"""Agent coordinator for multi-agent orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# Core agent imports
from agents.core.lane import AgentLane
from agents.core.router import MentionRouter

# Backend integration imports
from backends.base import BackendRegistry
from backends.wrappers.history import HistoryBackend

# Hafs model imports
from hafs.models.agent import Agent, AgentMessage, AgentRole, SharedContext

if TYPE_CHECKING:
    from hafs.core.history.logger import HistoryLogger
    from hafs.core.history.session import SessionManager


class CoordinatorMode(str, Enum):
    """Operating mode for the agent coordinator.

    PLANNING: Agents focus on analysis, planning, and discussion.
              No execution of code or file modifications.
    EXECUTION: Agents focus on implementation and action.
               Can execute code, modify files, and make changes.
    """

    PLANNING = "planning"
    EXECUTION = "execution"


class AgentCoordinator:
    """Coordinates multiple agents in a multi-agent orchestration system.

    The coordinator manages agent registration, message routing, shared
    context, and inter-agent communication.

    Example:
        config = CoordinatorConfig(max_agents=5)
        coordinator = AgentCoordinator(config)

        # Register agents
        await coordinator.register_agent(
            name="planner",
            role=AgentRole.PLANNER,
            backend_name="claude",
            system_prompt="You are a planning specialist..."
        )

        # Route a message
        response = await coordinator.route_message("@planner create a roadmap")

        # Broadcast a message
        await coordinator.broadcast("Project goal updated", sender="system")
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize the agent coordinator.

        Args:
            config: Optional configuration. Can be:
                - HafsConfig: Pydantic config model from hafs.config.schema
                - dict: Configuration dictionary with keys:
                    - max_agents: Maximum number of agents (default: 10)
                    - default_backend: Default backend name (default: "claude")
                    - enable_context_sharing: Enable shared context (default: True)
                    - enabled_backends: List of enabled backend names
        """
        self._enabled_backends: set[str] = set()

        # Handle HafsConfig Pydantic model
        if config is not None and hasattr(config, "orchestrator"):
            # It's a HafsConfig model
            self._max_agents = config.orchestrator.max_agents
            self._default_backend = "gemini"  # Default to gemini
            self._enable_context_sharing = True

            # Extract enabled backends
            if hasattr(config, "backends"):
                self._enabled_backends = {b.name for b in config.backends if b.enabled}
        elif isinstance(config, dict):
            self._max_agents = config.get("max_agents", 10)
            self._default_backend = config.get("default_backend", "claude")
            self._enable_context_sharing = config.get("enable_context_sharing", True)
            self._enabled_backends = set(config.get("enabled_backends", ["gemini", "claude"]))
        else:
            self._max_agents = 10
            self._default_backend = "claude"
            self._enable_context_sharing = True
            self._enabled_backends = {"gemini", "claude"}

        self._agents: dict[str, Agent] = {}
        self._lanes: dict[str, AgentLane] = {}
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._shared_context = SharedContext()
        self._router = MentionRouter()
        self._is_running = False
        self._mode = CoordinatorMode.PLANNING

        # History logging (optional)
        self._history_logger: Optional["HistoryLogger"] = None
        self._session_manager: Optional["SessionManager"] = None

    @property
    def agents(self) -> dict[str, AgentLane]:
        """Get all registered agent lanes.

        Returns:
            Dictionary of agent lanes keyed by name.
        """
        return self._lanes.copy()

    @property
    def shared_context(self) -> SharedContext:
        """Get the shared context.

        Returns:
            The current shared context state.
        """
        return self._shared_context

    @property
    def agent_count(self) -> int:
        """Get the number of registered agents.

        Returns:
            Number of currently registered agents.
        """
        return len(self._agents)

    @property
    def mode(self) -> CoordinatorMode:
        """Get the current coordinator mode.

        Returns:
            The current CoordinatorMode.
        """
        return self._mode

    @property
    def active_agents(self) -> list[str]:
        """Get names of all running agents.

        Returns:
            List of names of agents that are currently running.
        """
        return [name for name, lane in self._lanes.items() if lane.is_running]

    async def register_agent(
        self,
        name: str,
        role: AgentRole,
        backend_name: Optional[str] = None,
        system_prompt: str = "",
        persona: Optional[str] = None,
    ) -> AgentLane:
        """Register a new agent with the coordinator.

        Args:
            name: Unique name for the agent.
            role: The agent's role.
            backend_name: Backend to use (defaults to config default).
            system_prompt: System prompt for the agent.

        Returns:
            The created AgentLane instance.

        Raises:
            ValueError: If agent name already exists or max agents reached.
            RuntimeError: If backend cannot be created.
        """
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")

        if len(self._agents) >= self._max_agents:
            raise ValueError(f"Maximum agents ({self._max_agents}) reached")

        # Use default backend if not specified
        backend_name = backend_name or self._default_backend

        if self._enabled_backends and backend_name not in self._enabled_backends:
            raise ValueError(f"Backend '{backend_name}' is disabled or not configured")

        # Create backend instance
        backend = BackendRegistry.create(backend_name)
        if not backend:
            available = BackendRegistry.list_backends()
            if available:
                raise RuntimeError(
                    f"Backend '{backend_name}' not found. Available: {', '.join(available)}"
                )
            else:
                raise RuntimeError(
                    f"Backend '{backend_name}' not found. No backends registered. "
                    "Ensure backends package is imported."
                )

        # Wrap backend with history logging if enabled
        if self._history_logger:
            backend = HistoryBackend(
                wrapped=backend,
                logger=self._history_logger,
                agent_id=name,
                session_manager=self._session_manager,
                log_user_input=False,
            )

        if not system_prompt:
            try:
                from agents.core.roles import get_role_system_prompt

                system_prompt = get_role_system_prompt(role, persona=persona)
            except Exception:
                system_prompt = ""

        # Create agent
        agent = Agent(
            name=name,
            role=role,
            backend_name=backend_name,
            system_prompt=system_prompt,
        )

        # Create lane
        lane = AgentLane(agent, backend, self._shared_context)

        # Store
        self._agents[name] = agent
        self._lanes[name] = lane

        return lane

    async def unregister_agent(self, name: str) -> bool:
        """Unregister an agent from the coordinator.

        Args:
            name: Name of the agent to unregister.

        Returns:
            True if agent was unregistered, False if not found.
        """
        if name not in self._agents:
            return False

        # Stop the lane if running
        lane = self._lanes[name]
        if lane.is_running:
            await lane.stop()

        # Remove from storage
        del self._agents[name]
        del self._lanes[name]

        return True

    async def start_agent(self, name: str) -> bool:
        """Start a specific agent.

        Args:
            name: Name of the agent to start.

        Returns:
            True if agent started successfully.

        Raises:
            ValueError: If agent not found.
        """
        if name not in self._lanes:
            raise ValueError(f"Agent '{name}' not found")

        return await self._lanes[name].start()

    async def stop_agent(self, name: str) -> None:
        """Stop a specific agent.

        Args:
            name: Name of the agent to stop.

        Raises:
            ValueError: If agent not found.
        """
        if name not in self._lanes:
            raise ValueError(f"Agent '{name}' not found")

        await self._lanes[name].stop()

    async def start_all_agents(self) -> dict[str, bool]:
        """Start all registered agents in parallel.

        Returns:
            Dictionary mapping agent names to start success status.
        """
        results = {}
        names = list(self._lanes.keys())
        tasks = [self._lanes[name].start() for name in names]

        if not tasks:
            self._is_running = True
            return {}

        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        for name, outcome in zip(names, outcomes):
            if isinstance(outcome, Exception):
                results[name] = False
            else:
                results[name] = bool(outcome)

        self._is_running = True
        return results

    async def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        for lane in self._lanes.values():
            await lane.stop()
        self._is_running = False
        self.complete_session()

    async def route_message(
        self, message: str, sender: str = "user"
    ) -> str:
        """Route a message to the appropriate agent.

        Uses @mentions or content-based routing to determine the recipient.

        Args:
            message: The message content.
            sender: The sender identifier (default: "user").

        Returns:
            Name of the agent that will handle the message.

        Raises:
            ValueError: If no suitable agent can be found.
        """
        # Resolve recipient using the router
        recipient, cleaned_message = self._router.resolve_recipient(
            message, self._agents
        )

        if not recipient:
            # Default to the first general or planner agent
            for agent in self._agents.values():
                if agent.role in {AgentRole.GENERAL, AgentRole.PLANNER}:
                    recipient = agent.name
                    break

        if not recipient:
            # Fallback to first available agent if any
            if self._agents:
                recipient = list(self._agents.keys())[0]

        if not recipient:
            raise ValueError("No suitable agent found to handle message")

        # Create agent message
        agent_message = AgentMessage(
            content=cleaned_message,
            sender=sender,
            recipient=recipient,
            mentions=self._router.extract_mentions(message),
        )

        # Send to the agent's lane
        lane = self._lanes[recipient]
        await lane.receive_message(agent_message)

        # Log to history
        if self._history_logger:
            if sender == "user":
                self._history_logger.log_user_input(cleaned_message)
            else:
                self._history_logger.log_agent_message(sender, cleaned_message)

        self._log_history(
            "message_routed",
            {
                "sender": sender,
                "recipient": recipient,
                "message_length": len(message),
            },
        )

        return recipient

    async def broadcast(
        self, message: str, sender: str = "system"
    ) -> list[str]:
        """Broadcast a message to all agents.

        Args:
            message: The message content to broadcast.
            sender: The sender identifier (default: "system").

        Returns:
            List of agent names that received the message.
        """
        recipients = []
        for name, lane in self._lanes.items():
            agent_message = AgentMessage(
                content=message,
                sender=sender,
                recipient=name,
            )
            await lane.receive_message(agent_message)
            recipients.append(name)

        # Log to history
        if self._history_logger:
            if sender == "user":
                self._history_logger.log_user_input(message)
            else:
                self._history_logger.log_system_event(
                    "broadcast",
                    {"sender": sender, "message_length": len(message)},
                )

        self._log_history(
            "message_broadcast",
            {
                "sender": sender,
                "recipients": recipients,
                "message_length": len(message),
            },
        )

        return recipients

    async def stream_agent_response(
        self, agent_name: str
    ) -> AsyncIterator[str]:
        """Stream responses from a specific agent.

        Args:
            agent_name: Name of the agent to stream from.

        Yields:
            Response chunks from the agent.

        Raises:
            ValueError: If agent not found.
        """
        if agent_name not in self._lanes:
            raise ValueError(f"Agent '{agent_name}' not found")

        lane = self._lanes[agent_name]

        # Process the next message if available
        if lane.has_pending_messages:
            await lane.process_next_message()

        # Stream the output
        async for chunk in lane.stream_output():
            yield chunk

    def update_shared_context(
        self,
        task: Optional[str] = None,
        finding: Optional[str] = None,
        decision: Optional[str] = None,
        context_items: Optional[list[Path]] = None,
    ) -> None:
        """Update the shared context accessible to all agents.

        Args:
            task: Update the current task.
            finding: Add a finding to the shared context.
            decision: Add a decision to the shared context.
            context_items: Replace pinned context paths.
        """
        if task is not None:
            self._shared_context.active_task = task

        if finding is not None:
            self._shared_context.add_finding(finding)

        if decision is not None:
            self._shared_context.add_decision(decision)

        if context_items is not None:
            self._shared_context.set_context_items(context_items)

    def set_context_items(self, paths: list[Path]) -> list[Path]:
        """Set pinned context paths and return the normalized list.

        Args:
            paths: Paths to pin in shared context.

        Returns:
            The stored list of paths.
        """
        return self._shared_context.set_context_items(paths)

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name.

        Args:
            name: Name of the agent to retrieve.

        Returns:
            The Agent instance, or None if not found.
        """
        return self._agents.get(name)

    def get_lane(self, name: str) -> Optional[AgentLane]:
        """Get an agent lane by name.

        Args:
            name: Name of the agent whose lane to retrieve.

        Returns:
            The AgentLane instance, or None if not found.
        """
        return self._lanes.get(name)

    def list_agents(self) -> list[str]:
        """List all registered agent names.

        Returns:
            List of agent names.
        """
        return list(self._agents.keys())

    def list_agents_by_role(self, role: AgentRole) -> list[str]:
        """List agents filtered by role.

        Args:
            role: The role to filter by.

        Returns:
            List of agent names with the specified role.
        """
        return [
            name for name, agent in self._agents.items()
            if agent.role == role
        ]

    @property
    def is_running(self) -> bool:
        """Check if the coordinator is running.

        Returns:
            True if the coordinator has started agents.
        """
        return self._is_running

    def get_agent_status(self, name: str) -> dict[str, Any]:
        """Get the status of a specific agent.

        Args:
            name: Name of the agent.

        Returns:
            Dictionary containing agent status information.

        Raises:
            ValueError: If agent not found.
        """
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found")

        agent = self._agents[name]
        lane = self._lanes[name]

        return {
            "name": agent.name,
            "role": agent.role.value,
            "backend": agent.backend_name,
            "is_running": lane.is_running,
            "is_busy": lane.is_busy,
            "queue_size": lane.queue_size,
            "has_pending": lane.has_pending_messages,
        }

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all agents.

        Returns:
            Dictionary mapping agent names to their status information.
        """
        return {
            name: self.get_agent_status(name)
            for name in self._agents.keys()
        }

    def _get_mode_prompt(self) -> str:
        """Get the system prompt modifier for the current mode.

        Returns:
            System prompt text based on current mode.
        """
        if self._mode == CoordinatorMode.PLANNING:
            return (
                "\n\n=== MODE: PLANNING ===\n"
                "You are in PLANNING mode. Focus on:\n"
                "- Analyzing requirements and constraints\n"
                "- Discussing approaches and alternatives\n"
                "- Creating plans and strategies\n"
                "- Identifying potential issues\n"
                "- Collaborating with other agents on design\n"
                "\n"
                "DO NOT:\n"
                "- Execute code or commands\n"
                "- Modify files\n"
                "- Make any changes to the system\n"
                "=== END MODE ===\n"
            )
        else:  # EXECUTION mode
            return (
                "\n\n=== MODE: EXECUTION ===\n"
                "You are in EXECUTION mode. Focus on:\n"
                "- Implementing planned solutions\n"
                "- Executing code and commands\n"
                "- Modifying files as needed\n"
                "- Taking concrete actions\n"
                "- Completing implementation tasks\n"
                "=== END MODE ===\n"
            )

    async def set_mode(self, mode: CoordinatorMode) -> None:
        """Set the coordinator mode and update all agent prompts.

        This updates the mode and injects the mode-specific prompt
        into all running agent lanes.

        Args:
            mode: The new CoordinatorMode to set.
        """
        old_mode = self._mode
        self._mode = mode

        # Inject mode context into all running lanes
        mode_prompt = self._get_mode_prompt()
        for lane in self._lanes.values():
            if lane.is_running:
                await lane.inject_context(mode_prompt)

        # Log the mode change to shared context
        self.update_shared_context(
            decision=f"Mode changed from {old_mode.value} to {mode.value}"
        )

        # Log to history if enabled
        if self._history_logger:
            self._history_logger.log_system_event(
                "mode_changed",
                {
                    "old_mode": old_mode.value,
                    "new_mode": mode.value,
                },
            )

    def set_history_logger(self, logger: "HistoryLogger") -> None:
        """Set the history logger for operation logging.

        Args:
            logger: The history logger instance.
        """
        self._history_logger = logger
        # Wrap existing backends if possible
        for name, lane in self._lanes.items():
            if lane.is_running:
                continue
            backend = lane._backend
            if isinstance(backend, HistoryBackend):
                continue
            lane._backend = HistoryBackend(
                wrapped=backend,
                logger=logger,
                agent_id=name,
                session_manager=self._session_manager,
                log_user_input=False,
            )

    def set_session_manager(self, manager: "SessionManager") -> None:
        """Set the session manager for session tracking.

        Args:
            manager: The session manager instance.
        """
        self._session_manager = manager

    def complete_session(self) -> None:
        """Complete the active session if available."""
        if self._session_manager:
            self._session_manager.complete()

    def _log_history(
        self,
        event_name: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log an event to history if logger is configured.

        Args:
            event_name: Name of the event.
            data: Optional event data.
        """
        if self._history_logger:
            self._history_logger.log_system_event(event_name, data)
