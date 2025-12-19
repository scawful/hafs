"""Chat Adapter - Bridge between orchestration layer and UI events.

This adapter converts coordinator events to EventBus events, enabling
clean separation between the orchestration layer and UI components.

Responsibilities:
- Convert coordinator message events to ChatEvent/StreamTokenEvent
- Route tool execution results to ToolResultEvent
- Handle message streaming from agents
- Manage streaming context and state
- Provide high-level API for chat operations

Architecture:
    Coordinator → ChatAdapter → EventBus → UI Components
    (business logic) (translation) (pub/sub) (rendering)

Usage:
    # Initialize adapter
    adapter = ChatAdapter(coordinator, event_bus)

    # Start streaming a message
    await adapter.stream_agent_message(agent_id, message)

    # Handle tool execution
    await adapter.execute_tool(agent_id, tool_name, args)

    # Adapter publishes events to bus, UI widgets subscribe and react
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Optional

from hafs.ui.core.chat_protocol import (
    ChatMessage,
    MessageRole,
    MessageStatus,
    StreamingContext,
    create_streaming_context,
    emit_chat_message,
    emit_stream_complete,
    emit_stream_token,
    emit_tool_result,
)
from hafs.ui.core.event_bus import (
    AgentStatusEvent,
    EventBus,
    PhaseEvent,
    get_event_bus,
)

if TYPE_CHECKING:
    from hafs.agents.coordinator import AgentCoordinator
    from hafs.agents.lane import AgentLane

logger = logging.getLogger(__name__)


class ChatAdapter:
    """Adapter between agent orchestration and UI event system.

    Translates coordinator events and operations into EventBus events
    that UI components can subscribe to and react to.

    Attributes:
        coordinator: The AgentCoordinator instance
        bus: The EventBus for publishing events
    """

    def __init__(
        self,
        coordinator: "AgentCoordinator",
        bus: Optional[EventBus] = None,
    ):
        """Initialize the chat adapter.

        Args:
            coordinator: AgentCoordinator instance to bridge
            bus: EventBus instance (uses global if not provided)
        """
        self.coordinator = coordinator
        self.bus = bus or get_event_bus()

        # Track active streaming contexts
        self._streaming_contexts: Dict[str, StreamingContext] = {}

        # Setup hooks
        self._setup_coordinator_hooks()

    def _setup_coordinator_hooks(self) -> None:
        """Set up hooks to intercept coordinator events."""
        # This would hook into coordinator's event system if available
        # For now, we provide methods to manually trigger events
        pass

    async def send_user_message(
        self,
        message: str,
        agent_id: Optional[str] = None,
    ) -> None:
        """Send a user message through the coordinator.

        Publishes a ChatEvent and routes the message to the appropriate agent.

        Args:
            message: User's message text
            agent_id: Target agent (None for auto-routing)
        """
        # Create and emit user message event
        user_msg = ChatMessage.user(message)
        emit_chat_message(self.bus, user_msg)

        # Route through coordinator
        try:
            if agent_id:
                target = agent_id
            else:
                # Auto-route based on content
                target = await self.coordinator.route_message(message, sender="user")

            # Get the lane and send message
            lane = self.coordinator.get_lane(target)
            if lane:
                from hafs.models.agent import AgentMessage

                agent_message = AgentMessage(
                    content=message,
                    sender="user",
                    recipient=target,
                )
                await lane.receive_message(agent_message)

                # Publish agent status
                self.bus.publish(AgentStatusEvent(
                    agent_id=target,
                    status="thinking",
                    message=f"Processing message from user",
                ))

        except Exception as e:
            logger.error(f"Failed to send user message: {e}")
            # Publish error event
            error_msg = ChatMessage.system(f"Error routing message: {e}")
            emit_chat_message(self.bus, error_msg)

    async def stream_agent_response(
        self,
        agent_id: str,
    ) -> AsyncIterator[str]:
        """Stream an agent's response token-by-token.

        Creates a streaming context and emits StreamTokenEvent for each token.

        Args:
            agent_id: Agent to stream from

        Yields:
            Individual tokens from the agent's response
        """
        # Create streaming context
        context = create_streaming_context(agent_id)
        self._streaming_contexts[context.message_id] = context

        try:
            # Start streaming
            self.bus.publish(AgentStatusEvent(
                agent_id=agent_id,
                status="executing",
                message="Generating response",
            ))

            # Get the lane
            lane = self.coordinator.get_lane(agent_id)
            if not lane:
                raise ValueError(f"Agent {agent_id} not found")

            # Process message and stream response
            await lane.process_next_message()

            # Stream through coordinator
            async for chunk in self.coordinator.stream_agent_response(agent_id):
                # Append to context
                context.append_token(chunk)

                # Emit token event
                emit_stream_token(
                    self.bus,
                    token=chunk,
                    message_id=context.message_id,
                    agent_id=agent_id,
                    is_final=False,
                )

                yield chunk

            # Mark complete
            context.mark_complete()

            # Emit completion
            emit_stream_complete(
                self.bus,
                message_id=context.message_id,
                agent_id=agent_id,
                full_content=context.accumulated_content,
            )

            # Update agent status
            self.bus.publish(AgentStatusEvent(
                agent_id=agent_id,
                status="idle",
                message="Response complete",
            ))

        except Exception as e:
            logger.error(f"Failed to stream response from {agent_id}: {e}")

            # Emit error message
            error_msg = ChatMessage.system(
                f"Error streaming from {agent_id}: {e}",
                tags=["error"],
            )
            emit_chat_message(self.bus, error_msg)

            # Update agent status
            self.bus.publish(AgentStatusEvent(
                agent_id=agent_id,
                status="error",
                message=str(e),
            ))

        finally:
            # Clean up context
            if context.message_id in self._streaming_contexts:
                del self._streaming_contexts[context.message_id]

    async def execute_tool(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> None:
        """Execute a tool through the coordinator and publish result.

        Args:
            agent_id: Agent executing the tool
            tool_name: Name of the tool to execute
            args: Tool arguments
        """
        import time

        start_time = time.time()

        try:
            # Update agent status
            self.bus.publish(AgentStatusEvent(
                agent_id=agent_id,
                status="executing",
                message=f"Running {tool_name}",
            ))

            # Execute through coordinator
            # This is a placeholder - actual implementation depends on coordinator API
            result = await self._execute_tool_impl(agent_id, tool_name, args)

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Emit tool result event
            emit_tool_result(
                self.bus,
                tool_name=tool_name,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                duration_ms=duration_ms,
                success=result.get("success", True),
                agent_id=agent_id,
            )

            # Update agent status
            self.bus.publish(AgentStatusEvent(
                agent_id=agent_id,
                status="idle",
                message=f"Completed {tool_name}",
            ))

        except Exception as e:
            logger.error(f"Failed to execute tool {tool_name}: {e}")

            duration_ms = int((time.time() - start_time) * 1000)

            # Emit error result
            emit_tool_result(
                self.bus,
                tool_name=tool_name,
                stdout="",
                stderr=str(e),
                duration_ms=duration_ms,
                success=False,
                agent_id=agent_id,
            )

            # Update agent status
            self.bus.publish(AgentStatusEvent(
                agent_id=agent_id,
                status="error",
                message=f"Tool error: {e}",
            ))

    async def _execute_tool_impl(
        self,
        agent_id: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Implementation of tool execution.

        This is a placeholder that should be replaced with actual
        coordinator tool execution logic.

        Args:
            agent_id: Agent executing the tool
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Dictionary with stdout, stderr, success keys
        """
        # Placeholder implementation
        # Real implementation would call coordinator's tool execution
        await asyncio.sleep(0.1)  # Simulate work

        return {
            "stdout": f"Tool {tool_name} executed successfully",
            "stderr": "",
            "success": True,
        }

    def publish_agent_status(
        self,
        agent_id: str,
        status: str,
        message: Optional[str] = None,
        health: float = 1.0,
    ) -> None:
        """Publish an agent status update.

        Args:
            agent_id: Agent identifier
            status: Status string (thinking/executing/idle/error)
            message: Optional status message
            health: Health metric (0.0-1.0)
        """
        self.bus.publish(AgentStatusEvent(
            agent_id=agent_id,
            status=status,
            message=message,
            health=health,
        ))

    def publish_phase_event(
        self,
        phase: str,
        progress: float = 0.0,
        message: Optional[str] = None,
    ) -> None:
        """Publish a pipeline phase transition.

        Args:
            phase: Phase name (plan/execute/verify/summarize)
            progress: Progress percentage (0.0-1.0)
            message: Optional phase message
        """
        self.bus.publish(PhaseEvent(
            phase=phase,
            progress=progress,
            message=message,
        ))

    def get_streaming_context(self, message_id: str) -> Optional[StreamingContext]:
        """Get an active streaming context by message ID.

        Args:
            message_id: Message identifier

        Returns:
            StreamingContext if found, None otherwise
        """
        return self._streaming_contexts.get(message_id)

    def cleanup_streaming_context(self, message_id: str) -> None:
        """Remove a streaming context.

        Args:
            message_id: Message identifier to clean up
        """
        if message_id in self._streaming_contexts:
            del self._streaming_contexts[message_id]

    async def broadcast_message(
        self,
        message: str,
        exclude_agents: Optional[list[str]] = None,
    ) -> None:
        """Broadcast a message to all agents.

        Args:
            message: Message to broadcast
            exclude_agents: Optional list of agent IDs to exclude
        """
        exclude = exclude_agents or []

        for agent_name in self.coordinator.agents.keys():
            if agent_name not in exclude:
                try:
                    await self.send_user_message(message, agent_id=agent_name)
                except Exception as e:
                    logger.error(f"Failed to broadcast to {agent_name}: {e}")

    def register_message_callback(
        self,
        callback: Callable[[ChatMessage], None],
    ) -> None:
        """Register a callback for all chat messages.

        Useful for logging, persistence, or custom processing.

        Args:
            callback: Function to call for each message
        """
        def handler(event):
            try:
                # Convert event back to ChatMessage
                msg = ChatMessage(
                    content=event.data.get("content", ""),
                    role=MessageRole(event.data.get("role", "assistant")),
                    agent_id=event.data.get("agent_id", ""),
                    message_id=event.data.get("message_id"),
                )
                callback(msg)
            except Exception as e:
                logger.error(f"Message callback error: {e}")

        self.bus.subscribe("chat.message", handler)

    def get_agent_ids(self) -> list[str]:
        """Get list of registered agent IDs.

        Returns:
            List of agent identifiers
        """
        return list(self.coordinator.agents.keys())

    def get_lane(self, agent_id: str) -> Optional["AgentLane"]:
        """Get an agent lane by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentLane if found, None otherwise
        """
        return self.coordinator.get_lane(agent_id)

    async def start_all_agents(self) -> None:
        """Start all registered agents."""
        await self.coordinator.start_all_agents()

        # Publish status for each agent
        for agent_id in self.get_agent_ids():
            self.publish_agent_status(agent_id, "idle", "Agent started")

    async def stop_all_agents(self) -> None:
        """Stop all registered agents."""
        # Publish stopping status
        for agent_id in self.get_agent_ids():
            self.publish_agent_status(agent_id, "idle", "Agent stopping")

        # Stop coordinator
        await self.coordinator.stop_all_agents()
