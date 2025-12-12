import asyncio
from collections.abc import AsyncGenerator
from typing import Callable, Optional

from hafs.backends.base import BaseChatBackend
from hafs.models.agent import Agent, AgentMessage, SharedContext


class AgentLane:
    """Manages a single agent instance with its backend and message queue.

    An AgentLane represents an execution context for a single agent,
    managing its backend connection, message queue, and shared context.

    Example:
        agent = Agent(name="planner", role=AgentRole.PLANNER, ...)
        backend = BackendRegistry.get("claude")
        context = SharedContext()

        lane = AgentLane(agent, backend, context)
        await lane.start()
        await lane.receive_message(message)
        async for chunk in lane.stream_output():
            print(chunk, end="")
        await lane.stop()
    """

    def __init__(
        self,
        agent: Agent,
        backend: BaseChatBackend,
        shared_context: SharedContext,
    ) -> None:
        """Initialize an agent lane.

        Args:
            agent: The agent this lane manages.
            backend: The chat backend for this agent.
            shared_context: Shared context across all agents.
        """
        self._agent = agent
        self._backend = backend
        self._shared_context = shared_context
        self._message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._is_processing = False
        self._current_message: Optional[AgentMessage] = None

    @property
    def agent(self) -> Agent:
        """Get the agent managed by this lane."""
        return self._agent

    @property
    def is_running(self) -> bool:
        """Check if the lane's backend is running."""
        return self._backend.is_running

    @property
    def is_busy(self) -> bool:
        """Check if the lane is currently processing a message."""
        return self._is_processing or self._backend.is_busy

    @property
    def queue_size(self) -> int:
        """Get the current size of the message queue."""
        return self._message_queue.qsize()

    async def start(self) -> bool:
        """Start the agent lane (initialize backend).

        Returns:
            True if the backend started successfully.
        """
        success = await self._backend.start()
        if success and self._agent.system_prompt:
            # Inject the system prompt
            await self._backend.send_message(
                f"System: {self._agent.system_prompt}"
            )
        return success

    async def stop(self) -> None:
        """Stop the agent lane (terminate backend)."""
        await self._backend.stop()
        # Clear the queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message for processing.

        Args:
            message: The message to process.
        """
        await self._message_queue.put(message)

    def _build_context_prompt(self, user_message: str) -> str:
        """Build a prompt with injected shared context.

        Args:
            user_message: The user's message content.

        Returns:
            Message with shared context prepended.
        """
        context_text = self._shared_context.to_prompt_text()
        cognitive = None
        try:
            from hafs.core.protocol.prompt_context import get_prompt_context

            cognitive = get_prompt_context(Path.cwd())
        except Exception:
            cognitive = None

        if cognitive:
            return f"{context_text}\n\n{cognitive}\n\nUser: {user_message}"
        return f"{context_text}\n\nUser: {user_message}"

    async def process_next_message(self) -> bool:
        """Process the next message in the queue.

        Returns:
            True if a message was processed, False if queue is empty.
        """
        try:
            message = self._message_queue.get_nowait()
            self._current_message = message
            self._is_processing = True

            # Build prompt with shared context
            prompt = self._build_context_prompt(message.content)

            # Send to backend
            await self._backend.send_message(prompt)

            return True
        except asyncio.QueueEmpty:
            return False

    async def stream_output(self) -> AsyncGenerator[str, None]:
        """Stream the agent's response output.

        Yields:
            Response text chunks as they become available.

        Example:
            async for chunk in lane.stream_output():
                print(chunk, end="", flush=True)
        """
        try:
            async for chunk in self._backend.stream_response():
                yield chunk
        finally:
            self._is_processing = False
            self._current_message = None

    async def inject_context(self, context: str) -> None:
        """Inject additional context into the agent's session.

        Args:
            context: Context text to inject.
        """
        await self._backend.inject_context(context)

    def get_current_message(self) -> Optional[AgentMessage]:
        """Get the message currently being processed.

        Returns:
            The current message, or None if idle.
        """
        return self._current_message

    @property
    def has_pending_messages(self) -> bool:
        """Check if there are pending messages in the queue."""
        return not self._message_queue.empty()

    def to_prompt_text(self) -> str:
        """Convert the shared context to prompt text.

        This is a convenience method that delegates to the shared context.

        Returns:
            Formatted context text suitable for prompt injection.
        """
        return self._shared_context.to_prompt_text()

    async def send_key(self, key: str) -> None:
        """Send a special key to the backend.

        Forwards key input to the underlying PTY process for CLI backends.
        Useful for sending Ctrl+C, Ctrl+Y (YOLO mode), Shift+Tab (accept edits)
        to Gemini-CLI and similar tools.

        Args:
            key: Key name (e.g., "ctrl+c", "ctrl+y", "shift+tab").
        """
        if self._backend:
            self._backend.send_key(key)

    def write_raw(self, data: str) -> None:
        """Write raw data directly to the backend's PTY stdin.

        This is for sending arbitrary input (keypresses, text) to the terminal.

        Args:
            data: Raw string data to write.
        """
        if self._backend:
            self._backend.write_raw(data)

    def interrupt(self) -> None:
        """Send an interrupt signal (Ctrl+C) to the backend."""
        if self._backend:
            self._backend.interrupt()

    def set_raw_output_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """Set callback for raw PTY output (before parsing).

        This allows the UI to receive unprocessed terminal data for
        proper terminal emulation.

        Args:
            callback: Function called with raw output chunks, or None to clear.
        """
        if self._backend:
            self._backend.set_raw_output_callback(callback)


class AgentLaneManager:
    """Manages multiple agent lanes.

    This is a utility class for coordinating multiple AgentLane instances.
    It's useful when you need to manage several agents but don't need
    the full orchestration features of AgentCoordinator.
    """

    def __init__(self) -> None:
        """Initialize the lane manager."""
        self._lanes: dict[str, AgentLane] = {}

    def add_lane(self, name: str, lane: AgentLane) -> None:
        """Add a lane to the manager.

        Args:
            name: Unique name for the lane.
            lane: The AgentLane to manage.
        """
        self._lanes[name] = lane

    def remove_lane(self, name: str) -> bool:
        """Remove a lane from the manager.

        Args:
            name: Name of the lane to remove.

        Returns:
            True if the lane was removed, False if not found.
        """
        if name in self._lanes:
            del self._lanes[name]
            return True
        return False

    def get_lane(self, name: str) -> Optional[AgentLane]:
        """Get a lane by name.

        Args:
            name: Name of the lane to retrieve.

        Returns:
            The AgentLane, or None if not found.
        """
        return self._lanes.get(name)

    def list_lanes(self) -> list[str]:
        """List all managed lane names.

        Returns:
            List of lane names.
        """
        return list(self._lanes.keys())

    async def start_all(self) -> dict[str, bool]:
        """Start all managed lanes.

        Returns:
            Dictionary mapping lane names to start success status.
        """
        results = {}
        for name, lane in self._lanes.items():
            results[name] = await lane.start()
        return results

    async def stop_all(self) -> None:
        """Stop all managed lanes."""
        for lane in self._lanes.values():
            await lane.stop()

    @property
    def active_lanes(self) -> list[str]:
        """Get names of all running lanes.

        Returns:
            List of names of lanes that are currently running.
        """
        return [name for name, lane in self._lanes.items() if lane.is_running]

    @property
    def busy_lanes(self) -> list[str]:
        """Get names of all busy lanes.

        Returns:
            List of names of lanes that are currently processing.
        """
        return [name for name, lane in self._lanes.items() if lane.is_busy]
