"""History-aware backend wrapper.

Wraps any BaseChatBackend to log messages and tool calls to the history system.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable, Optional

from backends.base import BackendCapabilities, BaseChatBackend

if TYPE_CHECKING:
    from hafs.core.history.session import SessionManager

# Lazy imports to avoid circular dependencies
_HistoryLogger = None
_OperationType = None
_Provenance = None


def _get_history_imports():
    """Lazy load history module imports."""
    global _HistoryLogger, _OperationType, _Provenance
    if _HistoryLogger is None:
        from hafs.core.history.logger import HistoryLogger as HL
        from hafs.core.history.models import OperationType as OT
        from hafs.core.history.models import Provenance as P
        _HistoryLogger = HL
        _OperationType = OT
        _Provenance = P
    return _HistoryLogger, _OperationType, _Provenance


class HistoryBackend(BaseChatBackend):
    """Backend wrapper that logs all interactions to history.

    Wraps any BaseChatBackend and intercepts:
    - send_message: Logs user input
    - stream_response: Logs agent messages
    - Tool calls (if the backend supports them)

    Example:
        # Wrap an existing backend
        claude_backend = BackendRegistry.get("claude")
        history_backend = HistoryBackend(
            wrapped=claude_backend,
            logger=history_logger,
            agent_id="claude-main"
        )

        # Use normally - all calls are logged
        await history_backend.start()
        await history_backend.send_message("Hello!")
    """

    def __init__(
        self,
        wrapped: BaseChatBackend,
        logger: Any,  # HistoryLogger
        agent_id: Optional[str] = None,
        session_manager: Optional["SessionManager"] = None,
        log_user_input: bool = True,
    ) -> None:
        """Initialize the history-aware backend wrapper.

        Args:
            wrapped: The backend to wrap.
            logger: History logger instance.
            agent_id: Optional agent identifier for provenance.
            session_manager: Optional session manager.
        """
        self._wrapped = wrapped
        self._logger = logger
        self._agent_id = agent_id or wrapped.name
        self._session_manager = session_manager
        self._log_user_input = log_user_input

        # Track current message for response logging
        self._current_message: Optional[str] = None
        self._response_buffer: list[str] = []
        self._response_start_time: Optional[float] = None

    @property
    def name(self) -> str:
        """Return wrapped backend name with history suffix."""
        return f"{self._wrapped.name}+history"

    @property
    def display_name(self) -> str:
        """Return wrapped backend display name."""
        return self._wrapped.display_name

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return wrapped backend capabilities."""
        return self._wrapped.capabilities

    @property
    def is_running(self) -> bool:
        """Check if wrapped backend is running."""
        return self._wrapped.is_running

    @property
    def is_busy(self) -> bool:
        """Check if wrapped backend is busy."""
        return self._wrapped.is_busy

    async def start(self) -> bool:
        """Start the wrapped backend and log session start."""
        result = await self._wrapped.start()
        if result:
            self._logger.log_system_event(
                "backend_started",
                data={
                    "backend": self._wrapped.name,
                    "agent_id": self._agent_id,
                },
            )
        return result

    async def stop(self) -> None:
        """Stop the wrapped backend and log session end."""
        self._logger.log_system_event(
            "backend_stopped",
            data={
                "backend": self._wrapped.name,
                "agent_id": self._agent_id,
            },
        )
        await self._wrapped.stop()

    async def send_message(self, message: str) -> None:
        """Send message and log to history.

        Args:
            message: The message to send.
        """
        import time

        # Log user input
        if self._log_user_input:
            self._logger.log_user_input(message)
        else:
            self._logger.log_system_event(
                "prompt_sent",
                data={
                    "agent_id": self._agent_id,
                    "prompt_length": len(message),
                },
            )

        # Track for response correlation
        self._current_message = message
        self._response_buffer = []
        self._response_start_time = time.time()

        # Forward to wrapped backend
        await self._wrapped.send_message(message)

    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response and log complete message to history.

        Yields:
            Response chunks from the wrapped backend.
        """
        import time

        async for chunk in self._wrapped.stream_response():
            self._response_buffer.append(chunk)
            yield chunk

        # Log complete response when stream ends
        if self._response_buffer:
            full_response = "".join(self._response_buffer)
            duration_ms = None
            if self._response_start_time:
                duration_ms = int((time.time() - self._response_start_time) * 1000)

            self._logger.log_agent_message(
                agent_id=self._agent_id,
                message=full_response,
            )

            # Also log as operation with duration
            _, OperationType, Provenance = _get_history_imports()
            self._logger.log(
                operation_type=OperationType.AGENT_MESSAGE,
                name="response",
                input_data={"prompt": self._current_message},
                output=full_response[:10000],  # Truncate for storage
                duration_ms=duration_ms,
                provenance=Provenance(agent_id=self._agent_id),
            )

        # Reset tracking
        self._current_message = None
        self._response_buffer = []
        self._response_start_time = None

    async def inject_context(self, context: str) -> None:
        """Inject context and log to history.

        Args:
            context: Context text to inject.
        """
        self._logger.log_system_event(
            "context_injected",
            data={
                "agent_id": self._agent_id,
                "context_length": len(context),
            },
        )
        await self._wrapped.inject_context(context)

    def send_key(self, key: str) -> None:
        """Send key to wrapped backend.

        Args:
            key: Key name to send.
        """
        self._wrapped.send_key(key)

    def write_raw(self, data: str) -> None:
        """Write raw data to wrapped backend.

        Args:
            data: Raw data to write.
        """
        self._wrapped.write_raw(data)

    def interrupt(self) -> None:
        """Interrupt wrapped backend and log."""
        self._logger.log_system_event(
            "backend_interrupted",
            data={"agent_id": self._agent_id},
        )
        self._wrapped.interrupt()

    def set_raw_output_callback(
        self, callback: Callable[[str], None] | None
    ) -> None:
        """Set raw output callback on wrapped backend.

        Args:
            callback: Callback function or None.
        """
        self._wrapped.set_raw_output_callback(callback)


def wrap_with_history(
    backend: BaseChatBackend,
    history_dir: str,
    agent_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> HistoryBackend:
    """Convenience function to wrap a backend with history logging.

    Args:
        backend: Backend to wrap.
        history_dir: Directory for history logs.
        agent_id: Optional agent identifier.
        project_id: Optional project identifier.

    Returns:
        HistoryBackend wrapping the original.
    """
    from pathlib import Path

    HistoryLogger, _, _ = _get_history_imports()
    logger = HistoryLogger(
        history_dir=Path(history_dir),
        project_id=project_id,
    )

    return HistoryBackend(
        wrapped=backend,
        logger=logger,
        agent_id=agent_id,
    )
