"""Base chat backend abstraction layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from typing import Callable

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class BackendCapabilities(BaseModel):
    """Capabilities of a chat backend."""

    supports_streaming: bool = True
    supports_tool_use: bool = True
    supports_images: bool = False
    supports_files: bool = True
    max_context_tokens: int = 128000


class BaseChatBackend(ABC):
    """Abstract base class for AI chat backends.

    Provides a standardized interface for interacting with AI CLI tools
    via PTY subprocess management.

    Example:
        class MyBackend(BaseChatBackend):
            @property
            def name(self) -> str:
                return "my-backend"

            async def start(self) -> bool:
                # Spawn PTY process
                pass

            async def send_message(self, message: str) -> None:
                # Write to PTY
                pass

            async def stream_response(self) -> AsyncGenerator[str, None]:
                # Read from PTY
                yield ""
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique backend identifier.

        Returns:
            String identifier for this backend.
        """
        pass

    @property
    def display_name(self) -> str:
        """Human-readable display name.

        Returns:
            Display name, defaults to capitalized name.
        """
        return self.name.capitalize()

    @property
    def capabilities(self) -> BackendCapabilities:
        """Backend capabilities.

        Returns:
            BackendCapabilities describing what this backend supports.
        """
        return BackendCapabilities()

    @abstractmethod
    async def start(self) -> bool:
        """Start the backend (spawn PTY process).

        Returns:
            True if backend started successfully.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the backend (terminate PTY process)."""
        pass

    @abstractmethod
    async def send_message(self, message: str) -> None:
        """Send a message to the backend.

        Args:
            message: The message to send.
        """
        pass

    @abstractmethod
    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response chunks from the backend.

        Yields:
            Response text chunks as they become available.
        """
        yield ""

    async def inject_context(self, context: str) -> None:
        """Inject context into the conversation.

        Override this method to provide context injection support.

        Args:
            context: Context text to inject.
        """
        # Default: prepend to next message
        pass

    def send_key(self, key: str) -> None:
        """Send a special key to the backend.

        Override this method to support special key input for PTY backends.

        Args:
            key: Key name (e.g., "ctrl+c", "ctrl+y", "shift+tab").
        """
        # Default: no-op for backends that don't support PTY input
        pass

    def write_raw(self, data: str) -> None:
        """Write raw data directly to the PTY stdin.

        This is for sending arbitrary input (keypresses, text) to the terminal.
        Unlike send_message which may add formatting, this sends data as-is.

        Args:
            data: Raw string data to write to PTY.
        """
        # Default: no-op for backends that don't support raw PTY input
        pass

    def interrupt(self) -> None:
        """Send an interrupt signal (Ctrl+C) to the backend.

        Override this method to support interrupting the backend.
        """
        self.send_key("ctrl+c")

    def set_raw_output_callback(
        self, callback: "Callable[[str], None] | None"
    ) -> None:
        """Set callback for raw PTY output (before parsing).

        This allows widgets to receive unprocessed terminal data for
        proper terminal emulation.

        Args:
            callback: Function called with raw output chunks, or None to clear.
        """
        # Default: no-op for backends that don't support raw output
        pass

    @property
    def is_running(self) -> bool:
        """Check if backend is currently running.

        Returns:
            True if backend is running, False otherwise.
        """
        return False

    @property
    def is_busy(self) -> bool:
        """Check if backend is processing a request.

        Returns:
            True if backend is busy, False otherwise.
        """
        return False


class BackendRegistry:
    """Registry for chat backends.

    Provides centralized registration and lookup of backends.
    Backends are registered by their class and instantiated on demand.

    Example:
        # Register a backend
        BackendRegistry.register(MyBackend)

        # Get a backend instance
        backend = BackendRegistry.get("my-backend")
        if backend:
            await backend.start()
            await backend.send_message("Hello!")
    """

    _backends: dict[str, type[BaseChatBackend]] = {}
    _instances: dict[str, BaseChatBackend] = {}

    @classmethod
    def register(cls, backend_class: type[BaseChatBackend]) -> None:
        """Register a backend class.

        Args:
            backend_class: The backend class to register.
        """
        # Instantiate temporarily to get the name
        instance = backend_class()
        cls._backends[instance.name] = backend_class

    @classmethod
    def get(cls, name: str) -> BaseChatBackend | None:
        """Get or create a backend instance by name.

        Args:
            name: The backend name to look up.

        Returns:
            Backend instance, or None if not found.
        """
        if name not in cls._instances and name in cls._backends:
            cls._instances[name] = cls._backends[name]()
        return cls._instances.get(name)

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseChatBackend | None:
        """Create a new backend instance (not cached).

        Args:
            name: The backend name.
            **kwargs: Arguments to pass to backend constructor.

        Returns:
            New backend instance, or None if not found.
        """
        backend_class = cls._backends.get(name)
        if backend_class:
            return backend_class(**kwargs)
        return None

    @classmethod
    def list_backends(cls) -> list[str]:
        """List all registered backend names.

        Returns:
            List of registered backend names.
        """
        return list(cls._backends.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a backend by name.

        Args:
            name: The backend name to unregister.

        Returns:
            True if backend was unregistered, False if not found.
        """
        if name in cls._backends:
            del cls._backends[name]
            if name in cls._instances:
                del cls._instances[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered backends (useful for testing)."""
        cls._backends.clear()
        cls._instances.clear()
