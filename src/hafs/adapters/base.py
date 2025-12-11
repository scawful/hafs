"""Base adapter interface for external tool integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Generic, TypeVar

T = TypeVar("T")


class BaseAdapter(ABC, Generic[T]):
    """Abstract adapter for external tool integration.

    Provides a standardized interface for connecting to and fetching data
    from external tools. Subclass this to integrate with Google internal
    tools like Gemini-CLI (internal), Antigravity, etc.

    Type Parameters:
        T: The data type this adapter returns.

    Example:
        class GeminiInternalAdapter(BaseAdapter[InternalSession]):
            @property
            def name(self) -> str:
                return "gemini-internal"

            async def connect(self) -> bool:
                # Connect to internal RPC/API
                pass

            async def fetch_data(self, **kwargs) -> list[InternalSession]:
                # Fetch from internal data stores
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter name for registration.

        Returns:
            String identifier for this adapter.
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the external tool.

        Returns:
            True if connection was successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up connection to the external tool."""
        pass

    @abstractmethod
    async def fetch_data(self, **kwargs: Any) -> list[T]:
        """Fetch data from the external tool.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            List of data items.
        """
        pass

    async def stream_updates(self) -> AsyncIterator[T]:
        """Stream real-time updates from the tool.

        Override this method to provide streaming support.

        Yields:
            Data items as they become available.
        """
        # Default implementation: no streaming support
        return
        yield  # Make this a generator

    @property
    def is_connected(self) -> bool:
        """Check if adapter is currently connected.

        Override this to track connection state.

        Returns:
            True if connected, False otherwise.
        """
        return False


class AdapterRegistry:
    """Registry for tool adapters.

    Provides centralized registration and lookup of adapters.
    Adapters are registered by their class and instantiated on demand.

    Example:
        # Register an adapter
        AdapterRegistry.register(MyAdapter)

        # Get an adapter instance
        adapter = AdapterRegistry.get("my-adapter")
        if adapter:
            await adapter.connect()
            data = await adapter.fetch_data()
    """

    _adapters: dict[str, type[BaseAdapter]] = {}  # type: ignore[type-arg]
    _instances: dict[str, BaseAdapter] = {}  # type: ignore[type-arg]

    @classmethod
    def register(cls, adapter_class: type[BaseAdapter]) -> None:  # type: ignore[type-arg]
        """Register an adapter class.

        Args:
            adapter_class: The adapter class to register.
        """
        # Instantiate temporarily to get the name
        instance = adapter_class()
        cls._adapters[instance.name] = adapter_class

    @classmethod
    def get(cls, name: str) -> BaseAdapter | None:  # type: ignore[type-arg]
        """Get or create an adapter instance by name.

        Args:
            name: The adapter name to look up.

        Returns:
            Adapter instance, or None if not found.
        """
        if name not in cls._instances and name in cls._adapters:
            cls._instances[name] = cls._adapters[name]()
        return cls._instances.get(name)

    @classmethod
    def list_adapters(cls) -> list[str]:
        """List all registered adapter names.

        Returns:
            List of registered adapter names.
        """
        return list(cls._adapters.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister an adapter by name.

        Args:
            name: The adapter name to unregister.

        Returns:
            True if adapter was unregistered, False if not found.
        """
        if name in cls._adapters:
            del cls._adapters[name]
            if name in cls._instances:
                del cls._instances[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters (useful for testing)."""
        cls._adapters.clear()
        cls._instances.clear()
