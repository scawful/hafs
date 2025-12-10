"""Stub adapter for demonstration purposes.

This module provides a template for implementing Google-internal adapters.
It is not functional and serves only as documentation.

To create a real adapter:
1. Copy this file
2. Rename it (e.g., gemini_internal.py)
3. Implement the actual connection and data fetching logic
4. Register the adapter in the package __init__.py
"""

from typing import AsyncIterator

from hafs.adapters.base import BaseAdapter


class StubGoogleAdapter(BaseAdapter[dict]):
    """Stub adapter demonstrating the adapter interface.

    This adapter does nothing but shows the expected interface
    for Google-internal tool adapters.

    DO NOT register this adapter - it's for documentation only.
    """

    @property
    def name(self) -> str:
        """Return adapter name."""
        return "google-stub"

    async def connect(self) -> bool:
        """Stub connection - always returns False.

        In a real adapter, this would:
        - Initialize gRPC/stubby connections
        - Authenticate with internal systems
        - Set up streaming channels

        Returns:
            False (stub is not functional).
        """
        return False

    async def disconnect(self) -> None:
        """Stub disconnect - does nothing.

        In a real adapter, this would:
        - Close gRPC/stubby connections
        - Clean up resources
        - Flush pending operations
        """
        pass

    async def fetch_data(self, **kwargs) -> list[dict]:  # type: ignore[type-arg]
        """Stub data fetch - returns empty list.

        In a real adapter, this would:
        - Query internal databases (Spanner, Bigtable, etc.)
        - Transform data to the expected format
        - Handle pagination and filtering

        Args:
            **kwargs: Query parameters (ignored in stub).

        Returns:
            Empty list (stub is not functional).
        """
        return []

    async def stream_updates(self) -> AsyncIterator[dict]:
        """Stub streaming - yields nothing.

        In a real adapter, this would:
        - Subscribe to Pub/Sub topics
        - Stream real-time updates
        - Handle reconnection on failure

        Yields:
            Nothing (stub is not functional).
        """
        return
        yield  # Make this a generator

    @property
    def is_connected(self) -> bool:
        """Stub connection status - always False."""
        return False


# DO NOT uncomment - this is a stub for documentation only
# AdapterRegistry.register(StubGoogleAdapter)
