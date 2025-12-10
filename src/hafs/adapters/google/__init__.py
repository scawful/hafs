"""Google internal tool adapters.

This package provides extension points for Google-internal tools like:
- Gemini-CLI (internal version)
- Antigravity (internal)
- Other proprietary systems

IMPORTANT: This package contains stubs only. Actual implementations
should be added in a separate, internal package (e.g., hafs_google)
that is not distributed with the public hafs package.

To implement a Google-internal adapter:

1. Create a new file in this package (e.g., gemini_internal.py)
2. Subclass BaseAdapter with your implementation
3. Register with AdapterRegistry.register(YourAdapter)

Example:

    from hafs.adapters.base import BaseAdapter, AdapterRegistry
    from typing import AsyncIterator

    class GeminiInternalAdapter(BaseAdapter[InternalSession]):
        '''Adapter for Google-internal Gemini CLI.'''

        @property
        def name(self) -> str:
            return "gemini-internal"

        async def connect(self) -> bool:
            # Connect to internal RPC/API
            # e.g., stubby, grpc, etc.
            return True

        async def disconnect(self) -> None:
            # Clean up connections
            pass

        async def fetch_data(self, **kwargs) -> list[InternalSession]:
            # Fetch from internal data stores
            # e.g., Spanner, Bigtable, etc.
            return []

        async def stream_updates(self) -> AsyncIterator[InternalSession]:
            # Stream real-time updates via pubsub, etc.
            pass

    # Register on import
    AdapterRegistry.register(GeminiInternalAdapter)

The adapter will automatically be available to the UI through the registry.
"""

# Note: Actual Google-internal adapters should be implemented in a
# separate internal package and imported here conditionally.

try:
    # Try to import internal adapters if available
    import hafs_google  # noqa: F401
except ImportError:
    # Internal package not available (expected for external builds)
    pass
