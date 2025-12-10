"""Adapters for external tool integration.

This package provides extension points for integrating with external tools
like Google internal systems (Gemini-CLI internal, Antigravity).

To create a custom adapter:
1. Subclass BaseAdapter
2. Implement the required methods
3. Register with AdapterRegistry

Example:
    from hafs.adapters.base import BaseAdapter, AdapterRegistry

    class MyAdapter(BaseAdapter[MyDataType]):
        @property
        def name(self) -> str:
            return "my-adapter"

        async def connect(self) -> bool:
            # Connect to your service
            return True

        async def fetch_data(self, **kwargs) -> list[MyDataType]:
            # Fetch data from your service
            return []

    # Register the adapter
    AdapterRegistry.register(MyAdapter)
"""

from hafs.adapters.base import BaseAdapter, AdapterRegistry

__all__ = [
    "BaseAdapter",
    "AdapterRegistry",
]
