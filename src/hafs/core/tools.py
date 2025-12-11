"""Tool registry for HAFS."""

from typing import Type, Optional
from hafs.plugins.protocol import SearchProvider, ReviewProvider


class ToolRegistry:
    """Registry for tool providers."""

    _search_provider: Optional[Type[SearchProvider]] = None
    _review_provider: Optional[Type[ReviewProvider]] = None

    @classmethod
    def register_search_provider(cls, provider: Type[SearchProvider]) -> None:
        """Register a search provider."""
        cls._search_provider = provider

    @classmethod
    def get_search_provider(cls) -> Optional[Type[SearchProvider]]:
        """Get the registered search provider."""
        return cls._search_provider

    @classmethod
    def register_review_provider(cls, provider: Type[ReviewProvider]) -> None:
        """Register a review provider."""
        cls._review_provider = provider

    @classmethod
    def get_review_provider(cls) -> Optional[Type[ReviewProvider]]:
        """Get the registered review provider."""
        return cls._review_provider
