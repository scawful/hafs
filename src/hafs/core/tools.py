"""Tool registry for HAFS."""

from typing import Type, Optional, Protocol, Any
from pathlib import Path
from textual.message import Message
from hafs.plugins.protocol import SearchProvider, ReviewProvider


class ToolFileSelected(Message):
    """Event emitted when a tool selects a file to open."""
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        super().__init__()


class DevToolProvider(Protocol):
    """Protocol for development tool providers."""
    
    name: str
    slug: str
    category: Optional[str] = None  # e.g. "reviews", "search", "issues"
    
    def create_widget(self) -> Any:  # Returns a Textual Widget
        """Create and return the main widget for this tool."""
        ...


class ToolRegistry:
    """Registry for tool providers."""

    _search_provider: Optional[Type[SearchProvider]] = None
    _review_provider: Optional[Type[ReviewProvider]] = None
    _dev_tools: list[Type[DevToolProvider]] = []

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

    @classmethod
    def register_dev_tool(cls, tool: Type[DevToolProvider]) -> None:
        """Register a dev tool provider."""
        if tool not in cls._dev_tools:
            cls._dev_tools.append(tool)

    @classmethod
    def get_dev_tools(cls) -> list[Type[DevToolProvider]]:
        """Get all registered dev tools."""
        return cls._dev_tools
