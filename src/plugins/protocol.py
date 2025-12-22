"""Plugin protocols for hafs extensibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

from adapters.protocols import CodeReviewAdapter, CodeSearchAdapter, IssueTrackerAdapter

if TYPE_CHECKING:
    from textual.widget import Widget

    from backends.base import BaseChatBackend
    from core.parsers.base import BaseParser
    from tui.app import HafsApp


@dataclass
class SearchResult:
    """Result from a code search."""
    file: str
    line: int
    content: str
    match_type: str = "text"


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for code search providers."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Execute a code search.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of SearchResult objects.
        """
        ...


@dataclass
class ReviewStatus:
    """Status of a code review (PR/MR)."""
    id: str
    title: str
    status: str
    author: str
    url: str


@runtime_checkable
class ReviewProvider(Protocol):
    """Protocol for code review providers."""

    @property
    def name(self) -> str:
        """Provider name."""
        ...

    async def get_reviews(self, user: Optional[str] = None) -> List[ReviewStatus]:
        """Get active reviews for a user.

        Args:
            user: Username filter (optional).

        Returns:
            List of ReviewStatus objects.
        """
        ...


@runtime_checkable
class HafsPlugin(Protocol):
    """Protocol for hafs plugins.

    Plugins can extend hafs functionality by:
    - Adding new chat backends (BackendPlugin)
    - Adding new log parsers (ParserPlugin)
    - Adding new UI widgets (WidgetPlugin)
    - Providing custom hooks and integrations

    Example:
        class MyPlugin:
            @property
            def name(self) -> str:
                return "my-plugin"

            @property
            def version(self) -> str:
                return "1.0.0"

            def activate(self, app: HafsApp) -> None:
                # Register components, add bindings, etc.
                pass

            def deactivate(self) -> None:
                # Cleanup
                pass
    """

    @property
    def name(self) -> str:
        """Unique plugin name.

        Returns:
            Plugin identifier string.
        """
        ...

    @property
    def version(self) -> str:
        """Plugin version string.

        Returns:
            SemVer version string.
        """
        ...

    def activate(self, app: "HafsApp") -> None:
        """Called when plugin is activated.

        Args:
            app: The HafsApp instance.
        """
        ...

    def deactivate(self) -> None:
        """Called when plugin is deactivated."""
        ...


@runtime_checkable
class BackendPlugin(Protocol):
    """Protocol for backend plugins.

    Backend plugins provide new AI chat backend implementations.

    Example:
        class OllamaPlugin:
            def get_backend_class(self) -> type[BaseChatBackend]:
                return OllamaBackend
    """

    def get_backend_class(self) -> type["BaseChatBackend"]:
        """Return the backend class to register.

        Returns:
            A BaseChatBackend subclass.
        """
        ...


@runtime_checkable
class ParserPlugin(Protocol):
    """Protocol for parser plugins.

    Parser plugins provide new log/data parsers.

    Example:
        class CustomLogPlugin:
            def get_parser_class(self) -> type[BaseParser]:
                return CustomLogParser
    """

    def get_parser_class(self) -> type["BaseParser"]:
        """Return the parser class to register.

        Returns:
            A BaseParser subclass.
        """
        ...


@runtime_checkable
class WidgetPlugin(Protocol):
    """Protocol for widget plugins.

    Widget plugins provide new UI components.

    Example:
        class MetricsDashboardPlugin:
            def get_widget_class(self) -> type[Widget]:
                return MetricsDashboard

            def get_screen_position(self) -> str:
                return "sidebar"
    """

    def get_widget_class(self) -> type["Widget"]:
        """Return the widget class.

        Returns:
            A Textual Widget subclass.
        """
        ...

    def get_screen_position(self) -> str:
        """Where to place widget in the UI.

        Returns:
            Position hint: "sidebar", "main", "footer", "modal".
        """
        ...


@runtime_checkable
class ToolPlugin(Protocol):
    """Protocol for tool plugins.

    Tool plugins provide capabilities like search and code review.
    """

    def get_search_provider(self) -> Optional[type[SearchProvider]]:
        """Return a search provider class."""
        ...

    def get_review_provider(self) -> Optional[type[ReviewProvider]]:
        """Return a review provider class."""
        ...


@runtime_checkable
class IntegrationPlugin(Protocol):
    """Protocol for external provider adapters (issue tracker, code review, code search)."""

    def get_issue_tracker(self) -> Optional[type[IssueTrackerAdapter]]:
        """Return issue tracker adapter class."""
        ...

    def get_code_review(self) -> Optional[type[CodeReviewAdapter]]:
        """Return code review adapter class."""
        ...

    def get_code_search(self) -> Optional[type[CodeSearchAdapter]]:
        """Return code search adapter class."""
        ...
