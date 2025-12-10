"""Plugin protocols for hafs extensibility."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from textual.widget import Widget

    from hafs.backends.base import BaseChatBackend
    from hafs.core.parsers.base import BaseParser
    from hafs.ui.app import HafsApp


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
