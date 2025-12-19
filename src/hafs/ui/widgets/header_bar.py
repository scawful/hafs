"""macOS-style header bar for HAFS TUI."""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class HeaderBar(Widget):
    """Header bar with navigation and status info.

    Layout:
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ [Dashboard][Chat][Logs][Services]  HAFS  [Analysis][Config] [mode] [time] │
    └────────────────────────────────────────────────────────────────────────────┘
    """

    DEFAULT_CSS = """
    HeaderBar {
        height: 1;
        width: 100%;
        background: $primary-darken-2;
        color: $text;
    }

    HeaderBar #header-container {
        width: 100%;
        height: 1;
    }

    HeaderBar #nav-left {
        width: auto;
        height: 1;
    }

    HeaderBar #nav-right {
        width: auto;
        height: 1;
    }

    HeaderBar .nav-link {
        padding: 0 1;
        height: 1;
        color: $text-muted;
    }

    HeaderBar .nav-link:hover {
        background: $primary-darken-1;
        color: $text;
    }

    HeaderBar .nav-link.-active {
        color: $accent;
        text-style: bold;
    }

    HeaderBar #title-center {
        width: 1fr;
        height: 1;
        content-align: center middle;
        text-align: center;
    }

    HeaderBar #info-right {
        width: auto;
        height: 1;
    }

    HeaderBar .info-item {
        padding: 0 1;
    }
    """

    mode: reactive[str] = reactive("planning")
    active_screen: reactive[str] = reactive("dashboard")
    show_time: reactive[bool] = reactive(True)

    class MenuSelected(Message):
        """Emitted when a menu item is clicked."""

        def __init__(self, menu_id: str) -> None:
            self.menu_id = menu_id
            super().__init__()

    class NavigationRequested(Message):
        """Emitted when a nav button is clicked."""

        def __init__(self, screen: str) -> None:
            self.screen = screen
            super().__init__()

    def __init__(
        self,
        show_nav: bool = True,
        show_time: bool = True,
        active_screen: str = "dashboard",
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._show_nav = show_nav
        self.show_time = show_time
        self.active_screen = active_screen

    def compose(self) -> ComposeResult:
        """Compose the header layout."""
        with Horizontal(id="header-container"):
            # Left side - primary navigation
            with Horizontal(id="nav-left"):
                if self._show_nav:
                    yield Static("[1]Dash", id="nav-dashboard", classes="nav-link")
                    yield Static("[2]Chat", id="nav-chat", classes="nav-link")
                    yield Static("[3]Logs", id="nav-logs", classes="nav-link")
                    yield Static("[4]Svc", id="nav-services", classes="nav-link")

            # Center - title
            yield Static(
                "[bold]HAFS[/]",
                id="title-center",
            )

            # Right side - secondary nav + info
            with Horizontal(id="nav-right"):
                if self._show_nav:
                    yield Static("[5]Analysis", id="nav-analysis", classes="nav-link")
                    yield Static("[6]Config", id="nav-config", classes="nav-link")

            with Horizontal(id="info-right"):
                yield Static(
                    self._render_mode(),
                    classes="info-item",
                    id="mode-indicator",
                )
                if self.show_time:
                    yield Static(
                        self._render_time(),
                        classes="info-item",
                        id="time-display",
                    )

    def _render_mode(self) -> str:
        """Render the mode indicator."""
        if self.mode == "planning":
            return "[cyan]◉ plan[/]"
        else:
            return "[yellow]◉ exec[/]"

    def _render_time(self) -> str:
        """Render the current time."""
        return f"[dim]{datetime.now().strftime('%H:%M')}[/]"

    def on_mount(self) -> None:
        """Update active link on mount."""
        self._update_active_link()

    def watch_mode(self, mode: str) -> None:
        """React to mode changes."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
            indicator.update(self._render_mode())
        except Exception:
            pass

    def watch_active_screen(self, screen: str) -> None:
        """React to active screen changes."""
        self._update_active_link()

    def _update_active_link(self) -> None:
        """Update which nav link appears active."""
        screen_map = {
            "dashboard": "nav-dashboard",
            "chat": "nav-chat",
            "logs": "nav-logs",
            "services": "nav-services",
            "analysis": "nav-analysis",
            "config": "nav-config",
        }
        for screen_name, link_id in screen_map.items():
            try:
                link = self.query_one(f"#{link_id}", Static)
                if screen_name == self.active_screen:
                    link.add_class("-active")
                else:
                    link.remove_class("-active")
            except Exception:
                pass

    def on_click(self, event: Click) -> None:
        """Handle clicks on nav links."""
        try:
            widget = self.screen.get_widget_at(event.screen_x, event.screen_y)
            if widget and hasattr(widget, "id") and widget.id and widget.id.startswith("nav-"):
                screen = widget.id.replace("nav-", "")
                self.active_screen = screen
                self.post_message(self.NavigationRequested(screen))
        except Exception:
            pass

    def update_time(self) -> None:
        """Update the time display."""
        if self.show_time:
            try:
                time_widget = self.query_one("#time-display", Static)
                time_widget.update(self._render_time())
            except Exception:
                pass

    def set_active(self, screen: str) -> None:
        """Set the active screen (called externally)."""
        self.active_screen = screen
