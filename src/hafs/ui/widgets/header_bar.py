"""macOS-style header bar for HAFS TUI."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static

from hafs.ui.core.event_bus import Event

if TYPE_CHECKING:
    from hafs.ui.core.event_bus import NavigationEvent


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
        align: center middle;
    }

    HeaderBar #nav-left {
        width: auto;
        height: 1;
        padding: 0 0;
    }

    HeaderBar #nav-right {
        width: auto;
        height: 1;
        padding: 0 0;
    }

    HeaderBar .nav-btn {
        min-width: 6;
        height: 1;
        margin: 0;
        padding: 0 1;
        border: none;
        background: transparent;
    }

    HeaderBar .nav-btn:hover {
        background: $primary-darken-1;
    }

    HeaderBar .nav-btn.-active {
        background: $primary;
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
        padding: 0 1;
        align: center middle;
    }

    HeaderBar .info-item {
        padding: 0 1;
        height: 1;
    }

    HeaderBar #breadcrumb {
        padding: 0 2;
        height: 1;
        color: $text-muted;
    }
    """

    mode: reactive[str] = reactive("planning")
    active_screen: reactive[str] = reactive("dashboard")
    current_path: reactive[str] = reactive("/dashboard")
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
                    yield Button(
                        "Dashboard", id="nav-dashboard", classes="nav-btn", variant="default"
                    )
                    yield Button("Chat", id="nav-chat", classes="nav-btn", variant="default")
                    yield Button(
                        "Workspace", id="nav-workspace", classes="nav-btn", variant="default"
                    )
                    yield Button("Logs", id="nav-logs", classes="nav-btn", variant="default")
                    yield Button(
                        "Services", id="nav-services", classes="nav-btn", variant="default"
                    )

            # Center - title and breadcrumb
            yield Static(
                "[bold]HAFS[/]",
                id="title-center",
            )
            yield Static(
                self._render_breadcrumb(),
                id="breadcrumb",
            )

            # Right side - secondary nav + info
            with Horizontal(id="nav-right"):
                if self._show_nav:
                    yield Button(
                        "Analysis", id="nav-analysis", classes="nav-btn", variant="default"
                    )
                    yield Button("Config", id="nav-config", classes="nav-btn", variant="default")

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

    def _render_breadcrumb(self) -> str:
        """Render the breadcrumb from current_path."""
        path = self.current_path.strip("/")
        if not path:
            return ""
        parts = path.split("/")
        crumbs = " > ".join(p.title() for p in parts if p)
        return f"[dim]{crumbs}[/]"

    def watch_current_path(self, path: str) -> None:
        """Update breadcrumb when path changes."""
        try:
            breadcrumb = self.query_one("#breadcrumb", Static)
            breadcrumb.update(self._render_breadcrumb())
        except Exception:
            pass

    def on_mount(self) -> None:
        """Update active button on mount and subscribe to navigation events."""
        self._update_active_button()

        # Subscribe to navigation events for auto-updating breadcrumbs
        try:
            from hafs.ui.core.event_bus import get_event_bus

            bus = get_event_bus()
            bus.subscribe("navigation.*", self._on_navigation_event)
        except Exception:
            pass

    def _on_navigation_event(self, event: Event) -> None:
        """Handle navigation events to update breadcrumb."""
        from hafs.ui.core.event_bus import NavigationEvent

        if not isinstance(event, NavigationEvent):
            return
        try:
            path = event.data.get("path", "")
            if path:
                self.current_path = path
                # Also update active screen based on path
                screen_name = path.strip("/").split("/")[0]
                if screen_name:
                    self.active_screen = screen_name
        except Exception:
            pass

    def watch_mode(self, mode: str) -> None:
        """React to mode changes."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
            indicator.update(self._render_mode())
        except Exception:
            pass

    def watch_active_screen(self, screen: str) -> None:
        """React to active screen changes."""
        self._update_active_button()

    def _update_active_button(self) -> None:
        """Update which nav button appears active."""
        screen_map = {
            "dashboard": "nav-dashboard",
            "chat": "nav-chat",
            "workspace": "nav-workspace",
            "logs": "nav-logs",
            "services": "nav-services",
            "analysis": "nav-analysis",
            "config": "nav-config",
        }
        for screen_name, btn_id in screen_map.items():
            try:
                btn = self.query_one(f"#{btn_id}", Button)
                if screen_name == self.active_screen:
                    btn.add_class("-active")
                else:
                    btn.remove_class("-active")
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle nav button presses."""
        btn_id = event.button.id
        if btn_id and btn_id.startswith("nav-"):
            screen = btn_id.replace("nav-", "")
            self.active_screen = screen
            self.post_message(self.NavigationRequested(screen))

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

    def set_path(self, path: str) -> None:
        """Set the current navigation path for breadcrumbs."""
        self.current_path = path
