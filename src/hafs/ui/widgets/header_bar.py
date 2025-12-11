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
    """Header bar with essential controls and status info.

    Layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │ [Menu] [Context]    halext agentic file system      [mode] [time]│
    └──────────────────────────────────────────────────────────────────┘
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

    HeaderBar #menu-left {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    HeaderBar .menu-item {
        padding: 0 1;
    }

    HeaderBar .menu-item:hover {
        background: $primary;
        color: $text-accent;
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
    }

    HeaderBar .info-item {
        padding: 0 1;
    }
    """

    mode: reactive[str] = reactive("planning")
    show_time: reactive[bool] = reactive(True)

    class MenuSelected(Message):
        """Emitted when a menu item is clicked."""

        def __init__(self, menu_id: str) -> None:
            self.menu_id = menu_id
            super().__init__()

    def __init__(
        self,
        show_menus: bool = True,
        show_time: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self._show_menus = show_menus
        self.show_time = show_time

    def compose(self) -> ComposeResult:
        """Compose the header layout."""
        with Horizontal(id="header-container"):
            # Left side - functional menus
            with Horizontal(id="menu-left"):
                if self._show_menus:
                    yield Static("[bold]Menu[/]", classes="menu-item", id="menu-palette")
                    yield Static("[bold]Context[/]", classes="menu-item", id="menu-context")

            # Center - title
            yield Static(
                "[dim]halext[/] [bold]agentic file system[/]",
                id="title-center",
            )

            # Right side - info
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

    def watch_mode(self, mode: str) -> None:
        """React to mode changes."""
        try:
            indicator = self.query_one("#mode-indicator", Static)
            indicator.update(self._render_mode())
        except Exception:
            pass

    def on_click(self, event: Click) -> None:
        """Handle clicks on menu items."""
        try:
            widget = self.screen.get_widget_at(event.screen_x, event.screen_y)
            if widget and hasattr(widget, "id"):
                if widget.id == "menu-palette":
                    self.post_message(self.MenuSelected("palette"))
                elif widget.id == "menu-context":
                    self.post_message(self.MenuSelected("context"))
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
