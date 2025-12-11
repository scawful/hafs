"""Collapsible sidebar panel widget inspired by lazygit."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class SidebarPanel(Widget):
    """A collapsible panel for sidebar sections (lazygit-style).

    Features:
    - Collapsible with click or keybind
    - Title bar with item count indicator
    - Minimal borders for clean look
    - Focus highlighting

    Example:
        panel = SidebarPanel(
            title="Projects",
            panel_id="projects",
            collapsed=False,
        )
    """

    DEFAULT_CSS = """
    SidebarPanel {
        height: auto;
        min-height: 3;
        background: $surface;
        border: none;
    }

    SidebarPanel.collapsed {
        height: 1;
        min-height: 1;
    }

    SidebarPanel .panel-header {
        height: 1;
        background: $primary-darken-2;
        padding: 0 1;
    }

    SidebarPanel:focus-within .panel-header {
        background: $primary;
    }

    SidebarPanel .panel-header-text {
        width: 100%;
    }

    SidebarPanel .panel-content {
        height: auto;
        max-height: 12;
        padding: 0;
    }

    SidebarPanel.collapsed .panel-content {
        display: none;
    }

    SidebarPanel .item-count {
        color: $text-muted;
    }
    """

    collapsed: reactive[bool] = reactive(False)
    item_count: reactive[int] = reactive(0)

    class Toggled(Message):
        """Emitted when panel is collapsed/expanded."""

        def __init__(self, panel_id: str, collapsed: bool) -> None:
            self.panel_id = panel_id
            self.collapsed = collapsed
            super().__init__()

    class Selected(Message):
        """Emitted when panel is selected/focused."""

        def __init__(self, panel_id: str) -> None:
            self.panel_id = panel_id
            super().__init__()

    def __init__(
        self,
        title: str,
        panel_id: str,
        collapsed: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize sidebar panel.

        Args:
            title: Panel title text.
            panel_id: Unique identifier for this panel.
            collapsed: Whether panel starts collapsed.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._title = title
        self._panel_id = panel_id
        self.collapsed = collapsed

    def compose(self) -> ComposeResult:
        """Compose the panel layout."""
        yield Static(self._render_header(), classes="panel-header panel-header-text")
        with Vertical(classes="panel-content"):
            yield from self._compose_content()

    def _compose_content(self) -> ComposeResult:
        """Override to provide panel content. Yields nothing by default."""
        return
        yield  # Make this a generator

    def _render_header(self) -> str:
        """Render the header text with title and item count."""
        arrow = "▸" if self.collapsed else "▾"
        count_str = f" [dim]({self.item_count})[/]" if self.item_count > 0 else ""
        return f"{arrow} [bold]{self._title}[/]{count_str}"

    def watch_collapsed(self, collapsed: bool) -> None:
        """React to collapsed state changes."""
        if collapsed:
            self.add_class("collapsed")
        else:
            self.remove_class("collapsed")
        # Update header
        try:
            header = self.query_one(".panel-header", Static)
            header.update(self._render_header())
        except Exception:
            pass
        self.post_message(self.Toggled(self._panel_id, collapsed))

    def watch_item_count(self, count: int) -> None:
        """React to item count changes."""
        try:
            header = self.query_one(".panel-header", Static)
            header.update(self._render_header())
        except Exception:
            pass

    def toggle(self) -> None:
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed

    def on_click(self, event: Message) -> None:
        """Handle click to toggle or select."""
        # Check if click was on header
        if hasattr(event, "y") and event.y == 0:
            self.toggle()
        else:
            self.post_message(self.Selected(self._panel_id))

    @property
    def panel_id(self) -> str:
        """Get panel identifier."""
        return self._panel_id


class ProjectsPanel(SidebarPanel):
    """Panel for displaying AFS projects."""

    def __init__(
        self,
        collapsed: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            title="Projects",
            panel_id="projects",
            collapsed=collapsed,
            id=id,
            classes=classes,
        )


class MountsPanel(SidebarPanel):
    """Panel for displaying mounted contexts."""

    def __init__(
        self,
        collapsed: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            title="Mounts",
            panel_id="mounts",
            collapsed=collapsed,
            id=id,
            classes=classes,
        )


class AgentsPanel(SidebarPanel):
    """Panel for displaying active agents."""

    def __init__(
        self,
        collapsed: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            title="Agents",
            panel_id="agents",
            collapsed=collapsed,
            id=id,
            classes=classes,
        )


class RecentPanel(SidebarPanel):
    """Panel for displaying recent files."""

    def __init__(
        self,
        collapsed: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            title="Recent",
            panel_id="recent",
            collapsed=collapsed,
            id=id,
            classes=classes,
        )
