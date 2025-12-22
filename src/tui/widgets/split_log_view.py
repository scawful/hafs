"""Split log view widget combining session list and detail panel."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input

from tui.widgets.session_detail import SessionDetailPanel
from tui.widgets.session_list import SessionList, SessionSelected

if TYPE_CHECKING:
    pass


class SplitLogView(Widget):
    """Split view showing session list and detail panel.

    Layout:
    ┌──────────────┬──────────────────────────────────┐
    │ Session List │ Session Detail                   │
    │              │                                  │
    │ > Session 1  │ [User]: How do I...              │
    │   Session 2  │ [Assistant]: Here's how...       │
    │   Session 3  │                                  │
    ├──────────────┴──────────────────────────────────┤
    │ Search: [___________]                           │
    └─────────────────────────────────────────────────┘

    Example:
        view = SplitLogView(parser_type="gemini", id="gemini-view")
    """

    DEFAULT_CSS = """
    SplitLogView {
        layout: vertical;
        height: 100%;
        width: 100%;
    }

    SplitLogView .split-container {
        layout: horizontal;
        height: 1fr;
        width: 100%;
    }

    SplitLogView .list-panel {
        width: 35%;
        min-width: 25;
        height: 100%;
        border-right: solid $primary;
    }

    SplitLogView .detail-panel {
        width: 65%;
        height: 100%;
    }

    SplitLogView SessionList {
        height: 100%;
    }

    SplitLogView SessionDetailPanel {
        height: 100%;
    }

    SplitLogView .search-bar {
        height: 3;
        background: $surface;
        border-top: solid $primary;
        padding: 0 1;
    }

    SplitLogView .search-input {
        width: 100%;
    }
    """

    class SearchSubmitted(Message):
        """Message sent when search is submitted."""

        def __init__(self, query: str, parser_type: str):
            self.query = query
            self.parser_type = parser_type
            super().__init__()

    def __init__(
        self,
        parser_type: str = "gemini",
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize split log view.

        Args:
            parser_type: Type of parser ("gemini" or "antigravity").
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self.parser_type = parser_type

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Horizontal(classes="split-container"):
            # Left panel - session list
            with Container(classes="list-panel"):
                yield SessionList(
                    parser_type=self.parser_type,
                    id=f"{self.parser_type}-list",
                )

            # Right panel - session detail
            with Container(classes="detail-panel"):
                yield SessionDetailPanel(id=f"{self.parser_type}-detail")

        # Search bar at bottom
        with Container(classes="search-bar"):
            yield Input(
                placeholder=f"Search {self.parser_type} logs...",
                id=f"{self.parser_type}-search",
                classes="search-input",
            )

    def on_session_selected(self, event: SessionSelected) -> None:
        """Update detail panel when session selected.

        Args:
            event: The session selection event.
        """
        detail = self.query_one(f"#{self.parser_type}-detail", SessionDetailPanel)
        detail.set_session(event.session)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input.

        Args:
            event: The input submission event.
        """
        if event.input.id == f"{self.parser_type}-search":
            query = event.value.strip()
            if query:
                self.post_message(self.SearchSubmitted(query, self.parser_type))
                self._perform_search(query)

    def _perform_search(self, query: str) -> None:
        """Perform search on sessions.

        Args:
            query: Search query string.
        """
        session_list = self.query_one(f"#{self.parser_type}-list", SessionList)
        session_list.filter_by_query(query)

    def refresh_data(self) -> None:
        """Refresh the session list data."""
        session_list = self.query_one(f"#{self.parser_type}-list", SessionList)
        session_list.refresh_data()
        self.clear_detail()

    def clear_detail(self) -> None:
        """Clear the detail panel."""
        detail = self.query_one(f"#{self.parser_type}-detail", SessionDetailPanel)
        detail.clear()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live-filter sessions while typing."""
        if event.input.id == f"{self.parser_type}-search":
            query = event.value.strip()
            # Only trigger fuzzy search for small queries when at least 2 chars
            if not query:
                self._perform_search("")
            elif len(query) >= 2:
                self._perform_search(query)
