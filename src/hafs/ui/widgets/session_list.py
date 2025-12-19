"""Session list widget for displaying Gemini/Antigravity sessions."""

from typing import Any, Union

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import ListItem, ListView, Static

from hafs.core.parsers.registry import ParserRegistry
from hafs.core.search import fuzzy_filter_multi
from hafs.models.antigravity import AntigravityBrain
from hafs.models.gemini import GeminiSession


class SessionSelected(Message):
    """Message sent when a session is selected."""

    def __init__(self, session: GeminiSession | AntigravityBrain) -> None:
        self.session = session
        super().__init__()


class SessionListItem(ListItem):
    """A single session item in the list."""

    DEFAULT_CSS = """
    SessionListItem {
        height: 3;
        padding: 0 1;
    }

    SessionListItem:hover {
        background: $surface;
    }
    """

    def __init__(self, session: GeminiSession | AntigravityBrain) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        """Compose the list item."""
        if isinstance(self.session, GeminiSession):
            timestamp = "N/A"
            if self.session.start_time:
                timestamp = self.session.start_time.strftime("%Y-%m-%d %H:%M")
            msg_count = len(self.session.messages)
            tokens = self.session.total_tokens
            tools = self.session.tool_call_count
            models = ", ".join(sorted(self.session.models_used)) or "n/a"

            yield Static(
                f"[dim]{timestamp}[/dim] "
                f"[purple]{self.session.short_id}[/purple] "
                f"[cyan]{msg_count} msgs[/cyan] "
                f"[green]{tokens} tokens[/green] "
                f"[magenta]{tools} tools[/magenta] "
                f"[dim]models:[/dim] {models}"
            )
        elif isinstance(self.session, AntigravityBrain):
            done, total = self.session.progress
            updated = ""
            if getattr(self.session, "updated_at", None):
                updated = f" [dim]{self.session.updated_at.strftime('%m-%d %H:%M')}[/dim]"

            yield Static(
                f"[purple]{self.session.short_id}[/purple] "
                f"{self.session.title[:30]} "
                f"[dim]({done}/{total} tasks){updated}[/dim]"
            )


class SessionList(ListView):
    """List of Gemini/Antigravity sessions."""

    DEFAULT_CSS = """
    SessionList {
        background: $surface;
        border: solid $primary;
    }
    """

    def __init__(self, parser_type: str = "gemini", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.parser_type = parser_type
        self._all_items: list[Union[GeminiSession, AntigravityBrain]] = []
        self._parser_instance: Any = None

    def on_mount(self) -> None:
        """Load sessions when mounted."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh session list from parser."""
        self.clear()
        self._all_items = []
        self._parser_instance = None

        parser_class = ParserRegistry.get(self.parser_type)
        if not parser_class:
            self.append(ListItem(Static(f"[red]Parser '{self.parser_type}' not found[/red]")))
            return

        parser = parser_class()
        self._parser_instance = parser

        if not parser.exists():
            path = parser.base_path
            self.append(ListItem(Static(f"[dim]Path not found: {path}[/dim]")))
            return

        try:
            items = parser.parse(max_items=30)
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            self.append(ListItem(Static(f"[red]Failed to read logs: {exc}[/red]")))
            return

        if not items:
            # Show detailed error message if available
            error = parser.last_error
            if error:
                self.append(ListItem(Static(f"[yellow]{error}[/yellow]")))
            else:
                self.append(ListItem(Static("[dim]No sessions found[/dim]")))
            return

        # Ensure items are correctly typed by the parser or cast them
        self._all_items = [
            item for item in items if isinstance(item, (GeminiSession, AntigravityBrain))
        ]
        for item in self._all_items:
            self.append(SessionListItem(item))

    def filter_by_query(self, query: str) -> None:
        """Filter session list by search query (fuzzy matching).

        Args:
            query: Search query string.
        """
        if not query:
            # Reset to show all items
            self._display_items(self._all_items)
            return

        # Use fuzzy search from parser if available
        if self._parser_instance:
            results = self._parser_instance.fuzzy_search(query, self._all_items, threshold=40)
            filtered = [r.item for r in results]
        else:
            # Fallback to simple fuzzy filtering
            keys: dict[str, Any] = {}
            if self._all_items and isinstance(self._all_items[0], GeminiSession):
                keys = {
                    "session_id": lambda s: s.session_id,
                    "content": lambda s: " ".join(m.content for m in s.messages),
                }
            elif self._all_items and isinstance(self._all_items[0], AntigravityBrain):
                keys = {
                    "id": lambda b: b.id,
                    "title": lambda b: b.title or "",
                    "tasks": lambda b: " ".join(t.description for t in b.tasks),
                }

            if keys:
                results = fuzzy_filter_multi(query, self._all_items, keys, threshold=40)
                filtered = [r.item for r in results]
            else:
                filtered = []

        self._display_items(filtered)

    def _display_items(self, items: list[Union[GeminiSession, AntigravityBrain]]) -> None:
        """Display the given items in the list.

        Args:
            items: Items to display.
        """
        self.clear()

        if not items:
            self.append(ListItem(Static("[dim]No matching sessions[/dim]")))
            return

        for item in items:
            self.append(SessionListItem(item))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle session selection."""
        if isinstance(event.item, SessionListItem):
            self.post_message(SessionSelected(event.item.session))
