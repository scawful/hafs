"""Session list widget for displaying Gemini/Antigravity sessions."""

from textual.app import ComposeResult
from textual.widgets import ListView, ListItem, Static
from textual.message import Message

from hafs.core.parsers.registry import ParserRegistry
from hafs.models.gemini import GeminiSession
from hafs.models.antigravity import AntigravityBrain


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
            timestamp = self.session.start_time.strftime("%Y-%m-%d %H:%M")
            msg_count = len(self.session.messages)
            tokens = self.session.total_tokens

            yield Static(
                f"[dim]{timestamp}[/dim] "
                f"[purple]{self.session.short_id}[/purple] "
                f"[cyan]{msg_count} msgs[/cyan] "
                f"[green]{tokens} tokens[/green]"
            )
        elif isinstance(self.session, AntigravityBrain):
            done, total = self.session.progress

            yield Static(
                f"[purple]{self.session.short_id}[/purple] "
                f"{self.session.title[:30]} "
                f"[dim]({done}/{total} tasks)[/dim]"
            )


class SessionList(ListView):
    """List of Gemini/Antigravity sessions."""

    DEFAULT_CSS = """
    SessionList {
        background: $surface;
        border: solid $primary;
    }
    """

    def __init__(self, parser_type: str = "gemini", **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self.parser_type = parser_type

    def on_mount(self) -> None:
        """Load sessions when mounted."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh session list from parser."""
        self.clear()

        parser_class = ParserRegistry.get(self.parser_type)
        if not parser_class:
            return

        parser = parser_class()
        if not parser.exists():
            self.append(ListItem(Static("[dim]No data found[/dim]")))
            return

        items = parser.parse(max_items=30)

        if not items:
            self.append(ListItem(Static("[dim]No sessions found[/dim]")))
            return

        for item in items:
            self.append(SessionListItem(item))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle session selection."""
        if isinstance(event.item, SessionListItem):
            self.post_message(SessionSelected(event.item.session))
