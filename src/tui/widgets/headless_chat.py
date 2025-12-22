"""Headless chat view widget (no terminal emulation)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import RichLog, Static


class HeadlessChatView(Widget):
    """A lightweight chat transcript view for headless mode.

    This view is intended for quick answers without spawning terminal emulator UI.
    It displays user messages and streamed assistant output in a RichLog.
    """

    DEFAULT_CSS = """
    HeadlessChatView {
        width: 100%;
        height: 1fr;
        border: solid $primary;
    }

    HeadlessChatView #chat-header {
        height: 1;
        background: $primary;
        color: $text;
        padding: 0 1;
    }

    HeadlessChatView #chat-log {
        height: 1fr;
        padding: 0 1;
        background: $surface;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Headless Chat", id="chat-header")
            yield RichLog(id="chat-log", highlight=True, markup=True)

    def clear(self) -> None:
        """Clear the transcript."""
        self.query_one("#chat-log", RichLog).clear()

    def write_system(self, message: str) -> None:
        """Write a system/status line."""
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[dim]{message}[/]")
        log.scroll_end(animate=False)

    def write_user(self, message: str) -> None:
        """Write a user message."""
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold cyan]You:[/] {message}")
        log.scroll_end(animate=False)

    def start_assistant(self, name: str) -> None:
        """Start a new assistant response."""
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold green]{name}:[/]")
        log.scroll_end(animate=False)

    def write_assistant_chunk(self, chunk: str) -> None:
        """Append a streamed chunk from the assistant."""
        log = self.query_one("#chat-log", RichLog)
        for line in chunk.splitlines() or [""]:
            log.write(line)
        log.scroll_end(animate=False)

