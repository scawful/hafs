"""Virtual Chat Stream Widget.

Implements a high-performance chat stream by archiving old messages into a static
RichLog while keeping recent messages as interactive widgets.
"""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical
from textual.widget import Widget
from textual.widgets import RichLog, Static

from tui.widgets.streaming_message import StreamingMessage
from tui.widgets.tool_card import ToolCard


class VirtualChatStream(Widget):
    """A high-performance chat stream that archives old messages.

    Structure:
    - VerticalScroll
      - RichLog (Archive): Contains flattened text of old messages (99% of content)
      - Vertical (Active): Contains interactive widgets for recent messages (last 20)
    """

    DEFAULT_CSS = """
    VirtualChatStream {
        height: 100%;
        width: 100%;
        background: $background;
    }

    VirtualChatStream > VerticalScroll {
        height: 100%;
        width: 100%;
        scrollbar-gutter: stable;
    }

    #stream-archive {
        width: 100%;
        height: auto;
        background: $background;
        padding: 0 1;
    }

    #stream-active {
        width: 100%;
        height: auto;
        padding: 0 1;
        layout: vertical;
    }
    """

    def __init__(self, archive_threshold: int = 20, id: Optional[str] = None) -> None:
        super().__init__(id=id)
        self._archive_threshold = archive_threshold
        self._active_widgets: list[Widget] = []

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="stream-scroll"):
            yield RichLog(id="stream-archive", markup=True, wrap=True)
            yield Vertical(id="stream-active")

    def get_message(self, message_id: str) -> Optional[StreamingMessage]:
        """Find an active streaming message by ID."""
        for widget in self._active_widgets:
            if isinstance(widget, StreamingMessage) and widget.message_id == message_id:
                return widget
        return None

    def append_message(self, widget: Widget) -> None:
        """Append a new interactive message/tool widget."""
        active_container = self.query_one("#stream-active", Vertical)
        active_container.mount(widget)
        self._active_widgets.append(widget)

        # Check if we need to archive
        if len(self._active_widgets) > self._archive_threshold:
            self._archive_oldest()

        # Auto-scroll
        widget.scroll_visible()

    def _archive_oldest(self) -> None:
        """Move the oldest active widget to the archive log."""
        if not self._active_widgets:
            return

        widget = self._active_widgets.pop(0)

        # Extract content to archive
        # This is the tricky part: we need to serialize the widget to text/markdown
        content = ""
        if isinstance(widget, StreamingMessage):
            # Format: **Role**: Content
            role_style = "bold cyan" if widget.role == "user" else "bold green"
            name = widget.agent_name or widget.role.title()
            text = widget.get_content()
            content = f"[{role_style}]{name}[/]: {text}\n"

        elif isinstance(widget, ToolCard):
            # Format: [Tool] name ... output
            status_color = "green" if widget.success else "red"
            output = widget.stdout
            if widget.stderr:
                output += f"\n[red]Stderr:[/]\n{widget.stderr}"

            content = f"[bold yellow]Tool:[/] {widget.tool_name}\n[{status_color}]{output}[/]\n"

        # Write to log
        archive = self.query_one("#stream-archive", RichLog)
        if content:
            archive.write(content)

        # Remove widget
        widget.remove()

    def clear(self) -> None:
        """Clear all history."""
        self.query_one("#stream-archive", RichLog).clear()
        container = self.query_one("#stream-active", Vertical)
        for child in container.children:
            child.remove()
        self._active_widgets.clear()
