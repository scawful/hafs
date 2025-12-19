"""Streaming Message Widget - Token-by-token chat message rendering.

This widget provides real-time token-by-token rendering for chat messages
with support for markdown, message grouping, and sub-50ms latency display.

Features:
- Token accumulation with efficient rendering
- Markdown formatting via Rich markup
- Message grouping by agent_id
- Automatic scrolling to newest content
- Visual indicators for streaming state
- Support for user, assistant, and system messages

Performance:
- Target <50ms latency for token display
- Efficient text buffer management
- Minimal reflows during streaming

Usage:
    # In a screen
    msg = StreamingMessage(agent_id="planner", agent_name="Planner")
    msg.start_streaming(message_id="msg-123")

    # Subscribe to token events
    bus.subscribe("chat.stream_token", self._on_token)

    def _on_token(self, event):
        if event.message_id == msg.message_id:
            msg.append_token(event.token)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from rich.markdown import Markdown
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import RichLog, Static

from hafs.ui.core.event_bus import Event, StreamTokenEvent, get_event_bus

logger = logging.getLogger(__name__)


class StreamingMessage(Widget):
    """A widget that displays a streaming chat message.

    Renders messages token-by-token with markdown support and visual
    indicators for agent identity and streaming state.

    Attributes:
        agent_id: Unique identifier for the agent
        agent_name: Display name for the agent
        role: Message role (user/assistant/system)
        message_id: Current streaming message ID
        is_streaming: Whether actively receiving tokens
    """

    DEFAULT_CSS = """
    StreamingMessage {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    StreamingMessage .message-header {
        height: 1;
        width: 100%;
        padding: 0;
    }

    StreamingMessage .message-content {
        width: 100%;
        height: auto;
        min-height: 3;
        padding: 1;
        background: $panel;
        border-left: thick $primary;
    }

    StreamingMessage.user .message-content {
        border-left: thick $accent;
        background: $panel-darken-1;
    }

    StreamingMessage.system .message-content {
        border-left: thick $warning;
        background: $surface;
    }

    StreamingMessage .streaming-indicator {
        color: $warning;
    }

    StreamingMessage .complete-indicator {
        color: $success;
    }

    StreamingMessage .error-indicator {
        color: $error;
    }
    """

    is_streaming: reactive[bool] = reactive(False)
    token_count: reactive[int] = reactive(0)

    def __init__(
        self,
        agent_id: str,
        agent_name: str = "",
        role: str = "assistant",
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ):
        """Initialize streaming message widget.

        Args:
            agent_id: Unique identifier for the agent
            agent_name: Display name (defaults to agent_id if not provided)
            role: Message role (user/assistant/system)
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(id=id, classes=classes)
        self.agent_id = agent_id
        self.agent_name = agent_name or agent_id
        self.role = role
        self.message_id: Optional[str] = None

        self._content_buffer = ""
        self._started_at: Optional[datetime] = None
        self._bus = get_event_bus()
        self._subscription = None

        # Add role class for styling
        self.add_class(role)

    def compose(self) -> ComposeResult:
        """Compose the message widget."""
        with Vertical():
            yield Static(self._get_header_text(), id="header", classes="message-header")
            yield RichLog(
                id="content",
                classes="message-content",
                highlight=True,
                markup=True,
                wrap=True,
                auto_scroll=True,
            )

    def on_mount(self) -> None:
        """Subscribe to streaming events on mount."""
        self._subscription = self._bus.subscribe(
            "chat.stream_token",
            self._on_stream_token,
        )

    def on_unmount(self) -> None:
        """Unsubscribe from events on unmount."""
        if self._subscription:
            self._subscription.unsubscribe()
            self._subscription = None

    def _get_header_text(self) -> str:
        """Generate header text with agent info and status."""
        # Status indicator
        if self.is_streaming:
            indicator = "[yellow]●[/]"
            status = "[dim]streaming...[/]"
        elif self.token_count > 0:
            indicator = "[green]✓[/]"
            status = f"[dim]{self.token_count} tokens[/]"
        else:
            indicator = "[dim]○[/]"
            status = ""

        # Role-specific formatting
        if self.role == "user":
            name_color = "cyan"
        elif self.role == "system":
            name_color = "yellow"
        else:
            name_color = "green"

        name = f"[bold {name_color}]{self.agent_name}[/]"

        if status:
            return f"{indicator} {name} {status}"
        return f"{indicator} {name}"

    def _update_header(self) -> None:
        """Update the header display."""
        try:
            header = self.query_one("#header", Static)
            header.update(self._get_header_text())
        except Exception as e:
            logger.debug(f"Failed to update header: {e}")

    def watch_is_streaming(self, streaming: bool) -> None:
        """React to streaming state changes."""
        self._update_header()

    def watch_token_count(self, count: int) -> None:
        """React to token count changes."""
        self._update_header()

    def _on_stream_token(self, event: Event) -> None:
        """Handle streaming token events.

        Args:
            event: StreamTokenEvent from EventBus
        """
        # Type check
        if not isinstance(event, StreamTokenEvent):
            return

        # Only handle tokens for this message
        if event.message_id != self.message_id:
            return

        # Only handle tokens from this agent
        if event.agent_id != self.agent_id:
            return

        # Handle final marker
        if event.is_final:
            self.complete_streaming()
            return

        # Append token
        if event.token:
            self.append_token(event.token)

    def start_streaming(self, message_id: str) -> None:
        """Start streaming a new message.

        Args:
            message_id: Unique identifier for the message
        """
        self.message_id = message_id
        self.is_streaming = True
        self.token_count = 0
        self._content_buffer = ""
        self._started_at = datetime.now()

        # Clear previous content
        try:
            content = self.query_one("#content", RichLog)
            content.clear()
        except Exception as e:
            logger.debug(f"Failed to clear content: {e}")

        self._update_header()
        logger.debug(f"Started streaming message {message_id} for {self.agent_id}")

    def append_token(self, token: str) -> None:
        """Append a token to the message content.

        This is the hot path - optimized for <50ms latency.

        Args:
            token: Token text to append
        """
        if not self.is_streaming:
            return

        # Accumulate in buffer
        self._content_buffer += token
        self.token_count += 1

        # Render to display
        try:
            content = self.query_one("#content", RichLog)

            # For markdown rendering, we re-render the entire buffer
            # This is efficient enough for most use cases
            # For even faster rendering, we could use plain text mode
            content.clear()

            # Try markdown rendering, fall back to plain text on error
            try:
                md = Markdown(self._content_buffer)
                content.write(md)
            except Exception:
                # Fallback: plain text
                content.write(self._content_buffer)

            # Auto-scroll to end
            content.scroll_end(animate=False)

        except Exception as e:
            logger.debug(f"Failed to append token: {e}")

    def append_text(self, text: str) -> None:
        """Append a block of text (for non-streaming messages).

        Args:
            text: Text to append
        """
        self._content_buffer += text

        try:
            content = self.query_one("#content", RichLog)
            content.clear()

            try:
                md = Markdown(self._content_buffer)
                content.write(md)
            except Exception:
                content.write(self._content_buffer)

            content.scroll_end(animate=False)
        except Exception as e:
            logger.debug(f"Failed to append text: {e}")

    def complete_streaming(self) -> None:
        """Mark streaming as complete."""
        self.is_streaming = False
        self._update_header()

        # Log completion metrics
        if self._started_at:
            duration = datetime.now() - self._started_at
            duration_ms = int(duration.total_seconds() * 1000)
            logger.debug(
                f"Completed streaming {self.message_id}: "
                f"{self.token_count} tokens in {duration_ms}ms"
            )

    def set_content(self, content: str, is_complete: bool = True) -> None:
        """Set the complete message content.

        For non-streaming messages or when loading from history.

        Args:
            content: Full message content
            is_complete: Whether the message is complete
        """
        self._content_buffer = content
        self.is_streaming = not is_complete
        self.token_count = len(content.split())  # Rough token estimate

        try:
            content_widget = self.query_one("#content", RichLog)
            content_widget.clear()

            try:
                md = Markdown(content)
                content_widget.write(md)
            except Exception:
                content_widget.write(content)

        except Exception as e:
            logger.debug(f"Failed to set content: {e}")

        self._update_header()

    def get_content(self) -> str:
        """Get the current message content.

        Returns:
            Current content buffer
        """
        return self._content_buffer

    def clear(self) -> None:
        """Clear the message content."""
        self._content_buffer = ""
        self.token_count = 0
        self.is_streaming = False
        self.message_id = None

        try:
            content = self.query_one("#content", RichLog)
            content.clear()
        except Exception as e:
            logger.debug(f"Failed to clear: {e}")

        self._update_header()

    def set_error(self, error_message: str) -> None:
        """Display an error state.

        Args:
            error_message: Error message to display
        """
        self.is_streaming = False
        self._content_buffer = f"[red]Error:[/] {error_message}"

        try:
            content = self.query_one("#content", RichLog)
            content.clear()
            content.write(f"[red]Error:[/] {error_message}")
        except Exception as e:
            logger.debug(f"Failed to set error: {e}")

        self._update_header()
