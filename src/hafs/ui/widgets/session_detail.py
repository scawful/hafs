"""Session detail panel widget for detailed log viewing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static, Markdown

if TYPE_CHECKING:
    from hafs.models.gemini import GeminiSession, GeminiMessage
    from hafs.models.antigravity import AntigravityBrain


class SessionDetailPanel(Widget):
    """Detailed view for Gemini sessions and Antigravity brains.

    Shows full conversation history with syntax highlighting
    or task/notes details for brains.

    Example:
        panel = SessionDetailPanel(id="session-detail")
        panel.set_session(gemini_session)
        # or
        panel.set_brain(antigravity_brain)
    """

    DEFAULT_CSS = """
    SessionDetailPanel {
        height: 100%;
        background: $surface;
        border: solid $primary;
    }

    SessionDetailPanel .detail-header {
        height: auto;
        background: $primary;
        padding: 1;
        margin-bottom: 1;
    }

    SessionDetailPanel .detail-header-title {
        text-style: bold;
        color: $text;
    }

    SessionDetailPanel .detail-header-info {
        color: $text-muted;
    }

    SessionDetailPanel .message-container {
        height: auto;
        padding: 0 1;
    }

    SessionDetailPanel .message-user {
        background: $surface-highlight;
        margin: 1 0;
        padding: 1;
        border-left: thick $info;
    }

    SessionDetailPanel .message-assistant {
        background: $surface;
        margin: 1 0;
        padding: 1;
        border-left: thick $secondary;
    }

    SessionDetailPanel .message-role {
        text-style: bold;
        margin-bottom: 1;
    }

    SessionDetailPanel .message-content {
        color: $text;
    }

    SessionDetailPanel .task-item {
        padding: 1;
        margin: 0 1;
    }

    SessionDetailPanel .task-done {
        color: $success;
    }

    SessionDetailPanel .task-in-progress {
        color: $warning;
    }

    SessionDetailPanel .task-todo {
        color: $text-muted;
    }

    SessionDetailPanel .notes-section {
        padding: 1;
        margin-top: 1;
        border-top: solid $primary;
    }

    SessionDetailPanel .empty-state {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    """

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize session detail panel.

        Args:
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._session: "GeminiSession | None" = None
        self._brain: "AntigravityBrain | None" = None

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with VerticalScroll(id="detail-scroll"):
            yield Static(
                "[dim]Select a session to view details[/dim]",
                classes="empty-state",
                id="empty-message",
            )

    def set_session(self, session: "GeminiSession") -> None:
        """Display a Gemini session with all messages.

        Args:
            session: The GeminiSession to display.
        """
        self._session = session
        self._brain = None
        self._render_session()

    def set_brain(self, brain: "AntigravityBrain") -> None:
        """Display an Antigravity brain with tasks and notes.

        Args:
            brain: The AntigravityBrain to display.
        """
        self._brain = brain
        self._session = None
        self._render_brain()

    def _render_session(self) -> None:
        """Render the Gemini session view."""
        if not self._session:
            return

        scroll = self.query_one("#detail-scroll", VerticalScroll)

        # Clear existing content
        scroll.remove_children()

        # Header
        header_content = (
            f"[bold]{self._session.session_id[:16]}...[/bold]\n"
            f"[dim]Started: {self._session.start_time.strftime('%Y-%m-%d %H:%M')}[/dim]\n"
            f"[dim]Messages: {len(self._session.messages)} | "
            f"Tokens: {self._session.total_tokens}[/dim]"
        )
        scroll.mount(Static(header_content, classes="detail-header"))

        # Messages
        for msg in self._session.messages:
            role = msg.role.lower()
            role_class = "message-user" if role == "user" else "message-assistant"
            role_label = "[cyan]You[/cyan]" if role == "user" else "[magenta]Gemini[/magenta]"

            # Truncate very long messages for display
            content = msg.content
            if len(content) > 2000:
                content = content[:2000] + "\n\n[dim]... (truncated)[/dim]"

            msg_widget = Static(
                f"{role_label}\n{content}",
                classes=f"message-container {role_class}",
            )
            scroll.mount(msg_widget)

    def _render_brain(self) -> None:
        """Render the Antigravity brain view."""
        if not self._brain:
            return

        scroll = self.query_one("#detail-scroll", VerticalScroll)

        # Clear existing content
        scroll.remove_children()

        # Header
        header_content = (
            f"[bold]{self._brain.title or self._brain.short_id}[/bold]\n"
            f"[dim]ID: {self._brain.id}[/dim]\n"
            f"[dim]Progress: {self._brain.completed_tasks}/{self._brain.task_count} tasks[/dim]"
        )
        scroll.mount(Static(header_content, classes="detail-header"))

        # Tasks section
        scroll.mount(Static("[bold]Tasks[/bold]", classes="detail-header-title"))

        for task in self._brain.tasks:
            status = task.get("status", "todo")
            text = task.get("text", "")

            if status == "done":
                task_class = "task-done"
                prefix = "[green][x][/green]"
            elif status == "in_progress":
                task_class = "task-in-progress"
                prefix = "[yellow][/][/yellow]"
            else:
                task_class = "task-todo"
                prefix = "[dim][ ][/dim]"

            scroll.mount(Static(f"{prefix} {text}", classes=f"task-item {task_class}"))

        # Notes section
        if self._brain.notes:
            scroll.mount(Static("[bold]Notes[/bold]", classes="notes-section"))
            for note in self._brain.notes:
                scroll.mount(Markdown(note))

    def clear(self) -> None:
        """Clear the detail view."""
        self._session = None
        self._brain = None

        scroll = self.query_one("#detail-scroll", VerticalScroll)
        scroll.remove_children()
        scroll.mount(
            Static(
                "[dim]Select a session to view details[/dim]",
                classes="empty-state",
                id="empty-message",
            )
        )
