"""Session detail panel widget for detailed log viewing."""

from __future__ import annotations

from typing import Union

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Markdown, Static

from models.antigravity import AntigravityBrain
from models.gemini import GeminiSession


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
        color: $text-disabled;
    }

    SessionDetailPanel .message-container {
        height: auto;
        padding: 0 1;
    }

    SessionDetailPanel .message-user {
        background: $surface;
        margin: 1 0;
        padding: 1;
        border-left: thick $secondary;
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
        color: $text-disabled;
    }

    SessionDetailPanel .notes-section {
        padding: 1;
        margin-top: 1;
        border-top: solid $primary;
    }

    SessionDetailPanel .empty-state {
        color: $text-disabled;
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
        self._session: Union[GeminiSession, AntigravityBrain, None] = None
        self._brain: Union[AntigravityBrain, GeminiSession, None] = None

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with VerticalScroll(id="detail-scroll"):
            yield Static(
                "[dim]Select a session to view details[/dim]",
                classes="empty-state",
                id="empty-message",
            )

    def set_session(self, item: Union[GeminiSession, AntigravityBrain]) -> None:
        """Display a session or brain with all messages/tasks.

        Args:
            item: The GeminiSession or AntigravityBrain to display.
        """
        if isinstance(item, GeminiSession):
            self._session = item
            self._brain = None
            self._render_session()
        elif isinstance(item, AntigravityBrain):
            self._brain = item
            self._session = None
            self._render_brain()

    def _render_session(self) -> None:
        """Render the Gemini session view."""
        if not self._session or not isinstance(self._session, GeminiSession):
            return

        scroll = self.query_one("#detail-scroll", VerticalScroll)

        # Clear existing content
        scroll.remove_children()

        # Header
        header_content = (
            f"[bold]{escape(self._session.short_id)}[/bold]\n"
            f"[dim]Started: {self._session.start_time.strftime('%Y-%m-%d %H:%M')}[/dim]\n"
            f"[dim]Messages: {len(self._session.messages)} | "
            f"Tokens: {self._session.total_tokens}[/dim]\n"
            f"[dim]Project: {escape(self._session.project_hash[:12])} | "
            f"Duration: {int(self._session.duration // 60)}m | "
            f"Tools: {self._session.tool_call_count} | "
            f"Models: {', '.join(sorted(self._session.models_used)) or 'n/a'}[/dim]"
        )
        scroll.mount(Static(header_content, classes="detail-header", markup=True))

        # Messages
        tool_counts: dict[str, int] = {}
        for msg in self._session.messages:
            for tool in msg.tool_names:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        if tool_counts:
            top_tools = sorted(tool_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
            summary = "Top tools: " + ", ".join(f"{name} ({count})" for name, count in top_tools)
            scroll.mount(Static(f"[dim]{summary}[/dim]", classes="message-container", markup=True))

        for msg in self._session.messages:
            role = msg.type.lower()
            role_class = "message-user" if role == "user" else "message-assistant"
            role_label = "[cyan]You[/cyan]" if role == "user" else "[magenta]Gemini[/magenta]"

            # Truncate very long messages for display and escape markup
            content = msg.content or ""
            if len(content) > 2000:
                content = escape(content[:2000]) + "\n\n[dim]... (truncated)[/dim]"
            else:
                content = escape(content)

            msg_widget = Static(
                f"{role_label}\n{content}",
                classes=f"message-container {role_class}",
                markup=True,
            )
            scroll.mount(msg_widget)

    def _render_brain(self) -> None:
        """Render the Antigravity brain view."""
        if not self._brain or not isinstance(self._brain, AntigravityBrain):
            return

        scroll = self.query_one("#detail-scroll", VerticalScroll)

        # Clear existing content
        scroll.remove_children()

        # Header
        header_content = (
            f"[bold]{escape(self._brain.title or self._brain.short_id)}[/bold]\n"
            f"[dim]ID: {escape(self._brain.id)}[/dim]\n"
            f"[dim]Progress: {self._brain.completed_tasks}/{self._brain.task_count} tasks"
        )
        if self._brain.updated_at:
            header_content += (
                f" | Updated {self._brain.updated_at.strftime('%Y-%m-%d %H:%M')}"
            )
        header_content += "[/dim]"
        scroll.mount(Static(header_content, classes="detail-header", markup=True))

        # Tasks section
        scroll.mount(Static("[bold]Tasks[/bold]", classes="detail-header-title"))

        for task in self._brain.tasks:
            status = task.status
            text = escape(task.description or "")

            if status == "done":
                task_class = "task-done"
                prefix = "[green][x][/green]"
            elif status == "in_progress":
                task_class = "task-in-progress"
                prefix = "[yellow][~][/yellow]"
            else:
                task_class = "task-todo"
                prefix = "[dim][ ][/dim]"

            scroll.mount(
                Static(
                    f"{prefix} {text}",
                    classes=f"task-item {task_class}",
                    markup=True,
                )
            )

        # Notes section
        if self._brain.notes:
            scroll.mount(Static("[bold]Notes[/bold]", classes="notes-section"))
            for note in self._brain.notes:
                scroll.mount(Markdown(note))

        # Plan/walkthrough summaries
        if getattr(self._brain, "plan_summary", None) or getattr(
            self._brain, "walkthrough_summary", None
        ):
            scroll.mount(Static("[bold]Summaries[/bold]", classes="notes-section"))
            if self._brain.plan_summary:
                scroll.mount(
                    Static(
                        f"[cyan]Implementation Plan[/cyan]\n{self._brain.plan_summary}",
                        classes="message-container",
                    )
                )
            if self._brain.walkthrough_summary:
                scroll.mount(
                    Static(
                        f"[magenta]Walkthrough[/magenta]\n{self._brain.walkthrough_summary}",
                        classes="message-container",
                    )
                )

    def clear(self) -> None:
        """Clear the detail view."""
        self._session = None
        self._brain = None

        try:
            scroll = self.query_one("#detail-scroll", VerticalScroll)
        except Exception:
            return

        # Check if already showing empty message to avoid DuplicateIds
        if scroll.query("#empty-message"):
            # Ensure other children are removed (unlikely but safe)
            for child in scroll.children:
                if child.id != "empty-message":
                    child.remove()
            return

        scroll.remove_children()
        scroll.mount(
            Static(
                "[dim]Select a session to view details[/dim]",
                classes="empty-state",
                id="empty-message",
            )
        )
