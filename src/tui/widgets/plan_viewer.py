"""Plan viewer widget for displaying Claude plans."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Label

from core.parsers.claude import ClaudePlanParser
from models.claude import PlanDocument, TaskStatus


class PlanViewer(Widget):
    """Widget for viewing Claude plan documents."""

    DEFAULT_CSS = """
    PlanViewer {
        height: 100%;
        background: $surface;
        padding: 1;
    }

    .pv-title {
        color: $secondary;
        text-style: bold;
        margin-bottom: 1;
    }

    .pv-plan {
        margin-bottom: 2;
        padding: 1;
        border: solid $primary;
    }

    .pv-plan-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .pv-task {
        padding-left: 2;
    }

    .pv-empty {
        color: $text-disabled;
        text-align: center;
        padding: 2;
    }
    """

    def __init__(self, plans: list[PlanDocument] | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._plans = plans
        self._error_message: str | None = None

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with VerticalScroll():
            yield Label("Claude Plans", classes="pv-title")

            if self._error_message:
                yield Label(self._error_message, classes="pv-empty")
            elif self._plans:
                for plan in self._plans[:10]:  # Limit display
                    done, total = plan.progress
                    safe_title = escape(plan.title[:40])

                    with Vertical(classes="pv-plan"):
                        # Title and progress
                        progress_bar = self._make_progress_bar(done, total)
                        meta = ""
                        if plan.modified_at:
                            meta = f"  [dim]{plan.modified_at.strftime('%Y-%m-%d %H:%M')}[/dim]"
                        yield Label(
                            f"[bold]{safe_title}[/bold] {progress_bar}{meta}",
                            classes="pv-plan-title",
                        )
                        yield Label(f"[dim]{escape(plan.path.name)}[/dim]")

                        # Tasks
                        for task in plan.tasks[:5]:  # Limit tasks shown
                            icon, color = self._get_task_style(task.status)
                            task_text = escape(task.description[:50])
                            yield Label(
                                f"  {icon} [{color}]{task_text}[/{color}]",
                                classes="pv-task",
                            )

                        if len(plan.tasks) > 5:
                            yield Label(
                                f"  [dim]... and {len(plan.tasks) - 5} more tasks[/dim]"
                            )
            else:
                yield Label("No plans found", classes="pv-empty")

    def refresh_data(self) -> None:
        """Refresh plans from parser."""
        from core.parsers.registry import ParserRegistry

        parser_cls = ParserRegistry.get("claude") or ClaudePlanParser
        parser = parser_cls()
        self._error_message = None

        try:
            if parser.exists():
                self._plans = parser.parse(max_items=20)
            else:
                self._plans = []
                self._error_message = "[dim]No Claude plans directory found[/dim]"
        except Exception as exc:  # pragma: no cover - defensive for UI
            self._plans = []
            self._error_message = f"[red]Failed to load plans: {exc}[/red]"

        if not self._plans and not self._error_message and parser.last_error:
            self._error_message = f"[yellow]{parser.last_error}[/yellow]"

        self.refresh(recompose=True)

    def on_mount(self) -> None:
        """Load plans when mounted."""
        self.refresh_data()

    @staticmethod
    def _make_progress_bar(done: int, total: int, width: int = 10) -> str:
        """Create a text progress bar."""
        if total == 0:
            return ""
        filled = int(done / total * width)
        return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"

    @staticmethod
    def _get_task_style(status: TaskStatus) -> tuple[str, str]:
        """Get icon and color for task status."""
        return {
            TaskStatus.TODO: ("○", "dim"),
            TaskStatus.IN_PROGRESS: ("◐", "yellow"),
            TaskStatus.DONE: ("●", "green"),
        }.get(status, ("?", "white"))
