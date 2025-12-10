"""Plan viewer widget for displaying Claude plans."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical
from textual.widgets import Static, Label
from textual.widget import Widget

from hafs.core.parsers.claude import ClaudePlanParser
from hafs.models.claude import PlanDocument, TaskStatus


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
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    """

    def __init__(self, plans: list[PlanDocument] | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._plans = plans

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with VerticalScroll():
            yield Label("Claude Plans", classes="pv-title")

            if self._plans:
                for plan in self._plans[:10]:  # Limit display
                    done, total = plan.progress

                    with Vertical(classes="pv-plan"):
                        # Title and progress
                        progress_bar = self._make_progress_bar(done, total)
                        yield Label(
                            f"[bold]{plan.title[:40]}[/bold] {progress_bar}",
                            classes="pv-plan-title",
                        )
                        yield Label(f"[dim]{plan.path.name}[/dim]")

                        # Tasks
                        for task in plan.tasks[:5]:  # Limit tasks shown
                            icon, color = self._get_task_style(task.status)
                            yield Label(
                                f"  {icon} [{color}]{task.description[:50]}[/{color}]",
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
        parser = ClaudePlanParser()
        if parser.exists():
            self._plans = parser.parse(max_items=20)
        else:
            self._plans = []
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
