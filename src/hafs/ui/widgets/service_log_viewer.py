"""Service log viewer widget."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import RichLog, Static


class ServiceLogViewer(Vertical):
    """Widget for viewing service logs.

    Displays log output from a selected service with
    support for streaming updates.
    """

    DEFAULT_CSS = """
    ServiceLogViewer {
        height: 100%;
        padding: 1;
    }

    ServiceLogViewer .log-header {
        height: auto;
        padding-bottom: 1;
        border-bottom: solid $primary;
        margin-bottom: 1;
    }

    ServiceLogViewer .log-title {
        text-style: bold;
        color: $primary;
    }

    ServiceLogViewer .log-subtitle {
        color: $text-disabled;
    }

    ServiceLogViewer .empty-message {
        color: $text-disabled;
        text-style: italic;
        padding: 2;
        text-align: center;
    }

    ServiceLogViewer RichLog {
        height: 1fr;
        border: solid $primary;
        background: $surface;
    }
    """

    service_name: reactive[str | None] = reactive(None)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._log_content: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="log-header"):
            if self.service_name:
                yield Static(
                    f"[bold]Logs: {self.service_name}[/bold]",
                    classes="log-title",
                )
                yield Static(
                    "Press [bold]f[/bold] to follow, [bold]r[/bold] to refresh",
                    classes="log-subtitle",
                )
            else:
                yield Static("[bold]Logs[/bold]", classes="log-title")
                yield Static(
                    "Select a service to view logs",
                    classes="log-subtitle",
                )

        if self.service_name:
            yield RichLog(id="log-output", highlight=True, markup=True)
        else:
            yield Static(
                "No service selected\n\nSelect a service from the list to view its logs.",
                classes="empty-message",
            )

    def set_service(self, name: str) -> None:
        """Set the service to display logs for."""
        self.service_name = name
        self._log_content = ""
        self.refresh(recompose=True)

    def set_content(self, content: str) -> None:
        """Set the log content."""
        self._log_content = content
        try:
            log_widget = self.query_one("#log-output", RichLog)
            log_widget.clear()
            if content:
                for line in content.splitlines():
                    log_widget.write(line)
        except Exception:
            pass

    def append_line(self, line: str) -> None:
        """Append a line to the log output."""
        try:
            log_widget = self.query_one("#log-output", RichLog)
            log_widget.write(line.rstrip())
        except Exception:
            pass

    def clear(self) -> None:
        """Clear the log output."""
        self._log_content = ""
        try:
            log_widget = self.query_one("#log-output", RichLog)
            log_widget.clear()
        except Exception:
            pass

    def watch_service_name(self, name: str | None) -> None:
        """Handle service name changes."""
        self.refresh(recompose=True)
