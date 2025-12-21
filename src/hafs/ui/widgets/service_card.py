"""Service card widget for displaying individual service status."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static

from hafs.core.services.models import ServiceStatus, ServiceState


class ServiceCard(Widget):
    """Widget displaying a single service's status and controls.

    Shows:
    - Service name and description
    - Current state (running/stopped/failed)
    - PID if running
    - Quick action buttons
    """

    DEFAULT_CSS = """
    ServiceCard {
        height: auto;
        padding: 1;
        border: solid $primary;
        margin-bottom: 1;
    }

    ServiceCard:hover {
        background: $primary;
    }

    ServiceCard.selected {
        border: solid $accent;
        background: $primary;
    }

    ServiceCard .service-header {
        height: auto;
        margin-bottom: 1;
    }

    ServiceCard .service-name {
        text-style: bold;
    }

    ServiceCard .status-running {
        color: $success;
    }

    ServiceCard .status-stopped {
        color: $text-disabled;
    }

    ServiceCard .status-failed {
        color: $error;
    }

    ServiceCard .service-meta {
        color: $text-disabled;
        margin-left: 2;
    }

    ServiceCard .actions {
        height: auto;
        margin-top: 1;
    }

    ServiceCard .actions Button {
        margin-right: 1;
        min-width: 10;
    }
    """

    selected: reactive[bool] = reactive(False)

    class Selected(Message):
        """Posted when card is selected."""

        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class ActionRequested(Message):
        """Posted when action button is clicked."""

        def __init__(self, service_name: str, action: str) -> None:
            super().__init__()
            self.service_name = service_name
            self.action = action

    def __init__(self, status: ServiceStatus, **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = status

    def watch_selected(self, selected: bool) -> None:
        """Update styling when selection changes."""
        self.set_class(selected, "selected")

    def update_status(self, status: ServiceStatus) -> None:
        """Update the displayed status."""
        self._status = status
        self.refresh(recompose=True)

    def compose(self) -> ComposeResult:
        status = self._status

        # State indicator
        state_icon = {
            ServiceState.RUNNING: "[green]\u25cf[/]",
            ServiceState.STOPPED: "[dim]\u25cb[/]",
            ServiceState.FAILED: "[red]\u25cf[/]",
            ServiceState.STARTING: "[yellow]\u25d0[/]",
            ServiceState.STOPPING: "[yellow]\u25d1[/]",
            ServiceState.UNKNOWN: "[dim]?[/]",
        }.get(status.state, "[dim]?[/]")

        yield Static(
            f"{state_icon} [bold]{status.name}[/]",
            classes="service-name service-header",
        )

        # Build meta info
        meta_parts = [status.state.value]
        if status.pid:
            meta_parts.append(f"PID: {status.pid}")
        if status.enabled:
            meta_parts.append("installed")

        yield Static(f"  {' | '.join(meta_parts)}", classes="service-meta")

        # Action buttons
        with Horizontal(classes="actions"):
            if status.state == ServiceState.RUNNING:
                yield Button("Stop", id=f"stop-{status.name}", variant="error")
                yield Button("Restart", id=f"restart-{status.name}", variant="warning")
            else:
                yield Button("Start", id=f"start-{status.name}", variant="success")
            yield Button("Logs", id=f"logs-{status.name}", variant="default")

    def on_click(self) -> None:
        """Handle click to select this card."""
        self.post_message(self.Selected(self._status.name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle action button clicks."""
        button_id = event.button.id or ""
        action = None

        if button_id.startswith("start-"):
            action = "start"
        elif button_id.startswith("stop-"):
            action = "stop"
        elif button_id.startswith("restart-"):
            action = "restart"
        elif button_id.startswith("logs-"):
            action = "logs"

        if action:
            self.post_message(self.ActionRequested(self._status.name, action))
        event.stop()
