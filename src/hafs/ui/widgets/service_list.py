"""Service list widget for displaying all services."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from hafs.core.services.models import ServiceStatus
from hafs.ui.widgets.service_card import ServiceCard


class ServiceListWidget(VerticalScroll):
    """Widget containing a list of service cards.

    Displays all known services with their status and
    allows selection for viewing logs and performing actions.
    """

    DEFAULT_CSS = """
    ServiceListWidget {
        height: 100%;
        padding: 1;
    }

    ServiceListWidget .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    ServiceListWidget .empty-message {
        color: $text-disabled;
        text-style: italic;
        padding: 2;
    }
    """

    selected_service: reactive[str | None] = reactive(None)

    class ServiceSelected(Message):
        """Posted when a service is selected."""

        def __init__(self, service_name: str) -> None:
            super().__init__()
            self.service_name = service_name

    class ServiceActionRequested(Message):
        """Posted when a service action is requested."""

        def __init__(self, service_name: str, action: str) -> None:
            super().__init__()
            self.service_name = service_name
            self.action = action

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._services: dict[str, ServiceStatus] = {}

    def compose(self) -> ComposeResult:
        yield Static("[bold]Services[/bold]", classes="section-title")

        if not self._services:
            yield Static("No services found", classes="empty-message")
        else:
            for name, status in sorted(self._services.items()):
                yield ServiceCard(status, id=f"service-card-{name}")

    def update_services(self, services: dict[str, ServiceStatus]) -> None:
        """Update the service list with new status data."""
        self._services = services
        self.refresh(recompose=True)

        # Re-select the previously selected service if it still exists
        if self.selected_service and self.selected_service in self._services:
            self._update_selection(self.selected_service)

    def _update_selection(self, service_name: str | None) -> None:
        """Update visual selection state."""
        for card in self.query(ServiceCard):
            card.selected = card._status.name == service_name

    def watch_selected_service(self, service_name: str | None) -> None:
        """Handle selection changes."""
        self._update_selection(service_name)

    def on_service_card_selected(self, event: ServiceCard.Selected) -> None:
        """Handle service card selection."""
        self.selected_service = event.service_name
        self.post_message(self.ServiceSelected(event.service_name))

    def on_service_card_action_requested(
        self, event: ServiceCard.ActionRequested
    ) -> None:
        """Forward action requests."""
        self.post_message(
            self.ServiceActionRequested(event.service_name, event.action)
        )

    def select_next(self) -> None:
        """Select the next service in the list."""
        if not self._services:
            return

        names = sorted(self._services.keys())
        if self.selected_service is None:
            self.selected_service = names[0]
        else:
            try:
                idx = names.index(self.selected_service)
                self.selected_service = names[(idx + 1) % len(names)]
            except ValueError:
                self.selected_service = names[0]

        self.post_message(self.ServiceSelected(self.selected_service))

    def select_previous(self) -> None:
        """Select the previous service in the list."""
        if not self._services:
            return

        names = sorted(self._services.keys())
        if self.selected_service is None:
            self.selected_service = names[-1]
        else:
            try:
                idx = names.index(self.selected_service)
                self.selected_service = names[(idx - 1) % len(names)]
            except ValueError:
                self.selected_service = names[-1]

        self.post_message(self.ServiceSelected(self.selected_service))
