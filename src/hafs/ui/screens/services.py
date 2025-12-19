"""Services management screen for HAFS TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Static

from hafs.config.loader import load_config
from hafs.core.services import ServiceManager
from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.service_list import ServiceListWidget
from hafs.ui.widgets.service_log_viewer import ServiceLogViewer
from hafs.ui.widgets.which_key_bar import WhichKeyBar


class ServicesScreen(Screen, WhichKeyMixin):
    """Services management screen.

    Provides controls for managing HAFS background services including:
    - Model orchestrator daemon
    - Agent swarm coordinator
    - Web dashboard server

    Layout:
    +----------------------------------------------------------+
    | Header                                                    |
    | Platform: macOS (launchd)                                 |
    +------------------+---------------------------------------+
    | Service List     |  Service Logs                         |
    | [*] Orchestrator |  [log output...]                      |
    | [ ] Coordinator  |                                        |
    | [*] Dashboard    |                                        |
    +------------------+---------------------------------------+
    | WhichKeyBar                                    | Footer   |
    +----------------------------------------------------------+
    """

    BINDINGS = [
        Binding("q", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("j", "next_service", "Next", show=False),
        Binding("k", "prev_service", "Previous", show=False),
        Binding("s", "start_service", "Start"),
        Binding("S", "stop_service", "Stop"),
        Binding("R", "restart_service", "Restart"),
        Binding("e", "enable_service", "Enable"),
        Binding("d", "disable_service", "Disable"),
        Binding("i", "install_service", "Install"),
        Binding("u", "uninstall_service", "Uninstall"),
        Binding("l", "view_logs", "Logs"),
        Binding("f", "follow_logs", "Follow"),
    ]

    DEFAULT_CSS = """
    ServicesScreen {
        layout: vertical;
    }

    ServicesScreen #platform-info {
        height: 1;
        background: $surface;
        padding: 0 1;
        color: $text-muted;
    }

    ServicesScreen #main-container {
        height: 1fr;
    }

    ServicesScreen #service-list {
        width: 35;
        border-right: solid $primary-darken-2;
    }

    ServicesScreen #log-area {
        width: 1fr;
    }

    ServicesScreen #footer-area {
        height: auto;
        background: $surface;
    }

    ServicesScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }

    ServicesScreen #which-key-bar {
        width: 2fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._service_manager: ServiceManager | None = None
        self._selected_service: str | None = None
        self._following_logs: bool = False

    def compose(self) -> ComposeResult:
        yield HeaderBar(id="header-bar", active_screen="services")
        yield Static("Loading...", id="platform-info")

        with Horizontal(id="main-container"):
            yield ServiceListWidget(id="service-list")
            yield ServiceLogViewer(id="log-area")

        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield WhichKeyBar(id="which-key-bar")
                yield Footer()

    async def on_mount(self) -> None:
        """Initialize screen on mount."""
        self.title = "HAFS - Services"

        # Initialize service manager
        config = load_config()
        try:
            self._service_manager = ServiceManager(config)
            platform_info = self.query_one("#platform-info", Static)
            platform_info.update(f"Platform: {self._service_manager.platform_name}")
        except NotImplementedError as e:
            platform_info = self.query_one("#platform-info", Static)
            platform_info.update(f"[red]Error: {e}[/red]")
            return

        # Load initial service statuses
        await self._refresh_services()

    async def _refresh_services(self) -> None:
        """Refresh service status list."""
        if not self._service_manager:
            return

        statuses = await self._service_manager.status_all()
        service_list = self.query_one("#service-list", ServiceListWidget)
        service_list.update_services(statuses)

    async def _load_service_logs(self) -> None:
        """Load logs for selected service."""
        if not self._selected_service or not self._service_manager:
            return

        log_viewer = self.query_one("#log-area", ServiceLogViewer)
        log_viewer.set_service(self._selected_service)

        logs = await self._service_manager.logs(self._selected_service)
        log_viewer.set_content(logs)

    def on_service_list_widget_service_selected(
        self, event: ServiceListWidget.ServiceSelected
    ) -> None:
        """Handle service selection."""
        self._selected_service = event.service_name
        self.run_worker(self._load_service_logs())

    def on_service_list_widget_service_action_requested(
        self, event: ServiceListWidget.ServiceActionRequested
    ) -> None:
        """Handle service action requests from cards."""
        if event.action == "start":
            self.run_worker(self._do_start_service(event.service_name))
        elif event.action == "stop":
            self.run_worker(self._do_stop_service(event.service_name))
        elif event.action == "restart":
            self.run_worker(self._do_restart_service(event.service_name))
        elif event.action == "logs":
            self._selected_service = event.service_name
            self.run_worker(self._load_service_logs())

    async def _do_start_service(self, name: str) -> None:
        """Start a service."""
        if not self._service_manager:
            return

        definition = self._service_manager.get_service_definition(name)
        if not definition:
            self.notify(f"Unknown service: {name}", severity="error")
            return

        # Install if not already
        await self._service_manager.install(definition)

        success = await self._service_manager.start(name)
        if success:
            self.notify(f"Started {name}")
        else:
            self.notify(f"Failed to start {name}", severity="error")

        await self._refresh_services()

    async def _do_stop_service(self, name: str) -> None:
        """Stop a service."""
        if not self._service_manager:
            return

        success = await self._service_manager.stop(name)
        if success:
            self.notify(f"Stopped {name}")
        else:
            self.notify(f"Failed to stop {name}", severity="error")

        await self._refresh_services()

    async def _do_restart_service(self, name: str) -> None:
        """Restart a service."""
        if not self._service_manager:
            return

        success = await self._service_manager.restart(name)
        if success:
            self.notify(f"Restarted {name}")
        else:
            self.notify(f"Failed to restart {name}", severity="error")

        await self._refresh_services()

    async def _do_enable_service(self, name: str) -> None:
        """Enable auto-start for a service."""
        if not self._service_manager:
            return

        success = await self._service_manager.enable(name)
        if success:
            self.notify(f"Enabled auto-start for {name}")
        else:
            self.notify(f"Failed to enable {name}", severity="error")

        await self._refresh_services()

    async def _do_disable_service(self, name: str) -> None:
        """Disable auto-start for a service."""
        if not self._service_manager:
            return

        success = await self._service_manager.disable(name)
        if success:
            self.notify(f"Disabled auto-start for {name}")
        else:
            self.notify(f"Failed to disable {name}", severity="error")

        await self._refresh_services()

    async def _do_install_service(self, name: str) -> None:
        """Install a service."""
        if not self._service_manager:
            return

        definition = self._service_manager.get_service_definition(name)
        if not definition:
            self.notify(f"Unknown service: {name}", severity="error")
            return

        success = await self._service_manager.install(definition)
        if success:
            self.notify(f"Installed {name}")
        else:
            self.notify(f"Failed to install {name}", severity="error")

        await self._refresh_services()

    async def _do_uninstall_service(self, name: str) -> None:
        """Uninstall a service."""
        if not self._service_manager:
            return

        success = await self._service_manager.uninstall(name)
        if success:
            self.notify(f"Uninstalled {name}")
        else:
            self.notify(f"Failed to uninstall {name}", severity="error")

        await self._refresh_services()

    # Action methods

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_refresh(self) -> None:
        """Refresh service status."""
        self.run_worker(self._refresh_services())
        if self._selected_service:
            self.run_worker(self._load_service_logs())
        self.notify("Refreshed")

    def action_next_service(self) -> None:
        """Select next service."""
        service_list = self.query_one("#service-list", ServiceListWidget)
        service_list.select_next()

    def action_prev_service(self) -> None:
        """Select previous service."""
        service_list = self.query_one("#service-list", ServiceListWidget)
        service_list.select_previous()

    def action_start_service(self) -> None:
        """Start the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_start_service(self._selected_service))

    def action_stop_service(self) -> None:
        """Stop the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_stop_service(self._selected_service))

    def action_restart_service(self) -> None:
        """Restart the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_restart_service(self._selected_service))

    def action_enable_service(self) -> None:
        """Enable auto-start for the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_enable_service(self._selected_service))

    def action_disable_service(self) -> None:
        """Disable auto-start for the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_disable_service(self._selected_service))

    def action_install_service(self) -> None:
        """Install the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_install_service(self._selected_service))

    def action_uninstall_service(self) -> None:
        """Uninstall the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._do_uninstall_service(self._selected_service))

    def action_view_logs(self) -> None:
        """View logs for the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self.run_worker(self._load_service_logs())

    def action_follow_logs(self) -> None:
        """Toggle following logs for the selected service."""
        if not self._selected_service:
            self.notify("Select a service first", severity="warning")
            return
        self._following_logs = not self._following_logs
        if self._following_logs:
            self.notify(f"Following logs for {self._selected_service}")
            self.run_worker(self._stream_logs())
        else:
            self.notify("Stopped following logs")

    async def _stream_logs(self) -> None:
        """Stream logs for the selected service."""
        if not self._selected_service or not self._service_manager:
            return

        log_viewer = self.query_one("#log-area", ServiceLogViewer)

        async for line in self._service_manager.stream_logs(self._selected_service):
            if not self._following_logs:
                break
            log_viewer.append_line(line)

    def get_which_key_map(self):  # type: ignore[override]
        return {
            "s": ("start", "start_service"),
            "S": ("stop", "stop_service"),
            "R": ("restart", "restart_service"),
            "e": ("enable auto-start", "enable_service"),
            "d": ("disable auto-start", "disable_service"),
            "i": ("install", "install_service"),
            "u": ("uninstall", "uninstall_service"),
            "l": ("view logs", "view_logs"),
            "f": ("follow logs", "follow_logs"),
            "r": ("refresh", "refresh"),
            "q": ("back", "back"),
        }

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
        from hafs.ui.core.screen_router import get_screen_router

        route_map = {
            "dashboard": "/dashboard",
            "chat": "/chat",
            "logs": "/logs",
            "services": "/services",
            "analysis": "/analysis",
            "config": "/config",
        }
        route = route_map.get(event.screen)
        if route:
            router = get_screen_router()
            await router.navigate(route)
