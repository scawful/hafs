"""Modular Dashboard Screen for HAFS TUI.

This is the new modular dashboard that replaces the monolithic MainScreen.
It uses the core infrastructure (EventBus, StateStore, CommandRegistry,
BindingRegistry, NavigationController) for clean separation of concerns.

Architecture:
- DashboardScreen: Thin layout coordinator
- SidebarPanel: Collapsible sidebar with explorer, context summary, agents
- ContentPanel: Main content area with tabbed interface
- FooterPanel: Which-key bar and status

The screen delegates most logic to the core infrastructure and widgets,
keeping itself under 300 lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Static

from hafs.config.loader import load_config
from hafs.ui.core.command_registry import Command, CommandCategory, get_command_registry
from hafs.ui.core.event_bus import ContextEvent, get_event_bus
from hafs.ui.core.navigation_controller import get_navigation_controller
from hafs.ui.core.state_store import get_state_store
from hafs.ui.widgets.agent_status import AgentStatusWidget
from hafs.ui.widgets.context_summary import ContextSummaryWidget
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.dev_dashboard import DevDashboard
from hafs.ui.widgets.explorer import ExplorerWidget
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.which_key_bar import WhichKeyBar

if TYPE_CHECKING:
    from hafs.ui.widgets.protocol_widget import ProtocolWidget


class DashboardScreen(Screen):
    """Modular dashboard screen with project browser and context viewer.

    This screen uses the core infrastructure for:
    - Input handling via NavigationController
    - State management via StateStore
    - Commands via CommandRegistry
    - Events via EventBus

    The screen is intentionally thin, delegating most logic to widgets
    and the core infrastructure.
    """

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "command_palette", "Commands"),
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("ctrl+s", "save_file", "Save"),
    ]

    DEFAULT_CSS = """
    DashboardScreen {
        layout: vertical;
    }

    DashboardScreen #main-container {
        height: 1fr;
    }

    DashboardScreen #sidebar {
        width: 32;
        min-width: 0;
        max-width: 60;
        background: $surface;
        border-right: solid $primary-darken-2;
        height: 100%;
    }

    DashboardScreen #sidebar.collapsed {
        width: 0;
        min-width: 0;
        max-width: 0;
        border-right: none;
    }

    DashboardScreen #content {
        background: $background;
        height: 1fr;
    }

    DashboardScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary-darken-2;
    }

    DashboardScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        padding: 0 1;
    }

    DashboardScreen #which-key-bar {
        width: 2fr;
    }

    DashboardScreen Footer {
        width: auto;
        min-width: 24;
    }
    """

    # Reactive state
    sidebar_visible: reactive[bool] = reactive(True)
    sidebar_width: reactive[int] = reactive(32)

    def __init__(self) -> None:
        super().__init__()
        self._state = get_state_store()
        self._bus = get_event_bus()
        self._nav = get_navigation_controller()
        self._commands = get_command_registry()

        # Register screen-specific commands
        self._register_commands()

    def _register_commands(self) -> None:
        """Register dashboard-specific commands."""
        try:
            self._commands.register(Command(
                id="dashboard.toggle_sidebar",
                name="Toggle Sidebar",
                description="Show or hide the sidebar panel",
                handler=self.action_toggle_sidebar,
                category=CommandCategory.VIEW,
                keybinding="ctrl+b",
            ))
        except ValueError:
            pass  # Already registered

        try:
            self._commands.register(Command(
                id="dashboard.refresh",
                name="Refresh Dashboard",
                description="Refresh all dashboard data",
                handler=self.action_refresh,
                category=CommandCategory.VIEW,
                keybinding="r",
            ))
        except ValueError:
            pass

    def compose(self) -> ComposeResult:
        """Compose the modular dashboard layout."""
        yield HeaderBar(id="header-bar")

        with Horizontal(id="main-container"):
            # Sidebar panel
            with Vertical(id="sidebar"):
                config = load_config()
                yield ExplorerWidget(
                    workspace_dirs=config.general.workspace_directories,
                    id="explorer"
                )
                yield ContextSummaryWidget(id="context-summary")
                yield AgentStatusWidget(id="agent-status")

            # Main content area
            with Vertical(id="content"):
                yield DevDashboard(id="dev-dashboard")

        # Footer area
        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield WhichKeyBar(id="which-key-bar")
                yield Footer(compact=True, show_command_palette=False)

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        # Set navigation context
        self._nav.set_screen_context("dashboard")

        # Load sidebar state from store
        self.sidebar_visible = self._state.get("settings.sidebar_visible", True)

        # Subscribe to state changes
        self._state.subscribe("settings.sidebar_visible", self._on_sidebar_state_change)

        # Handle any pending file opens
        self._consume_pending_open()

        # Defer heavy loading to after first paint for snappier startup
        self.set_timer(0.1, self._deferred_load)

    def _deferred_load(self) -> None:
        """Load heavy data after initial render."""
        import asyncio

        # Refresh context stats in background
        try:
            context_summary = self.query_one("#context-summary", ContextSummaryWidget)
            asyncio.create_task(context_summary.refresh_stats())
        except Exception:
            pass

        # Refresh agent status in background
        try:
            agent_status = self.query_one("#agent-status", AgentStatusWidget)
            asyncio.create_task(agent_status.refresh_status())
        except Exception:
            pass

    def on_show(self) -> None:
        """Handle screen being shown."""
        self._consume_pending_open()

    def _consume_pending_open(self) -> None:
        """Open any file requested by another screen."""
        path = getattr(self.app, "_pending_open_path", None)
        if not isinstance(path, Path):
            return

        try:
            delattr(self.app, "_pending_open_path")
        except Exception:
            setattr(self.app, "_pending_open_path", None)

        if not path.exists():
            self.notify(f"File not found: {path}", severity="warning")
            return

        self._open_file(path)

    def _open_file(self, path: Path) -> None:
        """Open a file in the context viewer."""
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
        except Exception:
            pass

        try:
            viewer = self.query_one("#context-viewer", ContextViewer)
            viewer.set_file(path)

            # Publish context event
            self._bus.publish(ContextEvent(action="open", path=str(path)))
        except Exception as e:
            self.notify(f"Failed to open: {e}", severity="error")

    def _on_sidebar_state_change(self, path: str, old: bool, new: bool) -> None:
        """Handle sidebar visibility state change."""
        self.sidebar_visible = new

    def watch_sidebar_visible(self, visible: bool) -> None:
        """React to sidebar visibility changes."""
        try:
            sidebar = self.query_one("#sidebar", Vertical)
            if visible:
                sidebar.remove_class("collapsed")
                sidebar.styles.width = self.sidebar_width
            else:
                sidebar.add_class("collapsed")
                sidebar.styles.width = 0
        except Exception:
            pass

    # Event handlers

    def on_header_bar_menu_selected(self, event: HeaderBar.MenuSelected) -> None:
        """Handle header bar menu selections."""
        if event.menu_id == "palette":
            self.action_command_palette()
        elif event.menu_id == "context":
            self._open_context_chat()

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
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
            from hafs.ui.core.screen_router import get_screen_router
            router = get_screen_router()
            await router.navigate(route)

    def on_explorer_widget_file_selected(self, event: ExplorerWidget.FileSelected) -> None:
        """Handle file selection from explorer."""
        self._open_file(event.path)

    def on_explorer_widget_project_selected(self, event: ExplorerWidget.ProjectSelected) -> None:
        """Handle project selection from explorer."""
        try:
            self.query_one("#dev-dashboard", DevDashboard).active = "tab-context"
            viewer = self.query_one("#context-viewer", ContextViewer)
            viewer.set_project(event.project)
        except Exception:
            pass

    def on_context_viewer_file_saved(self, event: ContextViewer.FileSaved) -> None:
        """Handle file save events."""
        self.notify(f"Saved {event.path.name}")
        self._bus.publish(ContextEvent(action="save", path=str(event.path)))
        self._refresh_explorer()

    def on_context_viewer_file_error(self, event: ContextViewer.FileError) -> None:
        """Handle file operation errors."""
        self.notify(f"{event.error}", severity="error")

    def on_agent_status_widget_chat_requested(self, event: AgentStatusWidget.ChatRequested) -> None:
        """Handle chat request from agent status widget."""
        self._open_context_chat()

    # Actions

    def action_refresh(self) -> None:
        """Refresh all dashboard data."""
        self._refresh_explorer()

        try:
            import asyncio
            context_summary = self.query_one("#context-summary", ContextSummaryWidget)
            asyncio.create_task(context_summary.refresh_stats())
        except Exception:
            pass

        try:
            import asyncio
            agent_status = self.query_one("#agent-status", AgentStatusWidget)
            asyncio.create_task(agent_status.refresh_status())
        except Exception:
            pass

        self.notify("Dashboard refreshed", timeout=1)

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_toggle_sidebar(self) -> None:
        """Toggle sidebar visibility."""
        self.sidebar_visible = not self.sidebar_visible
        self._state.set("settings.sidebar_visible", self.sidebar_visible)
        status = "visible" if self.sidebar_visible else "hidden"
        self.notify(f"Sidebar {status}", timeout=1)

    def action_command_palette(self) -> None:
        """Open command palette."""
        from hafs.ui.screens.command_palette import CommandPalette
        self.app.push_screen(CommandPalette())

    def action_save_file(self) -> None:
        """Save current file if editing."""
        try:
            viewer = self.query_one("#context-viewer", ContextViewer)
            if viewer.is_editing:
                viewer.save_edits()
            else:
                self.notify("No file being edited", severity="warning")
        except Exception:
            pass

    def _open_context_chat(self) -> None:
        """Open chat with context selection."""
        from hafs.ui.screens.context_selection_modal import ContextSelectionModal
        from hafs.ui.screens.orchestrator import OrchestratorScreen

        def on_context_selected(paths: list[Path] | None) -> None:
            if not paths:
                return

            coordinator = getattr(self.app, "_coordinator", None)
            screen = OrchestratorScreen(coordinator=coordinator, context_paths=paths)
            self.app.push_screen(screen)

            if coordinator:
                coordinator.set_context_items(paths)

        self.app.push_screen(ContextSelectionModal(), on_context_selected)

    def _refresh_explorer(self) -> None:
        """Refresh the explorer widget."""
        try:
            explorer = self.query_one(ExplorerWidget)
            if hasattr(explorer, "refresh_data"):
                explorer.refresh_data()
        except Exception:
            pass
