"""Main HAFS TUI Application."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding

from hafs.config.loader import load_config
from hafs.plugins.loader import PluginLoader
from hafs.plugins.protocol import WidgetPlugin  # Explicitly import WidgetPlugin

# Core infrastructure
from hafs.ui.core.event_bus import get_event_bus, reset_event_bus
from hafs.ui.core.state_store import get_state_store, reset_state_store
from hafs.ui.core.command_registry import get_command_registry, reset_command_registry
from hafs.ui.core.binding_registry import get_binding_registry, reset_binding_registry
from hafs.ui.core.screen_router import (
    get_screen_router,
    reset_screen_router,
    register_default_routes,
)
from hafs.ui.core.navigation_controller import (
    get_navigation_controller,
    reset_navigation_controller,
)
from hafs.ui.core.accessibility import get_accessibility, reset_accessibility

# Screens (legacy)
from hafs.ui.screens.logs import LogsScreen
from hafs.ui.screens.main import MainScreen
from hafs.ui.screens.orchestrator import OrchestratorScreen
from hafs.ui.screens.services import ServicesScreen
from hafs.ui.screens.settings import SettingsScreen

if TYPE_CHECKING:
    from hafs.agents.coordinator import AgentCoordinator

logger = logging.getLogger(__name__)


class HafsApp(App):
    """HAFS - Halext Agentic File System TUI.

    A terminal user interface for managing AFS context directories
    and browsing AI agent logs.
    """

    TITLE = "HAFS - Halext AFS Manager"
    SUB_TITLE = "Agentic File System"

    # CSS_PATH disabled - conflicts with Textual 6.x theme system
    # CSS_PATH = Path(__file__).parent / "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "switch_main", "Dashboard", show=True),
        Binding("2", "switch_logs", "Logs", show=True),
        Binding("3", "switch_settings", "Settings", show=True),
        Binding("4", "switch_chat", "Chat", show=True),
        Binding("5", "switch_services", "Services", show=True),
        Binding("6", "switch_analysis", "Analysis", show=True),
        Binding("7", "switch_config", "Config", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("?", "help", "Help", show=True),
    ]

    def __init__(
        self,
        orchestrator_mode: bool = False,
        initial_agents: list[dict[str, str]] | None = None,
        default_backend: str = "gemini",
        use_modular_screens: bool = True,
    ) -> None:
        self.config = load_config()
        self._orchestrator_mode = orchestrator_mode
        self._initial_agents = initial_agents
        self._default_backend = default_backend
        self._use_modular = use_modular_screens
        self._coordinator: "AgentCoordinator | None" = None
        self.widget_plugins: list["WidgetPlugin"] = []

        # Initialize core infrastructure
        self._init_core_infrastructure()

        # Load and register theme
        from hafs.ui.theme import HalextTheme
        self.halext_theme = HalextTheme(self.config.theme)

        super().__init__()

        # Register custom theme with Textual
        self.register_theme(self.halext_theme.create_textual_theme())
        self.theme = "hafs-halext"

        # Initialize plugin loader
        self.plugin_loader = PluginLoader(
            plugin_dirs=getattr(self.config.plugins, "plugin_dirs", None)
            if hasattr(self.config, "plugins")
            else None
        )

    def _init_core_infrastructure(self) -> None:
        """Initialize the core TUI infrastructure."""
        # Reset singletons for clean state
        reset_event_bus()
        reset_state_store()
        reset_command_registry()
        reset_binding_registry()
        reset_screen_router()
        reset_navigation_controller()
        reset_accessibility()

        # Get fresh instances
        self._event_bus = get_event_bus()
        self._state_store = get_state_store()
        self._commands = get_command_registry()
        self._bindings = get_binding_registry()
        self._router = get_screen_router()
        self._nav_controller = get_navigation_controller()
        self._accessibility = get_accessibility()

        # Register default routes
        register_default_routes(self._router, use_modular=self._use_modular)

        # Set app reference on router
        self._router.set_app(self)

        logger.info(f"Core infrastructure initialized (modular={self._use_modular})")

    def register_widget_plugin(self, plugin: "WidgetPlugin") -> None:
        """Register a widget plugin.

        Args:
            plugin: The widget plugin to register.
        """
        self.widget_plugins.append(plugin)

    def get_css_variables(self) -> dict[str, str]:
        """Get CSS variables for the theme.

        Returns extended set of CSS variables from HalextTheme.
        """
        # Parse the TCSS variables string from the theme instance
        vars_dict = {}
        tcss = self.halext_theme.get_tcss_variables()
        for line in tcss.strip().split(";"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lstrip("$")
                value = value.strip()
                vars_dict[key] = value
        return vars_dict

    async def on_mount(self) -> None:
        """Initialize app on mount."""
        # Load and activate plugins
        for plugin_name in self.plugin_loader.discover_plugins():
            self.plugin_loader.activate_plugin(plugin_name, self)

        if self._orchestrator_mode:
            # Navigate to chat screen
            await self._router.navigate("/chat")
        else:
            # Navigate to dashboard
            await self._router.navigate("/dashboard")

    async def action_quit(self) -> None:
        """Quit the application."""
        if self._coordinator:
            try:
                self._coordinator.complete_session()
            except Exception:
                pass
        self.exit()

    async def action_switch_main(self) -> None:
        """Switch to main dashboard screen."""
        await self._router.navigate("/dashboard")

    async def action_switch_logs(self) -> None:
        """Switch to logs browser screen."""
        await self._router.navigate("/logs")

    async def action_switch_settings(self) -> None:
        """Switch to settings screen."""
        await self._router.navigate("/settings")

    async def action_switch_chat(self) -> None:
        """Switch to multi-agent chat screen."""
        await self._router.navigate("/chat")

    async def action_switch_services(self) -> None:
        """Switch to services management screen."""
        await self._router.navigate("/services")

    async def action_switch_analysis(self) -> None:
        """Switch to analysis dashboard screen."""
        await self._router.navigate("/analysis")

    async def action_switch_config(self) -> None:
        """Switch to configuration screen."""
        await self._router.navigate("/config")

    def action_refresh(self) -> None:
        """Refresh current screen data."""
        if hasattr(self.screen, "action_refresh"):
            self.screen.action_refresh()  # type: ignore[attr-defined]

    def action_help(self) -> None:
        """Show help information."""
        from hafs.ui.screens.help_modal import HelpModal

        # Get current screen name for context-aware help
        current_screen_name = type(self.screen).__name__
        self.push_screen(HelpModal(current_screen_name))


def run(use_modular: bool = True) -> None:
    """Entry point for running the TUI.

    Args:
        use_modular: If True, use new modular screens. If False, use legacy screens.
    """
    app = HafsApp(use_modular_screens=use_modular)
    app.run()


def run_chat(
    default_backend: str = "gemini",
    agents: list[dict[str, str]] | None = None,
    use_modular: bool = True,
) -> None:
    """Entry point for running the multi-agent chat TUI.

    Args:
        default_backend: Default backend to use for new agents.
        agents: List of agents to start (name, role dicts).
        use_modular: If True, use new modular screens. If False, use legacy screens.
    """
    app = HafsApp(
        orchestrator_mode=True,
        initial_agents=agents,
        default_backend=default_backend,
        use_modular_screens=use_modular,
    )
    app.run()


# Backwards compatibility alias
run_orchestrator = run_chat


if __name__ == "__main__":
    run()
