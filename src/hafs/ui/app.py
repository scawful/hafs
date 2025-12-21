"""Main HAFS TUI Application."""

from __future__ import annotations

import logging
from pathlib import Path
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

    # Note: CSS_PATH disabled - using Textual theme system instead
    # Widget-specific styles should use DEFAULT_CSS on individual widgets

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "switch_main", "Dashboard", show=True),
        Binding("2", "switch_logs", "Logs", show=True),
        Binding("3", "switch_settings", "Settings", show=True),
        Binding("4", "switch_chat", "Chat", show=True),
        Binding("5", "switch_workspace", "Workspace", show=True),
        Binding("6", "switch_services", "Services", show=True),
        Binding("7", "switch_analysis", "Analysis", show=True),
        Binding("8", "switch_config", "Config", show=True),
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

        # Load custom halext theme (provides $info, $text-muted, etc.)
        from hafs.ui.theme import HalextTheme

        self.halext_theme = HalextTheme(config=self.config.theme)

        super().__init__()

        # Register our custom theme (has all the CSS variables our widgets need)
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

        # Connect command handlers to app actions
        self._connect_command_handlers()

        logger.info(f"Core infrastructure initialized (modular={self._use_modular})")

    def _connect_command_handlers(self) -> None:
        """Connect default command handlers to app actions."""
        # Map command IDs to app methods
        command_handlers = {
            "nav.dashboard": lambda: self.run_action("switch_main"),
            "nav.chat": lambda: self.run_action("switch_chat"),
            "nav.workspace": lambda: self.run_action("switch_workspace"),
            "nav.logs": lambda: self.run_action("switch_logs"),
            "nav.settings": lambda: self.run_action("switch_settings"),
            "nav.services": lambda: self.run_action("switch_services"),
            "view.refresh": lambda: self.run_action("refresh"),
            "help.show": lambda: self.run_action("help"),
            "system.quit": lambda: self.run_action("quit"),
        }

        for cmd_id, handler in command_handlers.items():
            cmd = self._commands.get(cmd_id)
            if cmd:
                cmd.handler = handler

        # Connect theme commands
        themes = [
            "halext",
            "halext-light",
            "nord",
            "nord-light",
            "dracula",
            "dracula-light",
            "gruvbox",
            "gruvbox-light",
            "solarized",
            "solarized-light",
        ]
        for theme_id in themes:
            cmd_id = f"view.theme_{theme_id.replace('-', '_')}"
            cmd = self._commands.get(cmd_id)
            if cmd:
                # Use default arg to capture theme_id value
                cmd.handler = lambda t=theme_id: self._set_theme(t)

    async def _set_theme(self, theme_name: str) -> None:
        """Set the application theme."""
        from hafs.ui.theme import HalextTheme
        from hafs.config.schema import ThemeConfig, ThemeVariant
        from hafs.ui.theme_presets import get_preset_names
        from textual.theme import BUILTIN_THEMES

        # 1. Check if it's a legacy preset
        if theme_name in get_preset_names():
            # It's one of our custom presets
            config = ThemeConfig(preset=theme_name, variant=ThemeVariant.DARK)
            self.halext_theme = HalextTheme(config=config)
            self.register_theme(self.halext_theme.create_textual_theme())
            self.theme = "hafs-halext"

        elif (
            theme_name.endswith("-light") and theme_name.replace("-light", "") in get_preset_names()
        ):
            # Light variant of our preset
            preset = theme_name.replace("-light", "")
            config = ThemeConfig(preset=preset, variant=ThemeVariant.LIGHT)
            self.halext_theme = HalextTheme(config=config)
            self.register_theme(self.halext_theme.create_textual_theme())
            self.theme = "hafs-halext"

        # 2. Check if it's a builtin Textual theme
        elif theme_name in BUILTIN_THEMES:
            self.theme = theme_name
            # We need to update our adapter to reflect this new theme's colors
            # ensuring all $variables are updated for our widgets
            # Note: app.theme is just a string name. We need the object.
            active_theme = BUILTIN_THEMES[theme_name]
            self.halext_theme = HalextTheme(theme=active_theme)

        else:
            self.notify(f"Unknown theme '{theme_name}'", severity="error", timeout=3)
            return

        # Regenerate and apply CSS variables for the active theme
        # This fixes the "missing variable" issues for native themes
        css_vars = self.halext_theme.get_tcss_variables()
        # Parse the CSS block back into a dict for mapping (optimization opportunity: make get_tcss_variables return dict)
        # Since we optimized get_tcss_variables to return a dict, we can use it directly.
        # But wait, looking at theme.py replacement, it now returns a dict[str, str].
        # We need to inject these variables into the app's stylesheet.

        # NOTE: Textual doesn't have a public API to set multiple variables at runtime easily
        # without reloading CSS. However, we can use `self.stylesheet.set_variable` if available,
        # or we might rely on the fact that we updated `self.halext_theme` and if any widgets
        # were binding to it, they need a refresh.
        # But variables like $info are resolved at TCSS parsing time often?
        # Actually, variables in Textual 0.40+ are dynamic.

        # We need to manually inject these variables because they are not part of the standard theme definition.
        # There isn't a clean public API for bulk variable setting on the App in older Textual versions,
        # but let's check what we can do.

        # Approach: We can try to re-parse dynamic CSS.
        # Or simpler: The HalextTheme generates a `hafs-halext` theme.
        # If we selected a builtin theme (e.g. 'dracula'), we just set `self.theme = 'dracula'`.
        # BUT our widgets rely on `$info`. 'dracula' doesn't define `$info`.
        # So we MUST inject `$info`.

        # HACK: Re-define 'hafs-halext' to match the selected built-in theme + our variables,
        # and ALWAYS use 'hafs-halext'.
        # This effectively aliases the built-in theme into our custom theme wrapper.

        target_theme_obj = self.halext_theme.create_textual_theme()
        # Override the name to always be our internal usage name
        # Actually replace_file_content for theme.py sets name to `hafs-{preset_name}`.
        # Let's force it to standard name so we can switch to it.

        # Wait, if we use BUILTIN_THEMES, we want to benefit from it.
        # But if we rely on $info, we can't JUST use builtin themes without extending them.
        # So treating them as a source logic for HalextTheme is the correct approach.

        # Re-register our theme with the new colors derived from the builtin
        self.register_theme(target_theme_obj)
        self.theme = target_theme_obj.name

        self.refresh(layout=True)
        self.notify(f"Theme: {self.halext_theme.preset_name}", timeout=2)

    def register_widget_plugin(self, plugin: "WidgetPlugin") -> None:
        """Register a widget plugin.

        Args:
            plugin: The widget plugin to register.
        """
        self.widget_plugins.append(plugin)

    def get_css_variables(self) -> dict[str, str]:
        """Get CSS variables for the theme."""
        return self.halext_theme.get_tcss_variables()

    async def on_mount(self) -> None:
        """Initialize app on mount."""
        # Load and activate plugins
        for plugin_name in self.plugin_loader.discover_plugins():
            self.plugin_loader.activate_plugin(plugin_name, self)

        # Push initial screen directly (router used for subsequent navigation)
        if self._use_modular:
            from hafs.ui.screens.dashboard import DashboardScreen
            from hafs.ui.screens.chat import ChatScreen

            if self._orchestrator_mode:
                self.push_screen(ChatScreen())
            else:
                self.push_screen(DashboardScreen())
        else:
            if self._orchestrator_mode:
                self.push_screen(OrchestratorScreen(coordinator=None))
            else:
                self.push_screen(MainScreen())

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

    async def action_switch_workspace(self) -> None:
        """Switch to high-performance workspace screen."""
        await self._router.navigate("/workspace")

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
