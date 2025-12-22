"""Main HAFS TUI Application."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from textual.app import App
from textual.binding import Binding
from textual.theme import BUILTIN_THEMES, Theme

from config.loader import load_config
from config.schema import ThemeConfig, ThemeColors, ThemeVariant
from plugins.loader import PluginLoader
from plugins.protocol import WidgetPlugin  # Explicitly import WidgetPlugin

# Core infrastructure
from tui.core.event_bus import get_event_bus, reset_event_bus
from tui.core.state_store import get_state_store, reset_state_store
from tui.core.command_registry import get_command_registry, reset_command_registry
from tui.core.binding_registry import get_binding_registry, reset_binding_registry
from tui.core.screen_router import (
    get_screen_router,
    reset_screen_router,
    register_default_routes,
)
from tui.core.navigation_controller import (
    get_navigation_controller,
    reset_navigation_controller,
)
from tui.core.accessibility import get_accessibility, reset_accessibility

# Screens (legacy - only import those actually used directly)
from tui.screens.main import MainScreen
from tui.screens.orchestrator import OrchestratorScreen

if TYPE_CHECKING:
    from agents.core.coordinator import AgentCoordinator

logger = logging.getLogger(__name__)


DEFAULT_HALEXT_DARK = {
    "primary": "#4C3B52",
    "secondary": "#9B59B6",
    "accent": "#E74C3C",
    "background": "#000000",
    "surface": "#1F1F35",
    "panel": "#2A2A4E",
    "success": "#27AE60",
    "warning": "#F39C12",
    "error": "#E74C3C",
}

DEFAULT_HALEXT_LIGHT = {
    "primary": "#8E44AD",
    "secondary": "#6B4E74",
    "accent": "#C0392B",
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "panel": "#F0F0F0",
    "success": "#27AE60",
    "warning": "#E67E22",
    "error": "#C0392B",
}


def _theme_variables_from_custom(colors: ThemeColors) -> dict[str, str]:
    return {
        "text": colors.text,
        "text-muted": colors.text_muted,
        "text-disabled": colors.text_muted,
        "border": colors.border,
        "border-blurred": colors.border,
    }


def _build_halext_theme(
    config: Optional[ThemeConfig],
    *,
    light: bool,
) -> Theme:
    name = "halext-light" if light else "halext"
    base = DEFAULT_HALEXT_LIGHT if light else DEFAULT_HALEXT_DARK

    if config and config.custom:
        colors = config.custom
        return Theme(
            name=name,
            primary=colors.primary,
            secondary=colors.secondary,
            accent=colors.accent,
            background=colors.background,
            surface=colors.surface,
            panel=colors.surface_highlight,
            foreground=colors.text,
            success=colors.success,
            warning=colors.warning,
            error=colors.error,
            dark=not light,
            variables=_theme_variables_from_custom(colors),
        )

    primary = config.primary if config else None
    secondary = config.secondary if config else None
    accent = config.accent if config else None

    return Theme(
        name=name,
        primary=primary or base["primary"],
        secondary=secondary or base["secondary"],
        accent=accent or base["accent"],
        background=base["background"],
        surface=base["surface"],
        panel=base["panel"],
        success=base["success"],
        warning=base["warning"],
        error=base["error"],
        dark=not light,
    )


class HafsApp(App):
    """HAFS - Halext Agentic File System TUI.

    A terminal user interface for managing AFS context directories
    and browsing AI agent logs.
    """

    TITLE = "HAFS - Halext AFS Manager"
    SUB_TITLE = "Agentic File System"

    # Universal keybindings that work everywhere
    # Navigation is handled via SPC (which-key) on screens
    BINDINGS = [
        # Universal shortcuts (always available)
        Binding("r", "refresh", "Refresh", show=True),
        Binding("?", "help", "Help", show=True),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("escape", "back_or_cancel", "Back", show=False),
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

        super().__init__()

        # Register custom Halext themes (respect config overrides)
        theme_config = getattr(self.config, "theme", None)
        self.register_theme(_build_halext_theme(theme_config, light=False))
        self.register_theme(_build_halext_theme(theme_config, light=True))

        # Set initial theme (use config preset + variant)
        self._set_theme(self._resolve_theme_name(theme_config), notify=False)

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

        # Connect theme commands - use Textual's built-in theme names
        themes = [
            "halext",
            "halext-light",
            "textual-dark",
            "textual-light",
            "nord",
            "dracula",
            "gruvbox",
            "tokyo-night",
            "monokai",
            "solarized-light",
        ]
        for theme_id in themes:
            cmd_id = f"view.theme_{theme_id.replace('-', '_')}"
            cmd = self._commands.get(cmd_id)
            if cmd:
                cmd.handler = lambda t=theme_id: self._set_theme(t)

    def _resolve_theme_name(
        self,
        preset: ThemeConfig | str | None,
        variant: ThemeVariant | str | None = None,
    ) -> str:
        if isinstance(preset, ThemeConfig):
            variant = preset.variant
            preset_value = preset.preset
        else:
            preset_value = preset

        normalized = (preset_value or "").strip().lower().replace("_", "-")
        if not normalized:
            normalized = "halext"

        is_light = False
        if variant is not None:
            if isinstance(variant, ThemeVariant):
                is_light = variant == ThemeVariant.LIGHT
            else:
                is_light = str(variant).strip().lower() == "light"

        if normalized in {"halext", "halext-light"}:
            return "halext-light" if normalized == "halext" and is_light else normalized

        if normalized == "solarized":
            normalized = "solarized-light" if is_light else "textual-dark"
        elif is_light and f"{normalized}-light" in BUILTIN_THEMES:
            normalized = f"{normalized}-light"
        elif not is_light and f"{normalized}-dark" in BUILTIN_THEMES:
            normalized = f"{normalized}-dark"

        available = set(BUILTIN_THEMES) | {"halext", "halext-light"}
        if normalized not in available:
            fallback = "halext-light" if is_light else "halext"
            logger.warning("Unknown theme '%s'; falling back to '%s'", preset_value, fallback)
            return fallback

        return normalized

    def _set_theme(self, theme_name: str, *, notify: bool = True) -> None:
        """Set the application theme."""
        resolved = self._resolve_theme_name(theme_name)
        try:
            self.theme = resolved
            self.refresh(layout=True)
            if notify:
                if resolved != theme_name:
                    self.notify(f"Theme: {resolved} (from {theme_name})", timeout=2)
                else:
                    self.notify(f"Theme: {resolved}", timeout=2)
        except Exception as e:
            if notify:
                self.notify(f"Unknown theme '{theme_name}': {e}", severity="error", timeout=3)

    def register_widget_plugin(self, plugin: "WidgetPlugin") -> None:
        """Register a widget plugin.

        Args:
            plugin: The widget plugin to register.
        """
        self.widget_plugins.append(plugin)

    async def on_mount(self) -> None:
        """Initialize app on mount."""
        # Load and activate plugins
        for plugin_name in self.plugin_loader.discover_plugins():
            self.plugin_loader.activate_plugin(plugin_name, self)

        # Push initial screen directly (router used for subsequent navigation)
        if self._use_modular:
            from tui.screens.dashboard import DashboardScreen
            from tui.screens.chat import ChatScreen

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

    async def action_switch_training(self) -> None:
        """Switch to training data dashboard screen."""
        from tui.screens.training_dashboard import TrainingDashboardScreen

        self.push_screen(TrainingDashboardScreen())

    def action_refresh(self) -> None:
        """Refresh current screen data."""
        if hasattr(self.screen, "action_refresh"):
            self.screen.action_refresh()  # type: ignore[attr-defined]

    def action_help(self) -> None:
        """Show help information."""
        from tui.screens.help_modal import HelpModal

        # Get current screen name for context-aware help
        current_screen_name = type(self.screen).__name__
        self.push_screen(HelpModal(current_screen_name))

    def action_command_palette(self) -> None:
        """Open the command palette."""
        from tui.screens.command_palette import CommandPalette

        self.push_screen(CommandPalette())

    def action_back_or_cancel(self) -> None:
        """Go back or cancel current operation."""
        # If we're not on the base screen, pop the current screen
        if len(self.screen_stack) > 1:
            self.pop_screen()
        else:
            # On base screen, do nothing or show a hint
            pass


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
