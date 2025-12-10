"""Main HAFS TUI Application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding

from hafs.config.loader import load_config
from hafs.ui.screens.main import MainScreen
from hafs.ui.screens.logs import LogsScreen
from hafs.ui.screens.settings import SettingsScreen


class HafsApp(App):
    """HAFS - Halext Agentic File System TUI.

    A terminal user interface for managing AFS context directories
    and browsing AI agent logs.
    """

    TITLE = "HAFS - Halext AFS Manager"
    SUB_TITLE = "Agentic File System"

    CSS_PATH = Path(__file__).parent / "styles.tcss"

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("1", "switch_main", "Dashboard", show=True),
        Binding("2", "switch_logs", "Logs", show=True),
        Binding("3", "switch_settings", "Settings", show=True),
        Binding("4", "switch_orchestrator", "Orchestrate", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("?", "help", "Help", show=True),
    ]

    def __init__(
        self,
        orchestrator_mode: bool = False,
        initial_agents: list[dict[str, str]] | None = None,
        default_backend: str = "gemini",
    ) -> None:
        self.config = load_config()
        self._orchestrator_mode = orchestrator_mode
        self._initial_agents = initial_agents
        self._default_backend = default_backend
        self._coordinator = None

        # Load and register theme
        from hafs.ui.theme import HalextTheme
        self.halext_theme = HalextTheme(self.config.theme)

        super().__init__()

        # Register custom theme with Textual
        self.register_theme(HalextTheme.create_textual_theme())
        self.theme = "hafs-halext"

    def get_css_variables(self) -> dict[str, str]:
        """Get CSS variables for the theme.

        Returns extended set of CSS variables from HalextTheme.
        """
        from hafs.ui.theme import HalextTheme

        # Build variables dict directly from theme class for reliability
        vars_dict = {
            # Primary colors
            "primary": HalextTheme.PRIMARY,
            "primary-darken-1": HalextTheme.PRIMARY,
            "primary-darken-2": HalextTheme.PRIMARY,
            "secondary": HalextTheme.SECONDARY,
            "accent": HalextTheme.ACCENT,
            # Backgrounds
            "background": HalextTheme.BACKGROUND,
            "surface": HalextTheme.SURFACE,
            "surface-highlight": HalextTheme.SURFACE_HIGHLIGHT,
            "panel": HalextTheme.SURFACE,
            # Text
            "text": HalextTheme.TEXT,
            "foreground": HalextTheme.TEXT,
            "text-muted": HalextTheme.TEXT_MUTED,
            # Status
            "success": HalextTheme.SUCCESS,
            "warning": HalextTheme.WARNING,
            "error": HalextTheme.ERROR,
            "info": HalextTheme.INFO,
            # Policy
            "policy-readonly": HalextTheme.POLICY_READ_ONLY,
            "policy-writable": HalextTheme.POLICY_WRITABLE,
            "policy-executable": HalextTheme.POLICY_EXECUTABLE,
            # Scrollbar variables (required by Textual's Widget.DEFAULT_CSS)
            "scrollbar-background": HalextTheme.SURFACE,
            "scrollbar-background-hover": HalextTheme.SURFACE,
            "scrollbar-background-active": HalextTheme.SURFACE,
            "scrollbar-color": HalextTheme.PRIMARY,
            "scrollbar-color-hover": HalextTheme.SECONDARY,
            "scrollbar-color-active": HalextTheme.SECONDARY,
            "scrollbar": HalextTheme.PRIMARY,
            "scrollbar-hover": HalextTheme.SECONDARY,
            "scrollbar-active": HalextTheme.SECONDARY,
            "scrollbar-corner-color": HalextTheme.SURFACE,
        }

        return vars_dict

    async def on_mount(self) -> None:
        """Initialize app on mount."""
        if self._orchestrator_mode:
            # Show screen immediately with None coordinator
            from hafs.ui.screens.orchestrator import OrchestratorScreen
            
            screen = OrchestratorScreen(coordinator=None)
            self.push_screen(screen)
            
            # Run initialization in background worker
            self.run_worker(self._init_orchestrator(screen))
        else:
            self.push_screen(MainScreen())

    async def _init_orchestrator(self, screen: "OrchestratorScreen") -> None:
        """Initialize orchestrator components in background.
        
        Args:
            screen: The active OrchestratorScreen to update.
        """
        try:
            from hafs.agents.coordinator import AgentCoordinator
            from hafs.models.agent import AgentRole

            # Initialize coordinator
            self._coordinator = AgentCoordinator(self.config)

            # Register initial agents (or defaults)
            agents_to_init = self._initial_agents
            if not agents_to_init:
                agents_to_init = [
                    {"name": "Planner", "role": "planner"},
                    {"name": "Coder", "role": "coder"},
                    {"name": "Critic", "role": "critic"},
                ]

            for agent_spec in agents_to_init:
                try:
                    role = AgentRole(agent_spec.get("role", "general"))
                    await self._coordinator.register_agent(
                        name=agent_spec["name"],
                        role=role,
                        backend_name=self._default_backend,
                    )
                except Exception as e:
                    self.notify(
                        f"Failed to register agent {agent_spec['name']}: {e}",
                        severity="error",
                    )
            
            # Update screen with ready coordinator
            await screen.set_coordinator(self._coordinator)
            
        except ImportError:
            self.notify("Failed to load agent modules", severity="error")
        except Exception as e:
            self.notify(f"Orchestrator initialization failed: {e}", severity="error")

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_switch_main(self) -> None:
        """Switch to main dashboard screen."""
        # Clear screen stack and push main
        while len(self.screen_stack) > 1:
            self.pop_screen()
        if not isinstance(self.screen, MainScreen):
            self.switch_screen(MainScreen())

    def action_switch_logs(self) -> None:
        """Switch to logs browser screen."""
        if not isinstance(self.screen, LogsScreen):
            self.push_screen(LogsScreen())

    def action_switch_settings(self) -> None:
        """Switch to settings screen."""
        if not isinstance(self.screen, SettingsScreen):
            self.push_screen(SettingsScreen())

    def action_switch_orchestrator(self) -> None:
        """Switch to orchestrator screen."""
        from hafs.ui.screens.orchestrator import OrchestratorScreen

        if not isinstance(self.screen, OrchestratorScreen):
            self.push_screen(OrchestratorScreen(coordinator=self._coordinator))

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


def run() -> None:
    """Entry point for running the TUI."""
    app = HafsApp()
    app.run()


def run_orchestrator(
    default_backend: str = "gemini",
    agents: list[dict[str, str]] | None = None,
) -> None:
    """Entry point for running the orchestrator TUI.

    Args:
        default_backend: Default backend to use for new agents.
        agents: List of agents to start (name, role dicts).
    """
    app = HafsApp(
        orchestrator_mode=True,
        initial_agents=agents,
        default_backend=default_backend,
    )
    app.run()


if __name__ == "__main__":
    run()
