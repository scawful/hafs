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
        
        # Load theme
        from hafs.ui.theme import HalextTheme
        self.halext_theme = HalextTheme(self.config.theme)
        
        super().__init__()

    def get_css_variables(self) -> dict[str, str]:
        """Get CSS variables for the theme."""
        # Parse the TCSS variables string into a dictionary
        vars_dict = {}
        tcss = self.halext_theme.get_tcss_variables()
        for line in tcss.strip().split(";"):
            if ":" in line:
                key, value = line.split(":", 1)
                vars_dict[key.strip().lstrip("$")] = value.strip()
        
        # Add derived colors if needed
        vars_dict["primary-darken-1"] = vars_dict["primary"]  # Fallback
        vars_dict["primary-darken-2"] = vars_dict["primary"]  # Fallback
        
        # Ensure 'foreground' exists (mapped from 'text')
        if "text" in vars_dict:
            vars_dict["foreground"] = vars_dict["text"]
        
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

            # Register initial agents
            if self._initial_agents:
                for agent_spec in self._initial_agents:
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
        self.notify(
            "HAFS - Halext Agentic File System\n\n"
            "Keys:\n"
            "  1 - Dashboard\n"
            "  2 - Logs\n"
            "  3 - Settings\n"
            "  4 - Orchestrate\n"
            "  r - Refresh\n"
            "  q - Quit\n\n"
            "Navigate with arrow keys, Enter to select.",
            title="Help",
            timeout=10,
        )


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
