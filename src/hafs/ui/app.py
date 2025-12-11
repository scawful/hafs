"""Main HAFS TUI Application."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App
from textual.binding import Binding

from hafs.config.loader import load_config
from hafs.plugins.protocol import WidgetPlugin  # Explicitly import WidgetPlugin
from hafs.ui.screens.logs import LogsScreen
from hafs.ui.screens.main import MainScreen
from hafs.ui.screens.orchestrator import OrchestratorScreen
from hafs.ui.screens.settings import SettingsScreen

if TYPE_CHECKING:
    from hafs.agents.coordinator import AgentCoordinator


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
        self._coordinator: "AgentCoordinator | None" = None
        self.widget_plugins: list["WidgetPlugin"] = []

        # Load and register theme
        from hafs.ui.theme import HalextTheme
        self.halext_theme = HalextTheme(self.config.theme)

        super().__init__()

        # Register custom theme with Textual
        self.register_theme(self.halext_theme.create_textual_theme())
        self.theme = "hafs-halext"

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
        if self._orchestrator_mode:
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
        import asyncio

        try:
            # Ensure backends are registered by importing the module
            import hafs.backends  # noqa: F401
            from hafs.agents.coordinator import AgentCoordinator
            from hafs.models.agent import AgentRole

            # Initialize coordinator with timeout
            try:
                self._coordinator = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: AgentCoordinator(self.config)
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                self.notify("Agent coordinator initialization timed out", severity="error")
                return

            # Register initial agents (or defaults)
            agents_to_init = self._initial_agents
            if not agents_to_init:
                agents_to_init = [
                    {"name": "Planner", "role": "planner"},
                    {"name": "Coder", "role": "coder"},
                    {"name": "Critic", "role": "critic"},
                ]

            successful_agents = 0
            failed_agents = []

            for i, agent_spec in enumerate(agents_to_init, 1):
                try:
                    role = AgentRole(agent_spec.get("role", "general"))
                    if self._coordinator:
                        # Show progress
                        self.notify(
                            f"Registering agent {i}/{len(agents_to_init)}: {agent_spec['name']}",
                            timeout=2,
                        )
                        await self._coordinator.register_agent(
                            name=agent_spec["name"],
                            role=role,
                            backend_name=self._default_backend,
                        )
                        successful_agents += 1
                except Exception as e:
                    failed_agents.append(agent_spec["name"])
                    self.notify(
                        f"Failed to register agent {agent_spec['name']}: {e}",
                        severity="warning",
                    )

            # Update screen with ready coordinator
            if self._coordinator:
                await screen.set_coordinator(self._coordinator)

                # Show summary
                if failed_agents:
                    self.notify(
                        f"Initialized {successful_agents} agents." \
                        f"Failed: {', '.join(failed_agents)}",
                        severity="warning",
                        timeout=5,
                    )
                else:
                    self.notify(f"All {successful_agents} agents ready", timeout=3)

        except ImportError as e:
            self.notify(f"Failed to load agent modules: {e}", severity="error")
        except Exception as e:
            self.notify(f"Chat initialization failed: {e}", severity="error")

    async def action_quit(self) -> None:
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

    def action_switch_chat(self) -> None:
        """Switch to multi-agent chat screen."""
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


def run_chat(
    default_backend: str = "gemini",
    agents: list[dict[str, str]] | None = None,
) -> None:
    """Entry point for running the multi-agent chat TUI.

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


# Backwards compatibility alias
run_orchestrator = run_chat


if __name__ == "__main__":
    run()
