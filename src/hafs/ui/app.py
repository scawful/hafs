"""Main HAFS TUI Application."""

from pathlib import Path

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
        Binding("r", "refresh", "Refresh", show=True),
        Binding("?", "help", "Help", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()

    def on_mount(self) -> None:
        """Initialize app on mount."""
        self.push_screen(MainScreen())

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


if __name__ == "__main__":
    run()
