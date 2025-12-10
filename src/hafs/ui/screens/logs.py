"""Log browser screen for HAFS TUI."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static

from hafs.ui.widgets.session_list import SessionList
from hafs.ui.widgets.plan_viewer import PlanViewer


class LogsScreen(Screen):
    """Log browser screen with tabs for different sources."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "back", "Back"),
        ("1", "tab_gemini", "Gemini"),
        ("2", "tab_claude", "Claude"),
        ("3", "tab_antigravity", "Antigravity"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()

        with Container(id="logs-container"):
            with TabbedContent(id="logs-tabs"):
                with TabPane("Gemini Sessions", id="tab-gemini"):
                    yield Static(
                        "[bold]Gemini CLI Sessions[/bold]\n"
                        "[dim]Recent conversations from ~/.gemini/tmp/[/dim]",
                    )
                    yield SessionList(parser_type="gemini", id="gemini-sessions")

                with TabPane("Claude Plans", id="tab-claude"):
                    yield Static(
                        "[bold]Claude Code Plans[/bold]\n"
                        "[dim]Plan files from ~/.claude/plans/[/dim]",
                    )
                    yield PlanViewer(id="claude-plans")

                with TabPane("Antigravity", id="tab-antigravity"):
                    yield Static(
                        "[bold]Antigravity Brains[/bold]\n"
                        "[dim]Brain directories from ~/.gemini/antigravity/brain/[/dim]",
                    )
                    yield SessionList(parser_type="antigravity", id="antigravity-brains")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        self.title = "HAFS - Logs"

    def action_refresh(self) -> None:
        """Refresh all log data."""
        for session_list in self.query(SessionList):
            session_list.refresh_data()

        plan_viewer = self.query_one("#claude-plans", PlanViewer)
        plan_viewer.refresh_data()

    def action_back(self) -> None:
        """Go back to main screen."""
        self.app.pop_screen()

    def action_tab_gemini(self) -> None:
        """Switch to Gemini tab."""
        tabs = self.query_one("#logs-tabs", TabbedContent)
        tabs.active = "tab-gemini"

    def action_tab_claude(self) -> None:
        """Switch to Claude tab."""
        tabs = self.query_one("#logs-tabs", TabbedContent)
        tabs.active = "tab-claude"

    def action_tab_antigravity(self) -> None:
        """Switch to Antigravity tab."""
        tabs = self.query_one("#logs-tabs", TabbedContent)
        tabs.active = "tab-antigravity"
