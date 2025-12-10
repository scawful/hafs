"""Log browser screen for HAFS TUI."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Header, Footer, TabbedContent, TabPane, Static

from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.widgets.session_list import SessionList
from hafs.ui.widgets.split_log_view import SplitLogView
from hafs.ui.widgets.plan_viewer import PlanViewer
from hafs.core.parsers.registry import ParserRegistry


class LogsScreen(Screen, VimNavigationMixin):
    """Log browser screen with tabs for different sources."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "back", "Back"),
        Binding("1", "tab_gemini", "Gemini"),
        Binding("3", "tab_antigravity", "Antigravity"),
        # Vim navigation bindings
        *VimNavigationMixin.VIM_BINDINGS,
    ]

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()

        # Check if Claude is enabled/available
        claude_parser = ParserRegistry.get("claude")
        claude_enabled = claude_parser and claude_parser().exists()

        with Container(id="logs-container"):
            with TabbedContent(id="logs-tabs"):
                with TabPane("Gemini Sessions", id="tab-gemini"):
                    yield SplitLogView(
                        parser_type="gemini",
                        id="gemini-view",
                    )

                if claude_enabled:
                    with TabPane("Claude Plans", id="tab-claude"):
                        yield Static(
                            "[bold]Claude Code Plans[/bold]\n"
                            "[dim]Plan files from ~/.claude/plans/[/dim]",
                        )
                        yield PlanViewer(id="claude-plans")

                with TabPane("Antigravity", id="tab-antigravity"):
                    yield SplitLogView(
                        parser_type="antigravity",
                        id="antigravity-view",
                    )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        self.title = "HAFS - Logs"
        # Initialize vim navigation (disabled by default, toggle with Ctrl+V)
        self.init_vim_navigation(enabled=False)

        # Add binding for Claude tab if enabled
        claude_parser = ParserRegistry.get("claude")
        if claude_parser and claude_parser().exists():
            self.app.bind("2", "tab_claude", "Claude")

    def action_refresh(self) -> None:
        """Refresh all log data."""
        # Refresh split log views
        for split_view in self.query(SplitLogView):
            split_view.refresh_data()

        # Refresh any standalone session lists (fallback)
        for session_list in self.query(SessionList):
            session_list.refresh_data()

        # Only refresh plan viewer if it exists
        try:
            plan_viewer = self.query_one("#claude-plans", PlanViewer)
            plan_viewer.refresh_data()
        except Exception:
            pass

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
