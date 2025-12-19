"""Log browser screen for HAFS TUI."""

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from hafs.core.parsers.registry import ParserRegistry
from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.widgets.plan_viewer import PlanViewer
from hafs.ui.widgets.history_search import HistorySearchView
from hafs.ui.widgets.session_list import SessionList, SessionSelected
from hafs.ui.widgets.split_log_view import SplitLogView
from hafs.ui.widgets.which_key_bar import WhichKeyBar
from hafs.ui.screens.context_target_modal import ContextTargetModal

if TYPE_CHECKING:
    from hafs.models.antigravity import AntigravityBrain
    from hafs.models.gemini import GeminiSession


class LogsScreen(Screen, VimNavigationMixin, WhichKeyMixin):
    """Log browser screen with tabs for different sources."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "back", "Back"),
        Binding("1", "tab_gemini", "Gemini"),
        Binding("2", "tab_antigravity", "Antigravity"),
        Binding("3", "tab_claude", "Claude"),
        Binding("4", "tab_history", "History"),
        Binding("d", "delete_selected", "Delete", show=True),
        Binding("s", "save_to_context", "Save", show=True),
        # Vim navigation bindings
        *VimNavigationMixin.VIM_BINDINGS,
    ]

    DEFAULT_CSS = """
    LogsScreen {
        layout: vertical;
    }

    LogsScreen #logs-container {
        height: 1fr;
        width: 100%;
    }

    LogsScreen #logs-tabs {
        height: 100%;
    }

    LogsScreen TabbedContent ContentSwitcher {
        height: 100%;
    }

    LogsScreen TabPane {
        height: 100%;
        padding: 0;
    }

    LogsScreen SplitLogView {
        height: 100%;
    }

    LogsScreen #footer-area {
        height: auto;
        background: $surface;
    }

    LogsScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }

    LogsScreen #which-key-bar {
        width: 2fr;
    }

    LogsScreen #claude-plans-container {
        height: 100%;
    }

    LogsScreen #claude-header {
        height: auto;
        padding: 1;
    }

    LogsScreen #claude-plans {
        height: auto;
        min-height: 20;
    }
    """

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

                with TabPane("Antigravity", id="tab-antigravity"):
                    yield SplitLogView(
                        parser_type="antigravity",
                        id="antigravity-view",
                    )

                if claude_enabled:
                    with TabPane("Claude Plans", id="tab-claude"):
                        with VerticalScroll(id="claude-plans-container"):
                            yield Static(
                                "[bold]Claude Code Plans[/bold] [dim]~/.claude/plans/[/dim]",
                                id="claude-header",
                            )
                            yield PlanViewer(id="claude-plans")

                with TabPane("AFS History", id="tab-history"):
                    yield HistorySearchView(id="history-view")

        # Footer area with outline
        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
                yield WhichKeyBar(id="which-key-bar")
                yield Footer()

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize logs screen."""
        super().__init__(name=name, id=id, classes=classes)
        self._selected_session: "GeminiSession | AntigravityBrain | None" = None
        self._selected_parser_type: str | None = None

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        self.title = "HAFS - Logs"
        # Initialize vim navigation (loads setting from config)
        self.init_vim_navigation()

    def on_session_selected(self, event: SessionSelected) -> None:
        """Track the currently selected session."""
        self._selected_session = event.session
        # Determine parser type from session type
        from hafs.models.antigravity import AntigravityBrain
        from hafs.models.gemini import GeminiSession

        if isinstance(event.session, GeminiSession):
            self._selected_parser_type = "gemini"
        elif isinstance(event.session, AntigravityBrain):
            self._selected_parser_type = "antigravity"

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

    def action_tab_history(self) -> None:
        """Switch to history tab."""
        tabs = self.query_one("#logs-tabs", TabbedContent)
        tabs.active = "tab-history"

    def action_delete_selected(self) -> None:
        """Delete the currently selected session."""
        if not self._selected_session or not self._selected_parser_type:
            self.notify("No session selected", severity="warning")
            return

        # Get the parser
        parser_class = ParserRegistry.get(self._selected_parser_type)
        if not parser_class:
            self.notify(f"Parser not found: {self._selected_parser_type}", severity="error")
            return

        parser = parser_class()

        # Confirm deletion
        from hafs.models.gemini import GeminiSession

        if isinstance(self._selected_session, GeminiSession):
            session_id = self._selected_session.short_id
        else:
            session_id = getattr(self._selected_session, "short_id", "unknown")

        # Perform deletion
        if parser.delete_item(self._selected_session):
            self.notify(f"Deleted session: {session_id}", severity="information")
            self._selected_session = None
            self._selected_parser_type = None
            # Refresh the list
            self.action_refresh()
        else:
            self.notify(f"Failed to delete session: {parser.last_error}", severity="error")

    def action_save_to_context(self) -> None:
        """Save the currently selected session to context directory."""
        if not self._selected_session or not self._selected_parser_type:
            self.notify("No session selected", severity="warning")
            return

        def on_target_selected(target):  # type: ignore[no-untyped-def]
            if not target:
                return

            parser_class = ParserRegistry.get(self._selected_parser_type)  # type: ignore[arg-type]
            if not parser_class:
                self.notify(f"Parser not found: {self._selected_parser_type}", severity="error")
                return

            parser = parser_class()
            context_dir = Path.cwd() / ".context" / target.value

            saved_path = parser.save_to_context(self._selected_session, context_dir)
            if saved_path:
                self.notify(f"Saved to {target.value}: {saved_path.name}", severity="information")
            else:
                self.notify(f"Failed to save: {parser.last_error}", severity="error")

        self.app.push_screen(ContextTargetModal(), on_target_selected)

    def get_which_key_map(self):  # type: ignore[override]
        return {
            "t": (
                "+tabs",
                {
                    "1": ("gemini", "tab_gemini"),
                    "2": ("antigravity", "tab_antigravity"),
                    "3": ("claude", "tab_claude"),
                },
            ),
            "r": ("refresh", "refresh"),
            "s": ("save to context", "save_to_context"),
            "d": ("delete selected", "delete_selected"),
            "q": ("back", "back"),
        }
