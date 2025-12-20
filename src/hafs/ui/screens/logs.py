"""Log browser screen for HAFS TUI."""

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, TabbedContent, TabPane

from hafs.core.parsers.registry import ParserRegistry
from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.plan_viewer import PlanViewer
from hafs.ui.widgets.history_search import HistorySearchView
from hafs.ui.widgets.session_list import SessionList, SessionSelected
from hafs.ui.widgets.split_log_view import SplitLogView
from hafs.ui.widgets.which_key_bar import WhichKeyBar
from hafs.ui.core.standard_keymaps import get_standard_keymap
from hafs.ui.screens.context_target_modal import ContextTargetModal

if TYPE_CHECKING:
    from hafs.models.antigravity import AntigravityBrain
    from hafs.models.gemini import GeminiSession


class LogsScreen(WhichKeyMixin, VimNavigationMixin, Screen):
    """Log browser screen with tabs for different sources."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=False),
        Binding("q", "back", "Back", show=False),
        Binding("tab", "next_tab", "Next Tab", show=False),
        Binding("shift+tab", "prev_tab", "Prev Tab", show=False),
        Binding("d", "delete_selected", "Delete", show=False),
        Binding("s", "save_to_context", "Save", show=False),
        Binding("ctrl+p", "command_palette", "Commands", show=False),
        Binding("ctrl+k", "command_palette", "Commands", show=False),
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
        border-top: solid $primary-darken-2;
    }

    LogsScreen #which-key-bar {
        width: 100%;
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
        yield HeaderBar(id="header-bar", active_screen="logs")

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

        # Footer area with which-key bar only
        with Container(id="footer-area"):
            yield WhichKeyBar(id="which-key-bar")

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
        # Initialize which-key hints
        self.init_which_key_hints()
        # Set breadcrumb path
        try:
            header = self.query_one(HeaderBar)
            header.set_path("/logs")
        except Exception:
            pass

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

    def action_command_palette(self) -> None:
        """Open command palette."""
        from hafs.ui.screens.command_palette import CommandPalette
        self.app.push_screen(CommandPalette())

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

    def action_next_tab(self) -> None:
        """Switch to next tab."""
        tabs = self.query_one("#logs-tabs", TabbedContent)
        tab_ids = ["tab-gemini", "tab-antigravity", "tab-claude", "tab-history"]
        # Filter to only existing tabs
        existing = [t for t in tab_ids if self._tab_exists(t)]
        if not existing:
            return
        try:
            current_idx = existing.index(tabs.active)
            next_idx = (current_idx + 1) % len(existing)
            tabs.active = existing[next_idx]
        except ValueError:
            tabs.active = existing[0]

    def action_prev_tab(self) -> None:
        """Switch to previous tab."""
        tabs = self.query_one("#logs-tabs", TabbedContent)
        tab_ids = ["tab-gemini", "tab-antigravity", "tab-claude", "tab-history"]
        # Filter to only existing tabs
        existing = [t for t in tab_ids if self._tab_exists(t)]
        if not existing:
            return
        try:
            current_idx = existing.index(tabs.active)
            prev_idx = (current_idx - 1) % len(existing)
            tabs.active = existing[prev_idx]
        except ValueError:
            tabs.active = existing[0]

    def _tab_exists(self, tab_id: str) -> bool:
        """Check if a tab exists."""
        try:
            self.query_one(f"#{tab_id}")
            return True
        except Exception:
            return False

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
        """Return which-key bindings with standard navigation."""
        keymap = get_standard_keymap(self)
        # Add logs-specific bindings
        keymap["t"] = (
            "+tabs",
            {
                "g": ("gemini", self.action_tab_gemini),
                "a": ("antigravity", self.action_tab_antigravity),
                "c": ("claude", self.action_tab_claude),
                "h": ("history", self.action_tab_history),
                "n": ("next", self.action_next_tab),
                "p": ("prev", self.action_prev_tab),
            },
        )
        keymap["r"] = ("refresh", self.action_refresh)
        keymap["s"] = ("save to context", self.action_save_to_context)
        keymap["d"] = ("delete", self.action_delete_selected)
        return keymap

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
        from hafs.ui.core.screen_router import get_screen_router

        route_map = {
            "dashboard": "/dashboard",
            "chat": "/chat",
            "logs": "/logs",
            "services": "/services",
            "analysis": "/analysis",
            "config": "/config",
        }
        route = route_map.get(event.screen)
        if route:
            router = get_screen_router()
            await router.navigate(route)
