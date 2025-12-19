"""Command palette modal for quick access to commands and agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, ListItem, ListView, Static

from hafs.core.search import fuzzy_autocomplete


class PaletteActionType(Enum):
    """Type of action selected from palette."""

    SLASH_COMMAND = "slash_command"
    AGENT_MENTION = "agent_mention"
    RECENT_COMMAND = "recent_command"


@dataclass
class PaletteResult:
    """Result from command palette selection."""

    action_type: PaletteActionType
    value: str
    display_text: str


class CommandPalette(ModalScreen[PaletteResult | None]):
    """Quick-access command palette with fuzzy search.

    Features:
    - Fuzzy search through slash commands, agent names, and recent commands
    - Keyboard navigation (j/k or arrows)
    - Enter to select, Escape to close
    - Minimal, clean design

    Example:
        result = await app.push_screen_wait(CommandPalette(
            agent_names=["alice", "bob"],
            recent_commands=["/add agent1", "/task fix bug"]
        ))
        if result:
            print(f"Selected: {result.value} ({result.action_type})")
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("ctrl+c", "dismiss", "Close", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    DEFAULT_CSS = """
    CommandPalette {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }

    CommandPalette #palette-dialog {
        width: 70;
        height: auto;
        max-height: 25;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    CommandPalette #palette-input {
        width: 100%;
        height: 3;
        border: solid $secondary;
        margin-bottom: 1;
        padding: 0 1;
    }

    CommandPalette #palette-results {
        height: auto;
        max-height: 15;
        border: none;
        margin: 0;
        padding: 0;
        background: $surface;
    }

    CommandPalette .palette-item {
        height: auto;
        padding: 0 1;
    }

    CommandPalette .palette-item:hover {
        background: $primary-darken-1;
    }

    CommandPalette .palette-icon {
        color: $secondary;
    }

    CommandPalette .palette-recent {
        color: $text-muted;
    }

    CommandPalette #palette-hint {
        height: auto;
        color: $text-muted;
        text-align: center;
        padding: 1 0;
        border-top: solid $primary-darken-2;
        margin-top: 1;
    }
    """

    # Available slash commands
    SLASH_COMMANDS = [
        ("/add", "Add new agent"),
        ("/remove", "Remove agent"),
        ("/list", "List agents"),
        ("/task", "Set active task"),
        ("/clear", "Clear current lane"),
        ("/broadcast", "Message all agents"),
        ("/open", "Open protocol file"),
        ("/goal", "Set primary goal"),
        ("/defer", "Append deferred item"),
        ("/snapshot", "Snapshot state.md to history"),
        ("/help", "Show help"),
    ]

    def __init__(
        self,
        agent_names: list[str] | None = None,
        recent_commands: list[str] | None = None,
        max_recent: int = 10,
    ) -> None:
        """Initialize command palette.

        Args:
            agent_names: List of registered agent names for @mentions.
            recent_commands: List of recently used commands.
            max_recent: Maximum number of recent commands to track.
        """
        super().__init__()
        self.agent_names = agent_names or []
        self._max_recent = max_recent
        self._recent_commands: deque[str] = deque(
            recent_commands or [], maxlen=max_recent
        )
        self._all_items: list[tuple[str, str, PaletteActionType]] = []
        self._current_matches: list[tuple[str, str, PaletteActionType]] = []

    def compose(self) -> ComposeResult:
        """Compose the palette layout."""
        with Vertical(id="palette-dialog"):
            yield Input(
                placeholder="Search commands, @agents, or recent...",
                id="palette-input",
            )
            yield ListView(id="palette-results")
            yield Static(
                "[dim]Enter to select  \u2022  Esc to cancel  \u2022  "
                "\u2191\u2193 or j/k to navigate[/dim]",                id="palette-hint",
            )

    def on_mount(self) -> None:
        """Initialize palette when mounted."""
        self._build_item_list()
        self._update_results("")
        self.query_one("#palette-input", Input).focus()

    def _build_item_list(self) -> None:
        """Build complete list of searchable items."""
        self._all_items = []

        # Add slash commands
        for cmd, desc in self.SLASH_COMMANDS:
            self._all_items.append((cmd, desc, PaletteActionType.SLASH_COMMAND))

        # Add agent mentions
        for agent in self.agent_names:
            self._all_items.append(
                (f"@{agent}", f"Mention {agent}", PaletteActionType.AGENT_MENTION)
            )

        # Add recent commands (most recent first)
        for recent in reversed(list(self._recent_commands)):
            self._all_items.append(
                (recent, "Recent command", PaletteActionType.RECENT_COMMAND)
            )

    def _update_results(self, query: str) -> None:
        """Update results list based on query.

        Args:
            query: Search query from input.
        """
        results_view = self.query_one("#palette-results", ListView)
        results_view.clear()

        if not query:
            # Show all items when no query
            self._current_matches = self._all_items[:15]
        else:
            # Use fuzzy matching on the command/value part
            searchable_items = [item[0] for item in self._all_items]
            matches = fuzzy_autocomplete(query, searchable_items, limit=15, threshold=30)

            # Map back to full items
            item_map = {item[0]: item for item in self._all_items}
            self._current_matches = [
                item_map[name] for name, _score in matches if name in item_map
            ]

        # Display matches with icons
        for value, desc, action_type in self._current_matches:
            if action_type == PaletteActionType.SLASH_COMMAND:
                icon = "[palette-icon]/[/palette-icon]"
                label = f"{icon} {value}  [dim]{desc}[/dim]"
            elif action_type == PaletteActionType.AGENT_MENTION:
                icon = "[palette-icon]@[/palette-icon]"
                label = f"{icon} {value}  [dim]{desc}[/dim]"
            else:  # RECENT_COMMAND
                icon = "[palette-recent]\u21bb[/palette-recent]"
                label = f"{icon} [palette-recent]{value}[/palette-recent]"

            results_view.append(ListItem(Static(label), classes="palette-item"))

        # Auto-select first item if results exist
        if self._current_matches and results_view.children:
            results_view.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update results as user types.

        Args:
            event: Input change event.
        """
        self._update_results(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter on input.

        Args:
            event: Input submission event.
        """
        results_view = self.query_one("#palette-results", ListView)

        # Select highlighted item if available
        if results_view.highlighted_child and self._current_matches:
            idx = results_view.index
            if 0 <= idx < len(self._current_matches):
                value, desc, action_type = self._current_matches[idx]
                result = PaletteResult(
                    action_type=action_type, value=value, display_text=desc
                )
                self._add_to_recent(value)
                self.dismiss(result)
                return

        # No selection, dismiss without result
        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection from list.

        Args:
            event: ListView selection event.
        """
        idx = self.query_one("#palette-results", ListView).index

        if 0 <= idx < len(self._current_matches):
            value, desc, action_type = self._current_matches[idx]
            result = PaletteResult(
                action_type=action_type, value=value, display_text=desc
            )
            self._add_to_recent(value)
            self.dismiss(result)

    def _add_to_recent(self, command: str) -> None:
        """Add command to recent history.

        Args:
            command: Command to add to history.
        """
        # Remove if already exists to avoid duplicates
        try:
            self._recent_commands.remove(command)
        except ValueError:
            pass

        # Add to front of recent commands
        self._recent_commands.append(command)

    def action_dismiss(self) -> None:
        """Dismiss the palette without selection."""
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move cursor down (vim j key)."""
        results_view = self.query_one("#palette-results", ListView)
        if results_view.children:
            results_view.action_cursor_down()

    def action_cursor_up(self) -> None:
        """Move cursor up (vim k key)."""
        results_view = self.query_one("#palette-results", ListView)
        if results_view.children:
            results_view.action_cursor_up()

    def add_recent_command(self, command: str) -> None:
        """Add a command to recent history.

        This method can be called externally to track command usage.

        Args:
            command: Command to add to recent history.
        """
        self._add_to_recent(command)
        self._build_item_list()

    def update_agent_names(self, names: list[str]) -> None:
        """Update the list of available agent names.

        Args:
            names: New list of agent names.
        """
        self.agent_names = names
        self._build_item_list()
        self._update_results(self.query_one("#palette-input", Input).value)

    def get_recent_commands(self) -> list[str]:
        """Get list of recent commands.

        Returns:
            List of recent commands (most recent last).
        """
        return list(self._recent_commands)
