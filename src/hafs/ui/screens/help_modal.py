"""Help modal screen for HAFS TUI."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static, TabbedContent, TabPane


class HelpModal(ModalScreen[None]):
    """Modal screen showing help and keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("?", "dismiss", "Close", show=False),
        Binding("q", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }

    #help-dialog {
        width: 70;
        height: 28;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #help-title {
        text-align: center;
        text-style: bold;
        color: $secondary;
        padding-bottom: 1;
    }

    #help-subtitle {
        text-align: center;
        color: $text-muted;
        padding-bottom: 1;
    }

    .help-section-title {
        color: $secondary;
        text-style: bold;
        padding: 1 0;
    }

    #close-hint {
        text-align: center;
        color: $text-muted;
        padding-top: 1;
        dock: bottom;
    }

    DataTable {
        height: auto;
        max-height: 12;
    }
    """

    def __init__(self, current_screen_name: str = "") -> None:
        """Initialize help modal.

        Args:
            current_screen_name: Name of the current screen for context-aware help.
        """
        super().__init__()
        self._current_screen = current_screen_name

    def compose(self) -> ComposeResult:
        """Compose the help modal layout."""
        with Vertical(id="help-dialog"):
            yield Static("HAFS - Halext Agentic File System", id="help-title")
            yield Static("Keyboard Reference", id="help-subtitle")

            with TabbedContent():
                with TabPane("Global", id="tab-global"):
                    yield self._compose_global_keybindings()

                with TabPane("Navigation", id="tab-navigation"):
                    yield self._compose_navigation_keybindings()

                with TabPane("AFS", id="tab-afs"):
                    with VerticalScroll():
                        yield Static(
                            "Agentic File System & CLI",
                            classes="help-section-title",
                        )
                        yield self._compose_afs_commands()

                with TabPane("Workflows", id="tab-workflows"):
                    yield self._compose_workflow_help()

                with TabPane("Vim Mode", id="tab-vim"):
                    with VerticalScroll():
                        yield Static(
                            "Enable with [bold]Ctrl+V[/bold]",
                            classes="help-section-title",
                        )
                        yield self._compose_vim_keybindings()

                if self._current_screen:
                    with TabPane("Screen", id="tab-current"):
                        yield self._compose_screen_keybindings()

            yield Static(
                "Press [bold]Escape[/bold] or [bold]?[/bold] to close",
                id="close-hint",
            )

    def _compose_global_keybindings(self) -> DataTable:
        """Compose global keybindings table."""
        table: DataTable = DataTable(id="global-keys", show_cursor=False)
        table.add_columns("Key", "Action")
        table.add_rows([
            ("1", "Dashboard"),
            ("2", "Logs"),
            ("3", "Settings"),
            ("4", "Orchestrator"),
            ("r", "Refresh"),
            ("q", "Quit"),
            ("?", "Help"),
        ])
        return table

    def _compose_navigation_keybindings(self) -> DataTable:
        """Compose navigation keybindings table."""
        table: DataTable = DataTable(id="nav-keys", show_cursor=False)
        table.add_columns("Key", "Action")
        table.add_rows([
            ("Arrow Keys", "Navigate"),
            ("Enter", "Select / Activate"),
            ("Tab", "Next focusable"),
            ("Shift+Tab", "Previous focusable"),
            ("Ctrl+P", "Search / Focus search"),
        ])
        return table

    def _compose_vim_keybindings(self) -> DataTable:
        """Compose vim mode keybindings table."""
        table: DataTable = DataTable(id="vim-keys", show_cursor=False)
        table.add_columns("Key", "Action")
        table.add_rows([
            ("j / k", "Down / Up"),
            ("h / l", "Left / Right (Collapse / Expand)"),
            ("gg", "Go to first item"),
            ("G", "Go to last item"),
            ("/", "Enter search mode"),
            (":", "Enter command mode"),
            ("n / N", "Next / Previous search match"),
            ("Ctrl+V", "Toggle Vim mode"),
        ])
        return table

    def _compose_afs_commands(self) -> DataTable:
        """Compose AFS mount type commands table."""
        table: DataTable = DataTable(id="afs-keys", show_cursor=False)
        table.add_columns("Command / Type", "Description")
        table.add_rows([
            ("hafs init", "Initialize .context in current dir"),
            ("hafs mount <t> <s>", "Mount source 's' to type 't'"),
            ("hafs list", "List active context and mounts"),
            ("--", "--"),
            ("memory/", "Volatile session data & local docs"),
            ("knowledge/", "Indexed knowledge base & references"),
            ("tools/", "Executable scripts & AFS tools"),
            ("scratchpad/", "Active AI reasoning & plans"),
            ("history/", "Immutable interaction logs"),
        ])
        return table

    def _compose_workflow_help(self) -> DataTable:
        """Compose workflow help table."""
        table: DataTable = DataTable(id="workflow-keys", show_cursor=False)
        table.add_columns("Step", "Action")
        table.add_rows([
            ("1. Initialize", "hafs init (sets up .context)"),
            ("2. Mount", "hafs mount knowledge ./docs"),
            ("3. Orchestrate", "hafs orchestrate 'analyze codebase'"),
            ("4. Review", "Check scratchpad/ for plans"),
        ])
        return table

    def _compose_screen_keybindings(self) -> DataTable:
        """Compose context-specific keybindings for current screen."""
        screen_help = self._get_screen_specific_help()
        table: DataTable = DataTable(id="screen-keys", show_cursor=False)
        table.add_columns("Key", "Action")
        if screen_help:
            table.add_rows(screen_help)
        else:
            table.add_rows([("—", "No additional keybindings")])
        return table

    def _get_screen_specific_help(self) -> list[tuple[str, str]]:
        """Get keybindings specific to the current screen."""
        screen_bindings: dict[str, list[tuple[str, str]]] = {
            "MainScreen": [
                ("a", "Add File / Directory"),
                ("d", "Delete item"),
                ("e", "Edit file"),
                ("Ctrl+P", "Focus search"),
            ],
            "LogsScreen": [
                ("1", "Gemini tab"),
                ("2", "Claude tab"),
                ("3", "Antigravity tab"),
            ],
            "OrchestratorScreen": [
                ("Ctrl+1/2/3", "Focus lane 1/2/3"),
                ("Ctrl+N", "Add new agent"),
                ("Ctrl+K", "Kill agent"),
                ("Ctrl+C", "Toggle context panel"),
                ("Ctrl+P", "Manage permissions"),
                ("Ctrl+S", "Toggle synergy panel"),
                ("Escape", "Back"),
            ],
            "SettingsScreen": [
                ("r", "Reload config"),
            ],
        }
        return screen_bindings.get(self._current_screen, [])

    async def action_dismiss(self, result: Any = None) -> None:
        """Dismiss the modal."""
        await self.dismiss(result)
