"""Vim-style mode indicator widget.

Displays the current input mode in the UI:
- NORMAL: Standard navigation mode
- INSERT: Text input active
- WHICH-KEY: Leader key sequence in progress
- VISUAL: Selection mode (future)

The indicator uses color coding for quick recognition:
- NORMAL: Primary color
- INSERT: Success/green
- WHICH-KEY: Warning/yellow
"""

from __future__ import annotations

from enum import Enum

from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class InputMode(str, Enum):
    """Available input modes."""

    NORMAL = "NORMAL"
    INSERT = "INSERT"
    WHICH_KEY = "WHICH-KEY"
    VISUAL = "VISUAL"
    COMMAND = "COMMAND"


class ModeIndicator(Static):
    """Widget that displays the current input mode.

    Similar to vim's mode indicator, this shows users which
    mode they're in for better context awareness.
    """

    DEFAULT_CSS = """
    ModeIndicator {
        width: auto;
        height: 1;
        padding: 0 1;
        text-style: bold;
    }

    ModeIndicator.mode-normal {
        background: $primary;
        color: $text;
    }

    ModeIndicator.mode-insert {
        background: $success;
        color: $text;
    }

    ModeIndicator.mode-which-key {
        background: $warning;
        color: $background;
    }

    ModeIndicator.mode-visual {
        background: $accent;
        color: $text;
    }

    ModeIndicator.mode-command {
        background: $secondary;
        color: $text;
    }
    """

    mode: reactive[InputMode] = reactive(InputMode.NORMAL)

    def __init__(
        self,
        mode: InputMode = InputMode.NORMAL,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.mode = mode

    def render(self) -> str:
        """Render the mode indicator."""
        return f" {self.mode.value} "

    def watch_mode(self, old_mode: InputMode, new_mode: InputMode) -> None:
        """React to mode changes."""
        # Remove old mode class
        self.remove_class(f"mode-{old_mode.value.lower().replace('-', '-')}")
        # Add new mode class
        self.add_class(f"mode-{new_mode.value.lower().replace('-', '-')}")
        # Trigger re-render
        self.refresh()

    def on_mount(self) -> None:
        """Set initial mode class on mount."""
        self.add_class(f"mode-{self.mode.value.lower().replace('-', '-')}")

    def set_normal(self) -> None:
        """Set mode to NORMAL."""
        self.mode = InputMode.NORMAL

    def set_insert(self) -> None:
        """Set mode to INSERT."""
        self.mode = InputMode.INSERT

    def set_which_key(self) -> None:
        """Set mode to WHICH-KEY."""
        self.mode = InputMode.WHICH_KEY

    def set_visual(self) -> None:
        """Set mode to VISUAL."""
        self.mode = InputMode.VISUAL

    def set_command(self) -> None:
        """Set mode to COMMAND."""
        self.mode = InputMode.COMMAND
