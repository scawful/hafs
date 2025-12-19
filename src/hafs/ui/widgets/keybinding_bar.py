"""Compact keybinding bar widget for HAFS TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class KeyBindingBar(Widget):
    """Two-row keybinding display with bold, visible chips.

    Displays keybindings in two rows with high contrast styling:
    Row 1 (primary actions): bright background chips
    Row 2 (secondary actions): muted background chips

    Example:
        bar = KeyBindingBar(
            row1=[("c", "Chat"), ("1-4", "Screens")],
            row2=[("?", "Help"), ("q", "Quit")],
        )
    """

    DEFAULT_CSS = """
    KeyBindingBar {
        height: 2;
        width: 100%;
        background: $surface;
    }

    KeyBindingBar .keybinding-row {
        width: 100%;
        height: 1;
        padding: 0 1;
        content-align: center middle;
    }

    KeyBindingBar .keybinding-row-1 {
        background: $primary;
    }

    KeyBindingBar .keybinding-row-2 {
        background: $primary-darken-1;
    }
    """

    def __init__(
        self,
        bindings: list[tuple[str, str]] | None = None,
        row1: list[tuple[str, str]] | None = None,
        row2: list[tuple[str, str]] | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize keybinding bar.

        Args:
            bindings: Legacy single list - will be split into two rows.
            row1: Explicit first row bindings.
            row2: Explicit second row bindings.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)

        if row1 is not None or row2 is not None:
            self._row1 = row1 or []
            self._row2 = row2 or []
        elif bindings:
            # Split legacy bindings into two rows
            mid = (len(bindings) + 1) // 2
            self._row1 = bindings[:mid]
            self._row2 = bindings[mid:]
        else:
            self._row1 = []
            self._row2 = []

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Static(
            self._render_row(self._row1, "black", "cyan"),
            classes="keybinding-row keybinding-row-1",
        )
        yield Static(
            self._render_row(self._row2, "black", "yellow"),
            classes="keybinding-row keybinding-row-2",
        )

    def _render_row(self, bindings: list[tuple[str, str]], key_bg: str, key_fg: str) -> str:
        """Render a row of bindings with pill-style chips."""
        parts = []
        for key, label in bindings:
            # Bold key in colored box, white label
            parts.append(f"[bold {key_fg}]{key}[/]:[white]{label}[/]")
        return "   ".join(parts)

    def set_bindings(
        self,
        row1: list[tuple[str, str]] | None = None,
        row2: list[tuple[str, str]] | None = None,
    ) -> None:
        """Update the displayed bindings.

        Args:
            row1: First row bindings.
            row2: Second row bindings.
        """
        if row1 is not None:
            self._row1 = row1
        if row2 is not None:
            self._row2 = row2

        try:
            rows = self.query(".keybinding-row").results(Static)
            row_list = list(rows)
            if len(row_list) >= 1:
                row_list[0].update(self._render_row(self._row1, "black", "cyan"))
            if len(row_list) >= 2:
                row_list[1].update(self._render_row(self._row2, "black", "yellow"))
        except Exception:
            pass

    def add_binding(self, key: str, label: str, row: int = 1) -> None:
        """Add a single binding.

        Args:
            key: The key or key combo.
            label: The action label.
            row: Which row (1 or 2) to add to.
        """
        if row == 1:
            self._row1.append((key, label))
        else:
            self._row2.append((key, label))
        self.set_bindings(self._row1, self._row2)

    def remove_binding(self, key: str) -> None:
        """Remove a binding by key from both rows.

        Args:
            key: The key to remove.
        """
        self._row1 = [(k, label) for k, label in self._row1 if k != key]
        self._row2 = [(k, label) for k, label in self._row2 if k != key]
        self.set_bindings(self._row1, self._row2)


# Preset binding configurations for different screens
# Row 1: Primary actions (cyan)
# Row 2: Secondary actions (yellow)

MAIN_SCREEN_BINDINGS_ROW1 = [
    ("c", "Chat"),
    ("e", "Edit"),
    ("^s", "Save"),
    ("f2", "Rename"),
    ("f5", "Copy"),
]

MAIN_SCREEN_BINDINGS_ROW2 = [
    ("a", "Add"),
    ("x", "Context"),
    ("m", "MD View"),
    ("w", "Workspace"),
    ("g", "AI Gen"),
    ("p", "Policies"),
    ("r", "Refresh"),
    ("^p", "Search"),
    ("q", "Quit"),
]

# Legacy format for backwards compatibility
MAIN_SCREEN_BINDINGS = MAIN_SCREEN_BINDINGS_ROW1 + MAIN_SCREEN_BINDINGS_ROW2

ORCHESTRATOR_SCREEN_BINDINGS_ROW1 = [
    ("1-4", "Lane"),
    ("Tab", "Next"),
    ("m", "View"),
    ("^n", "Agent"),
]

ORCHESTRATOR_SCREEN_BINDINGS_ROW2 = [
    ("^y", "YOLO"),
    ("S-Tab", "Accept"),
    ("^x", "Context"),
    ("Esc", "Back"),
]

ORCHESTRATOR_SCREEN_BINDINGS = ORCHESTRATOR_SCREEN_BINDINGS_ROW1 + ORCHESTRATOR_SCREEN_BINDINGS_ROW2

LOGS_SCREEN_BINDINGS_ROW1 = [
    ("1", "Gemini"),
    ("2", "Antigravity"),
    ("3", "Claude"),
    ("d", "Delete"),
    ("s", "Save"),
]

LOGS_SCREEN_BINDINGS_ROW2 = [
    ("r", "Refresh"),
    ("q", "Back"),
    ("?", "Help"),
]

LOGS_SCREEN_BINDINGS = LOGS_SCREEN_BINDINGS_ROW1 + LOGS_SCREEN_BINDINGS_ROW2

SETTINGS_SCREEN_BINDINGS_ROW1 = [
    ("p", "Policies"),
    ("r", "Reload"),
]

SETTINGS_SCREEN_BINDINGS_ROW2 = [
    ("q", "Back"),
    ("?", "Help"),
]

SETTINGS_SCREEN_BINDINGS = SETTINGS_SCREEN_BINDINGS_ROW1 + SETTINGS_SCREEN_BINDINGS_ROW2
