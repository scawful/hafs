"""Compact keybinding bar widget for HAFS TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static


class KeyBindingBar(Widget):
    """Two-row keybinding display with colored chips.

    Displays keybindings in two rows with different color schemes:
    Row 1 (primary actions): cyan keys
    Row 2 (secondary actions): magenta keys

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
    }

    KeyBindingBar .keybinding-row-1 {
        background: $primary-darken-1;
    }

    KeyBindingBar .keybinding-row-2 {
        background: $primary-darken-2;
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
            row1: Explicit first row bindings (cyan).
            row2: Explicit second row bindings (magenta).
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
        with Vertical():
            yield Static(
                self._render_row(self._row1, "cyan"),
                classes="keybinding-row keybinding-row-1",
            )
            yield Static(
                self._render_row(self._row2, "magenta"),
                classes="keybinding-row keybinding-row-2",
            )

    def _render_row(self, bindings: list[tuple[str, str]], color: str) -> str:
        """Render a row of bindings with specified color."""
        parts = []
        for key, label in bindings:
            parts.append(f"[bold {color}][{key}][/] [dim]{label}[/]")
        return "  ".join(parts)

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
                row_list[0].update(self._render_row(self._row1, "cyan"))
            if len(row_list) >= 2:
                row_list[1].update(self._render_row(self._row2, "magenta"))
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
        self._row1 = [(k, l) for k, l in self._row1 if k != key]
        self._row2 = [(k, l) for k, l in self._row2 if k != key]
        self.set_bindings(self._row1, self._row2)


# Preset binding configurations for different screens
MAIN_SCREEN_BINDINGS = [
    ("c", "Chat"),
    ("1-4", "Screens"),
    ("a", "Add"),
    ("d", "Del"),
    ("e", "Edit"),
    ("g", "AI Gen"),
    ("?", "Help"),
    ("q", "Quit"),
]

ORCHESTRATOR_SCREEN_BINDINGS = [
    ("^1-3", "Lanes"),
    ("^n", "New"),
    ("^k", "Kill"),
    ("^c", "Context"),
    ("^p", "Perms"),
    ("Esc", "Back"),
    ("?", "Help"),
]

LOGS_SCREEN_BINDINGS = [
    ("1-3", "Tabs"),
    ("r", "Refresh"),
    ("q", "Back"),
    ("?", "Help"),
]

SETTINGS_SCREEN_BINDINGS = [
    ("r", "Reload"),
    ("q", "Back"),
    ("?", "Help"),
]
