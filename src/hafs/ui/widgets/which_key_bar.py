"""Which-key style bottom bar for leader key shortcuts.

Features:
- Idle state: Shows abbreviated key hints (e.g., "SPC g:git f:file ...")
- Active state: Shows full hints for current prefix
- No timeout waiting indicator
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class WhichKeyBar(Widget):
    """Bottom bar that displays available leader key commands.

    When inactive (with SHOW_PERSISTENT_HINTS), shows abbreviated hints.
    When active, shows current prefix and available keys.
    """

    DEFAULT_CSS = """
    WhichKeyBar {
        height: 2;
        width: 100%;
        background: $surface;
    }

    WhichKeyBar .wk-row {
        width: 100%;
        height: 1;
        padding: 0 1;
        content-align: left middle;
    }

    WhichKeyBar .wk-row-1 {
        background: $primary;
        color: $text;
    }

    WhichKeyBar .wk-row-2 {
        background: $primary;
        color: $text-disabled;
    }
    """

    active: reactive[bool] = reactive(False)
    prefix: reactive[str] = reactive("")
    hints: reactive[list[tuple[str, str]]] = reactive(list)
    abbreviated_hints: reactive[list[tuple[str, str]]] = reactive(list)

    def compose(self) -> ComposeResult:
        yield Static("", id="wk-row-1", classes="wk-row wk-row-1")
        yield Static("", id="wk-row-2", classes="wk-row wk-row-2")

    def show_hints(self, prefix: str, hints: Iterable[tuple[str, str]]) -> None:
        """Activate and render full hints (when SPC is pressed)."""
        self.active = True
        self.prefix = prefix
        self.hints = list(hints)

    def show_abbreviated_hints(self, which_key_map: Mapping[str, Any]) -> None:
        """Show abbreviated hints from the root which-key map (idle state)."""
        self.active = False
        self.abbreviated_hints = self._extract_abbreviated(which_key_map)
        self._render_bar()

    def hide_hints(self) -> None:
        """Hide which-key hints and show default help text."""
        self.active = False
        self.prefix = ""
        self.hints = []
        self.abbreviated_hints = []

    def _extract_abbreviated(
        self,
        node: Mapping[str, Any],
        max_items: int = 10,
    ) -> list[tuple[str, str]]:
        """Extract abbreviated hints (first word or + prefix of label).

        Args:
            node: The which-key map node
            max_items: Maximum number of hints to show

        Returns:
            List of (key, abbreviated_label) tuples
        """
        hints = []
        for k, v in list(node.items())[:max_items]:
            if isinstance(v, tuple) and len(v) == 2:
                label = v[0]
                # Keep + prefixes (group indicators) or take first word
                if label.startswith("+"):
                    abbrev = label
                else:
                    abbrev = label.split()[0] if label else k
                hints.append((k, abbrev))
            else:
                hints.append((k, k))
        return hints

    def watch_active(self, _active: bool) -> None:
        self._render_bar()

    def watch_prefix(self, _prefix: str) -> None:
        self._render_bar()

    def watch_hints(self, _hints: list[tuple[str, str]]) -> None:
        self._render_bar()

    def _render_bar(self) -> None:
        try:
            row1 = self.query_one("#wk-row-1", Static)
            row2 = self.query_one("#wk-row-2", Static)
        except Exception:
            return

        if not self.active:
            # Idle state - show abbreviated hints or default text
            if self.abbreviated_hints:
                parts = [f"[bold cyan]{k}[/]:{label}" for k, label in self.abbreviated_hints]
                hint_text = "  ".join(parts)
                row1.update(f"[dim]SPC[/] {hint_text}")
                row2.update("[dim]Press SPC to see all commands  |  Ctrl+P search  |  Ctrl+K palette[/dim]")
            else:
                row1.update("[dim]SPC for commands  |  Ctrl+P search  |  Ctrl+K palette[/dim]")
                row2.update("")
            return

        # Active state - full hints
        prefix_display = f"SPC {self.prefix}".strip()
        parts = []
        for key, label in self.hints:
            parts.append(f"[bold cyan]{key}[/] {label}")
        hints_text = "   ".join(parts) if parts else "[dim](no bindings)[/dim]"

        row1.update(f"[bold]{prefix_display}[/]  {hints_text}")
        row2.update("[dim]Esc to cancel  |  Type a key to continue...[/dim]")
