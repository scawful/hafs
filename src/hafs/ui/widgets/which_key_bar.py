"""Which-key style bottom bar for leader key shortcuts."""

from __future__ import annotations

from typing import Iterable

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class WhichKeyBar(Widget):
    """Bottom bar that displays available leader key commands.

    When inactive, shows a gentle hint about using SPC.
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
        background: $primary-darken-1;
        color: $text-muted;
    }
    """

    active: reactive[bool] = reactive(False)
    prefix: reactive[str] = reactive("")
    hints: reactive[list[tuple[str, str]]] = reactive(list)

    def compose(self) -> ComposeResult:
        yield Static("", id="wk-row-1", classes="wk-row wk-row-1")
        yield Static("", id="wk-row-2", classes="wk-row wk-row-2")

    def show_hints(self, prefix: str, hints: Iterable[tuple[str, str]]) -> None:
        """Activate and render hints."""
        self.active = True
        self.prefix = prefix
        self.hints = list(hints)

    def hide_hints(self) -> None:
        """Hide which-key hints and show idle help."""
        self.active = False
        self.prefix = ""
        self.hints = []

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
            row1.update("[dim]SPC for commands  •  Ctrl+P search  •  Ctrl+K palette[/dim]")
            row2.update("")
            return

        prefix_display = f"SPC {self.prefix}".strip()
        parts = []
        for key, label in self.hints:
            parts.append(f"[bold cyan]{key}[/] {label}")
        hints_text = "   ".join(parts) if parts else "[dim](no bindings)[/dim]"

        row1.update(f"[bold]{prefix_display}[/]  {hints_text}")
        row2.update("[dim]Esc to cancel  •  waiting for key...[/dim]")
