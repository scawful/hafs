"""Protocol helper widget for interacting with `.context/` artifacts."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, Static


class ProtocolWidget(VerticalScroll):
    """A small UI for interacting with cognitive protocol files."""

    DEFAULT_CSS = """
    ProtocolWidget {
        padding: 1 2;
        height: 100%;
    }
    ProtocolWidget .proto-row {
        height: auto;
        margin-bottom: 1;
    }
    ProtocolWidget Input {
        width: 1fr;
    }
    ProtocolWidget Button {
        margin-left: 1;
    }
    """

    class OpenFileRequested(Message):
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._root: Path = Path.cwd()

    def compose(self) -> ComposeResult:
        yield Label("[bold]Protocol[/bold]")
        yield Static(
            "Quick helpers for `.context/` (works with any agent).",
            classes="dim",
        )
        yield Static(f"[dim]Target:[/dim] {self._root}", id="proto-target")

        with Horizontal(classes="proto-row"):
            yield Input(placeholder="Set primary goal…", id="proto-goal")
            yield Button("Set", id="proto-goal-set", variant="primary")

        with Horizontal(classes="proto-row"):
            yield Input(placeholder="Add deferred item…", id="proto-defer")
            yield Button("Add", id="proto-defer-add", variant="primary")

        with Horizontal(classes="proto-row"):
            yield Input(placeholder="Snapshot reason (optional)…", id="proto-snap-reason")
            yield Button("Snapshot state.md", id="proto-snapshot")

        yield Static("[bold]Open[/bold] (in Context tab):", classes="proto-row")
        with Horizontal(classes="proto-row"):
            yield Button("state.md", id="proto-open-state")
            yield Button("goals.json", id="proto-open-goals")
            yield Button("deferred.md", id="proto-open-deferred")
            yield Button("metacognition.json", id="proto-open-meta")
            yield Button("fears.json", id="proto-open-fears")

    def set_target_root(self, root: Path) -> None:
        self._root = root.resolve()
        try:
            self.query_one("#proto-target", Static).update(f"[dim]Target:[/dim] {self._root}")
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "proto-goal-set":
            goal = self.query_one("#proto-goal", Input).value.strip()
            if not goal:
                self.app.notify("Enter a goal", severity="warning")
                return
            from core.protocol.actions import set_primary_goal

            set_primary_goal(self._root, goal)
            self.app.notify("Primary goal updated", timeout=2)
            return

        if bid == "proto-defer-add":
            item = self.query_one("#proto-defer", Input).value.strip()
            if not item:
                self.app.notify("Enter a deferred item", severity="warning")
                return
            from core.protocol.actions import append_deferred

            append_deferred(self._root, item)
            self.app.notify("Deferred item added", timeout=2)
            return

        if bid == "proto-snapshot":
            reason = self.query_one("#proto-snap-reason", Input).value.strip() or None
            from core.protocol.actions import snapshot_state

            dest = snapshot_state(self._root, reason=reason)
            if dest:
                self.app.notify(f"Snapshot saved: {dest.name}", timeout=2)
            else:
                self.app.notify("Snapshot failed", severity="error")
            return

        mapping = {
            "proto-open-state": "state",
            "proto-open-goals": "goals",
            "proto-open-deferred": "deferred",
            "proto-open-meta": "metacognition",
            "proto-open-fears": "fears",
        }
        if bid in mapping:
            from core.protocol.actions import open_protocol_file

            path = open_protocol_file(self._root, mapping[bid])  # type: ignore[arg-type]
            self.post_message(self.OpenFileRequested(path))

