"""Modal for selecting a target AFS mount to save context into."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from models.afs import MountType


class ContextTargetModal(ModalScreen[MountType | None]):
    """Pick where to save a log/session as context."""

    BINDINGS = [
        Binding("escape", "dismiss", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    ContextTargetModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }

    ContextTargetModal #dialog {
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    ContextTargetModal #title {
        width: 100%;
        text-align: center;
        color: $secondary;
        margin-bottom: 1;
    }

    ContextTargetModal Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("[bold]Save To Context[/bold]\nChoose target mount:", id="title")
            with Horizontal():
                yield Button("Memory", id="memory", variant="primary")
                yield Button("Knowledge", id="knowledge")
                yield Button("Scratchpad", id="scratchpad")
                yield Button("History", id="history")
                yield Button("Cancel", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "cancel":
            self.dismiss(None)
            return
        try:
            self.dismiss(MountType(button_id))
        except Exception:
            self.dismiss(None)

    def action_dismiss(self) -> None:
        self.dismiss(None)
