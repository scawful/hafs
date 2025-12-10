from textual.app import ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class InputModal(ModalScreen[str]):
    """Modal for getting text input."""

    DEFAULT_CSS = """
    InputModal {
        align: center middle;
    }

    #dialog {
        grid-size: 1;
        grid-rows: auto auto;
        grid-gutter: 1;
        padding: 1 2;
        width: 60;
        height: auto;
        border: thick $primary;
        background: $surface;
    }

    #question {
        width: 100%;
        content-align: center middle;
    }

    #input {
        width: 100%;
    }
    """

    def __init__(self, prompt: str, initial: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.initial_value = initial

    def compose(self) -> ComposeResult:
        with Grid(id="dialog"):
            yield Label(self.prompt, id="question")
            yield Input(self.initial_value, id="input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)
