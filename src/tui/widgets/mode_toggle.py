"""Mode toggle widget for switching between planning and execution modes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from agents.core.coordinator import CoordinatorMode


class ModeToggle(Widget):
    """Widget for displaying and toggling between coordinator modes.

    Shows the current mode (PLANNING or EXECUTION) with visual indicators.
    Click or press 'm' to toggle between modes.

    Styling:
    - Planning mode: Blue/info color
    - Execution mode: Green/success color

    Example:
        mode_toggle = ModeToggle()
        mode_toggle.set_mode(CoordinatorMode.PLANNING)
    """

    DEFAULT_CSS = """
    ModeToggle {
        width: auto;
        height: 1;
        padding: 0 1;
    }

    ModeToggle Static {
        width: auto;
        height: 1;
    }

    ModeToggle .mode-planning {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    ModeToggle .mode-execution {
        background: $success;
        color: $text;
        text-style: bold;
    }

    ModeToggle:hover {
        background: $primary-lighten-1;
    }
    """

    BINDINGS = [
        Binding("m", "toggle_mode", "Toggle Mode", show=False),
    ]

    current_mode: reactive[str] = reactive("planning")

    class ModeChanged(Message):
        """Posted when the mode is toggled.

        Attributes:
            mode: The new CoordinatorMode value.
        """

        def __init__(self, mode: str) -> None:
            """Initialize the message.

            Args:
                mode: The new mode value ("planning" or "execution").
            """
            super().__init__()
            self.mode = mode

    def __init__(
        self,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize mode toggle widget.

        Args:
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._mode_value = "planning"

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Static(
            "MODE: PLANNING",
            id="mode-display",
            classes="mode-planning",
        )

    def set_mode(self, mode: CoordinatorMode | str) -> None:
        """Set the current mode.

        Args:
            mode: CoordinatorMode enum or string ("planning" or "execution").
        """
        from agents.core.coordinator import CoordinatorMode

        # Convert to string if enum
        if isinstance(mode, CoordinatorMode):
            mode_str = mode.value
        else:
            mode_str = mode.lower()

        self._mode_value = mode_str
        self.current_mode = mode_str
        self._update_display()

    def _update_display(self) -> None:
        """Update the visual display of the mode."""
        try:
            display = self.query_one("#mode-display", Static)

            if self._mode_value == "planning":
                display.update("MODE: PLANNING")
                display.remove_class("mode-execution")
                display.add_class("mode-planning")
            else:
                display.update("MODE: EXECUTION")
                display.remove_class("mode-planning")
                display.add_class("mode-execution")
        except Exception:
            pass

    def action_toggle_mode(self) -> None:
        """Toggle between planning and execution modes."""
        new_mode = "execution" if self._mode_value == "planning" else "planning"
        self._mode_value = new_mode
        self.current_mode = new_mode
        self._update_display()
        self.post_message(self.ModeChanged(new_mode))

    def on_click(self) -> None:
        """Handle click events to toggle mode."""
        self.action_toggle_mode()

    @property
    def mode(self) -> str:
        """Get the current mode as a string.

        Returns:
            Current mode value ("planning" or "execution").
        """
        return self._mode_value
