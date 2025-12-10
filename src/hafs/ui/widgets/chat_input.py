"""Chat input widget with @mention autocomplete support."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Static


class ChatInput(Widget):
    """Chat input with @mention autocomplete support.

    Provides:
    - Text input for messages
    - @mention detection and autocomplete
    - Command prefix detection (/)
    - Submit message handling

    Example:
        chat_input = ChatInput(agent_names=["Planner", "Coder"])

        @on(ChatInput.Submitted)
        def handle_submit(self, event):
            print(f"Message: {event.value}")
            print(f"Mentions: {event.mentions}")
    """

    DEFAULT_CSS = """
    ChatInput {
        height: 5;
        dock: bottom;
        background: $surface;
        padding: 1;
        border-top: solid $primary;
    }

    ChatInput Input {
        width: 100%;
    }

    ChatInput .autocomplete-hint {
        height: 1;
        color: $text-muted;
        padding-left: 1;
    }

    ChatInput .autocomplete-visible {
        background: $primary-darken-1;
    }
    """

    MENTION_PATTERN = re.compile(r"@(\w*)$")

    autocomplete_text: reactive[str] = reactive("")

    class Submitted(Message):
        """Emitted when input is submitted."""

        def __init__(self, value: str, mentions: list[str]):
            self.value = value
            self.mentions = mentions
            super().__init__()

    class MentionSelected(Message):
        """Emitted when an @mention is autocompleted."""

        def __init__(self, agent_name: str):
            self.agent_name = agent_name
            super().__init__()

    def __init__(
        self,
        agent_names: list[str] | None = None,
        placeholder: str = "Message agents... (use @name to mention)",
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize chat input.

        Args:
            agent_names: List of agent names for autocomplete.
            placeholder: Input placeholder text.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._agent_names = agent_names or []
        self._placeholder = placeholder
        self._autocomplete_matches: list[str] = []
        self._autocomplete_index = 0

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical():
            yield Static("", id="autocomplete-hint", classes="autocomplete-hint")
            yield Input(placeholder=self._placeholder, id="chat-input")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for @mention autocomplete."""
        text = event.value

        # Check for partial @mention at end
        match = self.MENTION_PATTERN.search(text)
        if match:
            partial = match.group(1).lower()
            matches = [
                name for name in self._agent_names if name.lower().startswith(partial)
            ]
            if matches:
                self._show_autocomplete(matches)
                return

        self._hide_autocomplete()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()
        if not value:
            return

        # Extract mentions
        mentions = re.findall(r"@(\w+)", value)

        # Clear input
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""

        # Post message
        self.post_message(self.Submitted(value, mentions))

    def on_key(self, event) -> None:
        """Handle key events for autocomplete navigation."""
        if not self._autocomplete_matches:
            return

        if event.key == "tab":
            event.prevent_default()
            self._apply_autocomplete()
        elif event.key == "up":
            event.prevent_default()
            self._autocomplete_index = (self._autocomplete_index - 1) % len(
                self._autocomplete_matches
            )
            self._update_autocomplete_hint()
        elif event.key == "down":
            event.prevent_default()
            self._autocomplete_index = (self._autocomplete_index + 1) % len(
                self._autocomplete_matches
            )
            self._update_autocomplete_hint()
        elif event.key == "escape":
            self._hide_autocomplete()

    def _show_autocomplete(self, matches: list[str]) -> None:
        """Show autocomplete suggestions.

        Args:
            matches: List of matching agent names.
        """
        self._autocomplete_matches = matches
        self._autocomplete_index = 0
        self._update_autocomplete_hint()

    def _hide_autocomplete(self) -> None:
        """Hide autocomplete suggestions."""
        self._autocomplete_matches = []
        self._autocomplete_index = 0
        hint = self.query_one("#autocomplete-hint", Static)
        hint.update("")
        hint.remove_class("autocomplete-visible")

    def _update_autocomplete_hint(self) -> None:
        """Update the autocomplete hint text."""
        hint = self.query_one("#autocomplete-hint", Static)

        if not self._autocomplete_matches:
            hint.update("")
            hint.remove_class("autocomplete-visible")
            return

        hint.add_class("autocomplete-visible")

        # Build hint text with current selection highlighted
        parts = []
        for i, name in enumerate(self._autocomplete_matches[:5]):
            if i == self._autocomplete_index:
                parts.append(f"[bold reverse]@{name}[/]")
            else:
                parts.append(f"@{name}")

        if len(self._autocomplete_matches) > 5:
            parts.append(f"[dim]+{len(self._autocomplete_matches) - 5} more[/]")

        hint.update(" ".join(parts) + " [dim](Tab to select)[/]")

    def _apply_autocomplete(self) -> None:
        """Apply the current autocomplete selection."""
        if not self._autocomplete_matches:
            return

        selected = self._autocomplete_matches[self._autocomplete_index]
        input_widget = self.query_one("#chat-input", Input)
        text = input_widget.value

        # Replace partial @mention with full name
        match = self.MENTION_PATTERN.search(text)
        if match:
            new_text = text[: match.start()] + f"@{selected} "
            input_widget.value = new_text
            input_widget.cursor_position = len(new_text)

        self._hide_autocomplete()
        self.post_message(self.MentionSelected(selected))

    def set_agent_names(self, names: list[str]) -> None:
        """Update the list of agent names for autocomplete.

        Args:
            names: List of agent names.
        """
        self._agent_names = names

    def focus_input(self) -> None:
        """Focus the input field."""
        input_widget = self.query_one("#chat-input", Input)
        input_widget.focus()

    def set_value(self, value: str) -> None:
        """Set the input value.

        Args:
            value: Text to set.
        """
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = value
