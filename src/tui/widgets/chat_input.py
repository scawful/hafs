"""Chat input widget with @mention autocomplete and slash command support."""

from __future__ import annotations

import re

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Static

from core.search import fuzzy_autocomplete


class ChatInput(Widget):
    """Chat input with @mention autocomplete, slash commands, and message history.

    Provides:
    - Text input for messages
    - @mention detection and autocomplete (fuzzy matching)
    - Slash command autocomplete with argument hints
    - Message history navigation (up/down arrows)
    - Submit message handling

    Slash Commands:
    - /add <name> <role> - Add an agent
    - /remove <name> - Remove an agent
    - /task <description> - Create a task
    - /broadcast <message> - Broadcast to all agents
    - /list - List all agents
    - /clear - Clear chat history
    - /help - Show help
    - /mode <mode> - Change mode

    Example:
        chat_input = ChatInput(agent_names=["Planner", "Coder"])

        @on(ChatInput.Submitted)
        def handle_submit(self, event):
            print(f"Message: {event.value}")
            print(f"Mentions: {event.mentions}")
    """

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 3;
        max-height: 10;
        background: $surface;
        padding: 0 1;
        border-top: solid $primary;
    }

    ChatInput Input {
        width: 100%;
        margin: 0;
    }

    ChatInput .autocomplete-hint {
        height: auto;
        max-height: 6;
        color: $text-disabled;
        padding-left: 1;
        overflow-y: auto;
    }

    ChatInput .autocomplete-visible {
        background: $primary;
    }
    """

    # Regex patterns for autocomplete detection
    MENTION_PATTERN = re.compile(r"@(\w*)$")
    COMMAND_PATTERN = re.compile(r"^/(\w*)$")

    # Slash command definitions with their argument hints
    COMMANDS = {
        "add": "/add <name> <role> - Add an agent",
        "remove": "/remove <name> - Remove an agent",
        "task": "/task <description> - Create a task",
        "broadcast": "/broadcast <message> - Broadcast to all agents",
        "list": "/list - List all agents",
        "clear": "/clear - Clear chat history",
        "help": "/help - Show help",
        "mode": "/mode <mode> - Change mode",
        "ui": "/ui <headless|terminal> - Switch UI mode",
    }

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
        placeholder: str = "Message agents... (use @name to mention, /command for commands)",
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
        self._autocomplete_type: str = ""  # "mention" or "command"

        # Message history for up/down arrow navigation
        self._message_history: list[str] = []
        self._history_index: int = -1  # -1 means not browsing history
        self._current_draft: str = ""  # Save current input when browsing history

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        with Vertical():
            yield Static("", id="autocomplete-hint", classes="autocomplete-hint")
            yield Input(placeholder=self._placeholder, id="chat-input")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes for autocomplete (mentions and commands).

        Detects and handles:
        - @mention autocomplete (fuzzy matching) at end of text
        - /command autocomplete (prefix matching) at start of text
        """
        text = event.value

        # Reset history navigation when user types
        self._history_index = -1

        # Check for slash command at start (takes priority)
        command_match = self.COMMAND_PATTERN.match(text)
        if command_match:
            partial = command_match.group(1)
            # Filter commands by prefix match
            matches = [
                cmd for cmd in self.COMMANDS.keys()
                if cmd.startswith(partial.lower())
            ]
            if matches:
                self._show_autocomplete(matches, "command")
                return

        # Check for partial @mention at end
        mention_match = self.MENTION_PATTERN.search(text)
        if mention_match:
            partial = mention_match.group(1)
            if partial:
                # Use fuzzy matching for agent names
                results = fuzzy_autocomplete(partial, self._agent_names, limit=5, threshold=30)
                matches = [name for name, _score in results]
            else:
                # Show all agents if just @ is typed
                matches = self._agent_names[:5]

            if matches:
                self._show_autocomplete(matches, "mention")
                return

        self._hide_autocomplete()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission and add to message history."""
        value = event.value.strip()
        if not value:
            return

        # Add to message history (avoid duplicates of last message)
        if not self._message_history or self._message_history[-1] != value:
            self._message_history.append(value)

        # Reset history navigation
        self._history_index = -1
        self._current_draft = ""

        # Extract mentions
        mentions = re.findall(r"@(\w+)", value)

        # Clear input
        input_widget = self.query_one("#chat-input", Input)
        input_widget.value = ""

        # Post message
        self.post_message(self.Submitted(value, mentions))

    def on_key(self, event) -> None:
        """Handle key events for autocomplete and history navigation.

        Priority order:
        1. Autocomplete navigation (when autocomplete is active)
        2. Message history navigation (when autocomplete is not active)
        """
        # Handle autocomplete navigation first (takes priority)
        if self._autocomplete_matches:
            if event.key == "tab":
                event.prevent_default()
                self._apply_autocomplete()
                return
            elif event.key == "up":
                event.prevent_default()
                self._autocomplete_index = (self._autocomplete_index - 1) % len(
                    self._autocomplete_matches
                )
                self._update_autocomplete_hint()
                return
            elif event.key == "down":
                event.prevent_default()
                self._autocomplete_index = (self._autocomplete_index + 1) % len(
                    self._autocomplete_matches
                )
                self._update_autocomplete_hint()
                return
            elif event.key == "escape":
                self._hide_autocomplete()
                return

        # Handle message history navigation (when autocomplete is not active)
        if event.key in ("up", "down") and self._message_history:
            event.prevent_default()
            input_widget = self.query_one("#chat-input", Input)

            # Save current draft when first entering history mode
            if self._history_index == -1:
                self._current_draft = input_widget.value

            if event.key == "up":
                # Navigate backward in history (older messages)
                if self._history_index < len(self._message_history) - 1:
                    self._history_index += 1
            elif event.key == "down":
                # Navigate forward in history (newer messages)
                if self._history_index > -1:
                    self._history_index -= 1

            # Update input with history or draft
            if self._history_index == -1:
                # Back to current draft
                input_widget.value = self._current_draft
            else:
                # Show message from history (newest at end, so reverse index)
                history_pos = len(self._message_history) - 1 - self._history_index
                input_widget.value = self._message_history[history_pos]

            # Move cursor to end
            input_widget.cursor_position = len(input_widget.value)

    def _show_autocomplete(self, matches: list[str], autocomplete_type: str) -> None:
        """Show autocomplete suggestions.

        Args:
            matches: List of matching items (agent names or commands).
            autocomplete_type: Type of autocomplete ("mention" or "command").
        """
        self._autocomplete_matches = matches
        self._autocomplete_index = 0
        self._autocomplete_type = autocomplete_type
        self._update_autocomplete_hint()

    def _hide_autocomplete(self) -> None:
        """Hide autocomplete suggestions."""
        self._autocomplete_matches = []
        self._autocomplete_index = 0
        self._autocomplete_type = ""
        hint = self.query_one("#autocomplete-hint", Static)
        hint.update("")
        hint.remove_class("autocomplete-visible")

    def _update_autocomplete_hint(self) -> None:
        """Update the autocomplete hint text based on type (mention or command)."""
        hint = self.query_one("#autocomplete-hint", Static)

        if not self._autocomplete_matches:
            hint.update("")
            hint.remove_class("autocomplete-visible")
            return

        hint.add_class("autocomplete-visible")

        # Build hint text based on autocomplete type
        parts = []
        if self._autocomplete_type == "command":
            # Show command with argument hints
            for i, cmd in enumerate(self._autocomplete_matches[:5]):
                cmd_hint = self.COMMANDS.get(cmd, f"/{cmd}")
                if i == self._autocomplete_index:
                    parts.append(f"[bold reverse]{cmd_hint}[/]")
                else:
                    parts.append(f"[dim]{cmd_hint}[/]")
        else:
            # Show @mention suggestions
            for i, name in enumerate(self._autocomplete_matches[:5]):
                if i == self._autocomplete_index:
                    parts.append(f"[bold reverse]@{name}[/]")
                else:
                    parts.append(f"@{name}")

        if len(self._autocomplete_matches) > 5:
            parts.append(f"[dim]+{len(self._autocomplete_matches) - 5} more[/]")

        # Use newline separator for commands to show hints properly, space for mentions
        separator = "\n" if self._autocomplete_type == "command" else " "
        hint.update(separator.join(parts) + " [dim](Tab to select, ↑↓ to navigate)[/]")

    def _apply_autocomplete(self) -> None:
        """Apply the current autocomplete selection (mention or command)."""
        if not self._autocomplete_matches:
            return

        selected = self._autocomplete_matches[self._autocomplete_index]
        input_widget = self.query_one("#chat-input", Input)
        text = input_widget.value

        if self._autocomplete_type == "command":
            # Replace partial /command with full command and space
            new_text = f"/{selected} "
            input_widget.value = new_text
            input_widget.cursor_position = len(new_text)
        else:
            # Replace partial @mention with full name
            match = self.MENTION_PATTERN.search(text)
            if match:
                new_text = text[: match.start()] + f"@{selected} "
                input_widget.value = new_text
                input_widget.cursor_position = len(new_text)
                self.post_message(self.MentionSelected(selected))

        self._hide_autocomplete()

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
