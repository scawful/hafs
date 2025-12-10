"""Modal screen for managing AFS directory permissions."""

from __future__ import annotations

from typing import Iterable

from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList

from hafs.config.schema import AFSDirectoryConfig, PolicyType


class PermissionsModal(ModalScreen[list[AFSDirectoryConfig]]):
    """Interactive permissions manager for AFS directories."""

    CSS = """
    PermissionsModal {
        align: center middle;
    }

    #dialog {
        width: 60;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }

    #options {
        height: 10;
        margin-top: 1;
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align-horizontal: right;
        column-gap: 1;
    }
    """

    class PermissionsUpdated(Message):
        """Emitted when permissions are changed and saved."""

        def __init__(self, directories: list[AFSDirectoryConfig]):
            super().__init__()
            self.directories = directories

    def __init__(self, directories: Iterable[AFSDirectoryConfig]):
        super().__init__()
        # Work on copies to avoid mutating the app config until saved
        self._directories = [d.model_copy() for d in directories]

    def compose(self):
        with Vertical(id="dialog"):
            yield Label("Manage AFS Permissions", id="title")
            yield Label(
                "Select a directory and press Enter to cycle policy "
                "(read_only → writable → executable).",
                id="subtitle",
            )
            yield OptionList(id="options")
            with Vertical(id="buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel", variant="default")

    def _refresh_options(self) -> None:
        """Populate the option list with current policies."""
        options = self.query_one(OptionList)
        options.clear_options()
        for cfg in self._directories:
            options.add_option(self._format_option(cfg))

    def _format_option(self, cfg: AFSDirectoryConfig) -> str:
        """Format an option label for display."""
        return f"{cfg.name}  [{cfg.policy.value}]"

    def _cycle_policy(self, index: int) -> None:
        """Cycle the policy for a directory."""
        if index < 0 or index >= len(self._directories):
            return

        cfg = self._directories[index]
        order = [PolicyType.READ_ONLY, PolicyType.WRITABLE, PolicyType.EXECUTABLE]
        next_idx = (order.index(cfg.policy) + 1) % len(order)
        self._directories[index] = cfg.model_copy(update={"policy": order[next_idx]})

        self._refresh_options()
        # Restore selection
        options = self.query_one(OptionList)
        options.highlighted = index

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Cycle policy when an option is selected."""
        if event.option_index is not None:
            self._cycle_policy(event.option_index)

    def on_mount(self) -> None:
        """Populate options on mount."""
        self._refresh_options()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save":
            self.post_message(self.PermissionsUpdated(self._directories))
            self.dismiss()
        elif event.button.id == "cancel":
            self.dismiss()
