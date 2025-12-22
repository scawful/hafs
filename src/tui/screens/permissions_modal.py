"""Modal screen for managing AFS directory permissions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Label, OptionList

from config.schema import AFSDirectoryConfig, PolicyType


class PermissionsModal(ModalScreen[list[AFSDirectoryConfig]]):
    """Interactive permissions manager for AFS directories."""

    CSS = """
    PermissionsModal {
        align: center middle;
    }

    #dialog {
        width: 70;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }

    #options {
        height: 10;
        margin-top: 1;
        margin-bottom: 1;
    }

    #checkboxes {
        margin-top: 1;
        margin-bottom: 1;
    }

    #buttons {
        width: 100%;
        height: 3;
        align-horizontal: right;
    }
    """

    class PermissionsUpdated(Message):
        """Emitted when permissions are changed and saved."""

        def __init__(self, directories: list[AFSDirectoryConfig]):
            super().__init__()
            self.directories = directories

    def __init__(
        self,
        directories: Iterable[AFSDirectoryConfig],
        context_path: Optional[Path] = None,
    ):
        super().__init__()
        # Work on copies to avoid mutating the app config until saved
        self._directories = [d.model_copy() for d in directories]
        self._context_path = context_path
        self._save_to_global = True
        self._save_to_project = False

    def compose(self):
        with Vertical(id="dialog"):
            yield Label("Manage AFS Permissions", id="title")
            yield Label(
                "Select a directory and press Enter to cycle policy "
                "(read_only → writable → executable).",
                id="subtitle",
            )
            yield OptionList(id="options")
            with Vertical(id="checkboxes"):
                yield Checkbox(
                    "Save to global config (~/.config/hafs/config.toml)",
                    value=True,
                    id="save-global",
                )
                # Check if we're in a project with .context
                has_project = (
                    self._context_path is not None
                    and self._context_path.exists()
                    and (self._context_path / "metadata.json").exists()
                )
                yield Checkbox(
                    "Save to project (.context/metadata.json)",
                    value=False,
                    id="save-project",
                    disabled=not has_project,
                )
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

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        if event.checkbox.id == "save-global":
            self._save_to_global = event.value
        elif event.checkbox.id == "save-project":
            self._save_to_project = event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save":
            # Save to global config if checked
            if self._save_to_global:
                try:
                    from config.loader import load_config
                    from config.saver import save_config

                    config = load_config()
                    config.afs_directories = self._directories
                    save_config(config)
                    self.notify("Saved to global config")
                except Exception as e:
                    self.notify(f"Failed to save global config: {e}", severity="error")

            # Save to project metadata if checked and available
            if self._save_to_project and self._context_path:
                try:
                    from config.loader import load_config
                    from config.saver import save_afs_policies

                    config = load_config()
                    config.afs_directories = self._directories
                    save_afs_policies(config, self._context_path)
                    self.notify("Saved to project metadata")
                except Exception as e:
                    self.notify(f"Failed to save project metadata: {e}", severity="error")

            # Post message to update in-memory config
            self.post_message(self.PermissionsUpdated(self._directories))
            self.dismiss()
        elif event.button.id == "cancel":
            self.dismiss()
