"""AFS control panel widget for managing agent filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static

if TYPE_CHECKING:
    from hafs.models.afs import ContextRoot


class AFSControlPanel(Widget):
    """Panel for managing AFS mounts and context.

    Provides UI controls for:
    - Viewing mount points by type
    - Adding/removing mounts
    - Initializing AFS for new projects
    - Refreshing AFS state

    Example:
        panel = AFSControlPanel(project=context_root)
    """

    DEFAULT_CSS = """
    AFSControlPanel {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    AFSControlPanel .panel-title {
        text-style: bold;
        color: $secondary;
        margin-bottom: 1;
    }

    AFSControlPanel .mount-section {
        margin-bottom: 1;
    }

    AFSControlPanel .mount-type-header {
        color: $text-muted;
        text-style: bold;
    }

    AFSControlPanel .mount-entry {
        height: 2;
        padding: 0 1;
    }

    AFSControlPanel .mount-name {
        width: 1fr;
    }

    AFSControlPanel .mount-actions {
        width: auto;
    }

    AFSControlPanel .action-row {
        height: 3;
        margin-top: 1;
    }

    AFSControlPanel .action-btn {
        margin-right: 1;
    }

    AFSControlPanel .memory-mount { color: $info; }
    AFSControlPanel .knowledge-mount { color: $info; }
    AFSControlPanel .tools-mount { color: $error; }
    AFSControlPanel .scratchpad-mount { color: $success; }
    AFSControlPanel .history-mount { color: $text-muted; }

    AFSControlPanel .empty-state {
        color: $text-muted;
        padding: 1;
    }
    """

    class MountAdded(Message):
        """Message sent when a mount is added."""

        def __init__(self, mount_type: str, path: Path):
            self.mount_type = mount_type
            self.path = path
            super().__init__()

    class MountRemoved(Message):
        """Message sent when a mount is removed."""

        def __init__(self, mount_type: str, name: str):
            self.mount_type = mount_type
            self.name = name
            super().__init__()

    class AFSInitialized(Message):
        """Message sent when AFS is initialized."""

        def __init__(self, path: Path):
            self.path = path
            super().__init__()

    class AFSRefreshed(Message):
        """Message sent when AFS is refreshed."""

        pass

    def __init__(
        self,
        project: "ContextRoot | None" = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        """Initialize AFS control panel.

        Args:
            project: The ContextRoot to manage.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._project = project

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Static("[bold]AFS Control[/bold]", classes="panel-title")

        if self._project:
            # Show mounts by type
            for mount_type in ["memory", "knowledge", "tools", "scratchpad", "history"]:
                with Vertical(classes="mount-section"):
                    yield Static(
                        f"[{self._get_mount_color(mount_type)}]{mount_type.upper()}[/]",
                        classes="mount-type-header",
                    )
                    mounts = self._get_mounts_for_type(mount_type)
                    if mounts:
                        for mount in mounts:
                            with Horizontal(classes="mount-entry"):
                                yield Static(
                                    f"  {mount.name}",
                                    classes=f"mount-name {mount_type}-mount",
                                )
                    else:
                        yield Static("  [dim]No mounts[/dim]", classes="empty-state")

            # Action buttons
            with Horizontal(classes="action-row"):
                yield Button("+ Add Mount", id="add-mount", classes="action-btn")
                yield Button("Refresh", id="refresh-afs", classes="action-btn")
        else:
            yield Static("[dim]No project selected[/dim]", classes="empty-state")
            with Horizontal(classes="action-row"):
                yield Button("Initialize AFS", id="init-afs", classes="action-btn")

    def _get_mount_color(self, mount_type: str) -> str:
        """Get color markup for mount type."""
        colors = {
            "memory": "blue",
            "knowledge": "blue",
            "tools": "red",
            "scratchpad": "green",
            "history": "dim",
        }
        return colors.get(mount_type, "white")

    def _get_mounts_for_type(self, mount_type: str) -> list:
        """Get mounts for a specific type.

        Args:
            mount_type: The mount type string.

        Returns:
            List of MountPoint objects.
        """
        if not self._project:
            return []

        from hafs.models.afs import MountType

        type_map = {
            "memory": MountType.MEMORY,
            "knowledge": MountType.KNOWLEDGE,
            "tools": MountType.TOOLS,
            "scratchpad": MountType.SCRATCHPAD,
            "history": MountType.HISTORY,
        }

        mt = type_map.get(mount_type)
        if mt and mt in self._project.mounts:
            return self._project.mounts[mt]
        return []

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events.

        Args:
            event: The button press event.
        """
        if event.button.id == "init-afs":
            await self._initialize_afs()
        elif event.button.id == "refresh-afs":
            self._refresh()
        elif event.button.id == "add-mount":
            await self._add_mount_dialog()

    async def _initialize_afs(self) -> None:
        """Initialize AFS for current directory."""
        try:
            from hafs.config.loader import load_config
            from hafs.core.afs.manager import AFSManager

            config = load_config()
            manager = AFSManager(config)
            manager.init()
            self.post_message(self.AFSInitialized(Path.cwd()))

            if hasattr(self, "notify"):
                self.notify("AFS initialized successfully")
        except Exception as e:
            if hasattr(self, "notify"):
                self.notify(f"Failed to initialize AFS: {e}", severity="error")

    def _refresh(self) -> None:
        """Refresh the AFS state."""
        self.post_message(self.AFSRefreshed())
        if hasattr(self, "refresh"):
            self.refresh()

    async def _add_mount_dialog(self) -> None:
        """Show dialog to add a new mount."""
        # For now, just show a notification
        # In a full implementation, this would open a modal dialog
        if hasattr(self, "notify"):
            self.notify(
                "Use CLI: hafs afs mount <type> <source>\n"
                "Types: memory, knowledge, tools, scratchpad, history",
                title="Add Mount",
                timeout=5,
            )

    def set_project(self, project: "ContextRoot") -> None:
        """Update the displayed project.

        Args:
            project: New ContextRoot to display.
        """
        self._project = project
        self.refresh(recompose=True)

    def clear(self) -> None:
        """Clear the control panel."""
        self._project = None
        self.refresh(recompose=True)
