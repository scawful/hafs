"""Filesystem tree widget for browsing workspace directories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from hafs.config.schema import WorkspaceDirectory


class FilesystemTree(Tree):
    """Tree widget for browsing filesystem directories.

    Displays configured workspace directories and allows browsing/selecting
    files to add to context.

    Example:
        tree = FilesystemTree(workspace_dirs=[
            WorkspaceDirectory(path=Path("~/Code"), name="Code"),
            WorkspaceDirectory(path=Path("~/Documents"), name="Docs"),
        ])
    """

    DEFAULT_CSS = """
    FilesystemTree {
        height: auto;
        min-height: 4;
        max-height: 100%;
        scrollbar-size: 1 1;
    }

    FilesystemTree > .tree--guides {
        color: $primary-darken-2;
    }

    FilesystemTree > .tree--cursor {
        background: $primary;
    }

    FilesystemTree .directory {
        color: $primary-lighten-1;
    }

    FilesystemTree .file {
        color: $text;
    }

    FilesystemTree .workspace-root {
        color: $secondary;
        text-style: bold;
    }
    """

    class FileSelected(Message):
        """Emitted when a file is selected."""

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    class FileAddToContext(Message):
        """Emitted when user wants to add file to context."""

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    class DirectoryAddToContext(Message):
        """Emitted when user wants to add directory to context."""

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    def __init__(
        self,
        workspace_dirs: list["WorkspaceDirectory"] | None = None,
        show_hidden: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize filesystem tree.

        Args:
            workspace_dirs: List of workspace directories to display.
            show_hidden: Whether to show hidden files/directories.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__("Workspace [dim](w to add)[/]", id=id, classes=classes)
        self._workspace_dirs = workspace_dirs or []
        self._show_hidden = show_hidden
        self._loaded_dirs: set[str] = set()

    def on_mount(self) -> None:
        """Initialize tree on mount."""
        self.root.expand()
        self._populate_workspace_roots()

    def _populate_workspace_roots(self) -> None:
        """Add workspace directory roots to tree."""
        for ws_dir in self._workspace_dirs:
            if ws_dir.path.exists() and ws_dir.path.is_dir():
                display_name = ws_dir.name or ws_dir.path.name
                node = self.root.add(
                    f"[bold]{display_name}[/]",
                    data={"path": ws_dir.path, "type": "workspace", "recursive": ws_dir.recursive},
                    expand=False,
                )
                node.set_classes("workspace-root")
                # Add placeholder for lazy loading
                node.add_leaf("[dim]loading...[/]", data={"type": "placeholder"})

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Handle node expansion - lazy load directory contents."""
        node = event.node
        data = node.data

        if not data or data.get("type") == "placeholder":
            return

        path = data.get("path")
        if not path or not isinstance(path, Path):
            return

        # Check if already loaded
        path_str = str(path)
        if path_str in self._loaded_dirs:
            return

        # Remove placeholder
        for child in list(node.children):
            if child.data and child.data.get("type") == "placeholder":
                child.remove()

        # Load directory contents
        self._load_directory(node, path, data.get("recursive", True))
        self._loaded_dirs.add(path_str)

    def _load_directory(self, node: TreeNode, path: Path, recursive: bool = True) -> None:
        """Load directory contents into node.

        Args:
            node: Parent node to add children to.
            path: Directory path to load.
            recursive: Whether to allow expanding subdirectories.
        """
        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

            for entry in entries:
                # Skip hidden files unless configured to show
                if not self._show_hidden and entry.name.startswith("."):
                    continue

                if entry.is_dir():
                    # Directory node
                    dir_node = node.add(
                        f"[blue]{entry.name}/[/]",
                        data={"path": entry, "type": "directory", "recursive": recursive},
                        expand=False,
                    )
                    dir_node.set_classes("directory")
                    if recursive:
                        # Add placeholder for lazy loading
                        dir_node.add_leaf("[dim]...[/]", data={"type": "placeholder"})
                else:
                    # File node
                    icon = self._get_file_icon(entry)
                    file_node = node.add_leaf(
                        f"{icon} {entry.name}",
                        data={"path": entry, "type": "file"},
                    )
                    file_node.set_classes("file")

        except PermissionError:
            node.add_leaf("[red]Permission denied[/]", data={"type": "error"})
        except Exception as e:
            node.add_leaf(f"[red]Error: {e}[/]", data={"type": "error"})

    def _get_file_icon(self, path: Path) -> str:
        """Get icon for file based on extension."""
        suffix = path.suffix.lower()
        icons = {
            ".py": "[yellow]󰌠[/]",
            ".js": "[yellow]󰌞[/]",
            ".ts": "[blue]󰛦[/]",
            ".jsx": "[cyan]󰜈[/]",
            ".tsx": "[cyan]󰜈[/]",
            ".rs": "[orange]󱘗[/]",
            ".go": "[cyan]󰟓[/]",
            ".md": "[white]󰍔[/]",
            ".txt": "[white]󰈙[/]",
            ".json": "[yellow]󰘦[/]",
            ".toml": "[grey]󰅪[/]",
            ".yaml": "[red]󰅪[/]",
            ".yml": "[red]󰅪[/]",
            ".html": "[orange]󰌝[/]",
            ".css": "[blue]󰌜[/]",
            ".sh": "[green]󰆍[/]",
            ".bash": "[green]󰆍[/]",
            ".zsh": "[green]󰆍[/]",
        }
        return icons.get(suffix, "[dim]󰈔[/]")

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection."""
        node = event.node
        data = node.data

        if not data:
            return

        path = data.get("path")
        node_type = data.get("type")

        if not path or node_type in ("placeholder", "error"):
            return

        if node_type == "file":
            self.post_message(self.FileSelected(path))

    def action_add_to_context(self) -> None:
        """Add currently selected item to context."""
        node = self.cursor_node
        if not node or not node.data:
            return

        path = node.data.get("path")
        node_type = node.data.get("type")

        if not path or node_type in ("placeholder", "error"):
            return

        if node_type == "file":
            self.post_message(self.FileAddToContext(path))
        elif node_type in ("directory", "workspace"):
            self.post_message(self.DirectoryAddToContext(path))

    def set_workspace_dirs(self, workspace_dirs: list["WorkspaceDirectory"]) -> None:
        """Update workspace directories.

        Args:
            workspace_dirs: New list of workspace directories.
        """
        self._workspace_dirs = workspace_dirs
        self._loaded_dirs.clear()
        self.clear()
        self._populate_workspace_roots()

    def refresh_data(self) -> None:
        """Refresh tree data."""
        self._loaded_dirs.clear()
        self.clear()
        self._populate_workspace_roots()

    def set_show_hidden(self, show: bool) -> None:
        """Toggle hidden file visibility.

        Args:
            show: Whether to show hidden files.
        """
        self._show_hidden = show
        self.refresh_data()
