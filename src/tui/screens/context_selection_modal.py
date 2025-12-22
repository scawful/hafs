"""Context selection modal for choosing files before switching to chat."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static, Tree
from textual.widgets._tree import TreeNode

from core.afs.discovery import discover_projects
from models.afs import ContextRoot, MountType


class ContextSelectionModal(ModalScreen[list[Path] | None]):
    """Modal for selecting files/directories as context before chat.

    Features:
    - Multi-select file tree from discovered AFS projects
    - Quick select buttons for common selections
    - Selection counter
    - Confirm/Cancel buttons

    Returns:
        List of selected file paths on confirm, None on cancel.
    """

    DEFAULT_CSS = """
    ContextSelectionModal {
        align: center middle;
    }

    ContextSelectionModal #dialog {
        width: 90;
        height: 35;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    ContextSelectionModal #title {
        width: 100%;
        text-align: center;
        padding-bottom: 1;
        color: $primary;
    }

    ContextSelectionModal #tree-container {
        height: 1fr;
        border: solid $secondary;
        padding: 1;
    }

    ContextSelectionModal #quick-buttons {
        height: auto;
        padding: 1 0;
        align: center middle;
    }

    ContextSelectionModal .quick-btn {
        margin: 0 1;
        min-width: 12;
        height: 3;
    }

    ContextSelectionModal #selection-counter {
        height: 1;
        text-align: center;
        color: $accent;
        padding: 1 0;
    }

    ContextSelectionModal #action-buttons {
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    ContextSelectionModal .action-btn {
        margin: 0 1;
        min-width: 14;
        height: 3;
    }

    ContextSelectionModal #context-tree {
        height: 100%;
    }
    """

    def __init__(self) -> None:
        """Initialize context selection modal."""
        super().__init__()
        self._selected_paths: set[Path] = set()
        self._projects: list[ContextRoot] = []
        self._node_to_path: dict[TreeNode, Path] = {}
        self._path_to_node: dict[Path, TreeNode] = {}

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Container(id="dialog"):
            yield Label("[bold]Select Context for Chat[/bold]", id="title")

            # Quick selection buttons
            with Horizontal(id="quick-buttons"):
                yield Button("All", id="btn-all", classes="quick-btn")
                yield Button("Project", id="btn-project", classes="quick-btn")
                yield Button("Memory", id="btn-memory", classes="quick-btn")
                yield Button("Knowledge", id="btn-knowledge", classes="quick-btn")
                yield Button("Clear", id="btn-clear", classes="quick-btn")

            # File tree
            with Vertical(id="tree-container"):
                yield Tree("Projects", id="context-tree")

            # Selection counter
            yield Static("0 items selected", id="selection-counter")

            # Action buttons
            with Horizontal(id="action-buttons"):
                yield Button("Confirm", id="btn-confirm", variant="primary", classes="action-btn")
                yield Button("Cancel", id="btn-cancel", classes="action-btn")

    def on_mount(self) -> None:
        """Load projects when mounted."""
        self._load_projects()
        self._update_counter()

    def _load_projects(self) -> None:
        """Load all AFS projects into the tree."""
        tree = self.query_one("#context-tree", Tree)
        tree.root.expand()
        tree.show_root = True
        tree.guide_depth = 3

        self._projects = discover_projects()

        for project in self._projects:
            # Create project node
            project_node = tree.root.add(
                project.project_name,
                data={"type": "project", "project": project},
                expand=False,
            )
            project_node.allow_expand = True

            # Add mount type categories
            for mt in MountType:
                mounts = project.mounts.get(mt, [])
                if mounts:
                    # Color based on mount type


                    mount_node = project_node.add(
                        f"{mt.value} ({len(mounts)})",
                        data={"type": "mount_category", "mount_type": mt},
                        expand=False,
                    )
                    mount_node.allow_expand = True

                    # Add individual mounts
                    for mount in mounts:
                        self._add_path_node(mount_node, mount.name, mount.source)

    def _add_path_node(self, parent: TreeNode, label: str, path: Path) -> TreeNode:
        """Add a node for a path.

        Args:
            parent: Parent tree node.
            label: Display label for the node.
            path: Path object to associate with the node.

        Returns:
            The created tree node.
        """
        if path.is_dir():
            node = parent.add(
                f"\U0001f4c1 {label}",
                data={"type": "directory", "path": path},
                expand=False,
            )
            node.allow_expand = True
            # Add dummy child for lazy loading
            node.add_leaf("...", data={"type": "placeholder"})
        else:
            node = parent.add_leaf(
                f"\U0001f4c4 {label}",
                data={"type": "file", "path": path},
            )

        # Track mapping
        self._node_to_path[node] = path
        self._path_to_node[path] = node

        return node

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Handle lazy loading of directory contents.

        Args:
            event: The node expansion event.
        """
        node = event.node
        data = node.data

        if not isinstance(data, dict):
            return

        # Check if this is a directory that needs loading
        if data.get("type") == "directory" and len(node.children) == 1:
            # Check if the single child is a placeholder
            first_child = node.children[0]
            if isinstance(first_child.data, dict) and first_child.data.get("type") == "placeholder":
                # Remove placeholder
                node.remove_children()

                # Load directory contents
                path = data["path"]
                try:
                    items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                    for item in items:
                        # Skip hidden files
                        if item.name.startswith("."):
                            continue
                        self._add_path_node(node, item.name, item)

                    if not node.children:
                        node.add_leaf("[dim](empty)[/dim]", data={"type": "empty"})

                except PermissionError:
                    node.add_leaf("[red](permission denied)[/red]", data={"type": "error"})
                except Exception as e:
                    node.add_leaf(f"[red]Error: {e}[/red]", data={"type": "error"})

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle node selection to toggle selection state.

        Args:
            event: The node selection event.
        """
        node = event.node
        data = node.data

        if not isinstance(data, dict):
            return

        # Only allow selection of files and directories, not categories
        if data.get("type") in ("file", "directory"):
            path = data["path"]
            self._toggle_selection(path, node)

    def _toggle_selection(self, path: Path, node: TreeNode) -> None:
        """Toggle the selection state of a path.

        Args:
            path: Path to toggle.
            node: Associated tree node.
        """
        current_label = str(node.label)

        if path in self._selected_paths:
            self._selected_paths.remove(path)
            # Remove selection indicator from label
            if current_label.startswith("✓ "):
                node.set_label(current_label[2:])
        else:
            self._selected_paths.add(path)
            # Add selection indicator to label
            if not current_label.startswith("✓ "):
                node.set_label(f"✓ {current_label}")

        self._update_counter()

    def _update_counter(self) -> None:
        """Update the selection counter display."""
        counter = self.query_one("#selection-counter", Static)
        count = len(self._selected_paths)
        if count == 0:
            counter.update("0 items selected")
        elif count == 1:
            counter.update("1 item selected")
        else:
            counter.update(f"{count} items selected")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses.

        Args:
            event: Button press event.
        """
        button_id = event.button.id

        if button_id == "btn-confirm":
            self.dismiss(list(self._selected_paths) if self._selected_paths else None)
        elif button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-all":
            self._select_all()
        elif button_id == "btn-project":
            self._select_current_project()
        elif button_id == "btn-memory":
            self._select_mount_type(MountType.MEMORY)
        elif button_id == "btn-knowledge":
            self._select_mount_type(MountType.KNOWLEDGE)
        elif button_id == "btn-clear":
            self._clear_selection()

    def _select_all(self) -> None:
        """Select all files and directories from all projects."""
        for project in self._projects:
            for mount_list in project.mounts.values():
                for mount in mount_list:
                    path = mount.source
                    if path in self._path_to_node:
                        node = self._path_to_node[path]
                        if path not in self._selected_paths:
                            self._selected_paths.add(path)
                            current_label = str(node.label)
                            if not current_label.startswith("✓ "):
                                node.set_label(f"✓ {current_label}")

        self._update_counter()

    def _select_current_project(self) -> None:
        """Select all mounts from the currently focused project."""
        tree = self.query_one("#context-tree", Tree)
        current_node = tree.cursor_node

        # Find the project node
        project_node = current_node
        while project_node and project_node != tree.root:
            if isinstance(project_node.data, dict) and project_node.data.get("type") == "project":
                break
            project_node = project_node.parent

        if not project_node or project_node == tree.root:
            self.notify("No project selected", severity="warning")
            return

        # Select all paths under this project
        project = project_node.data["project"]
        for mount_list in project.mounts.values():
            for mount in mount_list:
                path = mount.source
                if path in self._path_to_node:
                    node = self._path_to_node[path]
                    if path not in self._selected_paths:
                        self._selected_paths.add(path)
                        current_label = str(node.label)
                        if not current_label.startswith("✓ "):
                            node.set_label(f"✓ {current_label}")

        self._update_counter()
        self.notify(f"Selected all mounts from {project.project_name}")

    def _select_mount_type(self, mount_type: MountType) -> None:
        """Select all mounts of a specific type from all projects.

        Args:
            mount_type: The mount type to select.
        """
        count = 0
        for project in self._projects:
            mounts = project.mounts.get(mount_type, [])
            for mount in mounts:
                path = mount.source
                if path in self._path_to_node:
                    node = self._path_to_node[path]
                    if path not in self._selected_paths:
                        self._selected_paths.add(path)
                        current_label = str(node.label)
                        if not current_label.startswith("✓ "):
                            node.set_label(f"✓ {current_label}")
                        count += 1

        self._update_counter()
        if count > 0:
            self.notify(f"Selected {count} {mount_type.value} mount(s)")
        else:
            self.notify(f"No {mount_type.value} mounts found", severity="warning")

    def _clear_selection(self) -> None:
        """Clear all selections."""
        for path in list(self._selected_paths):
            if path in self._path_to_node:
                node = self._path_to_node[path]
                current_label = str(node.label)
                if current_label.startswith("✓ "):
                    node.set_label(current_label[2:])

        self._selected_paths.clear()
        self._update_counter()
        self.notify("Selection cleared")

    def on_key(self, event) -> None:
        """Handle keyboard shortcuts.

        Args:
            event: Key event.
        """
        if event.key == "escape":
            self.dismiss(None)
        elif event.key == "enter":
            # Confirm on Enter if we have selections
            if self._selected_paths:
                self.dismiss(list(self._selected_paths))
