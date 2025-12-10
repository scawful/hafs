"""Project tree widget for browsing AFS-enabled projects."""

from pathlib import Path
from typing import Any

from textual.message import Message
from textual.widgets import Tree

from hafs.core.afs.discovery import discover_projects
from hafs.models.afs import ContextRoot, MountType


class ProjectSelected(Message):
    """Message sent when a project is selected."""

    def __init__(self, project: ContextRoot) -> None:
        self.project = project
        super().__init__()


class FileSelected(Message):
    """Message sent when a file is selected."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__()


class ProjectTree(Tree[Any]):
    """Tree widget showing AFS-enabled projects."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__("Projects", **kwargs)
        self.guide_depth = 3

    def on_mount(self) -> None:
        """Load projects when mounted."""
        self.refresh_data()

    def refresh_data(self) -> None:
        """Refresh the project list."""
        self.clear()

        # Reset root
        self.root.label = "Projects"
        self.root.data = None

        projects = discover_projects()
        for project in projects:
            # Project node
            node = self.root.add(project.project_name, data=project, expand=False)

            # Add AFS directories
            for mt in MountType:
                mounts = project.mounts.get(mt, [])
                if mounts:
                    # Color based on mount type
                    color = {
                        MountType.MEMORY: "blue",
                        MountType.KNOWLEDGE: "blue",
                        MountType.TOOLS: "red",
                        MountType.SCRATCHPAD: "green",
                        MountType.HISTORY: "dim",
                    }.get(mt, "white")

                    subnode = node.add(f"[{color}]{mt.value}[/{color}]", expand=False)
                    for mount in mounts:
                        self._add_path_node(subnode, mount.name, mount.source)

    def _add_path_node(self, parent: Any, label: str, path: Path) -> None:
        """Add a node for a path."""
        if path.is_dir():
            node = parent.add(label, data=path, expand=False)
            # Add dummy node to make it expandable
            node.add_leaf("loading...", data=None)
        else:
            parent.add_leaf(label, data=path)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Handle node expansion for lazy loading."""
        node = event.node
        path = node.data

        if isinstance(path, Path) and path.is_dir():
            # Check if we need to load (if children are dummy)
            if len(node.children) == 1 and str(node.children[0].label) == "loading...":
                node.remove_children()
                try:
                    items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                    for item in items:
                        # Skip hidden files
                        if item.name.startswith("."):
                            continue
                        self._add_path_node(node, item.name, item)

                    if not node.children:
                        node.add_leaf("[dim](empty)[/dim]", data=None)

                except PermissionError:
                    node.add_leaf("[red](permission denied)[/red]", data=None)
                except Exception as e:
                    node.add_leaf(f"[red]Error: {e}[/red]", data=None)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle selection."""
        data = event.node.data
        if isinstance(data, ContextRoot):
            self.post_message(ProjectSelected(data))
        elif isinstance(data, Path) and data.is_file():
            self.post_message(FileSelected(data))
