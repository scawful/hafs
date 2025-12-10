"""Project tree widget for browsing AFS-enabled projects."""

from textual.widgets import Tree
from textual.message import Message

from hafs.core.afs.discovery import discover_projects
from hafs.models.afs import ContextRoot, MountType


class ProjectSelected(Message):
    """Message sent when a project is selected."""

    def __init__(self, project: ContextRoot) -> None:
        self.project = project
        super().__init__()


class ProjectTree(Tree[ContextRoot]):
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

        projects = discover_projects()
        for project in projects:
            node = self.root.add(project.project_name, data=project)

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
                        subnode.add_leaf(mount.name)

        self.root.expand()

    def on_tree_node_selected(self, event: Tree.NodeSelected[ContextRoot]) -> None:
        """Handle project selection."""
        if event.node.data:
            self.post_message(ProjectSelected(event.node.data))
