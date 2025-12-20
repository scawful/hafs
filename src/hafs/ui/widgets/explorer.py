"""Unified Explorer Widget for browsing files and AFS projects."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import TabbedContent, TabPane, Tree
from textual.widgets.tree import TreeNode

from hafs.config.schema import WorkspaceDirectory
from hafs.core.afs.discovery import discover_projects
from hafs.core.vcs import GitProvider
from hafs.models.afs import ContextRoot, MountType


class BaseTree(Tree):
    """Base tree with Git integration helpers."""

    def __init__(self, label: str, git_provider: GitProvider, **kwargs):
        super().__init__(label, **kwargs)
        self.git = git_provider

    def _get_git_color(self, path: Path) -> str:
        if not self.git.is_available():
            return "white"

        # This is a bit expensive to do per file if not cached or optimized,
        # but for a TUI it might be acceptable for now.
        # Ideally we'd query status for the whole repo once.
        # For now, let's keep it simple or maybe just check if it's in the status list.
        # Optimized approach: We should probably pass the repo status down.
        return "white"

    def _get_file_icon(self, path: Path) -> str:
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


class ProjectTree(BaseTree):
    def on_mount(self) -> None:
        self.show_root = False
        # Show loading state immediately, defer actual discovery
        self.root.add_leaf("[dim]Loading projects...[/]", data={"type": "loading"})
        self.root.expand()
        # Defer heavy project discovery with longer delay
        self.set_timer(0.2, self._deferred_refresh)

    def _deferred_refresh(self) -> None:
        """Load projects after initial render in background."""
        import asyncio
        asyncio.create_task(self._async_refresh())

    async def _async_refresh(self) -> None:
        """Load projects in background thread."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Run discovery in thread to not block UI
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            projects = await loop.run_in_executor(
                executor,
                lambda: discover_projects(max_depth=2)  # Reduce depth for speed
            )

        # Update UI on main thread
        self._populate_projects(projects)

    def refresh_data(self) -> None:
        """Synchronous refresh - use for manual refresh only."""
        self.clear()
        self.root.expand()
        projects = discover_projects(max_depth=2)
        self._populate_projects(projects)

    def _populate_projects(self, projects: list) -> None:
        """Populate tree with discovered projects."""
        self.clear()
        self.root.expand()

        # Always surface Global Context, even if not initialized yet.
        global_path = (Path.home() / ".context").resolve()
        has_global = any(p.path.resolve() == global_path for p in projects)
        if not has_global:
            global_node = self.root.add(
                "Global Context",
                data={"type": "global_context", "path": global_path},
                expand=False,
            )
            global_node.add_leaf("loading...", data={"type": "placeholder"})

        for project in projects:
            if project.path.resolve() == global_path:
                project = ContextRoot(
                    path=project.path,
                    project_name="Global Context",
                    metadata=project.metadata,
                    mounts=project.mounts,
                )
            node = self.root.add(project.project_name, data=project, expand=False)
            node.add_leaf("loading...", data={"type": "placeholder", "context_root": project})

    def _add_path_node(self, parent: TreeNode, label: str, path: Path) -> None:
        if path.is_dir():
            node = parent.add(label, data=path, expand=False)
            node.add_leaf("loading...", data=None)
        else:
            icon = self._get_file_icon(path)
            parent.add_leaf(f"{icon} {label}", data=path)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        data = node.data

        # Handle Global Context placeholder expansion
        if isinstance(data, dict) and data.get("type") == "global_context":
            node.remove_children()
            try:
                from hafs.config.loader import load_config
                from hafs.core.afs.manager import AFSManager
                from hafs.core.afs.discovery import load_context_root

                manager = AFSManager(load_config())
                manager.ensure(Path.home())
                project = load_context_root(Path.home() / ".context")
                if project:
                    project = ContextRoot(
                        path=project.path,
                        project_name="Global Context",
                        metadata=project.metadata,
                        mounts=project.mounts,
                    )
                    node.label = "Global Context"
                    node.data = project
                    # Fall through to the standard ContextRoot expansion below
                    data = project
                else:
                    node.add_leaf("[red]Failed to load global context[/]", data=None)
                    return
            except Exception:
                node.add_leaf(
                    "[red]Failed to initialize global context[/]",
                    data=None,
                )
                return

        # Handle Project (ContextRoot) expansion
        if isinstance(data, ContextRoot):
            # Avoid duplicating mount-type categories if the node is expanded again.
            if node.children:
                first_child_data = node.children[0].data
                if not (
                    isinstance(first_child_data, dict)
                    and first_child_data.get("type") == "placeholder"
                ):
                    return
                node.remove_children()

            project = data
            for mt in MountType:
                mounts = project.mounts.get(mt, [])
                color = {
                    MountType.MEMORY: "blue",
                    MountType.KNOWLEDGE: "blue",
                    MountType.TOOLS: "red",
                    MountType.SCRATCHPAD: "green",
                    MountType.HISTORY: "dim",
                }.get(mt, "white")

                label = f"[{color}]{mt.value.upper()}[/{color}]"
                if not mounts:
                    label += " [dim](0)[/]"

                mount_type_node = node.add(
                    label,
                    expand=False,
                    data={"type": "mount_type_category", "mount_type": mt, "project": project},
                )

                if mounts:
                    # Add placeholder for lazy loading files within this mount type
                    mount_type_node.add_leaf(
                        "loading...",
                        data={"type": "placeholder", "mount_type": mt, "project": project},
                    )
                else:
                    mount_type_node.add_leaf("[dim](empty)[/dim]", data=None)
            return

        # Handle MountType Category expansion (to show actual files)
        if isinstance(data, dict) and data.get("type") == "mount_type_category":
            # Avoid duplicating children if expanded again.
            if node.children:
                first_child_data = node.children[0].data
                if not (
                    isinstance(first_child_data, dict)
                    and first_child_data.get("type") == "placeholder"
                ):
                    return
                node.remove_children()

            mount_type = data["mount_type"]
            project = data["project"]
            mounts = project.mounts.get(mount_type, [])

            if mounts:
                for mount in mounts:
                    status_color = self._get_git_color(mount.source)
                    if mount.source.is_dir():
                        dir_node = node.add(
                            f"[{status_color}]{mount.name}[/{status_color}]",
                            data=mount.source,
                            expand=False,
                        )
                        # Use the filesystem-style placeholder so Path expansion works.
                        dir_node.add_leaf("loading...", data=None)
                    else:
                        icon = self._get_file_icon(mount.source)
                        node.add_leaf(
                            f"[{status_color}]{icon} {mount.name}[/{status_color}]",
                            data=mount.source,
                        )

                if not node.children:
                    node.add_leaf("[dim](empty)[/dim]", data=None)
            else:
                node.add_leaf("[dim](empty)[/dim]", data=None)
            return

        # Original FilesystemTree path handling (still relevant for FileSystemTree)
        if isinstance(data, Path) and data.is_dir():
            if len(node.children) == 1 and str(node.children[0].label) == "loading...":
                node.remove_children()
                try:
                    items = sorted(
                        data.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
                    )
                    for item in items:
                        if item.name.startswith("."):
                            continue
                        self._add_path_node(node, item.name, item)
                    if not node.children:
                        node.add_leaf("[dim](empty)[/dim]", data=None)
                except Exception:
                    node.add_leaf("[red]Error[/]", data=None)


class FilesystemTree(BaseTree):
    def __init__(
        self, workspace_dirs: List[WorkspaceDirectory], git_provider: GitProvider, **kwargs
    ):
        super().__init__("Workspace", git_provider, **kwargs)
        self.workspace_dirs = workspace_dirs
        self._loaded_dirs = set()

    def on_mount(self) -> None:
        self.root.expand()
        for ws_dir in self.workspace_dirs:
            if ws_dir.path.exists():
                name = ws_dir.name or ws_dir.path.name
                node = self.root.add(
                    f"[bold]{name}[/]",
                    data={"path": ws_dir.path, "type": "workspace"},
                    expand=False
                )
                node.add_leaf("loading...", data={"type": "placeholder"})

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        data = node.data
        if not data or data.get("type") == "placeholder":
            return

        path = data.get("path")
        if not path:
            return

        path_str = str(path)
        if path_str in self._loaded_dirs:
            return

        # Clear placeholders
        if node.children and node.children[0].data.get("type") == "placeholder":
            node.remove_children()

        try:
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for entry in entries:
                if entry.name.startswith("."):
                    continue

                if entry.is_dir():
                    dir_node = node.add(
                        f"[blue]{entry.name}/[/]",
                        data={"path": entry, "type": "directory"},
                        expand=False
                    )
                    dir_node.add_leaf("...", data={"type": "placeholder"})
                else:
                    icon = self._get_file_icon(entry)
                    status_color = self._get_git_color(entry)
                    node.add_leaf(
                        f"[{status_color}]{icon} {entry.name}[/{status_color}]",
                        data={"path": entry, "type": "file"}
                    )
            self._loaded_dirs.add(path_str)
        except Exception:
            node.add_leaf("[red]Error[/]", data={"type": "error"})

    def refresh_data(self) -> None:
        """Reload workspace directories from disk."""
        self._loaded_dirs.clear()
        self.clear()
        self.root.label = "Workspace"
        self.root.data = {"type": "workspace-root"}
        self.on_mount()


class ExplorerWidget(Vertical):
    """
    Unified explorer for Projects and Filesystem.
    """

    DEFAULT_CSS = """
    ExplorerWidget {
        height: 100%;
        background: $surface;
    }

    ExplorerWidget Tree {
        background: $surface;
        padding: 0 1;
    }
    ExplorerWidget > TabbedContent {
        height: 100%;
    }
    """

    class FileSelected(Message):
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    class ProjectSelected(Message):
        def __init__(self, project: ContextRoot) -> None:
            self.project = project
            super().__init__()

    def __init__(self, workspace_dirs: Optional[List[WorkspaceDirectory]] = None, id: str | None = None):
        super().__init__(id=id)
        self.workspace_dirs = workspace_dirs or []
        self.git = GitProvider()
        self._loaded_dirs: Set[str] = set()

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Projects", id="tab-projects"):
                yield ProjectTree(label="Projects", id="project-tree", git_provider=self.git)
            with TabPane("Files", id="tab-files"):
                yield FilesystemTree(
                    workspace_dirs=self.workspace_dirs,
                    git_provider=self.git,
                    id="fs-tree"
                )

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        event.stop()
        self._handle_tree_data(event.node.data)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Preview items as the cursor moves."""
        event.stop()
        self._handle_tree_data(event.node.data)

    def _handle_tree_data(self, data) -> None:  # type: ignore[no-untyped-def]
        if not data:
            return

        if isinstance(data, Path):
            if data.is_file():
                self.post_message(self.FileSelected(data))
            elif data.is_dir():
                # Highlighting a directory doesn't auto-open
                return
        elif isinstance(data, ContextRoot):
            self.post_message(self.ProjectSelected(data))
        elif isinstance(data, dict):
            # Handle placeholder nodes for ProjectTree
            if data.get("type") == "placeholder" and "context_root" in data:
                self.post_message(self.ProjectSelected(data["context_root"]))
            elif data.get("type") == "global_context":
                try:
                    from hafs.config.loader import load_config
                    from hafs.core.afs.manager import AFSManager
                    from hafs.core.afs.discovery import load_context_root

                    manager = AFSManager(load_config())
                    manager.ensure(Path.home())
                    project = load_context_root(Path.home() / ".context")
                    if project:
                        project = ContextRoot(
                            path=project.path,
                            project_name="Global Context",
                            metadata=project.metadata,
                            mounts=project.mounts,
                        )
                        self.post_message(self.ProjectSelected(project))
                except Exception:
                    return
            elif data.get("type") == "mount_type_category" and "project" in data:
                self.post_message(self.ProjectSelected(data["project"]))
            elif "path" in data and isinstance(data["path"], Path):
                if data["path"].is_file():
                    self.post_message(self.FileSelected(data["path"]))

    def refresh_data(self) -> None:
        """Refresh both project and filesystem panes."""
        try:
            self.query_one("#project-tree", ProjectTree).refresh_data()
        except Exception:
            pass

        try:
            fs_tree = self.query_one("#fs-tree", FilesystemTree)
            fs_tree.workspace_dirs = self.workspace_dirs
            fs_tree.refresh_data()
        except Exception:
            pass
