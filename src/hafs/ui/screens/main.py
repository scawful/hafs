"""Main dashboard screen for HAFS TUI."""

import os
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static

from hafs.core.afs.discovery import discover_projects
from hafs.core.search import fuzzy_autocomplete
from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.screens.ai_context_modal import AIContextModal
from hafs.ui.screens.file_picker_modal import FilePickerModal
from hafs.ui.screens.input_modal import InputModal
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.project_tree import FileSelected, ProjectSelected, ProjectTree
from hafs.ui.widgets.stats_panel import StatsPanel


class MainScreen(Screen, VimNavigationMixin):
    """Main dashboard screen with project browser and context viewer."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "focus_search", "Search"),
        Binding("a", "add_item", "Add File/Dir"),
        Binding("d", "delete_item", "Delete"),
        Binding("e", "edit_item", "Edit"),
        Binding("g", "ai_generate", "AI Generate"),
        Binding("[", "shrink_sidebar", "Shrink"),
        Binding("]", "expand_sidebar", "Expand"),
        # Vim navigation bindings
        *VimNavigationMixin.VIM_BINDINGS,
    ]

    DEFAULT_CSS = """
    MainScreen #sidebar {
        width: 30;
        min-width: 15;
        max-width: 60;
        background: $surface;
        border-right: solid $primary;
    }

    MainScreen #sidebar-title {
        text-align: center;
        padding: 1;
    }
    """

    _sidebar_width: int = 30

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()

        # Search Bar
        yield Input(placeholder="Search context...", id="search-input")

        with Horizontal(id="main-container"):
            # Sidebar with project tree
            with Container(id="sidebar"):
                yield Static("[bold purple]PROJECTS[/bold purple]", id="sidebar-title")
                yield ProjectTree(id="project-tree")

            # Main content area
            with Vertical(id="content"):
                yield ContextViewer(id="context-viewer")
                yield StatsPanel(id="stats-panel")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        self.title = "HAFS - Dashboard"
        # Initialize vim navigation (disabled by default, toggle with Ctrl+V)
        self.init_vim_navigation(enabled=False)

    def on_project_selected(self, event: ProjectSelected) -> None:
        """Handle project selection from tree."""
        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_project(event.project)

    def on_file_selected(self, event: FileSelected) -> None:
        """Handle file selection from tree."""
        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_file(event.path)

    def action_focus_search(self) -> None:
        """Focus the search bar."""
        self.query_one("#search-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input (fuzzy search)."""
        if event.input.id == "search-input":
            query = event.value.strip()
            if not query:
                return

            # Collect all file paths from projects
            all_files: list[str] = []
            projects = discover_projects()
            for project in projects:
                for mount_list in project.mounts.values():
                    for mount in mount_list:
                        if mount.source.is_dir():
                            try:
                                for file_path in mount.source.rglob("*"):
                                    if file_path.is_file() and not file_path.name.startswith("."):
                                        all_files.append(str(file_path))
                            except PermissionError:
                                pass

            if not all_files:
                self.notify("No files found to search", severity="warning")
                return

            # Use fuzzy matching to find best matches
            results = fuzzy_autocomplete(query, all_files, limit=10, threshold=30)

            if results:
                # Select the best match and show it in context viewer
                best_match, score = results[0]
                best_path = Path(best_match)
                self.query_one("#context-viewer", ContextViewer).set_file(best_path)
                self.notify(f"Found: {best_path.name} (score: {score:.0f})")
            else:
                self.notify(f"No matches found for '{query}'", severity="warning")

    def action_edit_item(self) -> None:
        """Edit current file."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node
        if node and isinstance(node.data, Path) and node.data.is_file():
            self._edit_file(node.data)

    def _edit_file(self, path: Path) -> None:
        """Open file in editor."""
        editor = os.environ.get("EDITOR", "vim")
        self.app.suspend_process()  # type: ignore[attr-defined]
        subprocess.run([editor, str(path)])
        self.app.resume_process()  # type: ignore[attr-defined]
        # Refresh viewer
        self.query_one(ContextViewer).set_file(path)

    def action_add_item(self) -> None:
        """Add new file/folder with fuzzy file picker."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node

        # Determine parent path from current selection
        parent_path: Path | None = None

        if node:
            if isinstance(node.data, Path):
                parent_path = node.data if node.data.is_dir() else node.data.parent
            elif hasattr(node.data, "root_path"):
                # ContextRoot object
                parent_path = node.data.root_path

        # If no valid path selected, try to get a reasonable default
        if not parent_path:
            # Try to find a project root to add to
            projects = discover_projects()
            if projects:
                # Use first project's root
                parent_path = projects[0].root_path
                self.notify(f"Adding to: {parent_path.name}", timeout=2)
            else:
                self.notify(
                    "No project selected. Create a project first or select a directory.",
                    severity="warning"
                )
                return

        def on_file_selected(selected: Path | None) -> None:
            if not selected:
                return

            try:
                if selected.exists():
                    # Existing file/dir selected - open in viewer
                    if selected.is_file():
                        self.query_one("#context-viewer", ContextViewer).set_file(selected)
                        self.notify(f"Opened: {selected.name}")
                    else:
                        tree.refresh_data()
                        self.notify(f"Selected directory: {selected.name}")
                else:
                    # Create new file/directory
                    name = selected.name
                    if name.endswith("/"):
                        selected.mkdir(parents=True)
                        msg = f"Created directory: {name}"
                    else:
                        selected.parent.mkdir(parents=True, exist_ok=True)
                        selected.touch()
                        msg = f"Created file: {name}"

                    tree.refresh_data()
                    self.notify(msg)
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")

        self.app.push_screen(
            FilePickerModal(
                base_path=parent_path,
                prompt=f"Add file/directory in {parent_path.name}",
                allow_new=True,
            ),
            on_file_selected
        )

    def action_delete_item(self) -> None:
        """Delete item."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node
        if not node or not isinstance(node.data, Path):
            return

        path = node.data
        try:
            if path.is_dir():
                try:
                    path.rmdir()
                    self.notify(f"Deleted {path.name}")
                except OSError:
                    self.notify(
                        "Directory not empty. Recursive delete not supported.",
                        severity="error",
                    )
            else:
                path.unlink()
                self.notify(f"Deleted {path.name}")
            tree.refresh_data()
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.query_one(ProjectTree).refresh_data()
        self.query_one(StatsPanel).refresh_data()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_shrink_sidebar(self) -> None:
        """Shrink the sidebar width."""
        self._sidebar_width = max(15, self._sidebar_width - 5)
        self._update_sidebar_width()

    def action_expand_sidebar(self) -> None:
        """Expand the sidebar width."""
        self._sidebar_width = min(60, self._sidebar_width + 5)
        self._update_sidebar_width()

    def _update_sidebar_width(self) -> None:
        """Apply the current sidebar width."""
        try:
            sidebar = self.query_one("#sidebar", Container)
            sidebar.styles.width = self._sidebar_width
        except Exception:
            pass

    def action_ai_generate(self) -> None:
        """Open AI context generation modal."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node

        # Determine target path from current selection
        target_path: Path | None = None

        if node:
            if isinstance(node.data, Path):
                target_path = node.data if node.data.is_dir() else node.data.parent
            elif hasattr(node.data, "root_path"):
                target_path = node.data.root_path

        # Fall back to first project's root if nothing selected
        if not target_path:
            projects = discover_projects()
            if projects:
                target_path = projects[0].root_path
            else:
                self.notify(
                    "No project selected. Select a directory first.",
                    severity="warning"
                )
                return

        # Get config if available
        config = getattr(self.app, "config", None)

        self.app.push_screen(
            AIContextModal(
                target_path=target_path,
                config=config,
                backend="gemini",  # Default to Gemini
            )
        )
