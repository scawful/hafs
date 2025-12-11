"""Main dashboard screen for HAFS TUI."""

import os
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Static

from hafs.config.loader import load_config
from hafs.core.afs.discovery import discover_projects
from hafs.core.search import fuzzy_autocomplete
from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.screens.ai_context_modal import AIContextModal
from hafs.ui.screens.context_selection_modal import ContextSelectionModal
from hafs.ui.screens.file_picker_modal import FilePickerModal
from hafs.ui.screens.input_modal import InputModal
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.filesystem_tree import FilesystemTree
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.keybinding_bar import (
    KeyBindingBar,
    MAIN_SCREEN_BINDINGS_ROW1,
    MAIN_SCREEN_BINDINGS_ROW2,
)
from hafs.ui.widgets.project_tree import FileSelected, ProjectSelected, ProjectTree
from hafs.ui.widgets.sidebar_panel import SidebarPanel
from hafs.ui.widgets.stats_panel import StatsPanel


class MainScreen(Screen, VimNavigationMixin):
    """Main dashboard screen with project browser and context viewer."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "focus_search", "Search"),
        Binding("ctrl+k", "command_palette", "Commands"),
        Binding("a", "add_item", "Add File/Dir"),
        Binding("d", "delete_item", "Delete"),
        Binding("e", "edit_item", "Edit"),
        Binding("g", "ai_generate", "AI Generate"),
        Binding("c", "context_chat", "Chat w/ Context"),
        Binding("x", "add_to_context", "Add to Context"),
        Binding("w", "add_workspace", "Add Workspace"),
        Binding("[", "shrink_sidebar", "Shrink"),
        Binding("]", "expand_sidebar", "Expand"),
        # Vim navigation bindings
        *VimNavigationMixin.VIM_BINDINGS,
    ]

    LAYERS = ["base", "overlay"]

    DEFAULT_CSS = """
    MainScreen #search-container {
        height: auto;
        width: 100%;
        background: $surface;
        padding: 0 1;
    }

    MainScreen #search-input {
        width: 100%;
        height: 3;
        margin: 0;
        padding: 0 1;
        border: solid $primary-darken-2;
        background: $background;
    }

    MainScreen #search-input:focus {
        border: solid $primary;
    }

    MainScreen #search-results {
        display: none;
        height: auto;
        max-height: 10;
        background: $surface;
        border: solid $primary;
        border-top: none;
        padding: 0;
        margin: 0;
    }

    MainScreen #search-results.visible {
        display: block;
    }

    MainScreen .search-result-item {
        height: 1;
        padding: 0 1;
    }

    MainScreen .search-result-item:hover {
        background: $primary;
    }

    MainScreen .search-result-item.selected {
        background: $primary;
    }

    MainScreen #sidebar {
        width: 32;
        min-width: 20;
        max-width: 60;
        background: $surface;
        border-right: solid $primary-darken-2;
    }

    MainScreen .sidebar-section {
        height: auto;
        min-height: 3;
        max-height: 50%;
        border-bottom: solid $primary-darken-2;
    }

    MainScreen .sidebar-section:focus-within {
        border-left: solid $primary;
    }

    MainScreen .section-header {
        height: 1;
        background: $primary-darken-2;
        padding: 0 1;
    }

    MainScreen .section-header:hover {
        background: $primary;
    }

    MainScreen #workspace-panel {
        height: 1fr;
        min-height: 6;
        border-bottom: solid $primary-darken-2;
    }

    MainScreen #agents-panel {
        height: auto;
        min-height: 4;
        max-height: 10;
    }

    MainScreen #filesystem-tree {
        height: 100%;
    }

    MainScreen #content {
        background: $background;
    }

    MainScreen #stats-panel {
        height: auto;
        max-height: 6;
    }

    MainScreen #footer-area {
        height: auto;
        background: $surface;
    }

    MainScreen #keybinding-bar {
        border-top: solid $primary-darken-2;
    }
    """

    _sidebar_width: int = 30

    def compose(self) -> ComposeResult:
        """Compose the screen with lazygit-style multi-panel sidebar."""
        yield HeaderBar(id="header-bar")

        # Search bar at top (VS Code style with popup results)
        with Container(id="search-container"):
            yield Input(placeholder="Search files, commands... (Ctrl+P)", id="search-input")
            with Vertical(id="search-results"):
                yield Static("[dim]Type to search...[/]", id="search-placeholder")

        with Horizontal(id="main-container"):
            # Sidebar with multiple collapsible panels (lazygit-style)
            with Vertical(id="sidebar"):
                # Projects panel (tree only, no duplicate header)
                with Container(id="projects-panel", classes="sidebar-section"):
                    yield ProjectTree(id="project-tree")

                # Workspace panel (filesystem browser)
                with Container(id="workspace-panel", classes="sidebar-section"):
                    config = load_config()
                    yield FilesystemTree(
                        workspace_dirs=config.general.workspace_directories,
                        show_hidden=config.general.show_hidden_files,
                        id="filesystem-tree",
                    )

                # Agents panel (active agents status)
                with Container(id="agents-panel", classes="sidebar-section"):
                    yield Static(
                        "[bold]Agents[/] [dim](0 active)[/]\n"
                        "[dim]  No active agents[/]\n"
                        "[dim]  Press [bold]c[/] to start chat[/]",
                        id="agents-list",
                    )

            # Main content area
            with Vertical(id="content"):
                yield ContextViewer(id="context-viewer")
                yield StatsPanel(id="stats-panel")

        # Footer area with outline
        with Container(id="footer-area"):
            yield KeyBindingBar(
                row1=MAIN_SCREEN_BINDINGS_ROW1,
                row2=MAIN_SCREEN_BINDINGS_ROW2,
                id="keybinding-bar",
            )
            yield Footer()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        # Initialize vim navigation (loads setting from config)
        self.init_vim_navigation()

    def on_header_bar_menu_selected(self, event: HeaderBar.MenuSelected) -> None:
        """Handle header bar menu selections."""
        if event.menu_id == "palette":
            self._open_command_palette()
        elif event.menu_id == "view":
            # Toggle sidebar visibility or other view options
            pass
        elif event.menu_id == "file":
            # File operations menu
            pass

    def _open_command_palette(self) -> None:
        """Open the command palette."""
        try:
            from hafs.ui.screens.command_palette import CommandPalette
            self.app.push_screen(CommandPalette())
        except ImportError:
            self.notify("Command palette not available", severity="warning")

    def action_command_palette(self) -> None:
        """Open command palette (Ctrl+K)."""
        self._open_command_palette()

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

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input changes for live results."""
        if event.input.id == "search-input":
            query = event.value.strip()
            results_container = self.query_one("#search-results", Vertical)

            if not query:
                results_container.remove_class("visible")
                return

            # Show results container
            results_container.add_class("visible")

            # Get search results
            results = self._search_files(query)

            # Update results display
            self._update_search_results(results)

    def _search_files(self, query: str) -> list[tuple[str, float]]:
        """Search files and return matches with scores."""
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
            return []

        return fuzzy_autocomplete(query, all_files, limit=8, threshold=30)

    def _update_search_results(self, results: list[tuple[str, float]]) -> None:
        """Update the search results dropdown."""
        results_container = self.query_one("#search-results", Vertical)

        # Clear existing results
        results_container.remove_children()

        if not results:
            results_container.mount(Static("[dim]No matches found[/]", classes="search-result-item"))
            return

        # Add result items
        for i, (path, score) in enumerate(results):
            file_path = Path(path)
            display = f"[bold]{file_path.name}[/] [dim]{file_path.parent}[/]"
            item = Static(display, classes="search-result-item", id=f"result-{i}")
            item.data = file_path  # Store path for selection
            results_container.mount(item)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        if event.input.id == "search-input":
            query = event.value.strip()
            if not query:
                return

            # Get first result and open it
            results = self._search_files(query)
            if results:
                best_match, score = results[0]
                best_path = Path(best_match)
                self.query_one("#context-viewer", ContextViewer).set_file(best_path)
                self.notify(f"Opened: {best_path.name}")
            else:
                self.notify(f"No matches found for '{query}'", severity="warning")

            # Hide results and clear input
            self.query_one("#search-results", Vertical).remove_class("visible")
            event.input.value = ""

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
        try:
            self.query_one("#filesystem-tree", FilesystemTree).refresh_data()
        except Exception:
            pass

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

    def action_context_chat(self) -> None:
        """Open context selection modal and switch to chat with selected context."""
        def on_context_selected(selected_paths: list[Path] | None) -> None:
            if not selected_paths:
                return

            # Import OrchestratorScreen here to avoid circular imports
            from hafs.ui.screens.orchestrator import OrchestratorScreen

            # Create orchestrator screen with coordinator from app
            coordinator = getattr(self.app, "_coordinator", None)
            orch_screen = OrchestratorScreen(coordinator=coordinator)

            # Switch to orchestrator screen
            self.app.push_screen(orch_screen)

            # Show notification about selected context
            count = len(selected_paths)
            self.notify(
                f"Switched to chat with {count} context item{'s' if count != 1 else ''}",
                timeout=3,
            )

            # TODO: Pass selected_paths to coordinator/context panel
            # This could be done by updating the shared context with file paths
            # or by adding them to the context panel directly

        self.app.push_screen(ContextSelectionModal(), on_context_selected)

    def action_add_to_context(self) -> None:
        """Add selected file/directory from filesystem tree to context."""
        try:
            fs_tree = self.query_one("#filesystem-tree", FilesystemTree)
            fs_tree.action_add_to_context()
        except Exception:
            self.notify("Select a file in the workspace tree first", severity="warning")

    def action_add_workspace(self) -> None:
        """Add a new workspace directory via file picker."""
        def on_dir_selected(selected: Path | None) -> None:
            if not selected or not selected.is_dir():
                return

            # Load current config
            config = load_config()

            # Check if already exists
            existing_paths = [ws.path for ws in config.general.workspace_directories]
            if selected in existing_paths:
                self.notify(f"'{selected.name}' is already in workspace", severity="warning")
                return

            # Add new workspace directory
            from hafs.config.schema import WorkspaceDirectory
            config.general.workspace_directories.append(
                WorkspaceDirectory(path=selected, name=selected.name)
            )

            # Save config
            try:
                from hafs.config.saver import save_config
                save_config(config)
            except ImportError:
                self.notify("Install tomli-w to persist: pip install tomli-w", severity="warning")

            # Refresh filesystem tree
            try:
                fs_tree = self.query_one("#filesystem-tree", FilesystemTree)
                fs_tree.set_workspace_dirs(config.general.workspace_directories)
            except Exception:
                pass

            self.notify(f"Added '{selected.name}' to workspace")

        # Use file picker starting from home
        self.app.push_screen(
            FilePickerModal(
                base_path=Path.home(),
                prompt="Select directory to add to workspace",
                allow_new=False,
            ),
            on_dir_selected,
        )

    def on_filesystem_tree_file_selected(self, event: FilesystemTree.FileSelected) -> None:
        """Handle file selection from filesystem tree."""
        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_file(event.path)

    def on_filesystem_tree_file_add_to_context(
        self, event: FilesystemTree.FileAddToContext
    ) -> None:
        """Handle adding file to context."""
        # For now, just show in context viewer
        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_file(event.path)
        self.notify(f"Added '{event.path.name}' to context")

    def on_filesystem_tree_directory_add_to_context(
        self, event: FilesystemTree.DirectoryAddToContext
    ) -> None:
        """Handle adding directory to context."""
        self.notify(f"Added directory '{event.path.name}' to context")
