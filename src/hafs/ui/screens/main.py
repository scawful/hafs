"""Main dashboard screen for HAFS TUI."""

import os
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Static, TabbedContent, TabPane

from hafs.config.loader import load_config
from hafs.core.afs.discovery import discover_projects
from hafs.core.search import fuzzy_autocomplete
from hafs.ui.mixins.vim_navigation import VimNavigationMixin
from hafs.ui.screens.ai_context_modal import AIContextModal
from hafs.ui.screens.context_selection_modal import ContextSelectionModal
from hafs.ui.screens.file_picker_modal import FilePickerModal
from hafs.ui.screens.permissions_modal import PermissionsModal
from hafs.ui.utils.file_ops import can_edit_file, rename_path
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.dev_dashboard import DevDashboard
from hafs.ui.widgets.explorer import ExplorerWidget
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.keybinding_bar import (
    MAIN_SCREEN_BINDINGS_ROW1,
    MAIN_SCREEN_BINDINGS_ROW2,
    KeyBindingBar,
)
from hafs.ui.widgets.policy_summary import PolicySummary
from hafs.ui.widgets.stats_panel import StatsPanel


class MainScreen(Screen, VimNavigationMixin):
    """Main dashboard screen with project browser and context viewer."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("q", "quit", "Quit"),
        Binding("ctrl+p", "focus_search", "Search"),
        Binding("ctrl+k", "command_palette", "Commands"),
        Binding("a", "add_item", "Add File/Dir"),
        Binding("p", "edit_policies", "Policies"),
        Binding("d", "delete_item", "Delete"),
        Binding("e", "edit_item", "Edit"),
        Binding("ctrl+s", "save_file", "Save"),
        Binding("m", "toggle_markdown_preview", "MD View"),
        Binding("f2", "rename_item", "Rename"),
        Binding("f5", "duplicate_item", "Copy"),
        Binding("ctrl+o", "open_external", "Open OS"),
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
    MainScreen {
        layout: vertical;
    }

    MainScreen #main-container {
        height: 1fr;
    }

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
        border-top: solid $primary-darken-1;
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
        height: 100%;
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
        height: 1fr;
    }

    MainScreen #stats-panel {
        height: auto;
        max-height: 6;
    }

    MainScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary-darken-2;
        padding-top: 0;
    }

    MainScreen #footer-grid {
        height: auto;
        width: 100%;
        layout: horizontal;
        align: center middle;
        padding: 0 1;
    }

    MainScreen #keybinding-bar {
        width: 2fr;
    }

    MainScreen Footer {
        width: 1fr;
        min-width: 20;
    }
    """

    _sidebar_width: int = 30

    def compose(self) -> ComposeResult:
        """Compose the screen with consolidated sidebar."""
        yield HeaderBar(id="header-bar")

        with Horizontal(id="main-container"):
            # Sidebar with ExplorerWidget
            with Vertical(id="sidebar"):
                config = load_config()
                yield ExplorerWidget(
                    workspace_dirs=config.general.workspace_directories,
                    id="explorer"
                )

                # Agents panel (active agents status) - Optional, kept for now
                with Container(id="agents-panel", classes="sidebar-section"):
                    yield Static(
                        "[bold]Agents[/] [dim](0 active)[/]\n"
                        "[dim]  No active agents[/]\n"
                        "[dim]  Press [bold]c[/] to start chat[/]",
                        id="agents-list",
                    )

            # Main content area
            with Vertical(id="content"):
                with TabbedContent(id="main-tabs"):
                    with TabPane("Context", id="tab-context"):
                        yield ContextViewer(id="context-viewer")
                    with TabPane("Dev Tools", id="tab-devtools"):
                        yield DevDashboard(id="dev-dashboard")
                yield PolicySummary(id="policy-summary")
                yield StatsPanel(id="stats-panel")

        # Search bar near footer to reduce header clutter
        with Container(id="search-container"):
            yield Input(placeholder="Search files, commands... (Ctrl+P)", id="search-input")
            with Vertical(id="search-results"):
                yield Static("[dim]Type to search...[/]", id="search-placeholder")

        # Footer area with outline
        with Container(id="footer-area"):
            with Horizontal(id="footer-grid"):
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
        try:
            policies = getattr(self.app, "config", load_config()).afs_directories  # type: ignore[attr-defined]
            self.query_one(PolicySummary).set_policies(policies)
        except Exception:
            pass

    def on_header_bar_menu_selected(self, event: HeaderBar.MenuSelected) -> None:
        """Handle header bar menu selections."""
        if event.menu_id == "palette":
            self._open_command_palette()
        elif event.menu_id == "context":
            self.action_context_chat()

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

    def on_explorer_widget_project_selected(self, event: ExplorerWidget.ProjectSelected) -> None:
        """Handle project selection from tree."""
        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_project(event.project)

    def on_explorer_widget_file_selected(self, event: ExplorerWidget.FileSelected) -> None:
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

        try:
            config = load_config()
            for ws_dir in config.general.workspace_directories:
                if ws_dir.path.exists():
                    try:
                        for file_path in ws_dir.path.rglob("*"):
                            if file_path.is_file() and not file_path.name.startswith("."):
                                all_files.append(str(file_path))
                    except PermissionError:
                        continue
        except Exception:
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
            results_container.mount(Static(
                "[dim]No matches found[/]", classes="search-result-item"
            ))
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
        """Edit current file (inline when possible, otherwise external editor)."""
        path = self._get_selected_path()
        if not path or not path.exists() or path.is_dir():
            self.notify("Select a file to edit", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)

        if can_edit_file(path):
            if viewer.enter_edit_mode():
                self.notify(f"Editing {path.name} (Ctrl+S to save)", timeout=2)
        else:
            self._edit_file_external(path)

    def _edit_file_external(self, path: Path) -> None:
        """Open file in external editor."""
        editor = os.environ.get("EDITOR", "vim")
        self.app.suspend_process()  # type: ignore[attr-defined]
        subprocess.run([editor, str(path)])
        self.app.resume_process()  # type: ignore[attr-defined]
        self.query_one(ContextViewer).set_file(path)
        self._refresh_explorer()

    def action_save_file(self) -> None:
        """Save inline edits if active."""
        viewer = self.query_one("#context-viewer", ContextViewer)
        if viewer.is_editing:
            viewer.save_edits()
        else:
            self.notify("No inline edit in progress", severity="warning")

    def action_toggle_markdown_preview(self) -> None:
        """Toggle markdown preview/raw view."""
        self.query_one("#context-viewer", ContextViewer).toggle_markdown_preview()

    def action_add_item(self) -> None:
        """Add new file/folder with fuzzy file picker."""
        base_path = self._get_selected_directory()

        def on_file_selected(selected: Path | None) -> None:
            if not selected:
                return

            try:
                if selected.exists():
                    if selected.is_file():
                        self.query_one("#context-viewer", ContextViewer).set_file(selected)
                        self.notify(f"Opened: {selected.name}")
                    else:
                        self.notify(f"Selected directory: {selected.name}")
                else:
                    if selected.name.endswith("/"):
                        selected.mkdir(parents=True, exist_ok=True)
                        msg = f"Created directory: {selected.name.rstrip('/')}"
                    else:
                        selected.parent.mkdir(parents=True, exist_ok=True)
                        selected.touch()
                        msg = f"Created file: {selected.name}"
                    self.notify(msg)
                self._refresh_explorer()
            except Exception as exc:
                self.notify(f"Error: {exc}", severity="error")

        self.app.push_screen(
            FilePickerModal(
                base_path=base_path,
                prompt=f"Add file/directory in {base_path.name}",
                allow_new=True,
            ),
            on_file_selected,
        )

    def action_delete_item(self) -> None:
        """Delete selected file/directory (non-recursive)."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file or directory to delete", severity="warning")
            return

        try:
            if path.is_dir():
                path.rmdir()
                msg = f"Deleted empty directory {path.name}"
            else:
                path.unlink()
                msg = f"Deleted {path.name}"
            self.notify(msg)
            self._refresh_explorer()
        except OSError:
            self.notify("Directory not empty. Recursive delete not supported.", severity="error")
        except Exception as exc:
            self.notify(f"Error deleting item: {exc}", severity="error")

    def action_rename_item(self) -> None:
        """Rename the currently selected file/directory."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file to rename", severity="warning")
            return

        def on_new_path(selected: Path | None) -> None:
            if not selected:
                return

            if path.is_dir():
                try:
                    rename_path(path, selected)
                    self.notify(f"Renamed to {selected.name}")
                    self._refresh_explorer()
                except Exception as exc:
                    self.notify(f"Rename failed: {exc}", severity="error")
            else:
                viewer = self.query_one("#context-viewer", ContextViewer)
                viewer.set_file(path)
                viewer.rename_current(selected)

        self.app.push_screen(
            FilePickerModal(
                base_path=path.parent,
                prompt=f"Rename {path.name}",
                allow_new=True,
            ),
            on_new_path,
        )

    def action_duplicate_item(self) -> None:
        """Duplicate the selected file."""
        path = self._get_selected_path()
        if not path or not path.is_file():
            self.notify("Select a file to duplicate", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        viewer.duplicate_current()

    def action_open_external(self) -> None:
        """Open the selected file externally."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file first", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        if viewer.open_external():
            self.notify(f"Opening {path.name} externally", timeout=2)

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.query_one(StatsPanel).refresh_data()
        try:
            policies = getattr(self.app, "config", load_config()).afs_directories  # type: ignore[attr-defined]
            self.query_one(PolicySummary).set_policies(policies)
        except Exception:
            pass
        self._refresh_explorer()

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
            sidebar = self.query_one("#sidebar", Vertical)
            sidebar.styles.width = self._sidebar_width
            self.notify(f"Sidebar width: {self._sidebar_width}", timeout=1)
        except Exception as e:
            self.notify(f"Sidebar error: {e}", severity="warning")

    def action_ai_generate(self) -> None:
        """Open AI context generation modal."""
        target_path = self._get_selected_directory()
        config = getattr(self.app, "config", None)

        self.app.push_screen(
            AIContextModal(
                target_path=target_path,
                config=config,
                backend="gemini",
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
            orch_screen = OrchestratorScreen(
                coordinator=coordinator,
                context_paths=selected_paths,
            )

            # Switch to orchestrator screen
            self.app.push_screen(orch_screen)

            # Show notification about selected context
            count = len(selected_paths)
            self.notify(
                f"Switched to chat with {count} context item{'s' if count != 1 else ''}",
                timeout=3,
            )

            if coordinator:
                coordinator.set_context_items(selected_paths)

        self.app.push_screen(ContextSelectionModal(), on_context_selected)

    def action_add_to_context(self) -> None:
        """Add selected file/directory from filesystem tree to context."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file or directory first", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        self.notify(f"Added '{path.name}' to context")

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

            try:
                explorer = self.query_one(ExplorerWidget)
                explorer.workspace_dirs = config.general.workspace_directories
                if hasattr(explorer, "refresh_data"):
                    explorer.refresh_data()
            except Exception:
                pass

            self.notify(f"Added '{selected.name}' to workspace")

        # Use file picker starting from home
        self.app.push_screen(
            FilePickerModal(
                base_path=Path.home(),
                prompt=f"Select directory to add to workspace",
                allow_new=False,
            ),
            on_dir_selected,
        )

    def action_edit_policies(self) -> None:
        """Open policy editor from the main dashboard."""
        config = load_config()
        context_path = Path.cwd() / ".context"
        self.app.push_screen(PermissionsModal(config.afs_directories, context_path))

    def on_permissions_modal_permissions_updated(
        self, event: PermissionsModal.PermissionsUpdated
    ) -> None:
        """Handle policy updates from the modal."""
        if hasattr(self.app, "config"):
            self.app.config.afs_directories = event.directories  # type: ignore[attr-defined]
        try:
            self.query_one(PolicySummary).set_policies(event.directories)
        except Exception:
            pass
        try:
            self.query_one(StatsPanel).refresh_data()
        except Exception:
            pass

    def on_vim_mode_toggled(self, event: VimNavigationMixin.VimModeToggled) -> None:
        """Surface vim mode status so users discover the bindings."""
        status = "ON" if event.enabled else "OFF"
        self.notify(
            f"Vim mode {status} (j/k/h/l, gg/G, /, : available)",
            timeout=2,
        )

    def on_context_viewer_file_saved(self, event: ContextViewer.FileSaved) -> None:
        """React to inline save events."""
        self.notify(f"Saved {event.path.name}")
        self._refresh_explorer()

    def on_context_viewer_file_renamed(self, event: ContextViewer.FileRenamed) -> None:
        """Refresh explorer when files are renamed."""
        self.notify(f"Renamed {event.old_path.name} → {event.new_path.name}", timeout=2)
        self._refresh_explorer()

    def on_context_viewer_file_duplicated(self, event: ContextViewer.FileDuplicated) -> None:
        """Refresh explorer when files are duplicated."""
        self.notify(f"Copied to {event.new_path.name}", timeout=2)
        self._refresh_explorer()

    def on_context_viewer_file_error(self, event: ContextViewer.FileError) -> None:
        """Surface file operation errors."""
        path = f"{event.path}" if event.path else "file"
        self.notify(f"{path}: {event.error}", severity="error")

    def _get_selected_path(self) -> Path | None:
        """Return the current selection from viewer or explorer."""
        viewer = self.query_one("#context-viewer", ContextViewer)
        if viewer.current_path and viewer.current_path.exists():
            return viewer.current_path

        try:
            explorer = self.query_one(ExplorerWidget)
            tabbed = explorer.query_one(TabbedContent)
            active_tab = tabbed.active
            tree_id = "#project-tree" if active_tab == "tab-projects" else "#fs-tree"
            tree = explorer.query_one(tree_id)
            node = tree.cursor_node
        except Exception:
            return None

        if not node or not node.data:
            return None

        data = node.data
        if isinstance(data, Path):
            return data
        if isinstance(data, dict):
            path = data.get("path")
            if isinstance(path, Path):
                return path
        if hasattr(data, "root_path"):
            root_path = getattr(data, "root_path")
            if isinstance(root_path, Path):
                return root_path
        return None

    def _get_selected_directory(self) -> Path:
        """Pick a reasonable directory for creating files."""
        path = self._get_selected_path()
        if path:
            return path if path.is_dir() else path.parent

        config = load_config()
        if config.general.workspace_directories:
            default = config.general.workspace_directories[0].path
            if default.exists():
                return default
        return Path.cwd()

    def _refresh_explorer(self) -> None:
        """Refresh the explorer widget safely."""
        try:
            explorer = self.query_one(ExplorerWidget)
            if hasattr(explorer, "refresh_data"):
                explorer.refresh_data()
        except Exception:
            pass

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        # Initialize vim navigation (loads setting from config)
        self.init_vim_navigation()
        try:
            policies = getattr(self.app, "config", load_config()).afs_directories  # type: ignore[attr-defined]
            self.query_one(PolicySummary).set_policies(policies)
        except Exception:
            pass

    def on_header_bar_menu_selected(self, event: HeaderBar.MenuSelected) -> None:
        """Handle header bar menu selections."""
        if event.menu_id == "palette":
            self._open_command_palette()
        elif event.menu_id == "context":
            self.action_context_chat()

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

    def on_explorer_widget_project_selected(self, event: ExplorerWidget.ProjectSelected) -> None:
        """Handle project selection from tree."""
        context_viewer = self.query_one("#context-viewer", ContextViewer)
        context_viewer.set_project(event.project)

    def on_explorer_widget_file_selected(self, event: ExplorerWidget.FileSelected) -> None:
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

        try:
            config = load_config()
            for ws_dir in config.general.workspace_directories:
                if ws_dir.path.exists():
                    try:
                        for file_path in ws_dir.path.rglob("*"):
                            if file_path.is_file() and not file_path.name.startswith("."):
                                all_files.append(str(file_path))
                    except PermissionError:
                        continue
        except Exception:
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
            results_container.mount(Static(
                "[dim]No matches found[/]", classes="search-result-item"
            ))
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
        """Edit current file (inline when possible, otherwise external editor)."""
        path = self._get_selected_path()
        if not path or not path.exists() or path.is_dir():
            self.notify("Select a file to edit", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)

        if can_edit_file(path):
            if viewer.enter_edit_mode():
                self.notify(f"Editing {path.name} (Ctrl+S to save)", timeout=2)
        else:
            self._edit_file_external(path)

    def _edit_file_external(self, path: Path) -> None:
        """Open file in external editor."""
        editor = os.environ.get("EDITOR", "vim")
        self.app.suspend_process()  # type: ignore[attr-defined]
        subprocess.run([editor, str(path)])
        self.app.resume_process()  # type: ignore[attr-defined]
        self.query_one(ContextViewer).set_file(path)
        self._refresh_explorer()

    def action_save_file(self) -> None:
        """Save inline edits if active."""
        viewer = self.query_one("#context-viewer", ContextViewer)
        if viewer.is_editing:
            viewer.save_edits()
        else:
            self.notify("No inline edit in progress", severity="warning")

    def action_toggle_markdown_preview(self) -> None:
        """Toggle markdown preview/raw view."""
        self.query_one("#context-viewer", ContextViewer).toggle_markdown_preview()

    def action_add_item(self) -> None:
        """Add new file/folder with fuzzy file picker."""
        base_path = self._get_selected_directory()

        def on_file_selected(selected: Path | None) -> None:
            if not selected:
                return

            try:
                if selected.exists():
                    if selected.is_file():
                        self.query_one("#context-viewer", ContextViewer).set_file(selected)
                        self.notify(f"Opened: {selected.name}")
                    else:
                        self.notify(f"Selected directory: {selected.name}")
                else:
                    if selected.name.endswith("/"):
                        selected.mkdir(parents=True, exist_ok=True)
                        msg = f"Created directory: {selected.name.rstrip('/')}"
                    else:
                        selected.parent.mkdir(parents=True, exist_ok=True)
                        selected.touch()
                        msg = f"Created file: {selected.name}"
                    self.notify(msg)
                self._refresh_explorer()
            except Exception as exc:
                self.notify(f"Error: {exc}", severity="error")

        self.app.push_screen(
            FilePickerModal(
                base_path=base_path,
                prompt=f"Add file/directory in {base_path.name}",
                allow_new=True,
            ),
            on_file_selected,
        )

    def action_delete_item(self) -> None:
        """Delete selected file/directory (non-recursive)."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file or directory to delete", severity="warning")
            return

        try:
            if path.is_dir():
                path.rmdir()
                msg = f"Deleted empty directory {path.name}"
            else:
                path.unlink()
                msg = f"Deleted {path.name}"
            self.notify(msg)
            self._refresh_explorer()
        except OSError:
            self.notify("Directory not empty. Recursive delete not supported.", severity="error")
        except Exception as exc:
            self.notify(f"Error deleting item: {exc}", severity="error")

    def action_rename_item(self) -> None:
        """Rename the currently selected file/directory."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file to rename", severity="warning")
            return

        def on_new_path(selected: Path | None) -> None:
            if not selected:
                return

            if path.is_dir():
                try:
                    rename_path(path, selected)
                    self.notify(f"Renamed to {selected.name}")
                    self._refresh_explorer()
                except Exception as exc:
                    self.notify(f"Rename failed: {exc}", severity="error")
            else:
                viewer = self.query_one("#context-viewer", ContextViewer)
                viewer.set_file(path)
                viewer.rename_current(selected)

        self.app.push_screen(
            FilePickerModal(
                base_path=path.parent,
                prompt=f"Rename {path.name}",
                allow_new=True,
            ),
            on_new_path,
        )

    def action_duplicate_item(self) -> None:
        """Duplicate the selected file."""
        path = self._get_selected_path()
        if not path or not path.is_file():
            self.notify("Select a file to duplicate", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        viewer.duplicate_current()

    def action_open_external(self) -> None:
        """Open the selected file externally."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file first", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        if viewer.open_external():
            self.notify(f"Opening {path.name} externally", timeout=2)

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.query_one(StatsPanel).refresh_data()
        try:
            policies = getattr(self.app, "config", load_config()).afs_directories  # type: ignore[attr-defined]
            self.query_one(PolicySummary).set_policies(policies)
        except Exception:
            pass
        self._refresh_explorer()

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
            sidebar = self.query_one("#sidebar", Vertical)
            sidebar.styles.width = self._sidebar_width
            self.notify(f"Sidebar width: {self._sidebar_width}", timeout=1)
        except Exception as e:
            self.notify(f"Sidebar error: {e}", severity="warning")

    def action_ai_generate(self) -> None:
        """Open AI context generation modal."""
        target_path = self._get_selected_directory()
        config = getattr(self.app, "config", None)

        self.app.push_screen(
            AIContextModal(
                target_path=target_path,
                config=config,
                backend="gemini",
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
            orch_screen = OrchestratorScreen(
                coordinator=coordinator,
                context_paths=selected_paths,
            )

            # Switch to orchestrator screen
            self.app.push_screen(orch_screen)

            # Show notification about selected context
            count = len(selected_paths)
            self.notify(
                f"Switched to chat with {count} context item{'s' if count != 1 else ''}",
                timeout=3,
            )

            if coordinator:
                coordinator.set_context_items(selected_paths)

        self.app.push_screen(ContextSelectionModal(), on_context_selected)

    def action_add_to_context(self) -> None:
        """Add selected file/directory from filesystem tree to context."""
        path = self._get_selected_path()
        if not path:
            self.notify("Select a file or directory first", severity="warning")
            return

        viewer = self.query_one("#context-viewer", ContextViewer)
        viewer.set_file(path)
        self.notify(f"Added '{path.name}' to context")

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

            try:
                explorer = self.query_one(ExplorerWidget)
                explorer.workspace_dirs = config.general.workspace_directories
                if hasattr(explorer, "refresh_data"):
                    explorer.refresh_data()
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

    def action_edit_policies(self) -> None:
        """Open policy editor from the main dashboard."""
        config = load_config()
        context_path = Path.cwd() / ".context"
        self.app.push_screen(PermissionsModal(config.afs_directories, context_path))

    def on_permissions_modal_permissions_updated(
        self, event: PermissionsModal.PermissionsUpdated
    ) -> None:
        """Handle policy updates from the modal."""
        if hasattr(self.app, "config"):
            self.app.config.afs_directories = event.directories  # type: ignore[attr-defined]
        try:
            self.query_one(PolicySummary).set_policies(event.directories)
        except Exception:
            pass
        try:
            self.query_one(StatsPanel).refresh_data()
        except Exception:
            pass

    def on_vim_mode_toggled(self, event: VimNavigationMixin.VimModeToggled) -> None:
        """Surface vim mode status so users discover the bindings."""
        status = "ON" if event.enabled else "OFF"
        self.notify(
            f"Vim mode {status} (j/k/h/l, gg/G, /, : available)",
            timeout=2,
        )

    def on_context_viewer_file_saved(self, event: ContextViewer.FileSaved) -> None:
        """React to inline save events."""
        self.notify(f"Saved {event.path.name}")
        self._refresh_explorer()

    def on_context_viewer_file_renamed(self, event: ContextViewer.FileRenamed) -> None:
        """Refresh explorer when files are renamed."""
        self.notify(f"Renamed {event.old_path.name} → {event.new_path.name}", timeout=2)
        self._refresh_explorer()

    def on_context_viewer_file_duplicated(self, event: ContextViewer.FileDuplicated) -> None:
        """Refresh explorer when files are duplicated."""
        self.notify(f"Copied to {event.new_path.name}", timeout=2)
        self._refresh_explorer()

    def on_context_viewer_file_error(self, event: ContextViewer.FileError) -> None:
        """Surface file operation errors."""
        path = f"{event.path}" if event.path else "file"
        self.notify(f"{path}: {event.error}", severity="error")

    def _get_selected_path(self) -> Path | None:
        """Return the current selection from viewer or explorer."""
        viewer = self.query_one("#context-viewer", ContextViewer)
        if viewer.current_path and viewer.current_path.exists():
            return viewer.current_path

        try:
            explorer = self.query_one(ExplorerWidget)
            tabbed = explorer.query_one(TabbedContent)
            active_tab = tabbed.active
            tree_id = "#project-tree" if active_tab == "tab-projects" else "#fs-tree"
            tree = explorer.query_one(tree_id)
            node = tree.cursor_node
        except Exception:
            return None

        if not node or not node.data:
            return None

        data = node.data
        if isinstance(data, Path):
            return data
        if isinstance(data, dict):
            path = data.get("path")
            if isinstance(path, Path):
                return path
        if hasattr(data, "root_path"):
            root_path = getattr(data, "root_path")
            if isinstance(root_path, Path):
                return root_path
        return None

    def _get_selected_directory(self) -> Path:
        """Pick a reasonable directory for creating files."""
        path = self._get_selected_path()
        if path:
            return path if path.is_dir() else path.parent

        config = load_config()
        if config.general.workspace_directories:
            default = config.general.workspace_directories[0].path
            if default.exists():
                return default
        return Path.cwd()

    def _refresh_explorer(self) -> None:
        """Refresh the explorer widget safely."""
        try:
            explorer = self.query_one(ExplorerWidget)
            if hasattr(explorer, "refresh_data"):
                explorer.refresh_data()
        except Exception:
            pass
