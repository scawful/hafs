"""Main dashboard screen for HAFS TUI."""

import os
import shutil
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Header, Footer, Input

from hafs.ui.widgets.project_tree import ProjectTree, ProjectSelected, FileSelected
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.stats_panel import StatsPanel
from hafs.ui.screens.input_modal import InputModal


class MainScreen(Screen):
    """Main dashboard screen with project browser and context viewer."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
        ("ctrl+p", "focus_search", "Search"),
        ("a", "add_item", "Add File/Dir"),
        ("d", "delete_item", "Delete"),
        ("e", "edit_item", "Edit"),
    ]

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
        """Handle search input."""
        if event.input.id == "search-input":
            query = event.value
            self.notify(f"Fuzzy search for '{query}' not implemented yet.", severity="warning")

    def action_edit_item(self) -> None:
        """Edit current file."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node
        if node and isinstance(node.data, Path) and node.data.is_file():
            self._edit_file(node.data)

    def _edit_file(self, path: Path) -> None:
        """Open file in editor."""
        editor = os.environ.get("EDITOR", "vim")
        self.app.suspend_process()
        subprocess.run([editor, str(path)])
        self.app.resume_process()
        # Refresh viewer
        self.query_one(ContextViewer).set_file(path)

    def action_add_item(self) -> None:
        """Add new file/folder."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node
        if not node: return
        
        parent_path = node.data if isinstance(node.data, Path) else None
        
        if not parent_path:
            self.notify("Select a directory inside a mount to add to.", severity="warning")
            return
            
        if parent_path.is_file():
            parent_path = parent_path.parent
            
        def on_submit(name: str) -> None:
            if not name: return
            new_path = parent_path / name
            try:
                if new_path.exists():
                    self.notify("File already exists.", severity="error")
                    return
                
                if name.endswith("/"):
                    new_path.mkdir()
                    msg = f"Created directory {name}"
                else:
                    new_path.touch()
                    msg = f"Created file {name}"
                    
                tree.refresh_data()
                self.notify(msg)
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")

        self.app.push_screen(InputModal("Enter filename (end with / for dir):"), on_submit)

    def action_delete_item(self) -> None:
        """Delete item."""
        tree = self.query_one(ProjectTree)
        node = tree.cursor_node
        if not node or not isinstance(node.data, Path): return
        
        path = node.data
        try:
            if path.is_dir():
                try:
                    path.rmdir()
                    self.notify(f"Deleted {path.name}")
                except OSError:
                    self.notify("Directory not empty. Recursive delete not supported.", severity="error")
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
