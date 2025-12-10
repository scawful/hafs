"""Main dashboard screen for HAFS TUI."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Header, Footer

from hafs.ui.widgets.project_tree import ProjectTree, ProjectSelected
from hafs.ui.widgets.context_viewer import ContextViewer
from hafs.ui.widgets.stats_panel import StatsPanel


class MainScreen(Screen):
    """Main dashboard screen with project browser and context viewer."""

    BINDINGS = [
        ("r", "refresh", "Refresh"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the screen."""
        yield Header()

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

    def action_refresh(self) -> None:
        """Refresh all data."""
        self.query_one(ProjectTree).refresh_data()
        self.query_one(StatsPanel).refresh_data()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
