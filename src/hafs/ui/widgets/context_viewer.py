"""Context viewer widget for displaying AFS structure."""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Label
from textual.widget import Widget

from hafs.models.afs import ContextRoot, MountType


class ContextViewer(Widget):
    """Widget for viewing AFS context structure."""

    DEFAULT_CSS = """
    ContextViewer {
        height: 100%;
        background: $surface;
        padding: 1;
    }

    .cv-title {
        color: $secondary;
        text-style: bold;
        margin-bottom: 1;
    }

    .cv-section {
        margin-bottom: 1;
    }

    .cv-section-header {
        text-style: bold;
    }

    .cv-empty {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    """

    def __init__(self, project: ContextRoot | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._project = project

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with VerticalScroll():
            if self._project:
                yield Label(f"[bold]{self._project.project_name}[/bold]", classes="cv-title")
                yield Label(f"[dim]{self._project.path}[/dim]")
                yield Static("")  # Spacer

                for mt in MountType:
                    mounts = self._project.mounts.get(mt, [])
                    color = self._get_mount_color(mt)

                    with Vertical(classes="cv-section"):
                        yield Label(
                            f"[{color}]● {mt.value.upper()}[/{color}] ({len(mounts)})",
                            classes="cv-section-header",
                        )

                        if mounts:
                            for mount in mounts:
                                arrow = "→" if mount.is_symlink else "·"
                                yield Label(f"  {mount.name} {arrow} {mount.source}")
                        else:
                            yield Label("  [dim](empty)[/dim]")
            else:
                yield Label("Select a project to view context", classes="cv-empty")

    def set_project(self, project: ContextRoot | None) -> None:
        """Update the displayed project.

        Args:
            project: Project to display, or None to show empty state.
        """
        self._project = project
        self.refresh(recompose=True)

    @staticmethod
    def _get_mount_color(mt: MountType) -> str:
        """Get color for mount type."""
        return {
            MountType.MEMORY: "blue",
            MountType.KNOWLEDGE: "blue",
            MountType.TOOLS: "red",
            MountType.SCRATCHPAD: "green",
            MountType.HISTORY: "dim",
        }.get(mt, "white")
