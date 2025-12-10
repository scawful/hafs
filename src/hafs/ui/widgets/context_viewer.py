"""Context viewer widget for displaying AFS structure and files."""

from pathlib import Path
from rich.syntax import Syntax

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static, Label, Markdown
from textual.widget import Widget

from hafs.models.afs import ContextRoot, MountType


class ContextViewer(Widget):
    """Widget for viewing AFS context structure or file content."""

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

    def __init__(self, project: ContextRoot | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._file_path: Path | None = None

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with VerticalScroll():
            if self._file_path:
                yield from self._compose_file_view()
            elif self._project:
                yield from self._compose_project_view()
            else:
                yield Label("Select a project or file to view", classes="cv-empty")

    def _compose_file_view(self) -> ComposeResult:
        """Compose file content view."""
        if not self._file_path:
            return
            
        yield Label(f"[bold]{self._file_path.name}[/bold]", classes="cv-title")
        yield Label(f"[dim]{self._file_path}[/dim]")
        
        try:
            # Read content with limit
            # Note: For very large files, we should probably stream or lazily load
            # But for text files in context, 100KB is reasonable
            content = self._file_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 100000:
                content = content[:100000] + "\n... (truncated)"
            
            if self._file_path.suffix.lower() == ".md":
                yield Markdown(content)
            else:
                # Syntax highlighting
                try:
                    syntax = Syntax.from_path(
                        str(self._file_path),
                        content,
                        theme="monokai",
                        word_wrap=True
                    )
                    yield Static(syntax)
                except Exception:
                    # Fallback if syntax detection fails
                    yield Static(content)
                
        except Exception as e:
            yield Label(f"[red]Error reading file: {e}[/red]", classes="cv-empty")

    def _compose_project_view(self) -> ComposeResult:
        """Compose project structure view."""
        if not self._project:
            return

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

    def set_project(self, project: ContextRoot | None) -> None:
        """Update the displayed project."""
        self._project = project
        self._file_path = None
        self.refresh(recompose=True)

    def set_file(self, path: Path) -> None:
        """Display a file."""
        self._project = None
        self._file_path = path
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