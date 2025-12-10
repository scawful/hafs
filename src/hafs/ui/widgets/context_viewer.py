"""Context viewer widget for displaying AFS structure and files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Label, Markdown, Static

from hafs.models.afs import ContextRoot, MountType

# File type categories
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".org",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
    ".c", ".cpp", ".h", ".hpp", ".cc", ".cxx",
    ".java", ".kt", ".kts", ".scala",
    ".go", ".rs", ".rb", ".php", ".lua",
    ".sh", ".bash", ".zsh", ".fish",
    ".html", ".htm", ".xml", ".xhtml", ".svg",
    ".css", ".scss", ".sass", ".less",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sql", ".graphql", ".gql",
    ".r", ".R", ".jl", ".m", ".swift",
    ".vim", ".el", ".lisp", ".clj", ".hs",
    ".makefile", ".cmake", ".gradle",
    ".dockerfile", ".containerfile",
    ".gitignore", ".gitattributes", ".editorconfig",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg"}
BINARY_EXTENSIONS = {".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".db", ".sqlite"}


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


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

    .cv-info {
        color: $text-muted;
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

    .cv-file-info {
        background: $surface-darken-1;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
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

        # File info panel
        yield from self._compose_file_info()

        suffix = self._file_path.suffix.lower()

        # Handle different file types
        if suffix in IMAGE_EXTENSIONS:
            yield from self._compose_image_view()
        elif suffix in BINARY_EXTENSIONS:
            yield from self._compose_binary_view()
        elif suffix in TEXT_EXTENSIONS or self._is_text_file():
            yield from self._compose_text_view()
        else:
            # Try to read as text, fall back to binary info
            if self._is_text_file():
                yield from self._compose_text_view()
            else:
                yield from self._compose_binary_view()

    def _compose_file_info(self) -> ComposeResult:
        """Compose file information panel."""
        if not self._file_path:
            return

        try:
            stat = self._file_path.stat()
            size = _format_size(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            suffix = self._file_path.suffix.lower() or "(no extension)"

            info_text = (
                f"[dim]Path:[/dim] {self._file_path}\n"
                f"[dim]Size:[/dim] {size}  [dim]Modified:[/dim] {modified}  [dim]Type:[/dim] {suffix}"
            )

            yield Static(info_text, classes="cv-file-info")
        except Exception:
            yield Label(f"[dim]{self._file_path}[/dim]", classes="cv-info")

    def _compose_text_view(self) -> ComposeResult:
        """Compose text file content view."""
        if not self._file_path:
            return

        try:
            content = self._file_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 100000:
                content = content[:100000] + "\n\n... [truncated - file too large]"

            suffix = self._file_path.suffix.lower()

            # Markdown rendering
            if suffix in {".md", ".markdown"}:
                yield Markdown(content)
            # JSON with pretty print option
            elif suffix == ".json":
                try:
                    parsed = json.loads(content)
                    formatted = json.dumps(parsed, indent=2)
                    syntax = Syntax(formatted, "json", theme="monokai", word_wrap=True, line_numbers=True)
                    yield Static(syntax)
                except json.JSONDecodeError:
                    syntax = Syntax(content, "json", theme="monokai", word_wrap=True)
                    yield Static(syntax)
            # Syntax highlighting for code
            else:
                try:
                    syntax = Syntax.from_path(
                        str(self._file_path),
                        content,
                        theme="monokai",
                        word_wrap=True,
                        line_numbers=True,
                    )
                    yield Static(syntax)
                except Exception:
                    yield Static(content)

        except Exception as e:
            yield Label(f"[red]Error reading file: {e}[/red]", classes="cv-empty")

    def _compose_image_view(self) -> ComposeResult:
        """Compose image file info view."""
        if not self._file_path:
            return

        suffix = self._file_path.suffix.lower()

        yield Label(
            f"[cyan]Image File[/cyan]\n\n"
            f"Format: {suffix.upper()}\n"
            f"[dim]Preview not available in terminal.[/dim]\n\n"
            f"[dim]Press 'e' to open with system viewer[/dim]",
            classes="cv-empty"
        )

    def _compose_binary_view(self) -> ComposeResult:
        """Compose binary file info view."""
        if not self._file_path:
            return

        try:
            # Read first 256 bytes for hex preview
            with open(self._file_path, "rb") as f:
                header = f.read(256)

            # Format as hex dump
            hex_lines = []
            for i in range(0, len(header), 16):
                chunk = header[i:i+16]
                hex_part = " ".join(f"{b:02x}" for b in chunk)
                ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
                hex_lines.append(f"{i:08x}  {hex_part:<48}  {ascii_part}")

            yield Label("[cyan]Binary File[/cyan] (first 256 bytes)", classes="cv-section-header")
            yield Static("\n".join(hex_lines))

        except Exception as e:
            yield Label(f"[red]Error reading binary file: {e}[/red]", classes="cv-empty")

    def _is_text_file(self) -> bool:
        """Check if file appears to be text."""
        if not self._file_path:
            return False

        try:
            with open(self._file_path, "rb") as f:
                chunk = f.read(8192)

            # Check for null bytes (common in binary files)
            if b"\x00" in chunk:
                return False

            # Try to decode as UTF-8
            try:
                chunk.decode("utf-8")
                return True
            except UnicodeDecodeError:
                return False

        except Exception:
            return False

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
