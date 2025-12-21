"""Context viewer widget for displaying AFS structure and files."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from rich.markup import escape
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Label, Markdown, Static, TextArea

from hafs.models.afs import ContextRoot, MountPoint, MountType
from hafs.ui.utils.file_ops import can_edit_file, duplicate_file, read_text_file, rename_path

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp', '.svg'}
BINARY_EXTENSIONS = {'.exe', '.dll', '.so', '.dylib', '.bin', '.dat', '.db', '.sqlite'}


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class MountItem(Static):
    """Clickable item representing a mount point."""
    
    DEFAULT_CSS = """
    MountItem {
        padding-left: 1;
        color: $text;
    }
    MountItem:hover {
        background: $primary;
        color: $text;
    }
    """

    def __init__(self, mount: MountPoint) -> None:
        arrow = "→" if mount.is_symlink else "·"
        label = f"{mount.name} [dim]{arrow} {mount.source.name}[/dim]"
        super().__init__(label)
        self.mount_point = mount

    def on_click(self) -> None:
        """Handle click to open file."""
        for ancestor in self.ancestors:
            if isinstance(ancestor, ContextViewer):
                ancestor.set_file(self.mount_point.source)
                return


class ContextViewer(Widget):
    """Widget for viewing AFS context structure or file content."""

    class FileSaved(Message):
        """Emitted when a file is saved from the inline editor."""

        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path

    class FileRenamed(Message):
        """Emitted after a rename operation succeeds."""

        def __init__(self, old_path: Path, new_path: Path) -> None:
            super().__init__()
            self.old_path = old_path
            self.new_path = new_path

    class FileDuplicated(Message):
        """Emitted after duplicating a file."""

        def __init__(self, source: Path, new_path: Path) -> None:
            super().__init__()
            self.source = source
            self.new_path = new_path

    class FileError(Message):
        """Emitted when a file operation fails."""

        def __init__(self, path: Path | None, error: str) -> None:
            super().__init__()
            self.path = path
            self.error = error

    DEFAULT_CSS = """
    ContextViewer {
        height: 100%;
        width: 100%;
        background: $surface;
        padding: 1;
    }
    
    ContextViewer > VerticalScroll {
        height: 100%;
        width: 100%;
    }

    .cv-title {
        color: $secondary;
        text-style: bold;
        margin-bottom: 1;
    }

    .cv-info {
        color: $text-disabled;
        margin-bottom: 1;
    }

    .cv-section {
        margin-bottom: 1;
    }

    .cv-section-header {
        text-style: bold;
    }

    .cv-empty {
        color: $text-disabled;
        text-align: center;
        padding: 2;
    }

    .cv-file-info {
        background: $surface-darken-1;
        padding: 1;
        margin-bottom: 1;
        border: solid $primary;
    }

    .cv-toolbar {
        color: $text-disabled;
        margin-bottom: 1;
    }

    .cv-editor {
        min-height: 8;
        height: auto;
    }
    """

    def __init__(self, project: ContextRoot | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._file_path: Path | None = None
        self._edit_mode: bool = False
        self._edit_buffer: str = ""
        self._markdown_preview: bool = True

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

        suffix = self._file_path.suffix.lower()
        is_text = can_edit_file(self._file_path)
        content: str | None = None

        yield Label(f"[bold]{escape(self._file_path.name)}[/bold]", classes="cv-title")
        yield from self._compose_file_info()

        if self._file_path.is_dir():
            yield Label(
                "[dim]Directory selected. Use add/rename/delete shortcuts from the sidebar.[/dim]",
                classes="cv-empty",
            )
            return

        if is_text:
            try:
                content = read_text_file(self._file_path)
            except Exception as exc:
                yield Label(f"[red]Error reading file: {exc}[/red]", classes="cv-empty")
                return

        if self._edit_mode and is_text:
            hint = (
                "[dim]Ctrl+S to save  •  Esc/cancel-edit to exit  •  "
                "m toggles markdown preview[/dim]"
            )
            yield Static(hint, classes="cv-toolbar")
            editor_language = self._detect_language(suffix)
            yield TextArea(
                text=self._edit_buffer or content or "",
                id="context-editor",
                classes="cv-editor",
                language=editor_language,
                show_line_numbers=True,
                highlight_cursor_line=True,
            )
            return

        if is_text:
            yield Static(
                "[dim]e to edit • Ctrl+S to save edits • m to toggle raw/preview[/dim]",
                classes="cv-toolbar",
            )

        # Handle different file types
        if suffix in IMAGE_EXTENSIONS:
            yield from self._compose_image_view()
        elif suffix in BINARY_EXTENSIONS:
            yield from self._compose_binary_view()
        elif is_text and content is not None:
            yield from self._compose_text_view(content, suffix)
        else:
            # Try to read as text, fall back to binary info
            if is_text and content is not None:
                yield from self._compose_text_view(content, suffix)
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
                f"[dim]Path:[/dim] {escape(str(self._file_path))}\n"
                f"[dim]Size:[/dim] {size}  [dim]Modified:[/dim] {modified}  "
                f"[dim]Type:[/dim] {escape(suffix)}"
            )

            yield Static(info_text, classes="cv-file-info")
        except Exception:
            yield Label(f"[dim]{self._file_path}[/dim]", classes="cv-info")

    def _compose_text_view(self, content: str, suffix: str) -> ComposeResult:
        """Compose text file content view."""
        if not self._file_path:
            return

        # Markdown rendering with toggleable preview/raw
        if suffix in {'.md', '.markdown'}:
            if self._markdown_preview:
                yield Markdown(content)
            else:
                syntax = Syntax(
                    content,
                    "markdown",
                    theme="monokai",
                    word_wrap=True,
                    line_numbers=True,
                )
                yield Static(syntax)
            return

        # JSON with pretty print option
        if suffix == ".json":
            try:
                parsed = json.loads(content)
                formatted = json.dumps(parsed, indent=2)
                syntax = Syntax(
                    formatted,
                    "json",
                    theme="monokai",
                    word_wrap=True,
                    line_numbers=True,
                )
                yield Static(syntax)
            except json.JSONDecodeError:
                syntax = Syntax(content, "json", theme="monokai", word_wrap=True)
                yield Static(syntax)
            return

        # Syntax highlighting for code
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

    def _compose_image_view(self) -> ComposeResult:
        """Compose image file info view."""
        if not self._file_path:
            return

        suffix = self._file_path.suffix.lower()

        yield Label(
            f"[cyan]Image File[/cyan]\n\n"
            f"Format: {suffix.upper()}\n"
            f"[dim]Preview not available in terminal.[/dim]\n\n"
            f"[dim]Use Ctrl+O to open with system viewer[/dim]",
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

    def _detect_language(self, suffix: str) -> str | None:
        """Map common suffixes to syntax languages for editor highlight."""
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".json": "json",
            ".md": "markdown",
            ".markdown": "markdown",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".toml": "toml",
            ".sql": "sql",
            ".sh": "bash",
        }
        return mapping.get(suffix, suffix.lstrip(".") or None)

    def enter_edit_mode(self) -> bool:
        """Enter inline edit mode for the current text file."""
        if not self._file_path or not can_edit_file(self._file_path):
            self.post_message(self.FileError(self._file_path, "File is not editable"))
            return False

        try:
            self._edit_buffer = self._file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            self.post_message(self.FileError(self._file_path, str(exc)))
            return False

        self._edit_mode = True
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_editor)
        return True

    def _focus_editor(self) -> None:
        """Focus the text editor after composing."""
        try:
            editor = self.query_one("#context-editor", TextArea)
            editor.focus()
        except Exception:
            pass

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Keep edit buffer in sync as user types."""
        if event.text_area.id == "context-editor":
            self._edit_buffer = event.text_area.text

    def save_edits(self) -> bool:
        """Persist inline edits to disk."""
        if not self._edit_mode or not self._file_path:
            return False

        try:
            editor = self.query_one("#context-editor", TextArea)
            self._edit_buffer = editor.text
            self._file_path.write_text(self._edit_buffer, encoding="utf-8")
            self._edit_mode = False
            self.refresh(recompose=True)
            self.post_message(self.FileSaved(self._file_path))
            return True
        except Exception as exc:
            self.post_message(self.FileError(self._file_path, str(exc)))
            return False

    def cancel_edit(self) -> None:
        """Exit edit mode without saving."""
        if self._edit_mode:
            self._edit_mode = False
            self.refresh(recompose=True)

    def toggle_markdown_preview(self) -> None:
        """Toggle raw/preview display for markdown files."""
        if not self._file_path or self._edit_mode:
            return
        suffix = self._file_path.suffix.lower()
        if suffix in {'.md', '.markdown'}:
            self._markdown_preview = not self._markdown_preview
            self.refresh(recompose=True)

    def rename_current(self, new_path: Path) -> bool:
        """Rename the currently viewed file."""
        if not self._file_path:
            return False

        old_path = self._file_path
        try:
            dest = rename_path(old_path, new_path)
            self._file_path = dest
            self._edit_mode = False
            self.refresh(recompose=True)
            self.post_message(self.FileRenamed(old_path, dest))
            return True
        except Exception as exc:
            self.post_message(self.FileError(old_path, str(exc)))
            return False

    def duplicate_current(self) -> bool:
        """Duplicate the currently viewed file and open the copy."""
        if not self._file_path:
            return False

        source_path = self._file_path
        try:
            new_path = duplicate_file(source_path)
            self._file_path = new_path
            self._edit_mode = False
            self.refresh(recompose=True)
            self.post_message(self.FileDuplicated(source_path, new_path))
            return True
        except Exception as exc:
            self.post_message(self.FileError(source_path, str(exc)))
            return False

    def open_external(self) -> bool:
        """Open the current file with the system viewer."""
        if not self._file_path:
            return False

        cmd = (
            ["open", str(self._file_path)]
            if sys.platform == "darwin"
            else ["xdg-open", str(self._file_path)]
        )
        try:
            subprocess.Popen(cmd)
            return True
        except Exception as exc:
            self.post_message(self.FileError(self._file_path, str(exc)))
            return False

    @property
    def is_editing(self) -> bool:
        """Whether the viewer is currently in edit mode."""
        return self._edit_mode

    @property
    def current_path(self) -> Path | None:
        """Expose the currently active file path."""
        return self._file_path

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
                        yield MountItem(mount)
                else:
                    yield Label("  [dim](empty)[/dim]")

    def set_project(self, project: ContextRoot | None) -> None:
        """Update the displayed project."""
        self._project = project
        self._file_path = None
        self._edit_mode = False
        self._edit_buffer = ""
        self._markdown_preview = True
        self.refresh(recompose=True)

    def set_file(self, path: Path) -> None:
        """Display a file."""
        self._project = None
        self._file_path = path
        self._edit_mode = False
        self._edit_buffer = ""
        self._markdown_preview = True
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
