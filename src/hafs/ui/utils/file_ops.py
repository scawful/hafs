"""Small utilities for file editing/duplication used by TUI widgets."""

from __future__ import annotations

import shutil
from pathlib import Path

# Conservative list of extensions we treat as text-editable.
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".org",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".mjs",
    ".cjs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cc",
    ".cxx",
    ".java",
    ".kt",
    ".kts",
    ".scala",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".lua",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".html",
    ".htm",
    ".xml",
    ".xhtml",
    ".svg",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".graphql",
    ".gql",
    ".r",
    ".R",
    ".jl",
    ".m",
    ".swift",
    ".vim",
    ".el",
    ".lisp",
    ".clj",
    ".hs",
    ".makefile",
    ".cmake",
    ".gradle",
    ".dockerfile",
    ".containerfile",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
}


def is_probably_text(path: Path) -> bool:
    """Heuristic to see if file can be treated as text."""
    if not path.exists() or not path.is_file():
        return False

    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return False

    if b"\x00" in chunk:
        return False

    try:
        chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False

    return True


def can_edit_file(path: Path) -> bool:
    """Return True if a file looks editable in a text area."""
    if not path.exists() or not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return True

    return is_probably_text(path)


def read_text_file(path: Path, limit: int = 100_000) -> str:
    """Read a text file safely with a size cap."""
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) > limit:
        return content[:limit] + "\n\n... [truncated]"
    return content


def next_copy_path(path: Path) -> Path:
    """Return a non-conflicting copy path like foo_copy2.ext."""
    base = path.stem
    suffix = path.suffix
    parent = path.parent

    candidate = parent / f"{base}_copy{suffix}"
    if not candidate.exists():
        return candidate

    for idx in range(2, 100):
        candidate = parent / f"{base}_copy{idx}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError("Unable to find a free copy name")


def duplicate_file(path: Path) -> Path:
    """Duplicate a file and return the new path."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(path)

    dest = next_copy_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    return dest


def rename_path(src: Path, dest: Path) -> Path:
    """Rename a path, creating parents for the destination."""
    if not src.exists():
        raise FileNotFoundError(src)

    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)
    return dest
