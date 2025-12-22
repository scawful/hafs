"""Zelda Resource Discovery and Indexing System.

Scans the file system for Zelda-related resources (ASM, documentation, ROM hacks)
and builds a unified index for training data generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceFile:
    """A discovered resource file."""

    path: Path
    file_type: str  # asm, md, txt, inc, c, cpp, h
    size_bytes: int
    last_modified: str
    content_hash: str  # MD5 hash for deduplication
    source_dir: str  # Which root directory it came from
    relative_path: str  # Relative to source root
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "last_modified": self.last_modified,
            "content_hash": self.content_hash,
            "source_dir": self.source_dir,
            "relative_path": self.relative_path,
            "metadata": self.metadata,
        }


@dataclass
class IndexResult:
    """Result from resource indexing."""

    total_files: int
    by_type: dict[str, int]
    by_source: dict[str, int]
    files: list[ResourceFile]
    duplicates_found: int
    errors: list[str]
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "by_type": self.by_type,
            "by_source": self.by_source,
            "duplicates_found": self.duplicates_found,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "indexed_at": datetime.now().isoformat(),
        }


class ZeldaResourceIndexer:
    """Discovers and indexes all Zelda resources across the filesystem.

    Scans predefined roots for Zelda-related files (ASM, docs, C++),
    deduplicates by content hash, and builds a searchable index.
    """

    RESOURCE_ROOTS = [
        Path.home() / "Code" / "zelda3",
        Path.home() / "Code" / "Oracle-of-Secrets",
        Path.home() / "Code" / "alttp-hacker-workspace",
        Path.home() / "Code" / "book-of-mudora",
        Path.home() / "Code" / "hyrule-historian",
        Path.home() / "Code" / "docs" / "zelda",
        Path.home() / "Documents" / "Zelda",
    ]

    SEARCH_PATTERNS = [
        "**/*.asm",  # Assembly files
        "**/*.md",  # Markdown documentation
        "**/*.txt",  # Text files
        "**/*.inc",  # Include files (constants, macros)
        "**/*.s",  # Assembly files (alternative extension)
        "**/*.65s", # 65816 assembly files
        "**/*.65c", # 65816 assembly files
    ]

    # Patterns to exclude (build artifacts, dependencies, etc.)
    EXCLUDE_PATTERNS = [
        "**/node_modules/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/__pycache__/**",
        "**/venv/**",
        "**/.venv/**",
        "**/target/**",
    ]

    def __init__(self, index_path: Optional[Path] = None):
        """Initialize resource indexer.

        Args:
            index_path: Path to save index JSON (optional)
        """
        self.index_path = index_path or (
            Path.home() / ".context" / "training" / "resource_index.json"
        )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._files: list[ResourceFile] = []
        self._content_hashes: set[str] = set()  # For deduplication
        self._errors: list[str] = []

    def _should_exclude(self, path: Path) -> bool:
        """Check if path matches exclusion patterns."""
        path_str = str(path)
        for pattern in self.EXCLUDE_PATTERNS:
            # Simple glob matching
            if pattern.startswith("**/"):
                suffix = pattern[3:]
                if suffix in path_str:
                    return True
        return False

    def _get_file_type(self, path: Path) -> str:
        """Determine file type from extension."""
        suffix = path.suffix.lower()
        type_map = {
            ".asm": "asm",
            ".s": "asm",
            ".65s": "asm",
            ".65c": "asm",
            ".inc": "asm_include",
            ".md": "markdown",
            ".txt": "text",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "header",
        }
        return type_map.get(suffix, "unknown")

    def _compute_content_hash(self, path: Path) -> str:
        """Compute MD5 hash of file content."""
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to hash {path}: {e}")
            return ""

    def _index_file(self, path: Path, source_root: Path) -> Optional[ResourceFile]:
        """Index a single file.

        Args:
            path: File path to index
            source_root: Root directory path came from

        Returns:
            ResourceFile if indexed successfully, None otherwise
        """
        try:
            # Skip excluded paths
            if self._should_exclude(path):
                return None

            # Get file info
            stat = path.stat()
            file_type = self._get_file_type(path)

            # Skip unknown file types
            if file_type == "unknown":
                return None

            # Compute content hash for deduplication
            content_hash = self._compute_content_hash(path)
            if not content_hash:
                return None

            # Check for duplicates
            if content_hash in self._content_hashes:
                logger.debug(f"Duplicate file (hash): {path}")
                return None

            # Add to hash set
            self._content_hashes.add(content_hash)

            # Create resource file
            resource = ResourceFile(
                path=path,
                file_type=file_type,
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                content_hash=content_hash,
                source_dir=str(source_root),
                relative_path=str(path.relative_to(source_root)),
            )

            # Extract metadata based on file type
            if file_type in ("asm", "asm_include"):
                resource.metadata = self._extract_asm_metadata(path)
            elif file_type == "markdown":
                resource.metadata = self._extract_markdown_metadata(path)

            return resource

        except Exception as e:
            error_msg = f"Error indexing {path}: {e}"
            logger.error(error_msg)
            self._errors.append(error_msg)
            return None

    def _extract_asm_metadata(self, path: Path) -> dict[str, Any]:
        """Extract metadata from ASM file."""
        metadata = {
            "labels": [],
            "includes": [],
            "macros": [],
            "line_count": 0,
        }

        try:
            content = path.read_text(errors="replace")
            lines = content.split("\n")
            metadata["line_count"] = len(lines)

            for line in lines[:100]:  # First 100 lines for quick scan
                line = line.strip()

                # Extract labels (simplified - ends with :)
                if line and not line.startswith(";") and line.endswith(":"):
                    label = line[:-1].strip()
                    if label and not label.startswith("."):  # Skip local labels
                        metadata["labels"].append(label)

                # Extract includes
                if line.startswith("incsrc") or line.startswith("include"):
                    parts = line.split()
                    if len(parts) > 1:
                        metadata["includes"].append(parts[1])

                # Extract macro definitions
                if line.startswith("macro"):
                    parts = line.split()
                    if len(parts) > 1:
                        metadata["macros"].append(parts[1])

        except Exception as e:
            logger.warning(f"Failed to extract ASM metadata from {path}: {e}")

        return metadata

    def _extract_markdown_metadata(self, path: Path) -> dict[str, Any]:
        """Extract metadata from Markdown file."""
        metadata = {
            "title": "",
            "headings": [],
            "line_count": 0,
            "has_code_blocks": False,
        }

        try:
            content = path.read_text(errors="replace")
            lines = content.split("\n")
            metadata["line_count"] = len(lines)

            for line in lines[:50]:  # First 50 lines
                line = line.strip()

                # Extract title (first # heading)
                if line.startswith("# ") and not metadata["title"]:
                    metadata["title"] = line[2:].strip()

                # Extract headings
                if line.startswith("#"):
                    heading_level = len(line) - len(line.lstrip("#"))
                    heading_text = line.lstrip("#").strip()
                    metadata["headings"].append({
                        "level": heading_level,
                        "text": heading_text,
                    })

                # Check for code blocks
                if line.startswith("```"):
                    metadata["has_code_blocks"] = True

        except Exception as e:
            logger.warning(f"Failed to extract Markdown metadata from {path}: {e}")

        return metadata

    async def discover_and_index(self) -> IndexResult:
        """Scan all roots and build unified index.

        Returns:
            IndexResult with statistics and file list
        """
        start_time = datetime.now()
        logger.info("Starting resource discovery...")

        self._files = []
        self._content_hashes = set()
        self._errors = []
        duplicates_count = 0

        # Scan each root directory
        for root in self.RESOURCE_ROOTS:
            if not root.exists():
                logger.warning(f"Root directory not found: {root}")
                continue

            logger.info(f"Scanning: {root}")
            root_files = 0

            # Search for each pattern
            for pattern in self.SEARCH_PATTERNS:
                for path in root.rglob(pattern):
                    if not path.is_file():
                        continue

                    # Attempt to index
                    resource = self._index_file(path, root)
                    if resource:
                        self._files.append(resource)
                        root_files += 1
                    elif self._compute_content_hash(path) in self._content_hashes:
                        duplicates_count += 1

            logger.info(f"  Found {root_files} unique files in {root.name}")

        # Compute statistics
        by_type: dict[str, int] = {}
        by_source: dict[str, int] = {}

        for file in self._files:
            # Count by type
            by_type[file.file_type] = by_type.get(file.file_type, 0) + 1

            # Count by source
            source_name = Path(file.source_dir).name
            by_source[source_name] = by_source.get(source_name, 0) + 1

        duration = (datetime.now() - start_time).total_seconds()

        result = IndexResult(
            total_files=len(self._files),
            by_type=by_type,
            by_source=by_source,
            files=self._files,
            duplicates_found=duplicates_count,
            errors=self._errors,
            duration_seconds=duration,
        )

        logger.info(f"Discovery complete: {result.total_files} unique files in {duration:.1f}s")
        logger.info(f"  By type: {result.by_type}")
        logger.info(f"  By source: {result.by_source}")
        logger.info(f"  Duplicates skipped: {duplicates_count}")

        # Save index
        await self.save_index(result)

        return result

    async def save_index(self, result: IndexResult):
        """Save index to JSON file."""
        index_data = {
            "metadata": result.to_dict(),
            "files": [f.to_dict() for f in result.files],
        }

        with open(self.index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        logger.info(f"Index saved to: {self.index_path}")

    def load_index(self) -> Optional[IndexResult]:
        """Load index from JSON file."""
        if not self.index_path.exists():
            return None

        with open(self.index_path) as f:
            data = json.load(f)

        files = [
            ResourceFile(
                path=Path(f["path"]),
                file_type=f["file_type"],
                size_bytes=f["size_bytes"],
                last_modified=f["last_modified"],
                content_hash=f["content_hash"],
                source_dir=f["source_dir"],
                relative_path=f["relative_path"],
                metadata=f.get("metadata", {}),
            )
            for f in data["files"]
        ]

        metadata = data["metadata"]
        result = IndexResult(
            total_files=metadata["total_files"],
            by_type=metadata["by_type"],
            by_source=metadata["by_source"],
            files=files,
            duplicates_found=metadata["duplicates_found"],
            errors=metadata["errors"],
            duration_seconds=metadata["duration_seconds"],
        )

        return result

    def get_files_by_type(self, file_type: str) -> list[ResourceFile]:
        """Get all files of a specific type."""
        return [f for f in self._files if f.file_type == file_type]

    def get_files_by_source(self, source_name: str) -> list[ResourceFile]:
        """Get all files from a specific source directory."""
        return [f for f in self._files if Path(f.source_dir).name == source_name]
