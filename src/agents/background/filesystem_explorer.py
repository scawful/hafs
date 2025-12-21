"""Filesystem Explorer Agent - read-only filesystem analysis and reporting.

Scans local drives, network mounts, and remote systems to create comprehensive
filesystem inventories for consolidation planning. Completely read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.background.base import BackgroundAgent

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Information about a file."""

    path: str
    size_bytes: int
    modified: str
    extension: str
    hash_md5: Optional[str] = None


@dataclass
class DirectoryStats:
    """Statistics for a directory."""

    path: str
    total_files: int
    total_size_bytes: int
    file_count_by_ext: dict[str, int] = field(default_factory=dict)
    size_by_ext: dict[str, int] = field(default_factory=dict)
    largest_files: list[FileInfo] = field(default_factory=list)
    oldest_files: list[FileInfo] = field(default_factory=list)
    newest_files: list[FileInfo] = field(default_factory=list)


class FileSystemExplorerAgent(BackgroundAgent):
    """Filesystem explorer agent for read-only analysis.

    Scans directories, collects metadata, identifies patterns, and generates
    reports for consolidation planning. Never modifies files.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize filesystem explorer agent."""
        super().__init__(config_path, verbose)
        self.scan_paths = self.config.tasks.get("scan_paths", [])
        self.max_depth = self.config.tasks.get("max_depth", 5)
        self.compute_hashes = self.config.tasks.get("compute_hashes", False)
        self.hash_threshold_mb = self.config.tasks.get("hash_threshold_mb", 100)
        self.exclude_patterns = self.config.tasks.get(
            "exclude_patterns",
            [
                "*/.git/*",
                "*/.venv/*",
                "*/node_modules/*",
                "*/__pycache__/*",
                "*/build/*",
                "*/dist/*",
                "*/.cache/*",
            ],
        )

    def run(self) -> dict[str, Any]:
        """Execute filesystem exploration.

        Returns:
            Dictionary with exploration results
        """
        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "scan_paths": self.scan_paths,
            "directory_stats": [],
            "duplicate_candidates": [],
            "large_files": [],
            "extension_summary": {},
            "total_size_gb": 0.0,
            "total_files": 0,
        }

        # Track duplicates by hash
        hash_to_files: dict[str, list[FileInfo]] = defaultdict(list)

        # Scan each path
        for scan_path in self.scan_paths:
            path = Path(scan_path)
            if not path.exists():
                logger.warning(f"Path does not exist: {scan_path}")
                continue

            logger.info(f"Scanning: {scan_path}")

            dir_stats = self._scan_directory(path, hash_to_files)
            results["directory_stats"].append(self._stats_to_dict(dir_stats))

            # Update totals
            results["total_files"] += dir_stats.total_files
            results["total_size_gb"] += dir_stats.total_size_bytes / (1024**3)

            # Merge extension stats
            for ext, count in dir_stats.file_count_by_ext.items():
                if ext not in results["extension_summary"]:
                    results["extension_summary"][ext] = {"count": 0, "size_gb": 0.0}
                results["extension_summary"][ext]["count"] += count
                results["extension_summary"][ext]["size_gb"] += (
                    dir_stats.size_by_ext.get(ext, 0) / (1024**3)
                )

        # Identify duplicates
        for file_hash, files in hash_to_files.items():
            if len(files) > 1:
                duplicate_group = {
                    "hash": file_hash,
                    "count": len(files),
                    "size_bytes": files[0].size_bytes,
                    "size_mb": files[0].size_bytes / (1024**2),
                    "waste_mb": (files[0].size_bytes * (len(files) - 1)) / (1024**2),
                    "files": [f.path for f in files],
                }
                results["duplicate_candidates"].append(duplicate_group)

        # Sort duplicates by waste (most wasteful first)
        results["duplicate_candidates"].sort(
            key=lambda x: x["waste_mb"], reverse=True
        )

        # Save results
        self._save_output(results, "filesystem_inventory")

        # Generate summary report
        summary = self._generate_summary(results)
        self._save_output(summary, "filesystem_summary")

        return results

    def _scan_directory(
        self, path: Path, hash_tracker: dict[str, list[FileInfo]]
    ) -> DirectoryStats:
        """Scan a directory and collect statistics.

        Args:
            path: Directory to scan
            hash_tracker: Dictionary to track file hashes for duplicate detection

        Returns:
            DirectoryStats for this directory
        """
        stats = DirectoryStats(
            path=str(path),
            total_files=0,
            total_size_bytes=0,
        )

        all_files: list[FileInfo] = []

        try:
            for root, dirs, files in os.walk(path):
                # Filter out excluded patterns
                dirs[:] = [
                    d
                    for d in dirs
                    if not self._should_exclude(os.path.join(root, d))
                ]

                for filename in files:
                    file_path = os.path.join(root, filename)

                    # Skip excluded files
                    if self._should_exclude(file_path):
                        continue

                    try:
                        file_stat = os.stat(file_path)

                        # Skip very large files for safety (> 10 GB)
                        if file_stat.st_size > 10 * 1024**3:
                            logger.warning(f"Skipping very large file: {file_path}")
                            continue

                        ext = Path(filename).suffix.lower() or "no_extension"
                        modified = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

                        file_info = FileInfo(
                            path=file_path,
                            size_bytes=file_stat.st_size,
                            modified=modified,
                            extension=ext,
                        )

                        # Compute hash for duplicate detection
                        if self.compute_hashes:
                            size_mb = file_stat.st_size / (1024**2)
                            if size_mb <= self.hash_threshold_mb:
                                file_info.hash_md5 = self._compute_hash(file_path)
                                if file_info.hash_md5:
                                    hash_tracker[file_info.hash_md5].append(file_info)

                        all_files.append(file_info)

                        # Update stats
                        stats.total_files += 1
                        stats.total_size_bytes += file_stat.st_size
                        stats.file_count_by_ext[ext] = (
                            stats.file_count_by_ext.get(ext, 0) + 1
                        )
                        stats.size_by_ext[ext] = (
                            stats.size_by_ext.get(ext, 0) + file_stat.st_size
                        )

                    except (OSError, PermissionError) as e:
                        logger.debug(f"Cannot access file {file_path}: {e}")
                        continue

        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot access directory {path}: {e}")

        # Sort files for top-N lists
        all_files.sort(key=lambda f: f.size_bytes, reverse=True)
        stats.largest_files = all_files[:20]

        all_files.sort(key=lambda f: f.modified)
        stats.oldest_files = all_files[:10]
        stats.newest_files = all_files[-10:][::-1]

        return stats

    def _should_exclude(self, path: str) -> bool:
        """Check if path matches exclusion patterns.

        Args:
            path: Path to check

        Returns:
            True if should be excluded
        """
        import fnmatch

        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def _compute_hash(self, file_path: str) -> Optional[str]:
        """Compute MD5 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            MD5 hash hex string, or None on error
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.debug(f"Failed to hash {file_path}: {e}")
            return None

    def _stats_to_dict(self, stats: DirectoryStats) -> dict[str, Any]:
        """Convert DirectoryStats to dictionary.

        Args:
            stats: DirectoryStats object

        Returns:
            Dictionary representation
        """
        return {
            "path": stats.path,
            "total_files": stats.total_files,
            "total_size_gb": stats.total_size_bytes / (1024**3),
            "total_size_mb": stats.total_size_bytes / (1024**2),
            "file_count_by_ext": stats.file_count_by_ext,
            "size_by_ext_mb": {
                ext: size / (1024**2) for ext, size in stats.size_by_ext.items()
            },
            "largest_files": [
                {
                    "path": f.path,
                    "size_mb": f.size_bytes / (1024**2),
                    "extension": f.extension,
                }
                for f in stats.largest_files[:10]
            ],
        }

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary report.

        Args:
            results: Full exploration results

        Returns:
            Summary dictionary
        """
        # Calculate potential savings from duplicates
        total_waste_mb = sum(d["waste_mb"] for d in results["duplicate_candidates"])

        # Top extensions by count
        ext_by_count = sorted(
            results["extension_summary"].items(),
            key=lambda x: x[1]["count"],
            reverse=True,
        )[:10]

        # Top extensions by size
        ext_by_size = sorted(
            results["extension_summary"].items(),
            key=lambda x: x[1]["size_gb"],
            reverse=True,
        )[:10]

        return {
            "timestamp": results["scan_timestamp"],
            "total_files": results["total_files"],
            "total_size_gb": round(results["total_size_gb"], 2),
            "total_directories": len(results["directory_stats"]),
            "duplicate_groups": len(results["duplicate_candidates"]),
            "potential_savings_gb": round(total_waste_mb / 1024, 2),
            "top_extensions_by_count": [
                {"ext": ext, "count": data["count"]} for ext, data in ext_by_count
            ],
            "top_extensions_by_size": [
                {
                    "ext": ext,
                    "size_gb": round(data["size_gb"], 2),
                }
                for ext, data in ext_by_size
            ],
            "summary": f"Scanned {results['total_files']:,} files ({results['total_size_gb']:.1f} GB) "
            f"across {len(results['directory_stats'])} directories. "
            f"Found {len(results['duplicate_candidates'])} duplicate groups "
            f"with {total_waste_mb / 1024:.1f} GB potential savings.",
        }


def main():
    """CLI entry point for filesystem explorer agent."""
    parser = argparse.ArgumentParser(description="hafs Filesystem Explorer Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = FileSystemExplorerAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
