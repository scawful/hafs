"""Explorer agent - scans codebase and catalogs changes."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.background.base import BackgroundAgent

logger = logging.getLogger(__name__)


class ExplorerAgent(BackgroundAgent):
    """Explorer agent for codebase analysis.

    Scans directories, tracks file changes, analyzes dependencies,
    and generates reports on code structure.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize explorer agent."""
        super().__init__(config_path, verbose)
        self.scan_dirs = self.config.tasks.get("scan_directories", [])

    def run(self) -> dict[str, Any]:
        """Execute exploration tasks.

        Returns:
            Dictionary with exploration results
        """
        results = {
            "scanned_directories": [],
            "file_counts": {},
            "total_files": 0,
            "recent_changes": [],
            "dependencies": {},
        }

        for scan_dir in self.scan_dirs:
            dir_path = Path(scan_dir)
            if not dir_path.exists():
                logger.warning(f"Directory not found: {scan_dir}")
                continue

            logger.info(f"Scanning directory: {scan_dir}")
            dir_results = self._scan_directory(dir_path)
            results["scanned_directories"].append(scan_dir)
            results["file_counts"][scan_dir] = dir_results["file_count"]
            results["total_files"] += dir_results["file_count"]

            # Track recent changes if git repo
            if (dir_path / ".git").exists():
                changes = self._get_recent_changes(dir_path)
                results["recent_changes"].extend(changes)

            # Analyze dependencies if Python project
            if (dir_path / "pyproject.toml").exists() or (dir_path / "requirements.txt").exists():
                deps = self._analyze_dependencies(dir_path)
                results["dependencies"][scan_dir] = deps

        # Save results
        self._save_output(results, "exploration_report")

        # Generate summary
        summary = self._generate_summary(results)
        self._save_output(summary, "exploration_summary")

        return results

    def _scan_directory(self, path: Path) -> dict[str, Any]:
        """Scan a directory and collect file statistics.

        Args:
            path: Directory to scan

        Returns:
            Dictionary with scan results
        """
        file_count = 0
        file_types = {}

        for file_path in path.rglob("*"):
            if file_path.is_file():
                # Skip certain directories
                if any(
                    part in str(file_path)
                    for part in [".venv", "__pycache__", ".git", "node_modules", "build"]
                ):
                    continue

                file_count += 1
                suffix = file_path.suffix or "no_extension"
                file_types[suffix] = file_types.get(suffix, 0) + 1

        return {
            "path": str(path),
            "file_count": file_count,
            "file_types": file_types,
            "scan_time": datetime.now().isoformat(),
        }

    def _get_recent_changes(self, repo_path: Path, days: int = 7) -> list[dict[str, Any]]:
        """Get recent git changes.

        Args:
            repo_path: Path to git repository
            days: Number of days to look back

        Returns:
            List of recent changes
        """
        try:
            # Get recent commits
            result = subprocess.run(
                ["git", "log", f"--since={days}.days.ago", "--pretty=format:%h|%an|%s|%ad", "--date=short"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            changes = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    changes.append({
                        "commit": parts[0],
                        "author": parts[1],
                        "message": parts[2],
                        "date": parts[3],
                        "repo": str(repo_path),
                    })

            return changes

        except Exception as e:
            logger.warning(f"Failed to get git changes for {repo_path}: {e}")
            return []

    def _analyze_dependencies(self, project_path: Path) -> dict[str, Any]:
        """Analyze Python project dependencies.

        Args:
            project_path: Path to Python project

        Returns:
            Dictionary with dependency info
        """
        deps = {
            "has_pyproject": (project_path / "pyproject.toml").exists(),
            "has_requirements": (project_path / "requirements.txt").exists(),
            "packages": [],
        }

        # Parse requirements.txt if exists
        req_file = project_path / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Extract package name (before version specifiers)
                            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
                            deps["packages"].append(pkg_name)
            except Exception as e:
                logger.warning(f"Failed to parse requirements.txt: {e}")

        return deps

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary report.

        Args:
            results: Full exploration results

        Returns:
            Summary dictionary
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "directories_scanned": len(results["scanned_directories"]),
            "total_files": results["total_files"],
            "recent_commits": len(results["recent_changes"]),
            "projects_with_deps": len(results["dependencies"]),
            "summary": f"Scanned {results['total_files']} files across {len(results['scanned_directories'])} directories",
        }


def main():
    """CLI entry point for explorer agent."""
    parser = argparse.ArgumentParser(description="hafs Explorer Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    agent = ExplorerAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
