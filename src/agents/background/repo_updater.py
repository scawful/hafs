"""Repo updater agent - monitors and updates git repositories."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.background.base import BackgroundAgent


class RepoUpdaterAgent(BackgroundAgent):
    """Repo updater agent for git repository monitoring.

    Checks for updates, monitors changes, and optionally auto-pulls
    latest commits from remote repository.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize repo updater agent."""
        super().__init__(config_path, verbose)
        self.local_repo = Path(self.config.tasks.get("local_repo", "C:/hafs"))
        self.remote_origin = self.config.tasks.get("remote_origin", "")
        self.auto_pull = self.config.tasks.get("auto_pull", False)

    def run(self) -> dict[str, Any]:
        """Execute repository update tasks.

        Returns:
            Dictionary with repository status
        """
        results = {
            "repo_path": str(self.local_repo),
            "is_git_repo": False,
            "branch": None,
            "has_changes": False,
            "behind_remote": False,
            "commits_behind": 0,
            "recent_commits": [],
            "auto_pull_enabled": self.auto_pull,
            "pull_attempted": False,
            "pull_success": None,
        }

        # Check if valid git repo
        if not (self.local_repo / ".git").exists():
            self.logger.warning(f"Not a git repository: {self.local_repo}")
            return results

        results["is_git_repo"] = True

        # Get current branch
        branch = self._get_current_branch()
        results["branch"] = branch
        self.logger.info(f"Current branch: {branch}")

        # Check for local changes
        has_changes = self._has_local_changes()
        results["has_changes"] = has_changes
        if has_changes:
            self.logger.info("Local changes detected")

        # Fetch from remote
        self._fetch_remote()

        # Check if behind remote
        behind_count = self._count_commits_behind()
        results["commits_behind"] = behind_count
        results["behind_remote"] = behind_count > 0

        if behind_count > 0:
            self.logger.info(f"Repository is {behind_count} commits behind remote")

            if self.auto_pull and not has_changes:
                self.logger.info("Auto-pull enabled - pulling latest changes")
                results["pull_attempted"] = True
                results["pull_success"] = self._pull_changes()
            else:
                if has_changes:
                    self.logger.info("Skipping auto-pull due to local changes")
                else:
                    self.logger.info("Auto-pull disabled - manual pull required")

        # Get recent commits
        recent_commits = self._get_recent_commits(limit=5)
        results["recent_commits"] = recent_commits

        # Save results
        self._save_output(results, "repo_status")

        return results

    def _get_current_branch(self) -> str | None:
        """Get current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.local_repo,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            self.logger.warning(f"Failed to get current branch: {e}")
        return None

    def _has_local_changes(self) -> bool:
        """Check if repository has uncommitted changes."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.local_repo,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return bool(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Failed to check local changes: {e}")
            return False

    def _fetch_remote(self) -> bool:
        """Fetch latest changes from remote."""
        try:
            result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=self.local_repo,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.warning(f"Failed to fetch from remote: {e}")
            return False

    def _count_commits_behind(self) -> int:
        """Count how many commits behind remote the local branch is."""
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD..@{u}"],
                cwd=self.local_repo,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception as e:
            self.logger.warning(f"Failed to count commits behind: {e}")
        return 0

    def _pull_changes(self) -> bool:
        """Pull latest changes from remote."""
        try:
            result = subprocess.run(
                ["git", "pull", "origin"],
                cwd=self.local_repo,
                capture_output=True,
                text=True,
                timeout=60,
            )
            success = result.returncode == 0
            if success:
                self.logger.info("Successfully pulled latest changes")
            else:
                self.logger.error(f"Pull failed: {result.stderr}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to pull changes: {e}")
            return False

    def _get_recent_commits(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent commit history."""
        try:
            result = subprocess.run(
                ["git", "log", f"-{limit}", "--pretty=format:%h|%an|%s|%ad", "--date=short"],
                cwd=self.local_repo,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 3)
                if len(parts) == 4:
                    commits.append({
                        "hash": parts[0],
                        "author": parts[1],
                        "message": parts[2],
                        "date": parts[3],
                    })

            return commits

        except Exception as e:
            self.logger.warning(f"Failed to get recent commits: {e}")
            return []


def main():
    """CLI entry point for repo updater agent."""
    parser = argparse.ArgumentParser(description="hafs Repo Updater Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    agent = RepoUpdaterAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
