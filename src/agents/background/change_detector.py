"""Change Detector Agent.

Detects content changes on websites and git repositories.
Monitors RSS feeds, tracks page updates, and logs git commits.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import feedparser
import httpx

from agents.background.base import BackgroundAgent
from agents.background.ssh_utils import SSHClient
from agents.background.web_client import WebClient

logger = logging.getLogger(__name__)


class ChangeDetectorAgent(BackgroundAgent):
    """Agent for detecting changes across websites and repositories.

    Monitors:
    - Website content changes (page hash comparison)
    - RSS feeds for new posts
    - Git repository commits via SSH
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize change detector agent."""
        if config_path is None:
            config_path = Path("config/website_monitoring_agents.toml")

        super().__init__(config_path, verbose)

        self.websites = self.config.tasks.get("websites", [])
        self.git_repos = self.config.tasks.get("git_repos", [])
        self.ssh_host = self.config.tasks.get("ssh_host", "scawful@halext-server")
        self.ssh_key = self.config.tasks.get("ssh_key", "~/.ssh/id_rsa")

        # AI summarization
        self.use_ai_summarization = self.config.tasks.get("use_ai_summarization", True)
        self.ai_model = self.config.tasks.get("ai_model", "qwen2.5:7b")
        self.ollama_url = self.config.tasks.get(
            "ollama_url", "http://100.104.53.21:11434"
        )

        # History for change detection
        self.history_file = (
            Path(self.config.tasks.get("output_dir", "~/.context/monitoring/changes"))
            .expanduser()
            / "history.json"
        )
        self.history = self._load_history()

    def _default_config_path(self) -> Path:
        """Get default configuration path."""
        return Path("config/website_monitoring_agents.toml")

    def _load_history(self) -> dict[str, Any]:
        """Load historical data for change detection.

        Returns:
            Dictionary with historical snapshots
        """
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

        return {"pages": {}, "rss_feeds": {}}

    def _save_history(self):
        """Save change detection history to disk."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def run(self) -> dict[str, Any]:
        """Execute change detection for websites and repositories.

        Returns:
            Dictionary with detected changes
        """
        logger.info("Running change detection")

        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "changes_detected": 0,
            "new_posts": [],
            "git_commits": [],
            "content_changes": [],
        }

        # Check website changes
        for site in self.websites:
            site_name = site.get("name", "unknown")
            logger.info(f"Checking {site_name} for changes...")

            try:
                # Check RSS feed if configured
                if "rss" in site:
                    new_posts = asyncio.run(self._check_rss_feed(site))
                    results["new_posts"].extend(new_posts)
                    results["changes_detected"] += len(new_posts)

                # Check page content changes
                content_changes = asyncio.run(self._check_website_changes(site))
                results["content_changes"].extend(content_changes)
                results["changes_detected"] += len(content_changes)

            except Exception as e:
                logger.error(f"Failed to check {site_name}: {e}")

        # Check git repositories
        for repo in self.git_repos:
            repo_name = repo.get("name", "unknown")
            logger.info(f"Checking {repo_name} for commits...")

            try:
                commits = asyncio.run(self._check_git_changes(repo))
                if commits:
                    results["git_commits"].append(
                        {
                            "repo": repo_name,
                            "commits": len(commits),
                            "details": commits,
                        }
                    )
                    results["changes_detected"] += len(commits)

            except Exception as e:
                logger.error(f"Failed to check {repo_name}: {e}")

        # Save history
        self._save_history()

        # AI summarization
        if self.use_ai_summarization and results["changes_detected"] > 0:
            summary = asyncio.run(self._summarize_changes(results))
            results["ai_summary"] = summary

        # Save results
        self._save_output(results, "change_detection")

        summary_text = (
            f"Detected {results['changes_detected']} changes: "
            f"{len(results['new_posts'])} new posts, "
            f"{len(results['content_changes'])} content changes, "
            f"{sum(c['commits'] for c in results['git_commits'])} commits"
        )
        results["summary"] = summary_text

        logger.info(summary_text)

        return results

    async def _check_rss_feed(self, site: dict) -> list[dict[str, Any]]:
        """Check RSS feed for new posts.

        Args:
            site: Site configuration

        Returns:
            List of new posts
        """
        site_name = site.get("name", "unknown")
        rss_url = site.get("rss")

        if not rss_url:
            return []

        logger.info(f"Checking RSS feed: {rss_url}")

        try:
            # Fetch RSS feed
            async with httpx.AsyncClient() as client:
                response = await client.get(rss_url, timeout=10)

            if response.status_code != 200:
                logger.warning(f"RSS feed returned {response.status_code}")
                return []

            # Parse feed
            feed = feedparser.parse(response.text)

            # Load previous post IDs
            if "rss_feeds" not in self.history:
                self.history["rss_feeds"] = {}

            previous_posts = self.history["rss_feeds"].get(site_name, [])

            # Find new posts
            new_posts = []
            current_posts = []

            for entry in feed.entries:
                post_id = entry.get("id", entry.get("link"))
                current_posts.append(post_id)

                if post_id not in previous_posts:
                    new_posts.append(
                        {
                            "site": site_name,
                            "title": entry.get("title", "Untitled"),
                            "url": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "summary": entry.get("summary", "")[:200],
                        }
                    )

            # Update history
            self.history["rss_feeds"][site_name] = current_posts

            if new_posts:
                logger.info(f"Found {len(new_posts)} new posts on {site_name}")

            return new_posts

        except Exception as e:
            logger.error(f"Failed to check RSS feed {rss_url}: {e}")
            return []

    async def _check_website_changes(self, site: dict) -> list[dict[str, Any]]:
        """Check for content changes on website.

        Args:
            site: Site configuration

        Returns:
            List of detected changes
        """
        site_name = site.get("name", "unknown")
        url = site.get("url")

        changes = []

        try:
            # Fetch current homepage
            async with WebClient(url) as client:
                response = await client.get("/")
                current_hash = hashlib.md5(response.text.encode()).hexdigest()

            # Load previous hash
            if "pages" not in self.history:
                self.history["pages"] = {}

            previous_hash = self.history["pages"].get(url)

            # Check for changes
            if previous_hash and previous_hash != current_hash:
                changes.append(
                    {
                        "site": site_name,
                        "page": url,
                        "change_type": "updated",
                        "summary": "Homepage content changed",
                    }
                )
                logger.info(f"Content changed on {site_name}")

            # Update history
            self.history["pages"][url] = current_hash

        except Exception as e:
            logger.error(f"Failed to check {url}: {e}")

        return changes

    async def _check_git_changes(self, repo: dict) -> list[dict[str, str]]:
        """Check git repository for recent commits.

        Args:
            repo: Repository configuration

        Returns:
            List of commits
        """
        repo_name = repo.get("name", "unknown")
        repo_path = repo.get("path")

        if not repo_path:
            return []

        try:
            # Create SSH client
            ssh_client = SSHClient(
                host=self.ssh_host,
                key_path=Path(self.ssh_key).expanduser(),
            )

            # Test connection
            if not ssh_client.test_connection():
                logger.warning(f"Cannot connect to {self.ssh_host}")
                return []

            # Get git log (last 24 hours)
            commits = ssh_client.git_log(repo_path, since="24 hours ago")

            if commits:
                logger.info(f"Found {len(commits)} commits in {repo_name}")

            return commits

        except Exception as e:
            logger.error(f"Failed to check git repo {repo_name}: {e}")
            return []

    async def _summarize_changes(self, results: dict[str, Any]) -> str:
        """Use AI to summarize detected changes.

        Args:
            results: Change detection results

        Returns:
            AI-generated summary
        """
        # Build summary prompt
        prompt_parts = ["Summarize the following changes:\n"]

        if results["new_posts"]:
            prompt_parts.append(f"\nNew blog posts ({len(results['new_posts'])}):")
            for post in results["new_posts"][:5]:  # Limit to 5
                prompt_parts.append(f"- {post['title']}")

        if results["content_changes"]:
            prompt_parts.append(f"\nContent changes ({len(results['content_changes'])}):")
            for change in results["content_changes"][:5]:
                prompt_parts.append(f"- {change['site']}: {change['summary']}")

        if results["git_commits"]:
            prompt_parts.append(f"\nGit commits:")
            for repo_commits in results["git_commits"]:
                repo_name = repo_commits["repo"]
                commits = repo_commits["details"][:5]  # Limit to 5
                prompt_parts.append(f"- {repo_name} ({repo_commits['commits']} commits):")
                for commit in commits:
                    prompt_parts.append(f"  * {commit['subject']}")

        prompt = "\n".join(prompt_parts)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ai_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    summary = data.get("response", "")
                    logger.info("Generated AI summary")
                    return summary
                else:
                    logger.warning(f"AI summarization failed: {response.status_code}")
                    return "Summary generation failed"

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Summary generation failed"


def main():
    """CLI entry point for change detector agent."""
    parser = argparse.ArgumentParser(description="hafs Change Detector Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = ChangeDetectorAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
