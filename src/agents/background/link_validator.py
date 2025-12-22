"""Link Validator Agent.

Validates internal and external links on websites.
Detects broken links, redirects, and connection issues.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from agents.background.base import BackgroundAgent
from agents.background.web_client import WebClient

logger = logging.getLogger(__name__)


class LinkValidatorAgent(BackgroundAgent):
    """Agent for validating links on websites.

    Crawls configured websites and checks all internal and external links
    for validity, reporting broken links and errors.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize link validator agent."""
        if config_path is None:
            config_path = Path("config/website_monitoring_agents.toml")

        super().__init__(config_path, verbose)

        self.websites = self.config.tasks.get("websites", [])
        self.check_internal_links = self.config.tasks.get("check_internal_links", True)
        self.check_external_links = self.config.tasks.get("check_external_links", True)
        self.external_timeout = self.config.tasks.get("external_timeout_seconds", 10)
        self.max_retries = self.config.tasks.get("max_retries", 2)
        self.ignore_patterns = self.config.tasks.get("ignore_patterns", [])

    def _default_config_path(self) -> Path:
        """Get default configuration path."""
        return Path("config/website_monitoring_agents.toml")

    def run(self) -> dict[str, Any]:
        """Execute link validation for all configured websites.

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating links on {len(self.websites)} websites")

        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "websites_scanned": len(self.websites),
            "total_links_checked": 0,
            "broken_links": 0,
            "broken_by_site": {},
            "summary": {},
        }

        total_links = 0
        total_broken = 0

        # Validate each website
        for site in self.websites:
            site_name = site.get("name", "unknown")
            logger.info(f"Validating {site_name}...")

            try:
                site_result = asyncio.run(self._validate_website(site))

                results["broken_by_site"][site_name] = site_result["broken_links"]
                total_links += site_result["total_links"]
                total_broken += len(site_result["broken_links"])

            except Exception as e:
                logger.error(f"Failed to validate {site_name}: {e}")
                results["broken_by_site"][site_name] = []

        results["total_links_checked"] = total_links
        results["broken_links"] = total_broken

        # Summary
        failure_rate = (total_broken / total_links * 100) if total_links > 0 else 0

        summary_text = (
            f"{total_links} links checked, {total_broken} broken "
            f"({failure_rate:.1f}% failure rate)"
        )
        results["summary"]["text"] = summary_text
        results["summary"]["failure_rate_percent"] = round(failure_rate, 2)

        # Save results
        self._save_output(results, "link_validation")

        logger.info(summary_text)

        return results

    async def _validate_website(self, site_config: dict) -> dict[str, Any]:
        """Validate all links on a website.

        Args:
            site_config: Website configuration

        Returns:
            Dictionary with validation results
        """
        site_name = site_config.get("name", "unknown")
        url = site_config.get("url")

        logger.info(f"Crawling {site_name} for links...")

        # Crawl to extract all links
        all_links = []

        async with WebClient(url, rate_limit=1.0) as client:
            # Crawl homepage and a few pages
            pages = await client.bfs_crawl(max_pages=50, max_depth=2)

            # Collect all unique links from pages
            seen_links = set()
            for page in pages:
                for link in page.links:
                    if link not in seen_links:
                        seen_links.add(link)
                        all_links.append(
                            {
                                "url": link,
                                "source_page": page.url,
                                "is_internal": client.is_same_domain(link),
                            }
                        )

        logger.info(f"Found {len(all_links)} unique links on {site_name}")

        # Filter links
        links_to_check = self._filter_links(all_links)

        logger.info(f"Checking {len(links_to_check)} links...")

        # Validate links
        broken_links = await self._validate_links(links_to_check)

        logger.info(f"Found {len(broken_links)} broken links on {site_name}")

        return {
            "site": site_name,
            "total_links": len(all_links),
            "checked_links": len(links_to_check),
            "broken_links": broken_links,
        }

    def _filter_links(self, links: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter links based on configuration.

        Args:
            links: List of link dictionaries

        Returns:
            Filtered list of links to check
        """
        filtered = []

        for link_info in links:
            url = link_info["url"]
            is_internal = link_info["is_internal"]

            # Check internal/external filters
            if is_internal and not self.check_internal_links:
                continue

            if not is_internal and not self.check_external_links:
                continue

            # Check ignore patterns
            should_ignore = False
            for pattern in self.ignore_patterns:
                if pattern.startswith("*"):
                    # Suffix match
                    if url.endswith(pattern[1:]):
                        should_ignore = True
                        break
                elif pattern.endswith("*"):
                    # Prefix match
                    if url.startswith(pattern[:-1]):
                        should_ignore = True
                        break
                else:
                    # Exact match
                    if url == pattern:
                        should_ignore = True
                        break

            if should_ignore:
                continue

            filtered.append(link_info)

        return filtered

    async def _validate_links(self, links: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Validate a list of links concurrently.

        Args:
            links: List of link dictionaries

        Returns:
            List of broken links with error details
        """
        broken_links = []

        # Validate in batches to avoid overwhelming the server
        batch_size = 10

        for i in range(0, len(links), batch_size):
            batch = links[i : i + batch_size]

            # Validate batch concurrently
            tasks = [self._validate_link(link_info) for link_info in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect broken links
            for link_info, result in zip(batch, results):
                if isinstance(result, dict) and result.get("broken"):
                    broken_links.append(
                        {
                            "source_page": link_info["source_page"],
                            "broken_link": link_info["url"],
                            "error": result.get("error", "Unknown error"),
                            "link_type": "internal" if link_info["is_internal"] else "external",
                        }
                    )

            # Small delay between batches
            await asyncio.sleep(0.5)

            # Progress logging
            if (i + batch_size) % 50 == 0:
                logger.info(f"Checked {min(i + batch_size, len(links))}/{len(links)} links...")

        return broken_links

    async def _validate_link(self, link_info: dict[str, Any]) -> dict[str, Any]:
        """Validate a single link.

        Args:
            link_info: Link information dictionary

        Returns:
            Validation result
        """
        url = link_info["url"]
        is_internal = link_info["is_internal"]

        # Use appropriate timeout
        timeout = 5 if is_internal else self.external_timeout

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, follow_redirects=True
                ) as client:
                    # Use HEAD request for efficiency
                    response = await client.head(url)

                    # Check status code
                    if 200 <= response.status_code < 400:
                        return {"broken": False}
                    else:
                        return {
                            "broken": True,
                            "error": f"{response.status_code} {response.reason_phrase}",
                        }

            except httpx.TimeoutException:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return {"broken": True, "error": "Timeout"}

            except httpx.ConnectError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return {"broken": True, "error": f"Connection error: {str(e)}"}

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return {"broken": True, "error": str(e)}

        return {"broken": True, "error": "Max retries exceeded"}


def main():
    """CLI entry point for link validator agent."""
    parser = argparse.ArgumentParser(description="hafs Link Validator Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = LinkValidatorAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
