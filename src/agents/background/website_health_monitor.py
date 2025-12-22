"""Website Health Monitor Agent.

Monitors uptime and response times for configured websites.
Runs frequent health checks and alerts on downtime or degraded performance.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.background.base import BackgroundAgent
from agents.background.web_client import HealthStatus, WebClient

logger = logging.getLogger(__name__)


class WebsiteHealthMonitorAgent(BackgroundAgent):
    """Agent for monitoring website health and uptime.

    Performs periodic health checks on configured websites and tracks:
    - Response times
    - Status codes
    - SSL certificate validity
    - Downtime detection
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize health monitor agent."""
        # Override default config path for this agent
        if config_path is None:
            config_path = Path("config/website_monitoring_agents.toml")

        super().__init__(config_path, verbose)

        self.websites = self.config.tasks.get("websites", [])
        self.alert_on_downtime = self.config.tasks.get("alert_on_downtime", True)
        self.alert_on_slow_response = self.config.tasks.get("alert_on_slow_response", True)

        # Load historical data for downtime detection
        self.history_file = (
            Path(self.config.tasks.get("output_dir", "~/.context/monitoring/health"))
            .expanduser()
            / "history.json"
        )
        self.history = self._load_history()

    def _default_config_path(self) -> Path:
        """Get default configuration path."""
        return Path("config/website_monitoring_agents.toml")

    def _load_history(self) -> dict[str, list[dict]]:
        """Load historical health check data.

        Returns:
            Dictionary mapping site names to health check history
        """
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

        return {}

    def _save_history(self):
        """Save health check history to disk."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def _update_history(self, site_name: str, health: HealthStatus, max_history: int = 100):
        """Update health check history for a site.

        Args:
            site_name: Website name
            health: Health status result
            max_history: Maximum number of history entries to keep
        """
        if site_name not in self.history:
            self.history[site_name] = []

        # Add new entry
        self.history[site_name].append(
            {
                "timestamp": health.timestamp,
                "status": health.status,
                "status_code": health.status_code,
                "response_time_ms": health.response_time_ms,
                "error": health.error,
            }
        )

        # Trim to max history
        if len(self.history[site_name]) > max_history:
            self.history[site_name] = self.history[site_name][-max_history:]

    def _detect_downtime(self, site_name: str, consecutive_failures: int = 3) -> bool:
        """Detect if site is down based on recent history.

        Args:
            site_name: Website name
            consecutive_failures: Number of consecutive failures to trigger alert

        Returns:
            True if site is considered down
        """
        if site_name not in self.history:
            return False

        history = self.history[site_name]

        if len(history) < consecutive_failures:
            return False

        # Check last N entries
        recent = history[-consecutive_failures:]
        return all(entry["status"] == "offline" for entry in recent)

    def _send_alert(self, site_name: str, issue: str):
        """Send alert notification.

        Args:
            site_name: Website name
            issue: Issue description
        """
        # TODO: Implement email/Discord/Slack notifications
        logger.warning(f"ALERT [{site_name}]: {issue}")

        # For now, just log the alert
        alert_file = (
            Path(self.config.tasks.get("report_dir", "~/.context/logs/health_monitor"))
            .expanduser()
            / "alerts.log"
        )
        alert_file.parent.mkdir(parents=True, exist_ok=True)

        with open(alert_file, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} [{site_name}] {issue}\n")

    def run(self) -> dict[str, Any]:
        """Execute health checks for all configured websites.

        Returns:
            Dictionary with health check results
        """
        logger.info(f"Checking health of {len(self.websites)} websites")

        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "websites_checked": len(self.websites),
            "websites": [],
            "alerts": [],
            "summary": {},
        }

        # Run health checks
        health_statuses = asyncio.run(self._check_all_websites())

        # Process results
        online_count = 0
        offline_count = 0
        degraded_count = 0
        total_response_time = 0
        response_count = 0

        for health in health_statuses:
            site_name = health.name

            # Convert to dict for JSON serialization
            health_dict = {
                "name": health.name,
                "url": health.url,
                "status": health.status,
                "status_code": health.status_code,
                "response_time_ms": health.response_time_ms,
                "ssl_valid": health.ssl_valid,
                "ssl_expires": health.ssl_expires,
                "error": health.error,
            }

            results["websites"].append(health_dict)

            # Update history
            self._update_history(site_name, health)

            # Count statuses
            if health.status == "online":
                online_count += 1
                if health.response_time_ms:
                    total_response_time += health.response_time_ms
                    response_count += 1
            elif health.status == "offline":
                offline_count += 1

                # Check for downtime alert
                if self.alert_on_downtime and self._detect_downtime(site_name):
                    alert_msg = "Site is down (3+ consecutive failures)"
                    results["alerts"].append({"site": site_name, "issue": alert_msg})
                    self._send_alert(site_name, alert_msg)

            elif health.status == "degraded":
                degraded_count += 1

                # Check for slow response alert
                if self.alert_on_slow_response and health.response_time_ms:
                    threshold = next(
                        (
                            site.get("alert_threshold_ms", 5000)
                            for site in self.websites
                            if site.get("name") == site_name
                        ),
                        5000,
                    )
                    alert_msg = (
                        f"Slow response: {health.response_time_ms}ms "
                        f"(threshold: {threshold}ms)"
                    )
                    results["alerts"].append({"site": site_name, "issue": alert_msg})
                    self._send_alert(site_name, alert_msg)

        # Calculate summary
        avg_response_time = (
            int(total_response_time / response_count) if response_count > 0 else 0
        )

        results["summary"] = {
            "total_websites": len(self.websites),
            "online": online_count,
            "offline": offline_count,
            "degraded": degraded_count,
            "avg_response_time_ms": avg_response_time,
            "alerts_triggered": len(results["alerts"]),
        }

        summary_text = (
            f"{online_count} online, {offline_count} offline, "
            f"{degraded_count} degraded. "
            f"Avg response: {avg_response_time}ms"
        )
        results["summary"]["text"] = summary_text

        # Save results
        self._save_output(results, "health_check")

        # Save updated history
        self._save_history()

        logger.info(summary_text)

        if results["alerts"]:
            logger.warning(f"{len(results['alerts'])} alerts triggered")

        return results

    async def _check_all_websites(self) -> list[HealthStatus]:
        """Check health of all configured websites concurrently.

        Returns:
            List of HealthStatus objects
        """
        tasks = []

        for site in self.websites:
            task = self._check_website(site)
            tasks.append(task)

        return await asyncio.gather(*tasks)

    async def _check_website(self, site_config: dict) -> HealthStatus:
        """Check health of a single website.

        Args:
            site_config: Website configuration

        Returns:
            HealthStatus object
        """
        name = site_config.get("name", "unknown")
        url = site_config.get("url")
        alert_threshold_ms = site_config.get("alert_threshold_ms", 5000)

        logger.info(f"Checking {name}: {url}")

        async with WebClient(url) as client:
            health = await client.check_health(alert_threshold_ms=alert_threshold_ms)

        logger.info(f"{name}: {health.status} ({health.response_time_ms}ms)")

        return health


def main():
    """CLI entry point for website health monitor agent."""
    parser = argparse.ArgumentParser(description="hafs Website Health Monitor Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = WebsiteHealthMonitorAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
