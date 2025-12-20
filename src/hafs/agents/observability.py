"""Distributed Observability Agent.

Monitors distributed system health including:
- halext-org backend health
- Tailscale compute nodes
- Web endpoints (halext.org, justinscofield.com, etc.)
- Local services

Provides anomaly detection, alerting, and remediation suggestions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

# Lazy load aiohttp
_aiohttp = None

def _ensure_aiohttp():
    """Ensure aiohttp is available."""
    global _aiohttp
    if _aiohttp is None:
        try:
            import aiohttp
            _aiohttp = aiohttp
        except ImportError:
            raise ImportError("aiohttp not installed")
    return _aiohttp

from hafs.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Result of a health check."""

    endpoint: str
    name: str
    status: HealthStatus
    latency_ms: int
    timestamp: str
    status_code: Optional[int] = None
    message: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""

    severity: AlertSeverity
    source: str
    message: str
    timestamp: str
    details: dict = field(default_factory=dict)
    acknowledged: bool = False
    remediation: Optional[str] = None


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    timestamp: str
    labels: dict = field(default_factory=dict)


@dataclass
class Anomaly:
    """Detected anomaly in metrics."""

    metric: str
    expected_range: tuple
    actual_value: float
    deviation_pct: float
    timestamp: str
    description: str


class DistributedObservabilityAgent(BaseAgent):
    """Agent for monitoring distributed system health.

    Monitors:
    1. Web endpoints (health checks, latency, SSL)
    2. Tailscale nodes (Ollama instances)
    3. halext-org backend services
    4. Local services

    Features:
    - Configurable endpoint monitoring
    - Anomaly detection
    - Alert management
    - Remediation suggestions
    - Metric logging

    Example:
        agent = DistributedObservabilityAgent()
        await agent.setup()

        # Check all endpoints
        results = await agent.check_all_endpoints()

        # Get alerts
        alerts = agent.get_active_alerts()

        # Analyze recent issues
        analysis = await agent.analyze_issues()
    """

    # Default endpoints to monitor
    DEFAULT_ENDPOINTS = [
        {"name": "halext-org", "url": "https://halext.org/api/health", "type": "api"},
        {"name": "halext-org-main", "url": "https://halext.org", "type": "web"},
        {"name": "justinscofield", "url": "https://justinscofield.com", "type": "web"},
        {"name": "zeniea", "url": "https://zeniea.com", "type": "web"},
    ]

    # Thresholds
    LATENCY_WARNING_MS = 2000
    LATENCY_ERROR_MS = 5000
    ERROR_RATE_THRESHOLD = 0.1  # 10%
    SYNC_STALE_HOURS = 24

    def __init__(
        self,
        endpoints: Optional[list[dict]] = None,
        check_interval: int = 60,
    ):
        super().__init__(
            "DistributedObservability",
            "Monitor distributed system health and detect anomalies."
        )

        self.endpoints = endpoints or self.DEFAULT_ENDPOINTS
        self.check_interval = check_interval

        # State
        self._session: Optional[Any] = None
        self._alerts: list[Alert] = []
        self._metrics: list[MetricPoint] = []
        self._health_history: dict[str, list[HealthCheck]] = {}
        self._running = False

        # Storage paths
        self.metrics_dir = self.context_root / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.alerts_file = self.metrics_dir / "alerts.json"
        self.sync_status_file = self.metrics_dir / "afs_sync_status.json"

        # Use fast tier for analysis
        self.model_tier = "fast"

    async def setup(self):
        """Initialize the observability agent."""
        await super().setup()
        await self._ensure_session()
        self._load_alerts()
        logger.info(f"DistributedObservability initialized with {len(self.endpoints)} endpoints")

    async def _ensure_session(self):
        """Ensure HTTP session is available."""
        if self._session is None or self._session.closed:
            aio = _ensure_aiohttp()
            timeout = aio.ClientTimeout(total=10)
            self._session = aio.ClientSession(timeout=timeout)

    def _load_alerts(self):
        """Load saved alerts from disk."""
        if self.alerts_file.exists():
            try:
                data = json.loads(self.alerts_file.read_text())
                self._alerts = [
                    Alert(
                        severity=AlertSeverity(a["severity"]),
                        source=a["source"],
                        message=a["message"],
                        timestamp=a["timestamp"],
                        details=a.get("details", {}),
                        acknowledged=a.get("acknowledged", False),
                        remediation=a.get("remediation"),
                    )
                    for a in data
                ]
            except Exception as e:
                logger.warning(f"Failed to load alerts: {e}")

    def _save_alerts(self):
        """Save alerts to disk."""
        try:
            data = [
                {
                    "severity": a.severity.value,
                    "source": a.source,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "details": a.details,
                    "acknowledged": a.acknowledged,
                    "remediation": a.remediation,
                }
                for a in self._alerts[-100:]  # Keep last 100
            ]
            self.alerts_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")

    async def check_endpoint(self, endpoint: dict) -> HealthCheck:
        """Check health of a single endpoint.

        Args:
            endpoint: Endpoint config with name, url, type.

        Returns:
            Health check result.
        """
        await self._ensure_session()

        name = endpoint["name"]
        url = endpoint["url"]
        start_time = time.time()

        try:
            async with self._session.get(url) as resp:
                latency = int((time.time() - start_time) * 1000)
                status_code = resp.status

                if status_code == 200:
                    status = HealthStatus.HEALTHY
                    message = "OK"
                elif status_code < 500:
                    status = HealthStatus.DEGRADED
                    message = f"HTTP {status_code}"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"HTTP {status_code}"

                # Check latency thresholds
                if latency > self.LATENCY_ERROR_MS:
                    status = HealthStatus.UNHEALTHY
                    message = f"High latency: {latency}ms"
                elif latency > self.LATENCY_WARNING_MS:
                    status = HealthStatus.DEGRADED
                    message = f"Elevated latency: {latency}ms"

                # Try to get response body for health endpoints
                details = {}
                if endpoint.get("type") == "api":
                    try:
                        details = await resp.json()
                    except:
                        pass

                return HealthCheck(
                    endpoint=url,
                    name=name,
                    status=status,
                    latency_ms=latency,
                    timestamp=datetime.now().isoformat(),
                    status_code=status_code,
                    message=message,
                    details=details,
                )

        except asyncio.TimeoutError:
            return HealthCheck(
                endpoint=url,
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=10000,
                timestamp=datetime.now().isoformat(),
                message="Timeout",
            )
        except Exception as e:
            # Check for ClientConnectorError dynamically
            if "ClientConnectorError" in type(e).__name__:
                return HealthCheck(
                    endpoint=url,
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    latency_ms=0,
                    timestamp=datetime.now().isoformat(),
                    message=f"Connection failed: {e}",
                )
            # Re-raise explicit ImportError if aiohttp is missing (checking _ensure_aiohttp call logic)
            # Actually, if check_endpoint is called, _ensure_session was called, so aiohttp SHOULD exist.
            # But we must catch generic Exception to fallback.
            return HealthCheck(
                endpoint=url,
                name=name,
                status=HealthStatus.UNKNOWN,
                latency_ms=0,
                timestamp=datetime.now().isoformat(),
                message=f"Error: {e}",
            )

    async def check_all_endpoints(self) -> dict[str, HealthCheck]:
        """Check health of all configured endpoints.

        Returns:
            Dict mapping endpoint names to health check results.
        """
        tasks = [self.check_endpoint(ep) for ep in self.endpoints]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        health_map = {}
        for i, result in enumerate(results):
            name = self.endpoints[i]["name"]

            if isinstance(result, Exception):
                health_map[name] = HealthCheck(
                    endpoint=self.endpoints[i]["url"],
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    latency_ms=0,
                    timestamp=datetime.now().isoformat(),
                    message=str(result),
                )
            else:
                health_map[name] = result

            # Store in history
            if name not in self._health_history:
                self._health_history[name] = []
            self._health_history[name].append(health_map[name])
            # Keep last 100 checks per endpoint
            self._health_history[name] = self._health_history[name][-100:]

            # Generate alerts for unhealthy endpoints
            check = health_map[name]
            if check.status == HealthStatus.UNHEALTHY:
                self._add_alert(
                    AlertSeverity.ERROR,
                    name,
                    f"Endpoint unhealthy: {check.message}",
                    details=asdict(check),
                )
            elif check.status == HealthStatus.DEGRADED:
                self._add_alert(
                    AlertSeverity.WARNING,
                    name,
                    f"Endpoint degraded: {check.message}",
                    details=asdict(check),
                )

        # Log metrics
        await self._log_health_metrics(health_map)

        return health_map

    async def check_tailscale_nodes(self) -> dict[str, HealthCheck]:
        """Check health of Tailscale compute nodes.

        Returns:
            Dict mapping node names to health check results.
        """
        from hafs.core.nodes import node_manager

        await node_manager.load_config()
        status_map = await node_manager.health_check_all()

        health_map = {}
        for name, status in status_map.items():
            node = node_manager.get_node(name)

            if node:
                health_status = {
                    "online": HealthStatus.HEALTHY,
                    "offline": HealthStatus.UNHEALTHY,
                    "busy": HealthStatus.DEGRADED,
                    "error": HealthStatus.UNHEALTHY,
                }.get(status.value, HealthStatus.UNKNOWN)

                health_map[name] = HealthCheck(
                    endpoint=node.base_url,
                    name=name,
                    status=health_status,
                    latency_ms=node.latency_ms,
                    timestamp=datetime.now().isoformat(),
                    message=node.error_message or status.value,
                    details={
                        "models": node.models,
                        "capabilities": node.capabilities,
                        "is_local": node.is_local,
                    },
                )

                if health_status == HealthStatus.UNHEALTHY and not node.is_local:
                    self._add_alert(
                        AlertSeverity.WARNING,
                        f"node:{name}",
                        f"Compute node offline: {node.error_message}",
                    )

        return health_map

    async def check_local_services(self) -> dict[str, HealthCheck]:
        """Check health of local services.

        Returns:
            Dict mapping service names to health check results.
        """
        local_services = [
            {"name": "ollama-local", "url": "http://localhost:11434/api/tags", "type": "api"},
        ]

        health_map = {}
        for service in local_services:
            health_map[service["name"]] = await self.check_endpoint(service)

        return health_map

    async def check_sync_status(self) -> dict[str, HealthCheck]:
        """Check AFS sync status from the latest sync run."""
        if not self.sync_status_file.exists():
            return {}

        try:
            data = json.loads(self.sync_status_file.read_text())
        except Exception as exc:
            logger.warning("Failed to load sync status: %s", exc)
            return {}

        health_map: dict[str, HealthCheck] = {}
        profiles = data.get("profiles", {})
        now = datetime.now()

        for profile_name, profile_data in profiles.items():
            targets = profile_data.get("targets", {})
            for target_name, record in targets.items():
                timestamp = record.get("timestamp")
                last_seen = None
                if timestamp:
                    try:
                        last_seen = datetime.fromisoformat(timestamp)
                    except ValueError:
                        last_seen = None

                exit_code = record.get("exit_code", 0)
                age_hours = None
                if last_seen:
                    age_hours = (now - last_seen).total_seconds() / 3600

                if exit_code != 0:
                    status = HealthStatus.UNHEALTHY
                    message = record.get("stderr") or "Sync failed"
                    self._add_alert(
                        AlertSeverity.ERROR,
                        f"sync:{profile_name}:{target_name}",
                        f"AFS sync failed ({exit_code})",
                        details=record,
                    )
                elif age_hours is not None and age_hours > self.SYNC_STALE_HOURS:
                    status = HealthStatus.DEGRADED
                    message = f"Last sync {age_hours:.1f}h ago"
                    self._add_alert(
                        AlertSeverity.WARNING,
                        f"sync:{profile_name}:{target_name}",
                        message,
                        details=record,
                    )
                else:
                    status = HealthStatus.HEALTHY
                    message = "Sync healthy"

                health_map[f"sync:{profile_name}:{target_name}"] = HealthCheck(
                    endpoint=target_name,
                    name=f"sync:{profile_name}",
                    status=status,
                    latency_ms=0,
                    timestamp=timestamp or datetime.now().isoformat(),
                    message=message,
                    details={
                        "profile": profile_name,
                        "target": target_name,
                        "direction": record.get("direction"),
                        "exit_code": exit_code,
                        "last_seen": timestamp,
                        "duration_ms": record.get("duration_ms"),
                        "dry_run": record.get("dry_run"),
                    },
                )

        await self._log_health_metrics(health_map)
        return health_map

    def _add_alert(
        self,
        severity: AlertSeverity,
        source: str,
        message: str,
        details: dict = None,
    ):
        """Add an alert.

        Args:
            severity: Alert severity.
            source: Source of the alert.
            message: Alert message.
            details: Optional additional details.
        """
        # Avoid duplicate alerts in short time window
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        for alert in self._alerts:
            if (
                alert.source == source
                and alert.message == message
                and datetime.fromisoformat(alert.timestamp) > recent_cutoff
            ):
                return  # Duplicate

        alert = Alert(
            severity=severity,
            source=source,
            message=message,
            timestamp=datetime.now().isoformat(),
            details=details or {},
        )
        self._alerts.append(alert)
        self._save_alerts()

        logger.warning(f"[ALERT:{severity.value}] {source}: {message}")

    def get_active_alerts(
        self,
        severity_filter: Optional[AlertSeverity] = None,
    ) -> list[Alert]:
        """Get active (unacknowledged) alerts.

        Args:
            severity_filter: Optional severity to filter by.

        Returns:
            List of active alerts.
        """
        alerts = [a for a in self._alerts if not a.acknowledged]

        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def acknowledge_alert(self, timestamp: str) -> bool:
        """Acknowledge an alert.

        Args:
            timestamp: Alert timestamp to acknowledge.

        Returns:
            True if alert was found and acknowledged.
        """
        for alert in self._alerts:
            if alert.timestamp == timestamp:
                alert.acknowledged = True
                self._save_alerts()
                return True
        return False

    async def _log_health_metrics(self, health_map: dict[str, HealthCheck]):
        """Log health metrics to file.

        Args:
            health_map: Health check results.
        """
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        metrics_file = self.metrics_dir / f"health_{date_str}.jsonl"

        try:
            with open(metrics_file, "a") as f:
                for name, check in health_map.items():
                    metric = {
                        "name": name,
                        "status": check.status.value,
                        "latency_ms": check.latency_ms,
                        "timestamp": check.timestamp,
                    }
                    f.write(json.dumps(metric) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    async def detect_anomalies(
        self,
        hours: int = 24,
    ) -> list[Anomaly]:
        """Detect anomalies in recent metrics.

        Args:
            hours: How many hours to analyze.

        Returns:
            List of detected anomalies.
        """
        anomalies = []

        for name, history in self._health_history.items():
            if len(history) < 10:
                continue  # Not enough data

            # Calculate baseline latency
            latencies = [h.latency_ms for h in history if h.latency_ms > 0]
            if not latencies:
                continue

            avg_latency = sum(latencies) / len(latencies)
            std_latency = (sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)) ** 0.5

            # Check recent latency
            recent = history[-5:]
            for check in recent:
                if check.latency_ms > avg_latency + (2 * std_latency):
                    deviation = ((check.latency_ms - avg_latency) / avg_latency) * 100

                    anomalies.append(Anomaly(
                        metric=f"{name}_latency",
                        expected_range=(avg_latency - std_latency, avg_latency + std_latency),
                        actual_value=check.latency_ms,
                        deviation_pct=deviation,
                        timestamp=check.timestamp,
                        description=f"Latency spike: {check.latency_ms}ms vs avg {avg_latency:.0f}ms",
                    ))

            # Check error rate
            unhealthy_count = sum(1 for h in recent if h.status == HealthStatus.UNHEALTHY)
            error_rate = unhealthy_count / len(recent)

            if error_rate > self.ERROR_RATE_THRESHOLD:
                anomalies.append(Anomaly(
                    metric=f"{name}_errors",
                    expected_range=(0, self.ERROR_RATE_THRESHOLD),
                    actual_value=error_rate,
                    deviation_pct=(error_rate - self.ERROR_RATE_THRESHOLD) * 100,
                    timestamp=datetime.now().isoformat(),
                    description=f"High error rate: {error_rate:.0%}",
                ))

        return anomalies

    async def analyze_issues(self) -> str:
        """Analyze current issues and suggest remediations.

        Returns:
            Analysis with suggestions.
        """
        # Gather current state
        alerts = self.get_active_alerts()
        anomalies = await self.detect_anomalies()

        if not alerts and not anomalies:
            return "All systems healthy. No issues detected."

        # Build analysis prompt
        alert_summary = "\n".join([
            f"- [{a.severity.value}] {a.source}: {a.message}"
            for a in alerts[:10]
        ])

        anomaly_summary = "\n".join([
            f"- {a.metric}: {a.description} ({a.deviation_pct:.1f}% deviation)"
            for a in anomalies[:10]
        ])

        prompt = f"""Analyze these system issues and suggest remediations:

ACTIVE ALERTS:
{alert_summary or "None"}

DETECTED ANOMALIES:
{anomaly_summary or "None"}

Provide:
1. Root cause analysis
2. Priority ranking of issues
3. Specific remediation steps for each
4. Preventive measures"""

        analysis = await self.generate_thought(prompt)

        # Update alerts with remediation suggestions
        if analysis:
            for alert in alerts:
                if not alert.remediation:
                    alert.remediation = "See analysis for suggested remediation"
            self._save_alerts()

        return analysis

    async def get_system_summary(self) -> dict[str, Any]:
        """Get overall system health summary.

        Returns:
            Summary with status counts and issues.
        """
        # Check everything
        endpoint_health = await self.check_all_endpoints()
        local_health = await self.check_local_services()

        # Try node health (may fail if not configured)
        node_health = {}
        try:
            node_health = await self.check_tailscale_nodes()
        except Exception as e:
            logger.debug(f"Node check failed: {e}")

        sync_health = {}
        try:
            sync_health = await self.check_sync_status()
        except Exception as e:
            logger.debug(f"Sync check failed: {e}")

        # Aggregate
        all_health = {**endpoint_health, **local_health, **node_health, **sync_health}

        status_counts = {s: 0 for s in HealthStatus}
        for check in all_health.values():
            status_counts[check.status] += 1

        # Overall status
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            overall = HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall = HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall = HealthStatus.UNKNOWN
        else:
            overall = HealthStatus.HEALTHY

        # Get active alerts
        alerts = self.get_active_alerts()

        return {
            "overall_status": overall.value,
            "status_counts": {s.value: c for s, c in status_counts.items()},
            "endpoints": {
                name: {"status": h.status.value, "latency_ms": h.latency_ms}
                for name, h in all_health.items()
            },
            "active_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a.severity == AlertSeverity.CRITICAL]),
            "timestamp": datetime.now().isoformat(),
        }

    async def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        self._running = True
        logger.info(f"Starting monitoring loop (interval: {self.check_interval}s)")

        while self._running:
            try:
                await self.check_all_endpoints()
                await self.check_local_services()
                await self.check_sync_status()

                # Check for anomalies
                anomalies = await self.detect_anomalies()
                if anomalies:
                    for anomaly in anomalies:
                        self._add_alert(
                            AlertSeverity.WARNING,
                            f"anomaly:{anomaly.metric}",
                            anomaly.description,
                            details=asdict(anomaly),
                        )

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self._running = False

    async def run_task(self, task: str = "summary") -> dict[str, Any]:
        """Run an observability task.

        Args:
            task: Task to perform:
                - "summary": Get system summary
                - "check": Check all endpoints
                - "alerts": Get active alerts
                - "analyze": Analyze issues
                - "nodes": Check Tailscale nodes
                - "sync": Check sync status

        Returns:
            Task result.
        """
        if task == "summary":
            return await self.get_system_summary()

        elif task == "check":
            health = await self.check_all_endpoints()
            return {
                name: {"status": h.status.value, "latency_ms": h.latency_ms, "message": h.message}
                for name, h in health.items()
            }

        elif task == "alerts":
            alerts = self.get_active_alerts()
            return {
                "count": len(alerts),
                "alerts": [asdict(a) for a in alerts[:20]],
            }

        elif task == "analyze":
            analysis = await self.analyze_issues()
            return {"analysis": analysis}

        elif task == "nodes":
            health = await self.check_tailscale_nodes()
            return {
                name: {"status": h.status.value, "latency_ms": h.latency_ms}
                for name, h in health.items()
            }
        elif task == "sync":
            health = await self.check_sync_status()
            return {
                name: {"status": h.status.value, "message": h.message}
                for name, h in health.items()
            }

        else:
            return {
                "error": "Unknown task",
                "usage": ["summary", "check", "alerts", "analyze", "nodes", "sync"],
            }

    async def close(self):
        """Cleanup resources."""
        self.stop_monitoring()
        if self._session and not self._session.closed:
            await self._session.close()
