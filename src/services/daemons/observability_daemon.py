"""Observability daemon with optional safe remediations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from agents.utility.observability import (
    Alert,
    DistributedObservabilityAgent,
    HealthCheck,
    HealthStatus,
)
from config.loader import load_config
from config.schema import ObservabilityRemediationConfig
from core.config import CONTEXT_ROOT
from services.afs_sync import AFSSyncService
from services import ServiceManager, ServiceState

# Configure logging
LOG_DIR = CONTEXT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "observability_daemon.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemediationAction:
    """A safe remediation action."""

    action: str
    target: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RemediationPlanner:
    """Plan remediation actions within configurable safety guardrails."""

    def __init__(
        self,
        config: ObservabilityRemediationConfig,
        recent_actions: Optional[dict[str, str]] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self._config = config
        self._recent_actions = recent_actions or {}
        self._now = now or datetime.now()
        self._cooldown = timedelta(minutes=max(config.cooldown_minutes, 0))
        self._allowed_actions = {
            entry.strip().lower()
            for entry in config.allowed_actions
            if entry and entry.strip()
        }
        self._allowed_services = {
            self._normalize_name(name)
            for name in config.allowed_services
            if name and name.strip()
        }
        self._allowed_profiles = {
            name.strip()
            for name in config.allowed_sync_profiles
            if name and name.strip()
        }
        self._burst_severities = {
            severity.strip().lower()
            for severity in config.trigger_context_burst_on_alerts
            if severity and severity.strip()
        }

    def plan(
        self,
        *,
        service_statuses: dict[str, Any],
        sync_health: dict[str, HealthCheck],
        alerts: list[Alert],
    ) -> list[RemediationAction]:
        if not self._config.enabled or not self._allowed_actions:
            return []

        actions: list[RemediationAction] = []

        actions.extend(self._plan_service_actions(service_statuses))
        actions.extend(self._plan_sync_actions(sync_health))
        actions.extend(self._plan_context_burst(alerts))

        limit = max(self._config.max_actions_per_run, 0)
        if limit == 0:
            return []
        return actions[:limit]

    def _plan_service_actions(self, service_statuses: dict[str, Any]) -> list[RemediationAction]:
        if not self._allowed_services:
            return []

        actions: list[RemediationAction] = []
        for name in sorted(service_statuses.keys()):
            status = service_statuses[name]
            normalized = self._normalize_name(name)
            if normalized not in self._allowed_services:
                continue
            if not getattr(status, "enabled", False):
                continue
            if status.state == ServiceState.FAILED and self._action_allowed("restart_service", name):
                actions.append(
                    RemediationAction(
                        action="restart_service",
                        target=name,
                        reason="Service failed",
                    )
                )
            elif status.state == ServiceState.STOPPED and self._action_allowed("start_service", name):
                actions.append(
                    RemediationAction(
                        action="start_service",
                        target=name,
                        reason="Service stopped",
                    )
                )
        return actions

    def _plan_sync_actions(self, sync_health: dict[str, HealthCheck]) -> list[RemediationAction]:
        if "run_afs_sync" not in self._allowed_actions or not self._allowed_profiles:
            return []

        actions: list[RemediationAction] = []
        seen_profiles: set[str] = set()
        for check in sync_health.values():
            if check.status not in {HealthStatus.UNHEALTHY, HealthStatus.DEGRADED}:
                continue
            profile = check.details.get("profile")
            if not profile or profile not in self._allowed_profiles:
                continue
            if profile in seen_profiles:
                continue
            if not self._action_allowed("run_afs_sync", profile):
                continue
            seen_profiles.add(profile)
            actions.append(
                RemediationAction(
                    action="run_afs_sync",
                    target=profile,
                    reason=f"Sync {check.status.value}",
                    metadata={"check": check.name, "message": check.message},
                )
            )
        return actions

    def _plan_context_burst(self, alerts: list[Alert]) -> list[RemediationAction]:
        if "context_burst" not in self._allowed_actions:
            return []
        if not self._burst_severities:
            return []
        if not any(alert.severity.value in self._burst_severities for alert in alerts):
            return []
        if not self._action_allowed("context_burst", "context-agent-daemon"):
            return []
        return [
            RemediationAction(
                action="context_burst",
                target="context-agent-daemon",
                reason="Alerts triggered context burst",
            )
        ]

    def _action_allowed(self, action: str, target: str) -> bool:
        if action not in self._allowed_actions:
            return False
        key = self._action_key(action, target)
        last = self._recent_actions.get(key)
        if not last:
            return True
        try:
            last_time = datetime.fromisoformat(last)
        except ValueError:
            return True
        return (self._now - last_time) >= self._cooldown

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.strip().lower().replace("_", "-")

    @staticmethod
    def _action_key(action: str, target: str) -> str:
        return f"{action}:{target}"


class ObservabilityDaemon:
    """Daemon for distributed observability with optional auto-remediation."""

    def __init__(self, interval_seconds: Optional[int] = None) -> None:
        self._config = load_config()
        self._observability_config = self._config.observability
        self._remediation_config = self._observability_config.remediation
        self._interval_override = interval_seconds
        self.check_interval = interval_seconds or self._observability_config.check_interval_seconds

        self._running = False
        self._agent: Optional[DistributedObservabilityAgent] = None
        self._last_actions: dict[str, str] = {}

        self.data_dir = CONTEXT_ROOT / "observability_daemon"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.data_dir / "daemon.pid"
        self.status_file = self.data_dir / "daemon_status.json"
        self.state_file = self.data_dir / "remediation_state.json"

        self._load_state()

    async def start(self) -> None:
        """Start the daemon loop."""
        logger.info("Starting observability daemon...")
        logger.info("  Check interval: %ss", self.check_interval)

        self.pid_file.write_text(str(os.getpid()))

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._running = True
        await self._run_loop()

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %s, shutting down...", signum)
        self._running = False

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            payload = json.loads(self.state_file.read_text())
            self._last_actions = payload.get("last_actions", {})
        except Exception as exc:
            logger.warning("Failed to load remediation state: %s", exc)

    def _save_state(self) -> None:
        try:
            payload = {"last_actions": self._last_actions}
            self.state_file.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.error("Failed to save remediation state: %s", exc)

    async def _ensure_agent(self) -> DistributedObservabilityAgent:
        if self._agent is None:
            endpoints = [
                entry.model_dump()
                for entry in self._observability_config.endpoints
                if entry.enabled
            ]
            self._agent = DistributedObservabilityAgent(
                endpoints=endpoints or None,
                check_interval=self.check_interval,
            )
            await self._agent.setup()
        return self._agent

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Observability loop error: %s", exc)
            await asyncio.sleep(self.check_interval)

        self._cleanup()

    async def run_once(self) -> dict[str, Any]:
        """Run one observability pass."""
        if not self._observability_config.enabled:
            logger.info("Observability disabled in config.")
            summary = {"enabled": False, "timestamp": datetime.now().isoformat()}
            self._update_status(summary, actions=[])
            return summary

        agent = await self._ensure_agent()

        summary: dict[str, Any] = {}
        checks: dict[str, HealthCheck] = {}
        sync_health: dict[str, HealthCheck] = {}

        if self._observability_config.monitor_endpoints:
            checks.update(await agent.check_all_endpoints())
        if self._observability_config.monitor_local_services:
            checks.update(await agent.check_local_services())
        if self._observability_config.monitor_nodes:
            try:
                checks.update(await agent.check_tailscale_nodes())
            except Exception as exc:
                logger.warning("Node checks failed: %s", exc)
        if self._observability_config.monitor_sync:
            sync_health = await agent.check_sync_status()
            checks.update(sync_health)

        service_statuses: dict[str, Any] = {}
        if self._observability_config.monitor_services:
            try:
                manager = ServiceManager(self._config)
                service_statuses = await manager.status_all()
            except Exception as exc:
                logger.warning("Service status check failed: %s", exc)

        alerts = agent.get_active_alerts()

        actions = self._plan_actions(
            service_statuses=service_statuses,
            sync_health=sync_health,
            alerts=alerts,
        )
        results = []
        if actions:
            results = await self._execute_actions(actions)

        summary.update(
            {
                "timestamp": datetime.now().isoformat(),
                "checks": len(checks),
                "alerts": len(alerts),
                "actions": len(actions),
            }
        )
        self._update_status(summary, actions=results)
        return summary

    def _plan_actions(
        self,
        *,
        service_statuses: dict[str, Any],
        sync_health: dict[str, HealthCheck],
        alerts: list[Alert],
    ) -> list[RemediationAction]:
        planner = RemediationPlanner(
            config=self._remediation_config,
            recent_actions=self._last_actions,
        )
        return planner.plan(
            service_statuses=service_statuses,
            sync_health=sync_health,
            alerts=alerts,
        )

    async def _execute_actions(self, actions: list[RemediationAction]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        manager = ServiceManager(self._config)
        sync_service = AFSSyncService()
        sync_loaded = False

        for action in actions:
            timestamp = datetime.now().isoformat()
            success = False
            message = None
            try:
                if action.action == "restart_service":
                    success = await manager.restart(action.target)
                elif action.action == "start_service":
                    success = await manager.start(action.target)
                elif action.action == "run_afs_sync":
                    if not sync_loaded:
                        await sync_service.load()
                        sync_loaded = True
                    results_list = await sync_service.run_profile(action.target)
                    success = all(result.ok for result in results_list)
                    message = f"Ran {len(results_list)} sync targets"
                elif action.action == "context_burst":
                    success = await self._trigger_context_burst(
                        force=self._remediation_config.context_burst_force
                    )
                else:
                    message = "Unknown action"
            except Exception as exc:
                message = str(exc)

            result = {
                "action": action.action,
                "target": action.target,
                "success": success,
                "reason": action.reason,
                "message": message,
                "timestamp": timestamp,
            }
            results.append(result)
            self._record_action(action.action, action.target, timestamp)

        if results:
            self._save_state()
        return results

    async def _trigger_context_burst(self, force: bool) -> bool:
        if self._context_daemon_running():
            self._request_context_burst(force)
            return True

        try:
            from services.context_agent_daemon import ContextAgentDaemon

            daemon = ContextAgentDaemon()
            await daemon.run_burst(force=force)
            return True
        except Exception as exc:
            logger.error("Failed to run context burst: %s", exc)
            return False

    @staticmethod
    def _context_daemon_running() -> bool:
        pid_file = Path.home() / ".context" / "context_agent_daemon" / "daemon.pid"
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, ValueError):
            return False

    @staticmethod
    def _request_context_burst(force: bool) -> None:
        data_dir = Path.home() / ".context" / "context_agent_daemon"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "requested_at": datetime.now().isoformat(),
            "force": force,
        }
        (data_dir / "burst_request.json").write_text(json.dumps(payload, indent=2))

    def _record_action(self, action: str, target: str, timestamp: str) -> None:
        key = RemediationPlanner._action_key(action, target)
        self._last_actions[key] = timestamp

    def _update_status(self, summary: dict[str, Any], actions: list[dict[str, Any]]) -> None:
        try:
            status = {
                "pid": os.getpid(),
                "running": self._running,
                "last_update": datetime.now().isoformat(),
                "check_interval_seconds": self.check_interval,
                "summary": summary,
                "actions": actions,
            }
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception as exc:
            logger.error("Failed to update status: %s", exc)

    def _cleanup(self) -> None:
        logger.info("Stopping observability daemon...")
        if self.pid_file.exists():
            self.pid_file.unlink()
        try:
            status = {"running": False, "stopped": datetime.now().isoformat()}
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception:
            pass
        logger.info("Observability daemon stopped")


def get_status() -> dict[str, Any]:
    """Get daemon status."""
    status_file = CONTEXT_ROOT / "observability_daemon" / "daemon_status.json"
    pid_file = CONTEXT_ROOT / "observability_daemon" / "daemon.pid"

    result: dict[str, Any] = {"running": False}

    if status_file.exists():
        try:
            result = json.loads(status_file.read_text())
        except Exception:
            pass

    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            result["running"] = True
            result["pid"] = pid
        except (ProcessLookupError, ValueError):
            result["running"] = False

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAFS observability daemon")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Seconds between checks (overrides config)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check and exit",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check daemon status",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.status:
        status = get_status()
        print(json.dumps(status, indent=2))
        return

    daemon = ObservabilityDaemon(interval_seconds=args.interval)

    if args.once:
        asyncio.run(daemon.run_once())
        return

    asyncio.run(daemon.start())


if __name__ == "__main__":
    main()
