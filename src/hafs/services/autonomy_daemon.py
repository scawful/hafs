"""Autonomy Daemon Service.

Runs self-improvement, curiosity, self-healing, shadow observation, and
hallucination watcher loops on a schedule.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from hafs.agents.autonomy_agents import (
    CuriosityExplorerAgent,
    HallucinationWatcherAgent,
    LoopReport,
    SelfHealingAgent,
    SelfImprovementAgent,
    SwarmLogMonitorAgent,
)
from hafs.agents.shadow_observer import ShadowObserver
from hafs.agents.mission_agents import (
    ResearchMission,
    get_mission_agent,
    DEFAULT_MISSIONS,
)
from hafs.core.config import CONTEXT_ROOT
from hafs.core.runtime import resolve_python_executable

# Configure logging
LOG_DIR = CONTEXT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "autonomy_daemon.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class AutonomyTask:
    """A scheduled autonomy task."""

    name: str
    task_type: str
    interval_seconds: float
    last_run: Optional[datetime] = None
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    last_status: str = "pending"
    last_error: Optional[str] = None
    last_report: Optional[str] = None

    def is_due(self) -> bool:
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        elapsed = datetime.now() - self.last_run
        return elapsed.total_seconds() >= self.interval_seconds


class AutonomyDaemon:
    """Daemon for running autonomy loops."""

    DEFAULT_TASKS = [
        AutonomyTask(
            name="self_improvement",
            task_type="self_improvement",
            interval_seconds=6 * 3600,
        ),
        AutonomyTask(
            name="curiosity_explore",
            task_type="curiosity",
            interval_seconds=4 * 3600,
        ),
        AutonomyTask(
            name="self_heal",
            task_type="self_heal",
            interval_seconds=10 * 60,
            config={"auto_restart": True},
        ),
        AutonomyTask(
            name="shadow_observer",
            task_type="shadow_observer",
            interval_seconds=120,
        ),
        AutonomyTask(
            name="swarm_log_watch",
            task_type="swarm_log_watch",
            interval_seconds=120,
            config={
                "auto_restart": True,
                "disable_on_success": True,
                "execution_mode": "infra_ops",
                "progress_window_seconds": 300,
                "stall_seconds": 900,
                "restart_cooldown_seconds": 180,
            },
        ),
        AutonomyTask(
            name="hallucination_watch",
            task_type="hallucination_watch",
            interval_seconds=3600,
        ),
        # Mission agents for deep research
        AutonomyTask(
            name="mission_alttp_sprites",
            task_type="mission",
            interval_seconds=12 * 3600,
            config={"mission_id": "alttp_sprite_patterns"},
        ),
        AutonomyTask(
            name="mission_alttp_memory",
            task_type="mission",
            interval_seconds=24 * 3600,
            config={"mission_id": "alttp_memory_mapping"},
        ),
        AutonomyTask(
            name="mission_cross_reference",
            task_type="mission",
            interval_seconds=24 * 3600,
            config={"mission_id": "alttp_cross_reference"},
        ),
    ]

    def __init__(
        self,
        check_interval_seconds: int = 30,
    ):
        self.check_interval = check_interval_seconds
        self.context_root = CONTEXT_ROOT

        # State
        self._running = False
        self._tasks: list[AutonomyTask] = list(self.DEFAULT_TASKS)

        # Paths
        self.data_dir = self.context_root / "autonomy_daemon"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.data_dir / "daemon.pid"
        self.status_file = self.data_dir / "daemon_status.json"
        self.tasks_file = self.data_dir / "scheduled_tasks.json"
        self.reports_root = self.context_root / "background_agent" / "reports"
        self.reports_root.mkdir(parents=True, exist_ok=True)

        # Agents (lazy loaded)
        self._self_improvement: Optional[SelfImprovementAgent] = None
        self._curiosity: Optional[CuriosityExplorerAgent] = None
        self._self_healing: Optional[SelfHealingAgent] = None
        self._shadow_observer: Optional[ShadowObserver] = None
        self._hallucination: Optional[HallucinationWatcherAgent] = None
        self._swarm_watch: Optional[SwarmLogMonitorAgent] = None

    async def start(self):
        """Start the daemon."""
        logger.info("Starting autonomy daemon...")
        logger.info(f"  Check interval: {self.check_interval}s")

        self._load_tasks()

        self.pid_file.write_text(str(os.getpid()))

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._running = True
        await self._run_loop()

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %s, shutting down...", signum)
        self._running = False

    def _load_tasks(self):
        if self.tasks_file.exists():
            try:
                data = json.loads(self.tasks_file.read_text())
                self._tasks = []
                for t in data:
                    task = AutonomyTask(**t)
                    if task.last_run:
                        task.last_run = datetime.fromisoformat(task.last_run)
                    self._tasks.append(task)
                logger.info("Loaded %s scheduled tasks", len(self._tasks))
            except Exception as exc:
                logger.error("Failed to load tasks: %s", exc)

    def _save_tasks(self):
        try:
            data = []
            for t in self._tasks:
                payload = asdict(t)
                if payload["last_run"]:
                    payload["last_run"] = payload["last_run"].isoformat()
                data.append(payload)
            self.tasks_file.write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.error("Failed to save tasks: %s", exc)

    async def _ensure_self_improvement(self):
        if self._self_improvement is None:
            self._self_improvement = SelfImprovementAgent()
            await self._self_improvement.setup()

    async def _ensure_curiosity(self):
        if self._curiosity is None:
            self._curiosity = CuriosityExplorerAgent()
            await self._curiosity.setup()

    async def _ensure_self_healing(self, auto_restart: bool):
        if self._self_healing is None:
            self._self_healing = SelfHealingAgent(auto_restart=auto_restart)
            await self._self_healing.setup()
        else:
            self._self_healing.auto_restart = auto_restart

    async def _ensure_shadow_observer(self):
        if self._shadow_observer is None:
            self._shadow_observer = ShadowObserver()
            await self._shadow_observer.setup()

    async def _ensure_hallucination(self):
        if self._hallucination is None:
            self._hallucination = HallucinationWatcherAgent()
            await self._hallucination.setup()

    async def _ensure_swarm_watch(self, config: dict[str, Any]):
        if self._swarm_watch is None:
            self._swarm_watch = SwarmLogMonitorAgent(config=config)
            await self._swarm_watch.setup()
        else:
            self._swarm_watch.update_config(config)

    async def _run_loop(self):
        while self._running:
            try:
                await self._check_and_run_tasks()
                self._update_status()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Daemon loop error: %s", exc)
                await asyncio.sleep(60)

        self._cleanup()

    async def _check_and_run_tasks(self):
        for task in self._tasks:
            if not task.is_due():
                continue

            logger.info("Running task: %s", task.name)
            try:
                report = await self._run_task(task)
                task.last_run = datetime.now()
                task.last_status = "success"
                task.last_error = None
                if report:
                    report_path = self._write_report(task.task_type, report)
                    task.last_report = str(report_path) if report_path else None
                    if task.config.get("disable_on_success") and report.metrics.get("completed"):
                        task.enabled = False
                        task.last_status = "completed"
                self._save_tasks()
            except Exception as exc:
                task.last_run = datetime.now()
                task.last_status = "error"
                task.last_error = str(exc)
                logger.error("Task %s failed: %s", task.name, exc)
                self._save_tasks()

    async def _run_task(self, task: AutonomyTask) -> Optional[LoopReport]:
        if task.task_type == "self_improvement":
            await self._ensure_self_improvement()
            return await self._self_improvement.run_task()
        if task.task_type == "curiosity":
            await self._ensure_curiosity()
            return await self._curiosity.run_task()
        if task.task_type == "self_heal":
            auto_restart = bool(task.config.get("auto_restart", True))
            await self._ensure_self_healing(auto_restart=auto_restart)
            return await self._self_healing.run_task()
        if task.task_type == "shadow_observer":
            await self._ensure_shadow_observer()
            processed = await self._shadow_observer.check_history()
            # Get recent commands from memory for context
            recent_commands = await self._shadow_observer.get_recent_commands(limit=10)
            stats = await self._shadow_observer.get_stats()

            body = f"Processed {processed} new commands from shell history.\n\n"
            if recent_commands:
                body += "### Recent Commands\n"
                for cmd in recent_commands:
                    body += f"- `{cmd}`\n"
            else:
                body += "No recent shell activity detected.\n"

            if stats.get("total_commands", 0) > 0:
                body += f"\n### Session Statistics\n"
                body += f"- Total commands observed: {stats.get('total_commands', 0)}\n"
                body += f"- Most common: {stats.get('most_common', 'N/A')}\n"

            metrics = {
                "processed": processed,
                "total_observed": stats.get("total_commands", 0),
                "unique_commands": stats.get("unique_commands", 0),
            }
            return LoopReport(
                title="Shadow Observer Scan",
                body=body,
                tags=["shadow_observer"],
                metrics=metrics,
            )
        if task.task_type == "swarm_log_watch":
            await self._ensure_swarm_watch(task.config)
            if not self._swarm_watch:
                return None
            return await self._swarm_watch.run_task()
        if task.task_type == "hallucination_watch":
            await self._ensure_hallucination()
            return await self._hallucination.run_task()
        if task.task_type == "mission":
            return await self._run_mission(task)
        logger.warning("Unknown task type: %s", task.task_type)
        return None

    async def _run_mission(self, task: AutonomyTask) -> Optional[LoopReport]:
        """Run a mission agent task."""
        mission_id = task.config.get("mission_id")
        if not mission_id:
            logger.error("Mission task %s has no mission_id", task.name)
            return None

        # Find mission definition
        mission = None
        for m in DEFAULT_MISSIONS:
            if m.mission_id == mission_id:
                mission = m
                break

        if not mission:
            # Try loading from custom missions file
            custom_missions_file = self.context_root / "missions" / "custom_missions.json"
            if custom_missions_file.exists():
                try:
                    data = json.loads(custom_missions_file.read_text())
                    for m_data in data.get("missions", []):
                        if m_data.get("mission_id") == mission_id:
                            mission = ResearchMission.from_dict(m_data)
                            break
                except Exception as e:
                    logger.error("Failed to load custom missions: %s", e)

        if not mission:
            logger.error("Mission %s not found", mission_id)
            return None

        # Create and run the agent
        agent = get_mission_agent(mission)
        await agent.setup()
        return await agent.run_task()

    def _write_report(self, task_type: str, report: LoopReport) -> Optional[Path]:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in report.title.lower()
            )[:60]
            report_dir = self.reports_root / task_type
            report_dir.mkdir(parents=True, exist_ok=True)
            path = report_dir / f"{timestamp}_{safe_title}.md"

            body = f"# {report.title}\n\nGenerated: {datetime.now().isoformat()}\n\n{report.body}"
            if report.metrics:
                body += "\n\n## Metrics\n"
                body += json.dumps(report.metrics, indent=2)

            path.write_text(body)
            return path
        except Exception as exc:
            logger.error("Failed to write report: %s", exc)
            return None

    def _update_status(self):
        try:
            status = {
                "pid": os.getpid(),
                "running": self._running,
                "last_update": datetime.now().isoformat(),
                "check_interval_seconds": self.check_interval,
                "tasks": [
                    {
                        "name": t.name,
                        "type": t.task_type,
                        "enabled": t.enabled,
                        "interval_seconds": t.interval_seconds,
                        "last_run": t.last_run.isoformat() if t.last_run else None,
                        "last_status": t.last_status,
                        "last_error": t.last_error,
                        "last_report": t.last_report,
                        "is_due": t.is_due(),
                    }
                    for t in self._tasks
                ],
            }
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception as exc:
            logger.error("Failed to update status: %s", exc)

    def _cleanup(self):
        logger.info("Cleaning up autonomy daemon...")
        if self.pid_file.exists():
            self.pid_file.unlink()
        try:
            status = {"running": False, "stopped": datetime.now().isoformat()}
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception:
            pass
        logger.info("Autonomy daemon stopped")


def get_status() -> dict:
    """Get daemon status."""
    status_file = CONTEXT_ROOT / "autonomy_daemon" / "daemon_status.json"
    pid_file = CONTEXT_ROOT / "autonomy_daemon" / "daemon.pid"

    result = {"running": False}

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


def install_launchd():
    """Install launchd plist for macOS."""
    python_path = resolve_python_executable()

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hafs.autonomy-daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hafs.services.autonomy_daemon</string>
        <string>--interval</string>
        <string>30</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{Path.home() / "Code" / "hafs"}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{Path.home() / "Code" / "hafs" / "src"}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{CONTEXT_ROOT / "logs" / "autonomy_daemon.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{CONTEXT_ROOT / "logs" / "autonomy_daemon.err.log"}</string>
    <key>StartInterval</key>
    <integer>300</integer>
</dict>
</plist>
"""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hafs.autonomy-daemon.plist"
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content)

    print(f"Installed launchd plist: {plist_path}")
    print()
    print("To load the service:")
    print(f"  launchctl load {plist_path}")
    print()
    print("To unload:")
    print(f"  launchctl unload {plist_path}")
    print()
    print("To check status:")
    print("  launchctl list | grep hafs.autonomy")


def uninstall_launchd():
    """Uninstall launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hafs.autonomy-daemon.plist"

    if plist_path.exists():
        os.system(f"launchctl unload {plist_path} 2>/dev/null")
        plist_path.unlink()
        print(f"Uninstalled: {plist_path}")
    else:
        print("Service not installed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="Check interval seconds")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--install", action="store_true", help="Install as launchd service")
    parser.add_argument("--uninstall", action="store_true", help="Uninstall launchd service")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(get_status(), indent=2))
        return
    if args.install:
        install_launchd()
        return
    if args.uninstall:
        uninstall_launchd()
        return

    daemon = AutonomyDaemon(check_interval_seconds=args.interval)
    asyncio.run(daemon.start())


if __name__ == "__main__":
    main()
