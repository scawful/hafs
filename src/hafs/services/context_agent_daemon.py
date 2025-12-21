"""Context Agent Daemon Service.

Runs context building agents on a schedule to generate reports
and maintain up-to-date context documentation.

Supports:
- Daily summary reports
- Weekly full module analysis
- On-change detection for Oracle updates

Usage:
    # Run directly
    python -m hafs.services.context_agent_daemon

    # With options
    python -m hafs.services.context_agent_daemon --interval 3600

    # Install as launchd service
    python -m hafs.services.context_agent_daemon --install

    # Check status
    python -m hafs.services.context_agent_daemon --status
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from hafs.core.config import hafs_config
from hafs.core.runtime import resolve_python_executable

# Configure logging
LOG_DIR = Path.home() / ".context" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "context_agent_daemon.log"

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
class ScheduledTask:
    """A scheduled task for the daemon."""

    name: str
    task_type: str  # module_report, feature_report, kb_update, summary, afs_sync
    interval_hours: float
    last_run: Optional[datetime] = None
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)

    def is_due(self) -> bool:
        """Check if task is due to run."""
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        elapsed = datetime.now() - self.last_run
        return elapsed.total_seconds() >= self.interval_hours * 3600


class ContextAgentDaemon:
    """Daemon for scheduled context agent execution."""

    DEFAULT_TASKS = [
        ScheduledTask(
            name="daily_summary",
            task_type="summary",
            interval_hours=24,
            config={"projects": ["alttp", "oracle-of-secrets"]}
        ),
        ScheduledTask(
            name="weekly_module_analysis",
            task_type="module_report",
            interval_hours=168,  # Weekly
            config={"analyze_all": False, "sample_modules": [0x07, 0x09]}
        ),
        ScheduledTask(
            name="oracle_kb_update",
            task_type="kb_update",
            interval_hours=12,
            config={"project": "oracle-of-secrets"}
        ),
        ScheduledTask(
            name="routine_descriptions",
            task_type="kb_enhance",
            interval_hours=6,  # Every 6 hours
            config={"task": "routines:100", "batch_size": 100}
        ),
        ScheduledTask(
            name="semantic_tags",
            task_type="kb_enhance",
            interval_hours=24,  # Daily
            config={"task": "tags"}
        ),
    ]

    def __init__(
        self,
        check_interval_seconds: int = 300,
        max_daily_reports: int = 20,
    ):
        self.check_interval = check_interval_seconds
        self.max_daily_reports = max_daily_reports

        # State
        self._running = False
        self._daily_report_count = 0
        self._daily_reset = datetime.now().date()
        self._tasks: List[ScheduledTask] = list(self.DEFAULT_TASKS)
        self._model_tier: Optional[str] = None

        # Paths
        self.data_dir = Path.home() / ".context" / "context_agent_daemon"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.data_dir / "daemon.pid"
        self.status_file = self.data_dir / "daemon_status.json"
        self.tasks_file = self.data_dir / "scheduled_tasks.json"

        # Components (lazy loaded)
        self._module_analyzer = None
        self._oracle_analyzer = None
        self._oracle_kb_builder = None
        self._report_manager = None
        self._kb_enhancer = None

    async def start(self):
        """Start the daemon."""
        logger.info("Starting context agent daemon...")
        logger.info(f"  Check interval: {self.check_interval}s")
        logger.info(f"  Max daily reports: {self.max_daily_reports}")

        # Load saved tasks
        self._load_tasks()
        self._apply_model_policy()

        # Write PID file
        self.pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        self._running = True
        await self._run_loop()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _apply_model_policy(self) -> None:
        """Apply model policy configuration for context agents."""
        try:
            config = hafs_config.context_agents
        except Exception:
            config = None

        if not config:
            return

        if config.provider:
            os.environ["HAFS_MODEL_PROVIDER"] = config.provider
        if config.model:
            os.environ["HAFS_MODEL_MODEL"] = config.model
        if config.rotation:
            os.environ["HAFS_MODEL_ROTATION"] = ",".join(config.rotation)
        if config.prefer_gpu_nodes:
            os.environ["HAFS_PREFER_GPU_NODES"] = "1"
        if config.prefer_remote_nodes:
            os.environ["HAFS_PREFER_REMOTE_NODES"] = "1"

        if config.model_tier:
            self._model_tier = config.model_tier

    def _apply_agent_policy(self, agent: Any) -> None:
        """Apply runtime model tier overrides to an agent."""
        if self._model_tier:
            agent.model_tier = self._model_tier

    def _load_tasks(self):
        """Load scheduled tasks from disk."""
        if self.tasks_file.exists():
            try:
                data = json.loads(self.tasks_file.read_text())
                self._tasks = []
                for t in data:
                    task = ScheduledTask(**t)
                    if task.last_run:
                        task.last_run = datetime.fromisoformat(task.last_run)
                    self._tasks.append(task)
                logger.info(f"Loaded {len(self._tasks)} scheduled tasks")
            except Exception as e:
                logger.error(f"Failed to load tasks: {e}")

    def _save_tasks(self):
        """Save scheduled tasks to disk."""
        try:
            data = []
            for t in self._tasks:
                d = asdict(t)
                if d["last_run"]:
                    d["last_run"] = d["last_run"].isoformat()
                data.append(d)
            self.tasks_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    async def _ensure_module_analyzer(self):
        """Initialize ALTTP module analyzer."""
        if self._module_analyzer is None:
            try:
                from hafs.agents.alttp_module_analyzer import ALTTPModuleAnalyzer
                self._module_analyzer = ALTTPModuleAnalyzer()
                await self._module_analyzer.setup()
                self._apply_agent_policy(self._module_analyzer)
                logger.info("ALTTPModuleAnalyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize module analyzer: {e}")
                raise

    async def _ensure_oracle_analyzer(self):
        """Initialize Oracle analyzer."""
        if self._oracle_analyzer is None:
            try:
                from hafs.agents.oracle_analyzer import OracleOfSecretsAnalyzer
                self._oracle_analyzer = OracleOfSecretsAnalyzer()
                await self._oracle_analyzer.setup()
                self._apply_agent_policy(self._oracle_analyzer)
                logger.info("OracleOfSecretsAnalyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Oracle analyzer: {e}")
                raise

    async def _ensure_oracle_kb_builder(self):
        """Initialize Oracle KB builder."""
        if self._oracle_kb_builder is None:
            try:
                from hafs.agents.oracle_kb_builder import OracleKBBuilder
                self._oracle_kb_builder = OracleKBBuilder()
                await self._oracle_kb_builder.setup()
                self._apply_agent_policy(self._oracle_kb_builder)
                logger.info("OracleKBBuilder initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Oracle KB builder: {e}")
                raise

    async def _ensure_report_manager(self):
        """Initialize report manager."""
        if self._report_manager is None:
            try:
                from hafs.agents.report_manager import ReportManager
                self._report_manager = ReportManager()
                await self._report_manager.setup()
                self._apply_agent_policy(self._report_manager)
                logger.info("ReportManager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize report manager: {e}")
                raise

    async def _ensure_kb_enhancer(self):
        """Initialize KB enhancer."""
        if self._kb_enhancer is None:
            try:
                from hafs.agents.kb_enhancer import KBEnhancer
                self._kb_enhancer = KBEnhancer()
                await self._kb_enhancer.setup()
                self._apply_agent_policy(self._kb_enhancer)
                logger.info("KBEnhancer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize KB enhancer: {e}")
                raise

    async def _run_loop(self):
        """Main daemon loop."""
        while self._running:
            try:
                # Reset daily count at midnight
                today = datetime.now().date()
                if today != self._daily_reset:
                    self._daily_report_count = 0
                    self._daily_reset = today
                    logger.info("Daily report count reset")

                # Check for due tasks
                await self._check_and_run_tasks()

                # Update status
                self._update_status()

                # Sleep until next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                await asyncio.sleep(60)

        # Cleanup
        self._cleanup()

    async def _check_and_run_tasks(self):
        """Check for due tasks and run them."""
        for task in self._tasks:
            if not task.is_due():
                continue

            if self._daily_report_count >= self.max_daily_reports:
                logger.info("Daily report limit reached, skipping remaining tasks")
                break

            logger.info(f"Running scheduled task: {task.name}")

            try:
                await self._run_task(task)
                task.last_run = datetime.now()
                self._save_tasks()
            except Exception as e:
                logger.error(f"Task {task.name} failed: {e}")

    async def _run_task(self, task: ScheduledTask):
        """Execute a scheduled task."""
        if task.task_type == "module_report":
            await self._run_module_report_task(task)
        elif task.task_type == "kb_update":
            await self._run_kb_update_task(task)
        elif task.task_type == "summary":
            await self._run_summary_task(task)
        elif task.task_type == "feature_report":
            await self._run_feature_report_task(task)
        elif task.task_type == "afs_sync":
            await self._run_afs_sync_task(task)
        elif task.task_type == "kb_enhance":
            await self._run_kb_enhance_task(task)
        else:
            logger.warning(f"Unknown task type: {task.task_type}")

    async def _run_module_report_task(self, task: ScheduledTask):
        """Run module report generation."""
        await self._ensure_module_analyzer()

        config = task.config
        if config.get("analyze_all"):
            result = await self._module_analyzer.analyze_all_modules(delay_seconds=5.0)
            self._daily_report_count += result.get("analyzed", 0)
        else:
            modules = config.get("sample_modules", [0x07])
            for module_id in modules:
                try:
                    result = await self._module_analyzer.analyze_module(module_id)
                    self._daily_report_count += 1
                    logger.info(f"Module report generated: {result.get('report_path')}")
                except Exception as e:
                    logger.error(f"Module 0x{module_id:02X} analysis failed: {e}")
                await asyncio.sleep(2)

    async def _run_kb_update_task(self, task: ScheduledTask):
        """Run knowledge base update."""
        config = task.config
        project = config.get("project", "oracle-of-secrets")

        if project == "oracle-of-secrets":
            await self._ensure_oracle_kb_builder()
            result = await self._oracle_kb_builder.build_from_source(generate_embeddings=True)
            logger.info(f"Oracle KB updated: {result}")

    async def _run_summary_task(self, task: ScheduledTask):
        """Run daily summary generation."""
        await self._ensure_report_manager()

        projects = task.config.get("projects", ["alttp"])
        stats = self._report_manager.get_statistics()

        # Generate summary report
        summary_dir = Path.home() / ".context" / "reports" / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)

        summary = f"""# Daily Context Summary

Generated: {datetime.now().isoformat()}

## Report Statistics

{json.dumps(stats, indent=2)}

## Recent Reports

"""
        recent = self._report_manager.get_recent_reports(limit=10)
        for r in recent:
            summary += f"- **{r['topic']}** ({r['project']}/{r['report_type']})\n"
            summary += f"  Created: {r['created']}\n\n"

        path = summary_dir / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.md"
        path.write_text(summary)
        self._daily_report_count += 1
        logger.info(f"Daily summary generated: {path}")

    async def _run_feature_report_task(self, task: ScheduledTask):
        """Run feature report generation."""
        await self._ensure_oracle_analyzer()

        features = task.config.get("features", [])
        for feature in features:
            try:
                result = await self._oracle_analyzer.analyze_feature(feature)
                self._daily_report_count += 1
                logger.info(f"Feature report generated: {result.get('report_path')}")
            except Exception as e:
                logger.error(f"Feature '{feature}' analysis failed: {e}")
            await asyncio.sleep(2)

    async def _run_afs_sync_task(self, task: ScheduledTask) -> None:
        """Run AFS sync profiles."""
        from hafs.services.afs_sync import AFSSyncService

        profiles = task.config.get("profiles", [])
        direction = task.config.get("direction")
        dry_run = bool(task.config.get("dry_run", False))

        service = AFSSyncService()
        await service.load()

        if not profiles:
            profiles = [p.name for p in service.list_profiles() if p.enabled]

        for profile in profiles:
            try:
                await service.run_profile(profile, direction_override=direction, dry_run=dry_run)
                logger.info("AFS sync complete for profile: %s", profile)
            except Exception as exc:
                logger.error("AFS sync failed for profile %s: %s", profile, exc)

    async def _run_kb_enhance_task(self, task: ScheduledTask) -> None:
        """Run knowledge base enhancement task."""
        await self._ensure_kb_enhancer()

        task_name = task.config.get("task", "all")
        try:
            result = await self._kb_enhancer.run_task(task_name)
            logger.info(f"KB enhancement complete: {result}")
        except Exception as e:
            logger.error(f"KB enhancement failed: {e}")

    def _update_status(self):
        """Update daemon status file."""
        try:
            status = {
                "pid": os.getpid(),
                "running": self._running,
                "last_update": datetime.now().isoformat(),
                "daily_report_count": self._daily_report_count,
                "daily_limit": self.max_daily_reports,
                "check_interval_seconds": self.check_interval,
                "tasks": [
                    {
                        "name": t.name,
                        "type": t.task_type,
                        "enabled": t.enabled,
                        "last_run": t.last_run.isoformat() if t.last_run else None,
                        "is_due": t.is_due(),
                    }
                    for t in self._tasks
                ]
            }
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def _cleanup(self):
        """Cleanup on shutdown."""
        logger.info("Cleaning up...")
        if self.pid_file.exists():
            self.pid_file.unlink()

        # Final status update
        try:
            status = {"running": False, "stopped": datetime.now().isoformat()}
            self.status_file.write_text(json.dumps(status, indent=2))
        except Exception:
            pass

        logger.info("Context agent daemon stopped")


def get_status() -> dict:
    """Get daemon status."""
    status_file = Path.home() / ".context" / "context_agent_daemon" / "daemon_status.json"
    pid_file = Path.home() / ".context" / "context_agent_daemon" / "daemon.pid"

    result = {"running": False}

    if status_file.exists():
        try:
            result = json.loads(status_file.read_text())
        except Exception:
            pass

    # Check if process is actually running
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
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
    <string>com.hafs.context-agent-daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>hafs.services.context_agent_daemon</string>
        <string>--interval</string>
        <string>300</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{Path.home() / "Code" / "hafs"}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PYTHONPATH</key>
        <string>{Path.home() / "Code" / "hafs" / "src"}</string>
        <key>GEMINI_API_KEY</key>
        <string>{os.environ.get("GEMINI_API_KEY", "")}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{Path.home() / ".context" / "logs" / "context_agent_daemon.out.log"}</string>
    <key>StandardErrorPath</key>
    <string>{Path.home() / ".context" / "logs" / "context_agent_daemon.err.log"}</string>
    <key>StartInterval</key>
    <integer>3600</integer>
</dict>
</plist>
"""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hafs.context-agent-daemon.plist"
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
    print("  launchctl list | grep hafs.context")


def uninstall_launchd():
    """Uninstall launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.hafs.context-agent-daemon.plist"

    if plist_path.exists():
        os.system(f"launchctl unload {plist_path} 2>/dev/null")
        plist_path.unlink()
        print(f"Uninstalled: {plist_path}")
    else:
        print("Service not installed")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Context Agent Daemon Service")
    parser.add_argument("--interval", "-i", type=int, default=300,
                        help="Check interval in seconds (default: 300)")
    parser.add_argument("--max-daily", "-m", type=int, default=20,
                        help="Maximum reports per day (default: 20)")
    parser.add_argument("--install", action="store_true",
                        help="Install as launchd service")
    parser.add_argument("--uninstall", action="store_true",
                        help="Uninstall launchd service")
    parser.add_argument("--status", action="store_true",
                        help="Check daemon status")

    args = parser.parse_args()

    if args.install:
        install_launchd()
        return

    if args.uninstall:
        uninstall_launchd()
        return

    if args.status:
        status = get_status()
        print(json.dumps(status, indent=2))
        return

    # Run daemon
    daemon = ContextAgentDaemon(
        check_interval_seconds=args.interval,
        max_daily_reports=args.max_daily,
    )
    await daemon.start()


if __name__ == "__main__":
    asyncio.run(main())
