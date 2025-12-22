"""Training Daemon - Background service for training data generation.

Schedules and executes training data generation tasks,
coordinates with remote nodes, and manages quality pipelines.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrainingTask:
    """A scheduled training task."""

    name: str
    task_type: str  # generation, curation, quality_review, export
    domain: Optional[str] = None
    interval_seconds: int = 86400  # Default: daily
    limit: int = 100
    enabled: bool = True
    last_run: str = ""
    next_run: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def should_run(self) -> bool:
        """Check if task should run based on schedule."""
        if not self.enabled:
            return False
        if not self.last_run:
            return True

        last = datetime.fromisoformat(self.last_run)
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed >= self.interval_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type,
            "domain": self.domain,
            "interval_seconds": self.interval_seconds,
            "limit": self.limit,
            "enabled": self.enabled,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "config": self.config,
        }


class TrainingDaemon:
    """Background daemon for training data generation.

    Features:
    - Scheduled generation from multiple domains
    - Quality review of existing samples
    - Coordination with remote training nodes
    - Status tracking and metrics
    """

    DEFAULT_TASKS = [
        # Domain-specific generation
        TrainingTask(
            name="asm_generation",
            task_type="generation",
            domain="asm",
            interval_seconds=24 * 3600,  # Daily
            limit=100,
        ),
        TrainingTask(
            name="cpp_generation",
            task_type="generation",
            domain="cpp",
            interval_seconds=48 * 3600,  # Every 2 days
            limit=50,
        ),
        TrainingTask(
            name="text_generation",
            task_type="generation",
            domain="text",
            interval_seconds=72 * 3600,  # Every 3 days
            limit=50,
            enabled=False,  # Enable when text sources configured
        ),
        # Error and feedback mining (HIGH VALUE)
        TrainingTask(
            name="error_mining",
            task_type="error_mining",
            interval_seconds=6 * 3600,  # Every 6 hours
            limit=50,
            config={"lookback_hours": 24, "min_severity": "medium"},
        ),
        TrainingTask(
            name="history_mining",
            task_type="history_mining",
            interval_seconds=12 * 3600,  # Twice daily
            limit=30,
            config={"lookback_days": 7, "min_success_rate": 0.7},
        ),
        # Quality and curation
        TrainingTask(
            name="quality_review",
            task_type="quality_review",
            interval_seconds=12 * 3600,  # Twice daily
        ),
        TrainingTask(
            name="weekly_curation",
            task_type="curation",
            interval_seconds=7 * 24 * 3600,  # Weekly
            config={"target_count": 1000, "quality_threshold": 0.7},
        ),
        # Multi-teacher validation (premium quality)
        TrainingTask(
            name="multi_teacher_asm",
            task_type="multi_teacher",
            domain="asm",
            interval_seconds=7 * 24 * 3600,  # Weekly
            limit=25,
            config={"consensus_threshold": 2, "validate_locally": True},
        ),
    ]

    def __init__(
        self,
        check_interval: int = 60,
        context_root: Optional[Path] = None,
    ):
        """Initialize the daemon.

        Args:
            check_interval: Seconds between schedule checks
            context_root: Root path for context storage
        """
        self.check_interval = check_interval
        self.context_root = context_root or Path.home() / ".context"

        # State paths
        self.daemon_dir = self.context_root / "training_daemon"
        self.daemon_dir.mkdir(parents=True, exist_ok=True)

        self.pid_path = self.daemon_dir / "daemon.pid"
        self.status_path = self.daemon_dir / "daemon_status.json"
        self.tasks_path = self.daemon_dir / "scheduled_tasks.json"
        self.log_path = self.context_root / "logs" / "training_daemon.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # State
        self._tasks: list[TrainingTask] = []
        self._running = False
        self._curator = None
        self._quality_pipeline = None

    async def setup(self):
        """Initialize components."""
        # Load or initialize tasks
        self._load_tasks()

        # Initialize curator (lazy)
        from agents.training.curator import DataCurator

        self._curator = DataCurator()
        await self._curator.setup()
        await self._curator.register_default_generators()

        # Initialize quality pipeline
        from agents.training.quality import QualityPipeline

        self._quality_pipeline = QualityPipeline()
        await self._quality_pipeline.setup()

        logger.info("TrainingDaemon initialized")

    def _load_tasks(self):
        """Load tasks from disk or use defaults."""
        if self.tasks_path.exists():
            try:
                data = json.loads(self.tasks_path.read_text())
                self._tasks = [TrainingTask(**t) for t in data.get("tasks", [])]
                logger.info(f"Loaded {len(self._tasks)} scheduled tasks")
                return
            except Exception as e:
                logger.warning(f"Failed to load tasks: {e}")

        # Use defaults
        self._tasks = self.DEFAULT_TASKS.copy()
        self._save_tasks()

    def _save_tasks(self):
        """Save tasks to disk."""
        data = {"tasks": [t.to_dict() for t in self._tasks]}
        self.tasks_path.write_text(json.dumps(data, indent=2))

    def _update_status(self, status: str, **kwargs):
        """Update daemon status file."""
        data = {
            "status": status,
            "pid": os.getpid(),
            "updated": datetime.now().isoformat(),
            **kwargs,
        }
        self.status_path.write_text(json.dumps(data, indent=2))

    async def _run_task(self, task: TrainingTask):
        """Execute a scheduled task."""
        logger.info(f"Running task: {task.name}")
        self._update_status("running_task", current_task=task.name)

        try:
            if task.task_type == "generation":
                # Generate samples for a domain
                if task.domain:
                    result = await self._curator.generate_from_domain(
                        domain=task.domain,
                        limit=task.limit,
                    )
                    logger.info(
                        f"Generated {result.processed} samples for {task.domain}"
                    )

            elif task.task_type == "curation":
                # Run full curation pipeline
                result = await self._curator.curate_dataset(
                    target_count=task.config.get("target_count", 1000),
                    quality_threshold=task.config.get("quality_threshold", 0.7),
                    output_name="scheduled_curation",
                )
                logger.info(f"Curated dataset: {result.stats.final_count} samples")

            elif task.task_type == "quality_review":
                # Review quality of existing samples
                await self._run_quality_review()

            elif task.task_type == "export":
                # Export to specific format
                await self._run_export(task.config)

            elif task.task_type == "error_mining":
                # Mine training samples from errors and failures
                await self._run_error_mining(task)

            elif task.task_type == "history_mining":
                # Mine training samples from history logs
                await self._run_history_mining(task)

            elif task.task_type == "multi_teacher":
                # Generate with multi-teacher consensus
                await self._run_multi_teacher(task)

            task.last_run = datetime.now().isoformat()
            self._save_tasks()

        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
            import traceback
            traceback.print_exc()

    async def _run_quality_review(self):
        """Review and re-score existing samples."""
        samples_dir = self.context_root / "training" / "datasets"
        if not samples_dir.exists():
            return

        # Find recent datasets
        for dataset_dir in sorted(samples_dir.iterdir(), reverse=True)[:3]:
            if not dataset_dir.is_dir():
                continue

            train_path = dataset_dir / "train.jsonl"
            if not train_path.exists():
                continue

            # Load and re-score samples
            from agents.training.base import TrainingSample

            entries: list[tuple[dict[str, Any] | str, Optional[TrainingSample]]] = []
            with open(train_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        metadata = data.get("_metadata")
                        if metadata:
                            sample = TrainingSample(
                                instruction=data.get("instruction", ""),
                                input=data.get("input", ""),
                                output=data.get("output", ""),
                                domain=metadata.get("domain", ""),
                                source=metadata.get("source", ""),
                                sample_id=metadata.get("sample_id", ""),
                                quality_score=metadata.get("quality_score", 0.0),
                            )
                            entries.append((data, sample))
                        else:
                            entries.append((data, None))
                    except json.JSONDecodeError:
                        entries.append((line, None))
                        continue

            # Re-score samples
            rescored = 0
            updated = False
            scored = 0
            for data, sample in entries:
                if sample is None:
                    continue
                if scored >= 50:
                    break
                score = await self._quality_pipeline.score(sample)
                scored += 1
                if score.overall != sample.quality_score:
                    sample.quality_score = score.overall
                    if isinstance(data, dict):
                        data.setdefault("_metadata", {})["quality_score"] = score.overall
                    rescored += 1
                    updated = True

            if updated:
                temp_path = train_path.with_suffix(".jsonl.tmp")
                with open(temp_path, "w") as f:
                    for entry, _ in entries:
                        if isinstance(entry, str):
                            f.write(entry if entry.endswith("\n") else entry + "\n")
                        else:
                            f.write(json.dumps(entry) + "\n")
                temp_path.replace(train_path)

            logger.info(f"Re-scored {rescored} samples in {dataset_dir.name}")

    async def _run_export(self, config: dict[str, Any]):
        """Export dataset for a target model."""
        from agents.training.exporter import TrainingExporter

        exporter = TrainingExporter()
        model = config.get("model", "qwen2.5-coder-7b")
        output_dir = Path(config.get("output_dir", self.context_root / "exports"))

        # Find latest dataset
        samples_dir = self.context_root / "training" / "datasets"
        if not samples_dir.exists():
            return

        latest = max(samples_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        train_path = latest / "train.jsonl"

        if train_path.exists():
            from agents.training.base import TrainingSample

            samples = []
            with open(train_path) as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(
                        TrainingSample(
                            instruction=data.get("instruction", ""),
                            input=data.get("input", ""),
                            output=data.get("output", ""),
                            domain="mixed",
                            source="export",
                        )
                    )

            exporter.export_for_model(samples, model, output_dir)
            logger.info(f"Exported {len(samples)} samples for {model}")

    async def _run_error_mining(self, task: TrainingTask):
        """Mine training samples from errors and system failures."""
        from agents.training.generators.error_generator import ErrorSampleGenerator

        lookback_hours = task.config.get("lookback_hours", 24)
        min_severity = task.config.get("min_severity", "medium")

        generator = ErrorSampleGenerator(
            lookback_hours=lookback_hours,
            min_severity=min_severity,
        )
        await generator.setup()

        output_path = self.context_root / "training" / "errors" / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = await generator.run_generation(
            limit=task.limit,
            output_path=output_path,
        )

        logger.info(
            f"Error mining: {result.processed} samples from {lookback_hours}h lookback"
        )

        # Remember successful mining
        await self._curator.remember(
            content=f"Mined {result.processed} error samples",
            memory_type="error_mining",
            context={"lookback_hours": lookback_hours, "samples": result.processed},
            importance=0.6,
        )

    async def _run_history_mining(self, task: TrainingTask):
        """Mine training samples from history logs and workflows."""
        from agents.training.generators.history_miner import HistoryMiner

        lookback_days = task.config.get("lookback_days", 7)
        min_success_rate = task.config.get("min_success_rate", 0.7)

        miner = HistoryMiner(
            lookback_days=lookback_days,
            min_success_rate=min_success_rate,
        )
        await miner.setup()

        output_path = self.context_root / "training" / "workflows" / f"workflows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = await miner.run_generation(
            limit=task.limit,
            output_path=output_path,
        )

        logger.info(
            f"History mining: {result.processed} workflow samples from {lookback_days}d lookback"
        )

    async def _run_multi_teacher(self, task: TrainingTask):
        """Generate samples with multi-teacher consensus for high quality."""
        from agents.training.generators.error_generator import MultiTeacherGenerator

        domain = task.domain
        if not domain:
            logger.warning("Multi-teacher task requires a domain")
            return

        # Get the base generator for this domain
        generator = self._curator.get_generator(domain)
        if not generator:
            logger.warning(f"No generator for domain: {domain}")
            return

        consensus_threshold = task.config.get("consensus_threshold", 2)
        validate_locally = task.config.get("validate_locally", True)

        multi_gen = MultiTeacherGenerator(
            base_generator=generator,
            consensus_threshold=consensus_threshold,
            validate_locally=validate_locally,
        )
        await multi_gen.setup()

        output_path = self.context_root / "training" / "multi_teacher" / f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = await multi_gen.run_generation(
            limit=task.limit,
            output_path=output_path,
        )

        logger.info(
            f"Multi-teacher ({domain}): {result.processed} consensus samples "
            f"(threshold={consensus_threshold})"
        )

    async def run(self):
        """Main daemon loop."""
        await self.setup()

        self._running = True
        self.pid_path.write_text(str(os.getpid()))
        self._update_status("running")

        # Set up signal handlers
        def handle_signal(sig, frame):
            logger.info(f"Received signal {sig}, shutting down...")
            self._running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        logger.info(f"TrainingDaemon started (PID: {os.getpid()})")

        while self._running:
            try:
                # Check for tasks to run
                for task in self._tasks:
                    if task.should_run():
                        await self._run_task(task)

                self._update_status("idle")
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in daemon loop: {e}")
                await asyncio.sleep(self.check_interval)

        # Cleanup
        self._update_status("stopped")
        if self.pid_path.exists():
            self.pid_path.unlink()
        logger.info("TrainingDaemon stopped")

    def add_task(self, task: TrainingTask):
        """Add a new scheduled task."""
        self._tasks.append(task)
        self._save_tasks()

    def remove_task(self, name: str) -> bool:
        """Remove a task by name."""
        for i, task in enumerate(self._tasks):
            if task.name == name:
                del self._tasks[i]
                self._save_tasks()
                return True
        return False

    def get_status(self) -> dict[str, Any]:
        """Get current daemon status."""
        if self.status_path.exists():
            return json.loads(self.status_path.read_text())
        return {"status": "unknown"}

    def list_tasks(self) -> list[TrainingTask]:
        """List all scheduled tasks."""
        return self._tasks.copy()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="hafs Training Daemon")
    parser.add_argument("command", choices=["start", "stop", "status", "tasks"])
    parser.add_argument("--interval", type=int, default=60, help="Check interval")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    daemon = TrainingDaemon(check_interval=args.interval)

    if args.command == "start":
        asyncio.run(daemon.run())

    elif args.command == "stop":
        status = daemon.get_status()
        if status.get("pid"):
            try:
                os.kill(status["pid"], signal.SIGTERM)
                print(f"Sent SIGTERM to PID {status['pid']}")
            except ProcessLookupError:
                print("Daemon not running")
        else:
            print("Daemon not running")

    elif args.command == "status":
        status = daemon.get_status()
        print(json.dumps(status, indent=2))

    elif args.command == "tasks":
        daemon._load_tasks()
        for task in daemon.list_tasks():
            print(f"- {task.name}: {task.task_type} ({task.domain or 'all'})")
            print(f"  Interval: {task.interval_seconds}s, Enabled: {task.enabled}")
            if task.last_run:
                print(f"  Last run: {task.last_run}")


if __name__ == "__main__":
    main()
