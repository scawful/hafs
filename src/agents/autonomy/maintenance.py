"""Maintenance agents for system health, testing, and context discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from hafs.core.execution import ExecutionPolicy
from hafs.core.projects import ProjectRegistry
from hafs.core.tooling import ToolRunner, ToolResult


def _find_repo_root() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _safe_parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_result(result: ToolResult, max_lines: int = 20) -> str:
    lines = (result.stdout + "\n" + result.stderr).strip().splitlines()
    if not lines:
        return "(no output)"
    trimmed = lines[-max_lines:]
    if len(lines) > max_lines:
        trimmed.insert(0, f"... ({len(lines) - max_lines} lines trimmed)")
    return "\n".join(trimmed)


@dataclass
class ContextDiscoveryState:
    last_scan_at: Optional[str] = None


class ContextDiscoveryAgent(MemoryAwareAgent):
    """Find new/updated files and trigger context + embedding refreshes."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("ContextDiscovery", "Detect new context candidates and refresh knowledge.")
        defaults = {
            "lookback_hours": 24,
            "max_files": 200,
            "extensions": [
                ".md",
                ".txt",
                ".rst",
                ".py",
                ".asm",
                ".s",
                ".c",
                ".cpp",
                ".h",
                ".hpp",
                ".json",
            ],
            "trigger_context_burst": True,
            "force_context_burst": False,
            "trigger_embeddings": True,
            "embedding_batches": 1,
            "embedding_batch_size": 50,
        }
        merged = defaults.copy()
        merged.update(config or {})
        self._config = merged
        self._state_file = self.context_root / "autonomy_daemon" / "context_discovery_state.json"
        self._state = self._load_state()

    def _load_state(self) -> ContextDiscoveryState:
        try:
            if self._state_file.exists():
                payload = json.loads(self._state_file.read_text())
                return ContextDiscoveryState(last_scan_at=payload.get("last_scan_at"))
        except Exception:
            pass
        return ContextDiscoveryState()

    def _save_state(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"last_scan_at": self._state.last_scan_at}
            self._state_file.write_text(json.dumps(payload, indent=2))
        except Exception:
            return

    def _iter_recent_files(
        self,
        root: Path,
        since: datetime,
        max_files: int,
        extensions: set[str],
    ) -> list[Path]:
        results: list[Path] = []
        if not root.exists():
            return results
        for path in root.rglob("*"):
            if len(results) >= max_files:
                break
            if not path.is_file():
                continue
            if extensions and path.suffix.lower() not in extensions:
                continue
            try:
                if datetime.fromtimestamp(path.stat().st_mtime) > since:
                    results.append(path)
            except Exception:
                continue
        return results

    async def _trigger_context_burst(self, force: bool) -> int:
        try:
            from hafs.services.context_agent_daemon import ContextAgentDaemon

            daemon = ContextAgentDaemon()
            return await daemon.run_burst(force=force)
        except Exception:
            return 0

    async def _trigger_embeddings(self, batches: int, batch_size: int) -> int:
        try:
            from hafs.services.embedding_daemon import EmbeddingDaemon

            daemon = EmbeddingDaemon(batch_size=batch_size)
            generated_total = 0
            for _ in range(max(1, batches)):
                generated = await daemon.run_once()
                generated_total += generated
                if generated == 0:
                    break
            return generated_total
        except Exception:
            return 0

    async def run_task(self) -> LoopReport:
        now = datetime.now()
        last_scan = _safe_parse_datetime(self._state.last_scan_at)
        if not last_scan:
            lookback = int(self._config.get("lookback_hours", 24))
            last_scan = now - timedelta(hours=lookback)

        registry = ProjectRegistry.load()
        extensions = set(self._config.get("extensions") or [])
        max_files = int(self._config.get("max_files", 200))
        discovered: list[Path] = []

        projects = registry.list()
        roots_to_scan: list[Path] = []
        if projects:
            for project in projects:
                roots_to_scan.extend(project.knowledge_roots or [project.path])
        else:
            roots_to_scan.append(_find_repo_root() or Path.cwd())

        for root in roots_to_scan:
            discovered.extend(
                self._iter_recent_files(root, last_scan, max_files - len(discovered), extensions)
            )
            if len(discovered) >= max_files:
                break

        self._state.last_scan_at = now.isoformat()
        self._save_state()

        context_runs = 0
        embeddings_generated = 0
        if discovered:
            if self._config.get("trigger_context_burst", True):
                context_runs = await self._trigger_context_burst(
                    force=bool(self._config.get("force_context_burst", False))
                )
            if self._config.get("trigger_embeddings", True):
                embeddings_generated = await self._trigger_embeddings(
                    batches=int(self._config.get("embedding_batches", 1)),
                    batch_size=int(self._config.get("embedding_batch_size", 50)),
                )

        lines = [
            f"Scan window start: {last_scan.isoformat()}",
            f"Files discovered: {len(discovered)}",
        ]
        if discovered:
            preview = discovered[:10]
            lines.append("")
            lines.append("## Recent Files")
            lines.extend(f"- {path}" for path in preview)
            if len(discovered) > len(preview):
                lines.append(f"- ... ({len(discovered) - len(preview)} more)")
        if context_runs:
            lines.append(f"\nContext burst tasks run: {context_runs}")
        if embeddings_generated:
            lines.append(f"Embeddings generated: {embeddings_generated}")

        await self.remember(
            content=f"Discovered {len(discovered)} files; embeddings {embeddings_generated}",
            memory_type="insight",
            context={
                "files": [str(p) for p in discovered[:20]],
                "context_runs": context_runs,
                "embeddings": embeddings_generated,
            },
            importance=0.4 if discovered else 0.2,
        )

        return LoopReport(
            title="Context Discovery",
            body="\n".join(lines),
            tags=["context", "discovery", "embeddings"],
            metrics={
                "files": len(discovered),
                "context_burst": context_runs,
                "embeddings": embeddings_generated,
            },
        )


class TestRunnerAgent(MemoryAwareAgent):
    """Run test suites on a schedule."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("TestRunner", "Run scheduled test suites.")
        defaults = {
            "paths": ["tests"],
            "args": [],
            "timeout_seconds": 1800,
            "execution_mode": "build_only",
        }
        merged = defaults.copy()
        merged.update(config or {})
        self._config = merged
        self._runner: Optional[ToolRunner] = None
        self._root = _find_repo_root() or Path.cwd()

    async def setup(self) -> None:
        await super().setup()
        policy = ExecutionPolicy(execution_mode=str(self._config.get("execution_mode", "build_only")))
        profile = policy.resolve_tool_profile(None)
        self._runner = ToolRunner(root=self._root, profile=profile)

    async def _run_pytest(self, env: Optional[dict[str, str]] = None) -> ToolResult:
        if not self._runner:
            await self.setup()
        assert self._runner is not None
        args = list(self._config.get("args") or [])
        args.extend(self._config.get("paths") or [])
        env_vars = env or {}
        if (self._root / "src").exists():
            env_vars.setdefault("PYTHONPATH", str(self._root / "src"))
        return await self._runner.run(
            "pytest",
            args=args,
            timeout=int(self._config.get("timeout_seconds", 1800)),
            env=env_vars,
        )

    async def run_task(self) -> LoopReport:
        try:
            result = await self._run_pytest()
        except Exception as exc:
            await self.remember(
                content=f"Test runner failed: {exc}",
                memory_type="error",
                context={"error": str(exc)},
                importance=0.7,
            )
            return LoopReport(
                title="Test Suite",
                body=f"Test runner failed: {exc}",
                tags=["tests", "failure"],
                metrics={"ok": False},
            )

        output = _format_result(result)
        summary = "passed" if result.ok else "failed"
        body = f"Status: {summary}\nExit code: {result.exit_code}\n\n{output}"
        await self.remember(
            content=f"Tests {summary} (exit {result.exit_code})",
            memory_type="insight" if result.ok else "error",
            context={"exit_code": result.exit_code},
            importance=0.5 if result.ok else 0.8,
        )
        return LoopReport(
            title="Test Suite",
            body=body,
            tags=["tests", summary],
            metrics={"ok": result.ok, "exit_code": result.exit_code},
        )


class QualityAuditAgent(TestRunnerAgent):
    """Run the quality audit suite."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        defaults = {
            "paths": ["tests/test_quality_audit.py"],
            "args": [],
            "timeout_seconds": 1800,
            "execution_mode": "build_only",
        }
        merged = defaults.copy()
        merged.update(config or {})
        super().__init__(merged)
        self.name = "QualityAudit"
        self.role_description = "Run quality audit suites and report results."
        self.metrics.name = self.name

    async def run_task(self) -> LoopReport:
        try:
            result = await self._run_pytest(env={"HAFS_RUN_QUALITY_AUDIT": "1"})
        except Exception as exc:
            await self.remember(
                content=f"Quality audit failed: {exc}",
                memory_type="error",
                context={"error": str(exc)},
                importance=0.8,
            )
            return LoopReport(
                title="Quality Audit",
                body=f"Quality audit failed: {exc}",
                tags=["quality", "failure"],
                metrics={"ok": False},
            )

        output = _format_result(result)
        summary = "passed" if result.ok else "failed"
        body = f"Status: {summary}\nExit code: {result.exit_code}\n\n{output}"
        await self.remember(
            content=f"Quality audit {summary} (exit {result.exit_code})",
            memory_type="insight" if result.ok else "error",
            context={"exit_code": result.exit_code},
            importance=0.6 if result.ok else 0.9,
        )
        return LoopReport(
            title="Quality Audit",
            body=body,
            tags=["quality", summary],
            metrics={"ok": result.ok, "exit_code": result.exit_code},
        )
