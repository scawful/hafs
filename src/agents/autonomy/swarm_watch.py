"""Monitor swarm logs and restart stalled swarm runs."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from hafs.core.execution import ExecutionPolicy
from hafs.core.runtime import resolve_python_executable
from hafs.core.tooling import ToolRunner

_LOG_DIR = Path.home() / ".context" / "logs"
_STATE_DIR = Path.home() / ".context" / "autonomy_daemon"
_STATE_FILE = _STATE_DIR / "swarm_watch_state.json"

_ERROR_PATTERNS = ("Traceback", "ERROR", "Exception")
_NOISE_PATTERNS = (
    "Unclosed client session",
    "Unclosed connector",
)
_SUCCESS_PATTERNS = (
    '"status": "success"',
    "'status': 'success'",
    "status=success",
)
_PHASE_MARKERS = (
    ("synthesizing", "Swarm Synthesizing"),
    ("verifying", "Swarm Verifying"),
    ("collecting", "Swarm Collecting"),
    ("planning", "Swarm Planning"),
)


@dataclass
class SwarmTarget:
    name: str
    topic: str
    log_path: Path
    working_dir: Path
    command: list[str]
    env: dict[str, str]


def _find_repo_root() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _split_log_version(path: Path) -> tuple[str, int]:
    stem = path.stem
    if "_v" in stem:
        base, version = stem.rsplit("_v", 1)
        if version.isdigit():
            return base, int(version)
    return stem, 0


def _normalize_target_name(name: str) -> str:
    if name.startswith("swarm_"):
        return name[6:]
    return name


class SwarmLogMonitorAgent(MemoryAwareAgent):
    """Watches swarm logs for progress and restarts stalled runs."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__("SwarmLogMonitor", "Monitor swarm logs and restart stalled runs.")
        self.model_tier = "fast"
        self._config: dict[str, Any] = {}
        self._runner: Optional[ToolRunner] = None
        self._state: dict[str, Any] = {
            "restarts": {},
            "healthy_streak": {},
            "topics": {},
        }
        self._load_state()
        self.update_config(config or {})

    async def setup(self):
        await super().setup()
        policy = ExecutionPolicy(execution_mode=self._config.get("execution_mode", "infra_ops"))
        profile = policy.resolve_tool_profile(None)
        repo_root = _find_repo_root() or Path.cwd()
        self._runner = ToolRunner(root=repo_root, profile=profile)

    def update_config(self, config: dict[str, Any]) -> None:
        defaults = {
            "auto_restart": True,
            "progress_window_seconds": 300,
            "stall_seconds": 900,
            "restart_cooldown_seconds": 120,
            "restart_window_seconds": 3600,
            "max_restarts": 3,
            "execution_mode": "infra_ops",
            "targets": [],
            "env": {},
            "success_patterns": list(_SUCCESS_PATTERNS),
            "error_patterns": list(_ERROR_PATTERNS),
            "noise_patterns": list(_NOISE_PATTERNS),
        }
        merged = defaults.copy()
        merged.update(config or {})
        self._config = merged

    def _load_state(self) -> None:
        try:
            if _STATE_FILE.exists():
                self._state = json.loads(_STATE_FILE.read_text())
        except Exception:
            self._state = {"restarts": {}, "healthy_streak": {}}

    def _save_state(self) -> None:
        try:
            _STATE_DIR.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(json.dumps(self._state, indent=2))
        except Exception:
            return

    def _extract_topic_from_lines(self, lines: list[str]) -> str:
        for line in reversed(lines):
            if "Swarm Planning:" in line:
                return line.split("Swarm Planning:", 1)[1].strip()
        return ""

    def _extract_topic(self, path: Path) -> str:
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except Exception:
            return ""

        topics: list[str] = []
        for line in lines:
            if "Swarm Planning:" in line:
                topics.append(line.split("Swarm Planning:", 1)[1].strip())
        if not topics:
            topic = self._extract_topic_from_lines(lines[-200:])
            if topic:
                return topic
            return self._extract_topic_from_lines(lines)
        return max(topics, key=len)

    async def _tail_log(self, path: Path) -> list[str]:
        if self._runner:
            try:
                result = await self._runner.run("tail", args=[str(path)])
                if result.ok:
                    return result.stdout.splitlines()
            except Exception:
                pass
        try:
            return path.read_text(errors="ignore").splitlines()[-200:]
        except Exception:
            return []

    async def _ps_lines(self) -> list[str]:
        if not self._runner:
            return []
        try:
            result = await self._runner.run("ps")
            if result.ok:
                return result.stdout.splitlines()
        except Exception:
            return []
        return []

    def _parse_swarm_pids(self, lines: list[str]) -> list[tuple[int, str]]:
        matches: list[tuple[int, str]] = []
        for line in lines:
            if "hafs orchestrate run" not in line:
                continue
            if "--mode swarm" not in line:
                continue
            parts = line.split(None, 10)
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            matches.append((pid, line))
        return matches

    @staticmethod
    def _match_tokens(target: SwarmTarget) -> list[str]:
        tokens = [target.topic, target.name, _normalize_target_name(target.name)]
        return [token for token in tokens if token]

    def _detect_phase(self, lines: list[str]) -> str:
        for phase, marker in _PHASE_MARKERS:
            if any(marker in line for line in lines):
                return phase
        return "unknown"

    def _detect_errors(self, lines: list[str]) -> list[str]:
        patterns = tuple(self._config.get("error_patterns", list(_ERROR_PATTERNS)))
        noise = tuple(self._config.get("noise_patterns", list(_NOISE_PATTERNS)))
        hits = []
        for line in lines:
            if any(token in line for token in noise):
                continue
            if any(pattern in line for pattern in patterns):
                hits.append(line.strip())
        return hits[:5]

    def _detect_success(self, lines: list[str]) -> bool:
        patterns = tuple(self._config.get("success_patterns", list(_SUCCESS_PATTERNS)))
        return any(any(pattern in line for pattern in patterns) for line in lines)

    def _restart_allowed(self, name: str) -> bool:
        now = time.time()
        window = int(self._config.get("restart_window_seconds", 3600))
        history = self._state.setdefault("restarts", {}).setdefault(name, [])
        history = [t for t in history if now - t <= window]
        self._state["restarts"][name] = history
        max_restarts = int(self._config.get("max_restarts", 3))
        return len(history) < max_restarts

    def _record_restart(self, name: str) -> None:
        now = time.time()
        history = self._state.setdefault("restarts", {}).setdefault(name, [])
        history.append(now)
        self._state["restarts"][name] = history

    def _update_healthy_streak(self, name: str, healthy: bool) -> int:
        streaks = self._state.setdefault("healthy_streak", {})
        streaks[name] = (streaks.get(name, 0) + 1) if healthy else 0
        return streaks[name]

    def _build_env(self, overrides: dict[str, str]) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("HAFS_PREFER_USER_CONFIG", "1")
        env.setdefault("HAFS_ENABLE_OLLAMA", "1")
        env.update(self._config.get("env", {}))
        env.update(overrides)
        return env

    def _build_command(self, topic: str) -> list[str]:
        python = resolve_python_executable()
        return [python, "-m", "hafs", "orchestrate", "run", topic, "--mode", "swarm"]

    def _load_targets(self) -> list[SwarmTarget]:
        targets: list[SwarmTarget] = []
        repo_root = _find_repo_root() or Path.cwd()
        config_targets = self._config.get("targets", []) or []
        stored_topics = self._state.setdefault("topics", {})

        if config_targets:
            for entry in config_targets:
                name = entry.get("name") or entry.get("topic", "swarm")
                topic = entry.get("topic") or ""
                log_path = Path(entry.get("log_path", _LOG_DIR / f"swarm_{name}.log")).expanduser()
                env = entry.get("env", {}) or {}
                working_dir = Path(entry.get("working_dir", repo_root)).expanduser()
                command = entry.get("command") or self._build_command(topic or name)
                if isinstance(command, str):
                    command = command.split()
                targets.append(
                    SwarmTarget(
                        name=name,
                        topic=topic or name,
                        log_path=log_path,
                        working_dir=working_dir,
                        command=list(command),
                        env=env,
                    )
                )
            return targets

        if not _LOG_DIR.exists():
            return []

        candidates: dict[str, tuple[int, Path]] = {}
        for path in _LOG_DIR.glob("swarm_*.log"):
            base, version = _split_log_version(path)
            existing = candidates.get(base)
            if not existing or version > existing[0]:
                candidates[base] = (version, path)

        for base, (_, path) in candidates.items():
            stored_topic = stored_topics.get(base)
            extracted = self._extract_topic(path)
            if extracted and (not stored_topic or len(extracted) > len(stored_topic)):
                topic = extracted
            else:
                topic = stored_topic or _normalize_target_name(base)
            stored_topics[base] = topic
            targets.append(
                SwarmTarget(
                    name=base,
                    topic=topic,
                    log_path=path,
                    working_dir=repo_root,
                    command=self._build_command(topic),
                    env={},
                )
            )

        return targets

    async def _restart_target(self, target: SwarmTarget, pids: list[int]) -> str:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            except Exception:
                continue

        await asyncio.sleep(2)

        for pid in pids:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except Exception:
                continue

        target.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = self._build_env(target.env)
        with target.log_path.open("a") as handle:
            subprocess.Popen(
                target.command,
                cwd=str(target.working_dir),
                env=env,
                stdout=handle,
                stderr=handle,
                start_new_session=True,
            )

        return f"restarted {target.name}"

    async def run_task(self) -> LoopReport:
        targets = self._load_targets()
        if not targets:
            return LoopReport(
                title="Swarm Log Watch",
                body="No swarm logs found to monitor.",
                tags=["swarm", "monitor"],
                metrics={"targets": 0, "completed": False},
            )

        ps_lines = await self._ps_lines()
        process_entries = self._parse_swarm_pids(ps_lines)
        actions: list[str] = []
        issues: list[str] = []
        summaries: list[str] = []
        completed = True
        now = time.time()
        progress_window = int(self._config.get("progress_window_seconds", 300))
        stall_seconds = int(self._config.get("stall_seconds", 900))
        cooldown = int(self._config.get("restart_cooldown_seconds", 120))

        for target in targets:
            tokens = self._match_tokens(target)
            pids = [
                pid
                for pid, line in process_entries
                if any(token in line for token in tokens)
            ]
            running = bool(pids)
            log_exists = target.log_path.exists()
            log_age = None
            if log_exists:
                log_age = now - target.log_path.stat().st_mtime

            lines = await self._tail_log(target.log_path) if log_exists else []
            phase = self._detect_phase(lines)
            errors = self._detect_errors(lines)
            success = self._detect_success(lines)
            if (
                not running
                and not success
                and phase in {"synthesizing", "verifying"}
                and not errors
                and log_age is not None
                and log_age <= progress_window
            ):
                success = True

            recent = log_age is not None and log_age <= progress_window
            stalled = log_age is not None and log_age > stall_seconds and running

            status = "ok"
            if errors:
                status = "error"
            elif stalled:
                status = "stalled"
            elif not running and not success:
                status = "stopped"

            summaries.append(
                f"- {target.name}: {status}, phase={phase}, running={running}, log_age={int(log_age) if log_age is not None else 'n/a'}s"
            )

            healthy = (running and recent and not errors) or (success and not running)
            streak = self._update_healthy_streak(target.name, healthy)
            if not healthy:
                completed = False

            if status in {"error", "stalled", "stopped"}:
                issues.append(f"{target.name} is {status}")

                last_restarts = self._state.setdefault("restarts", {}).get(target.name, [])
                if last_restarts and now - last_restarts[-1] < cooldown:
                    actions.append(f"cooldown active for {target.name}")
                    continue

                if not self._config.get("auto_restart", True):
                    actions.append(f"restart recommended for {target.name}")
                    continue

                if not self._restart_allowed(target.name):
                    actions.append(f"restart limit hit for {target.name}")
                    continue

                action = await self._restart_target(target, pids)
                self._record_restart(target.name)
                actions.append(action)
                completed = False

            if healthy and streak < 2:
                completed = False

        self._save_state()

        body = "## Swarm Status\n" + "\n".join(summaries)
        if issues:
            body += "\n\n## Issues\n" + "\n".join(f"- {issue}" for issue in issues)
        if actions:
            body += "\n\n## Actions\n" + "\n".join(f"- {action}" for action in actions)

        await self.remember(
            content="; ".join(issues)[:500] if issues else "All swarms healthy",
            memory_type="error" if issues else "insight",
            context={"issues": issues, "actions": actions, "targets": len(targets)},
            importance=0.6 if issues else 0.3,
        )

        return LoopReport(
            title="Swarm Log Watch",
            body=body,
            tags=["swarm", "monitor"],
            metrics={
                "targets": len(targets),
                "issues": len(issues),
                "actions": len(actions),
                "completed": completed,
            },
        )
