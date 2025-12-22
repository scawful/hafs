"""AFS sync service for multi-node context replication."""

from __future__ import annotations

import logging
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - py311+
    import tomli as tomllib

from core.execution import ExecutionPolicy
from core.history.logger import HistoryLogger
from core.history.models import OperationType
from core.nodes import NodeManager, node_manager
from core.tooling import ToolRunner

logger = logging.getLogger(__name__)


@dataclass
class SyncTarget:
    """Sync destination or source target."""

    path: str
    node: Optional[str] = None
    host: Optional[str] = None
    user: Optional[str] = None
    port: Optional[int] = None
    direction: Optional[str] = None  # push | pull

    def label(self) -> str:
        if self.node:
            return f"{self.node}:{self.path}"
        if self.host:
            return f"{self.host}:{self.path}"
        return self.path


@dataclass
class SyncProfile:
    """Sync profile definition."""

    name: str
    source: str
    scope: str = "global"  # global | user | project
    direction: str = "push"  # push | pull | bidirectional
    transport: str = "rsync"
    exclude: list[str] = field(default_factory=list)
    delete: bool = False
    project: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    targets: list[SyncTarget] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SyncProfile":
        raw_targets = data.get("targets", []) or []
        targets = [SyncTarget(**t) for t in raw_targets]
        return cls(
            name=data.get("name", ""),
            source=data.get("source", ""),
            scope=data.get("scope", "global"),
            direction=data.get("direction", "push"),
            transport=data.get("transport", "rsync"),
            exclude=list(data.get("exclude", []) or []),
            delete=bool(data.get("delete", False)),
            project=data.get("project"),
            tags=list(data.get("tags", []) or []),
            enabled=bool(data.get("enabled", True)),
            targets=targets,
        )


@dataclass
class SyncResult:
    """Result for a single sync operation."""

    profile: str
    target: str
    direction: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class SyncRegistry:
    """Load sync profiles from config."""

    CONFIG_PATHS = [
        Path.home() / ".config" / "hafs" / "sync.toml",
        Path.home() / ".hafs" / "sync.toml",
        Path("sync.toml"),
    ]

    def __init__(self) -> None:
        self._profiles: dict[str, SyncProfile] = {}

    def load(self, config_path: Optional[Path] = None) -> list[SyncProfile]:
        path = None
        if config_path and config_path.exists():
            path = config_path
        else:
            for candidate in self.CONFIG_PATHS:
                if candidate.exists():
                    path = candidate
                    break

        if not path:
            return []

        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except Exception as exc:
            logger.warning("Failed to load sync config: %s", exc)
            return []

        profiles = []
        for profile_data in data.get("profiles", []):
            profile = SyncProfile.from_dict(profile_data)
            if profile.name:
                self._profiles[profile.name] = profile
                profiles.append(profile)

        return profiles

    def list(self) -> list[SyncProfile]:
        return list(self._profiles.values())

    def get(self, name: str) -> Optional[SyncProfile]:
        return self._profiles.get(name)


class AFSSyncService:
    """AFS sync runner with safe execution policy."""

    def __init__(
        self,
        registry: Optional[SyncRegistry] = None,
        nodes: Optional[NodeManager] = None,
        execution_mode: str = "infra_ops",
    ) -> None:
        self._registry = registry or SyncRegistry()
        self._nodes = nodes or node_manager
        self._execution_mode = execution_mode
        self._status_dir = Path.home() / ".context" / "metrics"
        self._status_file = self._status_dir / "afs_sync_status.json"
        self._history_logger = HistoryLogger(
            Path.home() / ".context" / "history",
            project_id="afs_sync",
        )

    async def load(self) -> list[SyncProfile]:
        await self._nodes.load_config()
        return self._registry.load()

    def list_profiles(self) -> list[SyncProfile]:
        return self._registry.list()

    def resolve_profile(self, name: str) -> Optional[SyncProfile]:
        return self._registry.get(name)

    async def run_profile(
        self,
        name: str,
        direction_override: Optional[str] = None,
        dry_run: bool = False,
    ) -> list[SyncResult]:
        profile = self._registry.get(name)
        if not profile:
            raise ValueError(f"Unknown profile: {name}")
        return await self._run_profile(profile, direction_override, dry_run)

    async def _run_profile(
        self,
        profile: SyncProfile,
        direction_override: Optional[str],
        dry_run: bool,
    ) -> list[SyncResult]:
        if not profile.enabled:
            return []

        source_root = Path(profile.source).expanduser().resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Sync source not found: {source_root}")

        policy = ExecutionPolicy(execution_mode=self._execution_mode)
        tool_profile = policy.resolve_tool_profile(None)
        runner = ToolRunner(root=source_root, profile=tool_profile)

        results: list[SyncResult] = []
        direction = direction_override or profile.direction

        for target in profile.targets:
            if direction == "bidirectional":
                results.extend(
                    await self._sync_target(
                        profile,
                        target,
                        "push",
                        runner,
                        dry_run,
                        update_only=True,
                    )
                )
                results.extend(
                    await self._sync_target(
                        profile,
                        target,
                        "pull",
                        runner,
                        dry_run,
                        update_only=True,
                    )
                )
            else:
                results.extend(
                    await self._sync_target(
                        profile,
                        target,
                        direction,
                        runner,
                        dry_run,
                        update_only=False,
                    )
                )

        return results

    async def _sync_target(
        self,
        profile: SyncProfile,
        target: SyncTarget,
        direction: str,
        runner: ToolRunner,
        dry_run: bool,
        update_only: bool,
    ) -> list[SyncResult]:
        resolved_direction = target.direction or direction
        if resolved_direction not in {"push", "pull"}:
            raise ValueError(f"Unsupported sync direction: {resolved_direction}")

        remote_spec, ssh_args = self._resolve_remote(target)
        local_path = self._format_path(Path(profile.source).expanduser().resolve())

        if resolved_direction == "push":
            source = local_path
            dest = remote_spec
        else:
            source = remote_spec
            dest = local_path

        args = []
        if dry_run:
            args.append("--dry-run")
        if profile.delete:
            args.append("--delete")
        if update_only:
            args.append("--update")
        for pattern in profile.exclude:
            args.extend(["--exclude", pattern])
        if ssh_args:
            args.extend(["-e", ssh_args])

        args.extend([source, dest])
        start_time = time.time()
        result = await runner.run("rsync", args=args)
        duration_ms = int((time.time() - start_time) * 1000)

        sync_result = SyncResult(
            profile=profile.name,
            target=target.label(),
            direction=resolved_direction,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        self._record_result(profile, sync_result, duration_ms, dry_run)

        return [sync_result]

    def _resolve_remote(self, target: SyncTarget) -> tuple[str, Optional[str]]:
        host = target.host
        if not host and target.node:
            node = self._nodes.get_node(target.node)
            if node:
                host = node.host

        if not host:
            raise ValueError(f"No host resolved for target: {target.label()}")

        user_prefix = f"{target.user}@" if target.user else ""
        remote = f"{user_prefix}{host}:{target.path}"

        ssh_args = None
        if target.port:
            ssh_args = f"ssh -p {target.port}"

        return remote, ssh_args

    @staticmethod
    def _format_path(path: Path) -> str:
        path_str = str(path)
        if path_str.endswith("/"):
            return path_str
        return f"{path_str}/"

    def _load_status(self) -> dict[str, Any]:
        if not self._status_file.exists():
            return {"profiles": {}}
        try:
            return json.loads(self._status_file.read_text())
        except Exception:
            return {"profiles": {}}

    def _save_status(self, data: dict[str, Any]) -> None:
        self._status_dir.mkdir(parents=True, exist_ok=True)
        self._status_file.write_text(json.dumps(data, indent=2))

    def _record_result(
        self,
        profile: SyncProfile,
        result: SyncResult,
        duration_ms: int,
        dry_run: bool,
    ) -> None:
        timestamp = datetime.now().isoformat()
        data = self._load_status()
        profile_entry = data.setdefault("profiles", {}).setdefault(profile.name, {})
        targets = profile_entry.setdefault("targets", {})

        targets[result.target] = {
            "timestamp": timestamp,
            "direction": result.direction,
            "exit_code": result.exit_code,
            "ok": result.ok,
            "duration_ms": duration_ms,
            "dry_run": dry_run,
            "stderr": result.stderr[:500],
            "stdout": result.stdout[:500],
        }
        profile_entry["last_run"] = timestamp
        profile_entry["direction"] = profile.direction
        profile_entry["scope"] = profile.scope
        data["updated_at"] = timestamp
        self._save_status(data)

        self._history_logger.log(
            operation_type=OperationType.SYSTEM_EVENT,
            name="afs_sync",
            input_data={
                "profile": profile.name,
                "target": result.target,
                "direction": result.direction,
                "dry_run": dry_run,
            },
            output={
                "exit_code": result.exit_code,
                "duration_ms": duration_ms,
            },
            success=result.ok,
            error=result.stderr.strip() if not result.ok else None,
            tags=["afs", "sync"],
        )
