"""Core protocol file operations (deterministic, no LLM)."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from hafs.config.schema import HafsConfig
from hafs.core.afs.manager import AFSManager

ProtocolFileKind = Literal[
    "state",
    "deferred",
    "goals",
    "metacognition",
    "fears",
]


@dataclass(frozen=True)
class ProtocolFiles:
    project_root: Path

    @property
    def context_root(self) -> Path:
        return self.project_root / ".context"

    @property
    def state_md(self) -> Path:
        return self.context_root / "scratchpad" / "state.md"

    @property
    def deferred_md(self) -> Path:
        return self.context_root / "scratchpad" / "deferred.md"

    @property
    def goals_json(self) -> Path:
        return self.context_root / "scratchpad" / "goals.json"

    @property
    def metacognition_json(self) -> Path:
        return self.context_root / "scratchpad" / "metacognition.json"

    @property
    def fears_json(self) -> Path:
        return self.context_root / "memory" / "fears.json"


def ensure_protocol(project_root: Path, config: HafsConfig | None = None) -> ProtocolFiles:
    """Ensure `.context/` protocol artifacts exist for a project root."""
    config = config or HafsConfig()
    AFSManager(config).ensure(project_root)
    return ProtocolFiles(project_root=project_root.resolve())


def open_protocol_file(project_root: Path, kind: ProtocolFileKind) -> Path:
    files = ensure_protocol(project_root)
    mapping: dict[ProtocolFileKind, Path] = {
        "state": files.state_md,
        "deferred": files.deferred_md,
        "goals": files.goals_json,
        "metacognition": files.metacognition_json,
        "fears": files.fears_json,
    }
    return mapping[kind]


def set_primary_goal(project_root: Path, goal: str) -> Path:
    """Set the primary goal in `goals.json`."""
    files = ensure_protocol(project_root)
    goal = " ".join(goal.strip().splitlines()).strip()
    if not goal:
        return files.goals_json

    try:
        raw: dict[str, Any] = json.loads(files.goals_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except Exception:
        raw = {}

    # Preserve oracle-code schema if present, otherwise use HAFS goal manager.
    try:
        from hafs.core.protocol.goals_compat import detect_wire_format, set_primary_goal_inplace

        wire = detect_wire_format(raw)
        updated = set_primary_goal_inplace(raw, description=goal, wire=wire)
        files.goals_json.write_text(
            json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return files.goals_json
    except Exception:
        pass

    try:
        from hafs.core.goals.manager import GoalManager

        manager = GoalManager(state_path=files.goals_json)
        manager.load_state()
        manager.set_primary_goal(goal)
        manager.save_state()
    except Exception:
        files.goals_json.write_text(
            json.dumps({"primary_goal": goal}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return files.goals_json


def append_deferred(project_root: Path, item: str) -> Path:
    """Append an item to `deferred.md` with a timestamp."""
    files = ensure_protocol(project_root)
    item = " ".join(item.strip().splitlines()).strip()
    if not item:
        return files.deferred_md

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"- [{stamp}] {item}"
    try:
        existing = files.deferred_md.read_text(encoding="utf-8", errors="replace").rstrip()
    except OSError:
        existing = "# Deferred"
    content = (existing + "\n" + line + "\n") if existing else (line + "\n")
    files.deferred_md.write_text(content, encoding="utf-8")
    return files.deferred_md


def snapshot_state(project_root: Path, reason: str | None = None) -> Path | None:
    """Copy `state.md` into `history/` for auditing."""
    files = ensure_protocol(project_root)
    if not files.state_md.exists():
        return None

    history_dir = files.context_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = ""
    if reason:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", reason.strip().lower()).strip("-")
        if cleaned:
            suffix = f"__{cleaned[:48]}"

    dest = history_dir / f"state__{timestamp}{suffix}.md"
    try:
        shutil.copyfile(files.state_md, dest)
    except OSError:
        return None
    return dest
