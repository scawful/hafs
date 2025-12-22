"""Cognitive protocol linting utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from config.schema import HafsConfig
from core.afs.mapping import resolve_directory_map
from core.protocol.goals_compat import detect_wire_format as detect_goals_wire
from core.protocol.metacognition_compat import (
    detect_wire_format as detect_meta_wire,
    normalize_metacognition,
)
from models.afs import MountType, ProjectMetadata
from models.goals import GoalHierarchy
from models.metacognition import MetacognitiveState

LintSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class LintIssue:
    severity: LintSeverity
    code: str
    message: str
    path: Path | None = None
    hint: str | None = None


@dataclass
class ProtocolLintReport:
    context_root: Path
    issues: list[LintIssue] = field(default_factory=list)

    def add(
        self,
        severity: LintSeverity,
        code: str,
        message: str,
        *,
        path: Path | None = None,
        hint: str | None = None,
    ) -> None:
        self.issues.append(LintIssue(severity, code, message, path, hint))

    @property
    def errors(self) -> list[LintIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[LintIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_clean(self) -> bool:
        return not self.errors


_STATE_SECTIONS = [
    "## 1. Current Context",
    "## 2. Theory of Mind",
    "## 3. Deliberation & Intent",
    "## 4. Action Outcome",
    "## 5. Emotional State & Risk Assessment",
    "## 6. Metacognitive Assessment",
]

_STATE_PLACEHOLDER_MARKERS = [
    "[Copy of the latest user prompt]",
    "[Brief summary of relevant past interactions",
    "[Key constraints or facts from `memory` or `knowledge`]",
    "[Inferred intent of the user]",
    "[What can I assume the user knows or sees?]",
    "[How might the user react to my proposed action?]",
    "[Option A: Pros/Cons]",
    "[Description of the action to be taken]",
    "[Why this action was chosen over others]",
    "[What this action is expected to achieve]",
    "[To be filled in after the action is executed",
    "[Immediate follow-up actions, if any]",
    "[List of potential negative outcomes",
    "[How to address the concerns",
]

_STAGE_ORDER = [
    "perception",
    "contextualization",
    "emotional_modeling",
    "deliberation",
    "action",
    "reflection",
]

_STAGE_ALIASES = {
    "context": "contextualization",
    "contextualisation": "contextualization",
    "emotion": "emotional_modeling",
    "emotional": "emotional_modeling",
    "emotional_model": "emotional_modeling",
    "post_action": "reflection",
}


def lint_protocol(context_root: Path, config: HafsConfig) -> ProtocolLintReport:
    report = ProtocolLintReport(context_root=context_root)

    if not context_root.exists() or not context_root.is_dir():
        report.add(
            "error",
            "context_missing",
            "Context root does not exist",
            path=context_root,
            hint="Run `hafs afs init` or point lint at an existing .context",
        )
        return report

    metadata = _load_metadata(context_root, report)
    directory_map = resolve_directory_map(
        afs_directories=config.afs_directories,
        metadata=metadata,
    )
    dir_names = {mt: directory_map.get(mt, mt.value) for mt in MountType}

    for mount_type, dir_name in dir_names.items():
        if not (context_root / dir_name).is_dir():
            report.add(
                "error",
                "directory_missing",
                f"Missing AFS directory for {mount_type.value}",
                path=context_root / dir_name,
            )

    scratchpad_dir = context_root / dir_names[MountType.SCRATCHPAD]
    memory_dir = context_root / dir_names[MountType.MEMORY]
    history_dir = context_root / dir_names[MountType.HISTORY]

    _lint_required_files(report, scratchpad_dir, memory_dir, history_dir)

    _lint_state_md(report, scratchpad_dir / "state.md")
    _lint_state_json(report, scratchpad_dir / "state.json")
    _lint_metacognition(report, scratchpad_dir / "metacognition.json")
    _lint_goals(report, scratchpad_dir / "goals.json")
    _lint_emotions(report, scratchpad_dir / "emotions.json")
    _lint_epistemic(report, scratchpad_dir / "epistemic.json")
    _lint_analysis_triggers(report, scratchpad_dir / "analysis-triggers.json")
    _lint_fears(report, memory_dir / "fears.json")
    _lint_sessions(report, history_dir / "sessions")

    return report


def _load_metadata(context_root: Path, report: ProtocolLintReport) -> ProjectMetadata | None:
    metadata_path = context_root / "metadata.json"
    if not metadata_path.exists():
        report.add(
            "warning",
            "metadata_missing",
            "metadata.json is missing; directory mapping may be inferred",
            path=metadata_path,
        )
        return None

    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(
            "error",
            "metadata_invalid",
            f"metadata.json could not be parsed: {exc}",
            path=metadata_path,
        )
        return None

    try:
        return ProjectMetadata(**raw)
    except Exception:
        cleaned = dict(raw)
        if not cleaned.get("created_at"):
            cleaned.pop("created_at", None)
        try:
            return ProjectMetadata(**cleaned)
        except Exception as exc:
            report.add(
                "error",
                "metadata_invalid",
                f"metadata.json failed schema validation: {exc}",
                path=metadata_path,
            )
            return None


def _lint_required_files(
    report: ProtocolLintReport,
    scratchpad_dir: Path,
    memory_dir: Path,
    history_dir: Path,
) -> None:
    required_files = [
        (scratchpad_dir / "state.md", "state.md"),
        (scratchpad_dir / "state.json", "state.json"),
        (scratchpad_dir / "deferred.md", "deferred.md"),
        (scratchpad_dir / "metacognition.json", "metacognition.json"),
        (scratchpad_dir / "goals.json", "goals.json"),
        (scratchpad_dir / "emotions.json", "emotions.json"),
        (scratchpad_dir / "epistemic.json", "epistemic.json"),
        (scratchpad_dir / "analysis-triggers.json", "analysis-triggers.json"),
        (memory_dir / "fears.json", "fears.json"),
    ]

    for path, label in required_files:
        if not path.exists():
            report.add(
                "error",
                "protocol_file_missing",
                f"Required protocol file is missing: {label}",
                path=path,
                hint="Run `hafs afs init` or regenerate the protocol scaffold",
            )

    if history_dir.exists():
        sessions_dir = history_dir / "sessions"
        if not sessions_dir.exists():
            report.add(
                "warning",
                "sessions_missing",
                "history/sessions directory is missing; session lifecycle checks skipped",
                path=sessions_dir,
            )


def _lint_state_md(report: ProtocolLintReport, path: Path) -> None:
    if not path.exists():
        return

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.add(
            "error",
            "state_md_unreadable",
            f"state.md could not be read: {exc}",
            path=path,
        )
        return

    missing_sections = [section for section in _STATE_SECTIONS if section not in content]
    if missing_sections:
        report.add(
            "error",
            "state_md_sections_missing",
            "state.md is missing required protocol sections",
            path=path,
            hint=", ".join(missing_sections),
        )

    placeholders = [marker for marker in _STATE_PLACEHOLDER_MARKERS if marker in content]
    if placeholders:
        report.add(
            "warning",
            "state_md_placeholders",
            "state.md still contains template placeholders",
            path=path,
            hint=f"{len(placeholders)} placeholders found",
        )


def _lint_state_json(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return

    if not isinstance(data, dict):
        report.add(
            "error",
            "state_json_invalid",
            "state.json must contain a JSON object",
            path=path,
        )
        return

    schema_version = data.get("schema_version")
    if not isinstance(schema_version, str):
        report.add(
            "error",
            "state_json_schema_version",
            "state.json must include schema_version",
            path=path,
        )
    elif schema_version != "0.3":
        report.add(
            "warning",
            "state_json_schema_version",
            f"Unexpected schema_version '{schema_version}'",
            path=path,
            hint="Expected 0.3",
        )

    producer = data.get("producer")
    if not isinstance(producer, dict) or not producer.get("name"):
        report.add(
            "error",
            "state_json_producer",
            "state.json must include producer.name",
            path=path,
        )

    last_updated = data.get("last_updated")
    if not isinstance(last_updated, str):
        report.add(
            "warning",
            "state_json_last_updated",
            "state.json should include last_updated timestamp",
            path=path,
        )

    entries = data.get("entries")
    if not isinstance(entries, list):
        report.add(
            "error",
            "state_json_entries",
            "state.json must include entries list",
            path=path,
        )
        return

    _lint_stage_transitions(report, entries, path)


def _lint_metacognition(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return

    if not isinstance(data, dict):
        report.add(
            "error",
            "metacognition_invalid",
            "metacognition.json must contain a JSON object",
            path=path,
        )
        return

    wire = detect_meta_wire(data)
    normalized = normalize_metacognition(data) if wire == "camel" else data

    try:
        MetacognitiveState.model_validate(normalized)
    except Exception as exc:
        report.add(
            "error",
            "metacognition_schema",
            f"metacognition.json failed schema validation: {exc}",
            path=path,
        )
        return

    if wire == "camel":
        report.add(
            "warning",
            "metacognition_wire_format",
            "metacognition.json uses legacy camelCase keys",
            path=path,
            hint="Consider migrating to snake_case for HAFS",
        )


def _lint_goals(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return

    if not isinstance(data, dict):
        report.add(
            "error",
            "goals_invalid",
            "goals.json must contain a JSON object",
            path=path,
        )
        return

    wire = detect_goals_wire(data)
    normalized = _normalize_goals(data) if wire == "camel" else data

    try:
        hierarchy = GoalHierarchy.model_validate(normalized)
    except Exception as exc:
        report.add(
            "error",
            "goals_schema",
            f"goals.json failed schema validation: {exc}",
            path=path,
        )
        return

    if wire == "camel":
        report.add(
            "warning",
            "goals_wire_format",
            "goals.json uses legacy camelCase keys",
            path=path,
            hint="Consider migrating to snake_case for HAFS",
        )

    _lint_goal_hierarchy(report, hierarchy, path)


def _lint_goal_hierarchy(
    report: ProtocolLintReport, hierarchy: GoalHierarchy, path: Path
) -> None:
    goal_ids = {goal.id for goal in hierarchy.all_goals}
    if hierarchy.goal_stack and not goal_ids:
        report.add(
            "warning",
            "goal_stack_orphaned",
            "goal_stack has entries but no goals exist",
            path=path,
        )

    for goal_id in hierarchy.goal_stack:
        if goal_id not in goal_ids:
            report.add(
                "warning",
                "goal_stack_invalid",
                f"goal_stack references unknown goal id '{goal_id}'",
                path=path,
            )

    if hierarchy.subgoals and not hierarchy.primary_goal:
        report.add(
            "warning",
            "primary_goal_missing",
            "subgoals exist but primary_goal is missing",
            path=path,
        )


def _lint_emotions(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return
    if not isinstance(data, dict):
        report.add(
            "warning",
            "emotions_invalid",
            "emotions.json should contain a JSON object",
            path=path,
        )


def _lint_epistemic(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return
    if not isinstance(data, dict):
        report.add(
            "warning",
            "epistemic_invalid",
            "epistemic.json should contain a JSON object",
            path=path,
        )


def _lint_analysis_triggers(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return
    if not isinstance(data, (dict, list)):
        report.add(
            "warning",
            "analysis_triggers_invalid",
            "analysis-triggers.json should contain a JSON object or array",
            path=path,
        )


def _lint_fears(report: ProtocolLintReport, path: Path) -> None:
    data = _read_json(report, path)
    if data is None:
        return
    if not isinstance(data, list):
        report.add(
            "warning",
            "fears_invalid",
            "fears.json should contain a JSON array",
            path=path,
        )
        return

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            report.add(
                "warning",
                "fears_entry_invalid",
                f"fears.json entry {idx} is not an object",
                path=path,
            )
            continue
        if not item.get("trigger") or not item.get("mitigation"):
            report.add(
                "warning",
                "fears_entry_fields",
                f"fears.json entry {idx} should include trigger and mitigation",
                path=path,
            )


def _lint_sessions(report: ProtocolLintReport, sessions_dir: Path) -> None:
    if not sessions_dir.exists():
        return

    for session_file in sorted(sessions_dir.glob("*.json")):
        data = _read_json(report, session_file)
        if data is None:
            continue
        if not isinstance(data, dict):
            report.add(
                "warning",
                "session_invalid",
                "Session file should contain a JSON object",
                path=session_file,
            )
            continue
        status = data.get("status")
        if status not in {"active", "suspended", "completed", "aborted"}:
            report.add(
                "warning",
                "session_status_invalid",
                f"Session status '{status}' is not a valid lifecycle state",
                path=session_file,
            )


def _lint_stage_transitions(
    report: ProtocolLintReport, entries: list[Any], path: Path
) -> None:
    stages: list[tuple[str, int]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        stage = entry.get("stage") or entry.get("phase") or entry.get("step")
        if not stage or not isinstance(stage, str):
            continue
        normalized = _normalize_stage(stage)
        if normalized not in _STAGE_ORDER:
            report.add(
                "warning",
                "stage_unknown",
                f"Unknown protocol stage '{stage}' in state.json",
                path=path,
            )
            continue
        stages.append((normalized, idx))

    if not stages:
        return

    order_index = {stage: i for i, stage in enumerate(_STAGE_ORDER)}
    last_stage = stages[0][0]
    last_index = order_index[last_stage]
    for stage, _ in stages[1:]:
        current_index = order_index[stage]
        if current_index < last_index and stage != _STAGE_ORDER[0]:
            report.add(
                "warning",
                "stage_transition",
                f"Stage transition from '{last_stage}' to '{stage}' breaks protocol order",
                path=path,
            )
        if stage == _STAGE_ORDER[0] and current_index < last_index:
            last_index = current_index
            last_stage = stage
            continue
        last_index = current_index
        last_stage = stage


def _normalize_stage(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return _STAGE_ALIASES.get(cleaned, cleaned)


def _read_json(report: ProtocolLintReport, path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(
            "error",
            "json_parse_error",
            f"Failed to parse JSON: {exc}",
            path=path,
        )
        return None


def _normalize_goals(data: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "primaryGoal": "primary_goal",
        "instrumentalGoals": "instrumental_goals",
        "goalStack": "goal_stack",
        "lastUpdated": "last_updated",
    }
    goal_map = {
        "goalType": "goal_type",
        "createdAt": "created_at",
        "updatedAt": "updated_at",
        "completedAt": "completed_at",
        "userStated": "user_stated",
        "successCriteria": "success_criteria",
        "estimatedEffort": "estimated_effort",
        "parentId": "parent_id",
    }
    conflict_map = {
        "goalAId": "goal_a_id",
        "goalBId": "goal_b_id",
        "conflictType": "conflict_type",
        "detectedAt": "detected_at",
    }

    normalized: dict[str, Any] = {}
    for key, value in data.items():
        normalized[mapping.get(key, key)] = value

    def normalize_goal(goal: Any) -> Any:
        if not isinstance(goal, dict):
            return goal
        return {goal_map.get(k, k): v for k, v in goal.items()}

    def normalize_conflict(conflict: Any) -> Any:
        if not isinstance(conflict, dict):
            return conflict
        return {conflict_map.get(k, k): v for k, v in conflict.items()}

    if "primary_goal" in normalized:
        normalized["primary_goal"] = normalize_goal(normalized["primary_goal"])

    for list_key in ("subgoals", "instrumental_goals"):
        if list_key in normalized and isinstance(normalized[list_key], list):
            normalized[list_key] = [normalize_goal(item) for item in normalized[list_key]]

    if "conflicts" in normalized and isinstance(normalized["conflicts"], list):
        normalized["conflicts"] = [
            normalize_conflict(item) for item in normalized["conflicts"]
        ]

    return normalized
