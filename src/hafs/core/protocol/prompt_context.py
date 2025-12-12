"""Build deterministic cognitive context for prompt injection.

This is inspired by oracle-code's CognitiveIntegration.getPromptContext(), but
kept intentionally lightweight and file-based for HAFS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_prompt_context(project_root: Path) -> str | None:
    """Return a `<cognitive_state>` block or None if nothing is available."""
    context_root = project_root / ".context"
    meta_path = context_root / "scratchpad" / "metacognition.json"
    goals_path = context_root / "scratchpad" / "goals.json"
    epistemic_path = context_root / "scratchpad" / "epistemic.json"
    outcomes_path = context_root / "scratchpad" / "metrics" / "task-outcomes.json"

    meta: dict[str, Any] | None = None
    goals: dict[str, Any] | None = None
    epistemic: dict[str, Any] | None = None
    outcomes: dict[str, Any] | None = None

    try:
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = None

    try:
        if goals_path.exists():
            goals = json.loads(goals_path.read_text(encoding="utf-8"))
    except Exception:
        goals = None

    try:
        if epistemic_path.exists():
            epistemic = json.loads(epistemic_path.read_text(encoding="utf-8"))
    except Exception:
        epistemic = None

    try:
        if outcomes_path.exists():
            outcomes = json.loads(outcomes_path.read_text(encoding="utf-8"))
    except Exception:
        outcomes = None

    if not meta and not goals:
        if not epistemic and not outcomes:
            return None

    lines: list[str] = ["<cognitive_state>"]

    if meta:
        try:
            from hafs.core.protocol.metacognition_compat import normalize_metacognition

            meta = normalize_metacognition(meta)
        except Exception:
            pass
        lines.append("## Metacognition")
        strategy = str(meta.get("current_strategy", "")).strip() or "unknown"
        progress = str(meta.get("progress_status", "")).strip() or "unknown"
        flow = bool(meta.get("flow_state", False))
        load = meta.get("cognitive_load", {}) or {}
        load_pct = load.get("current", None)
        items = load.get("items_in_focus", None)

        lines.append(f"- Strategy: {strategy}")
        lines.append(f"- Progress: {progress}")
        if isinstance(load_pct, (int, float)):
            lines.append(f"- Cognitive Load: {int(round(load_pct * 100))}%")
        if isinstance(items, int):
            lines.append(f"- Items in Focus: {items}")
        lines.append(f"- Flow State: {'Active' if flow else 'Inactive'}")

        spin = meta.get("spin_detection", {}) or {}
        similar = spin.get("similar_action_count", 0)
        threshold = spin.get("spinning_threshold", 0)
        if isinstance(similar, int) and isinstance(threshold, int) and threshold and similar >= threshold:
            lines.append("- WARNING: Spinning detected")

    primary_goal = None
    if goals:
        try:
            from hafs.core.protocol.goals_compat import extract_primary_goal_text

            primary_goal = extract_primary_goal_text(goals)
        except Exception:
            primary_goal = goals.get("primary_goal")  # type: ignore[assignment]

    if primary_goal:
        lines.append("")
        lines.append("## Goals")
        lines.append(f"- Primary: {str(primary_goal).strip()}")

    if isinstance(epistemic, dict) and epistemic:
        try:
            from hafs.core.protocol.epistemic_summary import (
                render_epistemic_prompt_section,
                summarize_epistemic,
            )

            summary = summarize_epistemic(epistemic)
            lines.append("")
            lines.extend(render_epistemic_prompt_section(summary))
        except Exception:
            pass

    if isinstance(outcomes, dict) and outcomes:
        try:
            from hafs.core.protocol.outcomes_summary import (
                render_outcomes_prompt_section,
                summarize_task_outcomes,
            )

            summary = summarize_task_outcomes(outcomes, window=10)
            section = render_outcomes_prompt_section(summary)
            if section:
                lines.append("")
                lines.extend(section)
        except Exception:
            pass

    lines.append("</cognitive_state>")
    return "\n".join(lines)
