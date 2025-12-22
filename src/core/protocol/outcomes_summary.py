"""Summaries for task-outcomes.json (optional efficiency hints)."""

from __future__ import annotations

from typing import Any


def summarize_task_outcomes(data: dict[str, Any], *, window: int = 10) -> dict[str, Any]:
    outcomes = data.get("outcomes")
    if not isinstance(outcomes, list):
        return {"count": 0}

    recent = [o for o in outcomes if isinstance(o, dict)][-window:]
    if not recent:
        return {"count": 0}

    successes = 0
    durations: list[int] = []
    errors = 0
    for o in recent:
        if o.get("success") is True:
            successes += 1
        dur = o.get("duration")
        if isinstance(dur, int):
            durations.append(dur)
        err = o.get("errorCount")
        if isinstance(err, int):
            errors += err

    avg_ms = int(sum(durations) / len(durations)) if durations else 0
    return {
        "count": len(recent),
        "success_rate": round(successes / len(recent), 2),
        "avg_duration_ms": avg_ms,
        "total_errors": errors,
    }


def render_outcomes_prompt_section(summary: dict[str, Any]) -> list[str]:
    if int(summary.get("count", 0) or 0) <= 0:
        return []

    lines = ["## Task Outcomes"]
    lines.append(f"- Recent tasks: {summary['count']}")
    lines.append(f"- Success rate: {int(round(summary['success_rate'] * 100))}%")
    if int(summary.get("avg_duration_ms", 0) or 0):
        lines.append(f"- Avg duration: {summary['avg_duration_ms']}ms")
    if int(summary.get("total_errors", 0) or 0):
        lines.append(f"- Tool errors: {summary['total_errors']}")
    return lines

