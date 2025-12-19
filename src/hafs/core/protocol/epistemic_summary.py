"""Summaries for oracle-style epistemic.json."""

from __future__ import annotations

from typing import Any


def summarize_epistemic(data: dict[str, Any]) -> dict[str, Any]:
    golden = data.get("goldenFacts") if isinstance(data.get("goldenFacts"), dict) else {}
    working = data.get("workingFacts") if isinstance(data.get("workingFacts"), dict) else {}
    assumptions = data.get("assumptions") if isinstance(data.get("assumptions"), dict) else {}
    unknowns = data.get("unknowns") if isinstance(data.get("unknowns"), list) else []
    contradictions = data.get("contradictions") if isinstance(data.get("contradictions"), list) else []
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}

    max_golden = settings.get("maxGoldenFacts", 10)
    max_working = settings.get("maxWorkingFacts", 100)

    confidences: list[float] = []
    for v in working.values():
        if isinstance(v, dict):
            c = v.get("confidence")
            if isinstance(c, (int, float)):
                confidences.append(float(c))
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0

    critical_unknowns: list[str] = []
    for u in unknowns:
        if not isinstance(u, dict):
            continue
        importance = str(u.get("importance", "")).lower()
        topic = u.get("topic")
        if not isinstance(topic, str) or not topic.strip():
            continue
        if importance == "critical":
            critical_unknowns.append(topic.strip())

    return {
        "golden_count": len(golden),
        "max_golden": int(max_golden) if isinstance(max_golden, int) else 10,
        "working_count": len(working),
        "max_working": int(max_working) if isinstance(max_working, int) else 100,
        "avg_confidence_pct": int(round(avg_conf * 100)),
        "assumption_count": len(assumptions),
        "unknown_count": len(unknowns),
        "critical_unknowns": critical_unknowns[:3],
        "contradiction_count": len([c for c in contradictions if isinstance(c, dict)]),
    }


def render_epistemic_prompt_section(summary: dict[str, Any]) -> list[str]:
    lines: list[str] = ["## Knowledge State"]
    lines.append(
        f"- Golden Facts: {summary.get('golden_count', 0)}/{summary.get('max_golden', 10)}"
    )
    lines.append(
        f"- Working Facts: {summary.get('working_count', 0)}/{summary.get('max_working', 100)} "
        f"({summary.get('avg_confidence_pct', 0)}% confident)"
    )
    unknown_count = int(summary.get("unknown_count", 0) or 0)
    if unknown_count:
        lines.append(f"- Unknowns: {unknown_count}")
    if int(summary.get("assumption_count", 0) or 0):
        lines.append(f"- Assumptions: {summary.get('assumption_count')}")
    if int(summary.get("contradiction_count", 0) or 0):
        lines.append(f"- WARNING: {summary.get('contradiction_count')} contradiction(s)")

    critical = summary.get("critical_unknowns") or []
    if isinstance(critical, list) and critical:
        lines.append("- WARNING: critical unknown(s)")
        for topic in critical:
            lines.append(f"  - {topic}")
    return lines

