from __future__ import annotations

from pathlib import Path

from hafs.core.protocol.prompt_context import get_prompt_context


def test_get_prompt_context_none_when_missing(tmp_path: Path) -> None:
    assert get_prompt_context(tmp_path) is None


def test_get_prompt_context_includes_meta_and_goal(tmp_path: Path) -> None:
    ctx = tmp_path / ".context" / "scratchpad"
    ctx.mkdir(parents=True, exist_ok=True)

    (ctx / "metacognition.json").write_text(
        """
        {
          "current_strategy": "incremental",
          "progress_status": "making_progress",
          "flow_state": false,
          "spin_detection": { "similar_action_count": 0, "spinning_threshold": 4 },
          "cognitive_load": { "current": 0.25, "items_in_focus": 2 }
        }
        """.strip(),
        encoding="utf-8",
    )
    (ctx / "goals.json").write_text(
        """
        { "primary_goal": "Ship v1" }
        """.strip(),
        encoding="utf-8",
    )

    block = get_prompt_context(tmp_path)
    assert block is not None
    assert "<cognitive_state>" in block
    assert "## Metacognition" in block
    assert "Strategy: incremental" in block
    assert "Cognitive Load: 25%" in block
    assert "## Goals" in block
    assert "Primary: Ship v1" in block
    assert "</cognitive_state>" in block


def test_get_prompt_context_reads_camelcase_metacognition(tmp_path: Path) -> None:
    ctx = tmp_path / ".context" / "scratchpad"
    ctx.mkdir(parents=True, exist_ok=True)

    (ctx / "metacognition.json").write_text(
        """
        {
          "currentStrategy": "incremental",
          "progressStatus": "making_progress",
          "flowState": true,
          "spinDetection": { "similarActionCount": 0, "spinningThreshold": 4 },
          "cognitiveLoad": { "current": 0.5, "itemsInFocus": 3 }
        }
        """.strip(),
        encoding="utf-8",
    )

    block = get_prompt_context(tmp_path)
    assert block is not None
    assert "Strategy: incremental" in block
    assert "Cognitive Load: 50%" in block
    assert "Flow State: Active" in block


def test_get_prompt_context_reads_camelcase_goals(tmp_path: Path) -> None:
    ctx = tmp_path / ".context" / "scratchpad"
    ctx.mkdir(parents=True, exist_ok=True)

    (ctx / "goals.json").write_text(
        """
        {
          "primaryGoal": { "description": "Ship v2", "goalType": "primary" },
          "subgoals": [],
          "instrumentalGoals": [],
          "goalStack": [],
          "conflicts": [],
          "lastUpdated": "2025-01-01T00:00:00Z"
        }
        """.strip(),
        encoding="utf-8",
    )

    block = get_prompt_context(tmp_path)
    assert block is not None
    assert "Primary: Ship v2" in block


def test_get_prompt_context_includes_epistemic_summary(tmp_path: Path) -> None:
    ctx = tmp_path / ".context" / "scratchpad"
    ctx.mkdir(parents=True, exist_ok=True)

    (ctx / "epistemic.json").write_text(
        """
        {
          "goldenFacts": {},
          "workingFacts": { "a": { "confidence": 0.9 } },
          "assumptions": {},
          "unknowns": [ { "topic": "deployment target", "importance": "critical" } ],
          "contradictions": [],
          "settings": { "maxGoldenFacts": 10, "maxWorkingFacts": 100 }
        }
        """.strip(),
        encoding="utf-8",
    )

    block = get_prompt_context(tmp_path)
    assert block is not None
    assert "## Knowledge State" in block
    assert "deployment target" in block
