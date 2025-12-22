from __future__ import annotations

from core.protocol.epistemic_summary import render_epistemic_prompt_section, summarize_epistemic


def test_summarize_epistemic_and_render() -> None:
    data = {
        "goldenFacts": {},
        "workingFacts": {"a": {"confidence": 0.8}, "b": {"confidence": 0.6}},
        "assumptions": {"x": {}},
        "unknowns": [{"topic": "preferred editor", "importance": "critical"}],
        "contradictions": [],
        "settings": {"maxGoldenFacts": 10, "maxWorkingFacts": 100},
    }
    summary = summarize_epistemic(data)
    assert summary["working_count"] == 2
    assert summary["assumption_count"] == 1
    assert summary["critical_unknowns"] == ["preferred editor"]

    lines = render_epistemic_prompt_section(summary)
    joined = "\n".join(lines)
    assert "## Knowledge State" in joined
    assert "Working Facts: 2/100" in joined
    assert "critical unknown(s)" in joined

