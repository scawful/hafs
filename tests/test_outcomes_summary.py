from __future__ import annotations

from hafs.core.protocol.outcomes_summary import summarize_task_outcomes


def test_summarize_task_outcomes_window() -> None:
    data = {
        "outcomes": [
            {"success": True, "duration": 100, "errorCount": 0},
            {"success": False, "duration": 200, "errorCount": 2},
        ]
    }
    summary = summarize_task_outcomes(data, window=10)
    assert summary["count"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["avg_duration_ms"] == 150
    assert summary["total_errors"] == 2

