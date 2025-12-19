from __future__ import annotations

import json
from pathlib import Path

from hafs.core.metacognition.monitor import MetacognitionMonitor
from hafs.models.metacognition import Strategy


def test_monitor_loads_camelcase_metacognition(tmp_path: Path) -> None:
    state_file = tmp_path / "metacognition.json"
    state_file.write_text(
        json.dumps(
            {
                "currentStrategy": "depth_first",
                "strategyEffectiveness": 0.8,
                "progressStatus": "making_progress",
                "spinDetection": {"similarActionCount": 0, "spinningThreshold": 4},
                "cognitiveLoad": {"current": 0.3, "itemsInFocus": 2},
                "helpSeeking": {"currentUncertainty": 0.0, "consecutiveFailures": 0},
                "flowState": False,
                "frustrationLevel": 0.2,
                "someFutureKey": {"x": 1},
            }
        ),
        encoding="utf-8",
    )

    monitor = MetacognitionMonitor(state_path=state_file)
    assert monitor.load_state() is True
    assert monitor.state.current_strategy == Strategy.DEPTH_FIRST
    assert monitor.state.cognitive_load.items_in_focus == 2

    # Ensure we preserve camelCase on save and keep unknown keys.
    assert monitor.save_state() is True
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert "currentStrategy" in saved
    assert "spinDetection" in saved
    assert "frustrationLevel" in saved
    assert "someFutureKey" in saved

