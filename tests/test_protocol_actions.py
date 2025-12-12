from __future__ import annotations

import json
from pathlib import Path

from hafs.core.protocol.actions import (
    append_deferred,
    open_protocol_file,
    set_primary_goal,
    snapshot_state,
)


def test_open_protocol_file_paths_exist(tmp_path: Path) -> None:
    state = open_protocol_file(tmp_path, "state")
    goals = open_protocol_file(tmp_path, "goals")
    deferred = open_protocol_file(tmp_path, "deferred")
    fears = open_protocol_file(tmp_path, "fears")

    assert state.exists()
    assert goals.exists()
    assert deferred.exists()
    assert fears.exists()


def test_set_primary_goal_updates_goals_json(tmp_path: Path) -> None:
    goals_path = set_primary_goal(tmp_path, "Ship v1")
    data = json.loads(goals_path.read_text(encoding="utf-8"))
    primary = data.get("primary_goal") or data.get("primaryGoal")
    if isinstance(primary, str):
        assert primary == "Ship v1"
    else:
        assert isinstance(primary, dict)
        assert primary.get("description") == "Ship v1"


def test_append_deferred_appends_line(tmp_path: Path) -> None:
    deferred_path = append_deferred(tmp_path, "Investigate flaky test")
    content = deferred_path.read_text(encoding="utf-8")
    assert "Investigate flaky test" in content


def test_snapshot_state_copies_to_history(tmp_path: Path) -> None:
    state_path = open_protocol_file(tmp_path, "state")
    state_path.write_text("# Agent State\n\nhello\n", encoding="utf-8")

    snap = snapshot_state(tmp_path, reason="after setup")
    assert snap is not None
    assert snap.exists()
    assert snap.parent.name == "history"
    assert "after-setup" in snap.name
