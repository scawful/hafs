from __future__ import annotations

from hafs.core.protocol.goals_compat import (
    detect_wire_format,
    extract_primary_goal_text,
    set_primary_goal_inplace,
)


def test_extract_primary_goal_text_snake_string() -> None:
    assert extract_primary_goal_text({"primary_goal": "Ship"}) == "Ship"


def test_extract_primary_goal_text_snake_object() -> None:
    assert extract_primary_goal_text({"primary_goal": {"description": "Ship"}}) == "Ship"


def test_extract_primary_goal_text_camel_object() -> None:
    assert extract_primary_goal_text({"primaryGoal": {"description": "Ship"}}) == "Ship"


def test_set_primary_goal_inplace_preserves_wire_format() -> None:
    data = {"primaryGoal": None, "subgoals": []}
    assert detect_wire_format(data) == "camel"
    updated = set_primary_goal_inplace(data, description="Launch", wire="camel")
    assert isinstance(updated.get("primaryGoal"), dict)
    assert updated["primaryGoal"]["description"] == "Launch"
    assert "lastUpdated" in updated

