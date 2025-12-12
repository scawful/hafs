"""Compatibility helpers for goals.json (HAFS vs oracle-code)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

WireFormat = Literal["snake", "camel"]


def detect_wire_format(data: dict[str, Any]) -> WireFormat:
    if "primaryGoal" in data or "goalStack" in data or "instrumentalGoals" in data:
        return "camel"
    return "snake"


def extract_primary_goal_text(data: dict[str, Any]) -> str | None:
    """Extract a primary goal description from either schema."""
    if "primary_goal" in data:
        value = data.get("primary_goal")
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            desc = value.get("description")
            if isinstance(desc, str):
                return desc.strip() or None

    if "primaryGoal" in data:
        value = data.get("primaryGoal")
        if isinstance(value, str):
            return value.strip() or None
        if isinstance(value, dict):
            desc = value.get("description")
            if isinstance(desc, str):
                return desc.strip() or None

    return None


def set_primary_goal_inplace(data: dict[str, Any], *, description: str, wire: WireFormat) -> dict[str, Any]:
    """Set the primary goal while preserving the existing schema style."""
    description = " ".join(description.strip().splitlines()).strip()
    if not description:
        return data

    now = datetime.now().isoformat(timespec="seconds")

    if wire == "camel":
        primary = data.get("primaryGoal")
        if not isinstance(primary, dict):
            primary = {}

        primary.setdefault("id", f"pg-{uuid4().hex[:8]}")
        primary["description"] = description
        primary.setdefault("goalType", "primary")
        primary.setdefault("status", "in_progress")
        primary.setdefault("priority", "medium")
        primary.setdefault("progress", 0)
        primary.setdefault("createdAt", now)
        primary["updatedAt"] = now
        primary.setdefault("completedAt", None)
        primary.setdefault("notes", "")
        primary.setdefault("userStated", description)
        primary.setdefault("successCriteria", [])
        primary.setdefault("constraints", [])

        data["primaryGoal"] = primary
        data.setdefault("subgoals", [])
        data.setdefault("instrumentalGoals", [])
        data.setdefault("goalStack", [])
        data.setdefault("conflicts", [])
        data["lastUpdated"] = now
        return data

    # snake
    primary = data.get("primary_goal")
    if not isinstance(primary, dict):
        primary = {}
    primary.setdefault("id", f"g-{uuid4().hex[:8]}")
    primary["description"] = description
    primary.setdefault("goal_type", "primary")
    primary.setdefault("status", "in_progress")
    primary.setdefault("priority", "medium")
    primary.setdefault("progress", 0.0)
    primary.setdefault("created_at", now)
    primary["updated_at"] = now
    primary.setdefault("completed_at", None)
    primary.setdefault("notes", "")
    primary.setdefault("user_stated", description)
    primary.setdefault("success_criteria", [])
    primary.setdefault("constraints", [])

    data["primary_goal"] = primary
    data.setdefault("subgoals", [])
    data.setdefault("instrumental_goals", [])
    data.setdefault("goal_stack", [])
    data.setdefault("conflicts", [])
    data["last_updated"] = now
    return data

