"""Compatibility helpers for emotions.json (HAFS vs oracle-code).

HAFS scaffolds oracle-code's camelCase schema by default, but we still normalize:
- camelCase (oracle-code)
- legacy snake_case or nested "categories" maps (older HAFS scaffolds)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

WireFormat = Literal["camel", "legacy"]


def detect_wire_format(data: Mapping[str, Any]) -> WireFormat:
    if "fears" in data and isinstance(data.get("session"), Mapping) and "anxietyLevel" in data.get("session", {}):
        return "camel"
    return "legacy"


def normalize_emotions(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize emotions payload to oracle-code-style camelCase."""
    fmt = detect_wire_format(data)
    if fmt == "camel":
        # Already oracle-code style; just copy
        return dict(data)

    normalized: dict[str, Any] = {
        "schema_version": data.get("schema_version") or data.get("schemaVersion") or "0.3",
        "producer": data.get("producer") or {"name": "unknown"},
        "last_updated": data.get("last_updated") or data.get("lastUpdated"),
    }

    # Session normalization
    session = data.get("session", {}) or {}
    mood = session.get("mood")
    if isinstance(mood, Mapping):
        mood_val = mood.get("current", "neutral")
    else:
        mood_val = mood or "neutral"
    normalized["session"] = {
        "mood": mood_val,
        "anxietyLevel": session.get("anxiety", {}).get("level", 0) if isinstance(session.get("anxiety"), Mapping) else session.get("anxietyLevel", 0),
        "confidenceLevel": int((session.get("confidence", 0.5) or 0) * 100) if "confidence" in session else session.get("confidenceLevel", 50),
        "recentEmotions": session.get("recentEmotions", []),
        "moodHistory": session.get("moodHistory", []),
        "mode": session.get("mode", "general"),
        "sessionStart": session.get("session_start") or session.get("sessionStart"),
    }

    # Legacy categories nested map → flat stores
    cats = data.get("categories", {}) or {}
    normalized.update(
        {
            "fears": data.get("fears") or cats.get("fear") or {},
            "curiosities": data.get("curiosities") or cats.get("curiosity") or {},
            "satisfactions": data.get("satisfactions") or cats.get("satisfaction") or {},
            "frustrations": data.get("frustrations") or cats.get("frustration") or {},
            "excitements": data.get("excitements") or cats.get("excitement") or {},
            "determinations": data.get("determinations") or cats.get("determination") or {},
            "cautions": data.get("cautions") or cats.get("caution") or {},
            "reliefs": data.get("reliefs") or cats.get("relief") or {},
        }
    )

    settings = data.get("settings") or {}
    normalized["settings"] = settings
    return normalized


def denormalize_emotions(data: Mapping[str, Any], *, wire_format: WireFormat) -> dict[str, Any]:
    """Convert camelCase to legacy nested categories if needed."""
    if wire_format == "camel":
        return dict(data)

    out = dict(data)
    # Re-nest categories
    categories = {
        "fear": data.get("fears", {}),
        "curiosity": data.get("curiosities", {}),
        "satisfaction": data.get("satisfactions", {}),
        "frustration": data.get("frustrations", {}),
        "excitement": data.get("excitements", {}),
        "determination": data.get("determinations", {}),
        "caution": data.get("cautions", {}),
        "relief": data.get("reliefs", {}),
    }
    out["categories"] = categories
    session = data.get("session", {}) or {}
    if session:
        out["session"] = {
            "mood": {"current": session.get("mood", "neutral"), "intensity": 0, "valence": 0},
            "anxiety": {"level": session.get("anxietyLevel", 0), "sources": []},
            "confidence": (session.get("confidenceLevel", 50) or 0) / 100,
            "mode": session.get("mode", "general"),
            "session_start": session.get("sessionStart"),
        }
    return out
