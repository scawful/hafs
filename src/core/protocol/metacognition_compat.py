"""Compatibility helpers for metacognition.json (HAFS vs oracle-code).

HAFS historically used snake_case keys (Pydantic models). oracle-code uses
camelCase keys (Zod schemas). This module provides normalization so HAFS can
read/write either format and preserve the existing "wire format" on save.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

WireFormat = Literal["snake", "camel"]


_TOP_LEVEL_MAP: dict[str, str] = {
    "currentStrategy": "current_strategy",
    "strategyEffectiveness": "strategy_effectiveness",
    "progressStatus": "progress_status",
    "spinDetection": "spin_detection",
    "cognitiveLoad": "cognitive_load",
    "helpSeeking": "help_seeking",
    "selfCorrections": "self_corrections",
    "flowState": "flow_state",
    "flowStateIndicators": "flow_state_indicators",
    "lastUpdated": "last_updated",
    "frustrationLevel": "frustration_level",
}

_SPIN_MAP: dict[str, str] = {
    "recentActions": "recent_actions",
    "similarActionCount": "similar_action_count",
    "lastDistinctActionTime": "last_distinct_action_time",
    "spinningThreshold": "spinning_threshold",
}

_LOAD_MAP: dict[str, str] = {
    "warningThreshold": "warning_threshold",
    "itemsInFocus": "items_in_focus",
    "maxRecommendedItems": "max_recommended_items",
}

_HELP_MAP: dict[str, str] = {
    "uncertaintyThreshold": "uncertainty_threshold",
    "currentUncertainty": "current_uncertainty",
    "consecutiveFailures": "consecutive_failures",
    "failureThreshold": "failure_threshold",
}

_FLOW_IND_MAP: dict[str, str] = {
    "minProgressRequired": "min_progress_required",
    "maxCognitiveLoad": "max_cognitive_load",
    "minStrategyEffectiveness": "min_strategy_effectiveness",
    "maxFrustration": "max_frustration",
    "noHelpNeeded": "no_help_needed",
}


def detect_wire_format(data: Mapping[str, Any]) -> WireFormat:
    if "currentStrategy" in data or "spinDetection" in data or "flowState" in data:
        return "camel"
    return "snake"


def normalize_metacognition(data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize oracle-code camelCase to HAFS snake_case.

    Returns a new dict; does not mutate the input.
    """
    normalized: dict[str, Any] = {}

    # First pass: translate known top-level keys
    for key, value in data.items():
        target = _TOP_LEVEL_MAP.get(key, key)
        normalized[target] = value

    # Nested known structures
    spin = normalized.get("spin_detection")
    if isinstance(spin, Mapping):
        normalized["spin_detection"] = {_SPIN_MAP.get(k, k): v for k, v in spin.items()}

    load = normalized.get("cognitive_load")
    if isinstance(load, Mapping):
        normalized["cognitive_load"] = {_LOAD_MAP.get(k, k): v for k, v in load.items()}

    help_seeking = normalized.get("help_seeking")
    if isinstance(help_seeking, Mapping):
        normalized["help_seeking"] = {_HELP_MAP.get(k, k): v for k, v in help_seeking.items()}

    flow_inds = normalized.get("flow_state_indicators")
    if isinstance(flow_inds, Mapping):
        normalized["flow_state_indicators"] = {
            _FLOW_IND_MAP.get(k, k): v for k, v in flow_inds.items()
        }

    return normalized


def denormalize_metacognition(
    data: Mapping[str, Any], *, wire_format: WireFormat
) -> dict[str, Any]:
    """Convert snake_case metacognition dict to requested wire format."""
    if wire_format == "snake":
        return dict(data)

    # Invert maps
    inv_top = {v: k for k, v in _TOP_LEVEL_MAP.items()}
    inv_spin = {v: k for k, v in _SPIN_MAP.items()}
    inv_load = {v: k for k, v in _LOAD_MAP.items()}
    inv_help = {v: k for k, v in _HELP_MAP.items()}
    inv_flow = {v: k for k, v in _FLOW_IND_MAP.items()}

    out: dict[str, Any] = {}
    for key, value in data.items():
        target = inv_top.get(key, key)
        out[target] = value

    spin = out.get("spinDetection")
    if isinstance(spin, Mapping):
        out["spinDetection"] = {inv_spin.get(k, k): v for k, v in spin.items()}

    load = out.get("cognitiveLoad")
    if isinstance(load, Mapping):
        out["cognitiveLoad"] = {inv_load.get(k, k): v for k, v in load.items()}

    help_seeking = out.get("helpSeeking")
    if isinstance(help_seeking, Mapping):
        out["helpSeeking"] = {inv_help.get(k, k): v for k, v in help_seeking.items()}

    flow_inds = out.get("flowStateIndicators")
    if isinstance(flow_inds, Mapping):
        out["flowStateIndicators"] = {inv_flow.get(k, k): v for k, v in flow_inds.items()}

    return out


def known_top_level_keys(wire_format: WireFormat) -> set[str]:
    """Return the set of known top-level keys for a given wire format."""
    if wire_format == "camel":
        return set(_TOP_LEVEL_MAP.keys())
    return set(_TOP_LEVEL_MAP.values())
