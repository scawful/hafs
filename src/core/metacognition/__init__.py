"""Metacognition and self-monitoring subsystem."""

from core.metacognition.monitor import MetacognitionMonitor
from core.metacognition.strategies import (
    STRATEGY_RECOMMENDATIONS,
    get_strategy_description,
    suggest_strategy_change,
)

__all__ = [
    "MetacognitionMonitor",
    "STRATEGY_RECOMMENDATIONS",
    "get_strategy_description",
    "suggest_strategy_change",
]
