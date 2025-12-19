"""Metacognition and self-monitoring subsystem."""

from hafs.core.metacognition.monitor import MetacognitionMonitor
from hafs.core.metacognition.strategies import (
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
