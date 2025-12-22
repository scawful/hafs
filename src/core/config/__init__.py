"""Configuration management for cognitive protocol."""

from .loader import (
    CognitiveProtocolConfig,
    ConfigLoader,
    get_config,
    MetacognitionConfig,
    EpistemicConfig,
    EmotionsConfig,
    GroundingConfig,
    AnalysisTriggersConfig,
    HivemindConfig,
    GoalsConfig,
    PerformanceConfig,
)

__all__ = [
    "CognitiveProtocolConfig",
    "ConfigLoader",
    "get_config",
    "MetacognitionConfig",
    "EpistemicConfig",
    "EmotionsConfig",
    "GroundingConfig",
    "AnalysisTriggersConfig",
    "HivemindConfig",
    "GoalsConfig",
    "PerformanceConfig",
]
