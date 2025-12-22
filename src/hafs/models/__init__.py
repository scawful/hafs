import importlib
import warnings
from typing import Any

_DEPRECATION_MESSAGE = "models is deprecated. Import from 'models' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_EXPORTS = [
    # AFS models
    "ContextRoot",
    "MountPoint",
    "MountType",
    "ProjectMetadata",
    # Gemini models
    "GeminiMessage",
    "GeminiProject",
    "GeminiSession",
    # Claude models
    "PlanDocument",
    "PlanTask",
    "TaskStatus",
    # Antigravity models
    "AntigravityBrain",
    "AntigravityTask",
    # Agent models
    "Agent",
    "AgentMessage",
    "AgentRole",
    "SharedContext",
    # Metacognition models
    "CognitiveLoad",
    "FlowStateIndicators",
    "HelpSeeking",
    "MetacognitiveState",
    "ProgressStatus",
    "SelfCorrection",
    "SpinDetection",
    "Strategy",
    # Goal hierarchy models
    "Goal",
    "GoalConflict",
    "GoalHierarchy",
    "GoalPriority",
    "GoalStatus",
    "GoalType",
    "InstrumentalGoal",
    "PrimaryGoal",
    "Subgoal",
    # Synergy models
    "ResponseQuality",
    "SynergyScore",
    "ToMMarker",
    "ToMMarkers",
    "ToMMarkerType",
    "UserPreferences",
    "UserProfile",
    # IRT models
    "AbilityEstimate",
    "AbilityType",
    "DifficultyLevel",
    "EnhancedUserProfile",
    "ItemResponse",
    "ToMAssessment",
    "TraitToMScore",
    # Synergy config models
    "AssessmentMode",
    "DifficultyEstimationConfig",
    "IRTConfig",
    "SynergyServiceConfig",
    "ToMAssessmentConfig",
    # Context engineering models
    "ContextItem",
    "ContextPriority",
    "ContextWindow",
    "MemoryType",
    "TokenBudget",
]

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        module = importlib.import_module("models")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS))

__all__ = _EXPORTS
