"""Data models for HAFS."""

from hafs.models.afs import ContextRoot, MountPoint, MountType, ProjectMetadata
from hafs.models.agent import Agent, AgentMessage, AgentRole, SharedContext
from hafs.models.antigravity import AntigravityBrain, AntigravityTask
from hafs.models.claude import PlanDocument, PlanTask, TaskStatus
from hafs.models.gemini import GeminiMessage, GeminiProject, GeminiSession
from hafs.models.metacognition import (
    CognitiveLoad,
    FlowStateIndicators,
    HelpSeeking,
    MetacognitiveState,
    ProgressStatus,
    SelfCorrection,
    SpinDetection,
    Strategy,
)
from hafs.models.goals import (
    Goal,
    GoalConflict,
    GoalHierarchy,
    GoalPriority,
    GoalStatus,
    GoalType,
    InstrumentalGoal,
    PrimaryGoal,
    Subgoal,
)
from hafs.models.synergy import (
    ResponseQuality,
    SynergyScore,
    ToMMarker,
    ToMMarkers,
    ToMMarkerType,
    UserPreferences,
    UserProfile,
)

__all__ = [
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
]
