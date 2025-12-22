"""Analysis Mode Protocol for AFS Cognitive Protocol v0.2.

This module implements the analysis modes defined in PROTOCOL_SPEC.md Section 5.
Analysis modes provide structured evaluation of agent behavior, collaboration quality,
and system performance based on peer-reviewed research findings.
"""

from core.analysis.modes import (
    AnalysisMode,
    AnalysisResult,
    BaseAnalyzer,
)
from core.analysis.critic import AdaptiveCritic, CriticReview, CriticTone
from core.analysis.triggers import (
    AnalysisGateMode,
    AnalysisTrigger,
    TriggerManager,
)

__all__ = [
    # Modes
    "AnalysisMode",
    "AnalysisResult",
    "BaseAnalyzer",
    # Critic
    "AdaptiveCritic",
    "CriticReview",
    "CriticTone",
    # Triggers
    "AnalysisGateMode",
    "AnalysisTrigger",
    "TriggerManager",
]
