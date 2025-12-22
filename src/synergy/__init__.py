"""Synergy (Theory of Mind) module for HAFS multi-agent orchestration.

Includes research-based enhancements from "Quantifying Human-AI Synergy":
- LLM-based ToM assessment (LMRA approach)
- Bayesian IRT ability estimation
- Separation of individual vs collaborative ability
"""

from .analyzer import PromptAnalyzer
from .evaluator import ResponseEvaluator
from .markers import TOM_PATTERNS, get_all_patterns, get_patterns_for_type
from .profile import UserProfileManager
from .scoring import SynergyCalculator
from .tom_assessor import ToMAssessor
from .irt_estimator import BayesianIRTEstimator

__all__ = [
    # Marker detection (regex-based)
    "TOM_PATTERNS",
    "get_all_patterns",
    "get_patterns_for_type",
    # Analysis
    "PromptAnalyzer",
    # Evaluation
    "ResponseEvaluator",
    # Profile management
    "UserProfileManager",
    # Scoring
    "SynergyCalculator",
    # Research-based enhancements
    "ToMAssessor",
    "BayesianIRTEstimator",
]
