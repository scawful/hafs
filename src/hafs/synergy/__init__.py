"""Synergy (Theory of Mind) module for HAFS multi-agent orchestration."""

from hafs.synergy.analyzer import PromptAnalyzer
from hafs.synergy.evaluator import ResponseEvaluator
from hafs.synergy.markers import TOM_PATTERNS, get_all_patterns, get_patterns_for_type
from hafs.synergy.profile import UserProfileManager
from hafs.synergy.scoring import SynergyCalculator

__all__ = [
    # Marker detection
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
]
