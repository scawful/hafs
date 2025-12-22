import importlib
import warnings
from typing import Any

_DEPRECATION_MESSAGE = "synergy is deprecated. Import from 'synergy' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_EXPORTS = [
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

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        module = importlib.import_module("synergy")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS))

__all__ = _EXPORTS
