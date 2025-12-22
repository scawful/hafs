import importlib
import warnings
from typing import Any

_DEPRECATION_MESSAGE = "context is deprecated. Import from 'context' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_EXPORTS = [
    # Original builder
    "ContextPromptBuilder",
    # Constructor (Phase 1)
    "ContextConstructor",
    "ConstructorConfig",
    "ContextStore",
    "SelectionCriteria",
    # Updater (Phase 2)
    "ContextUpdater",
    "UpdaterConfig",
    "RetentionPolicy",
    "SyncEvent",
    # Evaluator (Phase 3)
    "ContextEvaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    "TaskOutcome",
    # Token Budget Manager
    "TokenBudgetManager",
    "BudgetManagerConfig",
    "BudgetAllocation",
    "ModelConfig",
    "ModelCapability",
    "MODEL_CONFIGS",
    "create_budget_for_model",
]

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        module = importlib.import_module("context")
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORTS))

__all__ = _EXPORTS
