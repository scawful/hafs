"""Context Engineering Pipeline for AI agent prompts.

Implements the Context Engineering Pipeline from AFS research:
- Constructor: Select, prioritize, and compress context items
- Updater: Refresh and synchronize context with AFS state
- Evaluator: Validate context and learn from feedback

Based on "Everything is Context: Agentic File System Abstraction"
"""

from __future__ import annotations

from .builder import ContextPromptBuilder
from .constructor import (
    ContextConstructor,
    ConstructorConfig,
    ContextStore,
    SelectionCriteria,
)
from .updater import (
    ContextUpdater,
    UpdaterConfig,
    RetentionPolicy,
    SyncEvent,
)
from .evaluator import (
    ContextEvaluator,
    EvaluatorConfig,
    EvaluationResult,
    TaskOutcome,
)
from .token_budget import (
    TokenBudgetManager,
    BudgetManagerConfig,
    BudgetAllocation,
    ModelConfig,
    ModelCapability,
    MODEL_CONFIGS,
    create_budget_for_model,
)

__all__ = [
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
