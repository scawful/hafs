"""Training Data Pipeline.

Agent-based training data generation for finetuning local LLMs.
Supports multiple domains (ASM, C++, text) with quality refinement
via embeddings, knowledge graph validation, domain-specific validators,
feedback tracking, and active learning for coverage optimization.
"""

from agents.training.base import (
    DataGenerator,
    GenerationCheckpoint,
    GenerationResult,
    QualityScore,
    SourceItem,
    TrainingSample,
)
from agents.training.quality import QualityPipeline, DuplicateResult
from agents.training.active_learning import ActiveLearningSampler, CoverageReport

# Validators
from agents.training.validators import (
    AsmValidator,
    CompositeValidator,
    CppValidator,
    KGValidator,
    ValidationResult,
    Validator,
)

# Feedback
from agents.training.feedback import (
    QualityFeedbackTracker,
    QualityTrend,
    RejectionReason,
    SampleCorrelation,
    TrainingFeedback,
)

__all__ = [
    # Core
    "DataGenerator",
    "GenerationCheckpoint",
    "GenerationResult",
    "QualityScore",
    "SourceItem",
    "TrainingSample",
    # Quality
    "QualityPipeline",
    "DuplicateResult",
    # Active Learning
    "ActiveLearningSampler",
    "CoverageReport",
    # Validators
    "AsmValidator",
    "CompositeValidator",
    "CppValidator",
    "KGValidator",
    "ValidationResult",
    "Validator",
    # Feedback
    "QualityFeedbackTracker",
    "QualityTrend",
    "RejectionReason",
    "SampleCorrelation",
    "TrainingFeedback",
]
