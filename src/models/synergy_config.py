"""Synergy service configuration models.

Configuration for IRT-based ability estimation and LLM ToM assessment,
based on "Quantifying Human-AI Synergy" research.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AssessmentMode(str, Enum):
    """ToM assessment modes for cost/accuracy tradeoff.

    FULL: Assess every interaction (most accurate, highest cost)
    SAMPLE: Probabilistic sampling (configurable rate)
    BATCH: Queue and assess in batches (most efficient)
    """

    FULL = "full"
    SAMPLE = "sample"
    BATCH = "batch"


class ToMAssessmentConfig(BaseModel):
    """Configuration for LLM-based Theory of Mind assessment.

    Controls when and how often ToM assessments are performed
    using the LMRA (LLM-as-Research-Assistant) approach.
    """

    enabled: bool = Field(default=True, description="Enable ToM assessment")
    mode: AssessmentMode = Field(
        default=AssessmentMode.SAMPLE,
        description="Assessment mode (full/sample/batch)"
    )
    sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of assessing each interaction in SAMPLE mode"
    )
    max_per_hour: int = Field(
        default=20,
        ge=0,
        description="Maximum assessments per hour (0 = unlimited)"
    )
    max_daily_cost: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum daily cost in USD for assessments"
    )
    batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of interactions to batch in BATCH mode"
    )
    batch_interval_seconds: int = Field(
        default=300,
        ge=60,
        description="Seconds between batch processing runs"
    )
    min_prompt_length: int = Field(
        default=20,
        ge=0,
        description="Minimum prompt length to assess (skip trivial prompts)"
    )
    model_tier: str = Field(
        default="reasoning",
        description="Model tier for assessment (fast/balanced/reasoning)"
    )

    model_config = ConfigDict(frozen=False)

    @property
    def is_full_mode(self) -> bool:
        """Check if running in full assessment mode."""
        return self.mode == AssessmentMode.FULL

    @property
    def is_sample_mode(self) -> bool:
        """Check if running in sample mode."""
        return self.mode == AssessmentMode.SAMPLE

    @property
    def is_batch_mode(self) -> bool:
        """Check if running in batch mode."""
        return self.mode == AssessmentMode.BATCH

    @property
    def effective_rate(self) -> float:
        """Get effective assessment rate based on mode."""
        if self.mode == AssessmentMode.FULL:
            return 1.0
        elif self.mode == AssessmentMode.SAMPLE:
            return self.sample_rate
        else:  # BATCH
            return 1.0  # All batched items get assessed


class IRTConfig(BaseModel):
    """Configuration for Bayesian Item Response Theory estimation.

    Controls the 1PL Rasch model: Pr(Y=1) = logit^-1(θ - β)
    """

    enabled: bool = Field(default=True, description="Enable IRT ability tracking")
    prior_mean: float = Field(
        default=0.0,
        description="Prior mean for ability estimates"
    )
    prior_sd: float = Field(
        default=1.0,
        gt=0.0,
        description="Prior standard deviation for ability estimates"
    )
    min_responses: int = Field(
        default=5,
        ge=1,
        description="Minimum responses before ability is considered reliable"
    )
    max_history: int = Field(
        default=500,
        ge=10,
        description="Maximum response history to retain per ability type"
    )
    update_interval: int = Field(
        default=1,
        ge=1,
        description="Update ability estimate every N responses"
    )
    convergence_threshold: float = Field(
        default=0.001,
        gt=0.0,
        description="Newton-Raphson convergence threshold"
    )
    max_iterations: int = Field(
        default=100,
        ge=10,
        description="Maximum Newton-Raphson iterations"
    )

    model_config = ConfigDict(frozen=False)

    @property
    def prior_variance(self) -> float:
        """Get prior variance (sd^2)."""
        return self.prior_sd ** 2

    @property
    def prior_precision(self) -> float:
        """Get prior precision (1/variance)."""
        return 1.0 / self.prior_variance


class DifficultyEstimationConfig(BaseModel):
    """Configuration for task difficulty estimation.

    Maps task characteristics to difficulty beta values on the logit scale.
    """

    enabled: bool = Field(default=True, description="Enable difficulty estimation")
    default_difficulty: float = Field(
        default=0.0,
        description="Default difficulty for unknown tasks"
    )

    # Difficulty adjustments by task type
    task_difficulty_map: dict[str, float] = Field(
        default_factory=lambda: {
            "trivial": -2.0,
            "easy": -1.0,
            "medium": 0.0,
            "hard": 1.0,
            "expert": 2.0,
        },
        description="Mapping from qualitative difficulty to beta values"
    )

    # Context-based difficulty modifiers
    context_modifiers: dict[str, float] = Field(
        default_factory=lambda: {
            "code_generation": 0.5,
            "debugging": 0.3,
            "explanation": -0.3,
            "refactoring": 0.2,
            "documentation": -0.5,
        },
        description="Difficulty modifiers by task context"
    )

    model_config = ConfigDict(frozen=False)

    def get_difficulty(self, level: str, context: Optional[str] = None) -> float:
        """Calculate difficulty beta from level and optional context."""
        base = self.task_difficulty_map.get(level, self.default_difficulty)
        if context:
            modifier = self.context_modifiers.get(context, 0.0)
            return base + modifier
        return base


class SynergyServiceConfig(BaseModel):
    """Master configuration for the synergy service.

    Combines ToM assessment, IRT estimation, and service behavior settings.
    """

    enabled: bool = Field(default=True, description="Enable synergy service")
    tom_assessment: ToMAssessmentConfig = Field(default_factory=ToMAssessmentConfig)
    irt_estimation: IRTConfig = Field(default_factory=IRTConfig)
    difficulty_estimation: DifficultyEstimationConfig = Field(
        default_factory=DifficultyEstimationConfig
    )

    # Service behavior
    persist_profiles: bool = Field(
        default=True,
        description="Persist user profiles to disk"
    )
    profile_dir: str = Field(
        default="~/.context/synergy",
        description="Directory for synergy data persistence"
    )
    auto_start: bool = Field(
        default=False,
        description="Auto-start synergy service with coordinator"
    )

    # Quota integration
    use_quota_manager: bool = Field(
        default=True,
        description="Integrate with hafs quota manager for cost control"
    )
    quota_pool: str = Field(
        default="synergy",
        description="Quota pool name for synergy assessments"
    )

    model_config = ConfigDict(frozen=False)

    @property
    def is_fully_enabled(self) -> bool:
        """Check if service and all components are enabled."""
        return (
            self.enabled
            and self.tom_assessment.enabled
            and self.irt_estimation.enabled
        )

    @property
    def profile_path(self) -> str:
        """Get expanded profile directory path."""
        import os
        return os.path.expanduser(self.profile_dir)

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []

        if self.tom_assessment.mode == AssessmentMode.FULL:
            warnings.append(
                "FULL assessment mode will assess every interaction - "
                "consider SAMPLE mode for cost control"
            )

        if self.tom_assessment.max_daily_cost > 10.0:
            warnings.append(
                f"Daily cost limit is high (${self.tom_assessment.max_daily_cost}) - "
                "consider lowering for development"
            )

        if self.irt_estimation.min_responses < 5:
            warnings.append(
                "Low min_responses may produce unreliable ability estimates"
            )

        return warnings
