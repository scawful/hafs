"""Item Response Theory (IRT) models for synergy measurement.

Based on "Quantifying Human-AI Synergy" research paper:
- Separates individual ability (theta) from collaborative ability (kappa)
- Uses Bayesian IRT framework for ability estimation
- Tracks dynamic ToM fluctuations
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AbilityType(str, Enum):
    """Types of ability tracked in the IRT model."""

    THETA_INDIVIDUAL = "theta_individual"  # Solo ability without AI
    KAPPA_COLLABORATIVE = "kappa_collaborative"  # Human-AI collaborative ability


class DifficultyLevel(str, Enum):
    """Qualitative difficulty levels for task categorization."""

    TRIVIAL = "trivial"  # beta ~ -2.0
    EASY = "easy"  # beta ~ -1.0
    MEDIUM = "medium"  # beta ~ 0.0
    HARD = "hard"  # beta ~ 1.0
    EXPERT = "expert"  # beta ~ 2.0


class ItemResponse(BaseModel):
    """A single item response for IRT estimation.

    Records whether a task was completed successfully, along with
    the estimated difficulty of the task.
    """

    item_id: str
    response: bool  # True = success, False = failure
    difficulty_beta: float = Field(default=0.0, description="Task difficulty on logit scale")
    timestamp: datetime = Field(default_factory=datetime.now)
    context: str = Field(default="", description="Task type or category")

    model_config = ConfigDict(frozen=True)


class AbilityEstimate(BaseModel):
    """Current ability estimate with uncertainty.

    Uses the 1PL (Rasch) model: Pr(Y=1) = logit^-1(theta - beta)
    """

    ability_type: AbilityType
    theta: float = Field(default=0.0, description="Ability estimate on logit scale")
    se: float = Field(default=1.0, ge=0.0, description="Standard error of estimate")
    n_responses: int = Field(default=0, ge=0)
    last_updated: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)

    def probability_at_difficulty(self, beta: float) -> float:
        """Calculate Pr(Y=1) = logit^-1(theta - beta)."""
        z = self.theta - beta
        # Clamp to prevent overflow
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """Return 95% confidence interval for ability estimate."""
        return (self.theta - 1.96 * self.se, self.theta + 1.96 * self.se)

    @property
    def is_reliable(self) -> bool:
        """Check if estimate has enough responses to be reliable."""
        return self.n_responses >= 5 and self.se < 0.5


class TraitToMScore(BaseModel):
    """Theory of Mind score for a specific trait dimension.

    Tracks both the stable trait-level mean and moment-to-moment
    deviations, following the research finding that ToM varies
    both between and within users.
    """

    trait: str
    mean_score: float = Field(default=2.5, ge=0.0, le=5.0, description="Trait-level mean (0-5)")
    within_deviation: float = Field(default=0.0, description="Current deviation from mean")
    n_assessments: int = Field(default=0, ge=0)

    model_config = ConfigDict(frozen=False)

    @property
    def current_score(self) -> float:
        """Get current score (mean + deviation)."""
        return max(0.0, min(5.0, self.mean_score + self.within_deviation))

    @property
    def volatility(self) -> str:
        """Categorize score volatility."""
        if abs(self.within_deviation) < 0.5:
            return "stable"
        elif abs(self.within_deviation) < 1.0:
            return "moderate"
        else:
            return "volatile"


class ToMAssessment(BaseModel):
    """LLM-based Theory of Mind assessment result.

    Contains scores for 8 ToM dimensions, each on a 0-5 scale,
    following the LMRA (LLM-as-Research-Assistant) approach
    from the research paper.
    """

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)

    # 8 ToM dimensions (0-5 scale)
    perspective_taking: float = Field(ge=0.0, le=5.0, default=2.5)
    goal_inference: float = Field(ge=0.0, le=5.0, default=2.5)
    knowledge_gap_detection: float = Field(ge=0.0, le=5.0, default=2.5)
    communication_repair: float = Field(ge=0.0, le=5.0, default=2.5)
    confirmation_seeking: float = Field(ge=0.0, le=5.0, default=2.5)
    mental_state_attribution: float = Field(ge=0.0, le=5.0, default=2.5)
    plan_coordination: float = Field(ge=0.0, le=5.0, default=2.5)
    challenge_disagree: float = Field(ge=0.0, le=5.0, default=2.5)

    # Assessment metadata
    prompt_text: str = Field(default="", description="Truncated prompt that was assessed")
    response_text: str = Field(default="", description="Truncated response that was assessed")
    assessor_model: str = Field(default="", description="Model used for assessment")
    latency_ms: int = Field(default=0, ge=0, description="Assessment latency in milliseconds")
    reasoning: str = Field(default="", description="LLM's reasoning for scores")

    model_config = ConfigDict(frozen=True)

    @property
    def overall_score(self) -> float:
        """Calculate overall ToM score as average of all dimensions."""
        scores = [
            self.perspective_taking,
            self.goal_inference,
            self.knowledge_gap_detection,
            self.communication_repair,
            self.confirmation_seeking,
            self.mental_state_attribution,
            self.plan_coordination,
            self.challenge_disagree,
        ]
        return sum(scores) / len(scores)

    @property
    def dimension_scores(self) -> dict[str, float]:
        """Get all dimension scores as a dictionary."""
        return {
            "perspective_taking": self.perspective_taking,
            "goal_inference": self.goal_inference,
            "knowledge_gap_detection": self.knowledge_gap_detection,
            "communication_repair": self.communication_repair,
            "confirmation_seeking": self.confirmation_seeking,
            "mental_state_attribution": self.mental_state_attribution,
            "plan_coordination": self.plan_coordination,
            "challenge_disagree": self.challenge_disagree,
        }

    @property
    def strongest_dimension(self) -> tuple[str, float]:
        """Get the highest-scoring dimension."""
        scores = self.dimension_scores
        best = max(scores.items(), key=lambda x: x[1])
        return best

    @property
    def weakest_dimension(self) -> tuple[str, float]:
        """Get the lowest-scoring dimension."""
        scores = self.dimension_scores
        worst = min(scores.items(), key=lambda x: x[1])
        return worst


class EnhancedUserProfile(BaseModel):
    """Extended user profile with IRT ability tracking.

    Separates individual ability (theta) from collaborative ability (kappa),
    following the research finding that these are distinct latent abilities.
    """

    id: UUID = Field(default_factory=uuid4)

    # IRT Ability Estimates (key research finding: theta != kappa)
    theta_individual: AbilityEstimate = Field(
        default_factory=lambda: AbilityEstimate(ability_type=AbilityType.THETA_INDIVIDUAL)
    )
    kappa_collaborative: AbilityEstimate = Field(
        default_factory=lambda: AbilityEstimate(ability_type=AbilityType.KAPPA_COLLABORATIVE)
    )

    # Dynamic ToM tracking (trait-level means + within-user deviations)
    tom_traits: dict[str, TraitToMScore] = Field(default_factory=dict)

    # Assessment history
    tom_assessments: list[ToMAssessment] = Field(default_factory=list)
    item_responses: list[ItemResponse] = Field(default_factory=list)

    # Synergy tracking
    synergy_gain: float = Field(default=0.0, description="kappa - theta (positive = AI helps)")
    last_synergy_update: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=False)

    @property
    def ai_benefit(self) -> str:
        """Categorize the benefit from AI collaboration."""
        if self.synergy_gain > 0.5:
            return "significant_benefit"
        elif self.synergy_gain > 0.1:
            return "moderate_benefit"
        elif self.synergy_gain > -0.1:
            return "neutral"
        elif self.synergy_gain > -0.5:
            return "slight_hindrance"
        else:
            return "significant_hindrance"

    @property
    def recent_tom_score(self) -> Optional[float]:
        """Get most recent ToM assessment score."""
        if self.tom_assessments:
            return self.tom_assessments[-1].overall_score
        return None

    @property
    def tom_trend(self) -> str:
        """Calculate ToM trend from recent assessments."""
        if len(self.tom_assessments) < 3:
            return "insufficient_data"

        recent = self.tom_assessments[-5:]
        scores = [a.overall_score for a in recent]

        if len(scores) < 2:
            return "stable"

        # Simple trend: compare first half to second half
        mid = len(scores) // 2
        first_half = sum(scores[:mid]) / mid
        second_half = sum(scores[mid:]) / (len(scores) - mid)

        diff = second_half - first_half
        if diff > 0.3:
            return "improving"
        elif diff < -0.3:
            return "declining"
        else:
            return "stable"

    def initialize_tom_traits(self) -> None:
        """Initialize ToM trait tracking for all dimensions."""
        traits = [
            "perspective_taking",
            "goal_inference",
            "knowledge_gap_detection",
            "communication_repair",
            "confirmation_seeking",
            "mental_state_attribution",
            "plan_coordination",
            "challenge_disagree",
        ]
        for trait in traits:
            if trait not in self.tom_traits:
                self.tom_traits[trait] = TraitToMScore(trait=trait)

    def get_summary(self) -> dict:
        """Get a summary of the user's synergy profile."""
        return {
            "theta_individual": {
                "estimate": self.theta_individual.theta,
                "se": self.theta_individual.se,
                "n_responses": self.theta_individual.n_responses,
                "reliable": self.theta_individual.is_reliable,
            },
            "kappa_collaborative": {
                "estimate": self.kappa_collaborative.theta,
                "se": self.kappa_collaborative.se,
                "n_responses": self.kappa_collaborative.n_responses,
                "reliable": self.kappa_collaborative.is_reliable,
            },
            "synergy_gain": self.synergy_gain,
            "ai_benefit": self.ai_benefit,
            "tom_trend": self.tom_trend,
            "recent_tom_score": self.recent_tom_score,
            "tom_traits": {
                trait: {
                    "mean": data.mean_score,
                    "current": data.current_score,
                    "volatility": data.volatility,
                }
                for trait, data in self.tom_traits.items()
            },
        }
