"""Metacognition and self-monitoring models for cognitive protocol."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ProgressStatus(str, Enum):
    """Status of progress toward current goal."""

    MAKING_PROGRESS = "making_progress"
    SPINNING = "spinning"
    BLOCKED = "blocked"


class Strategy(str, Enum):
    """Available problem-solving strategies."""

    DIVIDE_AND_CONQUER = "divide_and_conquer"
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    INCREMENTAL = "incremental"
    RESEARCH_FIRST = "research_first"
    PROTOTYPE = "prototype"


STRATEGY_DESCRIPTIONS = {
    Strategy.DIVIDE_AND_CONQUER: "Break problem into smaller subproblems",
    Strategy.DEPTH_FIRST: "Fully explore one path before trying others",
    Strategy.BREADTH_FIRST: "Survey all options before committing",
    Strategy.INCREMENTAL: "Make small changes, validate frequently",
    Strategy.RESEARCH_FIRST: "Gather information before acting",
    Strategy.PROTOTYPE: "Build quick proof-of-concept first",
}


class SpinDetection(BaseModel):
    """Tracks repeated similar actions to detect 'spinning' behavior."""

    recent_actions: list[str] = Field(
        default_factory=list,
        description="Signatures of recent actions for comparison",
    )
    similar_action_count: int = Field(
        default=0,
        description="Count of consecutive similar actions",
    )
    last_distinct_action_time: datetime | None = Field(
        default=None,
        description="Timestamp of last meaningfully different action",
    )
    spinning_threshold: int = Field(
        default=4,
        ge=3,
        le=5,
        description="Number of similar actions before flagging as spinning",
    )

    @property
    def is_spinning(self) -> bool:
        """Check if currently in a spinning state."""
        return self.similar_action_count >= self.spinning_threshold


class CognitiveLoad(BaseModel):
    """Tracks mental workload and focus capacity."""

    current: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current cognitive load as fraction of capacity",
    )
    warning_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Load level that triggers warning",
    )
    items_in_focus: int = Field(
        default=0,
        ge=0,
        description="Number of items currently being tracked",
    )
    max_recommended_items: int = Field(
        default=7,
        description="Miller's Law - recommended max items in working memory",
    )

    @property
    def is_overloaded(self) -> bool:
        """Check if cognitive load exceeds warning threshold."""
        return self.current >= self.warning_threshold

    @property
    def load_percentage(self) -> int:
        """Return load as percentage for display."""
        return int(self.current * 100)


class HelpSeeking(BaseModel):
    """Tracks when the agent should ask for user assistance."""

    uncertainty_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Uncertainty level that triggers help-seeking",
    )
    current_uncertainty: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current uncertainty about approach/outcome",
    )
    consecutive_failures: int = Field(
        default=0,
        ge=0,
        description="Number of consecutive failed attempts",
    )
    failure_threshold: int = Field(
        default=2,
        description="Number of failures before seeking help",
    )

    @property
    def should_ask_user(self) -> bool:
        """Determine if agent should seek user help."""
        return (
            self.current_uncertainty > self.uncertainty_threshold
            or self.consecutive_failures > self.failure_threshold
        )


class SelfCorrection(BaseModel):
    """Records when the agent catches and fixes its own mistakes."""

    id: str = Field(description="Unique identifier for this correction")
    what: str = Field(description="What was wrong or needed correction")
    when: datetime = Field(default_factory=datetime.now)
    why: str = Field(description="Why the mistake happened")
    outcome: str = Field(
        default="",
        description="Result of the correction",
    )


class FlowStateIndicators(BaseModel):
    """Criteria for determining if agent is in flow state."""

    min_progress_required: bool = Field(
        default=True,
        description="Must be making progress",
    )
    max_cognitive_load: float = Field(
        default=0.7,
        description="Maximum cognitive load for flow state",
    )
    min_strategy_effectiveness: float = Field(
        default=0.6,
        description="Minimum strategy effectiveness for flow state",
    )
    max_frustration: float = Field(
        default=0.3,
        description="Maximum frustration level for flow state",
    )
    no_help_needed: bool = Field(
        default=True,
        description="Should not need user help",
    )


class MetacognitiveState(BaseModel):
    """Complete metacognitive state of the agent."""

    current_strategy: Strategy = Field(
        default=Strategy.INCREMENTAL,
        description="Current problem-solving strategy",
    )
    strategy_effectiveness: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How well current strategy is working",
    )
    progress_status: ProgressStatus = Field(
        default=ProgressStatus.MAKING_PROGRESS,
        description="Current progress assessment",
    )
    spin_detection: SpinDetection = Field(
        default_factory=SpinDetection,
        description="Spinning behavior detection state",
    )
    cognitive_load: CognitiveLoad = Field(
        default_factory=CognitiveLoad,
        description="Current cognitive load state",
    )
    help_seeking: HelpSeeking = Field(
        default_factory=HelpSeeking,
        description="Help-seeking state",
    )
    self_corrections: list[SelfCorrection] = Field(
        default_factory=list,
        description="Record of self-corrections made",
    )
    flow_state: bool = Field(
        default=False,
        description="Whether agent is currently in flow state",
    )
    flow_state_indicators: FlowStateIndicators = Field(
        default_factory=FlowStateIndicators,
        description="Configurable flow state criteria",
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When this state was last updated",
    )

    def to_state_markdown(self) -> str:
        """Generate markdown representation for state.md."""
        spinning_warning = "Yes" if self.spin_detection.is_spinning else "No"
        help_needed = "Yes" if self.help_seeking.should_ask_user else "No"
        flow = "Yes" if self.flow_state else "No"

        return f"""## 9. Metacognitive Assessment
- **Current Strategy:** {self.current_strategy.value}
- **Strategy Effectiveness (0-1):** {self.strategy_effectiveness:.2f}
- **Progress Status:** {self.progress_status.value}
- **Cognitive Load:** {self.cognitive_load.load_percentage}%
- **Items in Focus:** {self.cognitive_load.items_in_focus}
- **Spinning Warning:** {spinning_warning}
- **Help Needed:** {help_needed}
- **Flow State:** {flow}
"""

    model_config = ConfigDict(use_enum_values=False)
