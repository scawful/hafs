"""Theory of Mind and synergy scoring data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToMMarkerType(str, Enum):
    """Types of Theory of Mind markers detected in interactions."""

    PERSPECTIVE_TAKING = "perspective_taking"
    GOAL_INFERENCE = "goal_inference"
    KNOWLEDGE_GAP_DETECTION = "knowledge_gap_detection"
    COMMUNICATION_REPAIR = "communication_repair"
    CONFIRMATION_SEEKING = "confirmation_seeking"
    MENTAL_STATE_ATTRIBUTION = "mental_state_attribution"
    PLAN_COORDINATION = "plan_coordination"
    CHALLENGE_DISAGREE = "challenge_disagree"


class ToMMarker(BaseModel):
    """A detected Theory of Mind marker in interaction."""

    type: ToMMarkerType
    confidence: float = Field(ge=0.0, le=1.0)
    text_span: str
    context: str

    model_config = ConfigDict(frozen=True)

    @property
    def is_high_confidence(self) -> bool:
        """Check if marker has high confidence (>= 0.7)."""
        return self.confidence >= 0.7

    @property
    def is_low_confidence(self) -> bool:
        """Check if marker has low confidence (< 0.4)."""
        return self.confidence < 0.4


class ToMMarkers(BaseModel):
    """Collection of Theory of Mind markers for an interaction."""

    markers: dict[ToMMarkerType, list[ToMMarker]] = Field(default_factory=dict)

    def add_marker(self, marker: ToMMarker) -> None:
        """Add a marker to the collection."""
        if marker.type not in self.markers:
            self.markers[marker.type] = []
        self.markers[marker.type].append(marker)

    @property
    def total_count(self) -> int:
        """Count total number of markers detected."""
        return sum(len(markers) for markers in self.markers.values())

    @property
    def unique_types(self) -> int:
        """Count unique marker types detected."""
        return len(self.markers)

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across all markers."""
        all_markers = [m for markers in self.markers.values() for m in markers]
        if not all_markers:
            return 0.0
        return sum(m.confidence for m in all_markers) / len(all_markers)


class ResponseQuality(BaseModel):
    """Quality metrics for an agent response."""

    relevance: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    helpfulness: float = Field(ge=0.0, le=1.0)
    tom_awareness: float = Field(ge=0.0, le=1.0)

    model_config = ConfigDict(frozen=True)

    @property
    def overall(self) -> float:
        """Calculate overall quality score as average of all metrics."""
        return (self.relevance + self.clarity + self.helpfulness + self.tom_awareness) / 4.0

    @property
    def is_high_quality(self) -> bool:
        """Check if response is high quality (overall >= 0.7)."""
        return self.overall >= 0.7

    @property
    def is_low_quality(self) -> bool:
        """Check if response is low quality (overall < 0.4)."""
        return self.overall < 0.4


class SynergyScore(BaseModel):
    """Synergy score for human-AI interaction."""

    total: float = Field(ge=0.0, le=100.0)
    breakdown: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(frozen=True)

    @property
    def is_excellent(self) -> bool:
        """Check if synergy is excellent (>= 80)."""
        return self.total >= 80.0

    @property
    def is_good(self) -> bool:
        """Check if synergy is good (>= 60)."""
        return self.total >= 60.0

    @property
    def is_poor(self) -> bool:
        """Check if synergy is poor (< 40)."""
        return self.total < 40.0

    @property
    def component_count(self) -> int:
        """Count of components in breakdown."""
        return len(self.breakdown)


class UserPreferences(BaseModel):
    """User preferences for interaction style."""

    preferred_response_length: str = "medium"
    expertise_level: str = "intermediate"
    communication_style: str = "balanced"
    avg_response_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    interaction_count: int = Field(default=0, ge=0)
    detail_level: str = "balanced"
    prompt_history: list[str] = Field(default_factory=list)

    @property
    def is_beginner(self) -> bool:
        """Check if user is a beginner."""
        return self.expertise_level == "beginner"

    @property
    def is_expert(self) -> bool:
        """Check if user is an expert."""
        return self.expertise_level == "expert"

    @property
    def prefers_concise(self) -> bool:
        """Check if user prefers concise responses."""
        return self.preferred_response_length in {"short", "concise"}

    @property
    def prefers_detailed(self) -> bool:
        """Check if user prefers detailed responses."""
        return self.preferred_response_length in {"long", "detailed"}


class UserProfile(BaseModel):
    """User profile tracking preferences and interaction history."""

    id: UUID = Field(default_factory=uuid4)
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    last_interaction: Optional[datetime] = None
    tom_history: list[ToMMarkers] = Field(default_factory=list)

    def increment_interactions(self) -> None:
        """Increment interaction count and update last interaction time."""
        self.preferences.interaction_count += 1
        self.last_interaction = datetime.now()

    def add_tom_markers(self, markers: ToMMarkers) -> None:
        """Add ToM markers to history, maintaining last 100 entries."""
        self.tom_history.append(markers)
        if len(self.tom_history) > 100:
            self.tom_history.pop(0)

    @property
    def has_interaction_history(self) -> bool:
        """Check if user has any interaction history."""
        return self.preferences.interaction_count > 0

    @property
    def is_new_user(self) -> bool:
        """Check if user is new (< 5 interactions)."""
        return self.preferences.interaction_count < 5

    @property
    def is_experienced_user(self) -> bool:
        """Check if user is experienced (>= 20 interactions)."""
        return self.preferences.interaction_count >= 20

    @property
    def tom_history_size(self) -> int:
        """Count of ToM marker sets in history."""
        return len(self.tom_history)
