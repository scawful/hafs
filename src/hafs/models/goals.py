"""Goal hierarchy models for cognitive protocol.

This module defines the data structures for tracking goals at multiple levels:
- Primary goals: The user's main request/objective
- Subgoals: Decomposed steps to achieve the primary goal
- Instrumental goals: Meta-goals that support multiple subgoals

The goal hierarchy enables:
- Progress tracking at multiple granularities
- Goal conflict detection
- Focus management via goal stack
- Theory of Mind by understanding user's true objectives
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class GoalStatus(str, Enum):
    """Status of a goal."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    ABANDONED = "abandoned"


class GoalPriority(str, Enum):
    """Priority level for goals."""

    CRITICAL = "critical"  # Must complete, blocking
    HIGH = "high"  # Important, should complete soon
    MEDIUM = "medium"  # Normal priority
    LOW = "low"  # Nice to have
    DEFERRED = "deferred"  # Explicitly postponed


class GoalType(str, Enum):
    """Type of goal in the hierarchy."""

    PRIMARY = "primary"  # User's main request
    SUBGOAL = "subgoal"  # Step toward primary goal
    INSTRUMENTAL = "instrumental"  # Supports multiple goals


class Goal(BaseModel):
    """Base goal model with common attributes."""

    id: str = Field(default_factory=lambda: f"g-{uuid4().hex[:8]}")
    description: str = Field(description="What this goal aims to achieve")
    goal_type: GoalType = Field(description="Type of goal in hierarchy")
    status: GoalStatus = Field(default=GoalStatus.PENDING)
    priority: GoalPriority = Field(default=GoalPriority.MEDIUM)
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress toward completion (0-1)",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    notes: str = Field(default="", description="Additional context or notes")

    model_config = ConfigDict(use_enum_values=False)

    def mark_in_progress(self) -> None:
        """Mark goal as in progress."""
        self.status = GoalStatus.IN_PROGRESS
        self.updated_at = datetime.now()

    def mark_completed(self) -> None:
        """Mark goal as completed."""
        self.status = GoalStatus.COMPLETED
        self.progress = 1.0
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()

    def mark_blocked(self, reason: str = "") -> None:
        """Mark goal as blocked."""
        self.status = GoalStatus.BLOCKED
        if reason:
            self.notes = f"{self.notes}\nBlocked: {reason}".strip()
        self.updated_at = datetime.now()

    def update_progress(self, progress: float) -> None:
        """Update progress value."""
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now()
        if self.progress >= 1.0:
            self.mark_completed()


class PrimaryGoal(Goal):
    """The user's main objective.

    A primary goal represents what the user ultimately wants to achieve.
    It may be decomposed into multiple subgoals.
    """

    goal_type: GoalType = Field(default=GoalType.PRIMARY)
    user_stated: str = Field(
        default="",
        description="The user's original statement of the goal",
    )
    success_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria for considering this goal complete",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints or requirements to respect",
    )


class Subgoal(Goal):
    """A step toward achieving a primary goal.

    Subgoals break down primary goals into actionable steps.
    They form a tree structure under the primary goal.
    """

    goal_type: GoalType = Field(default=GoalType.SUBGOAL)
    parent_id: str = Field(description="ID of parent goal (primary or subgoal)")
    dependencies: list[str] = Field(
        default_factory=list,
        description="IDs of goals that must complete before this one",
    )
    estimated_effort: str = Field(
        default="",
        description="Estimated effort (e.g., 'small', 'medium', 'large')",
    )


class InstrumentalGoal(Goal):
    """A meta-goal that supports multiple other goals.

    Instrumental goals are things like "understand the codebase" that
    aren't directly requested but enable other goals.
    """

    goal_type: GoalType = Field(default=GoalType.INSTRUMENTAL)
    supports: list[str] = Field(
        default_factory=list,
        description="IDs of goals this instrumental goal supports",
    )
    reusable: bool = Field(
        default=True,
        description="Whether this goal's outcome can be reused",
    )


class GoalConflict(BaseModel):
    """Represents a detected conflict between goals."""

    id: str = Field(default_factory=lambda: f"conflict-{uuid4().hex[:8]}")
    goal_a_id: str = Field(description="First conflicting goal")
    goal_b_id: str = Field(description="Second conflicting goal")
    conflict_type: str = Field(
        description="Type of conflict (resource, logical, temporal, priority)"
    )
    description: str = Field(description="Description of the conflict")
    resolution: str = Field(default="", description="How the conflict was resolved")
    resolved: bool = Field(default=False)
    detected_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(use_enum_values=False)


class GoalHierarchy(BaseModel):
    """Complete goal hierarchy state.

    This represents the full goal structure including:
    - The primary goal (what the user wants)
    - Subgoals (decomposed steps)
    - Instrumental goals (supporting meta-goals)
    - Goal stack (current focus)
    - Detected conflicts
    """

    primary_goal: PrimaryGoal | None = Field(
        default=None,
        description="The main user objective",
    )
    subgoals: list[Subgoal] = Field(
        default_factory=list,
        description="Steps toward the primary goal",
    )
    instrumental_goals: list[InstrumentalGoal] = Field(
        default_factory=list,
        description="Meta-goals supporting multiple goals",
    )
    goal_stack: list[str] = Field(
        default_factory=list,
        description="Stack of goal IDs representing current focus (top = active)",
    )
    conflicts: list[GoalConflict] = Field(
        default_factory=list,
        description="Detected goal conflicts",
    )
    last_updated: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(use_enum_values=False)

    @property
    def current_focus(self) -> str | None:
        """Get the ID of the currently focused goal."""
        return self.goal_stack[-1] if self.goal_stack else None

    @property
    def all_goals(self) -> list[Goal]:
        """Get all goals in the hierarchy."""
        goals: list[Goal] = []
        if self.primary_goal:
            goals.append(self.primary_goal)
        goals.extend(self.subgoals)
        goals.extend(self.instrumental_goals)
        return goals

    @property
    def active_goals(self) -> list[Goal]:
        """Get all non-completed, non-abandoned goals."""
        return [
            g
            for g in self.all_goals
            if g.status not in (GoalStatus.COMPLETED, GoalStatus.ABANDONED)
        ]

    @property
    def completion_percentage(self) -> float:
        """Calculate overall completion percentage."""
        if not self.primary_goal:
            return 0.0
        # Weight: primary goal progress + average of subgoal progress
        if not self.subgoals:
            return self.primary_goal.progress * 100
        subgoal_progress = sum(s.progress for s in self.subgoals) / len(self.subgoals)
        # Primary goal progress is derived from subgoals
        return subgoal_progress * 100

    @property
    def unresolved_conflicts(self) -> list[GoalConflict]:
        """Get conflicts that haven't been resolved."""
        return [c for c in self.conflicts if not c.resolved]

    def get_goal_by_id(self, goal_id: str) -> Goal | None:
        """Find a goal by its ID."""
        for goal in self.all_goals:
            if goal.id == goal_id:
                return goal
        return None

    def get_subgoals_for(self, parent_id: str) -> list[Subgoal]:
        """Get all subgoals that have the given parent."""
        return [s for s in self.subgoals if s.parent_id == parent_id]

    def get_blocked_goals(self) -> list[Goal]:
        """Get all goals that are blocked."""
        return [g for g in self.all_goals if g.status == GoalStatus.BLOCKED]

    def to_summary_markdown(self) -> str:
        """Generate markdown summary for state.md."""
        if not self.primary_goal:
            return "## Goal Hierarchy\n\nNo primary goal set."

        lines = [
            "## Goal Hierarchy",
            f"**Primary Goal:** {self.primary_goal.description}",
            f"**Status:** {self.primary_goal.status.value}",
            f"**Progress:** {self.completion_percentage:.0f}%",
            "",
            "### Subgoals:",
        ]

        for subgoal in self.subgoals:
            status_icon = {
                GoalStatus.COMPLETED: "✓",
                GoalStatus.IN_PROGRESS: "→",
                GoalStatus.BLOCKED: "✗",
                GoalStatus.PENDING: "○",
                GoalStatus.ABANDONED: "⊘",
            }.get(subgoal.status, "?")
            lines.append(f"- [{status_icon}] {subgoal.description}")

        if self.instrumental_goals:
            lines.extend(["", "### Instrumental Goals:"])
            for ig in self.instrumental_goals:
                lines.append(f"- {ig.description} (supports {len(ig.supports)} goals)")

        if self.unresolved_conflicts:
            lines.extend(["", "### ⚠️ Conflicts:"])
            for conflict in self.unresolved_conflicts:
                lines.append(f"- {conflict.description}")

        return "\n".join(lines)
