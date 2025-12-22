"""Goal hierarchy management and conflict detection."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.config.loader import CognitiveProtocolConfig, get_config
from core.protocol.io_manager import get_io_manager
from core.protocol.validation import validate_goals_file
from models.goals import (
    Goal,
    GoalConflict,
    GoalHierarchy,
    GoalPriority,
    GoalStatus,
    InstrumentalGoal,
    PrimaryGoal,
    Subgoal,
)

if TYPE_CHECKING:
    pass


def get_conflict_patterns_dict(config: CognitiveProtocolConfig | None = None) -> dict:
    """Get conflict patterns from config as a dictionary.

    Args:
        config: Cognitive protocol configuration. If None, uses default config.

    Returns:
        Dictionary of conflict patterns keyed by pattern name.
    """
    cfg = config or get_config()
    patterns_dict = {}

    for pattern in cfg.goals.conflict_patterns:
        pattern_name = pattern.get("name", "")
        if pattern_name:
            patterns_dict[pattern_name] = {
                "keywords_a": pattern.get("keywords_a", []),
                "keywords_b": pattern.get("keywords_b", []),
                "description": pattern.get("description", ""),
            }

    return patterns_dict


# For backward compatibility, create default patterns using global config
CONFLICT_PATTERNS = get_conflict_patterns_dict()


class GoalManager:
    """Manages the goal hierarchy for the cognitive protocol.

    This class is responsible for:
    - Setting and tracking the primary goal
    - Decomposing goals into subgoals
    - Managing instrumental goals
    - Detecting and recording conflicts
    - Maintaining the goal stack (current focus)
    - Persisting goal state

    Example:
        manager = GoalManager()
        manager.set_primary_goal("Implement user authentication")
        manager.add_subgoal("Design database schema", parent_id=manager.hierarchy.primary_goal.id)
        manager.add_subgoal("Create login endpoint", parent_id=manager.hierarchy.primary_goal.id)
        manager.push_focus(subgoal_id)
        manager.save_state()
    """

    def __init__(
        self,
        state_path: Path | None = None,
        config: CognitiveProtocolConfig | None = None,
    ) -> None:
        """Initialize the goal manager.

        Args:
            state_path: Path to goals.json. Defaults to
                        .context/scratchpad/goals.json
            config: Cognitive protocol configuration. If None, uses default config.
        """
        self._config = config or get_config()
        self._state_path = state_path or (Path.cwd() / ".context" / "scratchpad" / "goals.json")
        self._hierarchy = GoalHierarchy()
        self._conflict_patterns = get_conflict_patterns_dict(self._config)

    @property
    def hierarchy(self) -> GoalHierarchy:
        """Get the current goal hierarchy."""
        return self._hierarchy

    @property
    def primary_goal(self) -> PrimaryGoal | None:
        """Get the primary goal."""
        return self._hierarchy.primary_goal

    @property
    def current_focus(self) -> Goal | None:
        """Get the currently focused goal."""
        focus_id = self._hierarchy.current_focus
        if focus_id:
            return self._hierarchy.get_goal_by_id(focus_id)
        return self._hierarchy.primary_goal

    @property
    def completion_percentage(self) -> float:
        """Get overall completion percentage."""
        return self._hierarchy.completion_percentage

    def load_state(self, validate: bool = False, auto_fix: bool = True) -> bool:
        """Load goal hierarchy from file.

        Args:
            validate: If True, use schema validation with Pydantic.
            auto_fix: If True and validation fails, use defaults (only if validate=True).

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        if not self._state_path.exists():
            return False

        try:
            if validate:
                # Use schema validation
                self._hierarchy = validate_goals_file(
                    self._state_path, auto_fix=auto_fix
                )
                return True

            # Legacy loading without strict validation
            io_manager = get_io_manager()
            data = io_manager.read_json(self._state_path)
            self._hierarchy = GoalHierarchy.model_validate(data)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def save_state(self, immediate: bool = False) -> bool:
        """Save goal hierarchy to file.

        Args:
            immediate: If True, write immediately. Otherwise batch the write.

        Returns:
            True if state was saved successfully, False otherwise.
        """
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._hierarchy.last_updated = datetime.now()

            data = self._hierarchy.model_dump(mode="json")
            io_manager = get_io_manager()
            io_manager.write_json(self._state_path, data, immediate=immediate)
            return True
        except OSError:
            return False

    def set_primary_goal(
        self,
        description: str,
        user_stated: str = "",
        success_criteria: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> PrimaryGoal:
        """Set or replace the primary goal.

        Args:
            description: What the goal aims to achieve.
            user_stated: The user's original statement.
            success_criteria: Criteria for completion.
            constraints: Constraints to respect.

        Returns:
            The created primary goal.
        """
        self._hierarchy.primary_goal = PrimaryGoal(
            description=description,
            user_stated=user_stated or description,
            success_criteria=success_criteria or [],
            constraints=constraints or [],
            status=GoalStatus.IN_PROGRESS,
        )

        # Clear existing subgoals and stack when primary changes
        self._hierarchy.subgoals = []
        self._hierarchy.goal_stack = []

        return self._hierarchy.primary_goal

    def add_subgoal(
        self,
        description: str,
        parent_id: str,
        dependencies: list[str] | None = None,
        priority: GoalPriority = GoalPriority.MEDIUM,
        estimated_effort: str = "",
    ) -> Subgoal:
        """Add a subgoal to the hierarchy.

        Args:
            description: What this subgoal achieves.
            parent_id: ID of parent goal.
            dependencies: IDs of goals that must complete first.
            priority: Priority level.
            estimated_effort: Effort estimate.

        Returns:
            The created subgoal.
        """
        subgoal = Subgoal(
            description=description,
            parent_id=parent_id,
            dependencies=dependencies or [],
            priority=priority,
            estimated_effort=estimated_effort,
        )
        self._hierarchy.subgoals.append(subgoal)

        # Check for conflicts with existing goals
        self._detect_conflicts_for_goal(subgoal)

        return subgoal

    def add_instrumental_goal(
        self,
        description: str,
        supports: list[str],
        reusable: bool = True,
    ) -> InstrumentalGoal:
        """Add an instrumental goal.

        Args:
            description: What this goal achieves.
            supports: IDs of goals this supports.
            reusable: Whether the outcome can be reused.

        Returns:
            The created instrumental goal.
        """
        goal = InstrumentalGoal(
            description=description,
            supports=supports,
            reusable=reusable,
        )
        self._hierarchy.instrumental_goals.append(goal)
        return goal

    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """Update progress on a goal.

        Args:
            goal_id: ID of the goal.
            progress: New progress value (0-1).

        Returns:
            True if goal was found and updated.
        """
        goal = self._hierarchy.get_goal_by_id(goal_id)
        if goal:
            goal.update_progress(progress)
            self._update_parent_progress(goal)
            return True
        return False

    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed.

        Args:
            goal_id: ID of the goal to complete.

        Returns:
            True if goal was found and completed.
        """
        goal = self._hierarchy.get_goal_by_id(goal_id)
        if goal:
            goal.mark_completed()
            self._update_parent_progress(goal)

            # Remove from stack if present
            if goal_id in self._hierarchy.goal_stack:
                self._hierarchy.goal_stack.remove(goal_id)

            return True
        return False

    def block_goal(self, goal_id: str, reason: str = "") -> bool:
        """Mark a goal as blocked.

        Args:
            goal_id: ID of the goal.
            reason: Why it's blocked.

        Returns:
            True if goal was found and blocked.
        """
        goal = self._hierarchy.get_goal_by_id(goal_id)
        if goal:
            goal.mark_blocked(reason)
            return True
        return False

    def push_focus(self, goal_id: str) -> bool:
        """Push a goal onto the focus stack.

        Args:
            goal_id: ID of the goal to focus on.

        Returns:
            True if goal exists and was pushed.
        """
        if self._hierarchy.get_goal_by_id(goal_id):
            if goal_id not in self._hierarchy.goal_stack:
                self._hierarchy.goal_stack.append(goal_id)
            return True
        return False

    def pop_focus(self) -> str | None:
        """Pop the current focus from the stack.

        Returns:
            The ID of the popped goal, or None if stack empty.
        """
        if self._hierarchy.goal_stack:
            return self._hierarchy.goal_stack.pop()
        return None

    def _update_parent_progress(self, goal: Goal) -> None:
        """Update parent goal progress based on children."""
        if isinstance(goal, Subgoal):
            parent = self._hierarchy.get_goal_by_id(goal.parent_id)
            if parent:
                # Calculate progress from all sibling subgoals
                siblings = self._hierarchy.get_subgoals_for(goal.parent_id)
                if siblings:
                    avg_progress = sum(s.progress for s in siblings) / len(siblings)
                    parent.progress = avg_progress
                    parent.updated_at = datetime.now()

    def _detect_conflicts_for_goal(self, new_goal: Goal) -> list[GoalConflict]:
        """Detect conflicts between new goal and existing goals.

        Args:
            new_goal: The newly added goal.

        Returns:
            List of detected conflicts.
        """
        conflicts = []
        description_lower = new_goal.description.lower()

        for existing_goal in self._hierarchy.all_goals:
            if existing_goal.id == new_goal.id:
                continue

            existing_lower = existing_goal.description.lower()

            # Check against known conflict patterns
            for pattern_name, pattern in self._conflict_patterns.items():
                # Check if new goal matches pattern A and existing matches B
                new_matches_a = any(kw in description_lower for kw in pattern["keywords_a"])
                existing_matches_b = any(kw in existing_lower for kw in pattern["keywords_b"])

                new_matches_b = any(kw in description_lower for kw in pattern["keywords_b"])
                existing_matches_a = any(kw in existing_lower for kw in pattern["keywords_a"])

                if (new_matches_a and existing_matches_b) or (new_matches_b and existing_matches_a):
                    conflict = GoalConflict(
                        goal_a_id=new_goal.id,
                        goal_b_id=existing_goal.id,
                        conflict_type=pattern_name,
                        description=pattern["description"],
                    )
                    conflicts.append(conflict)
                    self._hierarchy.conflicts.append(conflict)

        return conflicts

    def detect_all_conflicts(self) -> list[GoalConflict]:
        """Scan all goals for conflicts.

        Returns:
            List of all detected conflicts.
        """
        # Clear existing unresolved conflicts
        self._hierarchy.conflicts = [c for c in self._hierarchy.conflicts if c.resolved]

        all_goals = self._hierarchy.all_goals
        conflicts = []

        for i, goal_a in enumerate(all_goals):
            for goal_b in all_goals[i + 1 :]:
                desc_a = goal_a.description.lower()
                desc_b = goal_b.description.lower()

                for pattern_name, pattern in self._conflict_patterns.items():
                    a_matches_first = any(kw in desc_a for kw in pattern["keywords_a"])
                    b_matches_second = any(kw in desc_b for kw in pattern["keywords_b"])

                    a_matches_second = any(kw in desc_a for kw in pattern["keywords_b"])
                    b_matches_first = any(kw in desc_b for kw in pattern["keywords_a"])

                    if (a_matches_first and b_matches_second) or (
                        a_matches_second and b_matches_first
                    ):
                        conflict = GoalConflict(
                            goal_a_id=goal_a.id,
                            goal_b_id=goal_b.id,
                            conflict_type=pattern_name,
                            description=pattern["description"],
                        )
                        conflicts.append(conflict)
                        self._hierarchy.conflicts.append(conflict)

        return conflicts

    def resolve_conflict(self, conflict_id: str, resolution: str) -> bool:
        """Mark a conflict as resolved.

        Args:
            conflict_id: ID of the conflict.
            resolution: How it was resolved.

        Returns:
            True if conflict was found and resolved.
        """
        for conflict in self._hierarchy.conflicts:
            if conflict.id == conflict_id:
                conflict.resolution = resolution
                conflict.resolved = True
                return True
        return False

    def get_next_actionable_goal(self) -> Goal | None:
        """Get the next goal that can be worked on.

        Finds a pending/in_progress goal that has no unmet dependencies.

        Returns:
            The next actionable goal, or None.
        """
        for subgoal in self._hierarchy.subgoals:
            if subgoal.status in (GoalStatus.PENDING, GoalStatus.IN_PROGRESS):
                # Check dependencies
                deps_met = all(self._is_goal_completed(dep_id) for dep_id in subgoal.dependencies)
                if deps_met:
                    return subgoal

        # Fall back to primary goal if no subgoals
        if self._hierarchy.primary_goal:
            if self._hierarchy.primary_goal.status in (
                GoalStatus.PENDING,
                GoalStatus.IN_PROGRESS,
            ):
                return self._hierarchy.primary_goal

        return None

    def _is_goal_completed(self, goal_id: str) -> bool:
        """Check if a goal is completed."""
        goal = self._hierarchy.get_goal_by_id(goal_id)
        return goal is not None and goal.status == GoalStatus.COMPLETED

    def get_status_summary(self) -> dict:
        """Get a summary of goal status.

        Returns:
            Dictionary with key status indicators.
        """
        return {
            "has_primary_goal": self._hierarchy.primary_goal is not None,
            "primary_goal": (
                self._hierarchy.primary_goal.description if self._hierarchy.primary_goal else None
            ),
            "completion_percentage": self.completion_percentage,
            "total_subgoals": len(self._hierarchy.subgoals),
            "completed_subgoals": sum(
                1 for s in self._hierarchy.subgoals if s.status == GoalStatus.COMPLETED
            ),
            "blocked_goals": len(self._hierarchy.get_blocked_goals()),
            "unresolved_conflicts": len(self._hierarchy.unresolved_conflicts),
            "current_focus": (self.current_focus.description if self.current_focus else None),
        }

    def reset(self) -> None:
        """Reset all goal state to defaults."""
        self._hierarchy = GoalHierarchy()
