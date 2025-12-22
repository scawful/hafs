"""Tests for goal hierarchy management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.goals import (
    Goal,
    GoalConflict,
    GoalHierarchy,
    GoalPriority,
    GoalStatus,
    GoalType,
    InstrumentalGoal,
    PrimaryGoal,
    Subgoal,
)
from core.goals.manager import GoalManager


class TestGoalModels:
    """Tests for goal Pydantic models."""

    def test_goal_status_enum(self) -> None:
        """GoalStatus should have expected values."""
        assert GoalStatus.PENDING.value == "pending"
        assert GoalStatus.IN_PROGRESS.value == "in_progress"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.BLOCKED.value == "blocked"
        assert GoalStatus.ABANDONED.value == "abandoned"

    def test_goal_priority_enum(self) -> None:
        """GoalPriority should have expected values."""
        assert GoalPriority.CRITICAL.value == "critical"
        assert GoalPriority.HIGH.value == "high"
        assert GoalPriority.MEDIUM.value == "medium"
        assert GoalPriority.LOW.value == "low"
        assert GoalPriority.DEFERRED.value == "deferred"

    def test_goal_type_enum(self) -> None:
        """GoalType should have expected values."""
        assert GoalType.PRIMARY.value == "primary"
        assert GoalType.SUBGOAL.value == "subgoal"
        assert GoalType.INSTRUMENTAL.value == "instrumental"


class TestPrimaryGoal:
    """Tests for PrimaryGoal model."""

    def test_primary_goal_defaults(self) -> None:
        """PrimaryGoal should have sensible defaults."""
        goal = PrimaryGoal(description="Implement feature X")
        assert goal.goal_type == GoalType.PRIMARY
        assert goal.status == GoalStatus.PENDING
        assert goal.priority == GoalPriority.MEDIUM
        assert goal.progress == 0.0
        assert goal.success_criteria == []
        assert goal.constraints == []

    def test_primary_goal_with_criteria(self) -> None:
        """PrimaryGoal should accept success criteria and constraints."""
        goal = PrimaryGoal(
            description="Implement auth",
            user_stated="Add login functionality",
            success_criteria=["Users can log in", "Sessions persist"],
            constraints=["Must use existing database"],
        )
        assert goal.user_stated == "Add login functionality"
        assert len(goal.success_criteria) == 2
        assert len(goal.constraints) == 1

    def test_goal_mark_in_progress(self) -> None:
        """Goal should transition to in_progress status."""
        goal = PrimaryGoal(description="Test")
        assert goal.status == GoalStatus.PENDING

        goal.mark_in_progress()
        assert goal.status == GoalStatus.IN_PROGRESS

    def test_goal_mark_completed(self) -> None:
        """Goal should transition to completed with 100% progress."""
        goal = PrimaryGoal(description="Test")
        goal.mark_completed()

        assert goal.status == GoalStatus.COMPLETED
        assert goal.progress == 1.0
        assert goal.completed_at is not None

    def test_goal_mark_blocked(self) -> None:
        """Goal should transition to blocked with reason in notes."""
        goal = PrimaryGoal(description="Test")
        goal.mark_blocked("Missing dependency")

        assert goal.status == GoalStatus.BLOCKED
        assert "Missing dependency" in goal.notes

    def test_goal_update_progress(self) -> None:
        """Progress should be bounded and auto-complete at 100%."""
        goal = PrimaryGoal(description="Test")

        goal.update_progress(0.5)
        assert goal.progress == 0.5
        assert goal.status == GoalStatus.PENDING

        goal.update_progress(1.0)
        assert goal.progress == 1.0
        assert goal.status == GoalStatus.COMPLETED

    def test_goal_progress_bounds(self) -> None:
        """Progress should be clamped to 0-1 range."""
        goal = PrimaryGoal(description="Test")

        goal.update_progress(-0.5)
        assert goal.progress == 0.0

        goal.update_progress(1.5)
        assert goal.progress == 1.0


class TestSubgoal:
    """Tests for Subgoal model."""

    def test_subgoal_defaults(self) -> None:
        """Subgoal should have correct defaults."""
        subgoal = Subgoal(
            description="Design API",
            parent_id="pg-001",
        )
        assert subgoal.goal_type == GoalType.SUBGOAL
        assert subgoal.parent_id == "pg-001"
        assert subgoal.dependencies == []

    def test_subgoal_with_dependencies(self) -> None:
        """Subgoal should track dependencies."""
        subgoal = Subgoal(
            description="Implement API",
            parent_id="pg-001",
            dependencies=["sg-design", "sg-database"],
            estimated_effort="medium",
        )
        assert len(subgoal.dependencies) == 2
        assert subgoal.estimated_effort == "medium"


class TestInstrumentalGoal:
    """Tests for InstrumentalGoal model."""

    def test_instrumental_goal_defaults(self) -> None:
        """InstrumentalGoal should have correct defaults."""
        goal = InstrumentalGoal(
            description="Understand codebase",
            supports=["sg-001", "sg-002"],
        )
        assert goal.goal_type == GoalType.INSTRUMENTAL
        assert goal.reusable is True
        assert len(goal.supports) == 2

    def test_instrumental_goal_not_reusable(self) -> None:
        """InstrumentalGoal can be marked as not reusable."""
        goal = InstrumentalGoal(
            description="Run one-time migration",
            supports=["sg-001"],
            reusable=False,
        )
        assert goal.reusable is False


class TestGoalHierarchy:
    """Tests for GoalHierarchy model."""

    def test_hierarchy_defaults(self) -> None:
        """GoalHierarchy should start empty."""
        hierarchy = GoalHierarchy()
        assert hierarchy.primary_goal is None
        assert hierarchy.subgoals == []
        assert hierarchy.instrumental_goals == []
        assert hierarchy.goal_stack == []
        assert hierarchy.conflicts == []

    def test_hierarchy_current_focus(self) -> None:
        """Current focus should return top of goal stack."""
        hierarchy = GoalHierarchy()
        assert hierarchy.current_focus is None

        hierarchy.goal_stack = ["g-1", "g-2", "g-3"]
        assert hierarchy.current_focus == "g-3"

    def test_hierarchy_all_goals(self) -> None:
        """all_goals should include all goal types."""
        hierarchy = GoalHierarchy(
            primary_goal=PrimaryGoal(description="Main"),
            subgoals=[
                Subgoal(description="Sub1", parent_id="main"),
                Subgoal(description="Sub2", parent_id="main"),
            ],
            instrumental_goals=[
                InstrumentalGoal(description="Instrumental", supports=["sub1"]),
            ],
        )

        all_goals = hierarchy.all_goals
        assert len(all_goals) == 4

    def test_hierarchy_active_goals(self) -> None:
        """active_goals should exclude completed and abandoned."""
        hierarchy = GoalHierarchy(
            primary_goal=PrimaryGoal(description="Main"),
            subgoals=[
                Subgoal(description="Done", parent_id="main", status=GoalStatus.COMPLETED),
                Subgoal(description="Active", parent_id="main", status=GoalStatus.IN_PROGRESS),
            ],
        )

        active = hierarchy.active_goals
        assert len(active) == 2  # Primary + Active subgoal

    def test_hierarchy_completion_percentage(self) -> None:
        """Completion percentage should reflect subgoal progress."""
        hierarchy = GoalHierarchy(
            primary_goal=PrimaryGoal(description="Main"),
            subgoals=[
                Subgoal(description="Sub1", parent_id="main", progress=1.0),
                Subgoal(description="Sub2", parent_id="main", progress=0.5),
            ],
        )

        # (1.0 + 0.5) / 2 * 100 = 75%
        assert hierarchy.completion_percentage == 75.0

    def test_hierarchy_get_goal_by_id(self) -> None:
        """Should find goals by ID."""
        primary = PrimaryGoal(description="Main")
        hierarchy = GoalHierarchy(primary_goal=primary)

        found = hierarchy.get_goal_by_id(primary.id)
        assert found is not None
        assert found.description == "Main"

        not_found = hierarchy.get_goal_by_id("nonexistent")
        assert not_found is None

    def test_hierarchy_to_summary_markdown(self) -> None:
        """Should generate markdown summary."""
        hierarchy = GoalHierarchy(
            primary_goal=PrimaryGoal(
                description="Implement feature",
                status=GoalStatus.IN_PROGRESS,
            ),
            subgoals=[
                Subgoal(
                    description="Design",
                    parent_id="main",
                    status=GoalStatus.COMPLETED,
                ),
                Subgoal(
                    description="Implement",
                    parent_id="main",
                    status=GoalStatus.IN_PROGRESS,
                ),
            ],
        )

        markdown = hierarchy.to_summary_markdown()

        assert "## Goal Hierarchy" in markdown
        assert "Implement feature" in markdown
        assert "[✓] Design" in markdown
        assert "[→] Implement" in markdown


class TestGoalManager:
    """Tests for GoalManager class."""

    def test_manager_initialization(self, tmp_path: Path) -> None:
        """Manager should initialize with empty hierarchy."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        assert manager.primary_goal is None
        assert manager.current_focus is None
        assert manager.completion_percentage == 0.0

    def test_set_primary_goal(self, tmp_path: Path) -> None:
        """Setting primary goal should create and activate it."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        goal = manager.set_primary_goal(
            description="Build authentication system",
            user_stated="Add login",
            success_criteria=["Users can log in"],
        )

        assert manager.primary_goal is not None
        assert manager.primary_goal.description == "Build authentication system"
        assert manager.primary_goal.status == GoalStatus.IN_PROGRESS

    def test_add_subgoal(self, tmp_path: Path) -> None:
        """Adding subgoal should attach to parent."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main goal")
        subgoal = manager.add_subgoal(
            description="Sub task",
            parent_id=primary.id,
            priority=GoalPriority.HIGH,
        )

        assert len(manager.hierarchy.subgoals) == 1
        assert subgoal.parent_id == primary.id
        assert subgoal.priority == GoalPriority.HIGH

    def test_add_instrumental_goal(self, tmp_path: Path) -> None:
        """Adding instrumental goal should track supports."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main")
        sub1 = manager.add_subgoal("Sub1", primary.id)
        sub2 = manager.add_subgoal("Sub2", primary.id)

        instrumental = manager.add_instrumental_goal(
            description="Research codebase",
            supports=[sub1.id, sub2.id],
        )

        assert len(manager.hierarchy.instrumental_goals) == 1
        assert len(instrumental.supports) == 2

    def test_update_goal_progress(self, tmp_path: Path) -> None:
        """Updating progress should propagate to parent."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main")
        sub1 = manager.add_subgoal("Sub1", primary.id)
        sub2 = manager.add_subgoal("Sub2", primary.id)

        manager.update_goal_progress(sub1.id, 1.0)
        manager.update_goal_progress(sub2.id, 0.5)

        # Parent progress should be average of children
        assert primary.progress == 0.75

    def test_complete_goal(self, tmp_path: Path) -> None:
        """Completing a goal should update status and remove from stack."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main")
        subgoal = manager.add_subgoal("Sub", primary.id)
        manager.push_focus(subgoal.id)

        assert subgoal.id in manager.hierarchy.goal_stack

        manager.complete_goal(subgoal.id)

        assert subgoal.status == GoalStatus.COMPLETED
        assert subgoal.id not in manager.hierarchy.goal_stack

    def test_focus_stack_operations(self, tmp_path: Path) -> None:
        """Push and pop should manage focus correctly."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main")
        sub1 = manager.add_subgoal("Sub1", primary.id)
        sub2 = manager.add_subgoal("Sub2", primary.id)

        manager.push_focus(sub1.id)
        assert manager.current_focus.id == sub1.id

        manager.push_focus(sub2.id)
        assert manager.current_focus.id == sub2.id

        popped = manager.pop_focus()
        assert popped == sub2.id
        assert manager.current_focus.id == sub1.id

    def test_conflict_detection_minimize_vs_refactor(self, tmp_path: Path) -> None:
        """Should detect minimize vs refactor conflict."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Make changes")
        manager.add_subgoal("Make minimal changes to fix bug", primary.id)
        conflicts = manager.add_subgoal("Refactor the entire module", primary.id)

        # Conflict should be detected
        assert len(manager.hierarchy.conflicts) > 0
        assert any(c.conflict_type == "minimize_vs_refactor" for c in manager.hierarchy.conflicts)

    def test_conflict_detection_speed_vs_quality(self, tmp_path: Path) -> None:
        """Should detect speed vs quality conflict."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Build feature")
        manager.add_subgoal("Get this done fast, it's urgent", primary.id)
        manager.add_subgoal("Make it comprehensive and thorough", primary.id)

        assert len(manager.hierarchy.conflicts) > 0
        assert any(c.conflict_type == "speed_vs_quality" for c in manager.hierarchy.conflicts)

    def test_resolve_conflict(self, tmp_path: Path) -> None:
        """Should be able to resolve conflicts."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Build")
        manager.add_subgoal("Quick fix", primary.id)
        manager.add_subgoal("Thorough solution", primary.id)

        assert len(manager.hierarchy.unresolved_conflicts) > 0

        conflict = manager.hierarchy.conflicts[0]
        manager.resolve_conflict(conflict.id, "Prioritize quality over speed")

        assert conflict.resolved is True
        assert len(manager.hierarchy.unresolved_conflicts) == 0

    def test_get_next_actionable_goal(self, tmp_path: Path) -> None:
        """Should return goal with met dependencies."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main")
        sub1 = manager.add_subgoal("First step", primary.id)
        sub2 = manager.add_subgoal("Second step", primary.id, dependencies=[sub1.id])

        # sub2 depends on sub1, so sub1 should be next
        next_goal = manager.get_next_actionable_goal()
        assert next_goal.id == sub1.id

        # Complete sub1
        manager.complete_goal(sub1.id)

        # Now sub2 should be next
        next_goal = manager.get_next_actionable_goal()
        assert next_goal.id == sub2.id

    def test_state_persistence(self, tmp_path: Path) -> None:
        """State should persist across manager instances."""
        state_file = tmp_path / "goals.json"

        # Create and save state
        manager1 = GoalManager(state_path=state_file)
        manager1.set_primary_goal("Persistent goal")
        manager1.add_subgoal("Subgoal", manager1.primary_goal.id)
        manager1.save_state()

        # Load in new instance
        manager2 = GoalManager(state_path=state_file)
        loaded = manager2.load_state()

        assert loaded is True
        assert manager2.primary_goal is not None
        assert manager2.primary_goal.description == "Persistent goal"
        assert len(manager2.hierarchy.subgoals) == 1

    def test_get_status_summary(self, tmp_path: Path) -> None:
        """Status summary should include key metrics."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        primary = manager.set_primary_goal("Main")
        sub1 = manager.add_subgoal("Sub1", primary.id)
        manager.complete_goal(sub1.id)
        manager.add_subgoal("Sub2", primary.id)

        summary = manager.get_status_summary()

        assert summary["has_primary_goal"] is True
        assert summary["primary_goal"] == "Main"
        assert summary["total_subgoals"] == 2
        assert summary["completed_subgoals"] == 1
        assert summary["completion_percentage"] == 50.0

    def test_reset(self, tmp_path: Path) -> None:
        """Reset should clear all state."""
        state_file = tmp_path / "goals.json"
        manager = GoalManager(state_path=state_file)

        manager.set_primary_goal("Main")
        manager.add_subgoal("Sub", manager.primary_goal.id)

        manager.reset()

        assert manager.primary_goal is None
        assert len(manager.hierarchy.subgoals) == 0


class TestGoalConflict:
    """Tests for GoalConflict model."""

    def test_conflict_defaults(self) -> None:
        """GoalConflict should have correct defaults."""
        conflict = GoalConflict(
            goal_a_id="g-1",
            goal_b_id="g-2",
            conflict_type="test",
            description="Test conflict",
        )

        assert conflict.resolved is False
        assert conflict.resolution == ""
        assert conflict.detected_at is not None

    def test_conflict_id_generation(self) -> None:
        """Conflict ID should be auto-generated."""
        conflict = GoalConflict(
            goal_a_id="g-1",
            goal_b_id="g-2",
            conflict_type="test",
            description="Test",
        )

        assert conflict.id.startswith("conflict-")
