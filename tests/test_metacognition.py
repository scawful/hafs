"""Tests for metacognition monitoring and models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from models.metacognition import (
    CognitiveLoad,
    FlowStateIndicators,
    HelpSeeking,
    MetacognitiveState,
    ProgressStatus,
    SelfCorrection,
    SpinDetection,
    Strategy,
)
from core.metacognition.monitor import MetacognitionMonitor


class TestSpinDetection:
    """Tests for spin detection model and logic."""

    def test_spin_detection_defaults(self) -> None:
        """SpinDetection should have sensible defaults."""
        spin = SpinDetection()
        assert spin.spinning_threshold == 4
        assert spin.similar_action_count == 0
        assert spin.recent_actions == []
        assert not spin.is_spinning

    def test_spin_detection_triggers_at_threshold(self) -> None:
        """Spinning should be detected when similar_action_count >= threshold."""
        spin = SpinDetection(spinning_threshold=4)

        # Below threshold - not spinning
        spin.similar_action_count = 3
        assert not spin.is_spinning

        # At threshold - spinning
        spin.similar_action_count = 4
        assert spin.is_spinning

        # Above threshold - still spinning
        spin.similar_action_count = 5
        assert spin.is_spinning

    def test_spin_detection_configurable_threshold(self) -> None:
        """Spinning threshold should be configurable within bounds."""
        # Lower bound
        spin_low = SpinDetection(spinning_threshold=3)
        assert spin_low.spinning_threshold == 3

        # Upper bound
        spin_high = SpinDetection(spinning_threshold=5)
        assert spin_high.spinning_threshold == 5

    def test_spin_detection_threshold_validation(self) -> None:
        """Threshold outside 3-5 range should raise validation error."""
        with pytest.raises(ValueError):
            SpinDetection(spinning_threshold=2)

        with pytest.raises(ValueError):
            SpinDetection(spinning_threshold=6)


class TestCognitiveLoad:
    """Tests for cognitive load model."""

    def test_cognitive_load_defaults(self) -> None:
        """CognitiveLoad should have sensible defaults based on Miller's Law."""
        load = CognitiveLoad()
        assert load.max_recommended_items == 7
        assert load.warning_threshold == 0.8
        assert load.current == 0.0
        assert load.items_in_focus == 0

    def test_cognitive_load_calculation(self) -> None:
        """Load percentage should be calculated correctly."""
        load = CognitiveLoad(current=0.5)
        assert load.load_percentage == 50

        load.current = 0.75
        assert load.load_percentage == 75

        load.current = 1.0
        assert load.load_percentage == 100

    def test_cognitive_load_overload_detection(self) -> None:
        """Overload should be detected when current >= warning_threshold."""
        load = CognitiveLoad(warning_threshold=0.8)

        # Below threshold - not overloaded
        load.current = 0.79
        assert not load.is_overloaded

        # At threshold - overloaded
        load.current = 0.8
        assert load.is_overloaded

        # Above threshold - still overloaded
        load.current = 0.9
        assert load.is_overloaded

    def test_cognitive_load_bounds(self) -> None:
        """Load values should be constrained to 0.0-1.0."""
        with pytest.raises(ValueError):
            CognitiveLoad(current=-0.1)

        with pytest.raises(ValueError):
            CognitiveLoad(current=1.1)


class TestHelpSeeking:
    """Tests for help-seeking model."""

    def test_help_seeking_defaults(self) -> None:
        """HelpSeeking should have sensible defaults."""
        help_state = HelpSeeking()
        assert help_state.uncertainty_threshold == 0.3
        assert help_state.current_uncertainty == 0.0
        assert help_state.consecutive_failures == 0
        assert help_state.failure_threshold == 2

    def test_help_seeking_uncertainty_trigger(self) -> None:
        """Help should be sought when uncertainty exceeds threshold."""
        help_state = HelpSeeking(uncertainty_threshold=0.3)

        # Below threshold
        help_state.current_uncertainty = 0.29
        assert not help_state.should_ask_user

        # At threshold
        help_state.current_uncertainty = 0.3
        assert not help_state.should_ask_user  # Must exceed, not equal

        # Above threshold
        help_state.current_uncertainty = 0.31
        assert help_state.should_ask_user

    def test_help_seeking_failure_trigger(self) -> None:
        """Help should be sought after consecutive failures exceed threshold."""
        help_state = HelpSeeking(failure_threshold=2)

        # At threshold
        help_state.consecutive_failures = 2
        assert not help_state.should_ask_user  # Must exceed

        # Above threshold
        help_state.consecutive_failures = 3
        assert help_state.should_ask_user

    def test_help_seeking_combined_triggers(self) -> None:
        """Help seeking should trigger on either uncertainty OR failures."""
        help_state = HelpSeeking(uncertainty_threshold=0.3, failure_threshold=2)

        # High uncertainty, no failures
        help_state.current_uncertainty = 0.5
        help_state.consecutive_failures = 0
        assert help_state.should_ask_user

        # Low uncertainty, many failures
        help_state.current_uncertainty = 0.1
        help_state.consecutive_failures = 5
        assert help_state.should_ask_user


class TestFlowStateIndicators:
    """Tests for flow state criteria."""

    def test_flow_state_defaults(self) -> None:
        """FlowStateIndicators should have reasonable defaults."""
        indicators = FlowStateIndicators()
        assert indicators.min_progress_required is True
        assert indicators.max_cognitive_load == 0.7
        assert indicators.min_strategy_effectiveness == 0.6
        assert indicators.max_frustration == 0.3
        assert indicators.no_help_needed is True


class TestMetacognitiveState:
    """Tests for the complete metacognitive state model."""

    def test_metacognitive_state_defaults(self) -> None:
        """MetacognitiveState should initialize with sensible defaults."""
        state = MetacognitiveState()
        assert state.current_strategy == Strategy.INCREMENTAL
        assert state.strategy_effectiveness == 0.5
        assert state.progress_status == ProgressStatus.MAKING_PROGRESS
        assert state.flow_state is False

    def test_metacognitive_state_to_markdown(self) -> None:
        """State should serialize to markdown for state.md."""
        state = MetacognitiveState(
            current_strategy=Strategy.DIVIDE_AND_CONQUER,
            strategy_effectiveness=0.8,
            progress_status=ProgressStatus.MAKING_PROGRESS,
        )

        markdown = state.to_state_markdown()

        assert "## 9. Metacognitive Assessment" in markdown
        assert "divide_and_conquer" in markdown
        assert "0.80" in markdown
        assert "making_progress" in markdown

    def test_metacognitive_state_serialization(self) -> None:
        """State should serialize and deserialize correctly."""
        state = MetacognitiveState(
            current_strategy=Strategy.DEPTH_FIRST,
            strategy_effectiveness=0.7,
            progress_status=ProgressStatus.SPINNING,
        )

        # Serialize to JSON
        data = state.model_dump(mode="json")
        json_str = json.dumps(data, default=str)

        # Deserialize back
        loaded_data = json.loads(json_str)
        restored = MetacognitiveState.model_validate(loaded_data)

        assert restored.current_strategy == Strategy.DEPTH_FIRST
        assert restored.strategy_effectiveness == 0.7
        assert restored.progress_status == ProgressStatus.SPINNING


class TestMetacognitionMonitor:
    """Tests for the MetacognitionMonitor class."""

    def test_monitor_initialization(self, tmp_path: Path) -> None:
        """Monitor should initialize with clean state."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        assert not monitor.is_spinning
        assert not monitor.is_overloaded
        assert not monitor.should_seek_help
        assert not monitor.is_in_flow_state

    def test_spin_detection_via_actions(self, tmp_path: Path) -> None:
        """Monitor should detect spinning after repeated similar actions."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Different actions - not spinning
        monitor.record_action("edit file.py")
        assert not monitor.is_spinning

        monitor.record_action("read config.yaml")
        assert not monitor.is_spinning

        # Same action repeated consecutively - should trigger at threshold (4)
        # After "read config.yaml", "edit file.py" is different so counter = 0
        # Then 4 more "edit file.py" gives: 0, 1, 2, 3 (not spinning yet)
        # We need 5 total consecutive same actions to reach count=4
        for _ in range(5):
            monitor.record_action("edit file.py")

        assert monitor.is_spinning
        assert monitor.state.progress_status == ProgressStatus.SPINNING

    def test_spin_detection_reset_on_distinct_action(self, tmp_path: Path) -> None:
        """Spin counter should reset when a distinct action is taken."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Build up similar actions
        for _ in range(3):
            monitor.record_action("edit file.py")

        assert not monitor.is_spinning

        # Distinct action resets counter
        monitor.record_action("run tests")
        assert not monitor.is_spinning

        # Same actions but starting fresh
        for _ in range(3):
            monitor.record_action("edit file.py")

        assert not monitor.is_spinning  # Still under threshold

    def test_cognitive_load_calculation(self, tmp_path: Path) -> None:
        """Monitor should calculate cognitive load based on items in focus."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # 0 items = 0% load
        monitor.update_cognitive_load(0)
        assert monitor.state.cognitive_load.load_percentage == 0
        assert not monitor.is_overloaded

        # 7 items (Miller's max) = 100% load
        monitor.update_cognitive_load(7)
        assert monitor.state.cognitive_load.load_percentage == 100
        assert monitor.is_overloaded

        # 4 items ≈ 57% load
        monitor.update_cognitive_load(4)
        assert monitor.state.cognitive_load.load_percentage == 57
        assert not monitor.is_overloaded

    def test_flow_state_criteria(self, tmp_path: Path) -> None:
        """Flow state should only be active when all criteria are met."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Set up conditions for flow state
        monitor.set_strategy(Strategy.INCREMENTAL)
        monitor.update_strategy_effectiveness(0.3)  # Now at 0.8 (started at 0.5)
        monitor.update_cognitive_load(3)  # Low load
        monitor.record_success()  # Reset failures, reduce frustration

        is_in_flow = monitor.check_flow_state()
        assert is_in_flow
        assert monitor.is_in_flow_state

    def test_flow_state_blocked_by_spinning(self, tmp_path: Path) -> None:
        """Flow state should be blocked when spinning."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Set up good conditions
        monitor.update_strategy_effectiveness(0.3)
        monitor.update_cognitive_load(3)

        # But agent is spinning
        for _ in range(5):
            monitor.record_action("edit file.py")

        assert monitor.is_spinning
        is_in_flow = monitor.check_flow_state()
        assert not is_in_flow

    def test_flow_state_blocked_by_high_load(self, tmp_path: Path) -> None:
        """Flow state should be blocked when cognitive load is too high."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Set up good conditions
        monitor.update_strategy_effectiveness(0.3)

        # But load is too high
        monitor.update_cognitive_load(6)  # ~85% load

        is_in_flow = monitor.check_flow_state()
        assert not is_in_flow

    def test_strategy_effectiveness_decay(self, tmp_path: Path) -> None:
        """Strategy effectiveness should update based on outcomes."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        initial = monitor.state.strategy_effectiveness  # 0.5

        # Negative outcome decreases effectiveness
        monitor.update_strategy_effectiveness(-0.2)
        assert monitor.state.strategy_effectiveness == initial - 0.2

        # Positive outcome increases effectiveness
        monitor.update_strategy_effectiveness(0.4)
        assert monitor.state.strategy_effectiveness == initial + 0.2

    def test_strategy_effectiveness_bounds(self, tmp_path: Path) -> None:
        """Strategy effectiveness should be bounded to 0.0-1.0."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Try to go below 0
        monitor.update_strategy_effectiveness(-1.0)
        assert monitor.state.strategy_effectiveness >= 0.0

        monitor.reset()

        # Try to go above 1
        monitor.update_strategy_effectiveness(1.0)
        assert monitor.state.strategy_effectiveness <= 1.0

    def test_help_seeking_threshold(self, tmp_path: Path) -> None:
        """Help seeking should trigger based on uncertainty or failures."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Initially no help needed
        assert not monitor.should_seek_help

        # High uncertainty triggers help seeking
        monitor.update_uncertainty(0.5)
        assert monitor.should_seek_help

        # Reset uncertainty
        monitor.update_uncertainty(0.0)
        assert not monitor.should_seek_help

        # Multiple failures trigger help seeking
        for _ in range(3):
            monitor.record_failure()
        assert monitor.should_seek_help

    def test_self_correction_recording(self, tmp_path: Path) -> None:
        """Self-corrections should be recorded and limited."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Record a correction
        monitor.record_self_correction(
            what="Used wrong API endpoint",
            why="Didn't check docs first",
            outcome="Fixed and working",
        )

        assert len(monitor.state.self_corrections) == 1
        assert monitor.state.self_corrections[0].what == "Used wrong API endpoint"

        # Recording many corrections should trim to 10
        for i in range(15):
            monitor.record_self_correction(
                what=f"Correction {i}",
                why=f"Reason {i}",
            )

        assert len(monitor.state.self_corrections) == 10

    def test_state_persistence(self, tmp_path: Path) -> None:
        """State should persist across monitor instances."""
        state_file = tmp_path / "metacognition.json"

        # Create and modify state
        monitor1 = MetacognitionMonitor(state_path=state_file)
        monitor1.set_strategy(Strategy.DEPTH_FIRST)
        monitor1.update_cognitive_load(5)
        monitor1.save_state()

        # Create new instance and load
        monitor2 = MetacognitionMonitor(state_path=state_file)
        loaded = monitor2.load_state()

        assert loaded is True
        assert monitor2.state.current_strategy == Strategy.DEPTH_FIRST
        assert monitor2.state.cognitive_load.items_in_focus == 5

    def test_status_summary(self, tmp_path: Path) -> None:
        """Status summary should contain all key indicators."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        monitor.set_strategy(Strategy.BREADTH_FIRST)
        monitor.update_cognitive_load(4)

        summary = monitor.get_status_summary()

        assert summary["strategy"] == "breadth_first"
        assert "strategy_effectiveness" in summary
        assert "progress" in summary
        assert "cognitive_load" in summary
        assert "is_spinning" in summary
        assert "flow_state" in summary

    def test_reset(self, tmp_path: Path) -> None:
        """Reset should restore all state to defaults."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Modify state
        monitor.set_strategy(Strategy.DIVIDE_AND_CONQUER)
        for _ in range(5):
            monitor.record_action("edit file.py")
        monitor.update_cognitive_load(6)
        monitor.record_failure()

        assert monitor.is_spinning

        # Reset
        monitor.reset()

        assert monitor.state.current_strategy == Strategy.INCREMENTAL
        assert not monitor.is_spinning
        assert monitor.state.cognitive_load.current == 0.0

    def test_flow_state_change_detection(self, tmp_path: Path) -> None:
        """Flow state changes should be detectable."""
        state_file = tmp_path / "metacognition.json"
        monitor = MetacognitionMonitor(state_path=state_file)

        # Initially not in flow
        is_flow, changed = monitor.get_flow_state_changed()
        # First check - state changes from default

        # Set up flow conditions
        monitor.update_strategy_effectiveness(0.3)
        monitor.update_cognitive_load(3)
        monitor.record_success()

        is_flow, changed = monitor.get_flow_state_changed()
        assert is_flow is True
        assert changed is True

        # Check again - should not have changed
        is_flow, changed = monitor.get_flow_state_changed()
        assert is_flow is True
        assert changed is False


class TestStrategyEnum:
    """Tests for Strategy enum."""

    def test_all_strategies_have_descriptions(self) -> None:
        """All strategies should have descriptions."""
        from models.metacognition import STRATEGY_DESCRIPTIONS

        for strategy in Strategy:
            assert strategy in STRATEGY_DESCRIPTIONS
            assert len(STRATEGY_DESCRIPTIONS[strategy]) > 0


class TestProgressStatusEnum:
    """Tests for ProgressStatus enum."""

    def test_progress_status_values(self) -> None:
        """ProgressStatus should have expected values."""
        assert ProgressStatus.MAKING_PROGRESS.value == "making_progress"
        assert ProgressStatus.SPINNING.value == "spinning"
        assert ProgressStatus.BLOCKED.value == "blocked"
