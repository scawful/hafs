"""Metacognition monitoring and self-assessment logic."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from core.config.loader import CognitiveProtocolConfig, get_config
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

if TYPE_CHECKING:
    pass


class MetacognitionMonitor:
    """Monitors agent's cognitive processes and detects issues.

    This class is responsible for:
    - Detecting "spinning" behavior (repeated similar actions)
    - Tracking cognitive load
    - Assessing progress toward goals
    - Determining when to seek help
    - Evaluating strategy effectiveness
    - Detecting flow state conditions

    Example:
        monitor = MetacognitionMonitor()
        monitor.load_state()
        monitor.record_action("edit file.py")
        if monitor.is_spinning():
            print("Warning: Possible spinning detected")
        monitor.save_state()
    """

    def __init__(
        self,
        state_path: Path | None = None,
        max_action_history: int | None = None,
        config: CognitiveProtocolConfig | None = None,
    ) -> None:
        """Initialize the metacognition monitor.

        Args:
            state_path: Path to metacognition.json. Defaults to
                        .context/scratchpad/metacognition.json
            max_action_history: Maximum number of recent actions to track.
                                If None, uses config value.
            config: Cognitive protocol configuration. If None, uses default config.
        """
        self._config = config or get_config()
        self._state_path = state_path or (
            Path.cwd() / ".context" / "scratchpad" / "metacognition.json"
        )
        self._max_action_history = (
            max_action_history
            if max_action_history is not None
            else self._config.metacognition.max_action_history
        )
        self._state = MetacognitiveState()
        self._frustration_level: float = 0.0  # Track for flow state calculation
        self._wire_format: str = "snake"
        self._wire_extras: dict[str, object] = {}

    @property
    def state(self) -> MetacognitiveState:
        """Get current metacognitive state."""
        return self._state

    @property
    def is_spinning(self) -> bool:
        """Check if agent is currently spinning."""
        return self._state.spin_detection.is_spinning

    @property
    def is_overloaded(self) -> bool:
        """Check if cognitive load is too high."""
        return self._state.cognitive_load.is_overloaded

    @property
    def should_seek_help(self) -> bool:
        """Check if agent should ask user for help."""
        return self._state.help_seeking.should_ask_user

    @property
    def is_in_flow_state(self) -> bool:
        """Check if agent is currently in flow state."""
        return self._state.flow_state

    def load_state(self) -> bool:
        """Load metacognitive state from file.

        Returns:
            True if state was loaded successfully, False otherwise.
        """
        if not self._state_path.exists():
            return False

        try:
            content = self._state_path.read_text(encoding="utf-8", errors="replace")
            raw = json.loads(content)
            if not isinstance(raw, dict):
                return False

            from core.protocol.metacognition_compat import (
                detect_wire_format,
                known_top_level_keys,
                normalize_metacognition,
            )

            self._wire_format = detect_wire_format(raw)
            normalized = normalize_metacognition(raw)

            frustration = normalized.get("frustration_level")
            if isinstance(frustration, (int, float)):
                self._frustration_level = float(frustration)

            # Preserve unknown top-level keys so we don't clobber oracle-code data.
            known_raw = known_top_level_keys("camel" if self._wire_format == "camel" else "snake")
            self._wire_extras = {k: v for k, v in raw.items() if k not in known_raw}

            self._state = MetacognitiveState.model_validate(normalized)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def save_state(self) -> bool:
        """Save metacognitive state to file.

        Returns:
            True if state was saved successfully, False otherwise.
        """
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state.last_updated = datetime.now()

            base = self._state.model_dump(mode="json")
            base["frustration_level"] = self._frustration_level

            from core.protocol.metacognition_compat import denormalize_metacognition

            payload = denormalize_metacognition(
                base, wire_format="camel" if self._wire_format == "camel" else "snake"
            )

            # Merge any unknown top-level keys we saw on load.
            merged: dict[str, object] = dict(self._wire_extras)
            merged.update(payload)

            self._state_path.write_text(
                json.dumps(merged, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            return True
        except OSError:
            return False

    def record_action(self, action_description: str) -> None:
        """Record an action for spin detection.

        Args:
            action_description: Description of the action taken.
        """
        action_signature = self._compute_action_signature(action_description)
        recent = self._state.spin_detection.recent_actions

        # Check similarity to recent actions
        if recent and self._is_similar_action(action_signature, recent[-1]):
            self._state.spin_detection.similar_action_count += 1
        else:
            self._state.spin_detection.similar_action_count = 0
            self._state.spin_detection.last_distinct_action_time = datetime.now()

        # Maintain action history
        recent.append(action_signature)
        if len(recent) > self._max_action_history:
            self._state.spin_detection.recent_actions = recent[-self._max_action_history :]

        # Update progress status based on spinning
        self._update_progress_status()

    def _compute_action_signature(self, action: str) -> str:
        """Compute a normalized signature for an action.

        Args:
            action: Action description.

        Returns:
            Normalized action signature for comparison.
        """
        # Normalize: lowercase, remove extra whitespace, hash for privacy
        normalized = " ".join(action.lower().split())
        # Use first 8 chars of hash for comparison
        return hashlib.sha256(normalized.encode()).hexdigest()[:8]

    def _is_similar_action(self, sig1: str, sig2: str) -> bool:
        """Check if two action signatures are similar.

        For now, exact match. Could be extended to fuzzy matching.

        Args:
            sig1: First action signature.
            sig2: Second action signature.

        Returns:
            True if actions are similar.
        """
        return sig1 == sig2

    def _update_progress_status(self) -> None:
        """Update progress status based on current state."""
        if self._state.spin_detection.is_spinning:
            self._state.progress_status = ProgressStatus.SPINNING
        elif self._state.help_seeking.should_ask_user:
            self._state.progress_status = ProgressStatus.BLOCKED
        else:
            self._state.progress_status = ProgressStatus.MAKING_PROGRESS

    def update_cognitive_load(self, items_in_focus: int) -> None:
        """Update cognitive load based on items being tracked.

        Args:
            items_in_focus: Number of items currently in working memory.
        """
        max_items = self._state.cognitive_load.max_recommended_items
        self._state.cognitive_load.items_in_focus = items_in_focus
        self._state.cognitive_load.current = min(1.0, items_in_focus / max_items)

    def record_failure(self) -> None:
        """Record a failed attempt, incrementing failure counter."""
        self._state.help_seeking.consecutive_failures += 1
        self._frustration_level = min(1.0, self._frustration_level + 0.2)
        self._update_progress_status()

    def record_success(self) -> None:
        """Record a successful action, resetting failure counter."""
        self._state.help_seeking.consecutive_failures = 0
        self._frustration_level = max(0.0, self._frustration_level - 0.3)
        self._state.spin_detection.similar_action_count = 0

    def update_uncertainty(self, uncertainty: float) -> None:
        """Update current uncertainty level.

        Args:
            uncertainty: Uncertainty level from 0.0 to 1.0.
        """
        self._state.help_seeking.current_uncertainty = max(0.0, min(1.0, uncertainty))
        self._update_progress_status()

    def set_strategy(self, strategy: Strategy) -> None:
        """Set the current problem-solving strategy.

        Args:
            strategy: The strategy to use.
        """
        if strategy != self._state.current_strategy:
            self._state.current_strategy = strategy
            self._state.strategy_effectiveness = 0.5  # Reset effectiveness

    def update_strategy_effectiveness(self, delta: float) -> None:
        """Update strategy effectiveness based on outcomes.

        Args:
            delta: Change in effectiveness (-1.0 to 1.0).
        """
        new_effectiveness = self._state.strategy_effectiveness + delta
        self._state.strategy_effectiveness = max(0.0, min(1.0, new_effectiveness))

    def record_self_correction(self, what: str, why: str, outcome: str = "") -> None:
        """Record when the agent catches its own mistake.

        Args:
            what: What was wrong.
            why: Why it happened.
            outcome: Result of the correction.
        """
        correction = SelfCorrection(
            id=str(uuid4())[:8],
            what=what,
            why=why,
            outcome=outcome,
        )
        self._state.self_corrections.append(correction)

        # Keep only recent corrections (last 10)
        if len(self._state.self_corrections) > 10:
            self._state.self_corrections = self._state.self_corrections[-10:]

    def check_flow_state(self) -> bool:
        """Check if conditions for flow state are met.

        Returns:
            True if in flow state, False otherwise.
        """
        indicators = self._state.flow_state_indicators

        in_flow = (
            # Must be making progress
            (
                not indicators.min_progress_required
                or self._state.progress_status == ProgressStatus.MAKING_PROGRESS
            )
            # Cognitive load under threshold
            and self._state.cognitive_load.current < indicators.max_cognitive_load
            # Strategy working well
            and self._state.strategy_effectiveness >= indicators.min_strategy_effectiveness
            # Not frustrated
            and self._frustration_level < indicators.max_frustration
            # Doesn't need help
            and (not indicators.no_help_needed or not self._state.help_seeking.should_ask_user)
        )

        # Update flow state
        was_in_flow = self._state.flow_state
        self._state.flow_state = in_flow

        return in_flow

    def get_flow_state_changed(self) -> tuple[bool, bool]:
        """Check flow state and return if it changed.

        Returns:
            Tuple of (is_in_flow, state_changed).
        """
        was_in_flow = self._state.flow_state
        is_in_flow = self.check_flow_state()
        return is_in_flow, was_in_flow != is_in_flow

    def reset_spin_detection(self) -> None:
        """Reset spin detection state."""
        self._state.spin_detection = SpinDetection()

    def reset(self) -> None:
        """Reset all metacognitive state to defaults."""
        self._state = MetacognitiveState()
        self._frustration_level = 0.0

    def get_status_summary(self) -> dict:
        """Get a summary of current metacognitive status.

        Returns:
            Dictionary with key status indicators.
        """
        return {
            "strategy": self._state.current_strategy.value,
            "strategy_effectiveness": self._state.strategy_effectiveness,
            "progress": self._state.progress_status.value,
            "cognitive_load": self._state.cognitive_load.load_percentage,
            "is_spinning": self.is_spinning,
            "is_overloaded": self.is_overloaded,
            "should_seek_help": self.should_seek_help,
            "flow_state": self._state.flow_state,
            "frustration": self._frustration_level,
        }
