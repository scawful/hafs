"""Analysis Triggers - Automatic invocation of analysis modes.

Implements the trigger system per PROTOCOL_SPEC.md Section 5.3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field


class AnalysisGateMode(str, Enum):
    """Control how triggers are handled per PROTOCOL_SPEC.md Section 5.4."""

    CONFIRM_ALL = "confirm-all"   # All triggers require user confirmation (default)
    AUTO_ACCEPT = "auto-accept"   # Flow-friendly automatic acceptance
    AUTO_DENY = "auto-deny"       # Silently ignore all triggers
    SELECTIVE = "selective"       # Per-trigger configuration


class TriggerConditionType(str, Enum):
    """Types of trigger conditions."""

    THRESHOLD = "threshold"  # Metric exceeds threshold
    PATTERN = "pattern"      # Regex pattern match
    COUNT = "count"          # Event count within window
    TIME = "time"            # Time-based trigger


class TriggerPriority(str, Enum):
    """Trigger priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerCondition(BaseModel):
    """Condition for triggering analysis."""

    type: TriggerConditionType
    metric: Optional[str] = None
    threshold: Optional[float] = None
    pattern: Optional[str] = None
    count: Optional[int] = None
    within_ms: Optional[int] = None


class TriggerAction(BaseModel):
    """Action to take when trigger fires."""

    mode: str  # AnalysisMode value
    agent: Optional[str] = None
    auto_accept: bool = False
    priority: TriggerPriority = TriggerPriority.MEDIUM


class AnalysisTrigger(BaseModel):
    """Analysis trigger definition per PROTOCOL_SPEC.md Section 5.3."""

    id: str
    name: str
    description: str

    condition: TriggerCondition
    action: TriggerAction

    cooldown_ms: int = 0
    max_triggers_per_session: Optional[int] = None

    # State
    enabled: bool = True
    trigger_count: int = 0
    last_triggered: Optional[str] = None


# Default triggers from PROTOCOL_SPEC.md
DEFAULT_TRIGGERS = [
    AnalysisTrigger(
        id="spinning-critic",
        name="Spinning Detection",
        description="Activate critic when agent is spinning",
        condition=TriggerCondition(
            type=TriggerConditionType.PATTERN,
            metric="progress_status",
            pattern="spinning",
        ),
        action=TriggerAction(
            mode="critic",
            priority=TriggerPriority.HIGH,
        ),
    ),
    AnalysisTrigger(
        id="edits-without-tests",
        name="Edits Without Tests",
        description="Evaluate when edits made without tests",
        condition=TriggerCondition(
            type=TriggerConditionType.THRESHOLD,
            metric="edit_count",
            threshold=3,
        ),
        action=TriggerAction(
            mode="eval",
            priority=TriggerPriority.MEDIUM,
        ),
    ),
    AnalysisTrigger(
        id="high-anxiety-caution",
        name="High Anxiety Caution",
        description="Activate emotional analysis when anxiety is high",
        condition=TriggerCondition(
            type=TriggerConditionType.THRESHOLD,
            metric="anxiety_level",
            threshold=0.7,
        ),
        action=TriggerAction(
            mode="emotional",
            priority=TriggerPriority.HIGH,
        ),
    ),
    AnalysisTrigger(
        id="consecutive-failures",
        name="Consecutive Failures",
        description="Analyze metrics after consecutive failures",
        condition=TriggerCondition(
            type=TriggerConditionType.COUNT,
            metric="failures",
            count=3,
            within_ms=300000,  # 5 minutes
        ),
        action=TriggerAction(
            mode="metrics",
            priority=TriggerPriority.HIGH,
        ),
    ),
    AnalysisTrigger(
        id="high-cognitive-load",
        name="High Cognitive Load",
        description="Emotional analysis when cognitive load is high",
        condition=TriggerCondition(
            type=TriggerConditionType.THRESHOLD,
            metric="cognitive_load",
            threshold=0.8,
        ),
        action=TriggerAction(
            mode="emotional",
            priority=TriggerPriority.MEDIUM,
        ),
    ),
    AnalysisTrigger(
        id="tool-repetition",
        name="Tool Repetition",
        description="Critic when same tool called repeatedly",
        condition=TriggerCondition(
            type=TriggerConditionType.COUNT,
            metric="same_tool_calls",
            count=5,
            within_ms=60000,  # 1 minute
        ),
        action=TriggerAction(
            mode="critic",
            priority=TriggerPriority.LOW,
        ),
    ),
    AnalysisTrigger(
        id="baseline-too-high",
        name="Baseline Too High",
        description="Warn when multi-agent used but baseline is high",
        condition=TriggerCondition(
            type=TriggerConditionType.THRESHOLD,
            metric="baseline_accuracy",
            threshold=0.45,
        ),
        action=TriggerAction(
            mode="metrics",
            priority=TriggerPriority.MEDIUM,
        ),
    ),
    AnalysisTrigger(
        id="error-amplification-warning",
        name="Error Amplification Warning",
        description="Alert on high error amplification",
        condition=TriggerCondition(
            type=TriggerConditionType.THRESHOLD,
            metric="error_amplification",
            threshold=10,
        ),
        action=TriggerAction(
            mode="metrics",
            priority=TriggerPriority.HIGH,
        ),
    ),
]


class TriggerEvent(BaseModel):
    """Record of a trigger firing."""

    trigger_id: str
    timestamp: str
    metrics: dict[str, Any]
    action_taken: str
    accepted: bool


class TriggerManager:
    """Manages analysis triggers and their evaluation."""

    def __init__(
        self,
        gate_mode: AnalysisGateMode = AnalysisGateMode.CONFIRM_ALL,
        confirmation_callback: Optional[Callable[[AnalysisTrigger], bool]] = None,
    ) -> None:
        """Initialize the trigger manager.

        Args:
            gate_mode: How to handle trigger confirmations.
            confirmation_callback: Optional callback for user confirmation.
        """
        self.gate_mode = gate_mode
        self.confirmation_callback = confirmation_callback
        self.triggers: dict[str, AnalysisTrigger] = {}
        self.event_history: list[TriggerEvent] = []

        # Event counters for COUNT-type conditions
        self._event_counters: dict[str, list[float]] = {}

        # Load defaults
        for trigger in DEFAULT_TRIGGERS:
            self.register_trigger(trigger)

    def register_trigger(self, trigger: AnalysisTrigger) -> None:
        """Register a trigger.

        Args:
            trigger: The trigger to register.
        """
        self.triggers[trigger.id] = trigger

    def unregister_trigger(self, trigger_id: str) -> bool:
        """Unregister a trigger.

        Args:
            trigger_id: ID of trigger to remove.

        Returns:
            True if trigger was removed.
        """
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            return True
        return False

    def enable_trigger(self, trigger_id: str) -> None:
        """Enable a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = True

    def disable_trigger(self, trigger_id: str) -> None:
        """Disable a trigger."""
        if trigger_id in self.triggers:
            self.triggers[trigger_id].enabled = False

    def record_event(self, metric: str, value: Any = 1) -> None:
        """Record an event for count-based triggers.

        Args:
            metric: The metric name.
            value: The value (default 1 for counting).
        """
        import time
        now = time.time() * 1000  # ms

        if metric not in self._event_counters:
            self._event_counters[metric] = []

        self._event_counters[metric].append(now)

        # Cleanup old events (older than 10 minutes)
        cutoff = now - 600000
        self._event_counters[metric] = [
            t for t in self._event_counters[metric] if t > cutoff
        ]

    def _check_cooldown(self, trigger: AnalysisTrigger) -> bool:
        """Check if trigger is in cooldown period.

        Returns:
            True if trigger can fire (not in cooldown).
        """
        if trigger.cooldown_ms <= 0:
            return True

        if trigger.last_triggered is None:
            return True

        try:
            last = datetime.fromisoformat(trigger.last_triggered)
            now = datetime.now(timezone.utc)
            elapsed_ms = (now - last).total_seconds() * 1000
            return elapsed_ms >= trigger.cooldown_ms
        except Exception:
            return True

    def _check_max_triggers(self, trigger: AnalysisTrigger) -> bool:
        """Check if trigger has exceeded max per session.

        Returns:
            True if trigger can fire (under limit).
        """
        if trigger.max_triggers_per_session is None:
            return True
        return trigger.trigger_count < trigger.max_triggers_per_session

    def _evaluate_condition(
        self,
        condition: TriggerCondition,
        metrics: dict[str, Any],
    ) -> bool:
        """Evaluate a trigger condition.

        Args:
            condition: The condition to evaluate.
            metrics: Current metric values.

        Returns:
            True if condition is met.
        """
        import re
        import time

        if condition.type == TriggerConditionType.THRESHOLD:
            if condition.metric and condition.threshold is not None:
                value = metrics.get(condition.metric, 0)
                try:
                    return float(value) >= condition.threshold
                except (ValueError, TypeError):
                    return False

        elif condition.type == TriggerConditionType.PATTERN:
            if condition.metric and condition.pattern:
                value = str(metrics.get(condition.metric, ""))
                try:
                    return bool(re.search(condition.pattern, value))
                except re.error:
                    return False

        elif condition.type == TriggerConditionType.COUNT:
            if condition.metric and condition.count is not None:
                events = self._event_counters.get(condition.metric, [])

                # Filter by time window if specified
                if condition.within_ms:
                    cutoff = time.time() * 1000 - condition.within_ms
                    events = [t for t in events if t > cutoff]

                return len(events) >= condition.count

        elif condition.type == TriggerConditionType.TIME:
            # Time-based triggers would be handled by a scheduler
            pass

        return False

    def evaluate(
        self,
        metrics: dict[str, Any],
    ) -> list[tuple[AnalysisTrigger, bool]]:
        """Evaluate all triggers against current metrics.

        Args:
            metrics: Current metric values.

        Returns:
            List of (trigger, accepted) tuples for triggers that fired.
        """
        results: list[tuple[AnalysisTrigger, bool]] = []

        for trigger in self.triggers.values():
            if not trigger.enabled:
                continue

            if not self._check_cooldown(trigger):
                continue

            if not self._check_max_triggers(trigger):
                continue

            if not self._evaluate_condition(trigger.condition, metrics):
                continue

            # Trigger condition met - check gate mode
            accepted = self._handle_gate(trigger)

            # Update trigger state
            trigger.trigger_count += 1
            trigger.last_triggered = datetime.now(timezone.utc).isoformat()

            # Record event
            event = TriggerEvent(
                trigger_id=trigger.id,
                timestamp=trigger.last_triggered,
                metrics=metrics,
                action_taken=trigger.action.mode,
                accepted=accepted,
            )
            self.event_history.append(event)

            results.append((trigger, accepted))

        return results

    def _handle_gate(self, trigger: AnalysisTrigger) -> bool:
        """Handle gating logic for a fired trigger.

        Args:
            trigger: The trigger that fired.

        Returns:
            True if the action should be taken.
        """
        if self.gate_mode == AnalysisGateMode.AUTO_DENY:
            return False

        if self.gate_mode == AnalysisGateMode.AUTO_ACCEPT:
            return True

        if trigger.action.auto_accept:
            return True

        if self.gate_mode == AnalysisGateMode.CONFIRM_ALL:
            if self.confirmation_callback:
                return self.confirmation_callback(trigger)
            # No callback - default to deny for safety
            return False

        if self.gate_mode == AnalysisGateMode.SELECTIVE:
            # Check per-trigger auto_accept setting
            return trigger.action.auto_accept

        return False

    def get_recent_events(self, limit: int = 20) -> list[TriggerEvent]:
        """Get recent trigger events.

        Args:
            limit: Maximum events to return.

        Returns:
            List of recent trigger events.
        """
        return self.event_history[-limit:]

    def reset_session(self) -> None:
        """Reset per-session state (counters, histories)."""
        for trigger in self.triggers.values():
            trigger.trigger_count = 0
        self._event_counters.clear()
        self.event_history.clear()
