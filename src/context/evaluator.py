"""Context Evaluator for the Context Engineering Pipeline.

Implements the Evaluator phase from AFS research:
- Validate: Check context quality and consistency
- Score: Update relevance scores based on usage
- Feedback: Learn from task outcomes to improve selection

Based on "Everything is Context: Agentic File System Abstraction"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional
from uuid import UUID

from models.context import (
    ContextItem,
    ContextPriority,
    ContextWindow,
    MemoryType,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of evaluating a context window."""

    # Overall quality score (0-1)
    quality_score: float = 0.0

    # Coverage score (0-1) - how well context covers the task
    coverage_score: float = 0.0

    # Coherence score (0-1) - how well items relate to each other
    coherence_score: float = 0.0

    # Freshness score (0-1) - average recency of items
    freshness_score: float = 0.0

    # Efficiency score (0-1) - token utilization
    efficiency_score: float = 0.0

    # Issues found during validation
    issues: list[str] = field(default_factory=list)

    # Suggestions for improvement
    suggestions: list[str] = field(default_factory=list)

    # Items that should be updated
    items_to_refresh: list[UUID] = field(default_factory=list)

    # Items that should be promoted (higher priority)
    items_to_promote: list[UUID] = field(default_factory=list)

    # Items that should be demoted (lower priority)
    items_to_demote: list[UUID] = field(default_factory=list)


@dataclass
class TaskOutcome:
    """Outcome of a task for feedback learning."""

    task_id: str
    success: bool
    context_item_ids: list[UUID]

    # Optional scoring
    user_rating: Optional[float] = None  # 0-1
    completion_time_seconds: Optional[float] = None
    error_count: int = 0

    # What worked/didn't work
    helpful_items: list[UUID] = field(default_factory=list)
    unhelpful_items: list[UUID] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluatorConfig:
    """Configuration for context evaluator."""

    # Weights for quality score components
    coverage_weight: float = 0.3
    coherence_weight: float = 0.2
    freshness_weight: float = 0.25
    efficiency_weight: float = 0.25

    # Thresholds
    min_freshness_hours: float = 24  # Items older than this are stale
    min_coverage_score: float = 0.3  # Warn if below this
    min_quality_score: float = 0.4  # Warn if below this

    # Relevance score adjustments
    success_boost: float = 0.1   # Boost on successful task
    failure_penalty: float = 0.05  # Penalty on failed task
    helpful_boost: float = 0.15   # Boost for explicitly helpful items
    unhelpful_penalty: float = 0.1  # Penalty for explicitly unhelpful items

    # Priority adjustment thresholds
    promote_threshold: float = 0.8   # Promote if relevance > this
    demote_threshold: float = 0.2    # Demote if relevance < this

    # Learning rate for relevance updates
    learning_rate: float = 0.1


class ContextEvaluator:
    """Evaluates context quality and learns from feedback.

    The Evaluator phase of the Context Engineering Pipeline:
    1. Validate: Check context for issues
    2. Score: Compute quality metrics
    3. Feedback: Update relevance based on outcomes

    Example:
        evaluator = ContextEvaluator()

        # Evaluate before using
        result = evaluator.evaluate(window, task="implement auth")

        # After task completion
        evaluator.record_outcome(TaskOutcome(
            task_id="task-1",
            success=True,
            context_item_ids=[...],
            helpful_items=[...],
        ))

        # Update relevance scores
        evaluator.apply_feedback()
    """

    def __init__(
        self,
        store: Optional["ContextStore"] = None,  # type: ignore[name-defined]
        config: Optional[EvaluatorConfig] = None,
        semantic_scorer: Optional[Callable[[str, str], float]] = None,
    ):
        """Initialize the evaluator.

        Args:
            store: Context store for persisting updates
            config: Evaluator configuration
            semantic_scorer: Optional function(text1, text2) -> similarity(0-1)
                            For embedding-based relevance scoring
        """
        self.store = store
        self.config = config or EvaluatorConfig()
        self._semantic_scorer = semantic_scorer

        # Pending outcomes for batch processing
        self._pending_outcomes: list[TaskOutcome] = []

        # Relevance score history for smoothing
        self._score_history: dict[UUID, list[float]] = {}

    def evaluate(
        self,
        window: ContextWindow,
        task: Optional[str] = None,
        required_types: Optional[set[MemoryType]] = None,
    ) -> EvaluationResult:
        """Evaluate a context window.

        Args:
            window: The context window to evaluate
            task: Current task description (for coverage scoring)
            required_types: Memory types that should be present

        Returns:
            Evaluation result with scores and suggestions
        """
        result = EvaluationResult()

        # Validate
        self._validate(window, result, required_types)

        # Score components
        result.coverage_score = self._score_coverage(window, task)
        result.coherence_score = self._score_coherence(window)
        result.freshness_score = self._score_freshness(window)
        result.efficiency_score = self._score_efficiency(window)

        # Compute overall quality
        result.quality_score = (
            self.config.coverage_weight * result.coverage_score +
            self.config.coherence_weight * result.coherence_score +
            self.config.freshness_weight * result.freshness_score +
            self.config.efficiency_weight * result.efficiency_score
        )

        # Generate suggestions
        self._generate_suggestions(window, result)

        # Identify items for priority adjustment
        self._identify_adjustments(window, result)

        logger.info(
            f"Context evaluation: quality={result.quality_score:.2f}, "
            f"coverage={result.coverage_score:.2f}, "
            f"freshness={result.freshness_score:.2f}"
        )

        return result

    def _validate(
        self,
        window: ContextWindow,
        result: EvaluationResult,
        required_types: Optional[set[MemoryType]],
    ) -> None:
        """Validate context window for issues."""
        # Check for empty window
        if not window.items:
            result.issues.append("Context window is empty")

        # Check for missing required types
        if required_types:
            present_types = {item.memory_type for item in window.items}
            missing = required_types - present_types
            if missing:
                result.issues.append(
                    f"Missing required memory types: {[t.value for t in missing]}"
                )

        # Check for expired items
        expired = [item for item in window.items if item.is_expired()]
        if expired:
            result.issues.append(f"{len(expired)} expired items in context")
            result.items_to_refresh.extend([item.id for item in expired])

        # Check for very stale items
        stale = [
            item for item in window.items
            if item.staleness > self.config.min_freshness_hours
        ]
        if len(stale) > len(window.items) * 0.5:
            result.issues.append("More than 50% of items are stale")
            result.items_to_refresh.extend([item.id for item in stale])

        # Check for duplicate content (exact matches)
        seen_content = set()
        duplicates = []
        for item in window.items:
            content_hash = hash(item.content[:500])
            if content_hash in seen_content:
                duplicates.append(item.id)
            seen_content.add(content_hash)

        if duplicates:
            result.issues.append(f"{len(duplicates)} duplicate items found")

        # Check token budget
        if window.used_percentage > 95:
            result.issues.append("Token budget nearly exhausted (>95%)")
        elif window.used_percentage < 20:
            result.issues.append("Token budget under-utilized (<20%)")

    def _score_coverage(
        self,
        window: ContextWindow,
        task: Optional[str],
    ) -> float:
        """Score how well context covers the task."""
        if not task or not window.items:
            return 0.5  # Neutral score without task

        # Simple keyword-based coverage
        task_words = set(task.lower().split())
        task_words -= {"a", "an", "the", "is", "are", "to", "for", "in", "on"}

        if not task_words:
            return 0.5

        # Check coverage in context
        context_text = " ".join(
            item.content.lower() for item in window.items
        )

        covered_words = sum(
            1 for word in task_words
            if word in context_text
        )

        coverage = covered_words / len(task_words)

        # Use semantic scorer if available
        if self._semantic_scorer:
            semantic_score = self._semantic_scorer(
                task,
                window.to_prompt()[:4000],  # Limit for efficiency
            )
            coverage = 0.4 * coverage + 0.6 * semantic_score

        return min(1.0, coverage)

    def _score_coherence(self, window: ContextWindow) -> float:
        """Score how well items relate to each other."""
        if len(window.items) < 2:
            return 1.0  # Single item is coherent by definition

        # Check type distribution (mixed types = potentially less coherent)
        type_counts: dict[MemoryType, int] = {}
        for item in window.items:
            type_counts[item.memory_type] = (
                type_counts.get(item.memory_type, 0) + 1
            )

        # Entropy-based coherence
        # More uniform distribution = higher entropy = lower coherence
        total = len(window.items)
        import math
        entropy = 0
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize entropy (max entropy = log2(num_types))
        max_entropy = math.log2(len(MemoryType))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Invert: low entropy = high coherence
        coherence = 1.0 - normalized_entropy

        # Adjust based on priority distribution
        high_priority = sum(
            1 for item in window.items
            if item.priority in (ContextPriority.CRITICAL, ContextPriority.HIGH)
        )
        priority_ratio = high_priority / total

        # Good coherence if priorities are consistent
        coherence = 0.7 * coherence + 0.3 * min(1.0, priority_ratio * 2)

        return coherence

    def _score_freshness(self, window: ContextWindow) -> float:
        """Score recency of items."""
        if not window.items:
            return 0.0

        # Average freshness (inverse of staleness)
        freshness_scores = []
        for item in window.items:
            # Decay function: 0 hours -> 1.0, 24 hours -> 0.5
            hours = item.staleness
            freshness = 1.0 / (1.0 + hours / 24)
            freshness_scores.append(freshness)

        return sum(freshness_scores) / len(freshness_scores)

    def _score_efficiency(self, window: ContextWindow) -> float:
        """Score token budget utilization."""
        # Optimal is around 70-85% usage
        usage = window.used_percentage

        if usage < 50:
            # Under-utilized
            return usage / 50  # 0% -> 0, 50% -> 1
        elif usage <= 85:
            # Optimal range
            return 1.0
        else:
            # Over-utilized
            return max(0.5, 1.0 - (usage - 85) / 15)

    def _generate_suggestions(
        self,
        window: ContextWindow,
        result: EvaluationResult,
    ) -> None:
        """Generate improvement suggestions."""
        if result.coverage_score < self.config.min_coverage_score:
            result.suggestions.append(
                "Add more context items related to the current task"
            )

        if result.freshness_score < 0.4:
            result.suggestions.append(
                "Refresh stale items or add more recent context"
            )

        if result.efficiency_score < 0.5:
            if window.used_percentage < 50:
                result.suggestions.append(
                    "Context is under-utilized; add more relevant items"
                )
            else:
                result.suggestions.append(
                    "Context is over budget; compress or remove low-priority items"
                )

        if result.coherence_score < 0.4:
            result.suggestions.append(
                "Context has mixed memory types; consider focusing on relevant types"
            )

        # Suggest promotions for high-performing items
        high_relevance = [
            item for item in window.items
            if item.relevance_score > self.config.promote_threshold
            and item.priority not in (ContextPriority.CRITICAL, ContextPriority.HIGH)
        ]
        if high_relevance:
            result.suggestions.append(
                f"Consider promoting {len(high_relevance)} high-relevance items"
            )
            result.items_to_promote.extend([item.id for item in high_relevance])

    def _identify_adjustments(
        self,
        window: ContextWindow,
        result: EvaluationResult,
    ) -> None:
        """Identify items that need priority adjustment."""
        for item in window.items:
            # Check for promotion
            if (
                item.relevance_score > self.config.promote_threshold
                and item.priority not in (
                    ContextPriority.CRITICAL,
                    ContextPriority.HIGH,
                )
            ):
                if item.id not in result.items_to_promote:
                    result.items_to_promote.append(item.id)

            # Check for demotion
            if (
                item.relevance_score < self.config.demote_threshold
                and item.priority not in (
                    ContextPriority.LOW,
                    ContextPriority.BACKGROUND,
                )
            ):
                result.items_to_demote.append(item.id)

    def record_outcome(self, outcome: TaskOutcome) -> None:
        """Record a task outcome for feedback learning.

        Args:
            outcome: The task outcome with success/failure and item info
        """
        self._pending_outcomes.append(outcome)
        logger.debug(f"Recorded outcome for task {outcome.task_id}")

    def apply_feedback(self) -> int:
        """Apply feedback from recorded outcomes.

        Updates relevance scores based on task outcomes.

        Returns:
            Number of items updated
        """
        if not self._pending_outcomes:
            return 0

        if not self.store:
            logger.warning("No store configured, cannot apply feedback")
            return 0

        updated_count = 0
        outcomes = self._pending_outcomes
        self._pending_outcomes = []

        for outcome in outcomes:
            for item_id in outcome.context_item_ids:
                item = self.store.get(str(item_id))
                if not item:
                    continue

                # Base adjustment from success/failure
                if outcome.success:
                    adjustment = self.config.success_boost
                else:
                    adjustment = -self.config.failure_penalty

                # Extra adjustment for explicitly helpful/unhelpful
                if item_id in outcome.helpful_items:
                    adjustment += self.config.helpful_boost
                elif item_id in outcome.unhelpful_items:
                    adjustment -= self.config.unhelpful_penalty

                # Apply with learning rate
                old_score = item.relevance_score
                new_score = old_score + self.config.learning_rate * adjustment

                # Clamp to [0, 1]
                new_score = max(0.0, min(1.0, new_score))

                # Apply exponential smoothing with history
                if item_id in self._score_history:
                    history = self._score_history[item_id]
                    history.append(new_score)
                    # Keep last 10 scores
                    history = history[-10:]
                    self._score_history[item_id] = history
                    # Exponential moving average
                    smoothed = sum(
                        0.9 ** i * s
                        for i, s in enumerate(reversed(history))
                    )
                    smoothed /= sum(0.9 ** i for i in range(len(history)))
                    new_score = smoothed
                else:
                    self._score_history[item_id] = [new_score]

                item.relevance_score = new_score
                self.store.save(item)
                updated_count += 1

                logger.debug(
                    f"Updated item {item_id} relevance: "
                    f"{old_score:.2f} -> {new_score:.2f}"
                )

        logger.info(f"Applied feedback: updated {updated_count} items")
        return updated_count

    def get_item_stats(self, item_id: UUID) -> dict:
        """Get statistics for an item.

        Args:
            item_id: The item ID

        Returns:
            Statistics dictionary
        """
        history = self._score_history.get(item_id, [])

        return {
            "score_history": history,
            "avg_score": sum(history) / len(history) if history else 0.0,
            "score_trend": (
                history[-1] - history[0] if len(history) > 1 else 0.0
            ),
            "assessment_count": len(history),
        }

    def recommend_items(
        self,
        task: str,
        available_items: list[ContextItem],
        max_items: int = 10,
    ) -> list[ContextItem]:
        """Recommend items for a task based on learned relevance.

        Args:
            task: Task description
            available_items: Pool of available items
            max_items: Maximum items to recommend

        Returns:
            Recommended items sorted by expected relevance
        """
        scored: list[tuple[float, ContextItem]] = []

        for item in available_items:
            # Base score from item relevance
            base_score = item.relevance_score

            # Boost for high priority
            priority_boost = {
                ContextPriority.CRITICAL: 0.3,
                ContextPriority.HIGH: 0.2,
                ContextPriority.MEDIUM: 0.0,
                ContextPriority.LOW: -0.1,
                ContextPriority.BACKGROUND: -0.2,
            }.get(item.priority, 0.0)

            # Recency boost
            recency_boost = max(0, 0.2 * (1.0 - item.staleness / 168))

            # Compute task similarity if semantic scorer available
            task_similarity = 0.0
            if self._semantic_scorer:
                task_similarity = self._semantic_scorer(
                    task, item.content[:1000]
                )

            # Combined score
            score = (
                0.4 * base_score +
                0.1 * (priority_boost + 0.3) +  # Normalize to 0-0.6 range
                0.1 * recency_boost +
                0.4 * task_similarity
            )

            scored.append((score, item))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top items
        return [item for _, item in scored[:max_items]]
