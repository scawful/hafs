"""Quality Feedback Tracker for training data pipeline.

Tracks:
- Quality metrics over time
- Rejection reasons and patterns
- Threshold adjustments
- Generator performance
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agents.training.base import TrainingSample, QualityScore

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """Categorized reasons for sample rejection."""

    LOW_DIVERSITY = "low_diversity"
    KG_INCONSISTENT = "kg_inconsistent"
    HIGH_HALLUCINATION = "high_hallucination"
    LOW_COHERENCE = "low_coherence"
    DUPLICATE = "duplicate"
    SYNTAX_ERROR = "syntax_error"
    INVALID_INSTRUCTION = "invalid_instruction"
    EMPTY_OUTPUT = "empty_output"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    VALIDATION_FAILED = "validation_failed"
    OTHER = "other"


@dataclass
class QualityTrend:
    """Trend analysis for quality metrics over time."""

    domain: str
    metric: str
    values: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    @property
    def mean(self) -> float:
        """Average value."""
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def recent_mean(self, window: int = 10) -> float:
        """Average of recent values."""
        recent = self.values[-window:]
        return sum(recent) / len(recent) if recent else 0.0

    @property
    def trend_direction(self) -> str:
        """Whether quality is improving, declining, or stable."""
        if len(self.values) < 5:
            return "insufficient_data"

        recent = self.values[-5:]
        older = self.values[-10:-5] if len(self.values) >= 10 else self.values[:5]

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        diff = recent_avg - older_avg
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"

    def add(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a new data point."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.now())

        # Keep last 1000 data points
        if len(self.values) > 1000:
            self.values = self.values[-1000:]
            self.timestamps = self.timestamps[-1000:]


@dataclass
class GeneratorStats:
    """Statistics for a specific generator."""

    generator_name: str
    samples_generated: int = 0
    samples_accepted: int = 0
    samples_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    total_quality_sum: float = 0.0
    last_run: Optional[datetime] = None

    @property
    def acceptance_rate(self) -> float:
        """Percentage of samples accepted."""
        total = self.samples_accepted + self.samples_rejected
        return self.samples_accepted / total if total > 0 else 0.0

    def record_acceptance(self, quality_score: float) -> None:
        """Record an accepted sample."""
        self.samples_generated += 1
        self.samples_accepted += 1
        self.total_quality_sum += quality_score
        self.avg_quality_score = self.total_quality_sum / self.samples_generated
        self.last_run = datetime.now()

    def record_rejection(self, reason: RejectionReason) -> None:
        """Record a rejected sample."""
        self.samples_generated += 1
        self.samples_rejected += 1
        reason_key = reason.value
        self.rejection_reasons[reason_key] = self.rejection_reasons.get(reason_key, 0) + 1
        self.last_run = datetime.now()


class QualityFeedbackTracker:
    """Track quality metrics and provide feedback for threshold adjustment."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_adjust_thresholds: bool = True,
    ):
        """Initialize quality feedback tracker.

        Args:
            storage_path: Path to store feedback data
            auto_adjust_thresholds: Whether to automatically adjust quality thresholds
        """
        self.storage_path = storage_path or Path.home() / ".context" / "training" / "quality_feedback.json"
        self.auto_adjust = auto_adjust_thresholds

        # Quality trends by domain and metric
        self.trends: dict[str, dict[str, QualityTrend]] = defaultdict(dict)

        # Generator statistics
        self.generator_stats: dict[str, GeneratorStats] = {}

        # Current thresholds (can be adjusted)
        self.thresholds = {
            "min_quality_score": 0.7,
            "min_diversity": 0.3,
            "max_hallucination_risk": 0.5,
            "min_kg_consistency": 0.5,
            "min_coherence": 0.4,
            "similarity_threshold": 0.95,
        }

        # Rejection history for pattern analysis
        self.rejection_history: list[dict[str, Any]] = []

        # Load existing data
        self._load()

    def _load(self) -> None:
        """Load existing feedback data from disk."""
        if not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())

            # Load thresholds
            if "thresholds" in data:
                self.thresholds.update(data["thresholds"])

            # Load generator stats
            for name, stats_data in data.get("generator_stats", {}).items():
                self.generator_stats[name] = GeneratorStats(
                    generator_name=name,
                    samples_generated=stats_data.get("samples_generated", 0),
                    samples_accepted=stats_data.get("samples_accepted", 0),
                    samples_rejected=stats_data.get("samples_rejected", 0),
                    rejection_reasons=stats_data.get("rejection_reasons", {}),
                    avg_quality_score=stats_data.get("avg_quality_score", 0.0),
                    total_quality_sum=stats_data.get("total_quality_sum", 0.0),
                )

            # Load recent rejection history
            self.rejection_history = data.get("rejection_history", [])[-500:]

            logger.info(f"Loaded quality feedback data from {self.storage_path}")

        except Exception as e:
            logger.warning(f"Failed to load quality feedback: {e}")

    def save(self) -> None:
        """Save feedback data to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "thresholds": self.thresholds,
            "generator_stats": {
                name: {
                    "samples_generated": stats.samples_generated,
                    "samples_accepted": stats.samples_accepted,
                    "samples_rejected": stats.samples_rejected,
                    "rejection_reasons": stats.rejection_reasons,
                    "avg_quality_score": stats.avg_quality_score,
                    "total_quality_sum": stats.total_quality_sum,
                }
                for name, stats in self.generator_stats.items()
            },
            "rejection_history": self.rejection_history[-500:],
            "last_updated": datetime.now().isoformat(),
        }

        self.storage_path.write_text(json.dumps(data, indent=2))

    def record_sample(
        self,
        sample: TrainingSample,
        quality_score: QualityScore,
        accepted: bool,
        generator_name: str,
        rejection_reason: Optional[RejectionReason] = None,
    ) -> None:
        """Record feedback for a processed sample.

        Args:
            sample: The training sample
            quality_score: Quality scores for the sample
            accepted: Whether the sample was accepted
            generator_name: Name of the generator that created this sample
            rejection_reason: Reason for rejection if not accepted
        """
        # Update generator stats
        if generator_name not in self.generator_stats:
            self.generator_stats[generator_name] = GeneratorStats(generator_name=generator_name)

        stats = self.generator_stats[generator_name]

        if accepted:
            stats.record_acceptance(quality_score.overall)
        else:
            stats.record_rejection(rejection_reason or RejectionReason.OTHER)

            # Add to rejection history
            self.rejection_history.append({
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "generator": generator_name,
                "reason": rejection_reason.value if rejection_reason else "other",
                "scores": {
                    "diversity": quality_score.diversity_score,
                    "kg_consistency": quality_score.kg_consistency,
                    "hallucination_risk": quality_score.hallucination_risk,
                    "coherence": quality_score.semantic_coherence,
                    "overall": quality_score.overall,
                },
                "timestamp": datetime.now().isoformat(),
            })

        # Update trends
        domain = sample.domain

        if domain not in self.trends:
            self.trends[domain] = {}

        metrics = [
            ("diversity", quality_score.diversity_score),
            ("kg_consistency", quality_score.kg_consistency),
            ("hallucination_risk", quality_score.hallucination_risk),
            ("coherence", quality_score.semantic_coherence),
            ("overall", quality_score.overall),
        ]

        for metric_name, value in metrics:
            if metric_name not in self.trends[domain]:
                self.trends[domain][metric_name] = QualityTrend(domain=domain, metric=metric_name)
            self.trends[domain][metric_name].add(value)

        # Auto-adjust thresholds if enabled
        if self.auto_adjust:
            self._maybe_adjust_thresholds()

    def _maybe_adjust_thresholds(self) -> None:
        """Adjust thresholds based on observed patterns."""
        # Only adjust every 100 samples
        total_samples = sum(s.samples_generated for s in self.generator_stats.values())
        if total_samples % 100 != 0:
            return

        # Check overall acceptance rate
        total_accepted = sum(s.samples_accepted for s in self.generator_stats.values())
        acceptance_rate = total_accepted / total_samples if total_samples > 0 else 0.5

        # If acceptance rate is too low, loosen thresholds slightly
        if acceptance_rate < 0.3:
            self.thresholds["min_quality_score"] = max(0.5, self.thresholds["min_quality_score"] - 0.05)
            logger.info(f"Lowered min_quality_score to {self.thresholds['min_quality_score']}")

        # If acceptance rate is very high, tighten thresholds
        elif acceptance_rate > 0.9:
            self.thresholds["min_quality_score"] = min(0.85, self.thresholds["min_quality_score"] + 0.02)
            logger.info(f"Raised min_quality_score to {self.thresholds['min_quality_score']}")

        # Analyze rejection patterns
        reason_counts = defaultdict(int)
        for rejection in self.rejection_history[-100:]:
            reason_counts[rejection.get("reason", "other")] += 1

        # If duplicates are the main issue, tighten similarity threshold
        if reason_counts.get("duplicate", 0) > 30:
            self.thresholds["similarity_threshold"] = max(0.90, self.thresholds["similarity_threshold"] - 0.02)
            logger.info(f"Tightened similarity_threshold to {self.thresholds['similarity_threshold']}")

    def get_generator_report(self) -> dict[str, Any]:
        """Get a report on generator performance."""
        return {
            name: {
                "samples_generated": stats.samples_generated,
                "acceptance_rate": f"{stats.acceptance_rate:.1%}",
                "avg_quality": f"{stats.avg_quality_score:.2f}",
                "top_rejections": sorted(
                    stats.rejection_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3],
            }
            for name, stats in self.generator_stats.items()
        }

    def get_trend_report(self, domain: Optional[str] = None) -> dict[str, Any]:
        """Get quality trend report.

        Args:
            domain: Optional domain to filter by

        Returns:
            Report with trend information
        """
        report = {}

        domains = [domain] if domain else list(self.trends.keys())

        for d in domains:
            if d not in self.trends:
                continue

            report[d] = {}
            for metric_name, trend in self.trends[d].items():
                report[d][metric_name] = {
                    "mean": f"{trend.mean:.3f}",
                    "recent_mean": f"{trend.recent_mean:.3f}",
                    "direction": trend.trend_direction,
                    "data_points": len(trend.values),
                }

        return report

    def get_rejection_patterns(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get common rejection patterns.

        Args:
            limit: Maximum number of patterns to return

        Returns:
            List of rejection patterns with counts
        """
        # Count by reason and domain
        patterns = defaultdict(lambda: {"count": 0, "avg_scores": defaultdict(list)})

        for rejection in self.rejection_history:
            key = (rejection.get("reason", "other"), rejection.get("domain", "unknown"))
            patterns[key]["count"] += 1

            scores = rejection.get("scores", {})
            for score_name, score_val in scores.items():
                patterns[key]["avg_scores"][score_name].append(score_val)

        # Format results
        results = []
        for (reason, domain), data in sorted(
            patterns.items(), key=lambda x: x[1]["count"], reverse=True
        )[:limit]:
            avg_scores = {
                name: sum(vals) / len(vals)
                for name, vals in data["avg_scores"].items()
                if vals
            }
            results.append({
                "reason": reason,
                "domain": domain,
                "count": data["count"],
                "avg_scores": avg_scores,
            })

        return results

    def suggest_improvements(self, generator_name: str) -> list[str]:
        """Suggest improvements for a generator based on feedback.

        Args:
            generator_name: Name of the generator

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        if generator_name not in self.generator_stats:
            return ["No data available for this generator"]

        stats = self.generator_stats[generator_name]

        # Check acceptance rate
        if stats.acceptance_rate < 0.5:
            suggestions.append(
                f"Low acceptance rate ({stats.acceptance_rate:.1%}). "
                "Consider improving sample quality or adjusting generation parameters."
            )

        # Analyze rejection reasons
        if stats.rejection_reasons:
            top_reason = max(stats.rejection_reasons.items(), key=lambda x: x[1])
            reason_name, count = top_reason
            total_rejections = sum(stats.rejection_reasons.values())
            percentage = count / total_rejections if total_rejections > 0 else 0

            if percentage > 0.3:
                if reason_name == "low_diversity":
                    suggestions.append(
                        "High duplicate rate. Consider diversifying prompts or sampling strategies."
                    )
                elif reason_name == "kg_inconsistent":
                    suggestions.append(
                        "Many samples fail KG validation. Ensure generated content references known entities."
                    )
                elif reason_name == "high_hallucination":
                    suggestions.append(
                        "High hallucination risk detected. Consider using more grounded prompts."
                    )
                elif reason_name == "syntax_error":
                    suggestions.append(
                        "Frequent syntax errors. Review template formatting and code generation logic."
                    )

        # Check quality trend
        for domain, metrics in self.trends.items():
            if "overall" in metrics:
                trend = metrics["overall"]
                if trend.trend_direction == "declining":
                    suggestions.append(
                        f"Quality declining for {domain} domain. Review recent generation changes."
                    )

        if not suggestions:
            suggestions.append("Generator performing well. No immediate improvements suggested.")

        return suggestions
