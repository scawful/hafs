"""Data loading layer for training visualization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class QualityTrendData:
    """Quality trend for a single domain/metric."""

    domain: str
    metric: str
    values: list[float] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def trend_direction(self) -> str:
        if len(self.values) < 5:
            return "insufficient"
        recent = sum(self.values[-5:]) / 5
        older = sum(self.values[:5]) / 5
        diff = recent - older
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        return "stable"


@dataclass
class GeneratorStatsData:
    """Statistics for a single generator."""

    name: str
    samples_generated: int = 0
    samples_accepted: int = 0
    samples_rejected: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    avg_quality: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        total = self.samples_accepted + self.samples_rejected
        return self.samples_accepted / total if total > 0 else 0.0


@dataclass
class EmbeddingRegionData:
    """Embedding space region data."""

    index: int
    sample_count: int
    domain: str
    avg_quality: float


@dataclass
class TrainingRunData:
    """Training run metadata."""

    run_id: str
    model_name: str
    final_loss: float
    samples_count: int
    domain_distribution: dict[str, int] = field(default_factory=dict)
    eval_metrics: dict[str, float] = field(default_factory=dict)


class TrainingDataLoader:
    """Loads and caches training data from JSON files."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path.home() / ".context" / "training"

        # Cached data
        self._quality_trends: list[QualityTrendData] = []
        self._generator_stats: list[GeneratorStatsData] = []
        self._embedding_regions: list[EmbeddingRegionData] = []
        self._training_runs: list[TrainingRunData] = []
        self._coverage_score: float = 0.0
        self._rejection_reasons: dict[str, int] = {}

        self._last_load: Optional[datetime] = None

    def refresh(self) -> bool:
        """Reload all data from disk."""
        try:
            self._load_quality_feedback()
            self._load_active_learning()
            self._load_training_feedback()
            self._last_load = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return False

    def _load_quality_feedback(self) -> None:
        """Load quality_feedback.json."""
        path = self.data_path / "quality_feedback.json"
        if not path.exists():
            return

        data = json.loads(path.read_text())

        # Parse generator stats
        self._generator_stats = []
        self._rejection_reasons = {}
        for name, stats in data.get("generator_stats", {}).items():
            gen_stats = GeneratorStatsData(
                name=name,
                samples_generated=stats.get("samples_generated", 0),
                samples_accepted=stats.get("samples_accepted", 0),
                samples_rejected=stats.get("samples_rejected", 0),
                rejection_reasons=stats.get("rejection_reasons", {}),
                avg_quality=stats.get("avg_quality_score", 0.0),
            )
            self._generator_stats.append(gen_stats)

            # Aggregate rejection reasons
            for reason, count in gen_stats.rejection_reasons.items():
                self._rejection_reasons[reason] = (
                    self._rejection_reasons.get(reason, 0) + count
                )

        # Parse rejection history for trends
        history = data.get("rejection_history", [])
        trends_by_key: dict[tuple[str, str], QualityTrendData] = {}

        for entry in history:
            domain = entry.get("domain", "unknown")
            scores = entry.get("scores", {})

            for metric, value in scores.items():
                key = (domain, metric)
                if key not in trends_by_key:
                    trends_by_key[key] = QualityTrendData(domain=domain, metric=metric)
                trends_by_key[key].values.append(value)

        self._quality_trends = list(trends_by_key.values())

    def _load_active_learning(self) -> None:
        """Load active_learning.json."""
        path = self.data_path / "active_learning.json"
        if not path.exists():
            return

        data = json.loads(path.read_text())

        self._embedding_regions = []
        for i, region in enumerate(data.get("regions", [])):
            self._embedding_regions.append(
                EmbeddingRegionData(
                    index=i,
                    sample_count=region.get("sample_count", 0),
                    domain=region.get("domain", "unknown"),
                    avg_quality=region.get("avg_quality", 0.0),
                )
            )

        # Calculate coverage score
        if self._embedding_regions:
            counts = [r.sample_count for r in self._embedding_regions]
            avg = sum(counts) / len(counts)
            if avg > 0:
                std = (sum((c - avg) ** 2 for c in counts) / len(counts)) ** 0.5
                cv = std / avg
                self._coverage_score = max(0.0, min(1.0, 1.0 - cv))

    def _load_training_feedback(self) -> None:
        """Load training_feedback.json."""
        path = self.data_path / "training_feedback.json"
        if not path.exists():
            return

        data = json.loads(path.read_text())

        self._training_runs = []
        for run_id, run_data in data.get("training_runs", {}).items():
            if run_data.get("final_loss") is not None:
                self._training_runs.append(
                    TrainingRunData(
                        run_id=run_id,
                        model_name=run_data.get("model_name", ""),
                        final_loss=run_data.get("final_loss", 0.0),
                        samples_count=run_data.get("samples_count", 0),
                        domain_distribution=run_data.get("domain_distribution", {}),
                        eval_metrics=run_data.get("eval_metrics", {}),
                    )
                )

    # Properties for accessing data
    @property
    def quality_trends(self) -> list[QualityTrendData]:
        return self._quality_trends

    @property
    def generator_stats(self) -> list[GeneratorStatsData]:
        return self._generator_stats

    @property
    def embedding_regions(self) -> list[EmbeddingRegionData]:
        return self._embedding_regions

    @property
    def training_runs(self) -> list[TrainingRunData]:
        return self._training_runs

    @property
    def coverage_score(self) -> float:
        return self._coverage_score

    @property
    def rejection_reasons(self) -> dict[str, int]:
        return self._rejection_reasons

    @property
    def has_data(self) -> bool:
        return bool(
            self._generator_stats or self._embedding_regions or self._training_runs
        )
