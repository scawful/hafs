"""Training Feedback Analysis for post-training correlation.

Correlates training samples with model performance to identify:
- Which sample types improve model capabilities
- Samples that correlate with model failures
- Optimal dataset composition
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingRun:
    """Metadata about a training run."""

    run_id: str
    model_name: str
    base_model: str
    dataset_path: str
    samples_count: int
    domain_distribution: dict[str, int]
    start_time: datetime
    end_time: Optional[datetime] = None
    final_loss: Optional[float] = None
    eval_metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SampleCorrelation:
    """Correlation between a sample and training outcome."""

    sample_id: str
    domain: str
    quality_score: float
    training_run_id: str
    contribution_score: float  # How much this sample improved the model
    eval_delta: dict[str, float] = field(default_factory=dict)  # Change in eval metrics


class TrainingFeedback:
    """Analyze post-training feedback to improve sample selection."""

    def __init__(
        self,
        storage_path: Optional[Path] = None,
    ):
        """Initialize training feedback analyzer.

        Args:
            storage_path: Path to store feedback data
        """
        self.storage_path = storage_path or Path.home() / ".context" / "training" / "training_feedback.json"

        # Training runs
        self.training_runs: dict[str, TrainingRun] = {}

        # Sample correlations
        self.correlations: list[SampleCorrelation] = []

        # Aggregated insights
        self.domain_effectiveness: dict[str, float] = {}
        self.quality_threshold_effectiveness: dict[float, float] = {}

        self._load()

    def _load(self) -> None:
        """Load existing feedback data."""
        if not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())

            for run_id, run_data in data.get("training_runs", {}).items():
                self.training_runs[run_id] = TrainingRun(
                    run_id=run_id,
                    model_name=run_data.get("model_name", ""),
                    base_model=run_data.get("base_model", ""),
                    dataset_path=run_data.get("dataset_path", ""),
                    samples_count=run_data.get("samples_count", 0),
                    domain_distribution=run_data.get("domain_distribution", {}),
                    start_time=datetime.fromisoformat(run_data.get("start_time", datetime.now().isoformat())),
                    end_time=datetime.fromisoformat(run_data["end_time"]) if run_data.get("end_time") else None,
                    final_loss=run_data.get("final_loss"),
                    eval_metrics=run_data.get("eval_metrics", {}),
                    notes=run_data.get("notes", ""),
                )

            self.domain_effectiveness = data.get("domain_effectiveness", {})
            self.quality_threshold_effectiveness = {
                float(k): v for k, v in data.get("quality_threshold_effectiveness", {}).items()
            }

            logger.info(f"Loaded training feedback: {len(self.training_runs)} runs")

        except Exception as e:
            logger.warning(f"Failed to load training feedback: {e}")

    def save(self) -> None:
        """Save feedback data to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "training_runs": {
                run_id: {
                    "model_name": run.model_name,
                    "base_model": run.base_model,
                    "dataset_path": run.dataset_path,
                    "samples_count": run.samples_count,
                    "domain_distribution": run.domain_distribution,
                    "start_time": run.start_time.isoformat(),
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "final_loss": run.final_loss,
                    "eval_metrics": run.eval_metrics,
                    "notes": run.notes,
                }
                for run_id, run in self.training_runs.items()
            },
            "domain_effectiveness": self.domain_effectiveness,
            "quality_threshold_effectiveness": self.quality_threshold_effectiveness,
            "last_updated": datetime.now().isoformat(),
        }

        self.storage_path.write_text(json.dumps(data, indent=2))

    def register_training_run(
        self,
        run_id: str,
        model_name: str,
        base_model: str,
        dataset_path: str,
        samples_count: int,
        domain_distribution: dict[str, int],
    ) -> TrainingRun:
        """Register a new training run.

        Args:
            run_id: Unique identifier for this run
            model_name: Name of the model being trained
            base_model: Base model being finetuned
            dataset_path: Path to the training dataset
            samples_count: Number of samples in the dataset
            domain_distribution: Distribution of samples by domain

        Returns:
            Created TrainingRun object
        """
        run = TrainingRun(
            run_id=run_id,
            model_name=model_name,
            base_model=base_model,
            dataset_path=dataset_path,
            samples_count=samples_count,
            domain_distribution=domain_distribution,
            start_time=datetime.now(),
        )

        self.training_runs[run_id] = run
        self.save()

        return run

    def complete_training_run(
        self,
        run_id: str,
        final_loss: float,
        eval_metrics: dict[str, float],
        notes: str = "",
    ) -> Optional[TrainingRun]:
        """Record completion of a training run.

        Args:
            run_id: ID of the training run
            final_loss: Final training loss
            eval_metrics: Evaluation metrics
            notes: Additional notes

        Returns:
            Updated TrainingRun or None if not found
        """
        if run_id not in self.training_runs:
            logger.warning(f"Training run {run_id} not found")
            return None

        run = self.training_runs[run_id]
        run.end_time = datetime.now()
        run.final_loss = final_loss
        run.eval_metrics = eval_metrics
        run.notes = notes

        # Update domain effectiveness based on this run
        self._update_domain_effectiveness(run)

        self.save()
        return run

    def _update_domain_effectiveness(self, run: TrainingRun) -> None:
        """Update domain effectiveness scores based on training results."""
        if not run.eval_metrics:
            return

        # Use a simple heuristic: improvement = lower loss is better
        # Compare with previous runs if available
        prev_runs = [
            r for r in self.training_runs.values()
            if r.run_id != run.run_id
            and r.final_loss is not None
            and r.base_model == run.base_model
        ]

        if not prev_runs:
            # First run for this base model
            return

        avg_prev_loss = sum(r.final_loss for r in prev_runs) / len(prev_runs)

        if run.final_loss is not None:
            improvement = avg_prev_loss - run.final_loss

            # Attribute improvement to domains proportionally
            total_samples = sum(run.domain_distribution.values())
            if total_samples > 0:
                for domain, count in run.domain_distribution.items():
                    proportion = count / total_samples
                    domain_contribution = improvement * proportion

                    # Update running average
                    if domain in self.domain_effectiveness:
                        self.domain_effectiveness[domain] = (
                            self.domain_effectiveness[domain] * 0.7 + domain_contribution * 0.3
                        )
                    else:
                        self.domain_effectiveness[domain] = domain_contribution

    def record_sample_correlation(
        self,
        sample_id: str,
        domain: str,
        quality_score: float,
        training_run_id: str,
        contribution_score: float,
        eval_delta: Optional[dict[str, float]] = None,
    ) -> SampleCorrelation:
        """Record correlation between a sample and training outcome.

        Args:
            sample_id: ID of the training sample
            domain: Domain of the sample
            quality_score: Quality score of the sample
            training_run_id: ID of the training run
            contribution_score: Estimated contribution to model improvement
            eval_delta: Change in evaluation metrics attributable to this sample

        Returns:
            Created SampleCorrelation
        """
        correlation = SampleCorrelation(
            sample_id=sample_id,
            domain=domain,
            quality_score=quality_score,
            training_run_id=training_run_id,
            contribution_score=contribution_score,
            eval_delta=eval_delta or {},
        )

        self.correlations.append(correlation)

        # Update quality threshold effectiveness
        threshold_bucket = round(quality_score, 1)
        if threshold_bucket not in self.quality_threshold_effectiveness:
            self.quality_threshold_effectiveness[threshold_bucket] = 0.0

        # Running average of contribution scores at this quality level
        self.quality_threshold_effectiveness[threshold_bucket] = (
            self.quality_threshold_effectiveness[threshold_bucket] * 0.9 + contribution_score * 0.1
        )

        return correlation

    def get_recommended_domain_weights(self) -> dict[str, float]:
        """Get recommended domain weights based on effectiveness.

        Returns:
            Dictionary of domain -> recommended weight
        """
        if not self.domain_effectiveness:
            return {"asm": 0.4, "cpp": 0.3, "text": 0.3}  # Default weights

        # Normalize effectiveness scores to weights
        total = sum(abs(v) for v in self.domain_effectiveness.values())
        if total == 0:
            return {"asm": 0.4, "cpp": 0.3, "text": 0.3}

        weights = {}
        for domain, effectiveness in self.domain_effectiveness.items():
            # Use sigmoid-like transformation to keep weights reasonable
            normalized = (effectiveness / total + 1) / 2  # Map to 0-1
            weights[domain] = max(0.1, min(0.6, normalized))  # Clamp to 0.1-0.6

        # Normalize to sum to 1
        total_weight = sum(weights.values())
        weights = {d: w / total_weight for d, w in weights.items()}

        return weights

    def get_recommended_quality_threshold(self) -> float:
        """Get recommended quality threshold based on effectiveness.

        Returns:
            Recommended minimum quality score
        """
        if not self.quality_threshold_effectiveness:
            return 0.7  # Default

        # Find threshold with best effectiveness
        best_threshold = 0.7
        best_effectiveness = 0.0

        for threshold, effectiveness in self.quality_threshold_effectiveness.items():
            if effectiveness > best_effectiveness:
                best_effectiveness = effectiveness
                best_threshold = threshold

        return best_threshold

    def get_insights_report(self) -> dict[str, Any]:
        """Generate insights report from training feedback.

        Returns:
            Report with insights and recommendations
        """
        report = {
            "total_training_runs": len(self.training_runs),
            "total_correlations": len(self.correlations),
        }

        # Training run summary
        if self.training_runs:
            completed_runs = [r for r in self.training_runs.values() if r.final_loss is not None]
            if completed_runs:
                report["avg_final_loss"] = sum(r.final_loss for r in completed_runs) / len(completed_runs)
                report["best_loss"] = min(r.final_loss for r in completed_runs)
                report["worst_loss"] = max(r.final_loss for r in completed_runs)

        # Domain effectiveness
        report["domain_effectiveness"] = {
            domain: f"{effectiveness:+.4f}"
            for domain, effectiveness in sorted(
                self.domain_effectiveness.items(),
                key=lambda x: x[1],
                reverse=True
            )
        }

        # Quality threshold effectiveness
        report["quality_threshold_effectiveness"] = {
            f"{threshold:.1f}": f"{effectiveness:.4f}"
            for threshold, effectiveness in sorted(
                self.quality_threshold_effectiveness.items()
            )
        }

        # Recommendations
        report["recommendations"] = {
            "domain_weights": self.get_recommended_domain_weights(),
            "quality_threshold": self.get_recommended_quality_threshold(),
        }

        return report

    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str,
    ) -> dict[str, Any]:
        """Compare two training runs.

        Args:
            run_id_1: First run ID
            run_id_2: Second run ID

        Returns:
            Comparison report
        """
        if run_id_1 not in self.training_runs or run_id_2 not in self.training_runs:
            return {"error": "One or both runs not found"}

        run1 = self.training_runs[run_id_1]
        run2 = self.training_runs[run_id_2]

        comparison = {
            "run1": run_id_1,
            "run2": run_id_2,
            "samples_delta": run2.samples_count - run1.samples_count,
            "loss_delta": None,
            "domain_differences": {},
        }

        if run1.final_loss is not None and run2.final_loss is not None:
            comparison["loss_delta"] = run2.final_loss - run1.final_loss
            comparison["loss_improvement"] = run1.final_loss > run2.final_loss

        # Domain distribution differences
        all_domains = set(run1.domain_distribution.keys()) | set(run2.domain_distribution.keys())
        for domain in all_domains:
            count1 = run1.domain_distribution.get(domain, 0)
            count2 = run2.domain_distribution.get(domain, 0)
            comparison["domain_differences"][domain] = {
                "run1": count1,
                "run2": count2,
                "delta": count2 - count1,
            }

        # Eval metric differences
        if run1.eval_metrics and run2.eval_metrics:
            comparison["eval_deltas"] = {}
            all_metrics = set(run1.eval_metrics.keys()) | set(run2.eval_metrics.keys())
            for metric in all_metrics:
                val1 = run1.eval_metrics.get(metric, 0)
                val2 = run2.eval_metrics.get(metric, 0)
                comparison["eval_deltas"][metric] = val2 - val1

        return comparison
