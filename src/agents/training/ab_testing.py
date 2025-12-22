"""A/B Testing Framework for Training Data Generator Prompts.

Compares quality metrics between different prompt versions to guide
iterative prompt engineering improvements.

Usage:
    from agents.training.ab_testing import ABTestRunner, PromptVersion

    runner = ABTestRunner()

    # Define prompt versions
    v1 = PromptVersion(name="baseline", generator_cls=AsmDataGenerator)
    v2 = PromptVersion(name="enhanced", generator_cls=AsmDataGenerator,
                       use_enhanced=True)

    # Run A/B test
    results = await runner.run_test(
        versions=[v1, v2],
        num_samples=1000,
        source_limit=200
    )

    # Generate report
    runner.save_report(results, "ab_test_results.json")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

from agents.training.base import DataGenerator, TrainingSample
from agents.training.quality import QualityPipeline

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """Configuration for a prompt version to test."""

    name: str
    generator_cls: Type[DataGenerator]
    use_enhanced: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.name} ({'enhanced' if self.use_enhanced else 'baseline'})"


@dataclass
class ABTestResult:
    """Results from comparing two prompt versions."""

    version_name: str
    samples_generated: int
    samples_passed: int
    samples_failed: int
    pass_rate: float

    # Quality metrics
    avg_quality_score: float
    avg_diversity: float
    avg_kg_consistency: float
    avg_hallucination: float
    avg_coherence: float

    # Rejection breakdown
    rejected_validation: int
    rejected_quality: int
    rejected_duplicates: int

    # Performance
    generation_time_seconds: float
    samples_per_second: float

    # Sample quality distribution
    quality_distribution: dict[str, int]  # bins -> count

    # Top rejection reasons
    rejection_reasons: dict[str, int]

    # Example samples
    top_samples: list[dict[str, Any]]
    failed_samples: list[dict[str, Any]]


@dataclass
class ABTestComparison:
    """Side-by-side comparison of A/B test results."""

    test_id: str
    timestamp: str
    domain: str
    num_samples_target: int

    baseline: ABTestResult
    enhanced: ABTestResult

    # Improvement metrics
    pass_rate_improvement: float
    quality_improvement: float
    diversity_improvement: float
    coherence_improvement: float

    # Statistical significance (if enough samples)
    is_significant: bool
    confidence_level: float

    # Winner
    winner: str
    recommendation: str


class ABTestRunner:
    """A/B test runner for prompt comparison."""

    def __init__(
        self,
        output_dir: Path | None = None,
        enable_quality_pipeline: bool = True,
    ):
        """Initialize A/B test runner.

        Args:
            output_dir: Directory for test results
            enable_quality_pipeline: Whether to run quality filtering
        """
        self.output_dir = output_dir or Path.home() / ".context" / "training" / "ab_tests"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.enable_quality_pipeline = enable_quality_pipeline
        self.quality_pipeline: Optional[QualityPipeline] = None

    async def setup(self):
        """Initialize quality pipeline."""
        if self.enable_quality_pipeline:
            self.quality_pipeline = QualityPipeline(
                enable_validators=True,
                enable_feedback=True,
                enable_active_learning=False,  # Don't bias results
            )
            await self.quality_pipeline.setup()

    async def run_test(
        self,
        versions: list[PromptVersion],
        num_samples: int = 1000,
        source_limit: Optional[int] = None,
        domain: str = "asm",
    ) -> ABTestComparison:
        """Run A/B test comparing prompt versions.

        Args:
            versions: List of prompt versions to test (typically 2)
            num_samples: Target number of samples to generate per version
            source_limit: Limit source items (for faster testing)
            domain: Domain being tested

        Returns:
            Comparison results
        """
        if len(versions) != 2:
            raise ValueError("A/B test requires exactly 2 versions")

        if not self.quality_pipeline:
            await self.setup()

        logger.info(f"Starting A/B test: {versions[0]} vs {versions[1]}")
        logger.info(f"Target: {num_samples} samples per version")

        # Run both versions
        results = []
        for version in versions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {version}")
            logger.info(f"{'='*60}\n")

            result = await self._run_version(
                version=version,
                num_samples=num_samples,
                source_limit=source_limit,
            )
            results.append(result)

        # Compare results
        comparison = self._compare_results(
            baseline=results[0],
            enhanced=results[1],
            domain=domain,
            num_samples=num_samples,
        )

        # Save comparison
        self._save_comparison(comparison)

        return comparison

    async def _run_version(
        self,
        version: PromptVersion,
        num_samples: int,
        source_limit: Optional[int] = None,
    ) -> ABTestResult:
        """Run generation for a single prompt version.

        Args:
            version: Prompt version configuration
            num_samples: Target number of samples
            source_limit: Limit source items

        Returns:
            Test results for this version
        """
        start_time = datetime.now()

        # Initialize generator
        generator = version.generator_cls()

        # Configure to use enhanced prompts if requested
        if version.use_enhanced:
            generator.use_enhanced_prompts = True

        await generator.setup()

        # Extract source items
        source_items = await generator.extract_source_items()

        if source_limit:
            source_items = source_items[:source_limit]

        logger.info(f"Extracted {len(source_items)} source items")

        # Generate samples
        generated_samples: list[TrainingSample] = []

        for i, item in enumerate(source_items):
            if len(generated_samples) >= num_samples:
                break

            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(source_items)} items, {len(generated_samples)} samples")

            sample = await generator.generate_sample(item)
            if sample:
                generated_samples.append(sample)

        logger.info(f"Generated {len(generated_samples)} samples")

        # Apply quality filtering
        passed_samples: list[TrainingSample] = []
        rejected_validation = 0
        rejected_quality = 0
        rejected_duplicates = 0

        if self.quality_pipeline:
            passed_samples = await self.quality_pipeline.filter_samples(
                samples=generated_samples,
                min_quality=None,  # Use domain-specific thresholds
                deduplicate=True,
                generator_name=version.name,
            )

            stats = self.quality_pipeline.last_filter_stats
            if stats:
                rejected_validation = stats.rejected_validation
                rejected_quality = stats.rejected_quality
                rejected_duplicates = stats.rejected_duplicates
        else:
            passed_samples = generated_samples

        # Calculate metrics
        quality_scores = [s.quality_score for s in generated_samples if s.quality_score is not None]

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Quality component averages (from passed samples)
        diversity_scores = []
        kg_scores = []
        hallucination_scores = []
        coherence_scores = []

        for sample in generated_samples:
            if hasattr(sample, '_quality_components'):
                components = sample._quality_components
                diversity_scores.append(components.get('diversity_score', 0))
                kg_scores.append(components.get('kg_consistency', 0))
                hallucination_scores.append(components.get('hallucination_risk', 0))
                coherence_scores.append(components.get('semantic_coherence', 0))

        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
        avg_kg = sum(kg_scores) / len(kg_scores) if kg_scores else 0.0
        avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

        # Quality distribution (bins: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0)
        quality_bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for score in quality_scores:
            if score < 0.2:
                quality_bins["0.0-0.2"] += 1
            elif score < 0.4:
                quality_bins["0.2-0.4"] += 1
            elif score < 0.6:
                quality_bins["0.4-0.6"] += 1
            elif score < 0.8:
                quality_bins["0.6-0.8"] += 1
            else:
                quality_bins["0.8-1.0"] += 1

        # Rejection reasons
        rejection_reasons: dict[str, int] = {}
        if self.quality_pipeline and self.quality_pipeline._feedback_tracker:
            patterns = self.quality_pipeline._feedback_tracker.get_rejection_patterns()
            # patterns is a list of dicts, need to aggregate by reason
            for pattern in patterns:
                reason = pattern.get("reason", "unknown")
                count = pattern.get("count", 0)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + count

        # Top samples (highest quality)
        top_samples = sorted(
            [s for s in passed_samples if s.quality_score],
            key=lambda s: s.quality_score or 0,
            reverse=True
        )[:5]

        # Failed samples (lowest quality)
        failed = [s for s in generated_samples if s not in passed_samples]
        failed_samples = sorted(
            [s for s in failed if s.quality_score],
            key=lambda s: s.quality_score or 0,
        )[:5]

        # Performance metrics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        samples_per_sec = len(generated_samples) / duration if duration > 0 else 0

        return ABTestResult(
            version_name=version.name,
            samples_generated=len(generated_samples),
            samples_passed=len(passed_samples),
            samples_failed=len(generated_samples) - len(passed_samples),
            pass_rate=len(passed_samples) / len(generated_samples) if generated_samples else 0.0,
            avg_quality_score=avg_quality,
            avg_diversity=avg_diversity,
            avg_kg_consistency=avg_kg,
            avg_hallucination=avg_hallucination,
            avg_coherence=avg_coherence,
            rejected_validation=rejected_validation,
            rejected_quality=rejected_quality,
            rejected_duplicates=rejected_duplicates,
            generation_time_seconds=duration,
            samples_per_second=samples_per_sec,
            quality_distribution=quality_bins,
            rejection_reasons=rejection_reasons,
            top_samples=[self._sample_to_dict(s) for s in top_samples],
            failed_samples=[self._sample_to_dict(s) for s in failed_samples],
        )

    def _sample_to_dict(self, sample: TrainingSample) -> dict[str, Any]:
        """Convert sample to dict for reporting."""
        return {
            "instruction": sample.instruction[:200] + "..." if len(sample.instruction) > 200 else sample.instruction,
            "output_preview": sample.output[:200] + "..." if len(sample.output) > 200 else sample.output,
            "quality_score": sample.quality_score,
            "domain": sample.domain,
        }

    def _compare_results(
        self,
        baseline: ABTestResult,
        enhanced: ABTestResult,
        domain: str,
        num_samples: int,
    ) -> ABTestComparison:
        """Compare baseline vs enhanced results.

        Args:
            baseline: Baseline prompt results
            enhanced: Enhanced prompt results
            domain: Domain tested
            num_samples: Target sample count

        Returns:
            Comparison report
        """
        # Calculate improvements
        pass_rate_improvement = enhanced.pass_rate - baseline.pass_rate
        quality_improvement = enhanced.avg_quality_score - baseline.avg_quality_score
        diversity_improvement = enhanced.avg_diversity - baseline.avg_diversity
        coherence_improvement = enhanced.avg_coherence - baseline.avg_coherence

        # Determine winner
        winner = "enhanced" if pass_rate_improvement > 0 else "baseline"

        # Generate recommendation
        if pass_rate_improvement > 0.1:  # 10%+ improvement
            recommendation = f"STRONG RECOMMENDATION: Use enhanced prompts (+{pass_rate_improvement*100:.1f}% pass rate)"
        elif pass_rate_improvement > 0.05:  # 5-10% improvement
            recommendation = f"RECOMMENDATION: Use enhanced prompts (+{pass_rate_improvement*100:.1f}% pass rate)"
        elif pass_rate_improvement > 0:  # Slight improvement
            recommendation = f"MARGINAL: Enhanced prompts slightly better (+{pass_rate_improvement*100:.1f}%)"
        elif pass_rate_improvement > -0.05:  # Neutral
            recommendation = "NEUTRAL: No significant difference"
        else:  # Worse
            recommendation = f"WARNING: Enhanced prompts WORSE ({pass_rate_improvement*100:.1f}% pass rate)"

        # Statistical significance (simple threshold for now)
        is_significant = abs(pass_rate_improvement) > 0.05 and num_samples >= 100
        confidence = 0.95 if is_significant else 0.0

        test_id = f"ab_test_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ABTestComparison(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            domain=domain,
            num_samples_target=num_samples,
            baseline=baseline,
            enhanced=enhanced,
            pass_rate_improvement=pass_rate_improvement,
            quality_improvement=quality_improvement,
            diversity_improvement=diversity_improvement,
            coherence_improvement=coherence_improvement,
            is_significant=is_significant,
            confidence_level=confidence,
            winner=winner,
            recommendation=recommendation,
        )

    def _save_comparison(self, comparison: ABTestComparison):
        """Save comparison results to JSON."""
        output_file = self.output_dir / f"{comparison.test_id}.json"

        # Convert to dict
        data = {
            "test_id": comparison.test_id,
            "timestamp": comparison.timestamp,
            "domain": comparison.domain,
            "num_samples_target": comparison.num_samples_target,
            "baseline": self._result_to_dict(comparison.baseline),
            "enhanced": self._result_to_dict(comparison.enhanced),
            "improvements": {
                "pass_rate": comparison.pass_rate_improvement,
                "quality": comparison.quality_improvement,
                "diversity": comparison.diversity_improvement,
                "coherence": comparison.coherence_improvement,
            },
            "statistical": {
                "is_significant": comparison.is_significant,
                "confidence_level": comparison.confidence_level,
            },
            "winner": comparison.winner,
            "recommendation": comparison.recommendation,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved comparison to {output_file}")

        # Also print summary
        self._print_comparison(comparison)

    def _result_to_dict(self, result: ABTestResult) -> dict[str, Any]:
        """Convert result to dict."""
        return {
            "version_name": result.version_name,
            "samples_generated": result.samples_generated,
            "samples_passed": result.samples_passed,
            "samples_failed": result.samples_failed,
            "pass_rate": result.pass_rate,
            "avg_quality_score": result.avg_quality_score,
            "avg_diversity": result.avg_diversity,
            "avg_kg_consistency": result.avg_kg_consistency,
            "avg_hallucination": result.avg_hallucination,
            "avg_coherence": result.avg_coherence,
            "rejected_validation": result.rejected_validation,
            "rejected_quality": result.rejected_quality,
            "rejected_duplicates": result.rejected_duplicates,
            "generation_time_seconds": result.generation_time_seconds,
            "samples_per_second": result.samples_per_second,
            "quality_distribution": result.quality_distribution,
            "rejection_reasons": result.rejection_reasons,
            "top_samples": result.top_samples,
            "failed_samples": result.failed_samples,
        }

    def _print_comparison(self, comparison: ABTestComparison):
        """Print comparison results to console."""
        print("\n" + "="*80)
        print(f"A/B TEST RESULTS: {comparison.domain.upper()}")
        print("="*80)
        print(f"Test ID: {comparison.test_id}")
        print(f"Timestamp: {comparison.timestamp}")
        print(f"Target Samples: {comparison.num_samples_target}")
        print()

        print("BASELINE vs ENHANCED")
        print("-"*80)
        print(f"{'Metric':<30} {'Baseline':>15} {'Enhanced':>15} {'Improvement':>15}")
        print("-"*80)
        print(f"{'Samples Generated':<30} {comparison.baseline.samples_generated:>15} {comparison.enhanced.samples_generated:>15} {'':<15}")
        print(f"{'Samples Passed':<30} {comparison.baseline.samples_passed:>15} {comparison.enhanced.samples_passed:>15} {'':<15}")
        print(f"{'Pass Rate':<30} {comparison.baseline.pass_rate:>14.1%} {comparison.enhanced.pass_rate:>14.1%} {comparison.pass_rate_improvement:>+14.1%}")
        print(f"{'Avg Quality Score':<30} {comparison.baseline.avg_quality_score:>15.3f} {comparison.enhanced.avg_quality_score:>15.3f} {comparison.quality_improvement:>+15.3f}")
        print(f"{'Avg Diversity':<30} {comparison.baseline.avg_diversity:>15.3f} {comparison.enhanced.avg_diversity:>15.3f} {comparison.diversity_improvement:>+15.3f}")
        print(f"{'Avg Coherence':<30} {comparison.baseline.avg_coherence:>15.3f} {comparison.enhanced.avg_coherence:>15.3f} {comparison.coherence_improvement:>+15.3f}")
        print(f"{'Avg Hallucination':<30} {comparison.baseline.avg_hallucination:>15.3f} {comparison.enhanced.avg_hallucination:>15.3f} {comparison.enhanced.avg_hallucination - comparison.baseline.avg_hallucination:>+15.3f}")
        print()

        print("REJECTION BREAKDOWN")
        print("-"*80)
        print(f"{'Reason':<30} {'Baseline':>15} {'Enhanced':>15}")
        print("-"*80)
        print(f"{'Validation Failed':<30} {comparison.baseline.rejected_validation:>15} {comparison.enhanced.rejected_validation:>15}")
        print(f"{'Quality Too Low':<30} {comparison.baseline.rejected_quality:>15} {comparison.enhanced.rejected_quality:>15}")
        print(f"{'Duplicates':<30} {comparison.baseline.rejected_duplicates:>15} {comparison.enhanced.rejected_duplicates:>15}")
        print()

        print("VERDICT")
        print("-"*80)
        print(f"Winner: {comparison.winner.upper()}")
        print(f"Significant: {'YES' if comparison.is_significant else 'NO'} (confidence: {comparison.confidence_level:.0%})")
        print(f"\n{comparison.recommendation}")
        print("="*80 + "\n")


async def main():
    """Example A/B test run."""
    from agents.training.generators import AsmDataGenerator

    # Create test runner
    runner = ABTestRunner()

    # Define versions
    baseline = PromptVersion(
        name="baseline",
        generator_cls=AsmDataGenerator,
        use_enhanced=False,
    )

    enhanced = PromptVersion(
        name="enhanced_v1",
        generator_cls=AsmDataGenerator,
        use_enhanced=True,
    )

    # Run A/B test
    comparison = await runner.run_test(
        versions=[baseline, enhanced],
        num_samples=100,  # Small test
        source_limit=50,
        domain="asm",
    )

    print(f"\nTest complete! Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
