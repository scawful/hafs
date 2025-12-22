"""DataCurator - Coordinating agent for training data pipeline.

Orchestrates domain-specific generators, manages quality refinement,
and produces curated datasets for finetuning.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.autonomy.base import MemoryAwareAgent
from agents.training.augmentation import SyntheticAugmenter
from agents.training.base import DataGenerator, GenerationResult, TrainingSample
from agents.training.cross_domain import CrossDomainGenerator
from agents.training.quality import QualityPipeline

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Train/val/test split of samples."""

    train: list[TrainingSample]
    val: list[TrainingSample]
    test: list[TrainingSample]

    @property
    def total(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


@dataclass
class CurationStats:
    """Statistics from a curation run."""

    total_generated: int
    passed_quality: int
    deduplicated: int  # Duplicates removed after quality filtering
    augmented: int  # Synthetic augmented samples generated
    final_count: int
    domain_counts: dict[str, int] = field(default_factory=dict)
    quality_scores: dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_generated": self.total_generated,
            "passed_quality": self.passed_quality,
            "deduplicated": self.deduplicated,
            "augmented": self.augmented,
            "final_count": self.final_count,
            "domain_counts": self.domain_counts,
            "quality_scores": self.quality_scores,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class CurationResult:
    """Result from a curation run."""

    splits: DatasetSplit
    stats: CurationStats
    output_dir: Optional[Path] = None

    @property
    def samples(self) -> list[TrainingSample]:
        """All samples across splits."""
        return self.splits.train + self.splits.val + self.splits.test


class DataCurator(MemoryAwareAgent):
    """Coordinating agent for training data pipeline.

    Responsibilities:
    - Register and manage domain generators
    - Coordinate generation across domains
    - Deduplicate samples using embeddings
    - Validate against knowledge graph
    - Balance domains for final dataset
    - Generate train/val/test splits
    """

    # Default split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    def __init__(self):
        super().__init__(
            "DataCurator",
            "Coordinate training data generation and quality refinement",
        )
        self._generators: dict[str, DataGenerator] = {}
        self._quality_pipeline: Optional[QualityPipeline] = None
        self._augmenter: Optional[SyntheticAugmenter] = None
        self._cross_domain: Optional[CrossDomainGenerator] = None
        self._orchestrator = None

        # Paths
        self.training_dir = self.context_root / "training"
        self.output_dir = self.training_dir / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize components."""
        await super().setup()

        # Initialize quality pipeline
        self._quality_pipeline = QualityPipeline()
        await self._quality_pipeline.setup()

        # Share orchestrator with quality pipeline
        self._orchestrator = self._quality_pipeline.orchestrator

        # Initialize augmenter
        self._augmenter = SyntheticAugmenter(orchestrator=self._orchestrator)
        await self._augmenter.setup()

        # Initialize cross-domain generator
        self._cross_domain = CrossDomainGenerator(orchestrator=self._orchestrator)
        await self._cross_domain.setup()

    def register_generator(self, domain: str, generator: DataGenerator) -> None:
        """Register a domain-specific generator.

        Args:
            domain: Domain identifier (e.g., "asm", "cpp", "text")
            generator: DataGenerator instance for this domain
        """
        self._generators[domain] = generator
        logger.info(f"Registered generator for domain: {domain}")

    def unregister_generator(self, domain: str) -> None:
        """Unregister a generator."""
        if domain in self._generators:
            del self._generators[domain]

    def get_generator(self, domain: str) -> Optional[DataGenerator]:
        """Get a registered generator by domain."""
        return self._generators.get(domain)

    def list_domains(self) -> list[str]:
        """List all registered domains."""
        return list(self._generators.keys())

    async def _load_source_items(
        self,
        domain: str,
        max_items: Optional[int] = None,
    ) -> list[Any]:
        """Load and optionally downsample source items for a domain."""
        generator = self._generators.get(domain)
        if not generator:
            return []

        items = await generator.extract_source_items()
        if max_items and len(items) > max_items:
            return random.sample(items, max_items)
        return items

    async def register_default_generators(self) -> None:
        """Register the default set of generators."""
        from agents.training.generators import (
            AsmDataGenerator,
            CppDataGenerator,
            CuratedHackGenerator,
            TextDataGenerator,
        )

        # ASM generator
        asm_gen = AsmDataGenerator()
        await asm_gen.setup()
        self.register_generator("asm", asm_gen)

        # C++ generator (check if yaze path exists)
        cpp_gen = CppDataGenerator()
        if cpp_gen.yaze_path.exists():
            await cpp_gen.setup()
            self.register_generator("cpp", cpp_gen)
        else:
            logger.warning(f"Yaze path not found: {cpp_gen.yaze_path}")

        # Text generator
        text_gen = TextDataGenerator()
        await text_gen.setup()
        self.register_generator("text", text_gen)

        # Curated hack generator (allowlist)
        curated_gen = CuratedHackGenerator()
        await curated_gen.setup()
        if curated_gen.has_hacks:
            self.register_generator("hack_curated", curated_gen)
        else:
            logger.warning(
                f"Curated hack allowlist is empty or missing: {curated_gen.CONFIG_PATH}"
            )

    async def generate_from_domain(
        self,
        domain: str,
        limit: Optional[int] = None,
        resume: bool = True,
    ) -> GenerationResult:
        """Generate samples from a single domain.

        Args:
            domain: Domain to generate from
            limit: Maximum samples to generate
            resume: Whether to resume from checkpoint

        Returns:
            GenerationResult with samples and metrics
        """
        generator = self._generators.get(domain)
        if not generator:
            raise ValueError(f"No generator registered for domain: {domain}")

        return await generator.run_generation(limit=limit, resume=resume)

    async def curate_dataset(
        self,
        domains: Optional[list[str]] = None,
        target_count: int = 1000,
        quality_threshold: Optional[float] = None,
        balance_domains: bool = True,
        output_name: Optional[str] = None,
        resume: bool = False,
        cross_domain_samples: int = 0,
    ) -> CurationResult:
        """Curate a training dataset from multiple domains.

        Args:
            domains: List of domains to include (None = all registered)
            target_count: Target number of samples
            quality_threshold: Minimum quality score (None = use domain-specific thresholds)
            balance_domains: Whether to balance samples across domains
            output_name: Name for output files
            resume: Resume from checkpoints if available
            cross_domain_samples: Number of cross-domain samples to generate (0 = disabled)

        Returns:
            CurationResult with splits, stats, and output paths
        """
        import time

        start_time = time.time()

        if not self._quality_pipeline:
            await self.setup()

        domains = domains or list(self._generators.keys())
        if not domains:
            raise ValueError("No domains available for curation")

        # Calculate per-domain limits
        if balance_domains:
            per_domain_limit = target_count // len(domains)
        else:
            per_domain_limit = target_count

        # Generate from each domain
        all_samples: list[TrainingSample] = []
        domain_counts: dict[str, int] = {}

        for domain in domains:
            generator = self._generators.get(domain)
            if not generator:
                logger.warning(f"No generator for domain: {domain}")
                continue

            # Clear checkpoint if not resuming
            if not resume:
                generator.clear_checkpoint()

            logger.info(f"Generating from domain: {domain}")

            result = await generator.run_generation(
                limit=per_domain_limit,
                resume=resume,
            )

            all_samples.extend(result.samples)
            domain_counts[domain] = len(result.samples)

        # Generate cross-domain samples if requested
        cross_domain_count = 0
        cross_domain_generated: list[TrainingSample] = []
        if cross_domain_samples > 0 and self._cross_domain:
            logger.info(f"Generating {cross_domain_samples} cross-domain samples...")

            combo_specs: list[tuple[str, str, str]] = []
            if "asm" in domains and "oracle" in domains:
                combo_specs.append(("asm+oracle", "asm", "oracle"))
            if "asm" in domains and "gigaleak" in domains:
                combo_specs.append(("asm+gigaleak", "asm", "gigaleak"))

            if not combo_specs:
                logger.info("No compatible domain pairs for cross-domain generation")
            else:
                per_combo = max(cross_domain_samples // len(combo_specs), 1)
                remainder = max(cross_domain_samples - (per_combo * len(combo_specs)), 0)

                for idx, (combo_type, primary_domain, secondary_domain) in enumerate(combo_specs):
                    target_pairs = per_combo + (1 if idx < remainder else 0)
                    if target_pairs <= 0:
                        continue

                    primary_items = await self._load_source_items(primary_domain, max_items=target_pairs * 6)
                    secondary_items = await self._load_source_items(secondary_domain, max_items=target_pairs * 6)

                    # Favor vanilla routines for pairing
                    if combo_type in ("asm+oracle", "asm+gigaleak"):
                        primary_items = [item for item in primary_items if item.source == "vanilla"]

                    # Prefer hooks for Oracle pairing
                    if combo_type == "asm+oracle":
                        secondary_items = [
                            item for item in secondary_items
                            if getattr(item, "is_hook", False) or getattr(item, "hooks_vanilla", None)
                        ] or secondary_items

                    pairs = await self._cross_domain.find_related_pairs(
                        primary_items,
                        secondary_items,
                        combo_type,
                        max_pairs=target_pairs,
                    )

                    for pair in pairs:
                        if combo_type == "asm+oracle":
                            sample = await self._cross_domain.generate_asm_oracle_pair(
                                pair.primary,
                                pair.secondary,
                            )
                        elif combo_type == "asm+gigaleak":
                            sample = await self._cross_domain.generate_asm_gigaleak_pair(
                                pair.primary,
                                pair.secondary,
                            )
                        else:
                            sample = None

                        if sample:
                            cross_domain_generated.append(sample)

        if cross_domain_generated:
            cross_domain_count = len(cross_domain_generated)
            all_samples.extend(cross_domain_generated)
            for sample in cross_domain_generated:
                domain_counts[sample.domain] = domain_counts.get(sample.domain, 0) + 1

        total_generated = len(all_samples)
        logger.info(f"Total generated: {total_generated} ({cross_domain_count} cross-domain)")

        # Quality filter and deduplicate
        filtered = await self._quality_pipeline.filter_samples(
            all_samples,
            min_quality=quality_threshold,
            deduplicate=True,
        )

        passed_quality = len(filtered)
        deduplicated = 0
        if self._quality_pipeline and self._quality_pipeline.last_filter_stats:
            filter_stats = self._quality_pipeline.last_filter_stats
            passed_quality = filter_stats.passed_quality
            deduplicated = filter_stats.rejected_duplicates
        logger.info(f"Passed quality filter: {passed_quality}")

        # Augment high-quality samples for diversity
        augmented_count = 0
        if self._augmenter:
            logger.info("Augmenting high-quality samples for diversity...")
            augmented_samples = await self._augmenter.augment_batch(filtered)
            augmented_count = len(augmented_samples)

            if augmented_samples:
                # Add augmented samples to filtered set
                filtered.extend(augmented_samples)
                logger.info(
                    f"Generated {augmented_count} augmented samples "
                    f"from {len([s for s in filtered if s.quality_score >= self._augmenter.config.min_quality_threshold])} "
                    f"high-quality samples (threshold: {self._augmenter.config.min_quality_threshold})"
                )

        # Balance base domains if requested (keep cross-domain samples additive)
        if balance_domains and len(domains) > 1:
            base_samples = [s for s in filtered if "+" not in s.domain]
            cross_samples = [s for s in filtered if "+" in s.domain]
            filtered = self._balance_by_domain(base_samples, target_count)
            filtered.extend(cross_samples)

        # Create splits
        splits = self._create_splits(filtered)

        # Compute stats
        avg_quality = (
            sum(s.quality_score for s in filtered) / len(filtered)
            if filtered
            else 0.0
        )

        stats = CurationStats(
            total_generated=total_generated,
            passed_quality=passed_quality,
            deduplicated=deduplicated,
            augmented=augmented_count,
            final_count=splits.total,
            domain_counts=domain_counts,
            quality_scores={"average": avg_quality},
            duration_seconds=time.time() - start_time,
        )

        # Save outputs
        output_dir = None
        if output_name:
            output_dir = await self._save_dataset(splits, stats, output_name)

            # Save rejected samples for analysis
            if self._quality_pipeline and self._quality_pipeline.last_filter_stats:
                await self._save_rejected_samples(
                    output_dir,
                    self._quality_pipeline.last_filter_stats.rejected_samples
                )

        result = CurationResult(
            splits=splits,
            stats=stats,
            output_dir=output_dir,
        )

        # Remember the curation run
        await self.remember(
            content=f"Curated dataset with {stats.final_count} samples from {domains}",
            memory_type="curation_run",
            context=stats.to_dict(),
            importance=0.7,
        )

        return result

    def _balance_by_domain(
        self,
        samples: list[TrainingSample],
        target_count: int,
    ) -> list[TrainingSample]:
        """Balance samples across domains."""
        by_domain: dict[str, list[TrainingSample]] = {}

        for sample in samples:
            domain = sample.domain
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(sample)

        # Calculate per-domain quota
        num_domains = len(by_domain)
        if num_domains == 0:
            return []  # No samples to balance
        per_domain = target_count // num_domains

        balanced: list[TrainingSample] = []
        for domain, domain_samples in by_domain.items():
            # Sort by quality, take top N
            sorted_samples = sorted(
                domain_samples,
                key=lambda s: s.quality_score,
                reverse=True,
            )
            balanced.extend(sorted_samples[:per_domain])

        return balanced

    def _create_splits(self, samples: list[TrainingSample]) -> DatasetSplit:
        """Create train/val/test splits."""
        # Shuffle samples
        shuffled = samples.copy()
        random.shuffle(shuffled)

        total = len(shuffled)
        train_end = int(total * self.TRAIN_RATIO)
        val_end = train_end + int(total * self.VAL_RATIO)

        return DatasetSplit(
            train=shuffled[:train_end],
            val=shuffled[train_end:val_end],
            test=shuffled[val_end:],
        )

    async def _save_dataset(
        self,
        splits: DatasetSplit,
        stats: CurationStats,
        name: str,
        template: str = "alpaca",
    ) -> Path:
        """Save dataset to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_dir = self.output_dir / f"{name}_{timestamp}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Save splits as JSONL
        for split_name, samples in [
            ("train", splits.train),
            ("val", splits.val),
            ("test", splits.test),
        ]:
            path = dataset_dir / f"{split_name}.jsonl"
            with open(path, "w") as f:
                for sample in samples:
                    f.write(sample.to_jsonl_entry(template) + "\n")

        # Save stats
        stats_path = dataset_dir / "stats.json"
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2))

        # Save metadata
        metadata = {
            "name": name,
            "created": timestamp,
            "template": template,
            "train_count": len(splits.train),
            "val_count": len(splits.val),
            "test_count": len(splits.test),
            "domains": list(stats.domain_counts.keys()),
        }
        metadata_path = dataset_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        logger.info(f"Saved dataset to {dataset_dir}")
        return dataset_dir

    async def _save_rejected_samples(
        self,
        dataset_dir: Path,
        rejected_samples: list,
        template: str = "alpaca",
    ) -> None:
        """Save rejected samples to separate file for analysis."""
        if not rejected_samples:
            return

        rejected_path = dataset_dir / "rejected.jsonl"
        rejected_count = 0

        with open(rejected_path, "w") as f:
            for rejected in rejected_samples:
                sample = rejected.sample
                # Create JSONL entry with rejection metadata
                entry = {
                    "instruction": sample.instruction,
                    "input": sample.input or "",
                    "output": sample.output,
                    "domain": sample.domain,
                    "source": sample.source,
                    "rejection_reason": rejected.reason,
                    "quality_score": rejected.quality_score,
                }

                # Add rejection details if available
                if rejected.details:
                    entry["rejection_details"] = rejected.details

                f.write(json.dumps(entry) + "\n")
                rejected_count += 1

        logger.info(f"Saved {rejected_count} rejected samples to {rejected_path}")

        # Save rejection summary
        summary_path = dataset_dir / "rejection_summary.json"

        # Count by reason
        by_reason = {}
        by_domain = {}
        quality_scores = []

        for rejected in rejected_samples:
            reason = rejected.reason
            domain = rejected.sample.domain

            by_reason[reason] = by_reason.get(reason, 0) + 1
            by_domain[domain] = by_domain.get(domain, 0) + 1

            if rejected.quality_score is not None:
                quality_scores.append(rejected.quality_score)

        summary = {
            "total_rejected": len(rejected_samples),
            "by_reason": by_reason,
            "by_domain": by_domain,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            "min_quality_score": min(quality_scores) if quality_scores else 0.0,
            "max_quality_score": max(quality_scores) if quality_scores else 0.0,
        }

        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info(f"Saved rejection summary to {summary_path}")

    async def run_task(self, task: dict[str, Any]) -> str:
        """Run curation task (BaseAgent interface)."""
        domains = task.get("domains")
        target_count = task.get("target_count", 1000)
        quality_threshold = task.get("quality_threshold", 0.7)
        output_name = task.get("output_name", "curated_dataset")

        result = await self.curate_dataset(
            domains=domains,
            target_count=target_count,
            quality_threshold=quality_threshold,
            output_name=output_name,
        )

        return (
            f"Curated {result.stats.final_count} samples "
            f"(train: {len(result.splits.train)}, "
            f"val: {len(result.splits.val)}, "
            f"test: {len(result.splits.test)})"
        )
