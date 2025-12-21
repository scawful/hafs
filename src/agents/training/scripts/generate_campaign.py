#!/usr/bin/env python3
"""Training Data Generation Campaign Script.

Orchestrates large-scale training data generation across 5 domains:
1. ALTTP ASM (15K target)
2. Gigaleak source (8K target)
3. Oracle ROM hack (4K target)
4. YAZE C++ tools (6K target)
5. Error diagnostics (1.5K target)

Total target: 34.5K samples for dual-agent pipeline training.

Usage:
    python -m agents.training.scripts.generate_campaign --target 34500
    python -m agents.training.scripts.generate_campaign --pilot 1000
    python -m agents.training.scripts.generate_campaign --resume
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.training.curator import DataCurator, CurationResult
from agents.training.exporter import TrainingExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def register_all_generators(curator: DataCurator) -> None:
    """Register all 5 generators for the campaign.

    Generators:
    1. AsmDataGenerator - vanilla/hack ALTTP ASM routines
    2. GigaleakDataGenerator - Nintendo original source symbols
    3. OracleDataGenerator - Oracle ROM hack routines
    4. CppDataGenerator - YAZE C++ tools (yaze domain)
    5. ErrorSampleGenerator - system error diagnostics
    """
    logger.info("Registering all generators...")

    # Import generators
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.generators.cpp_generator import CppDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.oracle_generator import OracleDataGenerator
    from agents.training.generators.text_generator import TextDataGenerator

    # 1. ASM Generator (15K target)
    logger.info("Setting up AsmDataGenerator...")
    asm_gen = AsmDataGenerator()
    await asm_gen.setup()
    curator.register_generator("asm", asm_gen)
    logger.info("✓ ASM generator registered (target: 15K)")

    # 2. Gigaleak Generator (8K target)
    logger.info("Setting up GigaleakDataGenerator...")
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    curator.register_generator("gigaleak", gigaleak_gen)
    logger.info("✓ Gigaleak generator registered (target: 8K)")

    # 3. Oracle Generator (4K target)
    logger.info("Setting up OracleDataGenerator...")
    oracle_gen = OracleDataGenerator()
    await oracle_gen.setup()
    curator.register_generator("oracle", oracle_gen)
    logger.info("✓ Oracle generator registered (target: 4K)")

    # 4. YAZE/C++ Generator (6K target)
    logger.info("Setting up CppDataGenerator (YAZE)...")
    cpp_gen = CppDataGenerator()
    if cpp_gen.yaze_path.exists():
        await cpp_gen.setup()
        curator.register_generator("yaze", cpp_gen)  # Register as "yaze" domain
        logger.info("✓ YAZE generator registered (target: 6K)")
    else:
        logger.warning(
            f"YAZE path not found: {cpp_gen.yaze_path} - skipping YAZE generator"
        )

    # 5. Error Sample Generator (1.5K target)
    logger.info("Setting up ErrorSampleGenerator...")
    error_gen = ErrorSampleGenerator(
        lookback_hours=168,  # 7 days
        min_severity="low",  # Include all severities for more samples
    )
    await error_gen.setup()
    curator.register_generator("errors", error_gen)
    logger.info("✓ Error generator registered (target: 1.5K)")

    # 6. Text Generator (supplemental, 1.5K target)
    logger.info("Setting up TextDataGenerator...")
    text_gen = TextDataGenerator()
    await text_gen.setup()
    curator.register_generator("text", text_gen)
    logger.info("✓ Text generator registered (target: 1.5K)")

    logger.info(f"All generators registered. Total domains: {len(curator.list_domains())}")


async def run_generation_campaign(
    target_count: int = 34500,
    output_name: Optional[str] = None,
    pilot: bool = False,
    resume: bool = False,
    enable_active_learning: bool = True,
    quality_threshold: Optional[float] = None,
) -> CurationResult:
    """Run the full generation campaign.

    Args:
        target_count: Total number of samples to generate
        output_name: Custom output directory name
        pilot: If True, run pilot mode (smaller target for validation)
        resume: Resume from checkpoints if available
        enable_active_learning: Use coverage-driven generation
        quality_threshold: Base quality threshold (adaptive)

    Returns:
        CurationResult with generated samples and stats
    """
    start_time = datetime.now()

    # Pilot mode overrides
    if pilot:
        target_count = min(target_count, 1000)
        logger.info(f"PILOT MODE: Limiting to {target_count} samples")

    # Default output name
    if not output_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "pilot" if pilot else "alttp_yaze_full"
        output_name = f"{prefix}_{target_count}_{timestamp}"

    logger.info("=" * 80)
    logger.info(f"TRAINING DATA GENERATION CAMPAIGN")
    logger.info(f"Target: {target_count} samples")
    logger.info(f"Output: {output_name}")
    logger.info(f"Quality threshold: {quality_threshold}")
    logger.info(f"Active learning: {enable_active_learning}")
    logger.info(f"Resume: {resume}")
    logger.info("=" * 80)

    # Initialize curator
    logger.info("Initializing DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Register all generators
    await register_all_generators(curator)

    # Domain targets (proportional allocation)
    # Total = 34,500 if not pilot
    domain_allocations = {
        "asm": 0.435,  # 15K / 34.5K = 43.5%
        "gigaleak": 0.232,  # 8K / 34.5K = 23.2%
        "oracle": 0.116,  # 4K / 34.5K = 11.6%
        "yaze": 0.174,  # 6K / 34.5K = 17.4%
        "errors": 0.043,  # 1.5K / 34.5K = 4.3%
    }

    # Calculate per-domain targets
    domain_targets = {
        domain: int(target_count * allocation)
        for domain, allocation in domain_allocations.items()
    }

    logger.info("Domain targets:")
    for domain, target in domain_targets.items():
        logger.info(f"  {domain}: {target} samples ({domain_allocations[domain]*100:.1f}%)")

    # Run curation
    logger.info("Starting curation...")
    logger.info("This will take 20-30 hours for full 34.5K generation")
    logger.info("Progress will be saved every 100 samples (checkpoints)")

    result = await curator.curate_dataset(
        domains=list(domain_targets.keys()),
        target_count=target_count,
        quality_threshold=quality_threshold,
        balance_domains=True,
        output_name=output_name,
        resume=resume,
    )

    # Log results
    duration = (datetime.now() - start_time).total_seconds()
    logger.info("=" * 80)
    logger.info("GENERATION CAMPAIGN COMPLETED")
    logger.info(f"Duration: {duration / 3600:.2f} hours")
    logger.info(f"Total generated: {result.stats.total_generated}")
    logger.info(f"Passed quality: {result.stats.passed_quality}")
    logger.info(f"Deduplicated: {result.stats.deduplicated}")
    logger.info(f"Final count: {result.stats.final_count}")
    logger.info(f"Quality acceptance rate: {result.stats.passed_quality / max(result.stats.total_generated, 1) * 100:.1f}%")
    logger.info("")
    logger.info("Domain breakdown:")
    for domain, count in result.stats.domain_counts.items():
        logger.info(f"  {domain}: {count} samples")
    logger.info("")
    logger.info("Quality scores:")
    for domain, score in result.stats.quality_scores.items():
        logger.info(f"  {domain}: {score:.3f}")
    logger.info("=" * 80)

    return result


async def export_datasets(
    result: CurationResult,
    export_asm_dataset: bool = True,
    export_yaze_dataset: bool = True,
) -> dict[str, Path]:
    """Export datasets for training.

    Creates two separate datasets:
    1. ALTTP ASM Dataset (asm + gigaleak + oracle + errors)
    2. YAZE Tool Dataset (yaze + errors)

    Args:
        result: CurationResult from generation campaign
        export_asm_dataset: Export ASM-focused dataset
        export_yaze_dataset: Export YAZE-focused dataset

    Returns:
        Dictionary mapping dataset names to output paths
    """
    logger.info("Exporting datasets...")

    exporter = TrainingExporter()
    output_paths = {}

    # Split samples by domain
    asm_samples = [s for s in result.samples if s.domain in ("asm", "gigaleak", "oracle", "errors")]
    yaze_samples = [s for s in result.samples if s.domain in ("yaze", "errors")]

    # Export ASM dataset
    if export_asm_dataset and asm_samples:
        logger.info(f"Exporting ALTTP ASM dataset ({len(asm_samples)} samples)...")
        asm_output = result.output_dir.parent / f"{result.output_dir.name}_asm"
        asm_output.mkdir(parents=True, exist_ok=True)

        # Create splits
        random.shuffle(asm_samples)
        train_split = int(len(asm_samples) * 0.8)
        val_split = int(len(asm_samples) * 0.9)

        splits = {
            "train": asm_samples[:train_split],
            "val": asm_samples[train_split:val_split],
            "test": asm_samples[val_split:],
        }

        asm_paths = exporter.export_for_model(
            samples=asm_samples,
            model="qwen2.5-coder-14b",
            output_dir=asm_output,
            splits=splits,
        )
        output_paths["asm"] = asm_output
        logger.info(f"✓ ASM dataset exported to: {asm_output}")

    # Export YAZE dataset
    if export_yaze_dataset and yaze_samples:
        logger.info(f"Exporting YAZE Tool dataset ({len(yaze_samples)} samples)...")
        yaze_output = result.output_dir.parent / f"{result.output_dir.name}_yaze"
        yaze_output.mkdir(parents=True, exist_ok=True)

        # Create splits
        random.shuffle(yaze_samples)
        train_split = int(len(yaze_samples) * 0.8)
        val_split = int(len(yaze_samples) * 0.9)

        splits = {
            "train": yaze_samples[:train_split],
            "val": yaze_samples[train_split:val_split],
            "test": yaze_samples[val_split:],
        }

        yaze_paths = exporter.export_for_model(
            samples=yaze_samples,
            model="qwen2.5-coder-14b",
            output_dir=yaze_output,
            splits=splits,
        )
        output_paths["yaze"] = yaze_output
        logger.info(f"✓ YAZE dataset exported to: {yaze_output}")

    return output_paths


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Training data generation campaign")
    parser.add_argument(
        "--target",
        type=int,
        default=34500,
        help="Target number of samples (default: 34500)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        help="Custom output directory name",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot mode (max 1K samples for validation)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints if available",
    )
    parser.add_argument(
        "--no-active-learning",
        action="store_true",
        help="Disable coverage-driven active learning",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help="Base quality threshold (default: None, uses domain-specific thresholds)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export datasets after generation",
    )

    args = parser.parse_args()

    # Run generation
    result = await run_generation_campaign(
        target_count=args.target,
        output_name=args.output_name,
        pilot=args.pilot,
        resume=args.resume,
        enable_active_learning=not args.no_active_learning,
        quality_threshold=args.quality_threshold,
    )

    # Export if requested
    if args.export:
        export_paths = await export_datasets(result)
        logger.info("Exported datasets:")
        for name, path in export_paths.items():
            logger.info(f"  {name}: {path}")

    logger.info("Campaign complete!")
    return result


if __name__ == "__main__":
    import random

    asyncio.run(main())
