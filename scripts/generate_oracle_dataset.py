#!/usr/bin/env python3
"""Generate Oracle-Farore training dataset with correct generators.

Uses:
- OracleDataGenerator: Oracle-of-Secrets ROM hack routines, secrets system
- AsmDataGenerator: 65816 assembly with Oracle hooks and modifications
- GigaleakDataGenerator: Nintendo vanilla code for comparison

Target: 1000+ high-quality samples focused on Oracle ROM hacking.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.training.curator import DataCurator
from agents.training.generators.oracle_generator import OracleDataGenerator
from agents.training.generators.asm_generator import AsmDataGenerator
from agents.training.generators.gigaleak_generator import GigaleakDataGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Generate Oracle-focused dataset."""
    logger.info("=" * 80)
    logger.info("ORACLE-FARORE DATASET GENERATION")
    logger.info("=" * 80)
    logger.info("Target: 1000 samples")
    logger.info("Generators: Oracle + ASM (balanced) + cross-domain hooks")
    logger.info("Quality threshold: per-domain defaults")
    logger.info("=" * 80)

    # Initialize curator
    curator = DataCurator()
    await curator.setup()

    # Register Oracle generator (primary)
    logger.info("\nRegistering OracleDataGenerator...")
    oracle_gen = OracleDataGenerator(use_enhanced_prompts=True)
    await oracle_gen.setup()
    curator.register_generator("oracle", oracle_gen)
    logger.info("✓ Oracle generator (secrets system, ROM hack routines)")

    # Register ASM generator (secondary)
    logger.info("Registering AsmDataGenerator...")
    asm_gen = AsmDataGenerator(use_enhanced_prompts=True)
    await asm_gen.setup()
    curator.register_generator("asm", asm_gen)
    logger.info("✓ ASM generator (65816 with Oracle hooks)")

    # Register Gigaleak generator (cross-domain comparison only)
    logger.info("Registering GigaleakDataGenerator (cross-domain only)...")
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    curator.register_generator("gigaleak", gigaleak_gen)
    logger.info("✓ Gigaleak generator (production commentary for pairing)")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"oracle_farore_fixed_{timestamp}"

    logger.info(f"\nOutput: {output_name}")
    logger.info("Starting generation...\n")

    # Run curation with domain targets
    # Oracle: 500, ASM: 500, Cross-domain: 150
    result = await curator.curate_dataset(
        domains=["oracle", "asm"],
        target_count=1000,
        quality_threshold=None,
        balance_domains=True,
        output_name=output_name,
        resume=False,
        cross_domain_samples=150,
    )

    logger.info("\n" + "=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total generated: {result.stats.total_generated}")
    logger.info(f"Passed quality: {result.stats.passed_quality}")
    logger.info(f"Final count: {result.stats.final_count}")
    logger.info(f"Acceptance rate: {result.stats.passed_quality / max(result.stats.total_generated, 1) * 100:.1f}%")
    logger.info("\nDomain breakdown:")
    for domain, count in result.stats.domain_counts.items():
        logger.info(f"  {domain}: {count}")
    logger.info("\nQuality scores:")
    for domain, score in result.stats.quality_scores.items():
        logger.info(f"  {domain}: {score:.3f}")
    logger.info("=" * 80)

    logger.info(f"\nDataset saved to: {result.output_dir}")
    logger.info(f"Update config path to: {result.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
