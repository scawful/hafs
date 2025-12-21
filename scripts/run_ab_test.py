#!/usr/bin/env python3
"""Run A/B test comparing baseline vs enhanced prompts.

Usage:
    # Quick test (100 samples)
    python scripts/run_ab_test.py --domain asm --samples 100

    # Full test (1000 samples)
    python scripts/run_ab_test.py --domain asm --samples 1000

    # Oracle test
    python scripts/run_ab_test.py --domain oracle --samples 500
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.training.ab_testing import ABTestRunner, PromptVersion


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


async def run_asm_test(num_samples: int, source_limit: int | None = None):
    """Run A/B test for ASM generator."""
    from agents.training.generators.asm_generator import AsmDataGenerator

    logger.info("Starting ASM A/B test")

    runner = ABTestRunner()

    baseline = PromptVersion(
        name="baseline_asm",
        generator_cls=AsmDataGenerator,
        use_enhanced=False,
    )

    enhanced = PromptVersion(
        name="enhanced_asm_v1",
        generator_cls=AsmDataGenerator,
        use_enhanced=True,
    )

    comparison = await runner.run_test(
        versions=[baseline, enhanced],
        num_samples=num_samples,
        source_limit=source_limit,
        domain="asm",
    )

    return comparison


async def run_oracle_test(num_samples: int, source_limit: int | None = None):
    """Run A/B test for Oracle generator."""
    from agents.training.generators.oracle_generator import OracleDataGenerator

    logger.info("Starting Oracle A/B test")

    runner = ABTestRunner()

    baseline = PromptVersion(
        name="baseline_oracle",
        generator_cls=OracleDataGenerator,
        use_enhanced=False,
    )

    enhanced = PromptVersion(
        name="enhanced_oracle_v1",
        generator_cls=OracleDataGenerator,
        use_enhanced=True,
    )

    comparison = await runner.run_test(
        versions=[baseline, enhanced],
        num_samples=num_samples,
        source_limit=source_limit,
        domain="oracle",
    )

    return comparison


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run A/B test comparing baseline vs enhanced prompts"
    )
    parser.add_argument(
        "--domain",
        choices=["asm", "oracle"],
        required=True,
        help="Domain to test (asm or oracle)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Target number of samples to generate per version (default: 100)",
    )
    parser.add_argument(
        "--source-limit",
        type=int,
        default=None,
        help="Limit source items (for faster testing, default: no limit)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (50 samples, 25 source items)",
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.samples = 50
        args.source_limit = 25
        logger.info("Quick mode: 50 samples, 25 source items")

    # Run appropriate test
    if args.domain == "asm":
        comparison = asyncio.run(run_asm_test(args.samples, args.source_limit))
    elif args.domain == "oracle":
        comparison = asyncio.run(run_oracle_test(args.samples, args.source_limit))
    else:
        logger.error(f"Unknown domain: {args.domain}")
        return 1

    # Print summary
    print("\n" + "="*80)
    print("A/B TEST COMPLETE")
    print("="*80)
    print(f"\nTest ID: {comparison.test_id}")
    print(f"Domain: {comparison.domain}")
    print(f"\nWinner: {comparison.winner.upper()}")
    print(f"Recommendation: {comparison.recommendation}")
    print(f"\nPass Rate Improvement: {comparison.pass_rate_improvement:+.1%}")
    print(f"Quality Score Improvement: {comparison.quality_improvement:+.3f}")
    print("\n" + "="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
