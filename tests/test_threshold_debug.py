#!/usr/bin/env python3
"""Quick test to debug threshold logic with print statements."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def test_threshold_debug():
    """Generate 10 samples and check threshold logic."""
    from agents.training.curator import DataCurator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator

    print("=" * 80)
    print("THRESHOLD DEBUG TEST - 10 Gigaleak Samples")
    print("=" * 80)

    # Setup
    print("\n[1] Setting up...")
    curator = DataCurator()
    await curator.setup()

    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    curator.register_generator("gigaleak", gigaleak_gen)
    print("✓ Setup complete")

    # Generate 10 samples
    print("\n[2] Generating 10 samples...")
    result = await curator.curate_dataset(
        domains=["gigaleak"],
        target_count=10,
        quality_threshold=None,  # Should trigger per-sample thresholds
        balance_domains=False,
        output_name=None,
        resume=False,
    )

    # Results
    stats = result.stats
    print(f"\n[3] Results:")
    print(f"  Generated: {stats.total_generated}")
    print(f"  Passed: {stats.passed_quality}")
    print(f"  Final: {stats.final_count}")

    if stats.total_generated > 0:
        pass_rate = (stats.passed_quality / stats.total_generated) * 100
        print(f"  Pass rate: {pass_rate:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_threshold_debug())
