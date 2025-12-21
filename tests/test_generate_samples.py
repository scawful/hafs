#!/usr/bin/env python3
"""Test actual sample generation from Gigaleak + Errors."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def test_generation():
    """Test generating a few samples."""
    from agents.training.curator import DataCurator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator

    print("=" * 80)
    print("SAMPLE GENERATION TEST - 10 samples")
    print("=" * 80)

    # Setup curator
    print("\nStep 1: Setup curator...")
    curator = DataCurator()
    await curator.setup()
    print("✓ Curator ready")

    # Setup Gigaleak generator
    print("\nStep 2: Setup Gigaleak generator...")
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    curator.register_generator("gigaleak", gigaleak_gen)
    print("✓ Gigaleak registered")

    # Setup Error generator
    print("\nStep 3: Setup Error generator...")
    error_gen = ErrorSampleGenerator()
    await error_gen.setup()
    curator.register_generator("errors", error_gen)
    print("✓ Error generator registered")

    # Generate small batch
    print("\nStep 4: Generate 10 samples...")
    result = await curator.curate_dataset(
        domains=["gigaleak", "errors"],
        target_count=10,
        quality_threshold=None,  # Use domain-specific
        balance_domains=True,
        output_name=None,  # Don't save
        resume=False,
    )

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    stats = result.stats
    print(f"Total generated: {stats.total_generated}")
    print(f"Passed quality: {stats.passed_quality}")
    print(f"Final count: {stats.final_count}")

    if stats.total_generated > 0:
        pass_rate = (stats.passed_quality / stats.total_generated) * 100
        print(f"Quality pass rate: {pass_rate:.1f}%")

        if pass_rate > 0:
            print("\n✓✓ SUCCESS - Generators are working!")
            return True
        else:
            print("\n❌ FAIL - 0% pass rate")
            return False
    else:
        print("\n❌ FAIL - No samples generated")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_generation())
    sys.exit(0 if success else 1)
