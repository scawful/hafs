#!/usr/bin/env python3
"""Alpha pilot test: Generate 20 samples with fixed quality pipeline.

Validates that quality fixes work at scale before running full pilot.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def run_alpha_pilot():
    """Generate 20 samples across all domains and validate quality."""
    from agents.training.curator import DataCurator
    from agents.training.generators.asm_generator import AsmDataGenerator

    print("=" * 80)
    print("ALPHA PILOT TEST - 20 SAMPLES")
    print("=" * 80)

    # Create curator
    print("\n[1] Setting up DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Register ASM generator only (fastest)
    print("\n[2] Registering ASM generator...")
    asm_gen = AsmDataGenerator()
    await asm_gen.setup()
    curator.register_generator("asm", asm_gen)
    print(f"   ✓ ASM generator registered")

    # Run generation with quality filtering
    print("\n[3] Generating 20 samples (with quality validation)...")
    print("   This will take ~2-3 minutes...")

    result = await curator.curate_dataset(
        domains=["asm"],
        target_count=20,
        quality_threshold=None,  # Use domain-specific (0.4 for ASM)
        balance_domains=False,
        output_name="alpha_pilot_20",
        resume=False,
    )

    # Display results
    print("\n" + "=" * 80)
    print("ALPHA PILOT RESULTS")
    print("=" * 80)

    stats = result.stats
    print(f"\nGeneration:")
    print(f"  Total generated: {stats.total_generated}")
    print(f"  Passed quality: {stats.passed_quality}")
    print(f"  Deduplicated: {stats.deduplicated}")
    print(f"  Final count: {stats.final_count}")

    if stats.total_generated > 0:
        pass_rate = (stats.passed_quality / stats.total_generated) * 100
        print(f"\nQuality pass rate: {pass_rate:.1f}%")

        if pass_rate == 0:
            print("  ❌ REGRESSION: 0% pass rate!")
            return False
        elif pass_rate < 30:
            print(f"  ⚠️  WARNING: Low pass rate ({pass_rate:.1f}%)")
        elif pass_rate < 60:
            print(f"  ✓ ACCEPTABLE: {pass_rate:.1f}% pass rate")
        else:
            print(f"  ✓✓ GOOD: {pass_rate:.1f}% pass rate")

    print(f"\nDomain breakdown:")
    for domain, count in stats.domain_counts.items():
        print(f"  {domain}: {count} samples")

    print(f"\nQuality scores:")
    for domain, score in stats.quality_scores.items():
        print(f"  {domain}: {score:.3f}")

    print(f"\nDataset splits:")
    print(f"  Train: {len(result.splits.train)}")
    print(f"  Val: {len(result.splits.val)}")
    print(f"  Test: {len(result.splits.test)}")

    if result.output_dir:
        print(f"\nOutput: {result.output_dir}")

    # Success criteria
    success = (
        stats.final_count >= 10 and  # At least 10 samples
        pass_rate > 0  # Non-zero pass rate
    )

    print("\n" + "=" * 80)
    if success:
        print("✓ ALPHA PILOT PASSED - Quality fixes validated!")
        print("Ready to proceed with full pilot (1000 samples)")
    else:
        print("❌ ALPHA PILOT FAILED - More debugging needed")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = asyncio.run(run_alpha_pilot())
    sys.exit(0 if success else 1)
