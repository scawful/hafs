#!/usr/bin/env python3
"""VALIDATION pilot - 100 samples to confirm threshold fix.

Uses Gigaleak + Errors with domain-specific thresholds.
Should achieve >50% pass rate with fixed quality pipeline.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def run_validation_pilot():
    """Validation pilot - 100 samples with mixed domains."""
    from agents.training.curator import DataCurator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.parallel_generator import generate_batch_parallel

    print("=" * 80)
    print("VALIDATION PILOT - 100 SAMPLES (THRESHOLD FIX)")
    print("=" * 80)
    print("\nExpected:")
    print("  • >50% pass rate (was 1% before fix)")
    print("  • Gigaleak threshold: 0.5")
    print("  • Errors threshold: 0.3")
    print("  • Per-sample domain-specific thresholds")
    print()

    # Create curator
    print("[1] Setting up DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Register generators
    generators = []

    # Gigaleak
    print("  Setting up GigaleakDataGenerator...")
    try:
        gigaleak_gen = GigaleakDataGenerator()
        await gigaleak_gen.setup()
        curator.register_generator("gigaleak", gigaleak_gen)
        generators.append(("gigaleak", gigaleak_gen, 50))
    except Exception as e:
        print(f"  ⚠️  Gigaleak failed: {e}")

    # Errors
    print("  Setting up ErrorSampleGenerator...")
    try:
        error_gen = ErrorSampleGenerator()
        await error_gen.setup()
        curator.register_generator("errors", error_gen)
        generators.append(("errors", error_gen, 50))
    except Exception as e:
        print(f"  ⚠️  Error generator failed: {e}")

    if not generators:
        print("\n❌ NO GENERATORS AVAILABLE")
        return False

    print(f"\n  ✓ {len(generators)} generators registered")

    # Patch for 10x parallelism
    print("\n[2] Patching for 10x parallel...")
    for domain, gen, _ in generators:
        async def parallel_wrapper(items, batch_size=50, progress_callback=None, _gen=gen):
            return await generate_batch_parallel(
                _gen, items, batch_size=batch_size, max_concurrent=10, progress_callback=progress_callback
            )
        gen.generate_batch = parallel_wrapper
    print("  ✓ All generators patched")

    # Run generation
    print("\n[3] Generating 100 samples...")
    print("  ETA: 2-3 minutes")
    print()

    start_time = time.time()

    result = await curator.curate_dataset(
        domains=[d for d, _, _ in generators],
        target_count=100,
        quality_threshold=None,  # Use domain-specific (THIS IS THE FIX)
        balance_domains=True,
        output_name="validation_pilot_100",
        resume=False,
    )

    duration = time.time() - start_time

    # Results
    print("\n" + "=" * 80)
    print("VALIDATION PILOT RESULTS")
    print("=" * 80)

    stats = result.stats
    print(f"\nGeneration:")
    print(f"  Total generated: {stats.total_generated}")
    print(f"  Passed quality: {stats.passed_quality}")
    print(f"  Deduplicated: {stats.deduplicated}")
    print(f"  Final count: {stats.final_count}")
    print(f"  Duration: {duration / 60:.1f} minutes")

    if stats.total_generated > 0:
        pass_rate = (stats.passed_quality / stats.total_generated) * 100
        print(f"\nQuality pass rate: {pass_rate:.1f}%")

        if pass_rate == 0:
            print("  ❌ REGRESSION - 0% pass rate!")
        elif pass_rate < 30:
            print(f"  ⚠️  LOW - Fix may not be working")
        elif pass_rate < 50:
            print(f"  ~ BORDERLINE - Needs investigation")
        elif pass_rate < 80:
            print(f"  ✓ GOOD - Fix is working")
        else:
            print(f"  ✓✓ EXCELLENT - Fix confirmed!")

    print(f"\nDomain breakdown:")
    for domain, count in stats.domain_counts.items():
        pct = (count / stats.final_count * 100) if stats.final_count > 0 else 0
        print(f"  {domain}: {count} ({pct:.1f}%)")

    if result.output_dir:
        print(f"\nOutput: {result.output_dir}")

    samples_per_min = stats.final_count / (duration / 60) if duration > 0 else 0
    print(f"\nThroughput: {samples_per_min:.1f} samples/min")

    # Success criteria: >50% pass rate
    success = stats.final_count >= 50 and pass_rate >= 50

    print("\n" + "=" * 80)
    if success:
        print("✓✓ VALIDATION PASSED - READY FOR FULL CAMPAIGN")
        print("\nThreshold fix confirmed working:")
        print("  • Mixed domains use per-sample thresholds")
        print("  • Pass rate >50% (vs 1% before fix)")
    else:
        print("❌ VALIDATION FAILED")
        if pass_rate < 50:
            print(f"  • Pass rate too low: {pass_rate:.1f}% (need >=50%)")
        if stats.final_count < 50:
            print(f"  • Sample count too low: {stats.final_count} (need >=50)")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = asyncio.run(run_validation_pilot())
    sys.exit(0 if success else 1)
