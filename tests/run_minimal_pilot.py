#!/usr/bin/env python3
"""MINIMAL pilot - 200 samples, NO knowledge bases.

Uses only Gigaleak + Errors (no ASM, YAZE, or Oracle).
No KB loading, pure speed test.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def run_minimal_pilot():
    """Minimal 200-sample pilot without ANY knowledge base loading."""
    from agents.training.curator import DataCurator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.parallel_generator import generate_batch_parallel

    print("=" * 80)
    print("MINIMAL PILOT - 200 SAMPLES (NO KNOWLEDGE BASES)")
    print("=" * 80)
    print("\nOptimizations:")
    print("  • 10x concurrent generation on Gemini Flash")
    print("  • 2 generators ONLY (Gigaleak + Errors)")
    print("  • NO knowledge base loading")
    print("  • Domain-specific quality thresholds")
    print()

    # Create curator
    print("[1] Setting up DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Register ONLY generators that don't need knowledge bases
    generators = []

    # Gigaleak
    print("  Setting up GigaleakDataGenerator...")
    try:
        gigaleak_gen = GigaleakDataGenerator()
        await gigaleak_gen.setup()
        curator.register_generator("gigaleak", gigaleak_gen)
        generators.append(("gigaleak", gigaleak_gen, 100))
    except Exception as e:
        print(f"  ⚠️  Gigaleak failed: {e}")

    # Errors
    print("  Setting up ErrorSampleGenerator...")
    try:
        error_gen = ErrorSampleGenerator()
        await error_gen.setup()
        curator.register_generator("errors", error_gen)
        generators.append(("errors", error_gen, 100))
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
    print("\n[3] Generating 200 samples...")
    print("  ETA: 2-3 minutes")
    print()

    start_time = time.time()

    result = await curator.curate_dataset(
        domains=[d for d, _, _ in generators],
        target_count=200,
        quality_threshold=None,  # Use domain-specific
        balance_domains=True,
        output_name="minimal_pilot_200",
        resume=False,
    )

    duration = time.time() - start_time

    # Results
    print("\n" + "=" * 80)
    print("MINIMAL PILOT RESULTS")
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
            print("  ❌ REGRESSION!")
        elif pass_rate < 30:
            print(f"  ⚠️  LOW")
        elif pass_rate < 60:
            print(f"  ✓ ACCEPTABLE")
        else:
            print(f"  ✓✓ GOOD")

    print(f"\nDomain breakdown:")
    for domain, count in stats.domain_counts.items():
        pct = (count / stats.final_count * 100) if stats.final_count > 0 else 0
        print(f"  {domain}: {count} ({pct:.1f}%)")

    if result.output_dir:
        print(f"\nOutput: {result.output_dir}")

    samples_per_min = stats.final_count / (duration / 60)
    print(f"\nThroughput: {samples_per_min:.1f} samples/min")

    success = stats.final_count >= 100 and pass_rate > 0

    print("\n" + "=" * 80)
    if success:
        print("✓ MINIMAL PILOT PASSED!")
    else:
        print("❌ NEEDS REVIEW")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = asyncio.run(run_minimal_pilot())
    sys.exit(0 if success else 1)
