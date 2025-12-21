#!/usr/bin/env python3
"""FAST pilot - 1000 samples, skip slow Oracle KB.

Uses only: ASM, Gigaleak, YAZE, Errors (no Oracle)
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def run_fast_pilot():
    """Fast 1000-sample pilot without Oracle generator."""
    from agents.training.curator import DataCurator
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.cpp_generator import CppDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.parallel_generator import generate_batch_parallel

    print("=" * 80)
    print("FAST PILOT - 1000 SAMPLES (NO ORACLE)")
    print("=" * 80)
    print("\nOptimizations:")
    print("  • 10x concurrent generation on Gemini Flash")
    print("  • 4 generators (skip slow Oracle KB)")
    print("  • Domain-specific quality thresholds")
    print()

    # Create curator
    print("[1] Setting up DataCurator...")
    curator = DataCurator()
    await curator.setup()

    # Register FAST generators only (no Oracle - it's slow)
    generators = []

    # ASM
    print("  Setting up AsmDataGenerator...")
    asm_gen = AsmDataGenerator()
    await asm_gen.setup()
    curator.register_generator("asm", asm_gen)
    generators.append(("asm", asm_gen, 435))

    # Gigaleak
    print("  Setting up GigaleakDataGenerator...")
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    curator.register_generator("gigaleak", gigaleak_gen)
    generators.append(("gigaleak", gigaleak_gen, 232))

    # YAZE/C++
    print("  Setting up CppDataGenerator...")
    try:
        cpp_gen = CppDataGenerator()
        await cpp_gen.setup()
        curator.register_generator("yaze", cpp_gen)
        generators.append(("yaze", cpp_gen, 200))
    except Exception as e:
        print(f"  ⚠️  YAZE failed: {e}")

    # Errors
    print("  Setting up ErrorSampleGenerator...")
    try:
        error_gen = ErrorSampleGenerator()
        await error_gen.setup()
        curator.register_generator("errors", error_gen)
        generators.append(("errors", error_gen, 133))
    except Exception as e:
        print(f"  ⚠️  Error generator failed: {e}")

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
    print("\n[3] Generating 1000 samples...")
    print("  ETA: 8-10 minutes")
    print()

    start_time = time.time()

    result = await curator.curate_dataset(
        domains=[d for d, _, _ in generators],
        target_count=1000,
        quality_threshold=None,  # Use domain-specific
        balance_domains=True,
        output_name="fast_pilot_1000",
        resume=False,
    )

    duration = time.time() - start_time

    # Results
    print("\n" + "=" * 80)
    print("FAST PILOT RESULTS")
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

    success = stats.final_count >= 500 and pass_rate > 0

    print("\n" + "=" * 80)
    if success:
        print("✓ FAST PILOT PASSED!")
    else:
        print("❌ NEEDS REVIEW")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = asyncio.run(run_fast_pilot())
    sys.exit(0 if success else 1)
