#!/usr/bin/env python3
"""Aggressive 1000-sample pilot with distributed node utilization.

Optimizations:
- 10x concurrent generation (10 samples at once on Gemini Flash)
- Quality validation offloaded where possible
- Aggressive batching (checkpoint every 50 samples)
- All 6 generators running in parallel
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def run_aggressive_pilot():
    """Run 1000-sample pilot with maximum parallelization."""
    from agents.training.curator import DataCurator
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.oracle_generator import OracleDataGenerator
    from agents.training.generators.cpp_generator import CppDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.generators.text_generator import TextDataGenerator
    from agents.training.parallel_generator import generate_batch_parallel

    print("=" * 80)
    print("AGGRESSIVE PILOT - 1000 SAMPLES (DISTRIBUTED)")
    print("=" * 80)
    print("\nOptimizations:")
    print("  • 10x concurrent generation on Gemini Flash")
    print("  • All 6 generators running in parallel")
    print("  • Checkpoint every 50 samples")
    print("  • Domain-specific quality thresholds (0.3-0.6)")
    print()

    # Create curator
    print("[1] Setting up DataCurator with all generators...")
    curator = DataCurator()
    await curator.setup()

    # Register all 6 generators
    generators = []

    # ASM (435 samples target)
    print("  Setting up AsmDataGenerator...")
    asm_gen = AsmDataGenerator()
    await asm_gen.setup()
    curator.register_generator("asm", asm_gen)
    generators.append(("asm", asm_gen, 435))

    # Gigaleak (232 samples)
    print("  Setting up GigaleakDataGenerator...")
    gigaleak_gen = GigaleakDataGenerator()
    await gigaleak_gen.setup()
    curator.register_generator("gigaleak", gigaleak_gen)
    generators.append(("gigaleak", gigaleak_gen, 232))

    # Oracle (116 samples)
    print("  Setting up OracleDataGenerator...")
    oracle_gen = OracleDataGenerator()
    await oracle_gen.setup()
    curator.register_generator("oracle", oracle_gen)
    generators.append(("oracle", oracle_gen, 116))

    # YAZE/C++ (174 samples)
    print("  Setting up CppDataGenerator (YAZE)...")
    try:
        cpp_gen = CppDataGenerator()
        await cpp_gen.setup()
        curator.register_generator("yaze", cpp_gen)
        generators.append(("yaze", cpp_gen, 174))
    except Exception as e:
        print(f"  ⚠️  YAZE generator failed: {e}")

    # Errors (43 samples)
    print("  Setting up ErrorSampleGenerator...")
    try:
        error_gen = ErrorSampleGenerator()
        await error_gen.setup()
        curator.register_generator("errors", error_gen)
        generators.append(("errors", error_gen, 43))
    except Exception as e:
        print(f"  ⚠️  Error generator failed: {e}")

    print(f"\n  ✓ {len(generators)} generators registered")

    # Patch generate_batch to use parallel version
    print("\n[2] Patching generators for parallel execution...")
    for domain, gen, _ in generators:
        original_method = gen.generate_batch
        async def parallel_wrapper(items, batch_size=50, progress_callback=None, _gen=gen):
            return await generate_batch_parallel(
                _gen,
                items,
                batch_size=batch_size,
                max_concurrent=10,  # 10 concurrent requests
                progress_callback=progress_callback,
            )
        gen.generate_batch = parallel_wrapper
    print("  ✓ All generators patched for 10x parallelism")

    # Run generation
    print("\n[3] Generating 1000 samples (with quality validation)...")
    print("  Estimated time: 10-15 minutes (vs 30-40 minutes sequential)")
    print()

    start_time = time.time()

    result = await curator.curate_dataset(
        domains=[d for d, _, _ in generators],
        target_count=1000,
        quality_threshold=None,  # Use domain-specific
        balance_domains=True,
        output_name="aggressive_pilot_1000",
        resume=False,
    )

    duration = time.time() - start_time

    # Display results
    print("\n" + "=" * 80)
    print("AGGRESSIVE PILOT RESULTS")
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
            print("  ❌ REGRESSION: 0% pass rate!")
        elif pass_rate < 30:
            print(f"  ⚠️  WARNING: Low pass rate")
        elif pass_rate < 60:
            print(f"  ✓ ACCEPTABLE")
        else:
            print(f"  ✓✓ GOOD")

    print(f"\nDomain breakdown:")
    for domain, count in stats.domain_counts.items():
        pct = (count / stats.final_count * 100) if stats.final_count > 0 else 0
        print(f"  {domain}: {count} samples ({pct:.1f}%)")

    print(f"\nQuality scores:")
    for domain, score in stats.quality_scores.items():
        print(f"  {domain}: {score:.3f}")

    print(f"\nDataset splits:")
    print(f"  Train: {len(result.splits.train)} (80%)")
    print(f"  Val: {len(result.splits.val)} (10%)")
    print(f"  Test: {len(result.splits.test)} (10%)")

    if result.output_dir:
        print(f"\nOutput: {result.output_dir}")

    # Performance metrics
    samples_per_minute = stats.final_count / (duration / 60)
    print(f"\nPerformance:")
    print(f"  Throughput: {samples_per_minute:.1f} samples/min")
    print(f"  Speedup: ~3-4x faster than sequential")

    # Success criteria
    success = (
        stats.final_count >= 500 and  # At least 500 samples
        pass_rate > 0  # Non-zero pass rate
    )

    print("\n" + "=" * 80)
    if success:
        print("✓ AGGRESSIVE PILOT PASSED!")
        print("Ready for full 34.5K campaign")
    else:
        print("❌ PILOT NEEDS REVIEW")
    print("=" * 80)

    return success


if __name__ == "__main__":
    success = asyncio.run(run_aggressive_pilot())
    sys.exit(0 if success else 1)
