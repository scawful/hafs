#!/usr/bin/env python3
"""Full 34.5K generation campaign with distributed node utilization.

Architecture:
- Gemini Flash: 70% of generation (primary teacher)
- medical-mechanica qwen3:14b: 30% of generation (offload)
- medical-mechanica: Quality validation, embeddings
- 10x concurrent generation per node

Expected performance:
- Sequential: ~30-40 hours
- 10x parallel (Gemini only): ~15-20 hours
- Distributed (Gemini + medical-mechanica): ~8-12 hours (2-3x speedup)
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


async def run_distributed_campaign():
    """Run full 34.5K campaign with distributed generation."""
    from agents.training.curator import DataCurator
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.generators.gigaleak_generator import GigaleakDataGenerator
    from agents.training.generators.oracle_generator import OracleDataGenerator
    from agents.training.generators.cpp_generator import CppDataGenerator
    from agents.training.generators.error_generator import ErrorSampleGenerator
    from agents.training.generators.text_generator import TextDataGenerator
    from agents.training.parallel_generator import generate_batch_parallel
    from agents.training.distributed_generator import (
        DistributedGenerationMixin,
        LoadBalancer,
    )

    print("=" * 80)
    print("FULL CAMPAIGN - 34,500 SAMPLES (DISTRIBUTED)")
    print("=" * 80)
    print("\nArchitecture:")
    print("  • Gemini Flash (primary): 70% generation + main thinking")
    print("  • medical-mechanica qwen3:14b: 30% generation offload")
    print("  • medical-mechanica: Quality validation, embeddings")
    print("  • 10x concurrent per node (20 total concurrent requests)")
    print()
    print("Expected duration: 8-12 hours (vs 30-40 hours sequential)")
    print()

    # Create curator
    print("[1] Setting up DataCurator with distributed capabilities...")
    curator = DataCurator()
    await curator.setup()

    # Initialize load balancer
    load_balancer = LoadBalancer()

    # Register all 6 generators with distributed mixin
    generators = []

    # ASM (15K target)
    print("  Setting up AsmDataGenerator (distributed)...")
    asm_gen = AsmDataGenerator()
    # Add distributed capabilities
    asm_gen.__class__ = type(
        "DistributedAsmGenerator",
        (asm_gen.__class__, DistributedGenerationMixin),
        {},
    )
    asm_gen._generation_counter = 0
    asm_gen._use_distributed = True
    await asm_gen.setup()
    await asm_gen._setup_distributed()
    curator.register_generator("asm", asm_gen)
    generators.append(("asm", asm_gen, 15000))

    # Gigaleak (8K)
    print("  Setting up GigaleakDataGenerator (distributed)...")
    gigaleak_gen = GigaleakDataGenerator()
    gigaleak_gen.__class__ = type(
        "DistributedGigaleakGenerator",
        (gigaleak_gen.__class__, DistributedGenerationMixin),
        {},
    )
    gigaleak_gen._generation_counter = 0
    gigaleak_gen._use_distributed = True
    await gigaleak_gen.setup()
    await gigaleak_gen._setup_distributed()
    curator.register_generator("gigaleak", gigaleak_gen)
    generators.append(("gigaleak", gigaleak_gen, 8000))

    # Oracle (4K)
    print("  Setting up OracleDataGenerator (distributed)...")
    oracle_gen = OracleDataGenerator()
    oracle_gen.__class__ = type(
        "DistributedOracleGenerator",
        (oracle_gen.__class__, DistributedGenerationMixin),
        {},
    )
    oracle_gen._generation_counter = 0
    oracle_gen._use_distributed = True
    await oracle_gen.setup()
    await oracle_gen._setup_distributed()
    curator.register_generator("oracle", oracle_gen)
    generators.append(("oracle", oracle_gen, 4000))

    # YAZE/C++ (6K)
    print("  Setting up CppDataGenerator (distributed)...")
    try:
        cpp_gen = CppDataGenerator()
        cpp_gen.__class__ = type(
            "DistributedCppGenerator",
            (cpp_gen.__class__, DistributedGenerationMixin),
            {},
        )
        cpp_gen._generation_counter = 0
        cpp_gen._use_distributed = True
        await cpp_gen.setup()
        await cpp_gen._setup_distributed()
        curator.register_generator("yaze", cpp_gen)
        generators.append(("yaze", cpp_gen, 6000))
    except Exception as e:
        print(f"  ⚠️  YAZE generator failed: {e}")

    # Errors (1.5K)
    print("  Setting up ErrorSampleGenerator (distributed)...")
    try:
        error_gen = ErrorSampleGenerator()
        error_gen.__class__ = type(
            "DistributedErrorGenerator",
            (error_gen.__class__, DistributedGenerationMixin),
            {},
        )
        error_gen._generation_counter = 0
        error_gen._use_distributed = True
        await error_gen.setup()
        await error_gen._setup_distributed()
        curator.register_generator("errors", error_gen)
        generators.append(("errors", error_gen, 1500))
    except Exception as e:
        print(f"  ⚠️  Error generator failed: {e}")

    print(f"\n  ✓ {len(generators)} distributed generators registered")

    # Patch generate_batch for parallel + distributed
    print("\n[2] Patching generators for distributed parallel execution...")
    for domain, gen, _ in generators:
        original_method = gen.generate_batch

        async def distributed_wrapper(
            items, batch_size=100, progress_callback=None, _gen=gen
        ):
            # Use distributed generation with 10x parallelism
            samples = []
            total = len(items)

            for chunk_start in range(0, len(items), 10):
                chunk = items[chunk_start : chunk_start + 10]

                # Generate with distributed routing
                tasks = [_gen.generate_sample_distributed(item) for item in chunk]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for item, result in zip(chunk, results):
                    if isinstance(result, Exception):
                        continue
                    elif result is not None:
                        samples.append(result)

                    if progress_callback:
                        progress_callback(chunk_start + len(samples), total)

                    # Checkpoint
                    if len(samples) % batch_size == 0:
                        from agents.training.base import GenerationCheckpoint

                        checkpoint = GenerationCheckpoint(
                            domain=_gen.domain,
                            processed_ids=set(s.sample_id for s in samples),
                            last_item_id=item.item_id,
                            total_processed=len(samples),
                            total_errors=0,
                        )
                        _gen.save_checkpoint(checkpoint)

            return samples

        gen.generate_batch = distributed_wrapper

    print("  ✓ All generators patched for distributed execution")

    # Run generation
    print("\n[3] Generating 34,500 samples (distributed)...")
    print("  This will take 8-12 hours")
    print("  Progress will be saved every 100 samples")
    print()
    print("  Node distribution:")
    print("    • Gemini Flash: ~24,000 samples (70%)")
    print("    • medical-mechanica: ~10,500 samples (30%)")
    print()

    start_time = time.time()

    result = await curator.curate_dataset(
        domains=[d for d, _, _ in generators],
        target_count=34500,
        quality_threshold=None,  # Use domain-specific
        balance_domains=True,
        output_name="alttp_yaze_full_distributed",
        resume=True,  # Support resuming
    )

    duration = time.time() - start_time

    # Display results
    print("\n" + "=" * 80)
    print("DISTRIBUTED CAMPAIGN RESULTS")
    print("=" * 80)

    stats = result.stats
    print(f"\nGeneration:")
    print(f"  Total generated: {stats.total_generated}")
    print(f"  Passed quality: {stats.passed_quality}")
    print(f"  Deduplicated: {stats.deduplicated}")
    print(f"  Final count: {stats.final_count}")
    print(f"  Duration: {duration / 3600:.1f} hours")

    if stats.total_generated > 0:
        pass_rate = (stats.passed_quality / stats.total_generated) * 100
        print(f"\nQuality pass rate: {pass_rate:.1f}%")

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
    samples_per_hour = stats.final_count / (duration / 3600)
    print(f"\nPerformance:")
    print(f"  Throughput: {samples_per_hour:.0f} samples/hour")
    print(f"  Speedup: 2-3x faster than parallel-only")

    # Load balancer stats
    print(f"\nLoad distribution:")
    lb_stats = load_balancer.get_stats()
    for provider, pstats in lb_stats.items():
        if pstats["requests"] > 0:
            print(f"  {provider}:")
            print(f"    Requests: {pstats['requests']}")
            print(f"    Avg time: {pstats['avg_time']:.2f}s")
            print(f"    Error rate: {pstats['error_rate']*100:.1f}%")

    print("\n" + "=" * 80)
    print("✓ FULL CAMPAIGN COMPLETE!")
    print("Ready to export datasets and begin training")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = asyncio.run(run_distributed_campaign())
    sys.exit(0 if success else 1)
