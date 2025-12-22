#!/usr/bin/env python3
"""Debug script to test quality validation pipeline.

Tests why samples are failing quality validation.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_single_sample():
    """Generate ONE sample and test quality scoring."""
    from agents.training.generators.asm_generator import AsmDataGenerator
    from agents.training.quality import QualityPipeline

    print("=" * 80)
    print("QUALITY VALIDATION DEBUG TEST")
    print("=" * 80)

    # Create generator
    print("\n[1] Setting up ASM generator...")
    gen = AsmDataGenerator()
    await gen.setup()
    print(f"   ✓ Generator ready, domain: {gen.domain}")

    # Generate ONE sample
    print("\n[2] Generating ONE sample...")
    result = await gen.run_generation(limit=1)

    print(f"   Generation result:")
    print(f"     Processed: {result.processed}")
    print(f"     Skipped: {result.skipped}")
    print(f"     Errors: {result.errors}")
    print(f"     Samples: {len(result.samples)}")

    if not result.samples:
        print("   ✗ No samples generated!")
        print("\n   Debugging: trying direct generation...")

        # Try generating directly from one item
        items = await gen.extract_source_items()
        if items:
            print(f"   Found {len(items)} source items")
            print(f"   Trying to generate from first item: {items[0].name}")
            sample = await gen.generate_sample(items[0])
            if sample:
                print(f"   ✓ Direct generation succeeded!")
                result.samples = [sample]
            else:
                print(f"   ✗ Direct generation also failed!")
                return
        else:
            print(f"   ✗ No source items found!")
            return

    sample = result.samples[0]
    print(f"   ✓ Sample generated")
    print(f"     Instruction: {sample.instruction[:80]}...")
    print(f"     Output length: {len(sample.output)} chars")
    print(f"     Domain: {sample.domain}")
    print(f"     KG entities: {sample.kg_entities}")

    # Create quality pipeline
    print("\n[3] Setting up quality pipeline...")
    pipeline = QualityPipeline()
    await pipeline.setup()

    # Check what's available
    print(f"   Embedding index: {pipeline.embedding_index is not None}")
    print(f"   KG agent: {pipeline.kg_agent is not None}")
    print(f"   Orchestrator: {pipeline.orchestrator is not None}")
    print(f"   Validators: {len(pipeline._validators)}")

    # Run validation
    print("\n[4] Running validation...")
    is_valid, details = await pipeline.validate(sample)
    print(f"   Validation result: {is_valid}")
    if details:
        print(f"   Details: {details}")

    # Compute quality score
    print("\n[5] Computing quality score...")
    score = await pipeline.score(sample)
    print(f"   Diversity: {score.diversity_score:.3f}")
    print(f"   KG consistency: {score.kg_consistency:.3f}")
    print(f"   Hallucination risk: {score.hallucination_risk:.3f} (lower is better)")
    print(f"   Coherence: {score.semantic_coherence:.3f}")
    print(f"   OVERALL: {score.overall:.3f}")

    # Check against threshold
    threshold = 0.7
    print(f"\n[6] Quality check (threshold: {threshold})...")
    if score.overall >= threshold:
        print(f"   ✓ PASS (score {score.overall:.3f} >= {threshold})")
    else:
        print(f"   ✗ FAIL (score {score.overall:.3f} < {threshold})")

        # Analyze why
        print("\n   Failure analysis:")
        if score.diversity_score < 0.3:
            print(f"     - Low diversity: {score.diversity_score:.3f}")
        if score.kg_consistency < 0.5:
            print(f"     - Low KG consistency: {score.kg_consistency:.3f}")
        if score.hallucination_risk > 0.5:
            print(f"     - High hallucination risk: {score.hallucination_risk:.3f}")
        if score.semantic_coherence < 0.4:
            print(f"     - Low coherence: {score.semantic_coherence:.3f}")

    # Test filter_samples
    print("\n[7] Testing filter_samples()...")
    filtered = await pipeline.filter_samples([sample], min_quality=threshold)
    print(f"   Input: 1 sample")
    print(f"   Output: {len(filtered)} samples")

    if len(filtered) == 0:
        print(f"   ✗ Sample was rejected!")
        if pipeline.last_filter_stats:
            stats = pipeline.last_filter_stats
            print(f"     Total: {stats.total}")
            print(f"     Accepted: {stats.accepted}")
            print(f"     Rejected (validation): {stats.rejected_validation}")
            print(f"     Rejected (quality): {stats.rejected_quality}")
            print(f"     Rejected (duplicates): {stats.rejected_duplicates}")
    else:
        print(f"   ✓ Sample passed!")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_single_sample())
