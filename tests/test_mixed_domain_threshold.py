#!/usr/bin/env python3
"""Test mixed-domain quality threshold fix."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


async def test_mixed_domain():
    """Test that mixed domains use per-sample thresholds."""
    from agents.training.base import TrainingSample
    from agents.training.quality import QualityPipeline

    print("=" * 80)
    print("MIXED-DOMAIN THRESHOLD TEST")
    print("=" * 80)

    # Setup pipeline
    print("\n[1] Setting up QualityPipeline...")
    pipeline = QualityPipeline()
    await pipeline.setup()
    print("✓ Pipeline ready")

    # Create mock samples with different domains and scores
    print("\n[2] Creating mixed-domain samples...")
    samples = [
        TrainingSample(
            instruction="Test gigaleak 1",
            input="",
            output="Output with code patterns: LDA #$80, STA $2100",
            domain="gigaleak",
            source="test",
        ),
        TrainingSample(
            instruction="Test gigaleak 2",
            input="",
            output="Output with code patterns: JMP $8000",
            domain="gigaleak",
            source="test",
        ),
        TrainingSample(
            instruction="Test error 1",
            input="",
            output="Error occurred: timeout",
            domain="errors",
            source="test",
        ),
        TrainingSample(
            instruction="Test error 2",
            input="",
            output="Error: rate limit exceeded",
            domain="errors",
            source="test",
        ),
    ]
    print(f"✓ Created {len(samples)} samples (2 gigaleak, 2 errors)")

    # Filter with None threshold (should use domain-specific)
    print("\n[3] Filtering with domain-specific thresholds...")
    filtered = await pipeline.filter_samples(
        samples,
        min_quality=None,  # Use domain-specific
        deduplicate=False,
    )

    print(f"\n✓ Results:")
    print(f"  Input samples: {len(samples)}")
    print(f"  Filtered samples: {len(filtered)}")

    if pipeline.last_filter_stats:
        stats = pipeline.last_filter_stats
        print(f"  Rejected (validation): {stats.rejected_validation}")
        print(f"  Rejected (quality): {stats.rejected_quality}")
        print(f"  Rejected (duplicates): {stats.rejected_duplicates}")

        if len(samples) > 0:
            pass_rate = (len(filtered) / len(samples)) * 100
            print(f"\n  Pass rate: {pass_rate:.1f}%")

            if pass_rate > 50:
                print("\n✓✓ SUCCESS - Domain-specific thresholds working!")
                return True
            else:
                print("\n❌ FAIL - Low pass rate")
                return False

    return len(filtered) > 0


if __name__ == "__main__":
    success = asyncio.run(test_mixed_domain())
    sys.exit(0 if success else 1)
