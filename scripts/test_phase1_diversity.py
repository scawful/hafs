#!/usr/bin/env python3
"""Test Phase 1 diversity improvements.

Runs a small-scale generation campaign using:
1. Prompt variation templates
2. New resource discovery (zelda3, documentation)
3. Synthetic augmentation

Measures diversity improvements and rejection rates.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.training.curator import DataCurator
from agents.training.generators.zelda3_generator import Zelda3DisasmGenerator
from agents.training.generators.documentation_generator import DocumentationGenerator
from agents.training.generators.asm_generator import AsmDataGenerator


async def main():
    """Run Phase 1 diversity test campaign."""
    print("=" * 80)
    print("Phase 1 Diversity Improvements - Test Campaign")
    print("=" * 80)
    print()

    # Initialize curator
    curator = DataCurator()
    await curator.setup()

    # Register generators with new features
    print("Registering generators with Phase 1 improvements...")
    print()

    # 1. Zelda3 generator (NEW - resource discovery)
    print("✓ Zelda3DisasmGenerator (vanilla disassembly - NEW)")
    zelda3_gen = Zelda3DisasmGenerator(use_template_variation=True)
    await zelda3_gen.setup()
    curator.register_generator("zelda3", zelda3_gen)

    # 2. Documentation generator (NEW - resource discovery)
    print("✓ DocumentationGenerator (ROM hacking docs - NEW)")
    doc_gen = DocumentationGenerator(use_template_variation=True)
    await doc_gen.setup()
    curator.register_generator("documentation", doc_gen)

    # 3. ASM generator (ENHANCED - prompt variation)
    print("✓ AsmDataGenerator (with prompt variation - ENHANCED)")
    asm_gen = AsmDataGenerator(use_template_variation=True)
    await asm_gen.setup()
    curator.register_generator("asm", asm_gen)

    print()
    print("=" * 80)
    print("Phase 1 Features Enabled:")
    print("=" * 80)
    print("1. Prompt Variation: 15-20 templates per domain")
    print("2. Resource Discovery: 1,818 files indexed")
    print("3. Synthetic Augmentation: 3x multiplier for quality >= 0.6")
    print()

    # Run test campaign
    print("=" * 80)
    print("Running Test Campaign")
    print("=" * 80)
    print()
    print("Target: 90 samples (30 per domain)")
    print("This will test all Phase 1 improvements in action...")
    print()

    result = await curator.curate_dataset(
        domains=["zelda3", "documentation", "asm"],
        target_count=90,
        balance_domains=True,
        output_name="phase1_diversity_test",
        resume=False,
    )

    # Display results
    print()
    print("=" * 80)
    print("Test Campaign Results")
    print("=" * 80)
    print()

    stats = result.stats

    print(f"Total Generated:     {stats.total_generated}")
    print(f"Passed Quality:      {stats.passed_quality}")
    print(f"Deduplicated:        {stats.deduplicated}")
    print(f"Augmented (NEW):     {stats.augmented}")
    print(f"Final Count:         {stats.final_count}")
    print()

    print(f"Average Quality:     {stats.quality_scores.get('average', 0):.3f}")
    print(f"Duration:            {stats.duration_seconds:.1f}s")
    print()

    # Calculate metrics
    if stats.total_generated > 0:
        acceptance_rate = (stats.passed_quality / stats.total_generated) * 100
        rejection_rate = 100 - acceptance_rate
        augmentation_multiplier = (stats.augmented / stats.passed_quality) if stats.passed_quality > 0 else 0

        print("=" * 80)
        print("Diversity Metrics")
        print("=" * 80)
        print()
        print(f"Acceptance Rate:     {acceptance_rate:.1f}% (target: >70%)")
        print(f"Rejection Rate:      {rejection_rate:.1f}% (baseline: 85%, target: <20%)")
        print(f"Augmentation Gain:   {augmentation_multiplier:.1f}x (expected: ~3x)")
        print()

        # Assess improvement
        if rejection_rate < 20:
            status = "✓ EXCELLENT - Target achieved!"
        elif rejection_rate < 50:
            status = "✓ GOOD - Significant improvement"
        elif rejection_rate < 85:
            status = "⚠ MODERATE - Some improvement"
        else:
            status = "✗ NEEDS WORK - No improvement"

        print(f"Status: {status}")
        print()

    # Domain breakdown
    if stats.domain_counts:
        print("=" * 80)
        print("Domain Breakdown")
        print("=" * 80)
        print()
        for domain, count in sorted(stats.domain_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {domain:20s}: {count:4d} samples")
        print()

    # Output location
    if result.output_dir:
        print("=" * 80)
        print("Output Files")
        print("=" * 80)
        print()
        print(f"Dataset:             {result.output_dir}")
        print(f"  - train.jsonl:     {len(result.splits.train)} samples")
        print(f"  - val.jsonl:       {len(result.splits.val)} samples")
        print(f"  - test.jsonl:      {len(result.splits.test)} samples")
        print(f"  - stats.json:      Campaign statistics")
        print(f"  - rejected.jsonl:  Rejected samples for analysis")
        print()

        # Check for rejection summary
        rejection_summary = result.output_dir / "rejection_summary.json"
        if rejection_summary.exists():
            with open(rejection_summary) as f:
                rej_data = json.load(f)

            print("=" * 80)
            print("Rejection Analysis")
            print("=" * 80)
            print()
            print(f"Total Rejected:      {rej_data.get('total_rejected', 0)}")
            print()
            print("By Reason:")
            for reason, count in sorted(rej_data.get('by_reason', {}).items(), key=lambda x: x[1], reverse=True):
                pct = (count / rej_data['total_rejected'] * 100) if rej_data['total_rejected'] > 0 else 0
                print(f"  {reason:30s}: {count:4d} ({pct:5.1f}%)")
            print()

    print("=" * 80)
    print("Next Steps")
    print("=" * 80)
    print()
    print("1. Review rejected.jsonl to understand remaining diversity issues")
    print("2. Check augmented samples quality (domain suffix '+augmented')")
    print("3. Inspect template variation effectiveness in output samples")
    print("4. Compare embeddings of original vs augmented samples")
    print()
    print("If results are good (rejection rate <20%), proceed to Phase 2!")
    print()


if __name__ == "__main__":
    asyncio.run(main())
