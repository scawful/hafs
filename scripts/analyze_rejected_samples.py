#!/usr/bin/env python3
"""
Analyze Rejected Training Samples

This script helps you understand why training samples are being rejected
by analyzing the rejection reasons, quality scores, and patterns.

Usage:
  python scripts/analyze_rejected_samples.py <dataset_dir>
  python scripts/analyze_rejected_samples.py --latest
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def analyze_rejected_samples(dataset_dir: Path):
    """Analyze rejected samples from a dataset directory."""
    print("=" * 80)
    print(f"Analyzing Rejected Samples: {dataset_dir.name}")
    print("=" * 80)
    print()

    rejected_file = dataset_dir / "rejected.jsonl"
    summary_file = dataset_dir / "rejection_summary.json"

    if not rejected_file.exists():
        print(f"✗ No rejected samples file found: {rejected_file}")
        print("\nThis dataset may have been generated before rejected sample tracking was added.")
        return False

    # Load summary if available
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)

        print("SUMMARY")
        print("-" * 80)
        print(f"Total rejected: {summary['total_rejected']}")
        print(f"Average quality score: {summary['avg_quality_score']:.3f}")
        print(f"Min quality score: {summary['min_quality_score']:.3f}")
        print(f"Max quality score: {summary['max_quality_score']:.3f}")
        print()

        print("REJECTION REASONS")
        print("-" * 80)
        for reason, count in sorted(summary['by_reason'].items(), key=lambda x: -x[1]):
            pct = (count / summary['total_rejected']) * 100
            print(f"  {reason:25s} {count:6d} ({pct:5.1f}%)")
        print()

        print("BY DOMAIN")
        print("-" * 80)
        for domain, count in sorted(summary['by_domain'].items(), key=lambda x: -x[1]):
            pct = (count / summary['total_rejected']) * 100
            print(f"  {domain:25s} {count:6d} ({pct:5.1f}%)")
        print()

    # Load and analyze individual samples
    rejected_samples = []
    with open(rejected_file) as f:
        for line in f:
            rejected_samples.append(json.loads(line))

    print(f"Loaded {len(rejected_samples)} rejected samples")
    print()

    # Find samples closest to passing threshold
    print("CLOSEST TO PASSING (Top 10)")
    print("-" * 80)

    samples_with_scores = [
        s for s in rejected_samples
        if s.get('quality_score') is not None
    ]

    if samples_with_scores:
        sorted_samples = sorted(
            samples_with_scores,
            key=lambda x: x.get('quality_score', 0),
            reverse=True
        )

        for i, sample in enumerate(sorted_samples[:10], 1):
            reason = sample.get('rejection_reason', 'unknown')
            score = sample.get('quality_score', 0)
            domain = sample.get('domain', 'unknown')

            print(f"{i}. [{domain}] Score: {score:.3f} | Reason: {reason}")

            # Show rejection details if available
            details = sample.get('rejection_details')
            if details:
                if 'threshold' in details:
                    print(f"   Threshold: {details['threshold']:.3f}")
                if 'diversity' in details:
                    print(f"   Diversity: {details['diversity']:.3f}")
                if 'kg_consistency' in details:
                    print(f"   KG Consistency: {details['kg_consistency']:.3f}")
                if 'hallucination_risk' in details:
                    print(f"   Hallucination Risk: {details['hallucination_risk']:.3f}")
                if 'coherence' in details:
                    print(f"   Coherence: {details['coherence']:.3f}")

            # Show a snippet of the sample
            instruction = sample.get('instruction', '')[:100]
            print(f"   Instruction: {instruction}...")
            print()

    # Analyze quality score distribution
    if samples_with_scores:
        print("QUALITY SCORE DISTRIBUTION")
        print("-" * 80)

        scores = [s['quality_score'] for s in samples_with_scores]

        # Create bins
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        bin_counts = [0] * (len(bins) - 1)

        for score in scores:
            for i in range(len(bins) - 1):
                if bins[i] <= score < bins[i + 1]:
                    bin_counts[i] += 1
                    break

        for i in range(len(bin_counts)):
            count = bin_counts[i]
            pct = (count / len(scores)) * 100 if scores else 0
            bar = "█" * int(pct / 2)  # Scale to fit terminal
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {bar} {count:4d} ({pct:5.1f}%)")
        print()

    # Common patterns analysis
    print("SAMPLE INSPECTION")
    print("-" * 80)
    print("Sample domains and rejection reasons:")
    print()

    # Group by domain and reason
    domain_reason_counts = Counter()
    for sample in rejected_samples:
        key = (sample.get('domain', 'unknown'), sample.get('rejection_reason', 'unknown'))
        domain_reason_counts[key] += 1

    for (domain, reason), count in domain_reason_counts.most_common(10):
        print(f"  {domain:15s} | {reason:25s} | {count:5d}")

    print()
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Generate recommendations based on patterns
    if summary_file.exists():
        top_reason = max(summary['by_reason'].items(), key=lambda x: x[1])[0]
        avg_score = summary['avg_quality_score']

        print()
        if top_reason == 'low_quality' and avg_score < 0.4:
            print("• Most samples are being rejected for low overall quality")
            print("  - Consider lowering domain-specific thresholds in quality.py")
            print("  - Check if the generation prompts are producing good outputs")
            print()

        if top_reason == 'low_diversity':
            print("• Many samples lack diversity")
            print("  - Improve prompt variety in generators")
            print("  - Add more source material for training data")
            print()

        if top_reason == 'kg_inconsistent':
            print("• Knowledge graph consistency issues")
            print("  - Update knowledge graph with correct information")
            print("  - Review entity validation logic")
            print()

        if top_reason == 'high_hallucination':
            print("• High hallucination risk detected")
            print("  - Improve grounding in generators")
            print("  - Add more source context to prompts")
            print()

        if top_reason == 'duplicate':
            print("• Many duplicates detected")
            print("  - Increase prompt diversity")
            print("  - Add more source variation")
            print()

    return True


def find_latest_dataset():
    """Find the most recent dataset directory."""
    dataset_dir = Path.home() / ".context" / "training" / "datasets"

    if not dataset_dir.exists():
        return None

    dataset_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    if not dataset_dirs:
        return None

    # Sort by modification time
    return max(dataset_dirs, key=lambda x: x.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Analyze rejected training samples")
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        help="Dataset directory to analyze",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Analyze the most recent dataset",
    )

    args = parser.parse_args()

    if args.latest:
        dataset_dir = find_latest_dataset()
        if not dataset_dir:
            print("✗ No dataset directories found")
            return 1
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.exists():
            print(f"✗ Dataset directory not found: {dataset_dir}")
            return 1
    else:
        parser.print_help()
        print("\nExample:")
        print("  python scripts/analyze_rejected_samples.py --latest")
        print("  python scripts/analyze_rejected_samples.py ~/.context/training/datasets/my_dataset_20251221_123456")
        return 0

    success = analyze_rejected_samples(dataset_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
