#!/usr/bin/env python3
"""Analyze training sample quality by source file."""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def analyze_sources(dataset_path: Path) -> Dict[str, dict]:
    """Analyze sample quality metrics by source file."""

    accepted_file = dataset_path / "accepted.jsonl"
    rejected_file = dataset_path / "rejected.jsonl"

    if not accepted_file.exists():
        print(f"Error: {accepted_file} not found", file=sys.stderr)
        sys.exit(1)

    source_stats = defaultdict(lambda: {
        'accepted': 0,
        'rejected': 0,
        'quality_scores': [],
        'diversity_scores': [],
        'rejection_reasons': [],
    })

    # Process accepted samples
    if accepted_file.exists():
        with open(accepted_file) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                source = sample.get('metadata', {}).get('source_file', 'unknown')
                source_stats[source]['accepted'] += 1
                source_stats[source]['quality_scores'].append(
                    sample.get('quality_score', 0)
                )
                source_stats[source]['diversity_scores'].append(
                    sample.get('diversity_score', 0)
                )

    # Process rejected samples
    if rejected_file.exists():
        with open(rejected_file) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                source = sample.get('metadata', {}).get('source_file', 'unknown')
                source_stats[source]['rejected'] += 1
                source_stats[source]['rejection_reasons'].append(
                    sample.get('rejection_reason', 'unknown')
                )

    return dict(source_stats)


def print_source_report(source_stats: Dict[str, dict]):
    """Print detailed source quality report."""

    print("\n╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║                        Source Quality Analysis                                 ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝\n")

    # Calculate overall stats
    total_accepted = sum(s['accepted'] for s in source_stats.values())
    total_rejected = sum(s['rejected'] for s in source_stats.values())
    total_samples = total_accepted + total_rejected

    print(f"Overall Statistics:")
    print(f"  Total samples:   {total_samples}")
    print(f"  Accepted:        {total_accepted} ({total_accepted/total_samples*100:.1f}%)")
    print(f"  Rejected:        {total_rejected} ({total_rejected/total_samples*100:.1f}%)")
    print(f"  Unique sources:  {len(source_stats)}")
    print()

    # Sort sources by acceptance rate
    sorted_sources = sorted(
        source_stats.items(),
        key=lambda x: x[1]['accepted'] / (x[1]['accepted'] + x[1]['rejected'])
        if (x[1]['accepted'] + x[1]['rejected']) > 0 else 0,
        reverse=True
    )

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Source Quality Breakdown")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    for source, stats in sorted_sources:
        total = stats['accepted'] + stats['rejected']
        if total == 0:
            continue

        acceptance_rate = stats['accepted'] / total
        avg_quality = (
            sum(stats['quality_scores']) / len(stats['quality_scores'])
            if stats['quality_scores'] else 0
        )
        avg_diversity = (
            sum(stats['diversity_scores']) / len(stats['diversity_scores'])
            if stats['diversity_scores'] else 0
        )

        # Determine quality tier
        if acceptance_rate >= 0.8:
            tier = "🟢 HIGH"
        elif acceptance_rate >= 0.6:
            tier = "🟡 MEDIUM"
        else:
            tier = "🔴 LOW"

        print(f"{tier}  {source}")
        print(f"     Acceptance:  {acceptance_rate:.1%} ({stats['accepted']}/{total})")
        if avg_quality > 0:
            print(f"     Avg Quality: {avg_quality:.3f}")
        if avg_diversity > 0:
            print(f"     Avg Diversity: {avg_diversity:.3f}")

        # Show rejection reasons for low-quality sources
        if acceptance_rate < 0.6 and stats['rejection_reasons']:
            reason_counts = defaultdict(int)
            for reason in stats['rejection_reasons']:
                reason_counts[reason] += 1
            top_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"     Top rejections: {', '.join(f'{r}({c})' for r, c in top_reasons)}")

        print()


def print_recommendations(source_stats: Dict[str, dict]):
    """Print actionable recommendations."""

    print("╔════════════════════════════════════════════════════════════════════════════════╗")
    print("║                            Recommendations                                     ║")
    print("╚════════════════════════════════════════════════════════════════════════════════╝\n")

    # Find low-quality sources
    low_quality = []
    for source, stats in source_stats.items():
        total = stats['accepted'] + stats['rejected']
        if total < 5:  # Skip sources with too few samples
            continue

        acceptance_rate = stats['accepted'] / total if total > 0 else 0
        if acceptance_rate < 0.5:
            low_quality.append((source, acceptance_rate, stats))

    if low_quality:
        print("🔴 Low-Quality Sources (Consider filtering):\n")
        for source, rate, stats in sorted(low_quality, key=lambda x: x[1]):
            print(f"   • {source}")
            print(f"     Acceptance: {rate:.1%}")

            # Analyze rejection reasons
            if stats['rejection_reasons']:
                reason_counts = defaultdict(int)
                for reason in stats['rejection_reasons']:
                    reason_counts[reason] += 1

                if reason_counts['diversity_too_low'] > len(stats['rejection_reasons']) * 0.7:
                    print(f"     → Issue: Low diversity - add prompt variation")
                elif reason_counts['quality_too_low'] > len(stats['rejection_reasons']) * 0.5:
                    print(f"     → Issue: Low quality - improve prompts or filter source")
                elif any('irrelevant' in r for r in stats['rejection_reasons']):
                    print(f"     → Issue: Irrelevant content - add relevance filtering")

            print()

    # Find high-quality sources
    high_quality = []
    for source, stats in source_stats.items():
        total = stats['accepted'] + stats['rejected']
        if total < 5:
            continue

        acceptance_rate = stats['accepted'] / total if total > 0 else 0
        if acceptance_rate >= 0.8:
            high_quality.append((source, acceptance_rate, stats))

    if high_quality:
        print("\n🟢 High-Quality Sources (Generate more):\n")
        for source, rate, stats in sorted(high_quality, key=lambda x: x[1], reverse=True)[:5]:
            print(f"   • {source}")
            print(f"     Acceptance: {rate:.1%} ({stats['accepted']} samples)")

            avg_quality = (
                sum(stats['quality_scores']) / len(stats['quality_scores'])
                if stats['quality_scores'] else 0
            )
            if avg_quality > 0:
                print(f"     Avg Quality: {avg_quality:.3f}")

            print(f"     → Recommendation: Increase sample count from this source")
            print()


def main():
    parser = argparse.ArgumentParser(description="Analyze training sample sources")
    parser.add_argument("dataset_path", help="Path to dataset directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    source_stats = analyze_sources(dataset_path)

    if args.json:
        # Calculate summary stats for JSON output
        output = {}
        for source, stats in source_stats.items():
            total = stats['accepted'] + stats['rejected']
            output[source] = {
                'accepted': stats['accepted'],
                'rejected': stats['rejected'],
                'acceptance_rate': stats['accepted'] / total if total > 0 else 0,
                'avg_quality': (
                    sum(stats['quality_scores']) / len(stats['quality_scores'])
                    if stats['quality_scores'] else 0
                ),
                'avg_diversity': (
                    sum(stats['diversity_scores']) / len(stats['diversity_scores'])
                    if stats['diversity_scores'] else 0
                ),
            }
        print(json.dumps(output, indent=2))
    else:
        print_source_report(source_stats)
        print_recommendations(source_stats)


if __name__ == "__main__":
    main()
