#!/usr/bin/env python3
"""Generate comprehensive evaluation reports.

Combines all evaluation metrics into a single Markdown report.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_eval_results(eval_dir: Path) -> dict:
    """Load all evaluation results from directory."""
    results = {}

    # Load each metric file
    metric_files = {
        "perplexity": "perplexity.json",
        "generation": "generation.json",
        "llm_judge": "llm_judge.json",
    }

    for metric, filename in metric_files.items():
        filepath = eval_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                results[metric] = json.load(f)

    # Load benchmark results
    for benchmark_file in eval_dir.glob("*_benchmark.json"):
        benchmark_name = benchmark_file.stem.replace("_benchmark", "")
        with open(benchmark_file) as f:
            results[f"benchmark_{benchmark_name}"] = json.load(f)

    return results


def generate_report(eval_dir: Path, model_name: str) -> str:
    """Generate Markdown evaluation report."""
    results = load_eval_results(eval_dir)

    if not results:
        return "# Evaluation Report\n\nNo evaluation results found."

    # Build report
    report = []
    report.append("# Model Evaluation Report\n")
    report.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Model:** {model_name}\n")
    report.append(f"**Evaluation Directory:** {eval_dir}\n")
    report.append("\n---\n")

    # Automatic Metrics
    report.append("## Automatic Metrics\n")

    if "perplexity" in results:
        ppl = results["perplexity"]
        report.append(f"### Perplexity\n")
        report.append(f"- **Perplexity:** {ppl.get('perplexity', 'N/A'):.2f}\n")
        report.append(f"- **Average Loss:** {ppl.get('avg_loss', 'N/A'):.4f}\n")
        report.append(f"- **Test Samples:** {ppl.get('num_samples', 'N/A')}\n\n")

    if "generation" in results:
        gen = results["generation"]
        report.append(f"### Generation Quality\n")
        report.append(f"- **BLEU:** {gen.get('bleu', 'N/A'):.4f}\n")
        report.append(f"- **ROUGE-1:** {gen.get('rouge-1', 'N/A'):.4f}\n")
        report.append(f"- **ROUGE-2:** {gen.get('rouge-2', 'N/A'):.4f}\n")
        report.append(f"- **ROUGE-L:** {gen.get('rouge-L', 'N/A'):.4f}\n")
        report.append(f"- **Exact Match:** {gen.get('exact_match', 'N/A'):.2%}\n\n")

    # Benchmark Results
    benchmarks = [k for k in results if k.startswith("benchmark_")]
    if benchmarks:
        report.append("## Domain Benchmarks\n")

        for bench_key in benchmarks:
            bench = results[bench_key]
            bench_name = bench_key.replace("benchmark_", "").replace("_", " ").title()

            report.append(f"### {bench_name}\n")
            report.append(f"- **Overall Score:** {bench.get('overall_score', 0):.1%}\n")

            if "category_scores" in bench:
                report.append(f"\n**By Category:**\n")
                for cat, score in sorted(bench["category_scores"].items()):
                    report.append(f"- {cat}: {score:.1%}\n")

            report.append("\n")

    # LLM Judge Scores
    if "llm_judge" in results:
        judge = results["llm_judge"]
        avg = judge.get("average_scores", {})

        report.append("## LLM Judge Evaluation\n")
        report.append(f"**Judge:** {judge.get('judge', 'N/A')}\n")
        report.append(f"**Samples Evaluated:** {judge.get('num_samples', 'N/A')}\n\n")

        report.append("| Dimension | Score (1-10) |\n")
        report.append("|-----------|-------------|\n")
        report.append(f"| Correctness | {avg.get('correctness', 0):.2f} |\n")
        report.append(f"| Completeness | {avg.get('completeness', 0):.2f} |\n")
        report.append(f"| Clarity | {avg.get('clarity', 0):.2f} |\n")
        report.append(f"| Code Quality | {avg.get('code_quality', 0):.2f} |\n")
        report.append(f"| **Overall** | **{avg.get('overall', 0):.2f}** |\n\n")

    # Summary Table
    report.append("---\n")
    report.append("## Summary\n\n")
    report.append("| Metric | Value |\n")
    report.append("|--------|-------|\n")

    if "perplexity" in results:
        report.append(f"| Perplexity | {results['perplexity'].get('perplexity', 'N/A'):.2f} |\n")

    if "generation" in results:
        gen = results["generation"]
        report.append(f"| BLEU | {gen.get('bleu', 'N/A'):.4f} |\n")
        report.append(f"| ROUGE-L | {gen.get('rouge-L', 'N/A'):.4f} |\n")

    for bench_key in benchmarks:
        bench = results[bench_key]
        bench_name = bench_key.replace("benchmark_", "").upper()
        report.append(f"| {bench_name} Benchmark | {bench.get('overall_score', 0):.1%} |\n")

    if "llm_judge" in results:
        avg = results["llm_judge"].get("average_scores", {})
        report.append(f"| LLM Judge Overall | {avg.get('overall', 0):.2f}/10 |\n")

    report.append("\n")

    # Recommendations
    report.append("## Recommendations\n\n")

    # Analyze results and provide suggestions
    if "perplexity" in results:
        ppl_value = results["perplexity"].get("perplexity", float('inf'))
        if ppl_value > 15:
            report.append("- ⚠️ **High perplexity** - Model may need more training data or longer training\n")
        elif ppl_value < 5:
            report.append("- ✓ **Excellent perplexity** - Model has strong domain understanding\n")

    for bench_key in benchmarks:
        bench = results[bench_key]
        score = bench.get("overall_score", 0)
        bench_name = bench_key.replace("benchmark_", "")

        if score < 0.5:
            report.append(f"- ⚠️ **{bench_name}** score low - Generate more training data for this domain\n")
        elif score > 0.75:
            report.append(f"- ✓ **{bench_name}** performing well\n")

    if "llm_judge" in results:
        avg = results["llm_judge"].get("average_scores", {})
        if avg.get("overall", 0) < 6:
            report.append("- ⚠️ **LLM judge scores low** - Consider DPO training to improve quality\n")
        elif avg.get("overall", 0) > 8:
            report.append("- ✓ **Strong LLM judge scores** - Model quality is excellent\n")

    return "".join(report)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--eval_dir",
        type=str,
        required=True,
        help="Directory with evaluation results",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hafs-coder",
        help="Model name for report",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output markdown file (default: <eval_dir>/report.md)",
    )

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)

    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        return 1

    # Generate report
    logger.info("Generating evaluation report...")
    report = generate_report(eval_dir, args.model_name)

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = eval_dir / "report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"✓ Report saved to {output_path}")

    # Also print to stdout
    print(report)

    return 0


if __name__ == "__main__":
    exit(main())
