#!/usr/bin/env python3
"""Domain-specific benchmark evaluation.

Tests models on ASM, ROM hacking, and code understanding tasks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_benchmark(benchmark_path: Path) -> list[dict]:
    """Load benchmark test set."""
    samples = []
    with open(benchmark_path) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


def evaluate_sample(
    model,
    tokenizer,
    sample: dict,
    max_new_tokens: int = 512,
) -> dict:
    """Evaluate a single benchmark sample."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    reference = sample.get("output", "")
    category = sample.get("category", "general")

    # Build prompt
    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"

    messages = [
        {"role": "system", "content": "You are an expert SNES developer specializing in Zelda: A Link to the Past ROM hacking and 65816 assembly."},
        {"role": "user", "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    prediction = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )

    # Simple scoring heuristics
    # (In production, you'd use LLM judge or human eval)
    score = 0.0

    # Exact match (very strict)
    if prediction.strip() == reference.strip():
        score = 1.0
    # Substantial overlap
    elif len(set(prediction.lower().split()) & set(reference.lower().split())) > 10:
        score = 0.7
    # Some overlap
    elif len(set(prediction.lower().split()) & set(reference.lower().split())) > 5:
        score = 0.4
    # Minimal overlap
    else:
        score = 0.1

    return {
        "category": category,
        "score": score,
        "prediction": prediction,
        "reference": reference,
    }


def run_benchmark(
    model_path: str,
    benchmark_name: str = "asm",
    benchmark_dir: Optional[Path] = None,
) -> dict:
    """Run domain-specific benchmark.

    Args:
        model_path: Path to model or HuggingFace ID
        benchmark_name: "asm", "rom_hack", or "code_understanding"
        benchmark_dir: Directory with benchmark files

    Returns:
        Benchmark results with scores by category
    """
    logger.info(f"Loading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Load benchmark
    if benchmark_dir is None:
        benchmark_dir = Path.home() / ".context" / "training" / "benchmarks"

    benchmark_file = benchmark_dir / f"{benchmark_name}_benchmark.jsonl"

    if not benchmark_file.exists():
        logger.error(f"Benchmark not found: {benchmark_file}")
        logger.info(f"Please create benchmark at {benchmark_file}")
        logger.info("See: python -m agents.training.eval.create_benchmarks")
        return {"error": f"Benchmark not found: {benchmark_file}"}

    logger.info(f"Loading benchmark: {benchmark_file}")
    samples = load_benchmark(benchmark_file)
    logger.info(f"Loaded {len(samples)} benchmark samples")

    # Evaluate each sample
    results = []
    category_scores = {}

    for i, sample in enumerate(samples):
        result = evaluate_sample(model, tokenizer, sample)
        results.append(result)

        # Track by category
        cat = result["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(result["score"])

        logger.info(
            f"[{i+1}/{len(samples)}] {cat}: {result['score']:.2f}"
        )

    # Aggregate scores
    overall_score = sum(r["score"] for r in results) / len(results)

    category_averages = {
        cat: sum(scores) / len(scores)
        for cat, scores in category_scores.items()
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"BENCHMARK: {benchmark_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Score: {overall_score:.1%}")
    logger.info(f"\nBy Category:")
    for cat, score in sorted(category_averages.items()):
        logger.info(f"  {cat:30s}: {score:.1%}")
    logger.info(f"{'='*60}\n")

    return {
        "benchmark": benchmark_name,
        "overall_score": overall_score,
        "category_scores": category_averages,
        "num_samples": len(samples),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run domain-specific benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model path or ID")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="asm",
        choices=["asm", "rom_hack", "code_understanding"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        help="Directory with benchmark files",
    )
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model,
        benchmark_name=args.benchmark,
        benchmark_dir=Path(args.benchmark_dir) if args.benchmark_dir else None,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        print(json.dumps({
            "overall_score": results["overall_score"],
            "category_scores": results["category_scores"],
        }, indent=2))


if __name__ == "__main__":
    main()
