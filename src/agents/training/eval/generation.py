#!/usr/bin/env python3
"""Generation quality evaluation using BLEU and ROUGE metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def evaluate_generation_quality(
    model_path: str,
    test_file: Path,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> dict:
    """Evaluate generation quality with BLEU and ROUGE.

    Args:
        model_path: Path to model or HuggingFace ID
        test_file: JSONL with test samples
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary with BLEU and ROUGE scores
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

    # Load metrics
    bleu = load("bleu")
    rouge = load("rouge")

    # Load test samples
    logger.info(f"Loading test samples from {test_file}")
    samples = []
    with open(test_file) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(samples)} test samples")

    # Generate predictions
    predictions = []
    references = []

    for i, sample in enumerate(samples):
        # Build prompt
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reference = sample.get("output", "")

        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"

        messages = [
            {"role": "system", "content": "You are an expert SNES developer specializing in Zelda: A Link to the Past ROM hacking and 65816 assembly."},
            {"role": "user", "content": user_content},
        ]

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode (skip prompt)
        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        predictions.append(prediction)
        references.append(reference)

        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i + 1}/{len(samples)} samples")

    # Compute metrics
    logger.info("Computing BLEU...")
    bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])

    logger.info("Computing ROUGE...")
    rouge_score = rouge.compute(predictions=predictions, references=references)

    # Exact match
    exact_matches = sum(
        p.strip() == r.strip() for p, r in zip(predictions, references)
    )
    exact_match_ratio = exact_matches / len(predictions)

    results = {
        "bleu": bleu_score["bleu"],
        "rouge-1": rouge_score["rouge1"],
        "rouge-2": rouge_score["rouge2"],
        "rouge-L": rouge_score["rougeL"],
        "exact_match": exact_match_ratio,
        "num_samples": len(predictions),
    }

    logger.info(f"✓ BLEU: {results['bleu']:.4f}")
    logger.info(f"✓ ROUGE-L: {results['rouge-L']:.4f}")
    logger.info(f"✓ Exact Match: {results['exact_match']:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generation quality")
    parser.add_argument("--model", type=str, required=True, help="Model path or ID")
    parser.add_argument(
        "--test_set", type=str, required=True, help="Test set JSONL file"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3, help="Sampling temperature"
    )
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    results = evaluate_generation_quality(
        model_path=args.model,
        test_file=Path(args.test_set),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
