#!/usr/bin/env python3
"""Perplexity evaluation for fine-tuned models.

Measures how confident the model is on test data.
Lower perplexity = better understanding of the domain.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def compute_perplexity(
    model_path: str,
    test_file: Path,
    batch_size: int = 4,
    max_length: int = 2048,
) -> dict:
    """Compute perplexity on test set.

    Args:
        model_path: Path to model or HuggingFace model ID
        test_file: JSONL file with test samples
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length

    Returns:
        Dictionary with perplexity metrics
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

    # Load test samples
    logger.info(f"Loading test samples from {test_file}")
    samples = []
    with open(test_file) as f:
        for line in f:
            try:
                sample = json.loads(line)
                # Format as conversation
                text = f"{sample.get('instruction', '')}\n\n{sample.get('input', '')}\n\n{sample.get('output', '')}"
                samples.append(text)
            except json.JSONDecodeError:
                continue

    logger.info(f"Loaded {len(samples)} test samples")

    # Compute perplexity
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]

            # Tokenize
            encodings = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            # Forward pass
            outputs = model(**encodings, labels=encodings.input_ids)
            loss = outputs.loss

            # Accumulate
            batch_tokens = encodings.attention_mask.sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            if (i // batch_size) % 10 == 0:
                logger.info(f"Processed {i}/{len(samples)} samples")

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"✓ Perplexity: {perplexity:.2f}")

    return {
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on test set")
    parser.add_argument("--model", type=str, required=True, help="Model path or ID")
    parser.add_argument(
        "--test_set", type=str, required=True, help="Test set JSONL file"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Max sequence length"
    )
    parser.add_argument(
        "--output", type=str, help="Output JSON file for results"
    )

    args = parser.parse_args()

    results = compute_perplexity(
        model_path=args.model,
        test_file=Path(args.test_set),
        batch_size=args.batch_size,
        max_length=args.max_length,
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
