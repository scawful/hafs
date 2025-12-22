#!/usr/bin/env python3
"""Quick test script for newly trained models."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Test a trained model")
    parser.add_argument("--model-path", required=True, help="Path to trained model directory")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B", help="Base model name")
    parser.add_argument("--prompt", help="Single test prompt")
    parser.add_argument("--prompt-file", help="File with test prompts (one per line)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(base, str(model_path))
    model.eval()

    # Collect prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        # Default Oracle test prompts
        prompts = [
            "Explain the secrets system in Oracle of Secrets",
            "How does the ring menu work in Oracle ROM hacks?",
            "Write ASM code to add a new item to the inventory",
            "What are the main differences between Oracle of Ages and Oracle of Secrets?",
        ]

    print(f"\n{'='*80}")
    print(f"Running {len(prompts)} test prompts")
    print(f"{'='*80}\n")

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        print("-" * 80)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=True,
                top_p=0.9,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()

        print(f"Response: {response}\n")

    print(f"\n{'='*80}")
    print("Test complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
