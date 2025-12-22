#!/usr/bin/env python3
"""LLM-as-Judge evaluation using Claude Opus or GPT-4.

Uses frontier models to judge the quality of model outputs.
"""

from __future__ import annotations

import argparse
import asyncio
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


async def judge_with_claude(
    instruction: str, reference: str, prediction: str
) -> dict:
    """Use Claude Opus to judge response quality."""
    try:
        import anthropic

        client = anthropic.Anthropic()

        prompt = f"""You are evaluating the quality of a coding assistant response.

USER REQUEST:
{instruction}

GROUND TRUTH (REFERENCE):
{reference[:1000]}

MODEL PREDICTION:
{prediction[:1000]}

Rate the prediction on a scale of 1-10 for:
1. **Correctness**: Is the technical information accurate?
2. **Completeness**: Does it fully answer the question?
3. **Clarity**: Is the explanation clear and well-structured?
4. **Code Quality**: If code is present, is it correct and idiomatic?

Respond with JSON only:
{{
  "correctness": 1-10,
  "completeness": 1-10,
  "clarity": 1-10,
  "code_quality": 1-10,
  "overall": 1-10,
  "reasoning": "Brief explanation (1-2 sentences)"
}}"""

        response = await asyncio.to_thread(
            client.messages.create,
            model="claude-opus-4-5-20251101",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract JSON
        content = response.content[0].text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "{" in content:
            content = content[content.find("{") : content.rfind("}") + 1]

        return json.loads(content)

    except Exception as e:
        logger.error(f"Claude judge failed: {e}")
        return {
            "correctness": 0,
            "completeness": 0,
            "clarity": 0,
            "code_quality": 0,
            "overall": 0,
            "reasoning": f"Error: {str(e)}",
        }


async def judge_with_gemini(
    instruction: str, reference: str, prediction: str
) -> dict:
    """Use Gemini Pro to judge response quality."""
    try:
        import google.generativeai as genai
        import os

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-3-pro-preview")

        prompt = f"""Evaluate this coding assistant response (1-10 scale).

REQUEST: {instruction}
REFERENCE: {reference[:1000]}
PREDICTION: {prediction[:1000]}

Rate: correctness, completeness, clarity, code_quality.
JSON only: {{"correctness": X, "completeness": X, "clarity": X, "code_quality": X, "overall": X, "reasoning": "..."}}"""

        response = await asyncio.to_thread(model.generate_content, prompt)
        content = response.text

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "{" in content:
            content = content[content.find("{") : content.rfind("}") + 1]

        return json.loads(content)

    except Exception as e:
        logger.error(f"Gemini judge failed: {e}")
        return {
            "correctness": 0,
            "completeness": 0,
            "clarity": 0,
            "code_quality": 0,
            "overall": 0,
            "reasoning": f"Error: {str(e)}",
        }


async def llm_judge_samples(
    model_path: str,
    test_file: Path,
    judge: str = "claude",
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate model outputs using LLM judge.

    Args:
        model_path: Path to model or HuggingFace ID
        test_file: JSONL test file
        judge: "claude" or "gemini"
        max_samples: Limit number of samples (for cost control)

    Returns:
        Aggregated judge scores
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

    # Load samples
    logger.info(f"Loading test samples from {test_file}")
    samples = []
    with open(test_file) as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if max_samples:
        samples = samples[:max_samples]

    logger.info(f"Evaluating {len(samples)} samples with {judge} judge")

    # Generate and judge
    judge_fn = judge_with_claude if judge == "claude" else judge_with_gemini
    scores = []

    for i, sample in enumerate(samples):
        # Generate prediction
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reference = sample.get("output", "")

        user_content = instruction
        if input_text:
            user_content += f"\n\n{input_text}"

        messages = [
            {"role": "system", "content": "You are an expert SNES developer."},
            {"role": "user", "content": user_content},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        prediction = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Judge with LLM
        score = await judge_fn(instruction, reference, prediction)
        scores.append(score)

        logger.info(
            f"[{i+1}/{len(samples)}] Overall: {score.get('overall', 0)}/10 - {score.get('reasoning', 'N/A')[:50]}"
        )

        # Rate limiting
        await asyncio.sleep(1.0)

    # Aggregate scores
    avg_scores = {
        "correctness": sum(s.get("correctness", 0) for s in scores) / len(scores),
        "completeness": sum(s.get("completeness", 0) for s in scores) / len(scores),
        "clarity": sum(s.get("clarity", 0) for s in scores) / len(scores),
        "code_quality": sum(s.get("code_quality", 0) for s in scores) / len(scores),
        "overall": sum(s.get("overall", 0) for s in scores) / len(scores),
    }

    logger.info(f"✓ Average Overall Score: {avg_scores['overall']:.2f}/10")

    return {
        "average_scores": avg_scores,
        "individual_scores": scores,
        "num_samples": len(scores),
        "judge": judge,
    }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model path or ID")
    parser.add_argument(
        "--test_set", type=str, required=True, help="Test set JSONL"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="claude",
        choices=["claude", "gemini"],
        help="Judge model to use",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Max samples to evaluate (for cost control)",
    )
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    results = asyncio.run(
        llm_judge_samples(
            model_path=args.model,
            test_file=Path(args.test_set),
            judge=args.judge,
            max_samples=args.max_samples,
        )
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    else:
        print(json.dumps(results["average_scores"], indent=2))


if __name__ == "__main__":
    main()
