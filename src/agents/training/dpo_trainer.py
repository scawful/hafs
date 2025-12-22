#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) Trainer for hAFS.

Improves fine-tuned models using human preference data.
Requires preference pairs: (prompt, chosen_response, rejected_response).

Usage:
    python -m agents.training.dpo_trainer \
        --sft_model ~/.context/training/checkpoints/hafs-coder-sft \
        --preference_data ~/.context/training/preferences.jsonl \
        --output_dir ~/.context/training/checkpoints/hafs-coder-dpo \
        --beta 0.1 \
        --learning_rate 5e-6 \
        --num_epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_preference_data(data_path: Path) -> Dataset:
    """Load preference pairs from JSONL.

    Expected format:
    {
        "prompt": "Write a sprite DMA routine",
        "chosen": "Good ASM code...",
        "rejected": "Bad/buggy code..."
    }
    """
    logger.info(f"Loading preference data from {data_path}")

    samples = []
    with open(data_path) as f:
        for line in f:
            try:
                sample = json.loads(line)
                # Validate required fields
                if all(k in sample for k in ["prompt", "chosen", "rejected"]):
                    samples.append(sample)
                else:
                    logger.warning(f"Skipping sample missing required fields: {sample.keys()}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line: {e}")
                continue

    logger.info(f"Loaded {len(samples)} preference pairs")

    if len(samples) == 0:
        raise ValueError(f"No valid preference pairs found in {data_path}")

    return Dataset.from_list(samples)


def format_for_dpo(sample: dict, tokenizer) -> dict:
    """Format preference pair for DPO training.

    DPO expects:
    - prompt: The instruction/query
    - chosen: The preferred completion
    - rejected: The non-preferred completion
    """
    # Build messages for chosen response
    chosen_messages = [
        {"role": "system", "content": "You are an expert SNES developer specializing in Zelda: A Link to the Past ROM hacking and 65816 assembly."},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["chosen"]},
    ]

    # Build messages for rejected response
    rejected_messages = [
        {"role": "system", "content": "You are an expert SNES developer specializing in Zelda: A Link to the Past ROM hacking and 65816 assembly."},
        {"role": "user", "content": sample["prompt"]},
        {"role": "assistant", "content": sample["rejected"]},
    ]

    # Apply chat template
    chosen_text = tokenizer.apply_chat_template(
        chosen_messages, tokenize=False, add_generation_prompt=False
    )
    rejected_text = tokenizer.apply_chat_template(
        rejected_messages, tokenize=False, add_generation_prompt=False
    )

    # Extract just the prompt part (without assistant response)
    prompt_messages = chosen_messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    return {
        "prompt": prompt_text,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }


def main():
    parser = argparse.ArgumentParser(description="DPO training for hAFS")

    # Model args
    parser.add_argument(
        "--sft_model",
        type=str,
        required=True,
        help="Path to SFT fine-tuned model (base for DPO)",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Apply LoRA adapters on top of SFT model",
    )

    # Data args
    parser.add_argument(
        "--preference_data",
        type=str,
        required=True,
        help="Path to preference pairs JSONL",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation set split ratio",
    )

    # DPO args
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature (0.1=conservative, 0.5=aggressive)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (much lower than SFT!)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of epochs (1 is usually enough)",
    )

    # LoRA args (if using)
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank (lower than SFT)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # Training args
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_length", type=int, default=2048, help="Maximum sequence length"
    )
    parser.add_argument(
        "--max_prompt_length", type=int, default=512, help="Maximum prompt length"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint frequency")

    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer from SFT model: {args.sft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load preference data
    dataset = load_preference_data(Path(args.preference_data))

    # Format for DPO
    logger.info("Formatting preference pairs...")
    dataset = dataset.map(
        lambda x: format_for_dpo(x, tokenizer),
        remove_columns=dataset.column_names,
    )

    # Split train/val
    split = dataset.train_test_split(test_size=args.val_split)
    train_dataset = split["train"]
    val_dataset = split["test"]

    logger.info(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

    # Load SFT model
    logger.info(f"Loading SFT model: {args.sft_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load reference model (frozen copy of SFT)
    logger.info("Loading reference model (frozen SFT)")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.sft_model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # Optional: Apply LoRA for parameter-efficient DPO
    if args.use_lora:
        logger.info(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # DPO training config
    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,  # DPO-specific: preference temperature
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="tensorboard",
        remove_unused_columns=False,  # DPO needs all columns
    )

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train!
    logger.info("Starting DPO training...")
    logger.info(f"Beta (temperature): {args.beta}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_epochs}")

    trainer.train()

    # Save final model
    logger.info(f"Saving DPO model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    logger.info("✓ DPO training complete!")
    logger.info(f"Model saved to: {args.output_dir}/final")


if __name__ == "__main__":
    main()
