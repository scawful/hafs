#!/usr/bin/env python3
"""LoRA Fine-Tuning Script for hAFS.

Trains parameter-efficient LoRA adapters on top of base models like Qwen2.5-Coder.
Designed to run on medical-mechanica (RTX 4070 Ti SUPER, 16GB VRAM).

Usage:
    python -m agents.training.lora_trainer \
        --base_model Qwen/Qwen2.5-Coder-14B-Instruct \
        --dataset ~/.context/training/pilot_hybrid_100_*/train.jsonl \
        --output_dir ~/.context/training/checkpoints/hafs-coder-v1 \
        --lora_r 64 \
        --batch_size 4 \
        --num_epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> Dataset:
    """Load JSONL training data."""
    logger.info(f"Loading training data from {data_path}")

    samples = []
    for jsonl_file in data_path.parent.glob("*.jsonl"):
        if "train" in jsonl_file.name or "samples" in jsonl_file.name:
            logger.info(f"Reading {jsonl_file}")
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    logger.info(f"Loaded {len(samples)} training samples")

    if len(samples) == 0:
        raise ValueError(f"No training samples found in {data_path.parent}")

    return Dataset.from_list(samples)


def format_sample(sample: dict, tokenizer) -> dict:
    """Format instruction-tuning sample for Qwen2.5 chat format."""
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    # Combine instruction + input as user message
    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"

    # Qwen2.5 chat template
    messages = [
        {"role": "system", "content": "You are an expert SNES developer specializing in Zelda: A Link to the Past ROM hacking and 65816 assembly."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    return {"text": formatted}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for hAFS")

    # Model args
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-14B-Instruct",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization (lower VRAM, slight quality loss)",
    )

    # Data args
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training data (JSONL file or directory)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation set split ratio",
    )

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Modules to apply LoRA",
    )

    # Training args
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Maximum sequence length"
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
    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    data_path = Path(args.dataset)
    dataset = load_training_data(data_path)

    # Format samples
    logger.info("Formatting samples with chat template...")
    dataset = dataset.map(
        lambda x: format_sample(x, tokenizer), remove_columns=dataset.column_names
    )

    # Tokenize
    logger.info("Tokenizing dataset...")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
    )

    # Split train/val
    split = tokenized_dataset.train_test_split(test_size=args.val_split)
    train_dataset = split["train"]
    val_dataset = split["test"]

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Load model
    logger.info(f"Loading base model: {args.base_model}")

    if args.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if args.fp16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Configure LoRA
    logger.info(f"Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="tensorboard",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train!
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {args.output_dir}/final")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
