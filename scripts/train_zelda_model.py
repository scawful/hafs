#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Zelda-themed fine-tuned model
Usage: python train_zelda_model.py <zelda_name> <dataset_path> [base_model]
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Zelda naming scheme
ZELDA_NAMES = {
    # Legendary items (high quality)
    "master-sword": "The legendary blade - production ready",
    "triforce": "Ultimate power - highest quality",
    "light-arrows": "Sacred weapon - specialized task",
    "hylian-shield": "Ultimate defense - robust model",
    "ocarina": "Ocarina of Time - temporal specialist",

    # Regular items (good quality)
    "hookshot": "Versatile tool - general purpose",
    "boomerang": "Reliable tool - consistent performance",
    "bow": "Classic weapon - solid baseline",
    "bombs": "Explosive power - high impact",
    "mirror-shield": "Reflective defense - defensive specialist",

    # Experimental
    "deku-stick": "Quick and dirty - alpha test",
    "slingshot": "Early attempt - beta test",
    "bottle": "Flexible use - experimental",
}

# Parse arguments
if len(sys.argv) < 3:
    print("Usage: python train_zelda_model.py <zelda_name> <dataset_path> [base_model]")
    print()
    print("Available Zelda names:")
    for name, desc in ZELDA_NAMES.items():
        print(f"  {name:20s} - {desc}")
    print()
    print("Example:")
    print("  python train_zelda_model.py master-sword D:/.context/training/datasets/alttp_yaze_full_1000_20251221_195746")
    sys.exit(1)

ZELDA_NAME = sys.argv[1]
DATASET_PATH = Path(sys.argv[2])
BASE_MODEL = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen2.5-Coder-1.5B"

# Validate Zelda name
if ZELDA_NAME not in ZELDA_NAMES:
    print(f"✗ Unknown Zelda name: {ZELDA_NAME}")
    print(f"Available: {', '.join(ZELDA_NAMES.keys())}")
    sys.exit(1)

# Extract model size
import re
match = re.search(r'(\d+\.?\d*)B', BASE_MODEL, re.IGNORECASE)
MODEL_SIZE = match.group(0).lower() if match else "1.5b"

# Generate full model name: zelda-item-size-date
DATE = datetime.now().strftime("%Y%m%d")
MODEL_NAME = f"{ZELDA_NAME}-{MODEL_SIZE}-{DATE}"

# Output directory
OUTPUT_DIR = Path("D:/.context/training/models") / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check CUDA
if not torch.cuda.is_available():
    print("✗ CUDA not available! Training will be very slow.")
    device = "cpu"
else:
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✓ Using GPU: {gpu_name}")

print("=" * 80)
print(f"Training Zelda Model: {ZELDA_NAME.upper()}")
print("=" * 80)
print(f"Description:  {ZELDA_NAMES[ZELDA_NAME]}")
print(f"Dataset:      {DATASET_PATH}")
print(f"Base Model:   {BASE_MODEL}")
print(f"Output Model: {MODEL_NAME}")
print(f"Output Dir:   {OUTPUT_DIR}")
print(f"Device:       {device}")
print()

# Check dataset
if not (DATASET_PATH / "train.jsonl").exists():
    print(f"✗ Dataset not found: {DATASET_PATH}/train.jsonl")
    sys.exit(1)

# Load dataset stats
stats_file = DATASET_PATH / "stats.json"
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)
        train_count = stats.get('final_count', 0)
        avg_quality = stats.get('quality_scores', {}).get('average', 0)
        print(f"Training samples: {train_count}")
        print(f"Average quality: {avg_quality:.3f}")
        print()

# Import libraries
print("[1/6] Loading libraries...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import load_dataset
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

print()

# Load tokenizer
print(f"[2/6] Loading tokenizer: {BASE_MODEL}")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f"✗ Failed to load tokenizer: {e}")
    sys.exit(1)

print()

# Load base model
print(f"[3/6] Loading base model: {BASE_MODEL}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    # Add LoRA adapters
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("✓ Model loaded with LoRA adapters")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

print()

# Load dataset
print("[4/6] Loading training data...")
dataset = load_dataset("json", data_files={
    "train": str(DATASET_PATH / "train.jsonl"),
})["train"]
print(f"✓ Loaded {len(dataset)} samples")
print()

# Format and tokenize
def format_sample(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        prompt = f"""Below is an instruction for 65816 assembly, paired with context. Write a response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        prompt = f"""Below is an instruction for 65816 assembly. Write a response.

### Instruction:
{instruction}

### Response:
{output}"""

    return tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    format_sample,
    remove_columns=dataset.column_names,
    batched=False,
)
print("✓ Dataset tokenized")
print()

# Training configuration
print("[5/6] Configuring training...")
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=4 if device == "cuda" else 1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=(device == "cuda"),
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

print(f"  Epochs: 1")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: 2e-4")
print(f"  FP16: {training_args.fp16}")
print()

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
print(f"[6/6] Training {ZELDA_NAME.upper()}...")
start_time = datetime.now()
print(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
if device == "cuda":
    print(f"Estimated time: ~15-30 minutes on RTX GPU")
print()

try:
    trainer.train()
    duration = (datetime.now() - start_time).total_seconds()
    print()
    print(f"✓ Training completed in {duration:.0f}s ({duration/60:.1f} minutes)")
except KeyboardInterrupt:
    print("\n✗ Training interrupted")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Save model
print("Saving model...")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"✓ Model saved to: {OUTPUT_DIR}")

# Save metadata
metadata = {
    "name": MODEL_NAME,
    "zelda_name": ZELDA_NAME,
    "description": ZELDA_NAMES[ZELDA_NAME],
    "base_model": BASE_MODEL,
    "dataset": str(DATASET_PATH),
    "created": datetime.now().isoformat(),
    "training_samples": len(dataset),
    "training_duration_seconds": duration,
    "training_duration_minutes": duration / 60,
    "device": device,
    "model_size": MODEL_SIZE,
    "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "None",
}

metadata_file = OUTPUT_DIR / "metadata.json"
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved")
print()
print("=" * 80)
print(f"{ZELDA_NAME.upper()} Training Complete!")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Location: {OUTPUT_DIR}")
print(f"Description: {ZELDA_NAMES[ZELDA_NAME]}")
print()
