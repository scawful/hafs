#!/usr/bin/env python3
"""
Train a fine-tuned model on Windows GPU server
Usage: python train_model_windows.py <dataset_name> <base_model> <quality_tag>
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Parse arguments
if len(sys.argv) < 2:
    print("Usage: python train_model_windows.py <dataset_name> [base_model] [quality_tag]")
    print()
    print("Example:")
    print("  python train_model_windows.py alttp_yaze_full_1000_20251221_195746")
    print("  python train_model_windows.py alttp_yaze_full_1000_20251221_195746 unsloth/qwen2.5-coder-1.5b-bnb-4bit gold")
    sys.exit(1)

DATASET_NAME = sys.argv[1]
BASE_MODEL = sys.argv[2] if len(sys.argv) > 2 else "unsloth/qwen2.5-coder-1.5b-bnb-4bit"
QUALITY_TAG = sys.argv[3] if len(sys.argv) > 3 else "alpha"

# Extract model size
import re
match = re.search(r'(\d+\.?\d*)b', BASE_MODEL, re.IGNORECASE)
MODEL_SIZE = match.group(0) if match else "1.5b"

# Generate model name
DATE = datetime.now().strftime("%Y%m%d")
MODEL_NAME = f"hafs-asm-{MODEL_SIZE}-{DATE}-{QUALITY_TAG}"

# Paths (Windows)
DATASET_PATH = Path(f"D:/.context/training/datasets/{DATASET_NAME}")
OUTPUT_DIR = Path(f"D:/.context/training/models/{MODEL_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Training Configuration")
print("=" * 80)
print(f"Dataset:      {DATASET_PATH}")
print(f"Base Model:   {BASE_MODEL}")
print(f"Output Model: {MODEL_NAME}")
print(f"Output Dir:   {OUTPUT_DIR}")
print(f"Quality Tag:  {QUALITY_TAG}")
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
        print(f"Training samples: {stats.get('final_count', 'unknown')}")
        print(f"Average quality: {stats.get('quality_scores', {}).get('average', 0):.3f}")
        print()

# Import libraries
print("[1/6] Loading libraries...")
try:
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import load_dataset
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nInstall with:")
    print("  pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"")
    print("  pip install --no-deps trl peft accelerate bitsandbytes")
    print("  pip install datasets transformers")
    sys.exit(1)

print()

# Load base model
print(f"[2/6] Loading base model: {BASE_MODEL}")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

print()

# Add LoRA adapters
print("[3/6] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("✓ LoRA adapters added")
print()

# Load dataset
print("[4/6] Loading training data...")
dataset = load_dataset("json", data_files={
    "train": str(DATASET_PATH / "train.jsonl"),
})["train"]
print(f"✓ Loaded {len(dataset)} samples")
print()

# Format samples for training
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

    return {"text": prompt}

dataset = dataset.map(format_sample)

# Training configuration
print("[5/6] Training model...")
print(f"  Max steps: 500")
print(f"  Batch size: 2 (effective: 8 with gradient accumulation)")
print(f"  Learning rate: 2e-4")
print(f"  Optimizer: adamw_8bit")
print()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        report_to="none",  # Disable wandb
    ),
)

# Train
start_time = datetime.now()
print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

trainer.train()

duration = (datetime.now() - start_time).total_seconds()
print()
print(f"✓ Training completed in {duration:.0f}s ({duration/60:.1f} minutes)")
print()

# Save model
print("[6/6] Saving model...")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"✓ Model saved to: {OUTPUT_DIR}")

# Save metadata
metadata = {
    "name": MODEL_NAME,
    "base_model": BASE_MODEL,
    "dataset": str(DATASET_PATH),
    "dataset_name": DATASET_NAME,
    "quality_tag": QUALITY_TAG,
    "created": datetime.now().isoformat(),
    "training_samples": len(dataset),
    "training_duration_seconds": duration,
    "training_duration_minutes": duration / 60,
    "max_steps": 500,
    "model_size": MODEL_SIZE,
}

metadata_file = OUTPUT_DIR / "metadata.json"
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Metadata saved to: {metadata_file}")
print()
print("=" * 80)
print("Training Complete!")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Location: {OUTPUT_DIR}")
print()
