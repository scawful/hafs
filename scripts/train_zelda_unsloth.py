#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Zelda-themed models with Unsloth on RTX 5060 Ti
Usage: python train_zelda_unsloth.py <zelda_name> <dataset_path> [base_model]
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

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
    print("Usage: python train_zelda_unsloth.py <zelda_name> <dataset_path> [base_model]")
    print()
    print("Available Zelda names:")
    for name, desc in ZELDA_NAMES.items():
        print(f"  {name:20s} - {desc}")
    print()
    print("Example:")
    print("  python train_zelda_unsloth.py master-sword D:/.context/training/datasets/alttp_yaze_full_1000_20251221_195746")
    sys.exit(1)

ZELDA_NAME = sys.argv[1]
DATASET_PATH = Path(sys.argv[2])
BASE_MODEL = sys.argv[3] if len(sys.argv) > 3 else "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"

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

print("=" * 80)
print(f"Training Zelda Model with Unsloth: {ZELDA_NAME.upper()}")
print("=" * 80)
print(f"Description:  {ZELDA_NAMES[ZELDA_NAME]}")
print(f"Dataset:      {DATASET_PATH}")
print(f"Base Model:   {BASE_MODEL}")
print(f"Output Model: {MODEL_NAME}")
print(f"Output Dir:   {OUTPUT_DIR}")
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

# Step 1: Load model with Unsloth
print("[1/6] Loading model with Unsloth (4-bit quantization)...")
try:
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    print("✓ Model loaded with 4-bit quantization")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

print()

# Step 2: Add LoRA adapters
print("[2/6] Adding LoRA adapters...")
try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Optimized for speed
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    print("✓ LoRA adapters configured")
except Exception as e:
    print(f"✗ Failed to add LoRA: {e}")
    sys.exit(1)

print()

# Step 3: Load dataset
print("[3/6] Loading training data...")
dataset = load_dataset("json", data_files={
    "train": str(DATASET_PATH / "train.jsonl"),
})["train"]
print(f"✓ Loaded {len(dataset)} samples")
print()

# Format function for 65816 assembly instruction tuning
def format_sample(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        return f"""Below is an instruction for 65816 assembly, paired with context. Write a response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        return f"""Below is an instruction for 65816 assembly. Write a response.

### Instruction:
{instruction}

### Response:
{output}"""

# Step 4: Configure training
print("[4/6] Configuring training...")
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,  # RTX 5060 Ti can handle 2
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",  # 8-bit optimizer for memory efficiency
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_dir=str(OUTPUT_DIR / "logs"),
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    formatting_func=format_sample,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

print(f"  Epochs: 1")
print(f"  Batch size: 2 (effective: 8 with gradient accumulation)")
print(f"  Learning rate: 2e-4")
print(f"  Optimizer: adamw_8bit")
print()

# Step 5: Train
print(f"[5/6] Training {ZELDA_NAME.upper()} with Unsloth...")
start_time = datetime.now()
print(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Estimated time: ~15-30 minutes on RTX 5060 Ti")
print()

try:
    trainer_stats = trainer.train()
    duration = (datetime.now() - start_time).total_seconds()
    print()
    print("=" * 80)
    print(f"✓ Training completed in {duration:.0f}s ({duration/60:.1f} minutes)")
    print(f"  Samples/second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
except KeyboardInterrupt:
    print("\n✗ Training interrupted")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 6: Save model
print("[6/6] Saving model...")
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"✓ LoRA adapters saved to: {OUTPUT_DIR}")

# Save merged model (optional, for easier inference)
print("\nSaving merged model (16-bit)...")
merged_dir = OUTPUT_DIR.parent / f"{MODEL_NAME}-merged"
model.save_pretrained_merged(
    str(merged_dir),
    tokenizer,
    save_method="merged_16bit",
)
print(f"✓ Merged model saved to: {merged_dir}")

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
    "samples_per_second": trainer_stats.metrics['train_samples_per_second'],
    "model_size": MODEL_SIZE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
    "unsloth_version": True,
    "quantization": "4-bit",
    "lora_rank": 16,
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
print(f"LoRA adapters: {OUTPUT_DIR}")
print(f"Merged model: {merged_dir}")
print(f"Description: {ZELDA_NAMES[ZELDA_NAME]}")
print()
