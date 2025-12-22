#!/usr/bin/env python3
"""
Train a fine-tuned model on Mac M1 using MPS (Metal)
Usage: python train_model_mac.py <dataset_path> [base_model] [quality_tag]
"""

import os
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Check MPS
if not torch.backends.mps.is_available():
    print("✗ MPS not available. Training will use CPU (very slow).")
    device = "cpu"
else:
    print(f"✓ MPS available, using Metal GPU acceleration")
    device = "mps"

# Parse arguments
if len(sys.argv) < 2:
    print("Usage: python train_model_mac.py <dataset_path> [base_model] [quality_tag]")
    print()
    print("Example:")
    print(f"  python train_model_mac.py ~/.context/training/datasets/alttp_yaze_full_1000_20251221_195746")
    sys.exit(1)

DATASET_PATH = Path(sys.argv[1]).expanduser()
BASE_MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen2.5-Coder-1.5B"
QUALITY_TAG = sys.argv[3] if len(sys.argv) > 3 else "alpha"

# Extract model size
import re
match = re.search(r'(\d+\.?\d*)B', BASE_MODEL, re.IGNORECASE)
MODEL_SIZE = match.group(0).lower() if match else "1.5b"

# Generate model name
DATE = datetime.now().strftime("%Y%m%d")
MODEL_NAME = f"hafs-asm-{MODEL_SIZE}-{DATE}-{QUALITY_TAG}"

# Output directory
OUTPUT_DIR = Path.home() / "Code/hafs/models" / MODEL_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Training Configuration")
print("=" * 80)
print(f"Dataset:      {DATASET_PATH}")
print(f"Base Model:   {BASE_MODEL}")
print(f"Output Model: {MODEL_NAME}")
print(f"Output Dir:   {OUTPUT_DIR}")
print(f"Quality Tag:  {QUALITY_TAG}")
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
    print("\nInstall with:")
    print("  pip install transformers peft datasets accelerate")
    sys.exit(1)

print()

# Load tokenizer
print(f"[2/6] Loading tokenizer from: {BASE_MODEL}")
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
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=None,  # We'll move to device manually
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

# Format and tokenize samples
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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=device == "mps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    use_mps_device=(device == "mps"),
)

print(f"  Epochs: 1")
print(f"  Batch size: 1 (effective: 8 with gradient accumulation)")
print(f"  Learning rate: 2e-4")
print(f"  Device: {device}")
print()

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train
print("[6/6] Training model...")
start_time = datetime.now()
print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Estimated time: ~1-2 hours on M1 Mac")
print()

try:
    trainer.train()
    duration = (datetime.now() - start_time).total_seconds()
    print()
    print(f"✓ Training completed in {duration:.0f}s ({duration/60:.1f} minutes)")
except KeyboardInterrupt:
    print("\n✗ Training interrupted by user")
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
    "base_model": BASE_MODEL,
    "dataset": str(DATASET_PATH),
    "quality_tag": QUALITY_TAG,
    "created": datetime.now().isoformat(),
    "training_samples": len(dataset),
    "training_duration_seconds": duration,
    "training_duration_minutes": duration / 60,
    "device": device,
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
print("To use this model:")
print(f"  1. Test: python -c 'from transformers import AutoModel; AutoModel.from_pretrained(\"{OUTPUT_DIR}\")'")
print(f"  2. Use with hafs-lsp (configure in config/lsp.toml)")
print()
