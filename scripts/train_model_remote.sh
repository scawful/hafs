#!/bin/bash
# Train model on Windows GPU server (medical-mechanica)

set -e

# Configuration
WINDOWS_HOST="medical-mechanica"
DATASET_NAME="${1:-alttp_yaze_full_1000_20251221_195746}"
BASE_MODEL="${2:-unsloth/qwen2.5-coder-1.5b-bnb-4bit}"
QUALITY_TAG="${3:-alpha}"

# Extract model size for naming
MODEL_SIZE=$(echo "$BASE_MODEL" | grep -oE '[0-9.]+b' | head -1)
if [ -z "$MODEL_SIZE" ]; then
    MODEL_SIZE="1.5b"  # Default
fi

DATE=$(date +%Y%m%d)
MODEL_NAME="hafs-asm-${MODEL_SIZE}-${DATE}-${QUALITY_TAG}"

echo "========================================================================"
echo "Training Model on GPU Server"
echo "========================================================================"
echo ""
echo "GPU Server:    $WINDOWS_HOST"
echo "Dataset:       $DATASET_NAME"
echo "Base Model:    $BASE_MODEL"
echo "Output Model:  $MODEL_NAME"
echo "Quality Tag:   $QUALITY_TAG"
echo ""

# Check GPU server
echo "[1/5] Checking GPU server..."
if ! ssh -o ConnectTimeout=5 "$WINDOWS_HOST" "echo OK" > /dev/null 2>&1; then
    echo "✗ Cannot connect to $WINDOWS_HOST"
    exit 1
fi
echo "✓ Connected to GPU server"

# Check dataset exists
echo ""
echo "[2/5] Verifying dataset..."
DATASET_PATH="D:/.context/training/datasets/$DATASET_NAME"
ssh "$WINDOWS_HOST" << EOF
if [ ! -f "$DATASET_PATH/train.jsonl" ]; then
    echo "✗ Dataset not found: $DATASET_PATH"
    exit 1
fi

train_count=\$(wc -l < "$DATASET_PATH/train.jsonl")
echo "✓ Dataset found: \$train_count training samples"
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "Dataset not found on Windows. Syncing..."
    rsync -avz ~/.context/training/datasets/alttp_yaze_full_1000_20251221_195746_20251221_224740/ \
      /Users/scawful/Mounts/mm-d/.context/training/datasets/$DATASET_NAME/ \
      --exclude '__pycache__' --exclude '.DS_Store'
    echo "✓ Dataset synced"
fi

# Create training script on Windows
echo ""
echo "[3/5] Creating training script..."
ssh "$WINDOWS_HOST" << 'SCRIPT_EOF'
cat > D:/train_model.py << 'PYTHON_EOF'
import os
import json
from pathlib import Path
from datetime import datetime

# Set up paths
DATASET_PATH = Path("D:/.context/training/datasets/DATASET_NAME_PLACEHOLDER")
OUTPUT_DIR = Path("D:/.context/training/models/MODEL_NAME_PLACEHOLDER")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Training Configuration")
print("=" * 80)
print(f"Dataset: {DATASET_PATH}")
print(f"Output: {OUTPUT_DIR}")
print()

# Load dataset stats
stats_file = DATASET_PATH / "stats.json"
if stats_file.exists():
    with open(stats_file) as f:
        stats = json.load(f)
        print(f"Training samples: {stats.get('final_count', 'unknown')}")
        print(f"Average quality: {stats.get('quality_scores', {}).get('average', 'unknown'):.3f}")
        print()

# Import Unsloth
print("[1/6] Loading Unsloth...")
try:
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import load_dataset
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nInstall with:")
    print("  pip install unsloth transformers trl datasets")
    exit(1)

print()

# Load base model
print("[2/6] Loading base model: BASE_MODEL_PLACEHOLDER")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="BASE_MODEL_PLACEHOLDER",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
print("✓ Model loaded")
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

# Format samples
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

# Training
print("[5/6] Training model...")
print(f"  Max steps: 500")
print(f"  Batch size: 2 (effective: 8 with grad accumulation)")
print(f"  Learning rate: 2e-4")
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
    ),
)

# Start training
start_time = datetime.now()
trainer.train()
duration = (datetime.now() - start_time).total_seconds()

print()
print(f"✓ Training completed in {duration:.0f}s ({duration/60:.1f}m)")
print()

# Save model
print("[6/6] Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ Model saved to: {OUTPUT_DIR}")

# Save metadata
metadata = {
    "name": "MODEL_NAME_PLACEHOLDER",
    "base_model": "BASE_MODEL_PLACEHOLDER",
    "dataset": str(DATASET_PATH),
    "quality_tag": "QUALITY_TAG_PLACEHOLDER",
    "created": datetime.now().isoformat(),
    "training_samples": len(dataset),
    "training_duration_seconds": duration,
    "max_steps": 500,
}

with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata saved")
print()
print("=" * 80)
print("Training Complete!")
print("=" * 80)
PYTHON_EOF

echo "✓ Training script created"
SCRIPT_EOF

# Replace placeholders
echo ""
echo "[4/5] Configuring training parameters..."
ssh "$WINDOWS_HOST" "sed -i 's/DATASET_NAME_PLACEHOLDER/$DATASET_NAME/g' D:/train_model.py"
ssh "$WINDOWS_HOST" "sed -i 's/MODEL_NAME_PLACEHOLDER/$MODEL_NAME/g' D:/train_model.py"
ssh "$WINDOWS_HOST" "sed -i 's|BASE_MODEL_PLACEHOLDER|$BASE_MODEL|g' D:/train_model.py"
ssh "$WINDOWS_HOST" "sed -i 's/QUALITY_TAG_PLACEHOLDER/$QUALITY_TAG/g' D:/train_model.py"
echo "✓ Parameters configured"

# Run training
echo ""
echo "[5/5] Starting training on GPU..."
echo "This will take approximately 20-30 minutes for 500 steps..."
echo ""

ssh "$WINDOWS_HOST" "python D:/train_model.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Training Complete!"
    echo "========================================================================"
    echo ""
    echo "Model: $MODEL_NAME"
    echo "Location: D:/.context/training/models/$MODEL_NAME"
    echo ""
    echo "To use this model with hafs-lsp:"
    echo "  1. Copy model to Mac (if needed)"
    echo "  2. Use hafs-lsp-control.sh to configure"
    echo ""
else
    echo ""
    echo "✗ Training failed. Check logs above for errors."
    exit 1
fi
