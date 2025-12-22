#!/bin/bash
# Create uniquely named fine-tuned model from training data

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <base-model> <dataset-path> [quality-tag]"
    echo ""
    echo "Examples:"
    echo "  $0 qwen2.5-coder:1.5b ~/.context/training/datasets/latest gold"
    echo "  $0 qwen2.5-coder:7b ~/.context/training/datasets/my_dataset silver"
    echo ""
    echo "Quality tags: gold, silver, bronze, alpha, beta"
    echo ""
    exit 1
fi

BASE_MODEL="$1"
DATASET="$2"
QUALITY="${3:-alpha}"

# Extract model size from base model name
MODEL_SIZE=$(echo "$BASE_MODEL" | grep -oE '[0-9.]+b')

# Generate unique name: hafs-asm-{size}-{date}-{quality}
DATE=$(date +%Y%m%d)
MODEL_NAME="hafs-asm-${MODEL_SIZE}-${DATE}-${QUALITY}"

echo "========================================================================"
echo "Fine-tuning Model"
echo "========================================================================"
echo ""
echo "Base Model:    $BASE_MODEL"
echo "Dataset:       $DATASET"
echo "Output Name:   $MODEL_NAME"
echo "Quality Tag:   $QUALITY"
echo ""

# Check dataset exists
if [ ! -d "$DATASET" ]; then
    echo "✗ Dataset not found: $DATASET"
    exit 1
fi

# Count samples
train_samples=$(wc -l < "$DATASET/train.jsonl" 2>/dev/null || echo 0)
echo "Training samples: $train_samples"

if [ "$train_samples" -eq 0 ]; then
    echo "✗ No training samples found in dataset"
    exit 1
fi

echo ""
echo "Starting fine-tuning..."
echo ""

# Fine-tune with Unsloth
cd ~/Code/hafs

PYTHONPATH=src .venv/bin/python -c "
import sys
from pathlib import Path
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

print('[1/5] Loading base model: $BASE_MODEL')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='$BASE_MODEL',
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

print('[2/5] Adding LoRA adapters...')

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=3407,
)

print('[3/5] Loading dataset from $DATASET')

dataset = load_dataset('json', data_files={
    'train': str(Path('$DATASET') / 'train.jsonl'),
})['train']

print(f'Loaded {len(dataset)} samples')

def format_sample(example):
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text:
        prompt = f'''Below is an instruction for 65816 assembly, paired with context. Write a response.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}'''
    else:
        prompt = f'''Below is an instruction for 65816 assembly. Write a response.

### Instruction:
{instruction}

### Response:
{output}'''

    return {'text': prompt}

dataset = dataset.map(format_sample)

print('[4/5] Training...')

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field='text',
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir=f'./models/$MODEL_NAME',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=500,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        optim='adamw_8bit',
        save_strategy='steps',
        save_steps=100,
    ),
)

trainer.train()

print('[5/5] Saving model...')

output_dir = Path.home() / 'Code/hafs/models' / '$MODEL_NAME'
output_dir.mkdir(parents=True, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f'✓ Model saved to: {output_dir}')

# Create metadata file
import json
metadata = {
    'name': '$MODEL_NAME',
    'base_model': '$BASE_MODEL',
    'dataset': '$DATASET',
    'quality': '$QUALITY',
    'created': '$(date -u +"%Y-%m-%dT%H:%M:%SZ")',
    'samples': $train_samples,
    'size': '$MODEL_SIZE',
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print('✓ Metadata saved')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "Fine-tuning Complete!"
    echo "========================================================================"
    echo ""
    echo "Model: $MODEL_NAME"
    echo "Location: ~/Code/hafs/models/$MODEL_NAME"
    echo ""
    echo "To use this model:"
    echo "  ./scripts/hafs_lsp_control.sh custom ${MODEL_SIZE%b} ~/Code/hafs/models/$MODEL_NAME"
    echo ""
    echo "Or manually edit config/lsp.toml:"
    echo "  custom_fast = \"~/Code/hafs/models/$MODEL_NAME\""
    echo ""
else
    echo ""
    echo "✗ Fine-tuning failed"
    exit 1
fi
