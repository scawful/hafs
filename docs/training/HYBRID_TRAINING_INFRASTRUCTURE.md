# Hybrid Training Infrastructure (Mac + Windows GPU)

**Last updated:** 2025-12-21
**Status:** 🚧 In Development

## Overview

Hybrid training system using:
- **Mac (Mini M4)**: Gemini 2.0 Flash for dataset generation (parallel campaigns)
- **Windows (medical-mechanica)**: RTX 5060 Ti 16GB for model training with Unsloth

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Mac Mini M4                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Training Data Generation (Gemini 2.0 Flash)      │   │
│  │  ────────────────────────────────────────────────  │   │
│  │  • Parallel Generator Pools (8-16 concurrent)     │   │
│  │  • Quality Pipeline (validators, feedback loop)    │   │
│  │  • Active Learning (coverage-driven sampling)      │   │
│  │  • Checkpoint/Resume (fault-tolerant)              │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: High-quality JSONL datasets                       │
│  → ~/.context/training/datasets/                           │
│     - alttp_asm_full_24000_asm.jsonl                      │
│     - oracle_rom_hack_7000_oracle.jsonl                   │
│     - yaze_cpp_analysis_12000_cpp.jsonl                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ rsync/git sync
┌─────────────────────────────────────────────────────────────┐
│              medical-mechanica (Windows GPU)                │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Model Training (Unsloth + QLoRA)                 │   │
│  │  ────────────────────────────────────────────────  │   │
│  │  • RTX 5060 Ti 16GB VRAM                          │   │
│  │  • CUDA 11.2, PyTorch 2.x                         │   │
│  │  • Unsloth 4-bit quantization                     │   │
│  │  • Flash Attention 2                              │   │
│  │  • Gradient checkpointing                         │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  Output: Fine-tuned models                                 │
│  → D:/.context/training/models/                            │
│     - oracle-rauru-assembler-v1/                          │
│     - oracle-yaze-expert-v1/                              │
└─────────────────────────────────────────────────────────────┘
                            ↓ model deployment
                    Inference (Mac/Windows)
```

## Data Generation Pipeline (Mac)

### Current Status (2025-12-21)

**Active Campaign:** 34,500 sample generation (PID 13516)
- **Target:** 24K ALTTP ASM, 7K YAZE C++, 3.5K Oracle
- **Quality:** 62.7% pass rate (up from 1% after improvements)
- **Generator:** Gemini 2.0 Flash (coding tier)
- **Validation:** ASM validator + quality pipeline
- **Progress:** Managed by autonomous agent

### Quality Metrics

Domain-specific thresholds (from `quality.py:623-631`):
```python
DOMAIN_THRESHOLDS = {
    "asm": 0.4,        # ASM is hard - lower threshold
    "gigaleak": 0.45,  # Original source
    "oracle": 0.4,     # ROM hack - lower
    "yaze": 0.5,       # C++ code - medium
    "cpp": 0.5,
    "errors": 0.3,
    "text": 0.6,
}
```

**Quality Components:**
1. **Diversity Score** (0-1): Embedding distance from existing samples
2. **KG Consistency** (0-1): Entity validation against knowledge graph
3. **Hallucination Risk** (0-1): Pattern detection + LLM verification
4. **Semantic Coherence** (0-1): Instruction/output alignment

### Parallelization Strategy

**Mac Capabilities:**
- 16 GB unified memory (shared between CPU and LLM context)
- 10-core CPU (4 performance + 6 efficiency)
- Network: Gigabit Ethernet to medical-mechanica

**Generator Configuration:**
```python
# From parallel_generator.py
max_concurrent_workers = 8-16  # Depends on memory pressure
batch_size = 32                # Samples per batch
checkpoint_interval = 100      # Save every 100 samples
```

**Optimization:**
- Use Gemini Flash (fastest, cheapest, good quality for code)
- Batch API calls with async/await
- Stream results to disk (don't hold in memory)
- Quality filtering AFTER generation (not during)
- Checkpoint progress for resume on crash

### Data Transfer (Mac → Windows)

**Method 1: Direct Mount (Preferred)**
```bash
# Mac has medical-mechanica drives mounted
~/Mounts/mm-c/  # C: drive (code, venv)
~/Mounts/mm-d/  # D: drive (data, training)

# Copy datasets directly
cp ~/.context/training/datasets/*.jsonl ~/Mounts/mm-d/.context/training/datasets/
```

**Method 2: rsync over SSH**
```bash
# Sync training datasets
rsync -avz --progress \
  ~/.context/training/datasets/ \
  medical-mechanica:/D:/.context/training/datasets/

# Verify transfer
ssh medical-mechanica 'dir D:\.context\training\datasets\*.jsonl'
```

**Method 3: Git LFS (Future)**
```bash
# Track large datasets with Git LFS
git lfs track "*.jsonl"
git lfs push origin master
```

## Model Training Pipeline (Windows)

### Hardware Specifications

**GPU:** NVIDIA GeForce RTX 5060 Ti
- **VRAM:** 16 GB GDDR6
- **CUDA Cores:** 5632
- **Tensor Cores:** 176 (4th gen)
- **CUDA Version:** 11.2
- **Driver:** Latest stable

**Storage:**
- **C:/** 149 GB free (code, venv)
- **D:/** 1.56 TB free (training data, models, checkpoints)

**Python Environment:**
- Python 3.14.0
- PyTorch 2.x with CUDA 11.2
- Unsloth (to be installed)
- Flash Attention 2
- BitsAndBytes for 4-bit quantization

### Training Configuration

**Recommended Settings (16GB VRAM):**
```python
# Training hyperparameters
model_name = "unsloth/Qwen2.5-7B-bnb-4bit"  # 4-bit quantized base
max_seq_length = 2048                       # Context window
dtype = torch.bfloat16                      # Mixed precision

# LoRA configuration
lora_r = 16                                 # Rank
lora_alpha = 16                             # Scaling
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# Training arguments
per_device_train_batch_size = 2             # Fits in 16GB with gradient accumulation
gradient_accumulation_steps = 4             # Effective batch size = 8
num_train_epochs = 3
learning_rate = 2e-4
warmup_steps = 100
logging_steps = 10
save_steps = 500
fp16 = False
bf16 = True
optim = "adamw_8bit"                        # Memory-efficient optimizer
gradient_checkpointing = True
max_grad_norm = 1.0
```

**Memory Optimization:**
- 4-bit quantization saves ~4x memory
- Gradient checkpointing: trade compute for memory
- Batch size = 2 with accumulation = 4 (effective batch 8)
- Flash Attention 2: faster, less memory

**Expected Training Times:**
| Dataset Size | Epochs | Approximate Time |
|--------------|--------|------------------|
| 7,000 samples | 3 | ~4-6 hours |
| 12,000 samples | 3 | ~8-10 hours |
| 24,000 samples | 3 | ~16-20 hours |

### Model Training Scripts

**Location:** `scripts/train_on_windows.py` (to be created)

```python
#!/usr/bin/env python3
"""Train models on Windows GPU using Unsloth."""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def main():
    # Load 4-bit quantized model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-bnb-4bit",
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    # Configure LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing=True,
    )

    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": "D:/.context/training/datasets/oracle_rom_hack_7000_oracle.jsonl"
    })

    # Training arguments
    training_args = TrainingArguments(
        output_dir="D:/.context/training/models/oracle-rauru-assembler-v1",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=500,
        optim="adamw_8bit",
        warmup_steps=100,
        max_grad_norm=1.0,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=training_args,
        max_seq_length=2048,
    )

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained("D:/.context/training/models/oracle-rauru-assembler-v1")
    tokenizer.save_pretrained("D:/.context/training/models/oracle-rauru-assembler-v1")

if __name__ == "__main__":
    main()
```

### Monitoring Training

**Windows Task Manager:**
```powershell
# Monitor GPU usage
nvidia-smi -l 1  # Update every second

# Monitor process
Get-Process -Name python | Select-Object CPU, WorkingSet, Path
```

**Training Logs:**
```
D:/.context/logs/training/
├── oracle-rauru-v1.log
├── oracle-yaze-v1.log
└── tensorboard/
```

**TensorBoard (optional):**
```bash
tensorboard --logdir D:/.context/training/models/oracle-rauru-assembler-v1
# View at http://localhost:6006
```

## Hybrid Workflow

### Phase 1: Dataset Generation (Mac)

```bash
# Launch training campaign
cd ~/Code/hafs
hafs agents start-campaign \
  --target 24000 \
  --domains asm,oracle,yaze \
  --generator gemini-2.0-flash \
  --quality-threshold 0.4

# Monitor progress
hafs logs campaign --follow

# Check quality metrics
hafs agents quality-report
```

**Output:**
- `~/.context/training/datasets/alttp_asm_full_24000_asm.jsonl`
- `~/.context/training/datasets/oracle_rom_hack_7000_oracle.jsonl`
- `~/.context/training/datasets/yaze_cpp_12000_cpp.jsonl`

### Phase 2: Dataset Transfer

```bash
# Copy datasets to Windows
cp ~/.context/training/datasets/*.jsonl ~/Mounts/mm-d/.context/training/datasets/

# Verify on Windows
ssh medical-mechanica 'powershell -Command "Get-ChildItem D:\.context\training\datasets\*.jsonl | Select-Object Name, Length, LastWriteTime"'
```

### Phase 3: Model Training (Windows)

```bash
# SSH into Windows
ssh medical-mechanica

# Activate venv
cd C:\hafs
.venv\Scripts\activate

# Install training dependencies (first time only)
pip install unsloth transformers trl datasets accelerate

# Start training
python scripts/train_on_windows.py \
  --dataset D:/.context/training/datasets/oracle_rom_hack_7000_oracle.jsonl \
  --output D:/.context/training/models/oracle-rauru-assembler-v1 \
  --epochs 3 \
  --batch-size 2

# Monitor (separate terminal)
nvidia-smi -l 1
```

### Phase 4: Model Deployment

```bash
# Test inference on Windows
python scripts/test_model.py \
  --model D:/.context/training/models/oracle-rauru-assembler-v1 \
  --prompt "Write a hook that redirects sprite loading to bank $32"

# Copy model back to Mac (optional)
rsync -avz --progress \
  medical-mechanica:/D:/.context/training/models/oracle-rauru-assembler-v1/ \
  ~/.context/training/models/oracle-rauru-assembler-v1/
```

## Automation

### Background Agents (Windows)

**cataloger** (every 12 hours):
- Monitors `D:/.context/training/datasets/`
- Generates metadata (sample count, size, checksum)
- Tracks dataset growth

**training-monitor** (to be created):
- Monitors active training runs
- Tracks GPU utilization
- Alerts on completion or errors
- Exports metrics to logs

### Scheduled Tasks

**Daily Dataset Sync (1 AM):**
```powershell
# Pull latest datasets from Mac
rsync -avz mac-mini:~/.context/training/datasets/ D:/.context/training/datasets/
```

**Daily Model Backup (3 AM):**
```powershell
# Backup trained models
robocopy D:\.context\training\models\ D:\.backups\models\ /MIR /Z
```

## Quality Improvements

### Current Issues (from validation pilot)

1. **ASM Quality:** 0.4 threshold (40% pass) - needs improvement
2. **Oracle Quality:** 0.4 threshold - needs better prompts
3. **Coherence:** Code samples need better instruction/output alignment

### Improvement Strategies

**1. Enhanced Prompts (in progress)**
- Add reference examples to generator prompts
- Specify exact output format with code blocks
- Include quality anti-patterns to avoid
- Domain-specific validation criteria

**2. Iterative Refinement**
- Use rejected samples to improve prompts
- Analyze failure patterns from quality.py feedback
- A/B test different prompt variations
- Gradually raise quality thresholds as prompts improve

**3. Active Learning**
- Identify sparse regions in embedding space
- Generate targeted samples for underrepresented patterns
- Balance domain coverage (ASM vs Oracle vs YAZE)

**4. Human-in-the-Loop**
- Manual review of edge cases
- Curate "golden" examples
- Validate model outputs before deployment

## Cost Optimization

### Gemini 2.0 Flash Pricing
- **Input:** $0.075 per 1M tokens (~$0.000075 per 1K)
- **Output:** $0.30 per 1M tokens (~$0.0003 per 1K)
- **Batch (50% discount):** If using batch API

**Estimated Costs:**
| Campaign | Samples | Tokens (est) | Cost (est) |
|----------|---------|--------------|------------|
| ASM 24K | 24,000 | ~60M | ~$18-24 |
| Oracle 7K | 7,000 | ~21M | ~$6-8 |
| YAZE 12K | 12,000 | ~30M | ~$9-12 |
| **Total** | **43,000** | **111M** | **~$33-44** |

**Optimization:**
- Use batch API for 50% discount
- Cache common prompts
- Reuse embeddings across runs
- Filter low-quality samples early (save generation costs)

### GPU Training (Free)

medical-mechanica GPU is local hardware - no cloud costs.

## Future Enhancements

### Distributed Training (Multi-GPU)

If adding more GPUs:
```python
# DeepSpeed config for multi-GPU
from accelerate import Accelerator

accelerator = Accelerator()
model = accelerator.prepare(model)
```

### Model Merging

Merge LoRA adapters for different domains:
```python
# Merge ASM and Oracle adapters
from unsloth import merge_lora_adapters

merged_model = merge_lora_adapters(
    base_model="unsloth/Qwen2.5-7B-bnb-4bit",
    adapters=[
        "oracle-rauru-assembler-v1",
        "oracle-yaze-expert-v1"
    ],
    weights=[0.6, 0.4]  # 60% ASM, 40% YAZE
)
```

### Continuous Training

- Incrementally update models with new datasets
- Resume from checkpoints
- A/B test model versions

## Troubleshooting

### Mac: Generation Stalled

```bash
# Check campaign status
hafs agents status

# View logs
hafs logs campaign --tail 100

# Resume from checkpoint
hafs agents resume-campaign --checkpoint ~/.context/training/checkpoints/latest.json
```

### Windows: CUDA Out of Memory

**Solutions:**
1. Reduce batch size: `per_device_train_batch_size = 1`
2. Increase gradient accumulation: `gradient_accumulation_steps = 8`
3. Enable gradient checkpointing: `gradient_checkpointing = True`
4. Use smaller model: `Qwen2.5-1.5B` instead of `Qwen2.5-7B`
5. Reduce sequence length: `max_seq_length = 1024`

### Dataset Transfer Failed

```bash
# Check network connectivity
ping medical-mechanica

# Check mount status
ls ~/Mounts/mm-d/.context/training/

# Remount if needed
mount | grep medical-mechanica
# Remount via Finder → Go → Connect to Server
```

## References

- Quality pipeline: `src/agents/training/quality.py`
- ASM generator: `src/agents/training/generators/asm_generator.py`
- Oracle generator: `src/agents/training/generators/oracle_generator.py`
- Parallel generation: `src/agents/training/parallel_generator.py`
- Unsloth docs: https://github.com/unslothai/unsloth
- Training guide: `docs/training/TRAINING_PREPARATION.md`
