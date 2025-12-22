# Configuration-Based Training System

**Created**: 2025-12-22
**Status**: ✓ IMPLEMENTED

Unified configuration-driven training system for Oracle experts. No more hardcoded scripts!

---

## Overview

Instead of scattered training scripts with hardcoded settings, all training configuration is centralized in `config/training.toml`. A single unified trainer reads this config and handles training for any Oracle expert on any hardware.

### Benefits

- **Single source of truth** - All settings in one file
- **Hardware profiles** - Easy switching between Mac MPS, cloud GPUs, etc.
- **Expert definitions** - Oracle experts with their specializations
- **Reusable presets** - Common training configurations
- **No hardcoding** - Change settings without editing code
- **Metadata tracking** - Automatic logging of training parameters

---

## Quick Start

### List Available Presets

```bash
python scripts/train.py --list-presets
```

### Train Oracle Expert

```bash
# Train oracle-rauru-assembler on Mac MPS
python scripts/train.py oracle-rauru-mac

# Train on cloud GPU
python scripts/train.py oracle-rauru-cloud
```

That's it! The config handles everything else.

---

## Configuration Structure

### Hardware Profiles (`config/training.toml`)

Define hardware capabilities and constraints:

```toml
[hardware.mac-mps]
name = "Mac M1/M2/M3 with MPS"
device = "mps"
available_memory_gb = 20
max_batch_size = 1
max_sequence_length = 1024
supports_fp16 = true
supports_bf16 = false
supports_gradient_checkpointing = false  # Incompatible with LoRA on MPS
max_lora_rank = 8
recommended_gradient_accumulation = 2
```

Available profiles:
- `mac-mps` - Mac M1/M2/M3
- `windows-rtx-5060` - RTX 5060 Ti (currently unsupported by PyTorch)
- `cloud-rtx-4090` - Cloud RTX 4090
- `cloud-a100-40gb` - Cloud A100

### Oracle Experts

Define specialized AI agents:

```toml
[experts.oracle-rauru-assembler]
display_name = "Oracle: Rauru Assembler"
role = "asm"
group = "rom-tooling"
base_model = "Qwen/Qwen2.5-Coder-1.5B"
context_window = 8192
specialization = "65816 assembly, ALTTP routines, ROM patching"
```

### LoRA Configurations

Different LoRA settings for different constraints:

```toml
[lora.minimal]
# For memory-constrained systems (Mac MPS)
r = 8
lora_alpha = 8
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

[lora.standard]
# Balanced configuration
r = 16
lora_alpha = 16
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

[lora.full]
# Maximum quality (requires more VRAM)
r = 32
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Hyperparameter Presets

Common training configurations:

```toml
[hyperparameters.default]
num_epochs = 1
learning_rate = 2e-4
warmup_steps = 50
weight_decay = 0.01
lr_scheduler = "linear"

[hyperparameters.production]
num_epochs = 3
learning_rate = 2e-4
warmup_steps = 100
```

### Training Presets

Complete training configurations combining all components:

```toml
[presets.oracle-rauru-mac]
expert = "oracle-rauru-assembler"
hardware = "mac-mps"
dataset = "alttp_yaze_full"
hyperparameters = "default"
lora = "minimal"
sequence_length = 1024
batch_size = 1
gradient_accumulation = 2
```

---

## Usage Guide

### Listing Options

```bash
# List all training presets
python scripts/train.py --list-presets

# List all Oracle experts
python scripts/train.py --list-experts

# List hardware profiles
python scripts/train.py --list-hardware
```

### Training Examples

**Mac M1 Training:**
```bash
# Use the mac preset (automatically configured for MPS constraints)
python scripts/train.py oracle-rauru-mac
```

**Cloud GPU Training:**
```bash
# Use cloud preset (higher batch size, full LoRA rank)
python scripts/train.py oracle-rauru-cloud
```

**Verbose Logging:**
```bash
python scripts/train.py oracle-rauru-mac --verbose
```

### Custom Configuration

**Create Custom Preset:**

Edit `config/training.toml`:

```toml
[presets.my-custom-training]
expert = "oracle-sheik-debugger"
hardware = "mac-mps"
dataset = "alttp_yaze_full"
hyperparameters = "production"  # 3 epochs instead of 1
lora = "standard"  # Try higher rank if it fits
sequence_length = 1024
batch_size = 1
gradient_accumulation = 4  # Larger effective batch
```

Then train:
```bash
python scripts/train.py my-custom-training
```

---

## Adding New Components

### Add New Hardware Profile

Edit `config/training.toml`:

```toml
[hardware.my-workstation]
name = "My RTX 4090 Workstation"
device = "cuda"
available_memory_gb = 24
max_batch_size = 4
max_sequence_length = 4096
supports_fp16 = true
supports_bf16 = true
supports_gradient_checkpointing = true
max_lora_rank = 32
recommended_gradient_accumulation = 2
```

### Add New Oracle Expert

```toml
[experts.oracle-purah-profiler]
display_name = "Oracle: Purah Profiler"
role = "performance"
group = "rom-tooling"
base_model = "Qwen/Qwen2.5-Coder-1.5B"
context_window = 8192
specialization = "Performance tuning and WRAM/VRAM sanity"
prompt_template = """Below is a performance analysis task{context}. Provide optimization recommendations.

### Task:
{instruction}
{input_section}
### Analysis:
{output}"""
```

### Add New Dataset

```toml
[datasets.my_dataset]
path = "~/.context/training/datasets/my_dataset_20251222"
description = "My custom training dataset"
train_split = "train.jsonl"
val_split = "val.jsonl"
test_split = "test.jsonl"
quality_threshold = 0.6
domains = ["asm", "debugging"]
```

### Create Preset for New Combinations

```toml
[presets.oracle-purah-cloud]
expert = "oracle-purah-profiler"
hardware = "cloud-rtx-4090"
dataset = "my_dataset"
hyperparameters = "production"
lora = "standard"
sequence_length = 2048
batch_size = 4
gradient_accumulation = 2
```

---

## Output Structure

### Model Directory

Training produces:

```
~/Code/hafs/models/oracle-rauru-assembler-qwen25-coder-15b-20251222/
├── adapter_config.json          # LoRA configuration
├── adapter_model.bin            # LoRA weights
├── config.json                  # Model configuration
├── special_tokens_map.json      # Tokenizer special tokens
├── tokenizer_config.json        # Tokenizer configuration
├── tokenizer.json               # Tokenizer vocab
└── metadata.json                # Training metadata (custom)
```

### Metadata Format

```json
{
  "expert": {
    "name": "Oracle: Rauru Assembler",
    "role": "asm",
    "group": "rom-tooling",
    "specialization": "65816 assembly, ALTTP routines, ROM patching"
  },
  "model": {
    "base": "Qwen/Qwen2.5-Coder-1.5B",
    "lora_rank": 8,
    "lora_alpha": 8
  },
  "training": {
    "dataset": "/Users/scawful/Mounts/mm-d/.context/training/datasets/alttp_yaze_full_1000_20251221_195746",
    "num_samples": 504,
    "num_epochs": 1,
    "batch_size": 1,
    "gradient_accumulation": 2,
    "learning_rate": 0.0002,
    "sequence_length": 1024,
    "duration_seconds": 3847.2,
    "duration_minutes": 64.12
  },
  "hardware": {
    "name": "Mac M1/M2/M3 with MPS",
    "device": "mps",
    "fp16": true,
    "bf16": false
  },
  "metadata": {
    "created": "2025-12-22T00:09:14.123456",
    "git_commit": "a1b2c3d4"
  }
}
```

---

## Migration from Old Scripts

### Old Way

```bash
# Hardcoded Mac training script
python scripts/train_model_mac.py \
    ~/.context/training/datasets/dataset_name \
    "Qwen/Qwen2.5-Coder-1.5B" \
    gold
```

Problems:
- Settings scattered across multiple scripts
- No reusability
- Hardcoded paths and parameters
- Difficult to switch hardware
- No metadata tracking

### New Way

```bash
# Config-based training
python scripts/train.py oracle-rauru-mac
```

Benefits:
- All settings in `config/training.toml`
- Reusable presets
- Easy hardware switching
- Automatic metadata
- Single codebase for all training

---

## Advanced Features

### Programmatic Usage

Use the trainer in Python code:

```python
from hafs.training.config_trainer import ConfigTrainer

# Initialize
trainer = ConfigTrainer()

# Train with preset
output_dir = trainer.train("oracle-rauru-mac")

print(f"Model saved to: {output_dir}")
```

### Custom Configuration Path

```bash
# Use custom config file
python scripts/train.py oracle-rauru-mac --config my_config.toml
```

### Dynamic Preset Building

```python
from hafs.training.config_trainer import ConfigTrainer

trainer = ConfigTrainer()

# Build config from preset
config = trainer.build_training_config("oracle-rauru-mac")

# Modify config
config.learning_rate = 3e-4
config.num_epochs = 2

# Execute training with modified config
trainer.train_with_config(config)
```

---

## Troubleshooting

### Preset Not Found

```
Error: Preset 'oracle-rauru-mac' not found
```

**Solution**: Check available presets:
```bash
python scripts/train.py --list-presets
```

### Hardware Unsupported

```
RuntimeError: Hardware ... is unsupported: CUDA capability sm_120 not supported by PyTorch
```

**Solution**: Use a different hardware profile:
```bash
# Use Mac MPS instead
python scripts/train.py oracle-rauru-mac

# Or wait for PyTorch sm_120 support
```

### Out of Memory

```
RuntimeError: MPS backend out of memory
```

**Solution**: The hardware profile's memory limits may be incorrect. Edit `config/training.toml`:
```toml
[hardware.mac-mps]
max_sequence_length = 512  # Reduce from 1024
max_lora_rank = 4          # Reduce from 8
```

Or create a custom preset with lower settings.

---

## Best Practices

1. **Start with existing presets** - Use `oracle-rauru-mac` or `oracle-rauru-cloud` as templates
2. **Test on small datasets first** - Verify config before full training
3. **Document custom presets** - Add comments in `training.toml`
4. **Version control config** - Commit `training.toml` changes
5. **Track metadata** - Save `metadata.json` with trained models
6. **Use verbose logging** - Add `--verbose` when debugging

---

## Future Enhancements

- [ ] Auto-detect hardware and suggest presets
- [ ] Resume from checkpoint support
- [ ] Multi-GPU training presets
- [ ] Weights & Biases integration
- [ ] Automated hyperparameter tuning
- [ ] Cloud training orchestration (RunPod, Lambda Labs)
- [ ] Model registry integration
- [ ] Automatic model evaluation after training

---

## Summary

**What**: Unified configuration-based training system
**Why**: Eliminate hardcoded scripts, centralize settings, improve reusability
**How**: All config in `config/training.toml`, single trainer class, simple CLI

**Key Files**:
- `config/training.toml` - All training configuration
- `src/hafs/training/config_trainer.py` - Unified trainer class
- `scripts/train.py` - Simple CLI entry point

**Usage**:
```bash
python scripts/train.py oracle-rauru-mac
```

**Status**: ✓ Ready to use (replaces `train_model_mac.py`, `train_zelda_unsloth.py`, etc.)
