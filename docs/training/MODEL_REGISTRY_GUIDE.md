# Model Registry & Deployment Guide

**Last Updated:** 2025-12-22

Complete guide to the hafs model registry and cross-platform deployment system.

---

## Overview

The model registry provides:
- **Centralized model tracking** across Mac, Windows, and cloud
- **Cross-platform path handling** (no more manual slash conversion)
- **Model format conversion** (PyTorch → GGUF for ollama/llama.cpp)
- **Deployment automation** to ollama, llama.cpp, halext nodes
- **Model testing and validation**

---

## Architecture

### Components

1. **ModelRegistry** (`src/hafs/training/model_registry.py`)
   - Tracks all trained models with metadata
   - Stores locations, formats, backends
   - Registry file: `~/.context/models/registry.json`

2. **CrossPlatformPath** (`src/hafs/training/cross_platform.py`)
   - Handles path conversions (Mac ↔ Windows ↔ SSH)
   - No more manual slash/backslash issues
   - Intelligent mount point detection

3. **ModelDeployment** (`src/hafs/training/model_deployment.py`)
   - Pulls models from remote machines
   - Converts formats (PyTorch → GGUF)
   - Deploys to serving backends

4. **CLI Commands** (`src/cli/commands/models.py`)
   - `hafs models list` - List registered models
   - `hafs models info <model_id>` - Show model details
   - `hafs models pull <model_id>` - Pull from remote
   - `hafs models deploy <model_id> <backend>` - Deploy
   - `hafs models test <model_id> <backend>` - Test
   - `hafs models convert <model_id> <format>` - Convert

---

## Configuration

### Setup

1. **Create sync config:**
   ```bash
   cp config/sync.toml.example ~/.config/hafs/sync.toml
   ```

2. **Edit with your settings:**
   ```toml
   [mounts.windows-training]
   local_path = "~/Mounts/mm-d/hafs_training"
   remote_path = "D:/hafs_training"

   [hosts.medical-mechanica]
   hostname = "your-windows-host"
   username = "Administrator"
   platform = "windows"
   ssh_key = "~/.ssh/id_rsa"
   ```

3. **Verify connectivity:**
   ```bash
   # Test mount
   ls ~/Mounts/mm-d/hafs_training

   # Test SSH
   ssh Administrator@your-windows-host
   ```

---

## Workflows

### 1. Register a Trained Model

After training completes on Windows:

```bash
# Automatic registration (reads trainer_state.json)
hafs training register-model D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222

# Manual registration
hafs models register \
  /path/to/model \
  --id oracle-farore-v1 \
  --name "Oracle: Farore Secrets" \
  --base Qwen/Qwen2.5-Coder-1.5B \
  --role general \
  --location windows
```

**What gets stored:**
- Model ID and display name
- Training metrics (loss, perplexity)
- Dataset info (samples, quality metrics)
- Hardware used (RTX 5060 Ti, CUDA)
- LoRA configuration
- Locations where model exists

### 2. List Models

```bash
# List all models
hafs models list

# Filter by role
hafs models list --role asm

# Filter by location
hafs models list --location windows

# Filter by backend
hafs models list --backend ollama

# JSON output
hafs models list --json
```

### 3. Get Model Info

```bash
hafs models info oracle-farore-general-qwen25-coder-15b-20251222
```

**Output:**
```
╭─ Model Info ──────────────────────────────────╮
│            Oracle: Farore Secrets             │
╰───────────────────────────────────────────────╯

Model ID      oracle-farore-general-qwen25-coder-15b-20251222
Version       v1
Role          general
Base Model    Qwen/Qwen2.5-Coder-1.5B
Training Date 2025-12-22
Duration      75 minutes

Dataset:
  Name            oracle_farore_improved
  Samples         320 train / 40 val / 40 test
  acceptance_rate 86.70%
  avg_diversity   0.45

Metrics:
  Final Loss  0.8245
  Best Loss   0.7891

Locations:
  • windows (primary): D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222
  • mac: /Users/scawful/Code/hafs/models/oracle-farore-general-qwen25-coder-15b-20251222

Deployed To:
  • ollama
    Name: oracle-farore
```

### 4. Pull Model from Windows to Mac

```bash
# Auto-detect source
hafs models pull oracle-farore-general-qwen25-coder-15b-20251222

# Specify source
hafs models pull oracle-farore-v1 --source windows

# Custom destination
hafs models pull oracle-farore-v1 --dest ~/my-models/

# What happens:
# 1. Checks if mount available (fast)
# 2. Falls back to SSH/SCP if needed
# 3. Copies all model files
# 4. Updates registry with new location
```

**Transfer methods:**
- **Mount (preferred)**: Uses network mount if accessible (~30s for 3GB)
- **SSH/SCP (fallback)**: Direct SSH transfer (~2min for 3GB)

### 5. Convert to GGUF

```bash
# Convert with Q4_K_M quantization (recommended)
hafs models convert oracle-farore-v1 gguf --quant Q4_K_M

# Other quantizations
hafs models convert oracle-farore-v1 gguf --quant Q5_K_M  # Better quality
hafs models convert oracle-farore-v1 gguf --quant Q8_0    # Best quality
hafs models convert oracle-farore-v1 gguf --quant Q3_K_S  # Smallest size

# Requirements:
# - llama.cpp cloned to ~/Code/llama.cpp
# - Compiled with: make llama-quantize
```

**Quantization comparison:**
| Quantization | Size | Quality | Speed |
|--------------|------|---------|-------|
| Q3_K_S | 600MB | Good | Fast |
| Q4_K_M | 900MB | Very Good | Fast |
| Q5_K_M | 1.1GB | Excellent | Medium |
| Q8_0 | 1.6GB | Best | Slow |

### 6. Deploy to Ollama

```bash
# Deploy with auto-convert to GGUF
hafs models deploy oracle-farore-v1 ollama

# Custom name in Ollama
hafs models deploy oracle-farore-v1 ollama --name oracle-farore

# Specify quantization
hafs models deploy oracle-farore-v1 ollama --quant Q5_K_M

# What happens:
# 1. Converts to GGUF if needed
# 2. Creates Modelfile
# 3. Runs `ollama create`
# 4. Updates registry
```

**Use the model:**
```bash
ollama run oracle-farore "Write a DMA transfer in 65816 assembly"
```

### 7. Test Deployed Model

```bash
hafs models test oracle-farore-v1 ollama
```

**Output:**
```
✓ Test passed

Prompt: Write a simple NOP instruction in 65816 assembly:

Response:
NOP         ; No operation (1 byte, 2 cycles)

Or using EA opcode directly:
DB $EA      ; NOP instruction
```

### 8. Deploy to halext Node

```bash
# Deploy to halext AI node
hafs models deploy oracle-farore-v1 halext --name https://node1.halext.org/api

# What happens:
# 1. Uploads model to node (NOT YET IMPLEMENTED)
# 2. Registers in halext registry
# 3. Starts serving
```

---

## Cross-Platform Path Handling

### Problem

Manual path conversion is error-prone:
```python
# ❌ BAD - manual conversion
windows_path = "D:\\hafs_training\\models\\oracle-farore"
ssh_path = "/d/hafs_training/models/oracle-farore"  # Git Bash
```

### Solution

Use `CrossPlatformPath`:

```python
from hafs.training.cross_platform import CrossPlatformPath

# Create path object
path = CrossPlatformPath("D:/hafs_training/models", platform="windows")

# Convert as needed
windows_path = path.to_windows("backslash")  # D:\hafs_training\models
windows_fwd = path.to_windows("forward")     # D:/hafs_training/models
posix_path = path.to_posix()                 # /d/hafs_training/models (Git Bash)
ssh_path = path.to_ssh_path()                # /d/hafs_training/models
escaped = path.for_shell_escape()            # /d/hafs_training/models (with escaping)
```

### Path Resolver

Use `PathResolver` for mount/SSH operations:

```python
from hafs.training.cross_platform import get_path_resolver

resolver = get_path_resolver()

# Check if remote path exists
exists = resolver.check_remote_path("medical-mechanica", "D:/hafs_training/models")

# Convert remote to local mount
local = resolver.remote_to_local("D:/hafs_training/models/oracle-farore")
# Returns: /Users/scawful/Mounts/mm-d/hafs_training/models/oracle-farore

# Build SSH command with proper escaping
ssh_cmd = resolver.build_ssh_command(
    "medical-mechanica",
    "dir D:\\hafs_training\\models",
    working_dir="C:/hafs"
)

# Build SCP command
scp_cmd = resolver.build_scp_command(
    source="Administrator@host:/d/hafs_training/models/oracle-farore",
    dest="/Users/scawful/Code/hafs/models/oracle-farore",
    remote_host="medical-mechanica",
    recursive=True
)
```

---

## Registry Schema

Model metadata stored in `~/.context/models/registry.json`:

```json
{
  "version": "1.0",
  "updated_at": "2025-12-22T10:30:00",
  "models": {
    "oracle-farore-general-qwen25-coder-15b-20251222": {
      "model_id": "oracle-farore-general-qwen25-coder-15b-20251222",
      "display_name": "Oracle: Farore Secrets",
      "version": "v1",
      "base_model": "Qwen/Qwen2.5-Coder-1.5B",
      "role": "general",
      "group": "rom-tooling",
      "training_date": "2025-12-22T09:48:00",
      "training_duration_minutes": 75,

      "dataset_name": "oracle_farore_improved",
      "dataset_path": "D:/hafs_training/datasets/oracle_farore_improved_20251222_040629",
      "train_samples": 320,
      "val_samples": 40,
      "test_samples": 40,
      "dataset_quality": {
        "acceptance_rate": 0.867,
        "rejection_rate": 0.133,
        "avg_diversity": 0.45
      },

      "final_loss": 0.8245,
      "best_loss": 0.7891,
      "perplexity": 2.28,

      "lora_config": {
        "r": 16,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "lora_dropout": 0.05
      },

      "hyperparameters": {
        "learning_rate": 0.0002,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
        "sequence_length": 2048
      },

      "hardware": "windows-rtx-5060",
      "device": "cuda",
      "model_path": "D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222",

      "formats": ["pytorch", "gguf"],

      "locations": {
        "windows": "D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222",
        "mac": "/Users/scawful/Code/hafs/models/oracle-farore-general-qwen25-coder-15b-20251222"
      },
      "primary_location": "windows",

      "deployed_backends": ["ollama"],
      "ollama_model_name": "oracle-farore",
      "halext_node_id": null,

      "git_commit": "8a14bc8",
      "notes": "First production model with improved diverse dataset",
      "tags": ["production", "diverse-v2"],

      "created_at": "2025-12-22T09:48:00",
      "updated_at": "2025-12-22T11:30:00"
    }
  }
}
```

---

## Common Tasks

### Check Model Status

```bash
# Which models are deployed?
hafs models list --backend ollama

# Where is model available?
hafs models info oracle-farore-v1 | grep Locations

# What formats exist?
hafs models info oracle-farore-v1 | grep formats
```

### Pull & Deploy Workflow

Complete workflow from Windows training to local Ollama:

```bash
# 1. Register after training
hafs training register-model D:/hafs_training/models/oracle-farore-v1

# 2. Pull to Mac
hafs models pull oracle-farore-v1

# 3. Convert to GGUF
hafs models convert oracle-farore-v1 gguf

# 4. Deploy to Ollama
hafs models deploy oracle-farore-v1 ollama

# 5. Test it
hafs models test oracle-farore-v1 ollama

# 6. Use it
ollama run oracle-farore "Your prompt here"
```

### Cleanup Old Models

```bash
# List all models
hafs models list

# Get info to check if still needed
hafs models info old-model-v1

# Remove from registry (doesn't delete files)
hafs models delete old-model-v1

# Manually delete files if needed
rm -rf ~/Code/hafs/models/old-model-v1
```

---

## Troubleshooting

### Model Pull Fails

**Mount not accessible:**
```bash
# Check mount
ls ~/Mounts/mm-d/hafs_training

# Remount if needed
mount -t smbfs //hostname/share ~/Mounts/mm-d
```

**SSH fails:**
```bash
# Test SSH connection
ssh Administrator@hostname

# Check SSH key
ssh -i ~/.ssh/id_rsa Administrator@hostname

# Update ~/.config/hafs/sync.toml with correct hostname/key
```

### Conversion Fails

**llama.cpp not found:**
```bash
# Clone llama.cpp
cd ~/Code
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Compile
make llama-quantize

# Verify
./llama-quantize --help
```

**Python conversion script error:**
```bash
# Check dependencies
pip install transformers torch

# Check Python version (need 3.10+)
python --version
```

### Ollama Deployment Fails

**Ollama not running:**
```bash
# Start Ollama
ollama serve

# Or check if already running
ps aux | grep ollama
```

**Model create fails:**
```bash
# Check GGUF file exists
ls ~/Code/hafs/models/oracle-farore-v1/*.gguf

# Try manual create
ollama create oracle-farore -f ~/Code/hafs/models/oracle-farore-v1/Modelfile

# Check Ollama logs
journalctl -u ollama -f  # Linux
~/Library/Logs/ollama.log  # Mac
```

---

## Best Practices

### Model Naming

Use consistent naming:
```
oracle-<name>-<role>-<base>-<date>

Examples:
- oracle-farore-general-qwen25-coder-15b-20251222
- oracle-rauru-asm-qwen25-coder-15b-20251221
- oracle-sheik-debug-qwen25-coder-15b-20251220
```

### Quantization Selection

- **Development/Testing**: Q4_K_M (fast, good quality)
- **Production**: Q5_K_M (better quality, acceptable speed)
- **Highest Quality**: Q8_0 (slow, large, best results)
- **Minimal Size**: Q3_K_S (smallest, reduced quality)

### Storage Management

Models are large - manage storage:
```bash
# Check size
du -sh ~/Code/hafs/models/*

# Keep only GGUF versions on Mac
rm -rf ~/Code/hafs/models/*/pytorch_model.bin

# Use Windows for PyTorch originals
# Use Mac for GGUF versions only
```

### Registry Backup

Backup registry periodically:
```bash
cp ~/.context/models/registry.json ~/.context/models/registry.json.bak
```

---

## Integration with Training

Auto-register models after training:

```python
from hafs.training.model_registry import register_training_run

# At end of training
metadata = register_training_run(
    model_path=output_dir,
    config=training_config,
    metrics=trainer.state.log_history[-1],
    location="windows"
)

logger.info(f"Model registered: {metadata.model_id}")
```

---

## Future Enhancements

Planned features:
- [ ] Halext node deployment API implementation
- [ ] Model versioning and lineage tracking
- [ ] Automatic GGUF conversion on save
- [ ] Model comparison and benchmarking
- [ ] Cloud storage integration (S3, GCS)
- [ ] Model sharing and publishing
- [ ] Performance profiling
- [ ] A/B testing framework
