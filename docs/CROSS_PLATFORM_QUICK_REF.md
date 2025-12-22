# Cross-Platform Operations Quick Reference

**Date:** 2025-12-22

Quick reference for working with models across Mac, Windows, and cloud.

---

## Setup (One-Time)

```bash
# 1. Copy sync config
cp config/sync.toml.example ~/.config/hafs/sync.toml

# 2. Edit with your settings
vim ~/.config/hafs/sync.toml

# 3. Test connectivity
ls ~/Mounts/mm-d/hafs_training  # Test mount
ssh Administrator@hostname      # Test SSH
```

---

## Common Commands

### List Models

```bash
hafs models list                    # All models
hafs models list --role asm         # ASM experts only
hafs models list --location windows # On Windows
hafs models list --backend ollama   # Deployed to Ollama
```

### Get Model Info

```bash
hafs models info oracle-farore-v1
```

### Pull from Windows

```bash
# Auto-detect source
hafs models pull oracle-farore-v1

# Specify source
hafs models pull oracle-farore-v1 --source windows

# Custom destination
hafs models pull oracle-farore-v1 --dest ~/my-models/
```

### Convert to GGUF

```bash
# Q4_K_M (recommended)
hafs models convert oracle-farore-v1 gguf

# Q5_K_M (better quality)
hafs models convert oracle-farore-v1 gguf --quant Q5_K_M

# Q8_0 (best quality)
hafs models convert oracle-farore-v1 gguf --quant Q8_0
```

### Deploy to Ollama

```bash
# Default
hafs models deploy oracle-farore-v1 ollama

# Custom name
hafs models deploy oracle-farore-v1 ollama --name oracle-farore

# With quantization
hafs models deploy oracle-farore-v1 ollama --quant Q5_K_M
```

### Test Model

```bash
hafs models test oracle-farore-v1 ollama
```

### Use in Ollama

```bash
ollama run oracle-farore "Write a DMA transfer in 65816 assembly"
```

---

## Complete Workflow

From training on Windows to using in Ollama:

```bash
# After training completes on Windows
hafs training register-model D:/hafs_training/models/oracle-farore-v1
hafs models pull oracle-farore-v1
hafs models deploy oracle-farore-v1 ollama
hafs models test oracle-farore-v1 ollama
ollama run oracle-farore "Your prompt"
```

---

## Path Conversions

No more manual slash conversions:

```python
from hafs.training.cross_platform import CrossPlatformPath

path = CrossPlatformPath("D:/hafs_training/models", platform="windows")

path.to_windows("backslash")  # D:\hafs_training\models
path.to_windows("forward")    # D:/hafs_training/models
path.to_posix()               # /d/hafs_training/models
path.to_ssh_path()            # /d/hafs_training/models
```

---

## Troubleshooting

### Mount not accessible

```bash
ls ~/Mounts/mm-d  # Check mount
# Remount if needed
```

### SSH fails

```bash
ssh Administrator@hostname  # Test connection
# Update ~/.config/hafs/sync.toml if needed
```

### llama.cpp not found

```bash
cd ~/Code
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make llama-quantize
```

### Ollama not running

```bash
ollama serve  # Start Ollama
```

---

## File Locations

- **Registry**: `~/.context/models/registry.json`
- **Sync Config**: `~/.config/hafs/sync.toml`
- **Local Models**: `~/Code/hafs/models/`
- **Training Logs**: `~/.context/training/logs/`

---

## Quantization Guide

| Type | Size | Quality | Use Case |
|------|------|---------|----------|
| Q3_K_S | 600MB | Good | Dev/Testing |
| Q4_K_M | 900MB | Very Good | **Recommended** |
| Q5_K_M | 1.1GB | Excellent | Production |
| Q8_0 | 1.6GB | Best | Quality-Critical |

---

## Model Naming Convention

```
oracle-<name>-<role>-<base>-<date>

Examples:
oracle-farore-general-qwen25-coder-15b-20251222
oracle-rauru-asm-qwen25-coder-15b-20251221
```

---

## Registry Structure

```json
{
  "models": {
    "oracle-farore-v1": {
      "locations": {
        "windows": "D:/hafs_training/models/oracle-farore-v1",
        "mac": "/Users/scawful/Code/hafs/models/oracle-farore-v1"
      },
      "formats": ["pytorch", "gguf"],
      "deployed_backends": ["ollama"],
      "metrics": { "final_loss": 0.8245 }
    }
  }
}
```

---

## Documentation

- Full Guide: `docs/training/MODEL_REGISTRY_GUIDE.md`
- Training Guide: `docs/training/TRAINING_PIPELINE_OVERVIEW.md`
- Config Example: `config/sync.toml.example`
