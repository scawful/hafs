# Windows GPU Models Setup

Complete guide for setting up multiple models on medical-mechanica (RTX 5060 Ti 16GB).

## Quick Start

```bash
cd ~/Code/hafs
./scripts/deploy_models_windows.sh
```

This will:
1. Install Ollama models for data generation
2. Setup Unsloth for fine-tuning
3. Download base models from Hugging Face
4. Verify CUDA and disk space

## Current Models (Already Installed)

| Model | Size | Purpose | Quality |
|-------|------|---------|---------|
| qwen3:14b | 9GB | General purpose | High |
| deepseek-r1:14b | 9GB | Reasoning | High |
| magistral:24b | 14GB | Large general | Highest |
| ministral-3:8b | 6GB | Fast general | Medium |
| codellama:latest | 4GB | Code | Medium |
| gemma3:27b | 17GB | General | High |
| gemma3:12b | 8GB | General | High |

## Recommended Additions

### For Data Generation (Ollama)

**High Priority:**
```powershell
ollama pull qwen2.5-coder:14b    # 9GB - BEST for code (C++, Python)
ollama pull deepseek-coder:6.7b  # 4GB - BEST for ASM/low-level
ollama pull phi3.5:latest        # 2GB - Fast, efficient
```

**Optional (if space allows):**
```powershell
ollama pull qwen2.5-coder:7b     # 5GB - Faster alternative
ollama pull deepseek-coder:33b   # 19GB - Highest quality ASM
ollama pull mistral:7b           # 4GB - Fast general purpose
```

### For Fine-Tuning (Hugging Face)

**Primary (Recommended):**
- `Qwen/Qwen2.5-Coder-14B-Instruct` (28GB disk, 14GB VRAM)
  - Best balance of quality and speed
  - Excellent for SNES/ASM domain
  - Fits comfortably in 16GB VRAM with LoRA

**Fast Alternative:**
- `Qwen/Qwen2.5-Coder-7B-Instruct` (14GB disk, 8GB VRAM)
  - Faster training (2x)
  - Good quality
  - More experimentation iterations

**ASM Specialist:**
- `deepseek-ai/DeepSeek-Coder-33B-instruct` (66GB disk, 16GB VRAM)
  - Best for low-level code
  - Requires 8-bit quantization
  - Slower but highest quality

## Manual Setup

### 1. Install Ollama Models

SSH to medical-mechanica:

```bash
ssh medical-mechanica
```

Install essential models:

```powershell
# Code specialists (PRIORITY)
ollama pull qwen2.5-coder:14b
ollama pull qwen2.5-coder:7b
ollama pull deepseek-coder:6.7b

# Fast alternatives
ollama pull phi3.5:latest
ollama pull mistral:7b

# List installed
ollama list
```

### 2. Setup Fine-Tuning Environment

```powershell
cd D:\hafs_training

# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes

# Install training tools
pip install datasets transformers wandb

# Install HF CLI with fast downloads
pip install -U "huggingface_hub[cli,hf_transfer]"

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

### 3. Download Fine-Tuning Base Models

```powershell
# Set HF token (get from https://huggingface.co/settings/tokens)
$env:HF_TOKEN = "hf_your_token_here"

# Download recommended model (Qwen2.5-Coder-14B)
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct --local-dir D:\hafs_training\models\qwen2.5-coder-14b

# Or use the Python script
python D:\hafs_training\scripts\setup_finetuning_models.py --models recommended
```

## Model Selection Strategy

### By Domain

| Domain | Best Model | Alternative | Why |
|--------|-----------|-------------|-----|
| ASM | `deepseek-coder:6.7b` | `deepseek-coder:33b` | Low-level code understanding |
| C++ | `qwen2.5-coder:14b` | `qwen2.5-coder:7b` | General code quality |
| YAZE | `qwen2.5-coder:14b` | `codellama` | Emulator code |
| Gigaleak | `deepseek-coder:6.7b` | `deepseek-coder:33b` | Original SNES source |
| Oracle | `qwen2.5-coder:14b` | `qwen3:14b` | ROM hack mix |
| Text | `qwen2.5:14b` | `phi3.5` | Natural language |
| Errors | `deepseek-r1:14b` | `deepseek-r1:8b` | Reasoning |

### By Speed vs Quality

**Maximum Speed:**
- `phi3.5:latest` (2GB) - 3-4x faster than 14B models
- `mistral:7b` (4GB) - 2x faster
- `qwen2.5-coder:7b` (5GB) - Fast code generation

**Balanced:**
- `qwen2.5-coder:14b` (9GB) - Best balance
- `deepseek-coder:6.7b` (4GB) - Good for ASM
- `deepseek-r1:14b` (9GB) - Reasoning

**Maximum Quality:**
- `deepseek-coder:33b` (19GB) - Best for ASM
- `qwen2.5-coder:32b` (19GB) - Best for code
- `magistral:24b` (14GB) - General purpose

## Hybrid Campaign with Model Routing

Update `config/training_medical_mechanica.toml`:

```toml
[gpu.models]
# Available models on Ollama
available = [
    "qwen2.5-coder:14b",
    "deepseek-coder:6.7b",
    "qwen2.5:14b",
    "deepseek-r1:14b",
    "phi3.5:latest",
]

[gpu.routing]
# Route domains to best model
asm = "deepseek-coder:6.7b"
cpp = "qwen2.5-coder:14b"
yaze = "qwen2.5-coder:14b"
gigaleak = "deepseek-coder:6.7b"
oracle = "qwen2.5-coder:14b"
text = "qwen2.5:14b"
errors = "deepseek-r1:14b"

# Fallback chain
fallback = ["qwen3:14b", "ministral-3:8b"]
```

## Fine-Tuning Workflow

1. **Generate Training Data** (hybrid campaign):
   ```bash
   ./scripts/launch_hybrid_training.sh 34500
   ```

2. **Export to Training Format**:
   ```bash
   hafs training export --dataset ~/.context/training/datasets/hybrid_34500_*
   ```

3. **Transfer to Windows**:
   ```bash
   rsync -avz ~/.context/training/datasets/ medical-mechanica:D:/hafs_training/datasets/
   ```

4. **Start Fine-Tuning** (on Windows):
   ```powershell
   cd D:\hafs_training
   python scripts\fine_tune.py `
     --base-model qwen2.5-coder-14b `
     --dataset datasets\hybrid_34500 `
     --output-dir models\hafs-alttp-qwen-14b `
     --lora-r 64 `
     --lora-alpha 128 `
     --batch-size 2 `
     --epochs 3
   ```

5. **Monitor Training**:
   ```powershell
   # Watch logs
   tail -f D:\hafs_training\logs\training_*.log

   # Check GPU usage
   nvidia-smi -l 1
   ```

## Disk Space Management

Check space on D: drive:

```powershell
Get-PSDrive D | Format-Table -AutoSize
```

Remove old models:

```powershell
# Remove Ollama models
ollama rm old-model-name

# Remove downloaded base models
Remove-Item D:\hafs_training\models\old-model -Recurse -Force
```

## Testing Models

Test Ollama model:

```bash
curl http://100.104.53.21:11435/api/generate -d '{
  "model": "qwen2.5-coder:14b",
  "prompt": "Explain this SNES ASM: LDA $7E0010",
  "stream": false
}'
```

Test from hybrid campaign:

```bash
PYTHONPATH=src .venv/bin/python -m agents.training.scripts.hybrid_campaign --pilot --target 10
```

## Troubleshooting

### "Model not found" in Ollama

```powershell
# List installed models
ollama list

# Pull missing model
ollama pull model-name
```

### "Out of memory" during fine-tuning

1. Reduce batch size: `--batch-size 1`
2. Use smaller model: `qwen2.5-coder-7b`
3. Enable 8-bit quantization (already enabled)
4. Reduce sequence length: `--max-length 1024`

### Slow downloads from Hugging Face

```powershell
# Enable fast downloads
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
pip install hf-transfer
```

## Cost Savings

With local GPU models, you can save significantly on API costs:

| Scenario | API Cost | GPU Cost | Savings |
|----------|----------|----------|---------|
| 34,500 samples (API only) | ~$35 | $0 | 100% |
| Hybrid (70% GPU, 30% API) | ~$10 | $0 | 71% |
| Fine-tuning base | $0 | $0 | 100% |

**Estimated monthly savings**: $100-300 depending on usage

---

**Last Updated**: 2025-12-21
**GPU**: RTX 5060 Ti 16GB
**Tailscale IP**: 100.104.53.21:11435
