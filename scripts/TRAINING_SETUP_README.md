# PyTorch/Unsloth Training Environment Setup

Comprehensive installation scripts for setting up a PyTorch + Unsloth training environment on **medical-mechanica** (Windows 11 Pro with RTX 5060 Ti 16GB).

## System Information

- **Hostname**: medical-mechanica (alias: mm)
- **OS**: Windows 11 Pro
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **CUDA**: 11.2
- **Python**: 3.14.0
- **Storage**: 1.6TB free on D: drive
- **Access**: SSH via Tailscale

## Quick Start

### 1. Connect to the machine

```bash
# From your local machine (via Tailscale)
ssh medical-mechanica
# or
ssh mm
```

### 2. Run the installation script

```powershell
# Navigate to the scripts directory
cd path\to\hafs\scripts

# Run the installation (full setup)
.\install_training_env.ps1

# Or with custom installation path
.\install_training_env.ps1 -InstallPath "E:\my_training"
```

### 3. Validate the installation

```powershell
# Run validation script
python test_training_setup.py
```

## Files Included

### `install_training_env.ps1`

PowerShell script that:
- Installs PyTorch with CUDA 11.8 support (compatible with CUDA 11.2)
- Installs Unsloth for efficient LoRA training
- Installs all required training dependencies
- Creates directory structure at `D:\training`
- Runs validation tests
- Generates environment info and quick reference

**Usage:**

```powershell
# Full installation
.\install_training_env.ps1

# Skip PyTorch (if already installed)
.\install_training_env.ps1 -SkipPyTorch

# Skip Unsloth (if already installed)
.\install_training_env.ps1 -SkipUnsloth

# Skip validation
.\install_training_env.ps1 -SkipValidation

# Custom installation path
.\install_training_env.ps1 -InstallPath "E:\custom_path"

# Combine options
.\install_training_env.ps1 -SkipPyTorch -InstallPath "E:\training"
```

### `test_training_setup.py`

Python validation script that checks:
- Python version and environment
- PyTorch installation and CUDA availability
- GPU detection and memory
- Unsloth installation
- All training dependencies
- nvidia-smi availability
- Directory structure
- Quick GPU computation test

**Usage:**

```bash
# Run validation
python test_training_setup.py

# Should output a comprehensive report with PASS/FAIL for each component
```

## What Gets Installed

### Core Packages

- **PyTorch** (with CUDA 11.8 support)
- **torchvision** (computer vision utilities)
- **torchaudio** (audio processing)
- **Unsloth** (efficient LoRA training)

### Training Dependencies

- **transformers** - Hugging Face Transformers library
- **accelerate** - Distributed training utilities
- **bitsandbytes** - 8-bit/4-bit quantization
- **datasets** - Dataset loading and processing
- **peft** - Parameter-Efficient Fine-Tuning (LoRA, etc.)
- **trl** - Transformer Reinforcement Learning
- **xformers** - Memory-efficient attention
- **triton** - GPU kernels
- **einops** - Tensor operations
- **scipy** - Scientific computing
- **tensorboard** - Training visualization
- **sentencepiece** - Tokenization
- **protobuf** - Protocol buffers

### Optional

- **wandb** - Weights & Biases experiment tracking

## Directory Structure

After installation, the following structure is created at `D:\training`:

```
D:\training\
├── datasets/           # Training datasets
├── models/             # Saved models
├── checkpoints/        # Training checkpoints
├── logs/               # Training logs
├── configs/            # Configuration files
├── outputs/            # Generated outputs
├── environment_info.json  # Installation details
└── QUICK_START.txt     # Quick reference guide
```

## Post-Installation

### Quick Validation Commands

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU details
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check GPU status
nvidia-smi

# List installed packages
python -m pip list
```

### Test Unsloth Import

```python
from unsloth import FastLanguageModel

# Load a model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)
```

## Troubleshooting

### CUDA Not Available

```bash
# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors

- Reduce batch size in training config
- Use gradient accumulation
- Enable mixed precision training (fp16/bf16)
- Use 4-bit quantization with Unsloth

### Import Errors

```bash
# Reinstall specific package
python -m pip install --force-reinstall <package_name>

# Update all packages
python -m pip install --upgrade transformers accelerate peft trl
```

### Unsloth Installation Issues

If git installation fails, try:

```bash
# PyPI installation (fallback)
python -m pip install unsloth

# Or install from specific branch
python -m pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git@main"
```

## Remote Training Setup

### Transfer Files to medical-mechanica

```bash
# From local machine (macOS/Linux)
scp your_dataset.jsonl mm:D:/training/datasets/

# Or use rsync
rsync -avz --progress your_dataset.jsonl mm:D:/training/datasets/
```

### Monitor Training Remotely

```bash
# SSH into machine
ssh mm

# Monitor GPU
nvidia-smi -l 1  # Update every second

# Monitor training logs
tail -f D:\training\logs\training.log

# Or use tmux/screen for persistent sessions
```

### Run Training in Background

```powershell
# Start training in background
Start-Process python -ArgumentList "train.py" -NoNewWindow -RedirectStandardOutput "logs\training.log"

# Or use Windows Task Scheduler for scheduled training
```

## Integration with hafs

The training environment can be integrated with the hafs system:

```bash
# From hafs scripts directory
python launch_training.py --config training_config.json
```

See `launch_training.py` for hafs-integrated training launcher.

## Performance Tips

### RTX 5060 Ti (16GB) Optimization

- **Batch Size**: Start with 4-8, adjust based on model size
- **Gradient Accumulation**: Use 4-8 steps for effective larger batches
- **Mixed Precision**: Enable bf16 or fp16
- **Quantization**: Use 4-bit with Unsloth for 2-4x memory savings
- **Max Sequence Length**: 2048 tokens recommended, 4096 max

### Recommended Training Settings

```python
training_args = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_steps": 1000,
    "learning_rate": 2e-4,
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 100,
    "output_dir": "D:/training/outputs",
}
```

## Useful Links

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

## Support

For issues or questions:
1. Check validation output: `python test_training_setup.py`
2. Review installation logs
3. Check `D:\training\environment_info.json`
4. Consult QUICK_START.txt in training directory

---

**Created**: 2025-12-21
**Target System**: medical-mechanica (Windows 11 Pro, RTX 5060 Ti 16GB)
**hafs Integration**: Compatible with hafs training pipeline
