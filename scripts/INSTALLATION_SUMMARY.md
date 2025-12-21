# Training Environment Installation - Summary

**Created**: 2025-12-21
**Target**: medical-mechanica (Windows 11 Pro, RTX 5060 Ti 16GB)
**Purpose**: PyTorch + Unsloth training environment setup

## Files Created

### 1. Main Installation Script
**File**: `install_training_env.ps1` (11.8 KB)
- PowerShell script for complete environment setup
- Installs PyTorch with CUDA 11.8 support
- Installs Unsloth and all training dependencies
- Creates directory structure at D:\training
- Generates environment info and quick reference

**Usage**:
```powershell
# On Windows machine (medical-mechanica)
.\install_training_env.ps1

# Or with options
.\install_training_env.ps1 -InstallPath "E:\training"
.\install_training_env.ps1 -SkipPyTorch -SkipValidation
```

### 2. Validation Script
**File**: `test_training_setup.py` (11.5 KB)
- Comprehensive environment validation
- Tests PyTorch, CUDA, GPU, Unsloth, dependencies
- Runs GPU computation tests
- Provides detailed diagnostic output

**Usage**:
```bash
python test_training_setup.py
```

### 3. Quick GPU Test
**File**: `quick_gpu_test.py` (3.2 KB)
- Fast GPU functionality test
- Verifies PyTorch + CUDA working
- Tests memory allocation and computation
- Reports performance metrics

**Usage**:
```bash
python quick_gpu_test.py
```

### 4. Remote Installation Helper
**File**: `remote_install_training.sh` (4.8 KB)
- Bash script for remote installation via SSH
- Transfers scripts to Windows machine
- Executes installation remotely
- Cleans up temporary files

**Usage**:
```bash
# From macOS (local machine)
./remote_install_training.sh mm
# or
./remote_install_training.sh medical-mechanica
```

### 5. Windows Batch Launcher
**File**: `install_training.bat` (0.6 KB)
- Simple double-click installer for Windows
- Automatically runs PowerShell script
- User-friendly for non-technical use

**Usage**:
```
Double-click install_training.bat
```

### 6. Example Training Script
**File**: `example_unsloth_training.py` (6.4 KB)
- Complete working example of Unsloth training
- Optimized for RTX 5060 Ti (16GB)
- Demonstrates LoRA fine-tuning
- Includes best practices and comments

**Usage**:
```bash
python example_unsloth_training.py
```

### 7. Documentation
**File**: `TRAINING_SETUP_README.md` (8.9 KB)
- Comprehensive setup documentation
- Usage instructions for all scripts
- Troubleshooting guide
- Performance optimization tips
- Remote training setup instructions

## Installation Workflow

### Option A: Remote Installation (Recommended)

From your local macOS machine:

```bash
cd ~/Code/hafs/scripts
./remote_install_training.sh mm
```

This will:
1. Test SSH connection to medical-mechanica
2. Transfer installation scripts
3. Run installation remotely
4. Display results
5. Clean up temporary files

### Option B: Direct Installation on Windows

Method 1 - Double-click:
```
1. Navigate to scripts folder on Windows
2. Double-click install_training.bat
3. Wait for completion
```

Method 2 - PowerShell:
```powershell
cd path\to\hafs\scripts
.\install_training_env.ps1
```

Method 3 - Command Prompt:
```cmd
cd path\to\hafs\scripts
powershell -ExecutionPolicy Bypass -File install_training_env.ps1
```

## Post-Installation

### 1. Validate Installation
```bash
python test_training_setup.py
```

### 2. Quick GPU Test
```bash
python quick_gpu_test.py
```

### 3. Run Example Training
```bash
python example_unsloth_training.py
```

### 4. Check GPU Status
```bash
nvidia-smi
```

## What Gets Installed

### Core Components
- PyTorch 2.x (with CUDA 11.8)
- torchvision
- torchaudio
- Unsloth (latest)

### Training Libraries
- transformers (Hugging Face)
- accelerate (distributed training)
- bitsandbytes (quantization)
- datasets (data loading)
- peft (LoRA, QLoRA)
- trl (RLHF)
- xformers (efficient attention)
- triton (GPU kernels)
- einops (tensor ops)

### Utilities
- wandb (experiment tracking)
- tensorboard (visualization)
- scipy (scientific computing)
- sentencepiece (tokenization)

## Directory Structure

After installation:

```
D:\training\
├── datasets/           # Place your training data here
├── models/             # Saved model checkpoints
├── checkpoints/        # Training checkpoints
├── logs/               # Training logs
├── configs/            # Configuration files
├── outputs/            # Training outputs
├── environment_info.json   # Installation details
└── QUICK_START.txt     # Quick reference
```

## System Requirements

### Verified Configuration
- **OS**: Windows 11 Pro
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **CUDA**: 11.2 (compatible with cu118 PyTorch)
- **Python**: 3.14.0
- **Storage**: 1.6TB free on D: drive
- **Network**: Tailscale SSH access

### Minimum Requirements
- Windows 10/11
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.x or 12.x
- Python 3.8+
- 50GB free disk space
- Internet connection for downloads

## Remote Access

### SSH via Tailscale
```bash
ssh medical-mechanica
# or
ssh mm
```

### File Transfer
```bash
# Upload dataset
scp dataset.jsonl mm:D:/training/datasets/

# Download trained model
scp mm:D:/training/models/my_model.bin ./
```

### Remote Training
```bash
# SSH and start training
ssh mm
cd D:\training
python train.py

# Or in background (PowerShell)
Start-Process python -ArgumentList "train.py" -NoNewWindow
```

## Performance Tips for RTX 5060 Ti (16GB)

### Recommended Settings
```python
training_args = {
    "per_device_train_batch_size": 2,      # Small batch size
    "gradient_accumulation_steps": 4,      # Effective batch = 8
    "fp16": False,                         # Use bf16 if supported
    "bf16": True,                          # Better for training
    "max_seq_length": 2048,                # Or 4096 max
    "optim": "adamw_8bit",                 # Memory efficient
    "gradient_checkpointing": True,        # Save memory
}
```

### Model Sizes
- **7B models**: Comfortable with 4-bit quantization
- **13B models**: Use 4-bit + gradient checkpointing
- **70B models**: Not recommended (use smaller LoRA rank)

### Memory Optimization
1. Use 4-bit quantization (`load_in_4bit=True`)
2. Enable gradient checkpointing
3. Use 8-bit optimizer
4. Reduce batch size, increase gradient accumulation
5. Use Flash Attention via xformers
6. Pack sequences for efficiency

## Troubleshooting

### CUDA Not Available
```bash
# Verify installation
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory
- Reduce batch size to 1
- Increase gradient accumulation steps
- Use 4-bit quantization
- Enable gradient checkpointing
- Reduce max sequence length

### Import Errors
```bash
# Reinstall package
python -m pip install --force-reinstall <package>

# Update pip
python -m pip install --upgrade pip

# Check package list
python -m pip list
```

### Unsloth Installation Failed
```bash
# Try PyPI version
python -m pip install unsloth

# Or from git
python -m pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```

## Next Steps

1. **Validate Installation**
   ```bash
   python test_training_setup.py
   ```

2. **Test GPU**
   ```bash
   python quick_gpu_test.py
   ```

3. **Run Example**
   ```bash
   python example_unsloth_training.py
   ```

4. **Prepare Your Data**
   - Format as JSONL or dataset
   - Place in D:\training\datasets\

5. **Start Training**
   - Modify example_unsloth_training.py for your use case
   - Or integrate with hafs training pipeline

## Integration with hafs

The training environment integrates with the hafs ecosystem:

```bash
# From hafs root
hafs train --config training_config.json

# Or use launch script
python scripts/launch_training.py
```

See existing `launch_training.py` for hafs-integrated training.

## Support Resources

### Documentation
- [Unsloth Docs](https://docs.unsloth.ai/)
- [PyTorch Docs](https://pytorch.org/docs/)
- [Transformers Docs](https://huggingface.co/docs/transformers/)
- [PEFT Guide](https://huggingface.co/docs/peft/)

### Repositories
- [Unsloth](https://github.com/unslothai/unsloth)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)

### Community
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [Unsloth Discord](https://discord.gg/unsloth)

## File Checklist

Verify all files are present:

- [ ] `install_training_env.ps1` - Main installer
- [ ] `test_training_setup.py` - Validation script
- [ ] `quick_gpu_test.py` - Quick GPU test
- [ ] `remote_install_training.sh` - Remote installer
- [ ] `install_training.bat` - Windows batch launcher
- [ ] `example_unsloth_training.py` - Training example
- [ ] `TRAINING_SETUP_README.md` - Documentation
- [ ] `INSTALLATION_SUMMARY.md` - This file

## Script Locations

All scripts are in: `/Users/scawful/Code/hafs/scripts/`

Transfer to Windows machine at: `C:\path\to\hafs\scripts\`

Or run remotely via SSH using `remote_install_training.sh`

---

**Installation Ready**: All scripts created and tested
**Remote Access**: Configured via Tailscale (medical-mechanica / mm)
**Target GPU**: RTX 5060 Ti (16GB VRAM)
**Expected Setup Time**: 10-20 minutes (depending on internet speed)

For questions or issues, refer to TRAINING_SETUP_README.md or run validation scripts.
