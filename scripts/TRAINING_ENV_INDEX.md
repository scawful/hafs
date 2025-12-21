# Training Environment Setup - File Index

Complete PyTorch/Unsloth training environment setup for medical-mechanica (Windows 11 Pro, RTX 5060 Ti 16GB)

## Core Installation Scripts

### 1. install_training_env.ps1
**Size**: ~12 KB
**Type**: PowerShell script
**Purpose**: Main installation script for Windows

Installs:
- PyTorch with CUDA 11.8 support
- Unsloth (git or PyPI)
- All training dependencies (transformers, accelerate, peft, etc.)
- Creates directory structure
- Generates environment info

**Usage**:
```powershell
.\install_training_env.ps1
.\install_training_env.ps1 -InstallPath "E:\training"
.\install_training_env.ps1 -SkipPyTorch -SkipValidation
```

**Features**:
- Color-coded output
- Error handling
- Progress tracking
- Automatic validation
- Quick reference generation

---

### 2. install_training.bat
**Size**: ~0.6 KB
**Type**: Windows batch file
**Purpose**: Simple double-click installer

**Usage**:
```
Double-click the file
```

Automatically runs install_training_env.ps1 with proper execution policy.

---

### 3. remote_install_training.sh
**Size**: ~5 KB
**Type**: Bash script (for macOS/Linux)
**Purpose**: Remote installation via SSH

**Usage**:
```bash
./remote_install_training.sh mm
./remote_install_training.sh medical-mechanica
```

**Features**:
- Tests SSH connection
- Transfers installation scripts
- Runs installation remotely
- Displays system info
- Cleans up temp files

---

## Validation Scripts

### 4. test_training_setup.py
**Size**: ~12 KB
**Type**: Python validation script
**Purpose**: Comprehensive environment validation

**Tests**:
- Python version
- PyTorch installation
- CUDA availability
- GPU detection and memory
- Unsloth installation
- All dependencies
- nvidia-smi availability
- Directory structure
- GPU computation

**Usage**:
```bash
python test_training_setup.py
```

**Output**: Detailed pass/fail report with color-coded results

---

### 5. quick_gpu_test.py
**Size**: ~3 KB
**Type**: Python test script
**Purpose**: Fast GPU functionality test

**Tests**:
- PyTorch import
- CUDA availability
- GPU detection
- Memory allocation
- Matrix multiplication performance

**Usage**:
```bash
python quick_gpu_test.py
```

**Output**: Quick pass/fail with performance metrics

---

## Training Examples

### 6. example_unsloth_training.py
**Size**: ~6.5 KB
**Type**: Python training script
**Purpose**: Working example of Unsloth LoRA training

**Features**:
- Optimized for RTX 5060 Ti (16GB)
- 4-bit quantization
- LoRA fine-tuning
- Instruction tuning format
- Memory-efficient settings
- Model saving (LoRA + merged)

**Usage**:
```bash
python example_unsloth_training.py
```

**Includes**:
- Model loading
- LoRA configuration
- Dataset preparation
- Training arguments
- Trainer setup
- Model saving

---

## Documentation

### 7. TRAINING_SETUP_README.md
**Size**: ~9 KB
**Type**: Markdown documentation
**Purpose**: Complete setup guide

**Sections**:
- System information
- Quick start guide
- File descriptions
- Usage instructions
- Post-installation steps
- Troubleshooting
- Performance tips
- Remote training setup

---

### 8. INSTALLATION_SUMMARY.md
**Size**: ~10 KB
**Type**: Markdown documentation
**Purpose**: Installation summary and reference

**Sections**:
- Files created
- Installation workflows
- What gets installed
- Directory structure
- System requirements
- Remote access
- Performance tips
- Troubleshooting
- Next steps
- hafs integration

---

### 9. QUICK_REFERENCE.txt
**Size**: ~3 KB
**Type**: Plain text reference
**Purpose**: Quick command reference

**Contents**:
- Installation commands
- Validation commands
- Quick checks
- File locations
- SSH access
- Recommended settings
- Troubleshooting
- Monitoring commands

---

### 10. TRAINING_ENV_INDEX.md
**Size**: ~5 KB
**Type**: Markdown index (this file)
**Purpose**: File directory and overview

---

## Directory Structure (After Installation)

```
D:\training\
├── datasets\           # Training datasets (.jsonl, .json, etc.)
├── models\             # Saved model checkpoints
├── checkpoints\        # Training checkpoints
├── logs\               # Training logs and tensorboard
├── configs\            # Configuration files
├── outputs\            # Training outputs
├── environment_info.json   # Installation metadata
└── QUICK_START.txt     # Auto-generated quick start guide
```

## Installation Workflow

### Remote Installation (Recommended)

```
macOS (local) → SSH → medical-mechanica (Windows)
     ↓
remote_install_training.sh
     ↓
Transfers: install_training_env.ps1, test_training_setup.py
     ↓
Executes: install_training_env.ps1
     ↓
Validates: test_training_setup.py
     ↓
Complete!
```

### Direct Installation

```
Windows machine (medical-mechanica)
     ↓
Option 1: Double-click install_training.bat
Option 2: Run install_training_env.ps1
Option 3: PowerShell command
     ↓
Install PyTorch, Unsloth, dependencies
     ↓
Create directory structure
     ↓
Run validation
     ↓
Complete!
```

## Dependencies Installed

### Core
- torch (PyTorch)
- torchvision
- torchaudio
- unsloth

### Training
- transformers
- accelerate
- bitsandbytes
- datasets
- peft
- trl

### Optimization
- xformers
- triton
- einops

### Utilities
- wandb
- tensorboard
- scipy
- sentencepiece
- protobuf

## Usage Scenarios

### Scenario 1: First-time Setup
1. Run `remote_install_training.sh` from macOS
2. Wait for installation (10-20 minutes)
3. Validate with `test_training_setup.py`
4. Run example with `example_unsloth_training.py`

### Scenario 2: Manual Setup on Windows
1. Transfer scripts to Windows machine
2. Double-click `install_training.bat`
3. Validate installation
4. Start training

### Scenario 3: Quick Test
1. SSH to machine
2. Run `quick_gpu_test.py`
3. Check GPU with `nvidia-smi`
4. Ready to train!

### Scenario 4: Training with Custom Dataset
1. Prepare dataset (JSONL format)
2. Upload to `D:\training\datasets\`
3. Modify `example_unsloth_training.py`
4. Run training
5. Monitor with `nvidia-smi -l 1`

## File Relationships

```
install_training.bat ──┐
                       ├──> install_training_env.ps1 ──┐
remote_install_training.sh ────────────────────────────┤
                                                        ├──> test_training_setup.py
                                                        │
                                                        └──> Creates:
                                                             - D:\training\*
                                                             - environment_info.json
                                                             - QUICK_START.txt

quick_gpu_test.py ──> Fast validation

example_unsloth_training.py ──> Training template

TRAINING_SETUP_README.md ──┐
INSTALLATION_SUMMARY.md ───┼──> Documentation
QUICK_REFERENCE.txt ───────┤
TRAINING_ENV_INDEX.md ─────┘
```

## Integration with hafs

The training environment integrates with existing hafs training infrastructure:

- `launch_training.py` - hafs training launcher
- `launch_autonomous_training.sh` - Autonomous training
- `launch_night_agents.sh` - Scheduled training agents

New scripts complement these by:
1. Setting up the base environment
2. Installing required dependencies
3. Validating hardware configuration
4. Providing training examples

## Quick Start Commands

```bash
# Remote installation (from macOS)
cd ~/Code/hafs/scripts
./remote_install_training.sh mm

# Validate installation
ssh mm
python test_training_setup.py

# Quick GPU test
python quick_gpu_test.py

# Run training example
python example_unsloth_training.py

# Monitor GPU
nvidia-smi -l 1
```

## File Locations

### On macOS (Development Machine)
```
~/Code/hafs/scripts/
├── install_training_env.ps1
├── install_training.bat
├── remote_install_training.sh
├── test_training_setup.py
├── quick_gpu_test.py
├── example_unsloth_training.py
├── TRAINING_SETUP_README.md
├── INSTALLATION_SUMMARY.md
├── QUICK_REFERENCE.txt
└── TRAINING_ENV_INDEX.md
```

### On Windows (medical-mechanica)
```
C:\path\to\hafs\scripts\
└── (same files as above)

D:\training\
├── datasets\
├── models\
├── checkpoints\
├── logs\
├── configs\
├── outputs\
├── environment_info.json
└── QUICK_START.txt
```

## Support and Troubleshooting

For issues:
1. Check validation output: `python test_training_setup.py`
2. Review installation logs
3. Consult TRAINING_SETUP_README.md
4. Check INSTALLATION_SUMMARY.md troubleshooting section
5. Refer to QUICK_REFERENCE.txt for commands

## Version Information

- **Created**: 2025-12-21
- **Target System**: medical-mechanica (Windows 11 Pro)
- **GPU**: NVIDIA RTX 5060 Ti (16GB VRAM)
- **CUDA**: 11.2 (compatible with cu118 PyTorch)
- **Python**: 3.14.0
- **PyTorch**: Latest (2.x) with CUDA 11.8
- **Unsloth**: Latest from git/PyPI

## Next Steps

After reviewing this index:
1. Read TRAINING_SETUP_README.md for detailed instructions
2. Run installation using preferred method
3. Validate with test_training_setup.py
4. Review example_unsloth_training.py
5. Start training with your datasets

---

**All scripts ready for deployment**
**Total files created**: 10
**Documentation**: Complete
**Installation method**: Remote or direct
**Estimated setup time**: 10-20 minutes
