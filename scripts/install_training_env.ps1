# PowerShell Script: PyTorch/Unsloth Training Environment Setup
# Target: medical-mechanica (Windows 11 Pro, RTX 5060 Ti 16GB, CUDA 11.2)
# Created: 2025-12-21

param(
    [switch]$SkipPyTorch = $false,
    [switch]$SkipUnsloth = $false,
    [switch]$SkipValidation = $false,
    [string]$InstallPath = "D:\training"
)

# Color output functions
function Write-Step {
    param([string]$Message)
    Write-Host "`n[STEP] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Gray
}

# Main installation script
Write-Host @"
================================================================================
PyTorch/Unsloth Training Environment Installation
================================================================================
Target System: medical-mechanica
GPU: NVIDIA RTX 5060 Ti (16GB VRAM)
CUDA: 11.2
Python: 3.14.0
Installation Path: $InstallPath
================================================================================
"@ -ForegroundColor Cyan

# Step 1: Verify Python installation
Write-Step "Verifying Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Python found: $pythonVersion"

    # Check pip
    $pipVersion = python -m pip --version 2>&1
    Write-Success "pip found: $pipVersion"
} catch {
    Write-Error "Python not found or not in PATH. Please install Python 3.14.0 first."
    exit 1
}

# Step 2: Upgrade pip, setuptools, wheel
Write-Step "Upgrading pip, setuptools, and wheel..."
try {
    python -m pip install --upgrade pip setuptools wheel
    Write-Success "pip, setuptools, and wheel upgraded"
} catch {
    Write-Error "Failed to upgrade pip/setuptools/wheel: $_"
    exit 1
}

# Step 3: Install PyTorch with CUDA 11.8 support
if (-not $SkipPyTorch) {
    Write-Step "Installing PyTorch with CUDA 11.8 support..."
    Write-Info "This may take several minutes depending on your internet connection..."

    try {
        # Using cu118 as cu112 builds may not be available for newer PyTorch versions
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        Write-Success "PyTorch installed successfully"
    } catch {
        Write-Error "Failed to install PyTorch: $_"
        exit 1
    }
} else {
    Write-Warning "Skipping PyTorch installation (--SkipPyTorch flag set)"
}

# Step 4: Install Unsloth for efficient LoRA training
if (-not $SkipUnsloth) {
    Write-Step "Installing Unsloth..."
    Write-Info "Attempting to install Unsloth from git repository..."

    try {
        # Try git installation first (recommended)
        python -m pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
        Write-Success "Unsloth installed successfully from git"
    } catch {
        Write-Warning "Git installation failed, trying PyPI installation..."
        try {
            python -m pip install unsloth
            Write-Success "Unsloth installed successfully from PyPI"
        } catch {
            Write-Error "Failed to install Unsloth: $_"
            Write-Info "You may need to install it manually or check dependencies"
            # Don't exit - continue with other installations
        }
    }
} else {
    Write-Warning "Skipping Unsloth installation (--SkipUnsloth flag set)"
}

# Step 5: Install training dependencies
Write-Step "Installing training dependencies..."
$dependencies = @(
    "transformers",
    "accelerate",
    "bitsandbytes",
    "datasets",
    "peft",
    "trl",
    "wandb",
    "xformers",
    "triton",
    "einops",
    "scipy",
    "tensorboard",
    "sentencepiece",
    "protobuf"
)

foreach ($package in $dependencies) {
    Write-Info "Installing $package..."
    try {
        python -m pip install $package
        Write-Success "$package installed"
    } catch {
        Write-Warning "Failed to install $package (may not be critical): $_"
    }
}

# Step 6: Create directory structure
Write-Step "Creating training directory structure at $InstallPath..."
$directories = @(
    "$InstallPath\datasets",
    "$InstallPath\models",
    "$InstallPath\checkpoints",
    "$InstallPath\logs",
    "$InstallPath\configs",
    "$InstallPath\outputs"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        try {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created: $dir"
        } catch {
            Write-Error "Failed to create directory $dir: $_"
        }
    } else {
        Write-Info "Directory already exists: $dir"
    }
}

# Step 7: Create environment info file
Write-Step "Saving environment information..."
$envInfo = @{
    timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    hostname = $env:COMPUTERNAME
    python_version = (python --version 2>&1).ToString()
    pip_version = (python -m pip --version 2>&1).ToString()
    cuda_version = "11.2"
    gpu = "NVIDIA RTX 5060 Ti (16GB)"
    install_path = $InstallPath
    installed_packages = (python -m pip list 2>&1).ToString()
}

$envInfoPath = "$InstallPath\environment_info.json"
$envInfo | ConvertTo-Json | Out-File -FilePath $envInfoPath -Encoding UTF8
Write-Success "Environment info saved to: $envInfoPath"

# Step 8: Run validation tests
if (-not $SkipValidation) {
    Write-Step "Running validation tests..."

    # Check if validation script exists
    $validationScript = Join-Path $PSScriptRoot "test_training_setup.py"
    if (Test-Path $validationScript) {
        Write-Info "Running validation script: $validationScript"
        try {
            python $validationScript
            Write-Success "Validation completed"
        } catch {
            Write-Warning "Validation script encountered issues: $_"
        }
    } else {
        Write-Warning "Validation script not found at: $validationScript"
        Write-Info "You can run manual validation with: python test_training_setup.py"
    }
} else {
    Write-Warning "Skipping validation (--SkipValidation flag set)"
}

# Final summary
Write-Host @"

================================================================================
Installation Complete!
================================================================================

Summary:
- PyTorch: $(if ($SkipPyTorch) { "SKIPPED" } else { "INSTALLED" })
- Unsloth: $(if ($SkipUnsloth) { "SKIPPED" } else { "INSTALLED" })
- Training dependencies: INSTALLED
- Directory structure: CREATED at $InstallPath
- Environment info: SAVED to $envInfoPath

Next Steps:
1. Run validation: python $(Join-Path $PSScriptRoot "test_training_setup.py")
2. Test CUDA availability: python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
3. Check GPU: nvidia-smi
4. Start training with your datasets!

Installation Path: $InstallPath

For remote access via SSH (Tailscale):
  ssh medical-mechanica
  ssh mm

Training directories:
  - Datasets:     $InstallPath\datasets
  - Models:       $InstallPath\models
  - Checkpoints:  $InstallPath\checkpoints
  - Logs:         $InstallPath\logs
  - Configs:      $InstallPath\configs
  - Outputs:      $InstallPath\outputs

================================================================================
"@ -ForegroundColor Green

# Create a quick reference batch file
$quickRefPath = "$InstallPath\QUICK_START.txt"
@"
PyTorch/Unsloth Training Environment - Quick Reference
======================================================

Installation Date: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
System: medical-mechanica (Windows 11 Pro)
GPU: NVIDIA RTX 5060 Ti (16GB VRAM)
CUDA: 11.2
Python: 3.14.0

Quick Commands:
---------------

1. Check CUDA availability:
   python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

2. Check GPU status:
   nvidia-smi

3. Validate installation:
   python $($PSScriptRoot)\test_training_setup.py

4. List installed packages:
   python -m pip list

5. Start Jupyter (if installed):
   jupyter notebook --notebook-dir=$InstallPath

6. Monitor GPU during training:
   nvidia-smi -l 1

Directory Structure:
-------------------
$InstallPath\
  ├── datasets\      # Training datasets
  ├── models\        # Saved models
  ├── checkpoints\   # Training checkpoints
  ├── logs\          # Training logs
  ├── configs\       # Configuration files
  └── outputs\       # Generated outputs

Training with Unsloth:
---------------------
from unsloth import FastLanguageModel

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

Troubleshooting:
---------------
- Out of memory: Reduce batch size or use gradient accumulation
- CUDA errors: Check CUDA version compatibility with: python -c "import torch; print(torch.version.cuda)"
- Import errors: Reinstall package: python -m pip install --force-reinstall <package>

Useful Links:
------------
- Unsloth: https://github.com/unslothai/unsloth
- PyTorch: https://pytorch.org/
- Transformers: https://huggingface.co/docs/transformers/

"@ | Out-File -FilePath $quickRefPath -Encoding UTF8

Write-Success "Quick reference saved to: $quickRefPath"

Write-Host "`nInstallation script completed successfully!" -ForegroundColor Cyan
