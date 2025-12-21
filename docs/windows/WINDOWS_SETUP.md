# hafs Windows Setup Guide (medical-mechanica)

**System:** medical-mechanica (Windows 10 Pro 64-bit)
**GPU:** NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
**CUDA:** 11.2
**Storage:**
- C: (System) - hafs code and runtime
- D: (Data - 1.56 TB free) - Training data, models, .context

## Quick Start

### 1. Environment Setup

Open PowerShell as Administrator:

```powershell
# Set environment variables (replace with your actual keys)
[Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-...', 'User')
[Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIza...', 'User')
[Environment]::SetEnvironmentVariable('PYTHONPATH', 'C:\hafs\src', 'User')

# Refresh environment
$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
$env:GEMINI_API_KEY = [Environment]::GetEnvironmentVariable('GEMINI_API_KEY', 'User')
$env:PYTHONPATH = 'C:\hafs\src'
```

### 2. Python Virtual Environment

```powershell
cd C:\hafs

# Create virtual environment (if not exists)
if (!(Test-Path .venv)) {
    python -m venv .venv
}

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```powershell
# Test hafs import
python -c "import hafs; print('hafs installed successfully')"

# Check hafs CLI
python -m hafs.cli --help

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 5060 Ti
```

## Directory Structure

```
C:\hafs\                    # hafs codebase
├── src\                    # Python source code
├── config\                 # Configuration files
├── scripts\                # Utility scripts
├── docs\                   # Documentation
├── hafs.toml              # Main configuration
└── .venv\                  # Python virtual environment

D:\.context\                # AFS context storage
├── knowledge\              # Knowledge bases
├── embeddings\             # Vector embeddings
├── logs\                   # System logs
├── training\               # Training artifacts
│   ├── datasets\           # Generated datasets
│   ├── checkpoints\        # Training checkpoints
│   └── models\             # Trained models
├── history\                # Agent history
├── scratchpad\             # Working memory
├── memory\                 # Persistent memory
└── hivemind\               # Multi-agent coordination

D:\hafs_training\           # Training-specific storage
├── datasets\               # Training datasets (exported)
├── checkpoints\            # Campaign checkpoints
├── logs\                   # Training logs
├── models\                 # Trained models
└── temp\                   # Temporary files
```

## Configuration

The `hafs.toml` file is pre-configured for Windows with D drive paths:

```toml
[paths]
context_root = "D:/.context"
training_datasets = "D:/hafs_training/datasets"
training_models = "D:/hafs_training/models"

[gpu]
enabled = true
device = "cuda"
memory_fraction = 0.9
```

## Running Training Campaigns

### Generate Training Dataset

```powershell
cd C:\hafs
.\.venv\Scripts\Activate.ps1

# Generate 34,500 samples (8-12 hours)
python -m agents.training.scripts.generate_campaign `
    --target 34500 `
    --export `
    --output-dir D:\hafs_training\datasets

# Monitor progress
python -m agents.training.health_check --watch
```

### Monitor Campaign

```powershell
# Check status
python -m agents.training.health_check

# Watch logs (live)
Get-Content D:\hafs_training\logs\campaign_*.log -Wait -Tail 50

# View campaign summary
python -m hafs.cli training status
```

### Train Models

```powershell
# Train oracle-rauru-assembler on ALTTP ASM dataset
python -m agents.training.scripts.train_model `
    --dataset D:\hafs_training\datasets\alttp_yaze_full_*_asm `
    --model-name oracle-rauru-assembler `
    --output-dir D:\hafs_training\models\oracle-rauru-assembler `
    --config config\training_medical_mechanica.toml

# Train oracle-yaze-expert on YAZE dataset
python -m agents.training.scripts.train_model `
    --dataset D:\hafs_training\datasets\alttp_yaze_full_*_yaze `
    --model-name oracle-yaze-expert `
    --output-dir D:\hafs_training\models\oracle-yaze-expert `
    --config config\training_medical_mechanica.toml
```

## Background Services (Optional)

### Setup Windows Task Scheduler

Create scheduled tasks for background agents:

```powershell
# Create embedding service task
$action = New-ScheduledTaskAction -Execute 'C:\hafs\.venv\Scripts\python.exe' -Argument '-m hafs.services.embedding_daemon'
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId 'Administrator' -RunLevel Highest
Register-ScheduledTask -TaskName 'hafs-embedding-service' -Action $action -Trigger $trigger -Principal $principal

# Create context agent task
$action = New-ScheduledTaskAction -Execute 'C:\hafs\.venv\Scripts\python.exe' -Argument '-m hafs.services.context_agent_daemon'
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName 'hafs-context-agent' -Action $action -Trigger $trigger -Principal $principal
```

### Manage Services

```powershell
# Start services
Start-ScheduledTask -TaskName 'hafs-embedding-service'
Start-ScheduledTask -TaskName 'hafs-context-agent'

# Stop services
Stop-ScheduledTask -TaskName 'hafs-embedding-service'
Stop-ScheduledTask -TaskName 'hafs-context-agent'

# Check status
Get-ScheduledTask -TaskName 'hafs-*' | Select-Object TaskName, State
```

## Remote Access from Mac

### SSH Access

From Mac terminal:

```bash
# SSH into medical-mechanica
ssh Administrator@medical-mechanica

# Or use mounted drives (already configured)
ls ~/Mounts/mm-c/hafs
ls ~/Mounts/mm-d/hafs_training
```

### Copy Files

```bash
# Copy datasets FROM medical-mechanica to Mac
rsync -avzP medical-mechanica:D:/hafs_training/datasets/ \
    ~/.context/training/datasets_from_mechanica/

# Copy trained models FROM medical-mechanica to Mac
rsync -avzP medical-mechanica:D:/hafs_training/models/ \
    ~/.context/training/models_from_mechanica/
```

### Monitor Remotely

```bash
# Check campaign status via SSH
ssh medical-mechanica 'cd C:\hafs && .venv\Scripts\python.exe -m agents.training.health_check'

# Watch logs via SSH
ssh medical-mechanica 'Get-Content D:\hafs_training\logs\campaign_*.log -Wait -Tail 50'

# Or use mounted drives
tail -f ~/Mounts/mm-d/hafs_training/logs/campaign_*.log
```

## Troubleshooting

### GPU Not Detected

```powershell
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support:
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors

Edit `hafs.toml`:

```toml
[gpu]
memory_fraction = 0.7  # Reduce from 0.9

[training]
batch_size = 2  # Reduce from 4
gradient_accumulation_steps = 8  # Increase from 4
```

### API Rate Limits

Edit `hafs.toml`:

```toml
[api]
gemini_rpm = 500  # Reduce from 1000
max_retries = 5  # Increase from 3
retry_delay_seconds = 10  # Increase from 5
```

### Disk Space Issues

```powershell
# Check D drive space
Get-PSDrive D | Select-Object Used,Free

# Clean old checkpoints
Remove-Item D:\hafs_training\checkpoints\* -Recurse -Force

# Archive old logs
Compress-Archive -Path D:\hafs_training\logs\*.log -DestinationPath D:\hafs_training\logs\archive_$(Get-Date -Format 'yyyyMMdd').zip
Remove-Item D:\hafs_training\logs\*.log
```

## Next Steps

1. **Set environment variables** (API keys)
2. **Activate virtual environment** (`.venv\Scripts\Activate.ps1`)
3. **Test hafs CLI** (`python -m hafs.cli --help`)
4. **Run test campaign** (100 samples to verify setup)
5. **Launch full campaign** (34,500 samples)
6. **Train models** (oracle-rauru-assembler, oracle-yaze-expert)
7. **Setup background services** (optional)

## Support

- **Documentation**: `C:\hafs\docs\`
- **Configuration**: `C:\hafs\hafs.toml`
- **Logs**: `D:\hafs_training\logs\`
- **GitHub Issues**: https://github.com/scawful/hafs/issues
