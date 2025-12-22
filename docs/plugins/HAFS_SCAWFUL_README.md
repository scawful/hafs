# hafs_scawful Plugin

Personal hafs configuration and workflow automation for scawful's development environment.

## Overview

This plugin keeps all machine-specific configuration, deployment scripts, and workflow aliases out of the main hafs repository. It provides 30+ aliases and functions for streamlined development.

## Installation

The plugin is already set up in `~/.config/hafs/plugins/hafs_scawful/`.

**To activate in a new shell:**

```bash
# Add to ~/.zshrc (already done)
source ~/.config/hafs/plugins/hafs_scawful/aliases.sh

# Or reload current shell
source ~/.zshrc
```

## Quick Start

```bash
# Show all available commands
hafs-help

# Check system status
hafs-check-mounts
hafs-check-windows

# Full development cycle
hafs-train-dev
```

## Configuration

All settings are in `~/.config/hafs/plugins/hafs_scawful/config.toml`:

- **Machines:** medical-mechanica (Windows GPU), localhost (Mac dev)
- **Mounts:** mm-c, mm-d, mm-e (SMB mounts)
- **Paths:** Code, config, context directories
- **Ollama:** Local and remote endpoints

## Common Workflows

### Training Development

```bash
# 1. Make changes to training code
vim ~/Code/hafs/src/agents/training/quality.py

# 2. Run full development cycle (interactive)
hafs-train-dev

# Or manually:
hafs-presubmit              # Check code quality
git add . && git commit     # Commit changes
hafs-sync                   # Deploy to Windows
```

### Quick Commits

```bash
# Presubmit → commit → push → sync (all in one)
hafs-commit-sync
```

### Analyzing Training Results

```bash
# Find latest dataset
hafs-latest-dataset

# Analyze rejected samples from latest
hafs-analyze-latest

# Analyze specific dataset
hafs-analyze-rejected ~/.context/training/datasets/pilot_hybrid_1000_20251221_161454
```

### Checking Status

```bash
# Local training status
hafs-training-status

# Windows GPU server status
hafs-windows-status

# GPU utilization
hafs-gpu

# Daemon logs
hafs-logs
```

### LSP Management

```bash
# Check LSP status
hafs-lsp-status

# Enable/disable
hafs-lsp-enable
hafs-lsp-disable

# Set to manual trigger (no CPU usage)
hafs-lsp-manual
```

## All Available Aliases

### Navigation
- `cdhafs` - Go to hafs root
- `cdtraining` - Go to training code directory
- `cdlsp` - Go to LSP editor code
- `cdctx` - Go to context directory
- `hafsenv` - Activate hafs virtual environment

### Development Workflow
- `hafs-presubmit` - Run code quality checks before commit
- `hafs-sync` - Sync code to Windows GPU server
- `hafs-commit-sync` - Full cycle: presubmit → commit → push → sync
- `hafs-train-dev` - Interactive development cycle with prompts
- `hafs-test-imports` - Quick test that imports work

### Status Checks
- `hafs-training-status` - Show 10 most recent datasets
- `hafs-logs` - Show last 100 lines of daemon logs
- `hafs-check-windows` - Test SSH connectivity to Windows
- `hafs-windows-status` - Full Windows status (files, Ollama, GPU)
- `hafs-check-mounts` - Verify all SMB mounts are accessible

### LSP Management
- `hafs-lsp-status` - Show LSP configuration and status
- `hafs-lsp-enable` - Enable hafs-lsp server
- `hafs-lsp-disable` - Disable hafs-lsp server
- `hafs-lsp-manual` - Set to manual-trigger mode (Ctrl+Space only)

### Training Analysis
- `hafs-analyze-rejected <dataset>` - Analyze rejected samples with detailed breakdown
- `hafs-latest-dataset` - Find path to most recent dataset
- `hafs-analyze-latest` - Analyze rejected samples from latest dataset

### Windows GPU Server
- `hafs-ssh` - SSH to medical-mechanica
- `hafs-gpu` - Show GPU status (nvidia-smi)

### Maintenance
- `hafs-clean` - Clean Python __pycache__ and .pyc files
- `hafs-disk-usage` - Show disk usage for code, context, datasets
- `hafs-help` - Show help message

## Environment Variables

Set by the plugin:

```bash
HAFS_ROOT="/Users/scawful/Code/hafs"
HAFS_CONFIG="/Users/scawful/.config/hafs"
HAFS_CONTEXT="/Users/scawful/.context"
HAFS_VENV="/Users/scawful/Code/hafs/.venv"

HAFS_WINDOWS_HOST="medical-mechanica"
HAFS_WINDOWS_IP="100.104.53.21"
HAFS_WINDOWS_TRAINING="D:/hafs_training"

HAFS_MOUNT_MMD="/Users/scawful/Mounts/mm-d"
HAFS_TRAINING_MOUNT="/Users/scawful/Mounts/mm-d/hafs_training"
```

## Examples

### Complete Training Development Session

```bash
# 1. Navigate and activate venv
cdtraining
hafsenv

# 2. Check current status
hafs-training-status
hafs-check-mounts

# 3. Make code changes
vim quality.py

# 4. Test locally
hafs-test-imports

# 5. Run development cycle
hafs-train-dev
# This will:
# - Run presubmit checks
# - Show git status
# - Prompt to commit
# - Sync to Windows

# 6. Verify deployment
hafs-windows-status
```

### Quick Analysis After Training

```bash
# Check what datasets exist
hafs-training-status

# Analyze the latest
hafs-analyze-latest

# If you see interesting patterns, dive deeper
hafs-analyze-rejected $(hafs-latest-dataset)
```

### Monitoring Active Training

```bash
# Check logs
hafs-logs

# Check Windows GPU
hafs-gpu

# Full Windows status
hafs-windows-status
```

## Files in Plugin

```
~/.config/hafs/plugins/hafs_scawful/
├── config.toml          # Machine settings (hosts, mounts, paths)
├── aliases.sh           # All workflow aliases (source in ~/.zshrc)
└── scripts/             # User-specific deployment scripts (future)
    ├── deploy_models.sh
    └── backup_context.sh
```

## Why This Plugin Exists

**Problem:** The main hafs repo had machine-specific code:
- Hardcoded hostnames (medical-mechanica)
- Hardcoded paths (/Users/scawful/...)
- User-specific deployment scripts

**Solution:** Move all machine-specific code to user plugin:
- Main repo stays clean and portable
- Easy for other users to adopt hafs
- Deep customization without polluting repo
- Private settings never committed

## Extending the Plugin

### Add New Alias

Edit `~/.config/hafs/plugins/hafs_scawful/aliases.sh`:

```bash
# Custom alias
hafs-my-workflow() {
    echo "Running my custom workflow..."
    # Your commands here
}
```

Reload shell:
```bash
source ~/.zshrc
```

### Add Deployment Script

Create `~/.config/hafs/plugins/hafs_scawful/scripts/deploy_custom.sh`:

```bash
#!/bin/bash
# Load plugin config
source ~/.config/hafs/plugins/hafs_scawful/config.toml

# Your deployment logic
ssh $HAFS_WINDOWS_HOST "your commands"
```

Make executable:
```bash
chmod +x ~/.config/hafs/plugins/hafs_scawful/scripts/deploy_custom.sh
```

Create alias in `aliases.sh`:
```bash
alias hafs-deploy-custom="~/.config/hafs/plugins/hafs_scawful/scripts/deploy_custom.sh"
```

## Troubleshooting

### Aliases not working

```bash
# Check if plugin is sourced
type hafs-help

# If not found, source manually
source ~/.config/hafs/plugins/hafs_scawful/aliases.sh

# Or reload shell
source ~/.zshrc
```

### Windows connectivity issues

```bash
# Test basic connectivity
hafs-check-windows

# Test SSH manually
ssh medical-mechanica "echo OK"

# Check if host is in ~/.ssh/config
cat ~/.ssh/config | grep medical-mechanica
```

### Mount not accessible

```bash
# Check mount status
hafs-check-mounts

# Try remounting via Finder
# Or check network connectivity to Windows machine
```

## Next Steps

1. **Use aliases daily** - They'll become second nature
2. **Customize further** - Add your own workflow shortcuts
3. **Create deployment scripts** - Automate repetitive tasks
4. **Document patterns** - Add notes to this README

## Benefits

- **Speed:** One-command workflows instead of multi-step processes
- **Consistency:** Same commands work every time
- **Memory:** Don't need to remember complex paths/hosts
- **Portability:** Main hafs repo stays clean
- **Customization:** Deep personalization without repo pollution

Type `hafs-help` anytime for quick reference!
