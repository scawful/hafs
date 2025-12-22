# hafs_scawful Plugin - Quick Start

You now have a streamlined workflow system with 30+ aliases. This document shows the most useful commands.

## Instant Setup (Already Done)

```bash
# Plugin is already sourced in ~/.zshrc
# Reload your shell to activate:
source ~/.zshrc

# Or open a new terminal
```

## Most Useful Commands

### 🚀 Development Workflow

```bash
# Full development cycle (one command!)
hafs-commit-sync
# This runs: presubmit → commit → push → sync to Windows

# Interactive development (prompts at each step)
hafs-train-dev

# Just run presubmit checks
hafs-presubmit

# Just sync to Windows
hafs-sync
```

### 📊 Status Checks

```bash
# Check everything is working
hafs-check-mounts      # Verify SMB mounts
hafs-check-windows     # Test Windows SSH

# Show recent training datasets
hafs-training-status

# Full Windows GPU status
hafs-windows-status
```

### 🔍 Training Analysis

```bash
# Analyze latest rejected samples
hafs-analyze-latest

# Find latest dataset path
hafs-latest-dataset

# Analyze specific dataset
hafs-analyze-rejected ~/.context/training/datasets/pilot_hybrid_1000_20251221_161454
```

### 🧭 Quick Navigation

```bash
cdhafs         # Go to hafs root
cdtraining     # Go to training code
cdlsp          # Go to LSP code
cdctx          # Go to context directory
hafsenv        # Activate venv
```

## Example Workflows

### Making Changes to Training Code

```bash
# 1. Navigate and activate
cdtraining
hafsenv

# 2. Edit code
vim quality.py

# 3. Test locally
hafs-test-imports

# 4. One command to deploy
hafs-commit-sync
# Enter commit message when prompted
```

### Analyzing Training Results

```bash
# Check what's available
hafs-training-status

# Analyze the latest
hafs-analyze-latest

# Check Windows status
hafs-windows-status
```

### Managing LSP

```bash
# Check current state
hafs-lsp-status

# Enable/disable
hafs-lsp-enable
hafs-lsp-disable

# Set to manual mode (Ctrl+Space only, no CPU)
hafs-lsp-manual
```

## Environment Variables

These are automatically set:

```bash
$HAFS_ROOT              # /Users/scawful/Code/hafs
$HAFS_CONTEXT           # /Users/scawful/.context
$HAFS_WINDOWS_HOST      # medical-mechanica
$HAFS_TRAINING_MOUNT    # /Users/scawful/Mounts/mm-d/hafs_training
```

Use them in your own scripts:

```bash
cd $HAFS_ROOT
ls $HAFS_CONTEXT/training/datasets
```

## Getting Help

```bash
# Show all available commands
hafs-help

# Show this quick start
cat ~/.config/hafs/plugins/hafs_scawful/README.md
```

## Files in Plugin

```
~/.config/hafs/plugins/hafs_scawful/
├── config.toml    # Machine settings
├── aliases.sh     # All aliases (sourced in ~/.zshrc)
└── README.md      # Full documentation
```

## Next Steps

1. **Try the workflows** - Use `hafs-commit-sync` for your next change
2. **Check status often** - `hafs-training-status`, `hafs-windows-status`
3. **Analyze results** - `hafs-analyze-latest` after training
4. **Customize** - Edit `~/.config/hafs/plugins/hafs_scawful/aliases.sh` to add your own

## Troubleshooting

**"Command not found"**
```bash
# Reload shell
source ~/.zshrc
```

**"Cannot connect to Windows"**
```bash
# Check SSH
hafs-check-windows

# Try manual SSH
ssh medical-mechanica "echo OK"
```

**"Mount not accessible"**
```bash
# Check mounts
hafs-check-mounts

# Remount via Finder if needed
```

## Summary

**Before:** Multi-step workflows with long commands and paths
```bash
cd ~/Code/hafs
source .venv/bin/activate
~/Code/hafs/scripts/presubmit_training.sh
git add . && git commit -m "message"
git push
~/Code/hafs/scripts/sync_training_to_windows.sh
```

**After:** One command
```bash
hafs-commit-sync
```

Type `hafs-help` anytime for the full command list!
