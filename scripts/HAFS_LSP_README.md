# hafs-lsp Quick Reference

## Setup (First Time)
```bash
cd ~/Code/hafs
./scripts/setup_hafs_lsp.sh  # Downloads models, configures editors
```

## Daily Usage

### Enable/Disable
```bash
./scripts/hafs_lsp_control.sh enable   # Turn on
./scripts/hafs_lsp_control.sh disable  # Turn off
./scripts/hafs_lsp_control.sh status   # Check status
```

### In Your Editor
- **Neovim**: `:LspStart hafs_lsp` or press `Ctrl+Space`
- **Spacemacs**: `M-x lsp` or `SPC m l`
- **VSCode**: Opens automatically for .asm files
- **Terminal**: `Ctrl+Space` for completions

### Swap Models
```bash
./scripts/hafs_lsp_control.sh model fast qwen2.5-coder:1.5b   # Faster
./scripts/hafs_lsp_control.sh model quality qwen2.5-coder:14b # Better quality
```

### After Training
```bash
# Create fine-tuned model
./scripts/create_fine_tuned_model.sh qwen2.5-coder:1.5b ~/.context/training/datasets/latest gold

# Use it
./scripts/hafs_lsp_control.sh custom fast ~/Code/hafs/models/hafs-asm-1.5b-20251221-gold
```

## Config File
Edit `~/Code/hafs/config/lsp.toml` for all settings.

**Key settings:**
- `enabled = false` - Change to `true` to activate
- `strategy = "manual_trigger"` - Only runs when you ask (no CPU hit)
- Models, performance limits, ROM context options

## Strategies
- `manual_trigger` - Only on Ctrl+Space **(default, no surprises)**
- `fast_only` - Auto-complete with 1.5B model
- `quality_only` - Auto-complete with 7B model
- `adaptive` - Switches based on context

## Docs
- Full guide: `~/Code/hafs/docs/guides/HAFS_LSP_CONTROL.md`
- Setup guide: `~/Code/hafs/docs/guides/HAFS_LSP_QUICKSTART.md`
