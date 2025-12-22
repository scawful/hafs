# hafs-lsp Control Guide

Easy enable/disable and model swapping.

## Quick Commands

```bash
# Check status
~/Code/hafs/scripts/hafs_lsp_control.sh status

# Enable/disable
~/Code/hafs/scripts/hafs_lsp_control.sh enable
~/Code/hafs/scripts/hafs_lsp_control.sh disable

# Change strategy
~/Code/hafs/scripts/hafs_lsp_control.sh strategy manual_trigger

# Swap models
~/Code/hafs/scripts/hafs_lsp_control.sh model fast qwen2.5-coder:1.5b
~/Code/hafs/scripts/hafs_lsp_control.sh model quality qwen2.5-coder:14b-q4
```

## Configuration File

Edit `~/Code/hafs/config/lsp.toml`:

```toml
[server]
enabled = false        # ← Change to true to activate
auto_start = false     # ← Keep false to avoid surprises

[models]
strategy = "manual_trigger"  # ← No CPU hit unless you press Ctrl+Space

# Available strategies:
# - "fast_only"       - Always use 1.5B (instant, lower quality)
# - "quality_only"    - Always use 7B (slower, better quality)
# - "adaptive"        - Switch based on context size
# - "manual_trigger"  - Only when you press Ctrl+Space (RECOMMENDED)
```

## Strategies Explained

### manual_trigger (Default)
- **CPU usage:** Zero until you press Ctrl+Space
- **Behavior:** No auto-complete, only manual
- **Best for:** When you want full control

### fast_only
- **CPU usage:** Low (1.5B model)
- **Behavior:** Auto-complete as you type
- **Best for:** Fast coding, don't want to wait

### quality_only
- **CPU usage:** Medium (7B model)
- **Behavior:** Auto-complete with better quality
- **Best for:** When you want best suggestions

### adaptive
- **CPU usage:** Low to medium (switches models)
- **Behavior:** Fast model for short context, quality model for long
- **Best for:** Balanced approach

## Model Swapping

### Use Different Base Models

```bash
# Switch to faster model
./scripts/hafs_lsp_control.sh model fast qwen2.5-coder:1.5b

# Switch to better quality
./scripts/hafs_lsp_control.sh model quality qwen2.5-coder:14b-q4

# Use tiny model for ultra-fast completions
./scripts/hafs_lsp_control.sh model fast qwen2.5-coder:0.5b
```

### Use Fine-Tuned Models

After training completes:

```bash
# Create fine-tuned model with unique name
./scripts/create_fine_tuned_model.sh \
  qwen2.5-coder:1.5b \
  ~/.context/training/datasets/latest \
  gold

# Output: hafs-asm-1.5b-20251221-gold

# Use it
./scripts/hafs_lsp_control.sh custom fast ~/Code/hafs/models/hafs-asm-1.5b-20251221-gold
```

### Naming Scheme for Fine-Tuned Models

Format: `hafs-asm-{size}-{date}-{quality}`

Examples:
- `hafs-asm-1.5b-20251221-gold` - Best quality (>0.6 avg)
- `hafs-asm-1.5b-20251221-silver` - Good quality (0.5-0.6)
- `hafs-asm-1.5b-20251221-bronze` - Decent quality (0.4-0.5)
- `hafs-asm-1.5b-20251221-alpha` - Experimental
- `hafs-asm-1.5b-20251221-beta` - Testing

Quality tags:
- **gold**: Production-ready, high quality samples
- **silver**: Good quality, safe to use
- **bronze**: Acceptable quality
- **alpha**: First attempt, experimental
- **beta**: Second iteration, testing

## Enabling LSP Per Editor

Edit `config/lsp.toml`:

```toml
[editors]
neovim = true      # Enable for neovim
spacemacs = false  # Disable for spacemacs
vscode = false     # Disable for vscode
terminal = true    # Enable terminal completions
```

Or use manual start in each editor:

### Neovim
```vim
:LspStart hafs_lsp
```

Or add keybinding to `~/.config/nvim/lua/mappings.lua`:
```lua
vim.keymap.set("n", "<leader>lh", ":LspStart hafs_lsp<CR>", { desc = "Start hafs-lsp" })
```

### Spacemacs
```
M-x lsp
```

Or add to hooks:
```elisp
(add-hook 'asm-mode-hook #'lsp)
```

### VSCode
Just open an ASM file - status bar will show if active.

## Resource Monitoring

```bash
# Check if LSP is running
./scripts/hafs_lsp_control.sh status

# Output shows:
# - Enabled/disabled state
# - Active models
# - CPU and memory usage if running
```

Example output:
```
========================================================================
hafs-lsp Status
========================================================================

Status: ✓ ENABLED
Auto-start: false
Strategy: manual_trigger

Models:
  Fast:    qwen2.5-coder:1.5b
  Quality: qwen2.5-coder:7b-instruct-q4_K_M

Downloaded Models:
qwen2.5-coder:1.5b         2.1 GB  5 hours ago
qwen2.5-coder:7b-q4        4.8 GB  5 hours ago

LSP Server: RUNNING
  PID: 12345
  CPU: 5.2%  Memory: 3.1% (2048 MB)
```

## Avoiding Surprise CPU Usage

**Default config prevents surprises:**

1. `enabled = false` - Must explicitly enable
2. `auto_start = false` - Won't start automatically
3. `strategy = "manual_trigger"` - Only runs when you ask

**To activate:**
```bash
# 1. Enable in config
./scripts/hafs_lsp_control.sh enable

# 2. Choose strategy
./scripts/hafs_lsp_control.sh strategy manual_trigger

# 3. Start in editor when you want it
# Neovim: :LspStart hafs_lsp
# Spacemacs: M-x lsp
```

## Testing Different Models

```bash
# Benchmark models
for model in qwen2.5-coder:1.5b qwen2.5-coder:7b-q4 qwen2.5-coder:14b-q4; do
  echo "Testing $model..."
  time ollama run $model "LDA #$"
done

# Results will show latency differences:
# 1.5b: ~30ms
# 7b:   ~80ms
# 14b:  ~180ms
```

Pick the model that feels right for your workflow.

## Troubleshooting

**"LSP consuming too much CPU"**
```bash
# Switch to faster model
./scripts/hafs_lsp_control.sh model fast qwen2.5-coder:1.5b

# Or disable auto-start
./scripts/hafs_lsp_control.sh strategy manual_trigger
```

**"Completions are slow"**
```bash
# Check what model is active
./scripts/hafs_lsp_control.sh status

# Switch to smaller model
./scripts/hafs_lsp_control.sh model quality qwen2.5-coder:1.5b
```

**"Want to completely disable"**
```bash
./scripts/hafs_lsp_control.sh disable
# Restart editor
```

## Best Practices

1. **Start disabled** - Enable only when you want it
2. **Use manual_trigger** - No surprise CPU usage
3. **Benchmark models** - Find what works for your M1
4. **Fine-tune later** - Use base models first, train custom ones later
5. **Monitor resources** - Check status regularly

## Quick Reference

| Task | Command |
|------|---------|
| Show status | `./scripts/hafs_lsp_control.sh status` |
| Enable | `./scripts/hafs_lsp_control.sh enable` |
| Disable | `./scripts/hafs_lsp_control.sh disable` |
| Manual only | `./scripts/hafs_lsp_control.sh strategy manual_trigger` |
| Fast model | `./scripts/hafs_lsp_control.sh model fast qwen2.5-coder:1.5b` |
| Custom model | `./scripts/hafs_lsp_control.sh custom fast ~/path/to/model` |
| List models | `./scripts/hafs_lsp_control.sh list` |
