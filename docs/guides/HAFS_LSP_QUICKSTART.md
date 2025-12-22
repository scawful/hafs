# hafs-lsp Quick Start

ROM-aware AI completions for 65816 ASM in all your editors.

## One-Line Setup

```bash
cd ~/Code/hafs && ./scripts/setup_hafs_lsp.sh
```

This will:
1. Download models (~12GB)
2. Configure neovim, spacemacs, vscode
3. Set up terminal completions
4. Test the setup

**Time:** ~15 minutes (depending on download speed)

## What You Get

### Fast Completions (20-40ms)
Type in any editor:
```asm
LDA #$    <-- suggests valid hex values
STA $21   <-- suggests SNES registers
```

### ROM Context
```asm
; hafs-lsp knows you're in bank $00
LDA.w #$80        ; suggests $80 for INIDISP
STA.w $2100       ; suggests $2100 (screen register)
```

### Terminal Integration
```bash
$ echo "LDA #$" <Ctrl+Space>
$ echo "LDA #$80  STA $2100"  # Completion appeared!
```

## Verify It's Working

### Neovim
```vim
:LspInfo
" Should show: hafs_lsp (attached)
```

### Spacemacs
```
M-x lsp-describe-session
; Should show hafs-lsp client
```

### VSCode
Open .asm file, status bar should show "hafs-lsp"

### Terminal
```bash
source ~/.zshrc
echo "LDA" <Ctrl+Space>
# Should complete
```

## Performance

| Editor | First Token | Full Line |
|--------|-------------|-----------|
| Neovim | 30-60ms | 100-150ms |
| Spacemacs | 40-80ms | 120-180ms |
| VSCode | 50-100ms | 150-200ms |
| Terminal | 100-200ms | 200-300ms |

**M1 Max expected latency with 7B model**

## Models

Downloaded by setup script:

### qwen2.5-coder:1.5b (3GB)
- **Use:** Auto-complete while typing
- **Speed:** 20-40ms first token
- **Quality:** Good for simple completions

### qwen2.5-coder:7b-q4 (4GB)
- **Use:** Manual trigger (Ctrl+Space)
- **Speed:** 60-120ms first token
- **Quality:** Excellent, understands context

### Optional: qwen2.5-coder:14b-q4 (8GB)
```bash
ollama pull qwen2.5-coder:14b-instruct-q4_K_M
```
- **Use:** Complex ASM routines
- **Speed:** 150-250ms first token
- **Quality:** Best available

## Keybindings

### Neovim
- `<C-Space>` - Trigger completion
- `gd` - Go to definition
- `K` - Hover documentation
- `<leader>ca` - Code actions

### Spacemacs
- `SPC m =` - Format buffer
- `SPC m a` - Code actions
- `SPC m g` - Go to definition
- `SPC m h` - Describe symbol

### VSCode
- `Ctrl+Space` - Trigger completion
- `F12` - Go to definition
- `Shift+F12` - Find references

## Troubleshooting

### "No completions appearing"

**Check models:**
```bash
ollama list | grep qwen2.5-coder
```

**Check LSP server:**
```bash
cd ~/Code/hafs
.venv/bin/python src/hafs/editors/hafs_lsp.py
# Should start without errors
```

**Check editor LSP status:**
- Neovim: `:LspInfo`
- Spacemacs: `M-x lsp-workspace-show-log`
- VSCode: Output > hafs-lsp

### "Completions are slow (> 200ms)"

**Use faster model:**
Edit editor config to use 1.5B model:
```python
# In hafs_lsp.py
self.quality_model = "qwen2.5-coder:1.5b"  # Instead of 7b
```

**Check ollama performance:**
```bash
time ollama run qwen2.5-coder:1.5b "LDA #$"
# Should be < 50ms
```

### "ROM context not working"

**Check MCP servers:**
```bash
# Ensure yaze MCP server is running
python -m hafs.mcp.client test yaze-debugger
```

## Fine-Tuning (After Campaign Completes)

Once you have training data:

```bash
# Fine-tune 1.5B model on your ASM
python scripts/finetune_lsp_model.py \
  --base-model qwen2.5-coder:1.5b \
  --dataset ~/.context/training/datasets/latest \
  --output ~/Code/hafs/models/hafs-asm-1.5b \
  --epochs 3

# Use fine-tuned model
# Edit hafs_lsp.py:
self.fast_model = "~/Code/hafs/models/hafs-asm-1.5b"
```

**Expected improvement:**
- Learns YOUR coding patterns
- Knows YOUR symbol names
- Understands YOUR project structure
- 2-3x better completions for your code

## Next Steps

1. **Try all editors** - see which feels best
2. **Benchmark models** - compare 1.5B vs 7B vs 14B
3. **Fine-tune** - once training campaign completes
4. **Add LLVM** - for semantic analysis (future)

## Advanced Configuration

### Multi-Model Strategy

**Fast typing:**
```python
# Auto-complete uses 1.5B
if typing_mode:
    model = "qwen2.5-coder:1.5b"
```

**Manual trigger:**
```python
# Ctrl+Space uses 7B
if manual_trigger:
    model = "qwen2.5-coder:7b-instruct-q4_K_M"
```

**Complex completions:**
```python
# Long context uses 14B
if len(context) > 500:
    model = "qwen2.5-coder:14b-instruct-q4_K_M"
```

Edit `src/hafs/editors/hafs_lsp.py` method `_select_model()`.

### Terminal Only Fast Mode

If you only want terminal completions (no editor integration):

```bash
# Just source the completion function
source ~/Code/hafs/scripts/hafs_terminal_complete.sh

# Use Ctrl+Space in shell for completions
```

### Performance Monitoring

```bash
# Log completion times
tail -f ~/.hafs/lsp.log | grep "completion_time"

# See model performance
ollama ps  # Shows active model + memory usage
```

## FAQ

**Q: Can I use this without ollama?**
A: Not yet, but llama.cpp support coming soon.

**Q: Does this work offline?**
A: Yes! All models run locally on your M1.

**Q: Will this slow down my editor?**
A: No - completions are async and non-blocking.

**Q: Can I use my own fine-tuned models?**
A: Yes! Point `hafs_lsp.py` to any ollama model or local GGUF.

**Q: Does this send my code anywhere?**
A: No - everything runs locally on your machine.
