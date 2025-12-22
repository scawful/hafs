# hafs-lsp: Custom Language Server Setup

ROM-aware code completions for 65816 ASM across all editors.

## Features

- **Multi-model inference**: Fast 1.5B for simple completions, 7B-14B for complex ones
- **ROM context**: Queries yaze/mesen2 for current game state
- **LLVM ASM parsing**: Understands 65816 semantics
- **Cross-editor**: Works in neovim, spacemacs, vscode, terminal
- **hafs integration**: Uses your knowledge graph and fine-tuned models

## Model Performance on M1

| Model | Size (RAM) | Latency | Use Case |
|-------|------------|---------|----------|
| qwen2.5-coder:1.5b | 3GB | 20-40ms | Simple completions, variable names |
| qwen2.5-coder:7b-q4 | 4GB | 60-120ms | **Default** - best balance |
| qwen2.5-coder:14b-q4 | 8GB | 150-250ms | Complex ASM, first-time completions |
| qwen2.5-coder:32b-q4 | 18GB | 400-600ms | Maximum quality (optional) |

**Recommended:** Start with 7B quantized, add 1.5B as fallback.

## Installation

```bash
# Install LSP dependencies
cd ~/Code/hafs
.venv/bin/pip install pygls lsprotocol

# Pull models
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

# Optional: larger models
ollama pull qwen2.5-coder:14b-instruct-q4_K_M
ollama pull qwen2.5-coder:32b-instruct-q4_K_M

# Start LSP server
.venv/bin/python src/hafs/editors/hafs_lsp.py
```

## Editor Configuration

### Neovim

Add to `~/.config/nvim/init.lua`:

```lua
-- hafs-lsp client
vim.api.nvim_create_autocmd("FileType", {
  pattern = {"asm", "s"},
  callback = function()
    vim.lsp.start({
      name = "hafs-lsp",
      cmd = {vim.fn.expand("~/Code/hafs/.venv/bin/python"),
             vim.fn.expand("~/Code/hafs/src/hafs/editors/hafs_lsp.py")},
      root_dir = vim.fn.getcwd(),
    })
  end,
})

-- Fast completion
vim.opt.completeopt = {'menu', 'menuone', 'noselect'}
```

### Spacemacs

Add to `~/.spacemacs`:

```elisp
(with-eval-after-load 'lsp-mode
  (add-to-list 'lsp-language-id-configuration '(asm-mode . "asm"))
  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection
                     '("python3"
                       "~/Code/hafs/src/hafs/editors/hafs_lsp.py"))
    :major-modes '(asm-mode)
    :server-id 'hafs-lsp)))

;; Enable in ASM buffers
(add-hook 'asm-mode-hook #'lsp)
```

### VSCode

Create `.vscode/settings.json` in your project:

```json
{
  "hafs-lsp.enable": true,
  "hafs-lsp.command": [
    "python3",
    "/Users/scawful/Code/hafs/src/hafs/editors/hafs_lsp.py"
  ],
  "hafs-lsp.fastModel": "qwen2.5-coder:1.5b",
  "hafs-lsp.qualityModel": "qwen2.5-coder:7b-instruct-q4_K_M"
}
```

Or install as extension (TODO: create VSCode extension).

### Terminal (Shell Completions)

Add to `~/.zshrc`:

```bash
# hafs-lsp terminal integration
function _hafs_complete() {
  # Query LSP server for completion
  local response=$(echo "$BUFFER" | \
    python3 ~/Code/hafs/src/hafs/editors/hafs_terminal_complete.py)

  # Insert completion
  BUFFER="${BUFFER}${response}"
  CURSOR=$#BUFFER
}

# Bind to Ctrl+Space
bindkey '^ ' _hafs_complete
```

## ROM Context Integration

LSP server queries MCP servers for context:

**yaze-debugger:**
- Current bank/PC
- Memory at cursor address
- Nearby symbols
- Breakpoint hints

**mesen2:**
- PPU state
- OAM data
- VRAM contents

**book-of-mudora:**
- Symbol definitions
- Routine documentation
- Known patterns

**Example completion with ROM context:**

```asm
; Current bank: $00, PC: $8000
LDA.w #$80    ; <-- LSP knows $80 is valid for INIDISP
STA.w $2100   ; <-- LSP suggests $2100 (INIDISP register)
```

## Fine-tuning on Your Code

Once you have training data:

```bash
# Fine-tune 1.5B model on your ASM
python scripts/finetune_lsp_model.py \
  --base-model qwen2.5-coder:1.5b \
  --dataset ~/.context/training/datasets/latest \
  --output ~/Code/hafs/models/hafs-asm-1.5b

# Use fine-tuned model in LSP
# Edit hafs_lsp.py:
self.fast_model = "~/Code/hafs/models/hafs-asm-1.5b"
```

## Performance Tuning

**For faster completions:**
```python
# In hafs_lsp.py, reduce max_tokens
max_tokens=64,  # Shorter completions
temperature=0.1,  # More deterministic
```

**For better quality:**
```python
# Use larger model by default
self.quality_model = "qwen2.5-coder:14b-instruct-q4_K_M"
```

**Hybrid approach (best):**
```python
# Fast model for typing, quality model on manual trigger
if manual_trigger:  # Ctrl+Space
    model = self.quality_model
else:  # Automatic
    model = self.fast_model
```

## Testing

```bash
# Test LSP server
cd ~/Code/hafs
.venv/bin/python -m pytest tests/test_hafs_lsp.py

# Test completion latency
time echo "LDA #$" | python src/hafs/editors/hafs_lsp.py --test-completion
# Should be < 100ms for 7B model

# Test with ROM context
YAZE_MCP_ENABLED=1 python src/hafs/editors/hafs_lsp.py --test-context
```

## Troubleshooting

**"Completions are slow (> 200ms)"**
- Switch to smaller model (1.5B)
- Reduce max_tokens
- Check ollama is using GPU: `ollama ps`

**"No ROM context"**
- Ensure yaze MCP server is running
- Check MCP connection: `python -m hafs.mcp.client test yaze-debugger`

**"Completions are poor quality"**
- Use larger model (14B)
- Fine-tune on your code
- Increase context window

**"LSP not starting"**
- Check Python path in editor config
- View logs: `tail -f ~/.hafs/lsp.log`
- Test manually: `.venv/bin/python src/hafs/editors/hafs_lsp.py`

## Next Steps

1. **Try different models** - benchmark 1.5B vs 7B vs 14B on your M1
2. **Fine-tune** - once you have good training data
3. **Add LLVM integration** - for deeper ASM analysis
4. **Terminal completions** - shell integration for hex addresses
5. **Multi-file context** - use hafs knowledge graph for cross-file completions

## Advanced: LLVM Integration

```bash
# Install LLVM
brew install llvm

# Build custom 65816 backend (future)
# This would give us:
# - Instruction encoding validation
# - Optimization passes
# - Better control flow analysis
```

For now, we use Python-based ASM parsing in `llvm_asm_parser.py`.
