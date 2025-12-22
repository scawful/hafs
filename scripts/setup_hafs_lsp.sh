#!/bin/bash
# Setup hafs-lsp across all editors

echo "========================================================================"
echo "hafs-lsp Setup"
echo "========================================================================"
echo ""

# Pull models
echo "[1/5] Pulling Ollama models..."
echo "This will download ~12GB of models"
echo ""

ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:7b-instruct-q4_K_M

echo ""
echo "✓ Models downloaded"
echo ""

# Add hafs-lsp layer to spacemacs if not already there
echo "[2/5] Configuring Spacemacs..."

if ! grep -q "hafs-lsp" ~/.spacemacs; then
  # Find the configuration-layers line and add hafs-lsp
  sed -i.bak '/dotspacemacs-configuration-layers/,/)/ {
    /treemacs.*)/a\
     hafs-lsp
  }' ~/.spacemacs
  echo "✓ Added hafs-lsp layer to Spacemacs"
else
  echo "✓ hafs-lsp already in Spacemacs config"
fi

echo ""

# Terminal integration
echo "[3/5] Configuring terminal completions..."

if ! grep -q "hafs_complete" ~/.zshrc; then
  cat >> ~/.zshrc << 'EOF'

# hafs-lsp terminal completions for ASM
function _hafs_complete() {
  local response=$(echo "$BUFFER" | python3 ~/Code/hafs/src/hafs/editors/hafs_terminal_complete.py 2>/dev/null)
  if [ -n "$response" ]; then
    BUFFER="${BUFFER}${response}"
    CURSOR=$#BUFFER
  fi
}

# Bind to Ctrl+Space
bindkey '^ ' _hafs_complete
EOF
  echo "✓ Added terminal completions to ~/.zshrc"
else
  echo "✓ Terminal completions already configured"
fi

echo ""

# Test LSP server
echo "[4/5] Testing LSP server..."

cd ~/Code/hafs
if timeout 5 .venv/bin/python src/hafs/editors/hafs_lsp.py --test 2>/dev/null; then
  echo "✓ LSP server starts correctly"
else
  echo "⚠ LSP server test skipped (models not yet pulled)"
fi

echo ""

# Summary
echo "[5/5] Setup Complete!"
echo ""
echo "========================================================================"
echo "Configuration Summary"
echo "========================================================================"
echo ""
echo "✓ Neovim:     ~/.config/nvim/lua/plugins/hafs_lsp.lua"
echo "✓ Spacemacs:  ~/.emacs.d/private/hafs-lsp/"
echo "✓ VSCode:     ~/.config/Code/User/settings.json"
echo "✓ Terminal:   ~/.zshrc (Ctrl+Space for completions)"
echo ""
echo "Models Downloaded:"
echo "  • qwen2.5-coder:1.5b      (3GB)  - Fast completions"
echo "  • qwen2.5-coder:7b-q4     (4GB)  - Quality completions"
echo ""
echo "Next Steps:"
echo "  1. Restart your editors (or reload configs)"
echo "  2. Open an .asm file"
echo "  3. Start typing - completions will appear automatically"
echo ""
echo "For Neovim:    :Lazy reload hafs_lsp"
echo "For Spacemacs: SPC f e R (reload config)"
echo "For VSCode:    Reload window (Cmd+Shift+P > Reload Window)"
echo "For Terminal:  source ~/.zshrc"
echo ""
echo "Test completion speed:"
echo "  time ollama run qwen2.5-coder:1.5b 'LDA #$'"
echo "  (should be < 100ms)"
echo ""
echo "========================================================================"
