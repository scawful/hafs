#!/bin/bash
# Install/Update shell completion for HAFS (Zsh)

echo "Installing HAFS shell completion for Zsh..."

# Check if hafs is available
if ! command -v hafs &> /dev/null; then
    echo "Error: 'hafs' command not found."
    echo "Please ensure you are in the virtual environment or hafs is installed."
    echo "Try running: source .venv/bin/activate (or similar)"
    exit 1
fi

# Run the installation command
hafs --install-completion zsh

echo ""
echo "✅ Completion installation command executed."
echo "Please restart your shell or run 'compinit' (if using zsh) to enable it."
