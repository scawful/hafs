#!/bin/bash
# dev_hafs.sh
# Development setup and rebuild script for HAFS + optional plugin repos.

set -e

# Configuration
HAFS_PUBLIC="$HOME/Code/hafs"
VENV_PATH="$HOME/Code/hafs_venv"

echo "=== HAFS Development Environment ==="

# 1. Environment Check
if [ ! -d "$HAFS_PUBLIC" ]; then
    echo "Error: Public repo not found at $HAFS_PUBLIC"
    exit 1
fi

PLUGIN_REPOS=()
if [ -n "$HAFS_PLUGIN_REPOS" ]; then
    IFS=":" read -r -a PLUGIN_REPOS <<< "$HAFS_PLUGIN_REPOS"
elif [ -d "$HOME/Code/hafs-plugins" ]; then
    PLUGIN_REPOS+=("$HOME/Code/hafs-plugins")
fi

# 2. Virtualenv Check/Activation
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [ -d "$VENV_PATH" ]; then
        echo "Activating virtualenv at $VENV_PATH..."
        source "$VENV_PATH/bin/activate"
    else
        echo "Warning: No active virtualenv and $VENV_PATH not found."
        echo "Installing to user local environment (this might require --break-system-packages)."
    fi
fi

# 3. Clean Artifacts
echo "Cleaning build artifacts..."
find "$HAFS_PUBLIC" -name "*.pyc" -delete
find "$HAFS_PUBLIC" -name "__pycache__" -delete
rm -rf "$HAFS_PUBLIC/build" "$HAFS_PUBLIC/dist" "$HAFS_PUBLIC/"*.egg-info

for repo in "${PLUGIN_REPOS[@]}"; do
    if [ -d "$repo" ]; then
        find "$repo" -name "*.pyc" -delete
        find "$repo" -name "__pycache__" -delete
        rm -rf "$repo/build" "$repo/dist" "$repo/"*.egg-info
    fi
done

# 4. Install in Editable Mode
echo "Installing HAFS Core (Editable)..."
pip install -e "$HAFS_PUBLIC" --break-system-packages

for repo in "${PLUGIN_REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo "Installing plugin repo (Editable): $repo"
        pip install -e "$repo" --break-system-packages
    fi
done

# 5. Verification
echo "Verifying installation..."
if command -v hafs >/dev/null; then
    HAFS_LOC=$(which hafs)
    echo "✅ hafs found at: $HAFS_LOC"
else
    echo "❌ hafs command not found!"
fi

for repo in "${PLUGIN_REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo "✅ plugin repo installed: $repo"
    fi
done

echo "=== Ready for Development ==="
echo "Changes to source code in $HAFS_PUBLIC will be reflected immediately."
