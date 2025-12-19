#!/bin/bash
# dev_hafs.sh
# Development setup and rebuild script for HAFS (Public) + Google Internal (Private).

set -e

# Configuration
HAFS_PUBLIC="$HOME/Code/hafs"
HAFS_INTERNAL="$HOME/Code/hafs_google_internal"
VENV_PATH="$HOME/Code/hafs_venv"

echo "=== HAFS Development Environment ==="

# 1. Environment Check
if [ ! -d "$HAFS_PUBLIC" ]; then
    echo "Error: Public repo not found at $HAFS_PUBLIC"
    exit 1
fi

if [ ! -d "$HAFS_INTERNAL" ]; then
    echo "Error: Internal repo not found at $HAFS_INTERNAL"
    exit 1
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
find "$HAFS_PUBLIC" "$HAFS_INTERNAL" -name "*.pyc" -delete
find "$HAFS_PUBLIC" "$HAFS_INTERNAL" -name "__pycache__" -delete
rm -rf "$HAFS_PUBLIC/build" "$HAFS_PUBLIC/dist" "$HAFS_PUBLIC/"*.egg-info
rm -rf "$HAFS_INTERNAL/build" "$HAFS_INTERNAL/dist" "$HAFS_INTERNAL/"*.egg-info

# 4. Install in Editable Mode
echo "Installing HAFS Core (Editable)..."
pip install -e "$HAFS_PUBLIC" --break-system-packages

echo "Installing HAFS Internal (Editable)..."
pip install -e "$HAFS_INTERNAL" --break-system-packages

# 5. Verification
echo "Verifying installation..."
if command -v hafs >/dev/null; then
    HAFS_LOC=$(which hafs)
    echo "✅ hafs found at: $HAFS_LOC"
else
    echo "❌ hafs command not found!"
fi

if pip list | grep -q "hafs-google-internal"; then
    echo "✅ hafs-google-internal installed"
else
    echo "❌ hafs-google-internal NOT installed"
fi

echo "=== Ready for Development ==="
echo "Changes to source code in ~/Code/hafs or ~/Code/hafs_google_internal will be reflected immediately."
