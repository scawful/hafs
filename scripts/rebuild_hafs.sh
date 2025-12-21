#!/bin/bash
# rebuild_hafs.sh
# Rebuilds and reinstalls HAFS and optional plugin repos.

set -e

echo "=== HAFS Rebuild Script ==="

# Function to clean artifacts
clean_project() {
    local proj_dir=$1
    echo "Cleaning $proj_dir..."
    find "$proj_dir" -name "*.pyc" -delete
    find "$proj_dir" -name "__pycache__" -delete
    find "$proj_dir" -type d -name "*.egg-info" -exec rm -rf {} +
    find "$proj_dir" -type d -name "build" -exec rm -rf {} +
    find "$proj_dir" -type d -name "dist" -exec rm -rf {} +
}

# 1. Clean
if [ -d "$HOME/Code/hafs" ]; then
    clean_project "$HOME/Code/hafs"
fi

PLUGIN_REPOS=()
if [ -n "$HAFS_PLUGIN_REPOS" ]; then
    IFS=":" read -r -a PLUGIN_REPOS <<< "$HAFS_PLUGIN_REPOS"
elif [ -d "$HOME/Code/hafs-plugins" ]; then
    PLUGIN_REPOS+=("$HOME/Code/hafs-plugins")
fi

for repo in "${PLUGIN_REPOS[@]}"; do
    if [ -d "$repo" ]; then
        clean_project "$repo"
    fi
done

# 2. Install
echo "Installing..."

# Detect environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Detected active virtualenv: $VIRTUAL_ENV"
    INSTALL_CMD="pip install -e"
else
    echo "No virtualenv active. Installing to user local environment."
    # Check if --break-system-packages is needed (Debian 12+ / gLinux)
    if pip install --help | grep -q "break-system-packages"; then
        INSTALL_CMD="pip install -e . --break-system-packages"
    else
        INSTALL_CMD="pip install -e ."
    fi
fi

# Install HAFS Core
if [ -d "$HOME/Code/hafs" ]; then
    echo "Installing HAFS Core..."
    (cd "$HOME/Code/hafs" && $INSTALL_CMD)
else
    echo "Error: ~/Code/hafs not found!"
    exit 1
fi

for repo in "${PLUGIN_REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo "Installing plugin repo: $repo"
        (cd "$repo" && $INSTALL_CMD)
    fi
done

echo "=== Rebuild Complete ==="
echo "You can now run 'hafs'."
