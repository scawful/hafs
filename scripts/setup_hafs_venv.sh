#!/bin/bash
set -e

VENV_PATH="$HOME/Code/hafs_venv"

echo "Setting up HAFS virtual environment at $VENV_PATH..."

# Clean up broken venv if it exists
if [ -d "$VENV_PATH" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_PATH"
fi

# Create venv without pip (to avoid ensurepip error on gLinux)
echo "Creating venv..."
python3 -m venv --without-pip "$VENV_PATH"

# Activate
source "$VENV_PATH/bin/activate"

# Bootstrap pip
echo "Bootstrapping pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install projects
echo "Installing HAFS projects..."

if [ -d "$HOME/Code/hafs" ]; then
    echo "Installing hafs..."
    pip install -e "$HOME/Code/hafs"
else
    echo "Warning: ~/Code/hafs not found."
fi

PLUGIN_REPOS=()
if [ -n "$HAFS_PLUGIN_REPOS" ]; then
    IFS=":" read -r -a PLUGIN_REPOS <<< "$HAFS_PLUGIN_REPOS"
elif [ -d "$HOME/Code/hafs-plugins" ]; then
    PLUGIN_REPOS+=("$HOME/Code/hafs-plugins")
fi

for repo in "${PLUGIN_REPOS[@]}"; do
    if [ -d "$repo" ]; then
        echo "Installing plugin repo: $repo"
        pip install -e "$repo"
    fi
done

You can now run this script via:
~/.dotfiles/bin/setup_hafs_venv.sh

---------------------------------------------------
Setup complete!
To activate the environment, run:
source $VENV_PATH/bin/activate
