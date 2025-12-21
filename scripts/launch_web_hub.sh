#!/bin/bash
# HAFS Web Hub Launcher

# Define paths
VENV_PATH="$HOME/dotfiles/.venv"
REPO_ROOT="${HAFS_REPO_ROOT:-$HOME/Code/Experimental/hafs}"
SRC_PATH="$REPO_ROOT/src"

# Activate venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Set PYTHONPATH to include repo src and optional plugins
PLUGIN_PATHS="${HAFS_PLUGIN_PATHS:-}"
if [ -n "$PLUGIN_PATHS" ]; then
    export PYTHONPATH="$SRC_PATH:$PLUGIN_PATHS:$PYTHONPATH"
else
    export PYTHONPATH="$SRC_PATH:$PYTHONPATH"
fi

# Run Streamlit
echo "Starting HAFS Web Hub..."
echo "Access the dashboard at http://localhost:8501"

streamlit run "$SRC_PATH/hafs/ui/web_dashboard.py"
