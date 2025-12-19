#!/bin/bash
# HAFS Web Hub Launcher (Public)

# Define paths
VENV_PATH="$HOME/dotfiles/.venv"
REPO_ROOT="$HOME/Code/Experimental/hafs"
PLUGIN_ROOT="$HOME/Code/Experimental/hafs_google_internal"
SRC_PATH="$REPO_ROOT/src"
PLUGIN_SRC_PATH="$PLUGIN_ROOT/src"

# Activate venv
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Set PYTHONPATH to include both public core and internal plugin
export PYTHONPATH="$SRC_PATH:$PLUGIN_SRC_PATH:$PYTHONPATH"

# Run Streamlit
echo "Starting HAFS Web Hub..."
echo "Access the dashboard at http://localhost:8501"

streamlit run "$SRC_PATH/hafs/ui/web_dashboard.py"
