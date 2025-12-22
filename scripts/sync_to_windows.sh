#!/bin/bash
# Sync hafs and hafs_scawful to Windows via mounted shares
#
# This uses ~/Mounts/mm-d which is the D: drive on medical-mechanica
#
# Usage:
#   ./scripts/sync_to_windows.sh           # Full sync
#   ./scripts/sync_to_windows.sh training  # Just training files
#   ./scripts/sync_to_windows.sh data      # Just datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAFS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
HAFS_SCAWFUL="${HAFS_SCAWFUL_ROOT:-$HOME/Code/hafs_scawful}"

# Windows mount point
WINDOWS_MOUNT="${HAFS_WINDOWS_MOUNT:-$HOME/Mounts/mm-d}"
WINDOWS_TRAINING="${HAFS_WINDOWS_TRAINING_MOUNT:-$WINDOWS_MOUNT/hafs_training}"

if [ ! -d "$WINDOWS_MOUNT" ]; then
    echo "ERROR: Windows mount not found at $WINDOWS_MOUNT"
    echo "Mount the D: drive first with: mount_smbfs or similar"
    exit 1
fi

echo "========================================================================"
echo "SYNC TO WINDOWS"
echo "========================================================================"
echo "Source: $HAFS_ROOT"
echo "Plugin: $HAFS_SCAWFUL"
echo "Target: $WINDOWS_TRAINING"
echo ""

COMMAND=${1:-all}

sync_training() {
    echo "[1/3] Syncing training scripts..."
    mkdir -p "$WINDOWS_TRAINING/src/agents/training"

    # Core training code
    rsync -av --delete \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$HAFS_ROOT/src/agents/training/" \
        "$WINDOWS_TRAINING/src/agents/training/"

    echo "[2/3] Syncing hafs_scawful generators..."
    mkdir -p "$WINDOWS_TRAINING/src/hafs_scawful"

    if [ -d "$HAFS_SCAWFUL" ]; then
        rsync -av --delete \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='.git' \
            "$HAFS_SCAWFUL/" \
            "$WINDOWS_TRAINING/src/hafs_scawful/"
    fi

    echo "[3/3] Syncing config..."
    mkdir -p "$WINDOWS_TRAINING/config"
    if [ -f "$HAFS_SCAWFUL/config/training.toml" ]; then
        cp "$HAFS_SCAWFUL/config/training.toml" "$WINDOWS_TRAINING/config/" 2>/dev/null || true
    elif [ -f "$HAFS_ROOT/config/training.toml" ]; then
        cp "$HAFS_ROOT/config/training.toml" "$WINDOWS_TRAINING/config/" 2>/dev/null || true
    fi

    # Create platform-aware config
    cat > "$WINDOWS_TRAINING/config/paths_windows.toml" << 'EOF'
# Windows-specific paths for hafs_training
# Auto-generated - edit with caution

[paths]
# Data directories (D: drive)
datasets = "D:/hafs_training/datasets"
checkpoints = "D:/hafs_training/checkpoints"
models = "D:/hafs_training/models"
logs = "D:/hafs_training/logs"
temp = "D:/hafs_training/temp"

# Source code (synced from Mac)
src = "D:/hafs_training/src"
hafs_scawful = "D:/hafs_training/src/hafs_scawful"

# Context and knowledge (synced from ~/.context)
context_root = "D:/hafs_training/context"
knowledge = "D:/hafs_training/context/knowledge"
embeddings = "D:/hafs_training/context/embeddings"
cache = "D:/hafs_training/context/cache"

[knowledge_bases]
# Individual knowledge base paths
oracle_of_secrets = "D:/hafs_training/context/knowledge/oracle-of-secrets"
alttp = "D:/hafs_training/context/knowledge/alttp"
gigaleak = "D:/hafs_training/context/knowledge/gigaleak"
yaze = "D:/hafs_training/context/knowledge/yaze"

[training]
# GPU settings for RTX 5060 Ti 16GB
batch_size = 4
gradient_accumulation = 4
use_4bit = false
max_seq_length = 2048
EOF

    # Create Python path helper module
    cat > "$WINDOWS_TRAINING/src/platform_paths.py" << 'EOF'
"""Cross-platform path resolution for hafs training.

This module auto-detects the platform and returns appropriate paths.
Works on both Mac (development) and Windows (training).
"""
import os
import sys
from pathlib import Path

def is_windows() -> bool:
    return sys.platform == "win32"

def is_mac() -> bool:
    return sys.platform == "darwin"

def get_base_path() -> Path:
    """Get the base path for hafs training data."""
    if is_windows():
        return Path("D:/hafs_training")
    else:
        return Path.home() / ".context" / "training"

def get_context_path() -> Path:
    """Get the context root path."""
    if is_windows():
        return Path("D:/hafs_training/context")
    else:
        return Path.home() / ".context"

def get_knowledge_path(kb_name: str = None) -> Path:
    """Get path to knowledge bases."""
    base = get_context_path() / "knowledge"
    if kb_name:
        return base / kb_name
    return base

def get_embeddings_path() -> Path:
    """Get path to embeddings database."""
    if is_windows():
        return Path("D:/hafs_training/context/embeddings")
    else:
        return Path.home() / ".context" / "embedding_service"

def get_datasets_path() -> Path:
    """Get path to training datasets."""
    if is_windows():
        return Path("D:/hafs_training/datasets")
    else:
        return Path.home() / ".context" / "training" / "datasets"

def get_models_path() -> Path:
    """Get path to trained models."""
    if is_windows():
        return Path("D:/hafs_training/models")
    else:
        return Path.home() / ".context" / "training" / "models"

def setup_pythonpath():
    """Add necessary paths to sys.path for imports."""
    base = get_base_path()
    src_paths = [
        base / "src",
        base / "src" / "hafs",
        base / "src" / "hafs_scawful",
    ]
    for p in src_paths:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

# Environment variables for hafs
def setup_environment():
    """Set up environment variables for hafs."""
    os.environ["HAFS_CONTEXT_PATH"] = str(get_context_path())
    os.environ["HAFS_KNOWLEDGE_PATH"] = str(get_knowledge_path())
    os.environ["HAFS_EMBEDDINGS_PATH"] = str(get_embeddings_path())
    os.environ["HAFS_DATASETS_PATH"] = str(get_datasets_path())
    setup_pythonpath()
EOF

    echo "OK Training files synced"
}

sync_data() {
    echo "Syncing datasets..."
    mkdir -p "$WINDOWS_TRAINING/datasets"

    # Sync current datasets
    if [ -d "$HOME/.context/training/datasets" ]; then
        rsync -av --progress \
            "$HOME/.context/training/datasets/" \
            "$WINDOWS_TRAINING/datasets/"
    fi

    # Sync training_data if exists
    if [ -d "$HOME/training_data" ]; then
        rsync -av --progress \
            "$HOME/training_data/" \
            "$WINDOWS_TRAINING/datasets/"
    fi

    echo "OK Datasets synced"
}

sync_context() {
    echo "Syncing context and knowledge bases..."

    # Create context directory structure
    mkdir -p "$WINDOWS_TRAINING/context/knowledge"
    mkdir -p "$WINDOWS_TRAINING/context/embeddings"

    # Sync knowledge bases (oracle-of-secrets, alttp, etc.)
    if [ -d "$HOME/.context/knowledge" ]; then
        echo "  Syncing knowledge bases..."
        rsync -av --progress \
            --exclude='*.db-journal' \
            "$HOME/.context/knowledge/" \
            "$WINDOWS_TRAINING/context/knowledge/"
    fi

    # Sync embeddings
    if [ -d "$HOME/.context/embedding_service" ]; then
        echo "  Syncing embeddings database..."
        rsync -av --progress \
            "$HOME/.context/embedding_service/" \
            "$WINDOWS_TRAINING/context/embeddings/"
    fi

    # Sync any cached data
    if [ -d "$HOME/.context/cache" ]; then
        echo "  Syncing cache..."
        mkdir -p "$WINDOWS_TRAINING/context/cache"
        rsync -av --progress \
            "$HOME/.context/cache/" \
            "$WINDOWS_TRAINING/context/cache/"
    fi

    echo "OK Context synced"
}

sync_hafs_core() {
    echo "Syncing hafs core library..."
    mkdir -p "$WINDOWS_TRAINING/src/hafs"

    # Sync core hafs modules needed for agents
    rsync -av --delete \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='.venv' \
        --include='core/***' \
        --include='knowledge/***' \
        --include='services/***' \
        --include='agents/***' \
        --include='__init__.py' \
        --exclude='*' \
        "$HAFS_ROOT/src/hafs/" \
        "$WINDOWS_TRAINING/src/hafs/" 2>/dev/null || true

    # Also sync the full src/hafs for completeness
    rsync -av \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$HAFS_ROOT/src/hafs/" \
        "$WINDOWS_TRAINING/src/hafs/"

    echo "OK hafs core synced"
}

sync_all() {
    sync_hafs_core
    echo ""
    sync_training
    echo ""
    sync_data
    echo ""
    sync_context
}

case "$COMMAND" in
    training)
        sync_training
        ;;
    data)
        sync_data
        ;;
    context)
        sync_context
        ;;
    hafs)
        sync_hafs_core
        ;;
    all)
        sync_all
        ;;
    *)
        echo "Usage: $0 {all|training|data|context|hafs}"
        echo ""
        echo "Commands:"
        echo "  all       - Full sync (hafs + training + data + context)"
        echo "  training  - Just training scripts and generators"
        echo "  data      - Just datasets"
        echo "  context   - Knowledge bases and embeddings"
        echo "  hafs      - Core hafs library"
        exit 1
        ;;
esac

echo ""
echo "========================================================================"
echo "SYNC COMPLETE"
echo "========================================================================"
echo ""
echo "On Windows, run:"
echo "  cd D:\\hafs_training"
echo "  python train_euclid_asm.py --dataset datasets/euclid_asm_v0 --output models/euclid-asm-v0"
echo ""
