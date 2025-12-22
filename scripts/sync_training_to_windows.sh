#!/bin/bash
# Sync training code to Windows via configured mount or SSH
# Uses ~/.config/hafs/sync.toml for configuration

set -e

CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/hafs/sync.toml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "✗ Config file not found: $CONFIG_FILE"
    echo ""
    echo "Create it with:"
    echo ""
    echo "cat > $CONFIG_FILE << 'EOF'"
    echo "[windows]"
    echo "# Option 1: Mount path"
    echo "mount = \"/path/to/mount/hafs_training\""
    echo ""
    echo "# Option 2: SSH (if mount not available)"
    echo "# host = \"hostname\""
    echo "# path = \"D:/hafs_training\""
    echo "EOF"
    echo ""
    exit 1
fi

# Read config (simple TOML parsing)
MOUNT_PATH=$(grep '^mount' "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' "')
SSH_HOST=$(grep '^host' "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' "')
SSH_PATH=$(grep '^path' "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' "')

echo "=========================================="
echo "Syncing Training Code to Windows"
echo "=========================================="
echo ""

# Try mount first, fall back to SSH
if [ -n "$MOUNT_PATH" ] && [ -d "$MOUNT_PATH" ]; then
    echo "Using mount: $MOUNT_PATH"
    echo ""

    echo "[1/3] Syncing training code..."
    # Use cp instead of rsync for SMB mounts (avoids permission issues)
    mkdir -p "$MOUNT_PATH/src/agents/training"
    cp -r ~/Code/hafs/src/agents/training/* "$MOUNT_PATH/src/agents/training/" 2>/dev/null || {
        echo "  ⚠ Some files may not have copied (this is often OK for SMB mounts)"
    }
    echo "✓ Training code synced"
    echo ""

    echo "[2/3] Syncing config files..."
    mkdir -p "$MOUNT_PATH/config"
    cp ~/Code/hafs/config/*training*.toml "$MOUNT_PATH/config/" 2>/dev/null || true
    echo "✓ Config files synced"
    echo ""

    echo "[3/3] Syncing scripts..."
    mkdir -p "$MOUNT_PATH/scripts"
    cp ~/Code/hafs/scripts/analyze_rejected_samples.py "$MOUNT_PATH/scripts/" 2>/dev/null || true
    cp ~/Code/hafs/scripts/test_model_training.py "$MOUNT_PATH/scripts/" 2>/dev/null || true
    echo "✓ Scripts synced"
    echo ""

elif [ -n "$SSH_HOST" ] && [ -n "$SSH_PATH" ]; then
    echo "Using SSH: $SSH_HOST:$SSH_PATH"
    echo ""

    if ! ssh -o ConnectTimeout=5 "$SSH_HOST" "exit" 2>/dev/null; then
        echo "✗ Cannot connect to $SSH_HOST"
        exit 1
    fi

    echo "[1/3] Syncing training code..."
    scp -r ~/Code/hafs/src/agents/training/ "$SSH_HOST:$SSH_PATH/src/agents/training/"
    echo "✓ Training code synced"
    echo ""

    echo "[2/3] Syncing config files..."
    scp ~/Code/hafs/config/*training*.toml "$SSH_HOST:$SSH_PATH/config/" 2>/dev/null || true
    echo "✓ Config files synced"
    echo ""

    echo "[3/3] Syncing scripts..."
    scp ~/Code/hafs/scripts/analyze_rejected_samples.py "$SSH_HOST:$SSH_PATH/scripts/" 2>/dev/null || true
    echo "✓ Scripts synced"
    echo ""
else
    echo "✗ No valid sync method configured"
    echo "  Configure either 'mount' or 'host'+'path' in $CONFIG_FILE"
    exit 1
fi

# Verification
echo "Verifying sync..."
if [ -f "$MOUNT_PATH/src/agents/training/quality.py" ]; then
    echo "  ✓ quality.py"
    if grep -q "RejectedSample" "$MOUNT_PATH/src/agents/training/quality.py" 2>/dev/null; then
        echo "    ✓ RejectedSample tracking present"
    fi
fi

if [ -f "$MOUNT_PATH/src/agents/training/curator.py" ]; then
    echo "  ✓ curator.py"
    if grep -q "_save_rejected_samples" "$MOUNT_PATH/src/agents/training/curator.py" 2>/dev/null; then
        echo "    ✓ Rejected sample saving present"
    fi
fi

echo ""
echo "=========================================="
echo "✓ Sync Complete!"
echo "=========================================="
echo ""
