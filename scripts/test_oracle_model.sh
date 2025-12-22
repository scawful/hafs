#!/bin/bash
# Quick testing script for newly trained Oracle models

set -e

# Configuration
MODEL_ID="${1:-oracle-farore-general-qwen25-coder-15b-20251222}"
MODEL_NAME="${2:-oracle-farore}"
QUANTIZATION="${3:-Q4_K_M}"

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                        Oracle Model Testing Suite                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model ID:      $MODEL_ID"
echo "Model Name:    $MODEL_NAME"
echo "Quantization:  $QUANTIZATION"
echo ""

# Step 1: Check registration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking model registration..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if hafs models info "$MODEL_ID" &>/dev/null; then
    echo "✓ Model is registered"
    hafs models info "$MODEL_ID"
else
    echo "⚠  Model not registered. Registering now..."
    echo ""
    echo "Please provide the final training loss:"
    read -p "Final loss: " FINAL_LOSS

    hafs models register "$MODEL_ID" \
        --display-name "Oracle: Farore Secrets" \
        --role general \
        --base-model "Qwen/Qwen2.5-Coder-1.5B" \
        --location windows \
        --path "D:/hafs_training/models/$MODEL_ID" \
        --loss "$FINAL_LOSS" \
        --format pytorch

    echo "✓ Model registered successfully"
fi

# Step 2: Check if model is local
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Checking local availability..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

LOCAL_MODELS_DIR="$HOME/Code/hafs/models"
MOUNT_PATH="$HOME/Mounts/mm-d/hafs_training/models/$MODEL_ID"

if [ -d "$LOCAL_MODELS_DIR/$MODEL_ID" ]; then
    echo "✓ Model is available locally at $LOCAL_MODELS_DIR/$MODEL_ID"
elif [ -d "$MOUNT_PATH" ]; then
    echo "✓ Model is available via mount at $MOUNT_PATH"
    echo "  (Using mounted version for testing)"
else
    echo "⚠  Model not found locally or via mount"
    echo ""
    echo "Options:"
    echo "  1. Mount Windows drive: Check that ~/Mounts/mm-d is mounted"
    echo "  2. Pull via SSH: hafs models pull $MODEL_ID --source windows --dest mac"
    echo ""
    read -p "Pull model now? [y/N] " PULL_NOW
    if [ "$PULL_NOW" = "y" ] || [ "$PULL_NOW" = "Y" ]; then
        hafs models pull "$MODEL_ID" --source windows --dest mac
    else
        echo "Exiting. Please make model available and try again."
        exit 1
    fi
fi

# Step 3: Deploy to Ollama
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Deploying to Ollama..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if model already deployed
if ollama list | grep -q "$MODEL_NAME"; then
    echo "✓ Model already deployed to Ollama as '$MODEL_NAME'"
else
    echo "Deploying model to Ollama (this may take a few minutes)..."
    hafs models deploy "$MODEL_ID" ollama \
        --name "$MODEL_NAME" \
        --quantization "$QUANTIZATION"
    echo "✓ Model deployed successfully"
fi

# Step 4: Run test prompts
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Running test prompts..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_test() {
    local category="$1"
    local prompt="$2"

    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────────┐"
    echo "│ Category: $category"
    echo "├────────────────────────────────────────────────────────────────────────────┤"
    echo "│ Prompt: $prompt"
    echo "└────────────────────────────────────────────────────────────────────────────┘"
    echo ""

    ollama run "$MODEL_NAME" "$prompt" 2>/dev/null || {
        echo "[Error: Model failed to respond]"
        return 1
    }

    echo ""
}

# Test across different Oracle domains
run_test "General Knowledge" "What is Oracle of Secrets?"

run_test "Secrets System" "Explain how the secrets system works in Oracle of Secrets. What makes it different from the original Zelda games?"

run_test "Game Design" "I want to add a new dungeon secret that requires collecting 3 special rings. How should I design this?"

run_test "ASM/Technical" "Write 65816 assembly code to add a new item to the player's inventory"

run_test "YAZE Integration" "How do you use YAZE to edit Oracle of Secrets ROM? Walk me through the basic workflow."

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║                              Test Complete!                                    ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $MODEL_NAME"
echo "Deployment: Ollama ($QUANTIZATION)"
echo ""
echo "Next steps:"
echo "  • Chat interactively:     ollama run $MODEL_NAME"
echo "  • View training logs:     hafs training logs"
echo "  • Compare with oracle-rauru: scripts/compare_oracle_models.sh"
echo "  • Run full test suite:    scripts/run_oracle_test_suite.sh"
echo ""
