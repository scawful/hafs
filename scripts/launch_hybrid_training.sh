#!/bin/bash
# Launch Hybrid GPU + API Training Campaign
#
# Intelligently routes generation between:
# - Local GPU (medical-mechanica 5060TI) - FREE
# - Gemini API - PAID
#
# Routing logic:
# - GPU <70% util → Use GPU (FREE)
# - GPU 70-90% util → Mix GPU + API
# - GPU >90% util → Use API only
#
# This maximizes GPU usage (saves $$$) while preventing overload.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAFS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================================"
echo "HYBRID GPU + API TRAINING CAMPAIGN"
echo "========================================================================"
echo "hafs root: $HAFS_ROOT"
echo ""

# Configuration
TARGET=${1:-34500}
RESUME=${2:-true}
PILOT=${3:-false}

echo "Configuration:"
echo "  Target: $TARGET samples"
echo "  Resume: $RESUME"
echo "  Pilot mode: $PILOT"
echo ""

# Set Python path
export PYTHONPATH="$HAFS_ROOT/src:$PYTHONPATH"

# Log directory
LOG_DIR="$HOME/.context/logs"
mkdir -p "$LOG_DIR"

CAMPAIGN_LOG="$LOG_DIR/campaign_hybrid_$(date +%Y%m%d_%H%M%S).log"

echo "Checking GPU status..."
cd "$HAFS_ROOT"

# Test GPU connection
PYTHONPATH=src .venv/bin/python -m agents.training.hybrid_orchestrator

echo ""
echo "Launching hybrid campaign..."
echo "  Log: $CAMPAIGN_LOG"
echo ""

# Build command
CMD_ARGS="--target $TARGET"

if [ "$RESUME" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --resume"
fi

if [ "$PILOT" = "true" ]; then
    CMD_ARGS="$CMD_ARGS --pilot"
fi

# Launch campaign
cd "$HAFS_ROOT"

nohup .venv/bin/python -m agents.training.scripts.hybrid_campaign \
    $CMD_ARGS \
    > "$CAMPAIGN_LOG" 2>&1 &

CAMPAIGN_PID=$!

echo "✓ Hybrid campaign started (PID: $CAMPAIGN_PID)"
echo ""
echo "========================================================================"
echo "HYBRID MODE ACTIVE"
echo "========================================================================"
echo ""
echo "The campaign will intelligently route requests:"
echo "  🟢 GPU <70% load  → Use GPU (FREE)"
echo "  🟡 GPU 70-90%     → Mix GPU + API (optimize cost)"
echo "  🔴 GPU >90%       → Use API only (prevent overload)"
echo ""
echo "Monitoring:"
echo "  Campaign log: tail -f $CAMPAIGN_LOG"
echo "  Training status: hafs training status"
echo "  GPU load: ssh medical-mechanica 'nvidia-smi -l 5'"
echo ""
echo "Control:"
echo "  Stop campaign: hafs training stop"
echo "  Check status: ps -p $CAMPAIGN_PID"
echo ""
echo "Expected cost savings: 50-80% vs API-only"
echo "========================================================================"
