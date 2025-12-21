#!/bin/bash
# Launch Autonomous Training Campaign
#
# This script starts the autonomous training workflow in the background:
# 1. Monitors pilot generation quality
# 2. Validates pilot results
# 3. Launches full 34.5K campaign if validation passes
# 4. Monitors campaign progress

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAFS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================================"
echo "AUTONOMOUS TRAINING CAMPAIGN LAUNCHER"
echo "========================================================================"
echo "hafs root: $HAFS_ROOT"
echo ""

# Configuration
QUALITY_THRESHOLD=${QUALITY_THRESHOLD:-0.75}
CAMPAIGN_TARGET=${CAMPAIGN_TARGET:-34500}
AUTO_LAUNCH=${AUTO_LAUNCH:-true}

echo "Configuration:"
echo "  Quality threshold: $QUALITY_THRESHOLD"
echo "  Campaign target: $CAMPAIGN_TARGET samples"
echo "  Auto-launch: $AUTO_LAUNCH"
echo ""

# Set Python path
export PYTHONPATH="$HAFS_ROOT/src:$PYTHONPATH"

# Log directory
LOG_DIR="$HOME/.context/logs"
mkdir -p "$LOG_DIR"

ORCHESTRATOR_LOG="$LOG_DIR/training_orchestrator.log"
PID_FILE="$HOME/.context/training/orchestrator.pid"

mkdir -p "$(dirname "$PID_FILE")"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "ERROR: Training orchestrator already running (PID: $PID)"
        echo "  Log: $ORCHESTRATOR_LOG"
        echo "  Status: tail -f $ORCHESTRATOR_LOG"
        exit 1
    else
        echo "Removing stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Build command
CMD_ARGS="--quality-threshold $QUALITY_THRESHOLD --campaign-target $CAMPAIGN_TARGET"

if [ "$AUTO_LAUNCH" != "true" ]; then
    CMD_ARGS="$CMD_ARGS --no-auto-launch"
fi

# Launch orchestrator in background
echo "Launching training orchestrator..."
echo "  Command: python -m agents.autonomous.training_orchestrator --mode auto $CMD_ARGS"
echo "  Log: $ORCHESTRATOR_LOG"
echo ""

cd "$HAFS_ROOT"

nohup python3 -m agents.autonomous.training_orchestrator \
    --mode auto \
    $CMD_ARGS \
    > "$ORCHESTRATOR_LOG" 2>&1 &

ORCHESTRATOR_PID=$!

# Save PID
echo "$ORCHESTRATOR_PID" > "$PID_FILE"

echo "✓ Training orchestrator started (PID: $ORCHESTRATOR_PID)"
echo ""
echo "========================================================================"
echo "AUTONOMOUS WORKFLOW ACTIVE"
echo "========================================================================"
echo ""
echo "The orchestrator is now running autonomously and will:"
echo "  1. Monitor pilot generation (currently in progress)"
echo "  2. Validate pilot quality when complete"
echo "  3. Auto-launch full 34.5K campaign if quality passes threshold"
echo "  4. Monitor campaign until completion"
echo ""
echo "Monitoring:"
echo "  Live log: tail -f $ORCHESTRATOR_LOG"
echo "  Pilot status: cat ~/.context/training/pilot_monitor_status.json"
echo "  Validation: cat ~/.context/training/campaign_validation.json"
echo "  Campaign status: cat ~/.context/training/campaign_status.json"
echo ""
echo "Control:"
echo "  Stop orchestrator: kill $ORCHESTRATOR_PID"
echo "  Check status: ps -p $ORCHESTRATOR_PID"
echo ""
echo "========================================================================"
