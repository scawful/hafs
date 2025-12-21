#!/bin/bash
# Launch Background Agents for Overnight Context Building
#
# This script launches multiple agents that will run overnight to build
# comprehensive context, generate reports, and prepare knowledge bases.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAFS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================================================"
echo "OVERNIGHT BACKGROUND AGENTS LAUNCHER"
echo "========================================================================"
echo "hafs root: $HAFS_ROOT"
echo "Time: $(date)"
echo ""

# Set Python path
export PYTHONPATH="$HAFS_ROOT/src:$PYTHONPATH"

# Log directory
LOG_DIR="$HOME/.context/logs"
mkdir -p "$LOG_DIR"

# PID directory
PID_DIR="$HOME/.context/pids"
mkdir -p "$PID_DIR"

# Function to launch agent
launch_agent() {
    local name=$1
    local command=$2
    local log_file="$LOG_DIR/${name}.log"
    local pid_file="$PID_DIR/${name}.pid"

    echo "Launching $name..."
    echo "  Command: $command"
    echo "  Log: $log_file"

    # Launch in background
    cd "$HAFS_ROOT"
    nohup bash -c "$command" > "$log_file" 2>&1 &
    local pid=$!

    # Save PID
    echo "$pid" > "$pid_file"

    echo "  ✓ Started (PID: $pid)"
    echo ""
}

echo "Launching background agents for overnight context building..."
echo ""

# ========================================
# 1. ALTTP Module Analysis
# ========================================
launch_agent "alttp_module_analyzer" \
    "python3 -m hafs.agents.alttp_module_analyzer --output ~/.context/reports/alttp/modules"

# ========================================
# 2. Oracle of Secrets Analysis
# ========================================
launch_agent "oracle_analyzer" \
    "python3 -m hafs.agents.oracle_analyzer --output ~/.context/reports/oracle"

# ========================================
# 3. Oracle Knowledge Base Building
# ========================================
launch_agent "oracle_kb_builder" \
    "python3 -m hafs.agents.oracle_kb_builder --rebuild"

# ========================================
# 4. YAZE Performance Swarm
# ========================================
launch_agent "yaze_performance_swarm" \
    "python3 -m agents.swarm.launcher yaze-performance"

# ========================================
# 5. YAZE Audio Swarm
# ========================================
launch_agent "yaze_audio_swarm" \
    "python3 -m agents.swarm.launcher yaze-audio"

# ========================================
# 6. YAZE Input Swarm
# ========================================
launch_agent "yaze_input_swarm" \
    "python3 -m agents.swarm.launcher yaze-input"

# ========================================
# 7. Mesen2 Integration Swarm
# ========================================
launch_agent "mesen2_integration_swarm" \
    "python3 -m agents.swarm.launcher mesen2-integration"

# ========================================
# 8. Context Report Pipeline - ALTTP
# ========================================
launch_agent "context_report_alttp" \
    "python3 -m hafs.agents.context_report_pipeline --topic 'ALTTP Complete Analysis' --output ~/.context/reports/alttp/complete"

# ========================================
# 9. Context Report Pipeline - Oracle
# ========================================
launch_agent "context_report_oracle" \
    "python3 -m hafs.agents.context_report_pipeline --topic 'Oracle of Secrets Complete' --output ~/.context/reports/oracle/complete"

# ========================================
# 10. Context Report Pipeline - Gigaleak
# ========================================
launch_agent "context_report_gigaleak" \
    "python3 -m hafs.agents.context_report_pipeline --topic 'Nintendo Gigaleak Analysis' --output ~/.context/reports/gigaleak"

# ========================================
# 11. MoE Test Suite
# ========================================
launch_agent "moe_test" \
    "python3 -m hafs.agents.moe.test_moe > ~/.context/logs/moe_test_$(date +%Y%m%d).log 2>&1"

# ========================================
# 12. Embedding Service Rebuild
# ========================================
launch_agent "embedding_rebuild" \
    "python3 -m hafs.services.embedding_daemon --rebuild-all"

echo "========================================================================"
echo "BACKGROUND AGENTS LAUNCHED"
echo "========================================================================"
echo ""
echo "Launched Agents:"
echo "  1. ALTTP Module Analyzer"
echo "  2. Oracle Analyzer"
echo "  3. Oracle KB Builder"
echo "  4. YAZE Performance Swarm"
echo "  5. YAZE Audio Swarm"
echo "  6. YAZE Input Swarm"
echo "  7. Mesen2 Integration Swarm"
echo "  8. Context Report - ALTTP"
echo "  9. Context Report - Oracle"
echo " 10. Context Report - Gigaleak"
echo " 11. MoE Test Suite"
echo " 12. Embedding Service Rebuild"
echo ""
echo "Plus already running:"
echo "  - Training Orchestrator (PID: $(cat ~/.context/training/orchestrator.pid 2>/dev/null || echo 'N/A'))"
echo "  - Pilot Generation (background)"
echo ""
echo "Total Active Agents: 14+"
echo ""
echo "========================================================================"
echo "MONITORING"
echo "========================================================================"
echo ""
echo "Check status:"
echo "  python $HAFS_ROOT/scripts/launch_training.py status"
echo ""
echo "View PIDs:"
echo "  ls ~/.context/pids/"
echo ""
echo "Tail logs:"
echo "  tail -f ~/.context/logs/*.log"
echo ""
echo "Stop all agents:"
echo "  $HAFS_ROOT/scripts/stop_night_agents.sh"
echo ""
echo "========================================================================"
echo "ESTIMATED COMPLETION"
echo "========================================================================"
echo ""
echo "  ALTTP Module Analysis: ~2 hours"
echo "  Oracle Analysis: ~1.5 hours"
echo "  YAZE Swarms (3): ~4 hours total"
echo "  Mesen2 Integration: ~2 hours"
echo "  Context Reports (3): ~3 hours"
echo "  Embedding Rebuild: ~1 hour"
echo "  Training Workflow: ~21-26 hours (ongoing)"
echo ""
echo "Total estimated time: ~8-10 hours for context building"
echo "                     ~24-28 hours for complete training"
echo ""
echo "========================================================================"
