#!/bin/bash
# Stop All Background Night Agents

set -e

PID_DIR="$HOME/.context/pids"

echo "========================================================================"
echo "STOPPING BACKGROUND AGENTS"
echo "========================================================================"
echo ""

if [ ! -d "$PID_DIR" ]; then
    echo "No PID directory found - no agents to stop"
    exit 0
fi

# Count agents
agent_count=$(ls "$PID_DIR"/*.pid 2>/dev/null | wc -l | tr -d ' ')

if [ "$agent_count" -eq 0 ]; then
    echo "No agent PID files found"
    exit 0
fi

echo "Found $agent_count agent(s) to stop"
echo ""

# Stop each agent
for pid_file in "$PID_DIR"/*.pid; do
    if [ -f "$pid_file" ]; then
        agent_name=$(basename "$pid_file" .pid)
        pid=$(cat "$pid_file")

        echo "Stopping $agent_name (PID: $pid)..."

        # Check if process exists
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            echo "  ✓ Stopped"
        else
            echo "  ⊘ Not running (stale PID)"
        fi

        # Remove PID file
        rm "$pid_file"
    fi
done

echo ""
echo "========================================================================"
echo "ALL AGENTS STOPPED"
echo "========================================================================"
