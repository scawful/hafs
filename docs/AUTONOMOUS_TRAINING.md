# Autonomous Training Campaign System

**Created**: 2025-12-21
**Status**: ✓ FULLY IMPLEMENTED
**Purpose**: Autonomous monitoring, validation, and launch of training data generation campaigns

---

## Overview

The autonomous training system consists of background agents that:

1. **Monitor** pilot generation quality in real-time
2. **Validate** pilot results against quality thresholds
3. **Launch** full 34.5K sample campaign if validation passes
4. **Monitor** campaign progress until completion

This is a fully autonomous workflow - once started, it requires no human intervention.

---

## Quick Start

### Option 1: Bash Launcher (Recommended for Background)

```bash
# Launch autonomous workflow in background
./scripts/launch_autonomous_training.sh

# With custom configuration
QUALITY_THRESHOLD=0.80 CAMPAIGN_TARGET=50000 ./scripts/launch_autonomous_training.sh

# Disable auto-launch (manual approval required)
AUTO_LAUNCH=false ./scripts/launch_autonomous_training.sh
```

### Option 2: Python Launcher (More Control)

```bash
# Launch autonomous workflow (foreground)
python scripts/launch_training.py auto

# Launch in background
python scripts/launch_training.py auto --background

# Monitor pilot only
python scripts/launch_training.py monitor

# Validate pilot only
python scripts/launch_training.py validate

# Launch campaign only (skip monitoring)
python scripts/launch_training.py launch

# Show current status
python scripts/launch_training.py status

# Tail all logs
python scripts/launch_training.py logs
```

### Option 3: Direct Orchestrator

```bash
# Run orchestrator directly
PYTHONPATH=src python -m agents.autonomous.training_orchestrator \
    --mode auto \
    --quality-threshold 0.75 \
    --campaign-target 34500
```

---

## Architecture

### Component Overview

```
┌────────────────────────────────────────────────────────────┐
│                 Training Orchestrator                      │
│         (Coordinates autonomous workflow)                  │
└────────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────┼───────────────────────┐
    ↓                       ↓                       ↓
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Pilot     │     │  Campaign    │     │   Campaign      │
│   Quality   │  →  │  Validator   │  →  │   Launcher      │
│   Monitor   │     │              │     │                 │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                  ↓
                                         ┌─────────────────┐
                                         │   Campaign      │
                                         │   Monitor       │
                                         └─────────────────┘
```

### Agents

| Agent | Purpose | Location |
|-------|---------|----------|
| PilotQualityMonitor | Monitor pilot generation, extract quality metrics | `agents/autonomous/pilot_quality_monitor.py` |
| CampaignValidator | Validate pilot results, approve campaign launch | `agents/autonomous/pilot_quality_monitor.py` |
| CampaignLauncher | Launch full generation campaign | `agents/autonomous/campaign_launcher.py` |
| CampaignMonitor | Monitor campaign progress | `agents/autonomous/campaign_launcher.py` |
| TrainingOrchestrator | Coordinate entire workflow | `agents/autonomous/training_orchestrator.py` |

---

## Workflow

### Phase 1: Pilot Monitoring

The **PilotQualityMonitor** agent:
- Tails pilot generation log (`~/.context/training/pilot_campaign.log`)
- Extracts progress: `N/190` samples
- Extracts quality metrics (quality scores, acceptance rate)
- Checks every 10 seconds
- Saves status to `~/.context/training/pilot_monitor_status.json`

**Continues until**: Pilot reaches 190/190 samples

### Phase 2: Validation

The **CampaignValidator** agent:
- Loads pilot monitoring status
- Checks validation criteria:
  - ✓ Pilot complete (190/190)
  - ✓ Minimum samples (≥150)
  - ✓ Quality threshold (≥0.75)
- Makes launch decision
- Saves result to `~/.context/training/campaign_validation.json`

**Outcomes**:
- **APPROVED**: Proceed to Phase 3
- **REJECTED**: Stop workflow, require manual review

### Phase 3: Campaign Launch

The **CampaignLauncher** agent:
- Checks if campaign already running
- Builds launch command for 34.5K generation
- Launches campaign as background process
- Saves PID and status
- Saves launch result to `~/.context/training/campaign_status.json`

**Campaign command**:
```bash
python -m agents.training.scripts.generate_campaign \
    --target 34500 \
    --output-name full_campaign_34500 \
    --parallel \
    --use-active-learning
```

### Phase 4: Campaign Monitoring

The **CampaignMonitor** agent:
- Tails campaign log (`~/.context/training/full_campaign.log`)
- Extracts progress: `N/34500` samples
- Checks every 60 seconds
- Updates status file
- Continues until completion

**Workflow complete**: Campaign reaches 34,500/34,500 samples

---

## Configuration

### Environment Variables (Bash Launcher)

```bash
# Quality threshold (default: 0.75)
QUALITY_THRESHOLD=0.80

# Campaign target samples (default: 34500)
CAMPAIGN_TARGET=50000

# Auto-launch campaign if validation passes (default: true)
AUTO_LAUNCH=true
```

### Command-Line Arguments (Python Launcher)

```bash
--quality-threshold 0.75     # Minimum pilot quality
--campaign-target 34500      # Full campaign sample count
--no-auto-launch            # Require manual approval
--background                # Run in background (detached)
```

### Programmatic Configuration

```python
from agents.autonomous.training_orchestrator import TrainingOrchestrator

orchestrator = TrainingOrchestrator(
    quality_threshold=0.80,        # Stricter quality requirement
    min_pilot_samples=180,         # Require 180/190 samples minimum
    campaign_target=50000,         # Generate 50K samples
    auto_launch=True,              # Auto-launch if validated
)

await orchestrator.setup()
result = await orchestrator.run_autonomous()
```

---

## Monitoring

### Status Files

All status files are in `~/.context/training/`:

| File | Content |
|------|---------|
| `pilot_monitor_status.json` | Pilot progress and quality metrics |
| `campaign_validation.json` | Validation decision and reasoning |
| `campaign_status.json` | Campaign launch status and PID |
| `campaign_monitor_status.json` | Campaign progress |
| `orchestrator_state.json` | Overall workflow state |

### Log Files

All logs are in `~/.context/logs/` and `~/.context/training/`:

| File | Content |
|------|---------|
| `~/.context/logs/training_orchestrator.log` | Orchestrator activity |
| `~/.context/training/pilot_campaign.log` | Pilot generation output |
| `~/.context/training/full_campaign.log` | Full campaign output |

### Live Monitoring

```bash
# Watch orchestrator
tail -f ~/.context/logs/training_orchestrator.log

# Watch pilot
tail -f ~/.context/training/pilot_campaign.log

# Watch campaign
tail -f ~/.context/training/full_campaign.log

# Watch all
tail -f ~/.context/logs/training_orchestrator.log \
         ~/.context/training/pilot_campaign.log \
         ~/.context/training/full_campaign.log

# Or use Python launcher
python scripts/launch_training.py logs
```

### Check Status

```bash
# Bash
cat ~/.context/training/orchestrator_state.json | jq

# Python launcher
python scripts/launch_training.py status

# Direct check
jq . ~/.context/training/pilot_monitor_status.json
jq . ~/.context/training/campaign_validation.json
jq . ~/.context/training/campaign_status.json
```

---

## Control

### Stop Workflow

```bash
# Find orchestrator PID
cat ~/.context/training/orchestrator.pid

# Stop orchestrator
kill $(cat ~/.context/training/orchestrator.pid)

# Stop campaign (if running)
kill $(cat ~/.context/training/campaign.pid)
```

### Manual Validation

If auto-launch is disabled, manually approve:

```bash
# 1. Wait for pilot to complete
python scripts/launch_training.py monitor

# 2. Review validation
python scripts/launch_training.py validate

# 3. If approved, launch manually
python scripts/launch_training.py launch
```

---

## Example Workflow

### Scenario: Launch Autonomous Training

```bash
# 1. Start autonomous workflow
./scripts/launch_autonomous_training.sh

# Output:
# ========================================================================
# AUTONOMOUS WORKFLOW ACTIVE
# ========================================================================
#
# The orchestrator is now running autonomously and will:
#   1. Monitor pilot generation (currently in progress)
#   2. Validate pilot quality when complete
#   3. Auto-launch full 34.5K campaign if quality passes threshold
#   4. Monitor campaign until completion
```

### Scenario: Monitor Progress

```bash
# Check current status
python scripts/launch_training.py status

# Output:
# ========================================================================
# TRAINING WORKFLOW STATUS
# ========================================================================
#
# Orchestrator:
# {
#   "phase": "monitoring_pilot",
#   "started": "2025-12-21T06:50:00",
#   "pilot_result": {
#     "progress": 87,
#     "total": 190,
#     "estimated_quality": 0.82
#   }
# }
```

### Scenario: Tail Logs

```bash
# Watch all activity
python scripts/launch_training.py logs

# Output (live):
# 2025-12-21 06:50:15 [PilotQualityMonitor] INFO: Progress: 89/190 (46.8%) | Quality: 0.821 | Acceptance: 87.00%
# 2025-12-21 06:50:25 [PilotQualityMonitor] INFO: Progress: 90/190 (47.4%) | Quality: 0.819 | Acceptance: 86.50%
```

---

## Troubleshooting

### "Campaign already running"

**Cause**: Previous campaign still active

**Solution**:
```bash
# Check if running
ps -p $(cat ~/.context/training/campaign.pid)

# Stop if necessary
kill $(cat ~/.context/training/campaign.pid)

# Remove stale PID
rm ~/.context/training/campaign.pid
```

### "No pilot status available"

**Cause**: Pilot hasn't started or log file missing

**Solution**:
```bash
# Check pilot log exists
ls -lh ~/.context/training/pilot_campaign.log

# Verify pilot is running (background bash bc1ef76)
# Look for "Progress:" in log
tail ~/.context/training/pilot_campaign.log
```

### "Validation failed"

**Cause**: Pilot quality below threshold or incomplete

**Solution**:
```bash
# Check validation details
cat ~/.context/training/campaign_validation.json | jq

# Review pilot quality
cat ~/.context/training/pilot_monitor_status.json | jq '.estimated_quality'

# If quality is borderline, adjust threshold:
QUALITY_THRESHOLD=0.70 ./scripts/launch_autonomous_training.sh
```

### Campaign doesn't launch

**Cause**: Auto-launch disabled or validation failed

**Solution**:
```bash
# Check validation
python scripts/launch_training.py validate

# If approved, launch manually
python scripts/launch_training.py launch
```

---

## Integration with hafs

### Background Daemon Integration

The training orchestrator can run as a hafs background service:

```python
# In hafs autonomy daemon
from agents.autonomous.training_orchestrator import TrainingOrchestrator

# Add training task
training_task = AutonomyTask(
    name="training_campaign",
    task_type="training",
    interval_seconds=0,  # Run once
    enabled=True,
)

# Execute
orchestrator = TrainingOrchestrator()
await orchestrator.setup()
await orchestrator.run_autonomous()
```

### Status Integration

Training status integrates with hafs dashboard:

```python
# Read status in TUI
from pathlib import Path
import json

status_file = Path.home() / ".context/training/orchestrator_state.json"
if status_file.exists():
    status = json.loads(status_file.read_text())
    # Display in dashboard
```

---

## Next Steps After Campaign Completes

Once the autonomous workflow completes:

1. **Review Generated Data**:
   ```bash
   ls -lh ~/.context/training/datasets/
   ```

2. **Export Datasets**:
   ```bash
   python -m agents.training.exporter \
       --input full_campaign_34500 \
       --output alttp_asm_24k yaze_tools_7k \
       --format qwen
   ```

3. **Transfer to medical-mechanica**:
   ```bash
   rsync -avz ~/.context/training/datasets/ \
       medical-mechanica:~/training/datasets/
   ```

4. **Train Models**:
   ```bash
   ssh medical-mechanica
   cd ~/training
   python train_alttp_asm.py
   python train_yaze_tools.py
   ```

---

## Summary

**What**: Autonomous system for training data generation
**How**: Background agents monitor, validate, and launch campaigns
**Why**: Zero-touch workflow from pilot to full 34.5K generation
**Status**: ✓ READY TO USE

**Created**: 2025-12-21
**Components**: 5 agents, 2 launchers, full orchestration
**Test Status**: Pilot currently running (47/190, 25%)

---

**Quick Launch**:
```bash
./scripts/launch_autonomous_training.sh
```

**Monitor**:
```bash
python scripts/launch_training.py status
python scripts/launch_training.py logs
```

**Stop**:
```bash
kill $(cat ~/.context/training/orchestrator.pid)
```
