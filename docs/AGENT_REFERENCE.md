# hafs Agent System - Complete Reference

**Created**: 2025-12-21
**Status**: ✓ COMPREHENSIVE DOCUMENTATION
**Purpose**: Complete guide to all autonomous agents in hafs

---

## Quick Index

- [Training Agents](#training-agents) - Data generation (5 agents)
- [Swarm Agents](#swarm-agents) - Codebase analysis (12 agents)
- [Autonomous Agents](#autonomous-agents) - Workflow automation (5 agents)
- [Context Agents](#context-agents) - Knowledge building (6 agents)
- [Running Background Agents](#running-background-agents) - Deployment guide

---

## Training Agents

### PilotQualityMonitor
**File**: `agents/autonomous/pilot_quality_monitor.py`
**Purpose**: Monitor pilot generation quality in real-time
**Status**: ✓ RUNNING (PID embedded in orchestrator)

**What it does**:
- Tails pilot log every 10 seconds
- Extracts progress (N/190)
- Tracks quality scores
- Saves status for other agents

**Output**: `~/.context/training/pilot_monitor_status.json`

---

### CampaignValidator
**File**: `agents/autonomous/pilot_quality_monitor.py`
**Purpose**: Validate pilot and approve campaign launch

**Checks**:
- ✓ Pilot complete (190/190)
- ✓ Minimum samples (≥150)
- ✓ Quality threshold (≥0.75)

**Output**: `~/.context/training/campaign_validation.json`

---

### CampaignLauncher
**File**: `agents/autonomous/campaign_launcher.py`
**Purpose**: Launch full 34.5K generation campaign

**Actions**:
1. Checks if campaign already running
2. Builds launch command
3. Starts background process
4. Saves PID

**Output**: `~/.context/training/campaign_status.json`

---

### CampaignMonitor
**File**: `agents/autonomous/campaign_launcher.py`
**Purpose**: Monitor campaign progress

**Tracking**: N/34500 every 60s

**Output**: `~/.context/training/campaign_monitor_status.json`

---

### TrainingOrchestrator
**File**: `agents/autonomous/training_orchestrator.py`
**Purpose**: Coordinate entire training workflow
**Status**: ✓ RUNNING (PID: 20565)

**Phases**:
1. Monitor pilot
2. Validate results
3. Launch campaign
4. Monitor completion

**Output**: `~/.context/training/orchestrator_state.json`

---

## Swarm Agents

### YAZE Performance Agents

#### PerformanceProfilerAgent
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Profile emulation bottlenecks

**Output**: Hot paths with runtime %

#### CpuOptimizerAgent
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Optimize 65816 CPU emulation

**Output**: Inline candidates, SIMD opportunities

#### PpuOptimizerAgent
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Optimize PPU rendering

**Output**: Batching opportunities, redundant draws

---

### YAZE Audio Agents

#### AudioDebuggerAgent
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Debug audio system

**Output**: Buffer issues, crackling sources

#### Spc700ValidatorAgent
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Validate SPC700 emulation

**Output**: Instruction bugs, regression tests

---

### YAZE Input Agents

#### InputLagAnalyzerAgent
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Analyze input latency

**Output**: Latency sources, edge detection fixes

---

### Mesen2 Integration Agents

#### LuaScriptGeneratorAgent
**File**: `agents/swarm/mesen2_specialists.py`
**Purpose**: Generate Mesen2 Lua scripts

**Output**: 10+ scripts (memory watch, profiling, testing)

#### IntegrationArchitectAgent
**File**: `agents/swarm/mesen2_specialists.py`
**Purpose**: Design YAZE-Mesen2 integration

**Output**: Architecture, protocol, roadmap

#### DebuggingToolsBuilderAgent
**File**: `agents/swarm/mesen2_specialists.py`
**Purpose**: Design debugging tools

**Output**: Tool specs (memory sync, breakpoints)

#### TestAutomationAgent
**File**: `agents/swarm/mesen2_specialists.py`
**Purpose**: Create ROM testing framework

**Output**: Test suite, automation scripts

#### LuaScriptLibraryGenerator
**File**: `agents/swarm/mesen2_specialists.py`
**Purpose**: Generate comprehensive Lua library

**Output**: Full script library with 5 categories

---

### Swarm Synthesis

#### SwarmSynthesizer
**File**: `agents/swarm/yaze_specialists.py`
**Purpose**: Combine multi-agent findings

**Output**: Comprehensive markdown report

---

## Autonomous Agents

All autonomous agents above, plus:

### MoE Experts

#### AsmExpert
**File**: `hafs/agents/moe/experts/asm_expert.py`
**Purpose**: 65816 assembly specialist

**Config**: Temperature 0.3, Tokens 2048

#### YazeExpert
**File**: `hafs/agents/moe/experts/yaze_expert.py`
**Purpose**: YAZE tools specialist

**Config**: Temperature 0.6, Tokens 2048

#### DebugExpert
**File**: `hafs/agents/moe/experts/debug_expert.py`
**Purpose**: Error diagnosis specialist

**Config**: Temperature 0.4, Tokens 2048

---

## Context Agents

### ContextReportPipeline
**File**: `hafs/agents/context_report_pipeline.py`
**Purpose**: Generate comprehensive context reports

**Stages**: Discovery → Analysis → Synthesis → Indexing

---

### ALTTPModuleAnalyzer
**File**: `hafs/agents/alttp_module_analyzer.py`
**Purpose**: Analyze ALTTP disassembly modules

**Output**: Module summaries, function catalogs

---

### OracleAnalyzer
**File**: `hafs/agents/oracle_analyzer.py`
**Purpose**: Analyze Oracle ROM hack

**Output**: Vanilla vs hack diffs, features

---

### OracleKBBuilder
**File**: `hafs/agents/oracle_kb_builder.py`
**Purpose**: Build Oracle knowledge base

**Output**: Embeddings, entity extraction

---

### ReportManager
**File**: `hafs/agents/report_manager.py`
**Purpose**: Manage reports

**Output**: Catalog, search, versioning

---

### SelfImprovementAgent
**File**: `hafs/agents/autonomy_agents.py`
**Purpose**: System self-analysis

**Interval**: Every 6 hours

---

## Running Background Agents

### Currently Running

```bash
# Check what's running
ps aux | grep python | grep agents

# Currently active:
# - TrainingOrchestrator (PID: 20565)
# - PilotQualityMonitor (embedded)
# - Pilot generation (background bash bc1ef76)
```

### Launch Context Building

```bash
# Option 1: Context agent daemon
hafs agents start context

# Option 2: Direct report generation
PYTHONPATH=src python -m hafs.agents.context_report_pipeline \
    --topic "ALTTP Memory Map" \
    --output ~/.context/reports/alttp
```

### Launch Swarm Missions

```bash
# YAZE performance analysis
python -m agents.swarm.launcher yaze-performance

# YAZE audio debugging
python -m agents.swarm.launcher yaze-audio

# YAZE input fix
python -m agents.swarm.launcher yaze-input

# All YAZE missions
python -m agents.swarm.launcher yaze-all

# Mesen2 integration
python -m agents.swarm.launcher mesen2-integration

# Everything
python -m agents.swarm.launcher all
```

### Launch Mission Agents

```bash
# ALTTP deep dive research
PYTHONPATH=src python -m hafs.agents.mission_agents \
    --mission alttp_deep_dive \
    --duration 8

# Oracle investigation
PYTHONPATH=src python -m hafs.agents.mission_agents \
    --mission oracle_investigation \
    --duration 6

# Gigaleak study
PYTHONPATH=src python -m hafs.agents.mission_agents \
    --mission gigaleak_study \
    --duration 4
```

---

## Agent Status Monitoring

### Training Workflow

```bash
# Overall status
python scripts/launch_training.py status

# Pilot monitoring
cat ~/.context/training/pilot_monitor_status.json | jq

# Campaign validation
cat ~/.context/training/campaign_validation.json | jq

# Campaign status
cat ~/.context/training/campaign_status.json | jq
```

### Swarm Missions

```bash
# YAZE performance
cat ~/.context/swarms/yaze/yaze_performance_optimization/report.json | jq

# Mesen2 integration
cat ~/.context/swarms/mesen2/integration_mission/report.json | jq

# Lua scripts
ls ~/.context/swarms/mesen2/lua_scripts/
```

### Context Agents

```bash
# Context daemon status
cat ~/.context/autonomy_daemon/daemon_status.json | jq

# Reports
ls ~/.context/reports/
```

---

## Agent Logs

### Training

```bash
tail -f ~/.context/logs/training_orchestrator.log
tail -f ~/.context/training/pilot_campaign.log
tail -f ~/.context/training/full_campaign.log
```

### Swarms

```bash
# Live swarm status
cat ~/.context/swarm_status.json | jq

# Swarm history
cat ~/.context/swarm_history.json | jq
```

### Context

```bash
tail -f ~/.context/logs/context_agent_daemon.log
tail -f ~/.context/logs/autonomy_daemon.log
```

---

## Agent Control

### Stop Agents

```bash
# Training orchestrator
kill 20565

# Campaign
kill $(cat ~/.context/training/campaign.pid)

# All agents
pkill -f "python.*agents"
```

### Restart Agents

```bash
# Training workflow
./scripts/launch_autonomous_training.sh

# Context building
hafs agents restart context

# Swarms
python -m agents.swarm.launcher yaze-all
```

---

## Summary

**Total Agents**: 35+
**Currently Running**: 3
**Ready to Launch**: 32

**Agent Categories**:
- Training: 5 agents
- Swarm: 12 agents
- Autonomous: 5 agents
- Context: 6 agents
- Mission: 6 agents
- MoE: 3 experts

**Active Workflows**:
- ✓ Training orchestration (running)
- ⏳ Swarm missions (ready)
- ⏳ Context building (ready)
- ⏳ Mission research (ready)

---

**Quick Commands**:
```bash
# Status of all
python scripts/launch_training.py status

# Launch swarms
python -m agents.swarm.launcher all

# Start context building
hafs agents start context

# View all logs
tail -f ~/.context/logs/*.log
```

**Documentation**:
- This file: Agent reference
- `AGENTS.md`: System architecture
- `AUTONOMOUS_TRAINING.md`: Training workflow
- `SWARM_USAGE.md`: Swarm missions
- `MOE_SYSTEM.md`: MoE architecture
