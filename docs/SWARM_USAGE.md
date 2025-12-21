# Swarm System Usage Guide

**Created**: 2025-12-21
**Status**: ✓ IMPLEMENTED
**Components**: YAZE swarm, Mesen2 swarm, unified launcher

---

## Overview

The swarm system deploys specialized AI agents in parallel to analyze, debug, and improve codebases. Two swarm coordinators are available:

1. **YAZE Swarm** - Performance, audio, and input analysis for YAZE ROM editor
2. **Mesen2 Swarm** - Integration design and Lua script generation for Mesen2

---

## Quick Start

### Launch YAZE Performance Mission

```bash
# From hafs root
PYTHONPATH=src python -m agents.swarm.launcher yaze-performance

# Or with custom YAZE path
PYTHONPATH=src python -m agents.swarm.launcher yaze-performance \
    --yaze-path ~/Code/yaze
```

### Launch All YAZE Missions

```bash
PYTHONPATH=src python -m agents.swarm.launcher yaze-all
```

This runs:
1. Performance optimization (2 hours)
2. Audio debugging (1.5 hours)
3. Input fix (1 hour)

### Launch Mesen2 Integration Mission

```bash
PYTHONPATH=src python -m agents.swarm.launcher mesen2-integration \
    --yaze-path ~/Code/yaze \
    --mesen2-path ~/Code/mesen2
```

### Launch Full Campaign (All Missions)

```bash
PYTHONPATH=src python -m agents.swarm.launcher all
```

---

## Mission Details

### YAZE Performance Optimization

**Duration**: ~2 hours
**Agents**: 3 (performance_profiler, cpu_optimizer, ppu_optimizer)
**Focus**:
- CPU emulation loop profiling
- PPU rendering optimization
- Memory allocation analysis

**Output**:
- `~/.context/swarms/yaze/yaze_performance_optimization/report.json`
- `~/.context/swarms/yaze/yaze_performance_optimization/synthesis.md`

**Deliverables**:
- Hot path analysis with file:line references
- Optimization recommendations with estimated speedup
- Patch suggestions prioritized by impact

### YAZE Audio Debugging

**Duration**: ~1.5 hours
**Agents**: 2 (audio_debugger, spc700_validator)
**Focus**:
- Audio buffer underrun detection
- SPC700 emulation validation
- Audio/video sync analysis

**Output**:
- `~/.context/swarms/yaze/yaze_audio_system_debug/report.json`
- `~/.context/swarms/yaze/yaze_audio_system_debug/synthesis.md`

**Deliverables**:
- Buffer issue diagnostics
- SPC700 instruction bugs
- Test cases for validation

### YAZE Input Fix

**Duration**: ~1 hour
**Agents**: 1 (input_lag_analyzer)
**Focus**:
- Input polling latency
- Edge detection accuracy
- Frame-to-input delay

**Output**:
- `~/.context/swarms/yaze/yaze_input_system_fix/report.json`
- `~/.context/swarms/yaze/yaze_input_system_fix/synthesis.md`

**Deliverables**:
- Latency source analysis
- Edge detection fixes
- Input overlay design

### Mesen2 Integration

**Duration**: ~2 hours
**Agents**: 4 (lua_script_generator, integration_architect, debugging_tools_builder, test_automation)
**Focus**:
- Lua script library generation
- YAZE-Mesen2 integration design
- Debugging tools architecture
- Test automation framework

**Output**:
- `~/.context/swarms/mesen2/integration_mission/report.json`
- `~/.context/swarms/mesen2/integration_mission/synthesis.md`
- `~/.context/swarms/mesen2/integration_mission/integration_design.json`
- **Lua scripts**: `~/.context/swarms/mesen2/lua_scripts/`

**Deliverables**:
- 10+ Lua debugging scripts
- Integration architecture (plugin vs fork)
- Debugging tools design
- Test automation framework

---

## Programmatic Usage

### YAZE Swarm

```python
from pathlib import Path
from agents.swarm.yaze_swarm import YazeSwarmCoordinator

async def run_yaze_performance():
    coordinator = YazeSwarmCoordinator(yaze_path=Path("~/Code/yaze"))
    await coordinator.setup()

    results = await coordinator.launch_performance_mission()
    print(results["synthesis"])
```

### Mesen2 Swarm

```python
from pathlib import Path
from agents.swarm.mesen2_swarm import Mesen2SwarmCoordinator

async def run_mesen2_integration():
    coordinator = Mesen2SwarmCoordinator(
        yaze_path=Path("~/Code/yaze"),
        mesen2_path=Path("~/Code/mesen2")
    )
    await coordinator.setup()

    results = await coordinator.launch_integration_mission()
    print(results["synthesis"])
    print(f"Lua scripts: {coordinator.scripts_dir}")
```

### Custom Mission

```python
from agents.swarm.yaze_swarm import YazeSwarmMission, YazeSwarmCoordinator

# Define custom mission
mission = YazeSwarmMission(
    name="Custom Analysis",
    description="Analyze custom YAZE subsystem",
    target_codebase=Path("~/Code/yaze"),
    agents=["performance_profiler", "cpu_optimizer"],
    duration_hours=1.0
)

# Launch
coordinator = YazeSwarmCoordinator()
await coordinator.setup()
results = await coordinator.launch_mission(mission, parallel=True)
```

---

## Agent Descriptions

### YAZE Agents

| Agent | Purpose | Output |
|-------|---------|--------|
| PerformanceProfilerAgent | Profile emulation bottlenecks | Hot paths, runtime percentages |
| CpuOptimizerAgent | Optimize 65816 CPU emulation | Inline candidates, SIMD opportunities |
| PpuOptimizerAgent | Optimize PPU rendering | Redundant draws, batching opportunities |
| AudioDebuggerAgent | Debug audio glitches | Buffer issues, sync problems |
| Spc700ValidatorAgent | Validate SPC700 emulation | Instruction bugs, DSP issues |
| InputLagAnalyzerAgent | Analyze input latency | Latency sources, edge detection fixes |

### Mesen2 Agents

| Agent | Purpose | Output |
|-------|---------|--------|
| LuaScriptGeneratorAgent | Generate Lua debugging scripts | Script code with usage examples |
| IntegrationArchitectAgent | Design YAZE-Mesen2 integration | Architecture, communication protocol |
| DebuggingToolsBuilderAgent | Design debugging tools | Memory inspector, breakpoint manager |
| TestAutomationAgent | Design test framework | Test suite, automation scripts |
| LuaScriptLibraryGenerator | Generate comprehensive script library | 10+ scripts across 5 categories |

### Shared Agent

| Agent | Purpose | Output |
|-------|---------|--------|
| SwarmSynthesizer | Synthesize multi-agent findings | Markdown report with action items |

---

## Configuration

### Parallel Execution

By default, agents run in parallel for faster execution:

```python
results = await coordinator.launch_mission(mission, parallel=True)
```

Set `parallel=False` for sequential execution (useful for debugging).

### Output Directories

Default output locations:
- YAZE: `~/.context/swarms/yaze/{mission_name}/`
- Mesen2: `~/.context/swarms/mesen2/{mission_name}/`
- Lua scripts: `~/.context/swarms/mesen2/lua_scripts/`

### Logging

Enable verbose logging:

```bash
python -m agents.swarm.launcher yaze-performance -v
```

Or in code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Generated Lua Scripts

The Mesen2 integration mission generates Lua scripts in these categories:

1. **Memory Watch** - Monitor custom items, flags, game state
   - `watch_custom_items.lua`
   - `watch_game_flags.lua`
   - `track_player_state.lua`

2. **Performance Profiling** - Track CPU cycles and routine calls
   - `profile_performance.lua`
   - `track_cpu_cycles.lua`
   - `benchmark_routines.lua`

3. **Event Logging** - Log item pickups, room transitions
   - `log_item_pickups.lua`
   - `log_room_transitions.lua`
   - `log_boss_fights.lua`

4. **Input Recording** - Record and playback input sequences
   - `record_input.lua`
   - `playback_input.lua`
   - `tas_tools.lua`

5. **Automated Testing** - Validate ROM hack features
   - `test_suite.lua`
   - `validate_mechanics.lua`
   - `regression_tests.lua`

Each script includes:
- Usage documentation
- Configuration options
- Example test scenarios

---

## Integration with MoE System

Swarm agents can leverage the MoE (Mixture of Experts) system for specialized analysis:

```python
from agents.swarm.yaze_specialists import PerformanceProfilerAgent
from hafs.agents.moe import MoEOrchestrator

# Use MoE for enhanced analysis
moe = MoEOrchestrator()
await moe.initialize()

# Swarm agent can query MoE experts
profiler = PerformanceProfilerAgent()
profiler.moe = moe  # Inject MoE orchestrator

results = await profiler.run_task("~/Code/yaze")
# Profiler can now use ASM expert for code analysis
```

---

## Troubleshooting

### "YAZE path not found"

**Solution**: Specify correct path:
```bash
python -m agents.swarm.launcher yaze-performance \
    --yaze-path /path/to/yaze
```

### "Agent failed: ..."

**Solution**: Check logs for specific error. Common issues:
- Missing dependencies (install with `pip install -r requirements.txt`)
- API quota exceeded (switch to different provider)
- Invalid file paths (verify YAZE structure)

### Empty findings

**Solution**: Increase agent timeout or check if codebase path is correct:
```python
# Extend timeout (default is 120s per agent)
agent = PerformanceProfilerAgent()
agent.timeout = 300  # 5 minutes
```

---

## Next Steps

After swarm missions complete:

1. **Review Synthesis Reports** - Read `synthesis.md` files for actionable recommendations

2. **Apply Patches** - Implement suggested optimizations in YAZE codebase

3. **Test Lua Scripts** - Load scripts in Mesen2 and validate functionality

4. **Integrate Tools** - Build debugging tools based on architecture design

5. **Run Benchmarks** - Measure performance improvements

6. **Iterate** - Run swarms again after changes to validate improvements

---

## Examples

### Example 1: Quick YAZE Performance Analysis

```bash
# Launch performance mission
python -m agents.swarm.launcher yaze-performance

# Review results
cat ~/.context/swarms/yaze/yaze_performance_optimization/synthesis.md

# Check hot paths
jq '.findings.performance_profiler.hot_paths' \
    ~/.context/swarms/yaze/yaze_performance_optimization/report.json
```

### Example 2: Generate Mesen2 Scripts

```bash
# Launch Mesen2 integration
python -m agents.swarm.launcher mesen2-integration

# List generated scripts
ls -R ~/.context/swarms/mesen2/lua_scripts/

# Test a script
mesen2 --script ~/.context/swarms/mesen2/lua_scripts/memory_watch/watch_custom_items.lua \
    --rom ~/ROMs/alttp.sfc
```

### Example 3: Full YAZE + Mesen2 Campaign

```bash
# Launch everything
python -m agents.swarm.launcher all

# Results saved to:
# - ~/.context/swarms/yaze/* (3 missions)
# - ~/.context/swarms/mesen2/* (1 mission + scripts)
```

---

**Status**: ✓ READY
**Last Updated**: 2025-12-21
**Documentation**: See `SWARM_MISSIONS.md` for detailed mission specifications
