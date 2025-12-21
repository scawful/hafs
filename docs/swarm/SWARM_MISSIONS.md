# Swarm Missions: YAZE & Mesen2 Improvement

**Created**: 2025-12-21
**Target Systems**: YAZE ROM Editor, Mesen2 Emulator
**Objective**: Improve emulator performance, fix audio/input issues, enhance debugging

---

## Overview

Deploy agent swarms to analyze, debug, and improve the YAZE and Mesen2 codebases. Swarms consist of specialized agents working in parallel to achieve specific goals.

### Target Issues

**YAZE (ROM Editor)**:
- Emulator performance optimization
- Audio system bugs
- Edge detection input lag
- Testing infrastructure
- Integration with Mesen2 for debugging

**Mesen2 (Emulator)**:
- Lua script system for debugging
- YAZE integration hooks
- Performance profiling
- Potential fork with custom features

---

## Swarm Architecture

```
Swarm Coordinator
    ↓
┌────────────────┬─────────────────┬──────────────────┬─────────────────┐
│ Performance    │ Audio System    │ Input Detection  │ Integration     │
│ Profiler Agent │ Debug Agent     │ Fix Agent        │ Test Agent      │
└────────────────┴─────────────────┴──────────────────┴─────────────────┘
    ↓                  ↓                   ↓                   ↓
  Reports          Bug Fixes           Patches           Test Scripts
         ↓                  ↓                   ↓                   ↓
                  Swarm Results Synthesizer
                             ↓
                  Actionable Implementation Plan
```

---

## Mission 1: YAZE Performance Optimization

### Objectives
1. Profile emulation bottlenecks
2. Identify slow rendering paths
3. Optimize CPU/PPU emulation loops
4. Reduce input lag
5. Memory optimization

### Swarm Configuration

```toml
[swarm.yaze_performance]
name = "YAZE Performance Optimization Swarm"
target_codebase = "~/Code/yaze"
agents = [
    "performance_profiler",
    "cpu_optimizer",
    "ppu_optimizer",
    "memory_profiler",
]
parallel = true
duration = "2 hours"
output_dir = "~/.context/swarms/yaze_performance"
```

### Agent Assignments

**Performance Profiler Agent**:
- Scan C++ source for hot paths
- Identify function call overhead
- Find unnecessary allocations
- Generate profiling report

**CPU Optimizer Agent**:
- Analyze 65816 CPU emulation loop
- Suggest SIMD optimizations
- Identify branch mispredictions
- Recommend caching strategies

**PPU Optimizer Agent**:
- Analyze PPU rendering pipeline
- Find redundant draws
- Optimize tile/sprite handling
- Suggest frame skipping strategies

**Memory Profiler Agent**:
- Track memory allocation patterns
- Find memory leaks
- Suggest pooling opportunities
- Optimize data structures

### Expected Output

```
SWARM REPORT: YAZE Performance Optimization
Generated: 2025-12-21

=== Findings ===
1. CPU emulation loop: 45% of runtime (cpu.cpp:234-456)
   - Recommendation: Inline common instructions
   - Estimated speedup: 15-20%

2. PPU tile rendering: 30% of runtime (ppu.cpp:567-789)
   - Recommendation: Batch tile draws, use texture atlas
   - Estimated speedup: 25-30%

3. Memory allocations: 3.2M allocs/frame
   - Recommendation: Object pooling for sprites
   - Estimated speedup: 10-15%

=== Patches ===
- cpu_inline_opts.patch
- ppu_batching.patch
- memory_pooling.patch

=== Implementation Priority ===
1. PPU batching (highest impact)
2. CPU inlining (medium impact, low risk)
3. Memory pooling (medium impact, higher risk)
```

---

## Mission 2: Audio System Debugging

### Objectives
1. Identify audio glitches and crackling
2. Fix SPC700 emulation bugs
3. Improve audio sync with video
4. Reduce audio latency
5. Add debugging tools

### Swarm Configuration

```toml
[swarm.yaze_audio]
name = "YAZE Audio System Debug Swarm"
target_codebase = "~/Code/yaze"
focus_paths = [
    "src/app/emu/audio/",
    "src/app/emu/spc700/",
]
agents = [
    "audio_debugger",
    "spc700_validator",
    "sync_analyzer",
    "latency_profiler",
]
parallel = true
duration = "90 minutes"
```

### Agent Assignments

**Audio Debugger Agent**:
- Trace audio buffer underruns
- Find sample rate issues
- Identify crackling sources
- Generate audio test cases

**SPC700 Validator Agent**:
- Compare against known-good SPC700 implementations
- Find instruction emulation bugs
- Validate DSP operations
- Create regression tests

**Sync Analyzer Agent**:
- Analyze audio/video timing
- Find desync patterns
- Recommend sync strategies
- Test frame pacing

**Latency Profiler Agent**:
- Measure end-to-end latency
- Find buffering delays
- Optimize audio pipeline
- Suggest low-latency mode

---

## Mission 3: Input Edge Detection Fix

### Objectives
1. Fix input lag issues
2. Improve edge detection accuracy
3. Add input display overlay
4. Create input playback system
5. Reduce frame-to-input latency

### Swarm Configuration

```toml
[swarm.yaze_input]
name = "YAZE Input System Fix Swarm"
target_codebase = "~/Code/yaze"
focus_paths = [
    "src/app/emu/cpu.cc",
    "src/app/emu/controller.cc",
]
agents = [
    "input_lag_analyzer",
    "edge_detector",
    "input_overlay_generator",
    "playback_system_builder",
]
parallel = true
duration = "60 minutes"
```

### Expected Output

Patches for:
- Reduced input polling latency
- Improved edge detection algorithm
- Input display overlay (TAS-style)
- Input recording/playback for testing

---

## Mission 4: Mesen2 Integration & Lua Scripting

### Objectives
1. Create Lua debugging scripts for Mesen2
2. Design YAZE ↔ Mesen2 integration
3. Add memory watch/breakpoint helpers
4. Create ROM testing automation
5. Evaluate fork vs plugin architecture

### Swarm Configuration

```toml
[swarm.mesen2_integration]
name = "Mesen2 Integration Swarm"
target_codebases = [
    "~/Code/yaze",
    "~/Code/mesen2",  # If cloned
]
agents = [
    "lua_script_generator",
    "integration_architect",
    "debugging_tools_builder",
    "test_automation_agent",
]
parallel = true
duration = "2 hours"
```

### Agent Assignments

**Lua Script Generator**:
```lua
-- Generate Mesen2 debugging scripts

-- Example: Memory watch for custom items
function onFrame()
    local customItemSlot = emu.read(0x7EF340, emu.memType.snesMemory)
    if customItemSlot ~= 0 then
        emu.log("Custom item detected: " .. string.format("%02X", customItemSlot))

        -- Breakpoint on item pickup
        emu.addBreakpoint(0x0E8234, emu.breakType.exec,
            "Item pickup routine triggered")
    end
end

emu.addEventCallback(onFrame, emu.eventType.endFrame)
```

**Integration Architect**:
- Design YAZE → Mesen2 launch workflow
- Create shared memory interface
- Plan ROM → Emulator pipeline
- Evaluate fork vs plugin

**Debugging Tools Builder**:
- Memory inspector integration
- Disassembly viewer sync
- Breakpoint manager
- Watch expression system

**Test Automation Agent**:
- ROM testing framework
- Automated regression tests
- Performance benchmarks
- Input playback tests

---

## Mesen2 Fork Strategy

### Option 1: Plugin Architecture (Recommended)

**Advantages**:
- No forking needed
- Easier to maintain
- Can distribute separately
- Upstream changes flow through

**Implementation**:
```cpp
// YAZE plugin for Mesen2
class YazeIntegrationPlugin : public IMesenPlugin {
public:
    void OnRomLoaded(RomInfo& info) override {
        // Sync YAZE project with ROM
        yazeInterface.LoadProject(info.path);
    }

    void OnMemoryWrite(uint32_t addr, uint8_t value) override {
        // Track writes for debugging
        if (watchedAddresses.contains(addr)) {
            yazeInterface.NotifyMemoryWrite(addr, value);
        }
    }

    void OnBreakpoint(uint32_t addr) override {
        // Show YAZE disassembly at breakpoint
        yazeInterface.ShowDisassembly(addr);
    }
};
```

### Option 2: Fork with Custom Features

**Advantages**:
- Full control
- Can add YAZE-specific features
- Tight integration

**Disadvantages**:
- Maintenance burden
- Must sync upstream changes
- Distribution complexity

**If forking, focus on**:
1. YAZE project file integration
2. Synchronized disassembly view
3. Graphics/tile viewer integration
4. Shared breakpoint system

---

## Swarm Execution Plan

### Phase 1: Analysis (Week 1)

```bash
# Launch performance swarm
hafs swarm launch yaze_performance \
    --codebase ~/Code/yaze \
    --duration 2h \
    --output ~/.context/swarms/yaze_performance

# Launch audio swarm (parallel)
hafs swarm launch yaze_audio \
    --codebase ~/Code/yaze \
    --focus src/app/emu/audio \
    --duration 90m

# Launch input swarm (parallel)
hafs swarm launch yaze_input \
    --codebase ~/Code/yaze \
    --focus src/app/emu/controller.cc \
    --duration 60m
```

### Phase 2: Implementation (Week 2)

```bash
# Apply patches from swarms
hafs swarm apply yaze_performance \
    --patches all \
    --test

# Validate fixes
hafs swarm test yaze_performance \
    --benchmark
```

### Phase 3: Integration (Week 3)

```bash
# Clone Mesen2
git clone https://github.com/SourMesen/Mesen2 ~/Code/mesen2

# Launch integration swarm
hafs swarm launch mesen2_integration \
    --codebases ~/Code/yaze ~/Code/mesen2 \
    --duration 2h
```

### Phase 4: Testing (Week 4)

```bash
# Run automated tests
hafs swarm test integration \
    --rom alttp.sfc \
    --test-suite full

# Generate test report
hafs swarm report integration \
    --format markdown \
    --output docs/reports/INTEGRATION_REPORT.md
```

---

## Lua Debugging Scripts

### Memory Watch Scripts

```lua
-- watch_custom_items.lua
-- Monitors custom item slots and triggers breakpoints

local CUSTOM_ITEM_START = 0x7EF340
local CUSTOM_ITEM_COUNT = 16

function logItemChange(slot, oldValue, newValue)
    emu.log(string.format(
        "Item slot $%02X changed: $%02X → $%02X",
        slot, oldValue, newValue
    ))
end

local itemCache = {}

function checkItems()
    for i = 0, CUSTOM_ITEM_COUNT - 1 do
        local addr = CUSTOM_ITEM_START + i
        local value = emu.read(addr, emu.memType.snesMemory)

        if itemCache[i] ~= value then
            logItemChange(i, itemCache[i] or 0, value)
            itemCache[i] = value

            -- Breakpoint on first change
            if not itemCache[i.."_bp"] then
                emu.addBreakpoint(addr, emu.breakType.write)
                itemCache[i.."_bp"] = true
            end
        end
    end
end

emu.addEventCallback(checkItems, emu.eventType.endFrame)
```

### Performance Profiling

```lua
-- profile_performance.lua
-- Profiles CPU usage by routine

local profiles = {}

function profileStart(name, address)
    profiles[name] = {
        address = address,
        callCount = 0,
        totalCycles = 0,
        startCycle = 0,
    }

    -- Breakpoint at routine entry
    emu.addBreakpoint(address, emu.breakType.exec, function()
        profiles[name].startCycle = emu.getCycle()
        profiles[name].callCount = profiles[name].callCount + 1
    end)

    -- Breakpoint at routine exit (RTL)
    emu.addBreakpoint(address + 0x100, emu.breakType.exec, function()
        local cycles = emu.getCycle() - profiles[name].startCycle
        profiles[name].totalCycles = profiles[name].totalCycles + cycles
    end)
end

-- Profile key routines
profileStart("NPC_Handler", 0x0E8234)
profileStart("Item_Pickup", 0x0E9456)
profileStart("Player_Update", 0x0EA123)

function printProfiles()
    emu.log("=== Performance Profile ===")
    for name, data in pairs(profiles) do
        local avgCycles = data.totalCycles / math.max(data.callCount, 1)
        emu.log(string.format(
            "%s: %d calls, avg %d cycles",
            name, data.callCount, avgCycles
        ))
    end
end

emu.addEventCallback(printProfiles, emu.eventType.endFrame)
```

---

## Expected Deliverables

### YAZE Improvements
- [ ] 30-50% emulation performance boost
- [ ] Fixed audio crackling
- [ ] <5ms input latency
- [ ] Input display overlay
- [ ] Improved testing infrastructure

### Mesen2 Integration
- [ ] 10+ Lua debugging scripts
- [ ] YAZE plugin or fork decision
- [ ] Synchronized memory watch
- [ ] Automated testing framework
- [ ] Performance benchmarks

### Documentation
- [ ] Swarm reports for each mission
- [ ] Implementation guides
- [ ] Testing procedures
- [ ] Integration architecture docs

---

## Resource Requirements

**Hardware**: Mac (development) + medical-mechanica (intensive analysis)
**Time**: 4 weeks (1 week per phase)
**API Costs**: ~$50-100 for swarm analysis (Gemini 3 Flash for bulk work)
**Storage**: ~5GB for swarm outputs, patches, test results

---

## Next Steps

1. **Clone Mesen2**: `git clone https://github.com/SourMesen/Mesen2 ~/Code/mesen2`
2. **Implement Swarm System**: Create swarm coordinator and agent templates
3. **Launch Performance Swarm**: Start with YAZE performance as pilot
4. **Develop Lua Scripts**: Generate initial debugging script library
5. **Evaluate Integration**: Test plugin vs fork approach

---

**Status**: PLANNED
**Priority**: HIGH (after pilot generation completes)
**Dependencies**: Pilot generation, MoE system (for swarm agents)
**Timeline**: 4 weeks for full implementation
