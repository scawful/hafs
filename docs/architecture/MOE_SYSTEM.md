# Mixture of Experts (MoE) System

**Created**: 2025-12-21
**Status**: ✓ IMPLEMENTED
**Architecture**: Multi-agent ROM hacking system with specialized experts

---

## Overview

The Mixture of Experts (MoE) system coordinates multiple specialized AI agents to handle complex ROM hacking tasks. Instead of using a single general-purpose model, tasks are routed to domain experts that excel in specific areas.

## Oracle Naming Schema (Locked)

Use dash-separated IDs so the models read like standard community names.

**Short form**:
```
oracle-<name>-<role>
```

**Verbose form**:
```
oracle-<name>-<role>-<base>-<yyyymmdd>
```

**Examples**:
```
oracle-nayru-canon
oracle-zelda-plotweaver-gemma3-12b-it-20250115
oracle-rauru-assembler-qwen3-coder-14b-20250115
```

### Base Defaults (Open, but Recommended)

Story/lore/dialogue experts:
- Primary: `gemma3-12b-it`
- Heavy: `gemma3-27b-it`

ROM tooling/ASM/debug experts:
- Primary: `qwen3-coder-14b`
- Fast: `qwen3-coder-7b`

Synthesis/verification experts:
- Primary: `magistral-24b`
- Fast: `deepseek-r1-14b`

You can swap in Llama 3.1 or Mixtral variants later without renaming the expert role.

## Oracle of Secrets MoE (Planned)

| Expert ID | Purpose | Base Default |
| --- | --- | --- |
| `oracle-nayru-canon` | Lore bible, continuity, timeline sanity | gemma3-12b-it |
| `oracle-zelda-plotweaver` | Plot arcs, reveals, pacing | gemma3-12b-it |
| `oracle-farore-pathfinder` | Quests, world layout, progression beats | gemma3-12b-it |
| `oracle-din-forge` | Combat balance, item tuning, difficulty | gemma3-12b-it |
| `oracle-saria-voice` | Dialogue, character voice, banter | gemma3-12b-it |
| `oracle-impa-archivist` | Consistency checks, citations, canon audits | gemma3-12b-it |
| `oracle-zelda-scribe` | UI copy, quest logs, item text | gemma3-12b-it |
| `oracle-midna-voice` | Alt voice for snarky or wry dialogue | gemma3-12b-it |
| `oracle-hylia-curator` | Mythos, cosmology, pantheon continuity | gemma3-12b-it |

## ROM Tooling MoE (Planned)

| Expert ID | Purpose | Base Default |
| --- | --- | --- |
| `oracle-rauru-assembler` | 65816 routines, hooks, patches | qwen3-coder-14b |
| `oracle-sheik-debugger` | Crash triage, trace reading, root cause | qwen3-coder-14b |
| `oracle-purah-profiler` | Performance, WRAM/VRAM/ROM map sanity | qwen3-coder-14b |
| `oracle-kaepora-banker` | Bank layout, freespace strategy | qwen3-coder-14b |
| `oracle-robbie-toolsmith` | Core ROM tooling workflows (build, patch, assets, pipelines) | qwen3-coder-14b |
| `oracle-yaze-expert` | YAZE-specific workflows and C++ API usage | qwen3-coder-14b |
| `oracle-koume-compressor` | Compression/decompression pipelines (LC_LZ2, HM) | qwen3-coder-14b |
| `oracle-kotake-tilesmith` | Tiles, palettes, graphics formats | qwen3-coder-14b |
| `oracle-agahnim-patcher` | Patch merges, IPS/BPS workflows | qwen3-coder-14b |
| `oracle-fi-indexer` | Symbol maps, metadata catalogs, index hygiene | qwen3-coder-14b |
| `oracle-kass-audio` | SPC/BRR samples, audio tooling | qwen3-coder-14b |
| `oracle-sheik-prover` | Regression verification, repro scripts | deepseek-r1-14b |

## MoE Core (Planned)

| Expert ID | Purpose | Base Default |
| --- | --- | --- |
| `oracle-council-synth` | Multi-expert synthesis and adjudication | magistral-24b |

## Routing Table (Draft)

Use these keywords as the first-pass classifier hints.

**Oracle of Secrets**
- lore, canon, timeline, retcon, continuity -> `oracle-nayru-canon`
- mythos, pantheon, cosmology, goddess -> `oracle-hylia-curator`
- plot, act, reveal, pacing, arc -> `oracle-zelda-plotweaver`
- quest, dungeon, progression, gating -> `oracle-farore-pathfinder`
- balance, damage, economy, tuning -> `oracle-din-forge`
- dialogue, voice, character, banter -> `oracle-saria-voice`
- copy, ui text, menu, quest log, item description -> `oracle-zelda-scribe`
- snark, sarcastic, wry, sassy -> `oracle-midna-voice`
- verify, cite, consistency, source -> `oracle-impa-archivist`

**ROM tooling**
- asm, routine, hook, patch -> `oracle-rauru-assembler`
- crash, trace, bug, fix -> `oracle-sheik-debugger`
- perf, optimize, vram, wram, rom map -> `oracle-purah-profiler`
- bank, org, freespace -> `oracle-kaepora-banker`
- build, pipeline, tooling, assets, conversion -> `oracle-robbie-toolsmith`
- yaze, editor, map editor, ui, palette -> `oracle-yaze-expert`
- compression, decompress, lz, packer -> `oracle-koume-compressor`
- tile, tileset, palette, gfx, sprite -> `oracle-kotake-tilesmith`
- ips, bps, xdelta, patch merge -> `oracle-agahnim-patcher`
- index, symbol, metadata, registry -> `oracle-fi-indexer`
- audio, music, spc, brr, sfx -> `oracle-kass-audio`
- regression, repro, verify fix -> `oracle-sheik-prover`

**MoE core**
- synthesize, adjudicate, resolve conflicts -> `oracle-council-synth`

Template files:
- `docs/config/routing.toml` (copy to `~/.context/models/routing.toml`)
- `docs/config/model_registry.toml` (copy to `~/.context/models/registry.toml`)

### Architecture

```
User Intent → Classifier → Expert(s) → [Synthesizer] → Result
                ↓              ↓
          Determines    Specialized
          which expert(s)  responses
```

### Components

1. **Task Classifier** - Routes tasks to appropriate expert(s)
2. **Expert Agents** - Specialized models for specific domains
   - ASM Expert (hyrule-asm-v1): 65816 assembly code
   - YAZE Expert (yaze-sage-v1): ROM editor tools and C++ API
   - Debug Expert (debug-oracle-v1): Error diagnosis and debugging
3. **Synthesizer** - Combines outputs from multiple experts
4. **Orchestrator** - Coordinates the entire workflow

---

## Quick Start

### Basic Usage

```python
from hafs.agents.moe import MoEOrchestrator

# Initialize
orchestrator = MoEOrchestrator()
await orchestrator.initialize()

# Execute a task
result = await orchestrator.execute(
    "Write a routine to add a new item to ALTTP"
)

print(result.content)
print(f"Experts used: {result.experts_used}")
```

### Multi-Expert Task

```python
# Complex task requiring multiple experts
task = """
Create a new custom item in ALTTP that:
1. Has a unique item ID
2. Uses custom graphics loaded from YAZE
3. Has an assembly routine to handle item usage
"""

result = await orchestrator.execute(task)

# Results from both ASM and YAZE experts are synthesized
print(f"Experts: {result.experts_used}")  # ['asm', 'yaze']
print(f"Synthesis used: {result.synthesis_used}")  # True
```

---

## Configuration

### Temperature Presets

Predefined temperature settings for common use cases:

```python
from hafs.agents.moe import TempPreset

TempPreset.DETERMINISTIC  # 0.1 - Very low, for exact repeatable outputs
TempPreset.LOW           # 0.3 - Precise, focused responses (code, classification)
TempPreset.MEDIUM        # 0.7 - Balanced creativity and precision
TempPreset.HIGH          # 1.0 - More creative and varied outputs
TempPreset.CREATIVE      # 1.2 - Maximum creativity (synthesis, brainstorming)
```

### Token Presets

Predefined token limits for different task types:

```python
from hafs.agents.moe import TokenPreset

TokenPreset.TINY      # 100   - Quick classifications
TokenPreset.SHORT     # 200   - Brief responses
TokenPreset.MEDIUM    # 512   - Standard responses
TokenPreset.LONG      # 1024  - Detailed explanations
TokenPreset.VERY_LONG # 2048  - Code with explanations
TokenPreset.MAXIMUM   # 4096  - Full synthesis or complex solutions
```

### Using Presets

```python
from hafs.agents.moe import TaskClassifier, TempPreset, TokenPreset

# Use presets
classifier = TaskClassifier(
    temperature=TempPreset.LOW.value,
    max_tokens=TokenPreset.SHORT.value,
)

# Or get expert-specific preset
from hafs.agents.moe import get_preset

asm_preset = get_preset("asm")
print(f"ASM: temp={asm_preset.temperature}, tokens={asm_preset.max_tokens}")
# ASM: temp=0.3, tokens=2048
```

### Custom Configuration

```python
# Full customization
from hafs.agents.moe import MoEOrchestrator, TaskClassifier, Synthesizer

classifier = TaskClassifier(
    max_tokens=150,
    temperature=0.2,  # Very deterministic
)

synthesizer = Synthesizer(
    max_tokens=3000,
    temperature=0.9,  # More creative synthesis
)

orchestrator = MoEOrchestrator()
orchestrator.classifier = classifier
orchestrator.synthesizer = synthesizer

await orchestrator.initialize()
```

---

## Expert Details

### ASM Expert

**Name**: `hyrule-asm-v1`
**Specialization**: 65816 assembly for ALTTP ROM hacking
**Default Config**:
- Temperature: 0.3 (LOW - precise code)
- Max tokens: 2048 (VERY_LONG)

**Capabilities**:
- Generate assembly routines
- Optimize existing code
- Explain assembly patterns
- Memory map navigation
- Bank allocation strategies

**Example**:
```python
from hafs.agents.moe.experts import AsmExpert

asm_expert = AsmExpert()
await asm_expert.initialize()

# Optimize a routine
optimized = await asm_expert.optimize_routine(
    routine_code=my_asm_code,
    optimization_goal="speed"
)
```

### YAZE Expert

**Name**: `yaze-sage-v1`
**Specialization**: YAZE ROM editor tools and C++ API
**Default Config**:
- Temperature: 0.6 (MEDIUM)
- Max tokens: 2048 (VERY_LONG)

**Capabilities**:
- YAZE tool usage
- ROM file manipulation
- Graphics format conversion
- Compression/decompression
- Map and dungeon editing

**Example**:
```python
from hafs.agents.moe.experts import YazeExpert

yaze_expert = YazeExpert()
await yaze_expert.initialize()

# Generate YAZE API calls
code = await yaze_expert.generate_tool_calls(
    "Replace Link sprite graphics with custom tiles"
)
```

### Debug Expert

**Name**: `debug-oracle-v1`
**Specialization**: Error diagnostics and debugging
**Default Config**:
- Temperature: 0.4 (LOW - focused debugging)
- Max tokens: 2048 (VERY_LONG)

**Capabilities**:
- Error message interpretation
- Stack trace analysis
- Debugging strategies
- Root cause analysis
- Fix recommendations

**Example**:
```python
from hafs.agents.moe.experts import DebugExpert

debug_expert = DebugExpert()
await debug_expert.initialize()

# Diagnose an error
diagnosis = await debug_expert.diagnose_error(
    error_description="ROM crashes when picking up custom item",
    error_logs=crash_log
)
```

---

## Advanced Usage

### Force Specific Experts

```python
# Force use of specific experts (bypass classification)
result = await orchestrator.execute(
    "Explain sprite graphics handling",
    force_experts=["asm", "yaze"]  # Use both
)
```

### Get Routing Explanation

```python
# See which experts would be used (without executing)
explanation = await orchestrator.explain_routing(
    "Add a new dungeon room with custom tiles"
)
print(explanation)
```

### List Available Experts

```python
# Get all experts and their specializations
experts = await orchestrator.list_experts()
for name, specialization in experts.items():
    print(f"{name}: {specialization}")

# Get detailed info for specific expert
info = await orchestrator.get_expert_info("asm")
print(info)
```

### Custom Expert Configuration

```python
from pathlib import Path
from hafs.agents.moe.experts import AsmExpert

# Custom ASM expert with specific LoRA adapter
custom_asm = AsmExpert(
    model_name="my-custom-asm-model",
    lora_adapter_path=Path("~/.context/models/custom/adapters"),
)

# Replace default expert
orchestrator.experts["asm"] = custom_asm
await custom_asm.initialize()
```

---

## Integration with Fine-Tuned Models

Once the models are trained on medical-mechanica, they'll automatically be loaded:

```python
# After training hyrule-asm-v1 and yaze-sage-v1:

orchestrator = MoEOrchestrator()
await orchestrator.initialize()

# Experts will load their LoRA adapters from:
# - ASM: ~/.context/models/alttp_asm_agent/lora_adapters
# - YAZE: ~/.context/models/yaze_tool_agent/lora_adapters
# - Debug: ~/.context/models/debug_agent/lora_adapters (if trained)

# If adapters aren't found, experts fall back to base models with
# specialized system prompts (still effective!)
```

---

## Task Classification

### Keyword-Based (Fast Path)

The classifier first tries keyword matching for quick decisions:

```python
# ASM keywords: asm, assembly, routine, bank, memory, etc.
# YAZE keywords: yaze, rom, graphics, sprite, tile, etc.
# Debug keywords: error, bug, crash, fix, debug, etc.

# High confidence (>0.8) keyword matches skip LLM classification
```

### LLM-Based (Nuanced)

For ambiguous tasks, uses LLM for classification:

```python
classifier = TaskClassifier()
classification = await classifier.classify(
    "Create a sprite that uses optimized code"
)

# Result:
# experts: ['asm', 'yaze']
# confidences: [0.85, 0.75]
# is_multi_expert: True
```

---

## Synthesis Process

When multiple experts are used, their outputs are synthesized:

```python
# Multiple expert outputs → Synthesizer → Unified solution

synthesizer = Synthesizer()
result = await synthesizer.synthesize(
    user_intent="Add custom item with graphics",
    expert_responses=[asm_response, yaze_response]
)

# Synthesizer:
# 1. Integrates insights from all experts
# 2. Creates step-by-step approach
# 3. Resolves conflicts
# 4. Provides complete, actionable answer
```

---

## Testing

Run comprehensive test suite:

```bash
PYTHONPATH=src .venv/bin/python -m hafs.agents.moe.test_moe
```

**Tests included**:
1. Task classifier (keyword + LLM)
2. Single expert execution
3. Multi-expert execution with synthesis
4. Debug expert
5. Expert routing explanation
6. Forced expert selection
7. Configurable parameters
8. Expert information queries

---

## File Structure

```
src/hafs/agents/moe/
├── __init__.py              # Package exports
├── config.py                # Temperature/token presets
├── classifier.py            # Task classification
├── expert.py                # Base expert class
├── experts/
│   ├── __init__.py
│   ├── asm_expert.py       # ASM specialist
│   ├── yaze_expert.py      # YAZE specialist
│   └── debug_expert.py     # Debug specialist
├── synthesizer.py          # Multi-expert synthesis
├── orchestrator.py         # Main coordinator
└── test_moe.py            # Test suite
```

---

## Performance

### Single Expert

```
User Intent → Classifier → Expert → Result
              (~200ms)     (~1-3s)
Total: ~1.5-3.5 seconds
```

### Multi-Expert (Parallel)

```
User Intent → Classifier → [ASM Expert  → Result
              (~200ms)      YAZE Expert]
                            (~1-3s each, parallel)
                                ↓
                           Synthesizer
                            (~2-4s)
Total: ~3.5-7.5 seconds
```

### Optimization

- Experts execute in parallel (asyncio.gather)
- Keyword classification shortcuts LLM when confident
- Results are cached per session
- LoRA adapters stay loaded between requests

---

## Example Use Cases

### 1. Simple ASM Task (Single Expert)

```python
result = await orchestrator.execute(
    "Write a routine that checks if Link has the Master Sword"
)
# Uses: ASM Expert only
# Time: ~2 seconds
```

### 2. YAZE Graphics Task (Single Expert)

```python
result = await orchestrator.execute(
    "Load custom sprite graphics at offset 0x80000 using YAZE"
)
# Uses: YAZE Expert only
# Time: ~2 seconds
```

### 3. Complex Multi-Domain Task (Multi-Expert)

```python
result = await orchestrator.execute(
    """
    Create a new boss enemy that:
    - Uses custom sprite graphics (YAZE)
    - Has an AI routine in assembly (ASM)
    - Loads at dungeon entrance
    """
)
# Uses: ASM + YAZE Experts → Synthesizer
# Time: ~5-6 seconds
# Output: Integrated solution with both code and graphics instructions
```

### 4. Debugging Task (Single Expert)

```python
result = await orchestrator.execute(
    """
    My ROM hack crashes when entering room $45 in dungeon 3.
    The screen goes black and freezes. I recently modified
    the tile data for this room.
    """
)
# Uses: Debug Expert only
# Time: ~2-3 seconds
# Output: Diagnostic steps and likely causes
```

---

## Future Enhancements

### Planned Features

1. **Expert Memory** - Remember context across sessions
2. **Confidence Tuning** - Auto-adjust expert thresholds based on performance
3. **Expert Voting** - Multiple experts vote on conflicting solutions
4. **Specialized Sub-Experts** - E.g., Graphics Expert, Audio Expert
5. **Performance Profiling** - Track which experts perform best on which tasks

### Potential Experts to Add

- **Oracle Expert** - Oracle of Secrets ROM hack specialist
- **Gigaleak Expert** - Nintendo source code analysis
- **Audio Expert** - SPC700 sound programming
- **Graphics Expert** - Advanced tile/sprite manipulation
- **Randomizer Expert** - Randomizer logic and generation

---

## Troubleshooting

### Expert Not Loading

```
Error: No LoRA adapter found at path
```

**Solution**: Experts fall back to base models automatically. Train models or specify custom path:
```python
expert = AsmExpert(lora_adapter_path=Path("/custom/path"))
```

### Classification Issues

```
Wrong expert selected for task
```

**Solution 1**: Force specific experts:
```python
result = await orchestrator.execute(task, force_experts=["asm"])
```

**Solution 2**: Add keywords to task description:
```python
task = "Using YAZE tool, create sprite graphics..."  # "YAZE" triggers YAZE expert
```

### Synthesis Quality

```
Multi-expert synthesis seems incoherent
```

**Solution**: Adjust synthesizer temperature:
```python
synthesizer = Synthesizer(
    max_tokens=4096,
    temperature=0.5  # Lower for more focused synthesis
)
```

---

## Summary

**Status**: ✓ FULLY IMPLEMENTED

The MoE system provides:
- ✓ 3 specialized expert agents (ASM, YAZE, Debug)
- ✓ Intelligent task classification (keyword + LLM)
- ✓ Multi-expert synthesis for complex tasks
- ✓ Configurable parameters (temperature, tokens)
- ✓ Temperature/token presets for common scenarios
- ✓ Parallel expert execution
- ✓ Comprehensive test suite
- ✓ Ready for fine-tuned model integration

**Next Steps**:
1. Complete pilot generation (145/190, 76%)
2. Train models on medical-mechanica
3. Test MoE with trained adapters
4. Benchmark performance vs single-model baseline

---

**Created**: 2025-12-21
**Last Updated**: 2025-12-21
**Version**: 1.0.0
