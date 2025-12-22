# Cognitive Protocol Architecture

**Version**: 1.0
**Schema Version**: 0.3
**Last Updated**: 2025-12-22

---

## Table of Contents

1. [Overview](#overview)
2. [File System Structure](#file-system-structure)
3. [Data Flow](#data-flow)
4. [Configuration System](#configuration-system)
5. [I/O Performance](#io-performance)
6. [Schema Validation](#schema-validation)
7. [Integration Patterns](#integration-patterns)
8. [Performance Benchmarks](#performance-benchmarks)

---

## Overview

The Cognitive Protocol is a multi-layer system for tracking agent cognitive state, implementing metacognitive monitoring, and enabling autonomous adaptation. It provides:

- **Metacognition**: Self-awareness, spin detection, flow state tracking
- **Goal Management**: Hierarchical goal decomposition and conflict detection
- **Epistemic Tracking**: Knowledge confidence with temporal decay
- **Emotional State**: Multi-dimensional emotional tracking
- **Analysis Triggers**: Automatic mode switching based on thresholds
- **Configuration**: 60+ tunable parameters via TOML files
- **Performance**: Batched I/O with caching (90-95% latency reduction)
- **Validation**: Runtime schema validation with auto-fix

---

## File System Structure

### Global Context (`~/.context/`)

**Purpose**: Cross-project shared state and background services

```
~/.context/
├── autonomy_daemon/          # Autonomous training orchestration
│   ├── daemon_status.json    # Service health and metrics
│   └── schedules/            # Training schedules
├── background_agent/         # Background task state
├── context_agent_daemon/     # Context management service
├── embedding_service/        # Vector embeddings
│   └── index/               # FAISS indexes
├── monitoring/              # System monitoring
├── swarms/                  # Multi-agent coordination
│   ├── councils/            # Council decisions
│   └── missions/            # Active mission state
└── models/                  # Trained model registry
    ├── oracle-rauru-assembler/
    └── oracle-yaze-expert/
```

### Per-Project Context (`./.context/`)

**Purpose**: Project-specific working memory and long-term learning

```
./.context/
├── metadata.json            # Project metadata
├── scratchpad/             # Working memory (ephemeral, reset per session)
│   ├── state.md            # Human-readable state (rendered view)
│   ├── state.json          # Canonical state (schema v0.3)
│   ├── metacognition.json  # Metacognitive monitoring
│   ├── epistemic.json      # Knowledge confidence
│   ├── emotions.json       # Emotional state
│   ├── grounding.json      # Crisis detection triggers
│   ├── analysis-triggers.json  # Pending analysis requests
│   ├── goals.json          # Goal hierarchy
│   └── deferred.md         # Deferred tasks
├── memory/                 # Long-term learned patterns
│   ├── fears.json          # Learned error patterns
│   └── patterns/           # Recurring patterns
├── knowledge/              # Domain knowledge
│   ├── facts/              # Verified facts
│   └── embeddings/         # Vector index
├── hivemind/              # Cross-agent shared state
│   ├── manifest.json      # Hivemind configuration
│   ├── decisions.json     # Council decisions (golden facts)
│   └── preferences.json   # Shared preferences
└── history/               # State snapshots
    ├── agents/            # Agent execution history
    │   ├── explore_<id>/
    │   ├── critic_<id>/
    │   └── general_<id>/
    └── snapshots/         # Periodic state backups
```

### Key Files

#### `scratchpad/metacognition.json`
**Schema**: v0.3
**Purpose**: Real-time metacognitive monitoring
**Updates**: Every action

```json
{
  "schema_version": "0.3",
  "producer": {"name": "hafs", "version": "0.5.0"},
  "last_updated": "2025-12-22T10:30:00Z",

  "current_strategy": "incremental",
  "strategy_effectiveness": 0.75,
  "progress_status": "making_progress",

  "spin_detection": {
    "recent_actions": ["edit_file", "run_test", "edit_file"],
    "similar_action_count": 2,
    "spinning_threshold": 4,
    "last_distinct_action_time": "2025-12-22T10:28:00Z"
  },

  "cognitive_load": {
    "current": 0.65,
    "items_in_focus": 5,
    "max_recommended_items": 7
  },

  "help_seeking": {
    "current_uncertainty": 0.25,
    "consecutive_failures": 0,
    "should_ask_user": false
  },

  "flow_state": true,
  "self_corrections": []
}
```

#### `scratchpad/goals.json`
**Schema**: v0.3
**Purpose**: Goal hierarchy and conflict tracking
**Updates**: When goals change

```json
{
  "schema_version": "0.3",
  "primary_goal": {
    "id": "g_abc123",
    "description": "Implement user authentication",
    "user_stated": "Add login system",
    "status": "in_progress",
    "progress": 0.60
  },

  "subgoals": [
    {
      "id": "g_sub_001",
      "description": "Design database schema",
      "parent_id": "g_abc123",
      "status": "completed",
      "progress": 1.0
    }
  ],

  "conflicts": [
    {
      "id": "conflict_001",
      "goal_a_id": "g_sub_002",
      "goal_b_id": "g_sub_003",
      "conflict_type": "speed_vs_quality",
      "resolved": false
    }
  ]
}
```

#### `scratchpad/epistemic.json`
**Schema**: v0.3
**Purpose**: Knowledge confidence tracking with temporal decay
**Updates**: When facts are learned/verified

```json
{
  "schema_version": "0.3",
  "golden_facts": {
    "fact_001": {
      "content": "WRAM $7E0000-$7E1FFF is working RAM",
      "confidence": 1.0,
      "source": "verified_from_disassembly",
      "learned_at": "2025-12-22T10:00:00Z"
    }
  },

  "working_facts": {
    "fact_002": {
      "content": "Bank $09 has 8KB freespace at $098000",
      "confidence": 0.75,
      "source": "user_stated",
      "learned_at": "2025-12-22T10:15:00Z"
    }
  },

  "settings": {
    "decay_rate_per_hour": 0.05,
    "prune_threshold": 0.1,
    "max_golden_facts": 10,
    "max_working_facts": 100
  }
}
```

---

## Data Flow

### Read-Process-Update-Reflect Loop

```
┌─────────────────────────────────────────────────────────┐
│                    COGNITIVE LOOP                        │
└─────────────────────────────────────────────────────────┘

1. READ
   ├─ Load metacognition.json (cached)
   ├─ Load goals.json (cached)
   ├─ Load epistemic.json (cached)
   └─ Load emotions.json (cached)

2. PROCESS
   ├─ Record action → update spin detection
   ├─ Update cognitive load
   ├─ Check triggers
   │  ├─ Spinning? → suggest critic mode
   │  ├─ High anxiety? → suggest emotional analysis
   │  └─ Failures? → suggest metrics review
   └─ Update flow state

3. UPDATE (Batched)
   ├─ Queue metacognition.json write
   ├─ Queue goals.json write
   ├─ Queue epistemic.json write
   └─ Flush every 5 seconds (configurable)

4. REFLECT
   ├─ Evaluate strategy effectiveness
   ├─ Check for conflicts
   ├─ Suggest strategy changes
   └─ Recommend next actions
```

### Performance Characteristics

**Before Optimization** (v0.4):
- Cold load: ~100ms (4 files × 25ms each)
- Save: ~80ms (4 files × 20ms each)
- Per-turn overhead: ~180ms

**After Optimization** (v0.5 with IOManager):
- Cold load: ~10ms (4 files × 2.5ms cached)
- Save (batched): ~1ms (queued)
- Flush: ~20ms (4 files batched)
- Per-turn overhead: ~11ms (94% reduction)

---

## Configuration System

### Architecture

```
config/
├── cognitive_protocol.toml        # Base configuration (60+ parameters)
└── agent_personalities.toml       # Personality overrides

Personalities:
- cautious      (conservative, asks for help early)
- aggressive    (risk-tolerant, rarely asks for help)
- researcher    (exploration-focused, tracks more items)
- builder       (implementation-focused, tolerates prototyping)
- critic        (quality-focused, strict about testing)
```

### Configuration Hierarchy

```
1. Base Defaults (hardcoded in Pydantic models)
        ↓
2. cognitive_protocol.toml (overrides defaults)
        ↓
3. Personality Profile (overrides base config)
        ↓
4. Runtime Parameters (optional, overrides everything)
```

### Example: Loading Configuration

```python
from core.config.loader import get_config

# Load default configuration
config = get_config()

# Load with personality
config = get_config(personality="cautious")

# Access values
threshold = config.metacognition.spinning_threshold  # 3 for cautious, 4 default
uncertainty = config.metacognition.help_seeking.uncertainty_threshold  # 0.2 for cautious

# Use in managers
from core.metacognition.monitor import MetacognitionMonitor
monitor = MetacognitionMonitor(config=config)
```

### Key Configuration Sections

#### Metacognition
```toml
[metacognition]
spinning_threshold = 4        # Detect spinning after 4 repetitions
max_action_history = 10       # Track last 10 actions
cognitive_load_warning = 0.8  # Warn at 80% capacity
max_items_in_focus = 7        # Miller's Law

[metacognition.help_seeking]
uncertainty_threshold = 0.3   # Ask for help at 30% uncertainty
failure_threshold = 2         # After 2 consecutive failures

[metacognition.flow_state]
max_cognitive_load = 0.7      # Flow requires <70% load
min_strategy_effectiveness = 0.6  # Flow requires >60% effectiveness
max_frustration = 0.3         # Flow requires <30% frustration
```

#### Analysis Triggers
```toml
[analysis_triggers]
edits_without_tests = 3              # Trigger eval mode after 3 edits
high_anxiety_threshold = 0.7         # Emotional analysis at 70% anxiety
consecutive_failures_count = 3       # Metrics review after 3 failures
tool_repetition_count = 5            # Critic after 5 same-tool calls

[analysis_triggers.cooldowns]
spinning_critic = 600000             # 10 minutes (ms)
edits_without_tests = 900000         # 15 minutes
high_anxiety_caution = 1200000       # 20 minutes
```

---

## I/O Performance

### IOManager Architecture

```python
from core.protocol.io_manager import get_io_manager

# Singleton instance
io_manager = get_io_manager()

# Read (with caching)
data = io_manager.read_json(Path("state.json"))
# First call: 25ms (disk)
# Subsequent calls: 0.1ms (cache hit)

# Write (with batching)
io_manager.write_json(Path("state.json"), data)
# Queued: 0.5ms
# Flushed every 5 seconds: 20ms (batched)

# Force immediate write
io_manager.write_json(Path("state.json"), data, immediate=True)

# Manual flush
io_manager.flush()

# Stats
stats = io_manager.get_stats()
# {
#   "cache_hits": 150,
#   "cache_misses": 10,
#   "cache_hit_rate": 0.938,
#   "writes_batched": 45,
#   "writes_immediate": 5
# }
```

### Batching Strategy

**Configuration**:
```toml
[performance]
enable_batching = true
batch_flush_interval_ms = 5000  # Flush every 5 seconds
enable_caching = true
cache_ttl_seconds = 60          # Cache for 1 minute
lazy_load = true
```

**Behavior**:
1. Writes are queued in memory
2. Every 5 seconds (configurable), queue is flushed
3. Critical operations (init, shutdown) use `immediate=True`
4. Reduced disk I/O: 4-6 writes/turn → 1 batched write

### Caching Strategy

**Features**:
- In-memory cache with TTL (default: 60 seconds)
- Thread-safe with locks
- Automatic expiration
- Manual invalidation support

**Cache Hit Scenarios**:
- Repeated reads within TTL: 100x faster
- Unchanged state files: No disk I/O
- High-frequency polling: Cached responses

---

## Schema Validation

### SchemaValidator

```python
from core.protocol.validation import SchemaValidator

# Validate a single file
from models.metacognition import MetacognitiveState
state = SchemaValidator.validate_file(
    Path(".context/scratchpad/metacognition.json"),
    MetacognitiveState
)

# Validate with auto-fix
state, was_fixed = SchemaValidator.validate_file_with_auto_fix(
    Path(".context/scratchpad/metacognition.json"),
    MetacognitiveState
)
if was_fixed:
    print("File was corrupted, used defaults")

# Validate entire directory
results = SchemaValidator.validate_directory(
    Path(".context"),
    auto_fix=True
)
# {
#   ".context/scratchpad/metacognition.json": True,
#   ".context/scratchpad/goals.json": True,
#   ".context/scratchpad/emotions.json": False  # Validation failed
# }

# Get detailed errors
errors = SchemaValidator.get_validation_errors(
    Path("metacognition.json"),
    MetacognitiveState
)
# [
#   {"field": "cognitive_load.current", "message": "value must be <= 1.0"},
#   {"field": "strategy_effectiveness", "message": "field required"}
# ]
```

### Integration with Managers

```python
from core.metacognition.monitor import MetacognitionMonitor

monitor = MetacognitionMonitor()

# Load with validation (optional, opt-in)
monitor.load_state(validate=True, auto_fix=True)

# Load without validation (default, backward compatible)
monitor.load_state()
```

---

## Integration Patterns

### Oracle-Code Integration

```python
# Oracle-code bridge for external integration
from core.protocol.manager import CognitiveProtocolManager

class OracleCodeBridge:
    def __init__(self):
        self.protocol = CognitiveProtocolManager()

    def before_turn(self) -> str:
        """Returns cognitive context for LLM prompt."""
        return self.protocol.get_prompt_context()

    def after_tool_use(self, tool_name: str, success: bool):
        """Record tool use."""
        self.protocol.record_action(
            action_type="tool_use",
            action_data={"tool": tool_name},
            success=success
        )

    def after_turn(self):
        """Reflect and save state."""
        self.protocol.save_state()
        suggestions = self.protocol.reflect()

        if suggestions["strategy_change"]:
            print(f"Consider switching to: {suggestions['strategy_change']}")

        if suggestions["seek_help"]:
            print("Agent should ask user for help")
```

### HAFS TUI Integration

```python
# Real-time monitoring in Textual UI
from core.metacognition.monitor import MetacognitionMonitor

monitor = MetacognitionMonitor()
monitor.load_state()

# Get status for display
status = monitor.get_status_summary()
# {
#   "strategy": "incremental",
#   "progress": "making_progress",
#   "cognitive_load": 0.65,
#   "is_spinning": False,
#   "flow_state": True
# }

# Render in TUI
tui.render_cognitive_status(status)
```

---

## Performance Benchmarks

### I/O Latency (macOS, SSD)

| Operation | Before (v0.4) | After (v0.5) | Improvement |
|-----------|---------------|--------------|-------------|
| Cold load | 100ms | 10ms | **90%** |
| Cached read | N/A | 0.1ms | **N/A** |
| Write (batched) | 80ms | 1ms | **99%** |
| Flush (4 files) | N/A | 20ms | **N/A** |
| Per-turn overhead | 180ms | 11ms | **94%** |

### Memory Usage

| Component | Memory |
|-----------|--------|
| IOManager | ~2 MB (cache) |
| MetacognitionMonitor | ~1 MB |
| GoalManager | ~1 MB |
| Total (per agent) | ~5 MB |

### Scaling Characteristics

**Single Agent**:
- Initialization: 10ms
- Per-action: 11ms (includes I/O)
- Per-turn: 11ms + flush (20ms every 5s)

**10 Concurrent Agents**:
- Shared IOManager cache
- Independent write queues
- Flush: 200ms (10 agents × 4 files each, batched)

---

## Schema Versioning

**Current Version**: 0.3
**Compatibility**: Forward-compatible (old agents can read new schemas)

### Version History

- **v0.1** (2025-11-15): Initial schema
- **v0.2** (2025-12-01): Added flow state indicators
- **v0.3** (2025-12-20): Added self-corrections, improved help seeking

### Migration Strategy

When updating schemas:
1. Add `schema_version` field to all JSON files
2. Include `producer` metadata (name, version)
3. Maintain backward compatibility for 2 versions
4. Provide auto-migration via SchemaValidator

---

## Summary

The Cognitive Protocol provides a **complete metacognitive framework** for autonomous agents:

- **60+ configurable parameters** via TOML
- **5 personality profiles** for different tasks
- **90-95% I/O performance improvement** with batching and caching
- **Runtime validation** with auto-fix capability
- **Schema versioning** with migration support
- **Multi-agent coordination** via hivemind
- **Seamless integration** with oracle-code and HAFS TUI

**Next Steps**:
1. Review `docs/guides/CONFIGURATION.md` for configuration tuning
2. See your plugin training plan (`MODEL_TRAINING_PLAN.md`) for specialized experts
3. Explore `docs/architecture/MOE_SYSTEM.md` for multi-expert orchestration
