# Code Porting Plan: Consolidate hafs.* to Top-Level Modules

## Overview

Port code from `src/hafs/agents/`, `src/hafs/backends/`, and `src/hafs/core/services/` to new top-level directories with backward compatibility via re-exports.

**Goals:**
- New `src/agents/`, `src/backends/`, `src/services/` become canonical locations
- Old `hafs.*` imports continue working with deprecation warnings
- Agents organized by category for easier navigation

---

## Target Structure

### `src/agents/` (organized by category)

```
src/agents/
├── __init__.py              # Re-exports all public symbols
├── core/                    # Infrastructure
│   ├── base.py              # BaseAgent
│   ├── coordinator.py       # AgentCoordinator
│   ├── lane.py              # AgentLane
│   ├── router.py            # MentionRouter
│   └── roles.py             # Role definitions
├── autonomy/                # Self-improvement agents
│   ├── base.py              # MemoryAwareAgent
│   ├── self_improvement.py
│   ├── curiosity.py
│   ├── self_healing.py
│   └── hallucination.py
├── knowledge/               # ROM/KB agents
│   ├── alttp_*.py           # ALTTP knowledge files
│   ├── rom_specialist.py
│   ├── kb_enhancer.py
│   └── oracle_*.py
├── pipeline/                # Development workflow
│   ├── architect_council.py
│   ├── builder_council.py
│   └── validator_council.py
├── analysis/                # Context/analysis
│   ├── context_report_pipeline.py
│   ├── embedding_analysis.py
│   └── code_describer.py
├── swarm/                   # Swarm orchestration
│   ├── swarm.py
│   └── specialists.py
├── mission/                 # Autonomous mission agents
│   └── mission_agents.py
└── utility/                 # Standalone agents
    ├── cartographer.py
    ├── chronos.py
    ├── daily_briefing.py
    └── ... (remaining agents)
```

### `src/backends/`

```
src/backends/
├── __init__.py              # Re-exports + registry init
├── base.py                  # BaseChatBackend, BackendRegistry
├── cli/                     # PTY-based
│   ├── claude.py
│   ├── gemini.py
│   └── pty.py
├── api/                     # Direct API
│   ├── anthropic.py
│   ├── openai.py
│   └── ollama.py
├── oneshot/                 # One-shot backends
│   ├── claude.py
│   └── gemini.py
└── wrappers/
    └── history.py           # HistoryBackend
```

### `src/services/`

```
src/services/
├── __init__.py
├── manager.py               # ServiceManager
├── models.py                # ServiceDefinition, ServiceState, etc.
└── adapters/
    ├── base.py
    ├── launchd.py
    └── systemd.py
```

---

## Implementation Order

### Phase 1: Backends (lowest dependencies)

1. Create `src/backends/base.py` - copy from `hafs/backends/base.py`
2. Create `src/backends/cli/` - port `pty.py`, `claude.py`, `gemini.py`
3. Create `src/backends/api/` - port `anthropic.py`, `openai.py`, `ollama.py`
4. Create `src/backends/oneshot/` - split `oneshot.py` into two files
5. Create `src/backends/wrappers/history.py`
6. Create `src/backends/__init__.py` with registry initialization
7. Update `src/hafs/backends/__init__.py` for backward compat:
   ```python
   from backends import *  # Re-export from new location
   ```

### Phase 2: Services

1. Create `src/services/models.py`
2. Create `src/services/adapters/` with base, launchd, systemd
3. Create `src/services/manager.py`
4. Create `src/services/__init__.py`
5. Update `src/hafs/core/services/__init__.py` for backward compat

### Phase 3: Agents Core

1. Create `src/agents/core/` with:
   - `base.py` (BaseAgent)
   - `roles.py` (role definitions)
   - `router.py` (MentionRouter)
   - `lane.py` (AgentLane)
   - `coordinator.py` (AgentCoordinator)
2. Update imports to use `backends.*` and `services.*`

### Phase 4-10: Remaining Agent Categories

Port each category in order:
4. `autonomy/` - Split `autonomy_agents.py` into individual files
5. `knowledge/` - All ALTTP and KB-related agents
6. `pipeline/` - Architect, Builder, Validator councils
7. `analysis/` - Context pipelines, embedding analysis
8. `swarm/` - SwarmCouncil and specialists
9. `mission/` - Mission agents
10. `utility/` - All remaining standalone agents

### Phase 11: Backward Compatibility

Update `src/hafs/agents/__init__.py`:
```python
import warnings
warnings.warn(
    "hafs.agents is deprecated. Import from 'agents' instead.",
    DeprecationWarning, stacklevel=2
)
from agents import *
```

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/hafs/backends/base.py` | BackendRegistry pattern to preserve |
| `src/hafs/agents/coordinator.py` | Most complex, many dependencies |
| `src/hafs/agents/autonomy_agents.py` | Needs splitting into multiple files |
| `src/hafs/core/services/manager.py` | Service management logic |

## Dependencies to Keep in hafs.*

These modules stay in place (agents/backends depend on them):
- `hafs.models.agent` - Core data models
- `hafs.core.orchestrator` - Model orchestration
- `hafs.core.history` - History logging
- `hafs.core.orchestration` - Pipeline orchestration
- `hafs.core.execution` - Execution policies
- `hafs.config.*` - Configuration

## Testing Approach

1. **Before each phase**: Run existing tests as baseline
2. **After each phase**:
   - Verify old imports still work via backward compat
   - Verify new imports work
   - Check backend registry has all backends registered
3. **Integration test**: Full agent creation with new import paths

---

## Estimated Scope

- **Backends**: ~10 files to port, minimal import changes
- **Services**: ~7 files to port, minimal changes
- **Agents**: ~54 files to port, most work in organizing categories
