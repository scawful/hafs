# HAFS Code Quality Report

Generated: 2025-12-19

## Executive Summary

The codebase has solid architectural foundations with sophisticated multi-agent orchestration. However, several critical issues need attention before production use:

| Area | Critical | High | Medium | Low |
|------|----------|------|--------|-----|
| Background Agents | 2 | 3 | 4 | 2 |
| Swarm/Council | 1 | 3 | 4 | 2 |
| Service Management | 5 | 6 | 16 | 12 |
| Plugin Architecture | 2 | 1 | 2 | 1 |
| **Total** | **10** | **13** | **26** | **17** |

---

## 1. Critical Issues (Immediate Action Required)

### 1.1 Inconsistent Registry Pattern

**Files Affected:**
- `src/hafs/agents/context_builder.py` - Uses hardcoded `GenericAdapter()`
- `src/hafs/agents/daily_briefing.py` - Uses hardcoded `GenericAdapter()`
- `src/hafs/agents/trend_watcher.py` - Correctly uses `agent_registry.get_adapter()`

**Problem:** Two of three background agents bypass the plugin registry, making adapter injection impossible.

**Fix Required:**
```python
# In AutonomousContextAgent.setup() and DailyBriefingAgent.setup()
from hafs.core.registry import agent_registry

async def setup(self):
    try:
        self.bugs = agent_registry.get_adapter("issue_tracker")
        await self.bugs.connect()
    except:
        self.bugs = GenericAdapter()  # Fallback
```

### 1.2 Naming Inconsistency in ContextBuilder

**Current:** Uses old internal names (`bugs`, `critique`, `codesearch`)
**Expected:** Standard names (`issue_tracker`, `code_review`, `code_search`)

**Impact:** Plugins cannot inject adapters because names don't match.

### 1.3 Service Management Security Vulnerabilities

**Command Injection (systemd.py:42):**
```python
f"ExecStart={' '.join(definition.command)}"  # No escaping!
```

**Environment Variable Injection (systemd.py:48-49):**
```python
f"Environment={key}={value}"  # Newlines can inject directives
```

**Fix:** Implement proper escaping for systemd unit files.

### 1.4 Missing Error Handling Everywhere

**Pattern Found:**
```python
except: pass  # Silent failures in swarm.py, trend_watcher.py
```

**Impact:** Failures are invisible; debugging is impossible.

### 1.5 Incomplete Orchestrator Fallback

**File:** `src/hafs/core/orchestrator.py:112-116`

```python
if self.gemini_cli_path:
    logger.info("Falling back to CLI execution...")
    # NO ACTUAL CLI EXECUTION - JUST LOGS
raise Exception(...)  # Always fails
```

---

## 2. Architectural Issues

### 2.1 Two Disconnected Agent Systems

| SwarmCouncil | AgentCoordinator |
|--------------|------------------|
| Simple BaseAgent pattern | Pydantic models with roles |
| Manual phase orchestration | MentionRouter + SharedContext |
| Used for autonomous research | Used for interactive coordination |
| No inter-agent communication | Full message passing |

**Recommendation:** Consolidate into single system using AgentCoordinator as foundation.

### 2.2 Missing SharedContext in SwarmCouncil

```python
# Current: String concatenation loses structure
f"DATA:\n{str(results)}\n\nCRITIQUE:\n{critique}"

# Should use: SharedContext pattern from AgentCoordinator
shared_context.add_finding(agent_name, structured_result)
```

### 2.3 Background Agents Not Using Config System

```python
# Hardcoded paths in multiple agents
self.context_root = Path.home() / ".context"

# Should use:
from hafs.core.config import hafs_config
self.context_root = hafs_config.context_root
```

---

## 3. Missing Functionality

### 3.1 Background Agents

| Feature | Status |
|---------|--------|
| `AutonomousContextAgent.explore_environment()` | Only reads AgentWorkspaces, never explores codebase |
| `ShadowObserver.process_command()` | Prints commands, does nothing with them |
| Multi-turn synthesis | Removed during sanitization |
| Metrics logging | Removed during sanitization |

### 3.2 Service Management

| Feature | Status |
|---------|--------|
| Health checks | Not implemented |
| Service dependencies | Not implemented |
| Log rotation | Not implemented |
| Resource limits | Not implemented |
| Timeout configuration | Hardcoded values |

### 3.3 Swarm Council

| Feature | Status |
|---------|--------|
| Inter-agent delegation | Not implemented |
| Consensus/voting | Not implemented |
| Conflict resolution | Not implemented |
| Retry logic | Not implemented |

---

## 4. Quick Wins (Easy Fixes)

### 4.1 Fix Registry Usage in Background Agents

```python
# daily_briefing.py - Replace lines 28-30
from hafs.core.registry import agent_registry

async def setup(self):
    await super().setup()
    try:
        self.code_review = agent_registry.get_adapter("code_review")
        await self.code_review.connect()
    except:
        self.code_review = GenericAdapter()

    try:
        self.issue_tracker = agent_registry.get_adapter("issue_tracker")
        await self.issue_tracker.connect()
    except:
        self.issue_tracker = GenericAdapter()
```

### 4.2 Fix ContextBuilder Naming

```python
# context_builder.py - Rename adapters
self.issue_tracker = GenericAdapter()  # was: self.bugs
self.code_review = GenericAdapter()    # was: self.critique
self.code_search = GenericAdapter()    # was: self.codesearch
```

### 4.3 Add Basic Error Logging

```python
# Replace all `except: pass` with:
import logging
logger = logging.getLogger(__name__)

except Exception as e:
    logger.warning(f"Operation failed: {e}")
```

### 4.4 Use Config for Paths

```python
from hafs.core.config import hafs_config, CONTEXT_ROOT, BRIEFINGS_DIR

# Replace hardcoded Path.home() / ".context" everywhere
```

---

## 5. Service Definition Issues

### 5.1 Built-in Services Reference Missing Modules

```python
# manager.py:76-104 - These modules don't exist:
"hafs.core.orchestrator", "--daemon"     # No orchestrator daemon mode
"hafs.agents.coordinator", "--daemon"    # No coordinator daemon mode
"hafs.ui.web_dashboard"                  # File doesn't exist
```

### 5.2 Streamlit Dependency Not Checked

```python
command=["streamlit", "run", ...]  # streamlit may not be installed
```

---

## 6. Test Coverage Gaps

| Component | Tests Exist |
|-----------|-------------|
| Background Agents | No |
| SwarmCouncil | Yes (basic) |
| AgentCoordinator | Yes |
| Service Management | No |
| Plugin Loader | No |

---

## 7. Recommended Priority Order

### Phase 1: Critical Fixes (This Week)
1. Fix registry pattern in DailyBriefingAgent and AutonomousContextAgent
2. Fix adapter naming in ContextBuilder
3. Add error handling to replace `except: pass`
4. Fix service management security issues

### Phase 2: Stability (Next Week)
1. Add timeout configuration to subprocess calls
2. Implement proper status synchronization
3. Add comprehensive logging
4. Verify built-in service definitions

### Phase 3: Completeness (Following Week)
1. Restore multi-turn synthesis in ContextBuilder
2. Implement SharedContext in SwarmCouncil
3. Add health checks to services
4. Create test coverage for new code

### Phase 4: Architecture (Ongoing)
1. Consolidate SwarmCouncil with AgentCoordinator
2. Implement inter-agent communication
3. Add service dependencies
4. Implement proper plugin discovery

---

## 8. Files Requiring Immediate Attention

| File | Issues | Priority |
|------|--------|----------|
| `src/hafs/agents/context_builder.py` | Registry pattern, naming, incomplete methods | Critical |
| `src/hafs/agents/daily_briefing.py` | Registry pattern | Critical |
| `src/hafs/core/services/adapters/systemd.py` | Security (command injection) | Critical |
| `src/hafs/agents/swarm.py` | Error handling, context propagation | High |
| `src/hafs/core/orchestrator.py` | Incomplete fallback | High |
| `src/hafs/agents/shadow_observer.py` | No-op implementation | Medium |
| `src/hafs/core/services/manager.py` | Missing module verification | Medium |
