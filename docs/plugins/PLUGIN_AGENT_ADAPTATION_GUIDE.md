# Plugin Agent Adaptation Guide

This guide helps plugin providers port custom agents into HAFS and wire them into the plugin system.
It assumes you already have an agent implementation and want it to run inside HAFS reliably.

---

## 1. Choose the right agent shape

HAFS supports two common agent styles. Pick the one that matches your existing agent.

### 1.1 BaseAgent (standalone task runner)
- Best for single-task agents that generate a report or perform a focused action.
- Uses `BaseAgent` in `src/hafs/agents/base.py`.
- Runs via `await agent.run_task(...)`.

### 1.2 Coordinator agents (multi-agent orchestration)
- Use `AgentCoordinator` with `AgentLane` for multi-agent workflows.
- Best when your agent is part of a coordinated plan with a shared context.

If you are porting a single agent, start with `BaseAgent`.

---

## 2. Adapt your agent to BaseAgent

Minimum requirements:
- A no-arg constructor if it will be registered in the global registry.
- Call `super().__init__(name, role_description)`.
- Implement `async def run_task(...)`.

```python
from hafs.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__("MyCustomAgent", "Explain and summarize build failures")
        self.model_tier = "fast"  # fast | reasoning | coding | research | creative

    async def run_task(self, payload: str) -> str:
        prompt = f"Analyze this failure:\n{payload}"
        return await self.generate_thought(prompt)
```

### No-arg constructor requirement
`AgentRegistry.get_agent()` instantiates agents with no arguments. If your agent needs
extra constructor args (paths, credentials), either:
- Provide defaults in `__init__`, or
- Register via a custom factory in your own orchestration entrypoint (not the global registry).

---

## 3. Hook into memory and history (recommended)

If your agent produces recurring analysis, write to AgentMemory or History for traceability.

### AgentMemory (per-agent long-term memory)
```python
from hafs.core.history import AgentMemoryManager

memory = AgentMemoryManager(context_root).get_agent_memory("MyCustomAgent")
await memory.remember(
    content="Detected recurring failure in build step",
    memory_type="learning",
    context={"build": "backend"},
    importance=0.6,
)
```

### HistoryLogger (global session log)
```python
from hafs.core.history import HistoryLogger

logger = HistoryLogger(context_root / "history")
logger.log_system_event("custom_agent_ran", {"agent": "MyCustomAgent"})
```

---

## 4. Use safe tool execution

If your agent needs shell access, use `ToolRunner` with an ExecutionPolicy.
This respects AFS policies and configured tool profiles.

```python
from hafs.core.execution import ExecutionPolicy
from hafs.core.projects import ProjectRegistry
from hafs.core.tooling import ToolRunner

registry = ProjectRegistry.load()
project = registry.match_path(Path.cwd())
policy = ExecutionPolicy(registry=registry, execution_mode=None)
profile = policy.resolve_tool_profile(project)
runner = ToolRunner(Path.cwd(), profile)
result = await runner.run("rg", args=["TODO", "src"])
```

---

## 5. Register your agent in a plugin

You can register agents using the plugin protocol or the legacy `register` function.

### Plugin class (recommended)
```python
from hafs.plugins.protocol import HafsPlugin
from hafs.core.registry import agent_registry
from .agents import MyCustomAgent

class Plugin(HafsPlugin):
    @property
    def name(self) -> str:
        return "my-hafs-plugin"

    @property
    def version(self) -> str:
        return "0.1.0"

    def activate(self, app) -> None:
        agent_registry.register_agent(MyCustomAgent)

    def deactivate(self) -> None:
        pass
```

### Legacy register function
```python
from hafs.core.registry import agent_registry
from .agents import MyCustomAgent

def register(registry):
    registry.register_agent(MyCustomAgent)
```

Enable the plugin in `hafs.toml` or `~/.config/hafs/config.toml`:

```toml
[plugins]
enabled_plugins = ["my-hafs-plugin"]
```

---

## 6. Optional: override a built-in agent

If you want to replace a core agent, register with `name_override`.

```python
registry.register_agent(MyCustomAgent, name_override="ReportManager")
```

Only do this if you fully own the behavior, because it affects core workflows.

---

## 7. Porting checklist

- [ ] Agent uses `BaseAgent` and has a no-arg `__init__`.
- [ ] `run_task()` accepts a clear input contract.
- [ ] Model tier set (`fast` for quick summaries, `reasoning` for deep analysis).
- [ ] Tool usage goes through `ToolRunner` (no direct `subprocess`).
- [ ] Optional: writes to `AgentMemory` or `HistoryLogger`.
- [ ] Registered via plugin activation or legacy `register()`.
- [ ] Config pulled from `hafs.toml` or environment variables.

---

## 8. Related docs

- `docs/PLUGIN_DEVELOPMENT.md`
- `docs/PLUGIN_ADAPTER_GUIDE.md`
- `docs/AGENTS_QUICKSTART.md`

