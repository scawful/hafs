# Plugin Development Guide

Extending HAFS is easy. The system uses a dynamic discovery mechanism to find and load plugins at runtime.

## 1. Structure of a Plugin

A HAFS plugin is a Python package. It can be installed via `pip` or loaded from a plugin directory.
You can implement either the **new Plugin protocol** or the **legacy register function** (still supported).

```
my-hafs-plugin/
├── pyproject.toml
└── src/
    └── my_hafs_plugin/
        ├── __init__.py
        ├── hafs_plugin.py  <-- Legacy entry point (optional)
        ├── plugin.py       <-- Plugin class (recommended)
        ├── agents.py
        └── adapters.py
```

Plugins can be discovered via:
- Entry points: `hafs.plugins`
- Plugin dirs: `plugins.plugin_dirs` in `hafs.toml` / `~/.config/hafs/config.toml`

## 2. The Legacy `register` Function (Optional)

Legacy plugins can still export a `register` function that accepts an `AgentRegistry`.

```python
from hafs.core.registry import AgentRegistry
from .agents import MySuperAgent

def register(registry: AgentRegistry):
    # Register your custom agent
    registry.register_agent(MySuperAgent)
    print("MySuperAgent loaded!")
```

## 3. The Plugin Protocol (Recommended)

New plugins should implement `HafsPlugin` and optional capability protocols
(`BackendPlugin`, `IntegrationPlugin`, `WidgetPlugin`, etc.).

```python
from hafs.plugins.protocol import HafsPlugin, IntegrationPlugin
from hafs.adapters.protocols import IssueTrackerAdapter

class Plugin(HafsPlugin, IntegrationPlugin):
    @property
    def name(self) -> str:
        return "my-plugin"

    @property
    def version(self) -> str:
        return "1.0.0"

    def activate(self, app) -> None:
        # Optional: register UI widgets, bindings, etc.
        pass

    def deactivate(self) -> None:
        pass

    def get_issue_tracker(self) -> type[IssueTrackerAdapter] | None:
        from .adapters import MyIssueTracker
        return MyIssueTracker
```

## 4. Creating a Custom Agent

Inherit from `BaseAgent`.

```python
from hafs.agents.base import BaseAgent

class MySuperAgent(BaseAgent):
    def __init__(self):
        super().__init__("MySuperAgent", "I do super things.")
        # 'fast' (Flash), 'reasoning' (Pro), or 'coding'
        self.model_tier = "reasoning" 

    async def run_task(self, input_data: str) -> str:
        # 1. Use the LLM
        prompt = f"Analyze this: {input_data}"
        thought = await self.generate_thought(prompt)
        
        # 2. Do something with it
        return f"Analysis complete: {thought}"
```

## 5. Advanced Patterns

### Overriding Core Agents
You can replace a core agent (e.g., `ReviewUploader`) with a specialized implementation (e.g., `GitHubUploader`) by registering it with the same name.

```python
# In plugin.py (Plugin.activate) or legacy hafs_plugin.py
from .agents import GitHubUploader

def register(registry):
    # This replaces any existing "ReviewUploader" in the registry
    registry.register_agent(GitHubUploader, name_override="ReviewUploader")
```

### UI Extensions
Plugins can add pages to the Web Hub sidebar.

```python
# In plugin.py (Plugin.activate) or legacy hafs_plugin.py
from hafs.core.ui_registry import ui_registry
from .ui import render_my_page

def register(registry):
    ui_registry.register_page("My Custom Page", render_my_page)
```

### Instantiation Safety
If your agent requires arguments in `__init__` (e.g., `workspace_path`), **do not** expect it to be auto-instantiated by the default `SwarmCouncil`. These agents are typically used by specific pipelines or need a custom factory.

If you must register it, ensure you handle its instantiation logic in your own orchestration code, or provide a default no-arg constructor.

## 6. Testing Your Plugin

You can test your plugin using the HAFS CLI or Web Hub.

1.  Enable it in `~/.config/hafs/config.toml`: `[plugins].enabled_plugins = ["my_hafs_plugin"]`.
2.  Run `hafs-hub`.
3.  Go to the **"Agents"** page. You should see `MySuperAgent` listed there.
