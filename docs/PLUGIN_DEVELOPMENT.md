# Plugin Development Guide

Extending HAFS is easy. The system uses a dynamic discovery mechanism to find and load plugins at runtime.

## 1. Structure of a Plugin

A HAFS plugin is simply a Python package. It can be installed via `pip` or just sit in your `PYTHONPATH`.

The only requirement is that it must contain a `hafs_plugin.py` file at its root.

```
my-hafs-plugin/
├── pyproject.toml
└── src/
    └── my_hafs_plugin/
        ├── __init__.py
        ├── hafs_plugin.py  <-- Entry Point
        ├── agents.py
        └── adapters.py
```

## 2. The `register` Function

Your `hafs_plugin.py` MUST export a `register` function that accepts an `AgentRegistry`.

```python
from hafs.core.registry import AgentRegistry
from .agents import MySuperAgent

def register(registry: AgentRegistry):
    # Register your custom agent
    registry.register_agent(MySuperAgent)
    print("MySuperAgent loaded!")
```

## 3. Creating a Custom Agent

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

## 4. Advanced Patterns

### Overriding Core Agents
You can replace a core agent (e.g., `ReviewUploader`) with a specialized implementation (e.g., `GitHubUploader`) by registering it with the same name.

```python
# In hafs_plugin.py
from .agents import GitHubUploader

def register(registry):
    # This replaces any existing "ReviewUploader" in the registry
    registry.register_agent(GitHubUploader, name_override="ReviewUploader")
```

### UI Extensions
Plugins can add pages to the Web Hub sidebar.

```python
# In hafs_plugin.py
from hafs.core.ui_registry import ui_registry
from .ui import render_my_page

def register(registry):
    ui_registry.register_page("My Custom Page", render_my_page)
```

### Instantiation Safety
If your agent requires arguments in `__init__` (e.g., `workspace_path`), **do not** expect it to be auto-instantiated by the default `SwarmCouncil`. These agents are typically used by specific pipelines or need a custom factory.

If you must register it, ensure you handle its instantiation logic in your own orchestration code, or provide a default no-arg constructor.

## 5. Testing Your Plugin

You can test your plugin using the HAFS CLI or Web Hub.

1.  Enable it in `~/.context/hafs_config.toml`: `plugins = ["my_hafs_plugin"]`.
2.  Run `hafs-hub`.
3.  Go to the **"Agents"** page. You should see `MySuperAgent` listed there.