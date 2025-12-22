# hAFS (Halext Agentic File System)

hAFS is a framework for orchestrating agents, managing context, and executing tool-gated workflows across projects and nodes. It is designed to be:

*   **Modular:** Core logic is separated from specific tools. Use plugins to connect to issue trackers, code review systems, and version control.
*   **Cognitive:** Agents track session state (Anxiety, Confidence, Curiosity) to adjust their prompts.
*   **Orchestrated:** Built-in pipelines for plan → execute → verify → summarize flows.
*   **Observable:** History logging plus a Streamlit dashboard for monitoring swarms and knowledge.

## Key Components

### 1. The Swarm Council
The orchestrator that manages specialized agents.
*   **Strategist:** Breaks down high-level goals into actionable plans.
*   **Reviewer:** Critiques findings and ensures quality.
*   **Documenter:** Synthesizes findings into reports.
*   **Trend Watcher:** Flags emerging issues.

### 2. Pipelines
Three-stage pipeline for multi-step work:
1.  **Architect:** Generates a Technical Design Doc (TDD) and a structured `plan.json`.
2.  **Builder:** Writes code iteratively and handles build errors.
3.  **Validator:** Writes and runs tests, then prepares a review request.

### 3. Plugin Architecture
hAFS defines interfaces for agents and integrations.
*   **Core:** `SwarmCouncil`, `CognitiveLayer`, and `ModelOrchestrator`.
*   **Plugins:** Provide integrations (issue trackers, code review, version control).

### 4. Semantic Context
Vector-based semantic search pulls relevant "Verified Knowledge" from the local context store (`~/.context`) before actions are taken.

### 5. Context Engineering Pipeline
Constructor, updater, and evaluator stages build agent-ready context windows with token budgeting and feedback loops.

## Installation

```bash
pip install hafs
```

## Configuration

hAFS loads configuration from a layered TOML setup:
1. `~/.config/hafs/config.toml` (local overrides, auto-created)
2. `hafs.toml` (project defaults)
3. `~/.context/hafs_config.toml` (legacy fallback)

On first run, hAFS creates the local config and seeds it with discovered AFS projects.

```toml
[general]
context_root = "~/.context"
agent_workspaces_dir = "~/AgentWorkspaces"

[plugins]
enabled_plugins = ["hafs_plugin_github"]
```

See `docs/guides/CONFIGURATION.md` for full examples.

## Usage

### Launch the Web Hub
Streamlit dashboard for monitoring agents, knowledge, and infrastructure.
```bash
./scripts/launch_web_hub.sh
```

### CLI Commands
Examples for headless operation.
```bash
# Run an orchestration pipeline
hafs orchestrate run "Investigate ALTTP routine X"

# Start an interactive coordinator chat
hafs chat

# Inspect context pipeline status
hafs context status

# Query agent memory
hafs memory recall --agent Generalist "what did we decide?"

# Launch the TUI (default)
hafs
```

## Training & Improvement

hAFS includes a self-improvement loop via the `agents.training` module.
*   **HafsSystemGenerator:** Generates training data for HAFS tool usage (CLI, scripts, context).
*   **Plugins:** Domain-specific generators (e.g., ASM, C++) are loaded from plugins like `my_hafs_plugin`.

## Developing Plugins

Create a standard Python package with a `hafs_plugin.py` entry point.

```python
# my_plugin/hafs_plugin.py
from hafs.core.registry import AgentRegistry
from my_plugin.agents import MyCustomAgent

def register(registry: AgentRegistry):
    registry.register_agent(MyCustomAgent)
    print("My Custom Plugin Loaded!")
```

See [Plugin Development Guide](docs/plugins/PLUGIN_DEVELOPMENT.md) for details.

## Documentation

See `docs/README.md` for the full navigation map.

*   [Architecture Overview](docs/architecture/ARCHITECTURE.md)
*   [Agentic System Guide](docs/guides/AGENTIC_SYSTEM_GUIDE.md)
*   [Configuration Guide](docs/guides/CONFIGURATION.md)
*   [Usage Guide](docs/guides/USAGE.md)
*   [Improvements Roadmap](docs/backlog/IMPROVEMENTS.md)
*   [Research Alignment Plan](docs/plans/RESEARCH_ALIGNMENT_PLAN.md)
*   [Chat Mode + Renderer Plan](docs/plans/CHAT_MODE_RENDERER_PLAN.md)
*   [Plugin Adapter Guide](docs/plugins/PLUGIN_ADAPTER_GUIDE.md)

## License

MIT
