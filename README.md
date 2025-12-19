# HAFS (Halext Agentic File System)

**A Modular, Autonomous Agent Framework.**

HAFS is a professional-grade framework for building, orchestrating, and deploying autonomous AI agent swarms. It is designed to be:

*   **Modular:** Core logic is separated from specific tools. Use plugins to connect to GitHub, Jira, or internal corporate systems (like Google's).
*   **Cognitive:** Agents track their "Emotional State" (Anxiety, Confidence, Curiosity) to adjust their behavior dynamically.
*   **Autonomous:** Built-in pipelines for "Prompt-to-Product" workflows: Architecting, Building, and Validating code without human intervention.
*   **Observable:** A unified Web Hub dashboard for monitoring swarms, managing knowledge, and intervening when necessary.

![HAFS Hub](https://via.placeholder.com/800x400?text=HAFS+Web+Hub+Dashboard)

## 🚀 Key Features

### 1. The Swarm Council
A sophisticated orchestrator that manages a team of specialized agents.
*   **Strategist:** Breaks down high-level goals into actionable plans.
*   **Reviewer:** Critiques findings and ensures quality.
*   **Documenter:** Synthesizes vast amounts of information into clean reports.
*   **Trend Watcher:** Proactively identifies emerging issues.

### 2. Autonomous Pipelines
HAFS implements a three-stage pipeline to turn a simple prompt into working code:
1.  **Architect:** Generates a Technical Design Doc (TDD) and a structured `plan.json`.
2.  **Builder:** Writes code iteratively, attempting to fix build errors automatically.
3.  **Validator:** Writes and runs tests, then prepares a Change List (CL/PR) for review.

### 3. Plugin Architecture
HAFS is agnostic. It defines the *interfaces* for agents.
*   **Core:** Contains the `SwarmCouncil`, `CognitiveLayer`, and `ModelOrchestrator`.
*   **Plugins:** define specific capabilities.
    *   *Example:* Adapter plugins can provide integrations for issue trackers, code review systems, and version control.
    *   *Example:* `hafs-github-adapter` (Community) could provide agents for Issues and PRs.

### 4. Semantic Context
Agents don't just "guess." They use vector-based semantic search to pull the most relevant "Verified Knowledge" from your local context store (`~/.context`) before taking any action.

## 📦 Installation

```bash
pip install hafs
```

### Optional: Google Internal Adapter
If you are a Googler, install the internal adapter (requires internal repository access):
```bash
pip install hafs-google-adapter
```

## 🛠️ Configuration

HAFS uses a single, centralized configuration file at `~/.context/hafs_config.toml`.

```toml
[core]
context_root = "~/.context"
agent_workspaces = "~/AgentWorkspaces"

[llm]
# Your API Key (or use env var AISTUDIO_API_KEY)
aistudio_api_key = "..."
default_reasoning_model = "gemini-3-pro-preview"

[user_preferences]
username = "your_username"
```

## 🖥️ Usage

### Launch the Web Hub
The central command center for your agent swarms.
```bash
hafs-hub
```
Access it at `http://localhost:8501`.

### CLI Commands
HAFS also provides a CLI for headless operation.
```bash
# Run a specific research session
hafs run --topic "Legacy Code Migration"

# Launch the Architect Pipeline on a prompt file
hafs pipeline architect --prompt ./my_feature_idea.md
```

## 🧩 Developing Plugins

Create a standard Python package with a `hafs_plugin.py` entry point.

```python
# my_plugin/hafs_plugin.py
from hafs.core.registry import AgentRegistry
from my_plugin.agents import MyCustomAgent

def register(registry: AgentRegistry):
    registry.register_agent(MyCustomAgent)
    print("My Custom Plugin Loaded!")
```

See [Plugin Development Guide](docs/PLUGIN_DEVELOPMENT.md) for details.

## 📄 Documentation

*   [Architecture Overview](docs/ARCHITECTURE.md)
*   [Autonomous Workflow](docs/AUTONOMOUS_WORKFLOW.md)
*   [Configuration Guide](docs/CONFIGURATION.md)

## License

MIT