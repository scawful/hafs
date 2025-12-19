# HAFS Architecture Overview

HAFS is designed as a layered system, moving from abstract cognition to concrete execution.

## 1. The Cognitive Core (`hafs.core`)

At the center of the system is the **Cognitive Layer**, which ensures agents behave intelligently and consistently.

*   **`ModelOrchestrator`**: A robust wrapper around LLM APIs (like Gemini). It handles:
    *   **Tiered Fallback:** Automatically downgrades from "Reasoning" (Gemini 3 Pro) to "Fast" (Flash) if quotas are exceeded (429 errors).
    *   **Cost Tracking:** Logs token usage for every call.
*   **`CognitiveLayer`**: Maintains the system's "Emotional State" (Anxiety, Confidence, Curiosity). This state is injected into every agent prompt.
*   **`AgentRegistry`**: The dynamic phonebook. It knows about every available agent class.
*   **`HAFSConfig`**: A singleton that loads configuration from `~/.context/hafs_config.toml`. It is the source of truth for paths, API keys, and plugin lists.

## 2. The Agent Layer (`hafs.agents`)

Agents are the workers. All agents inherit from `BaseAgent`, giving them shared DNA:
*   **Context Injection:** They automatically pull relevant "Verified Knowledge" via vector search.
*   **Tool Access:** They know which tools are available.
*   **Metric Logging:** Every action is recorded.

### Agent Archetypes
*   **Collectors:** Go out and get data (e.g., `BugCollector`, `CodeExplorer`).
*   **Specialists:** Perform complex reasoning (e.g., `SwarmStrategist`, `DeepDiveDocumenter`).
*   **Doers:** Execute changes (e.g., `CodeWriter`, `PiperAgent` / `ShellAgent`).

## 3. The Orchestration Layer (`SwarmCouncil`)

The **Swarm Council** is the "manager."
*   **Dependency Injection:** It accepts a dictionary of *instantiated* agents at startup. This allows for easy mocking in tests and swapping of implementations via plugins.
*   **Execution:** It defines the multi-phase process (Planning -> Collection -> Refinement -> Synthesis).

## 4. The Plugin Layer (`hafs_plugin.py`)

This is where HAFS becomes extensible.
*   A plugin is a Python package that exports a `register(registry)` function.
*   It can register **Agents**, **Adapters**, and **UI Pages**.

*Example:* `hafs-google-adapter` registers a `GoogleReviewUploader`. When the core `ValidatorCouncil` asks for a `ReviewUploader`, it gets the Google version if the plugin is loaded.

## 5. The User Interface (`hafs.ui`)

The **Web Hub** is the cockpit.
*   **Registry-Based:** Pages are registered via `hafs.core.ui_registry`. Plugins can add their own views (e.g., "My Work").
*   **Live:** Connects to the shared `.context` state to show real-time agent updates.
