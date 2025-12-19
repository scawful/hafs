# HAFS Architecture Overview

HAFS is designed as a layered system, moving from abstract cognition to concrete execution.

## 1. The Cognitive Core (`hafs.core`)

At the center of the system is the **Cognitive Layer**, which ensures agents behave intelligently and consistently.

*   **`ModelOrchestrator`**: Compatibility wrapper that now delegates to `UnifiedOrchestrator` (multi-provider) when available. It handles:
    *   **Tiered Fallback:** Automatically downgrades from "Reasoning" (Gemini 3 Pro) to "Fast" (Flash) if quotas are exceeded (429 errors).
    *   **Cost Tracking:** Logs token usage for every call.
*   **`CognitiveLayer`**: Maintains the system's "Emotional State" (Anxiety, Confidence, Curiosity). This state is injected into every agent prompt.
*   **`AgentRegistry`**: The dynamic phonebook. It knows about every available agent class.
*   **`HAFSConfig`**: A singleton wrapper over the unified config loader (`hafs.toml`, `~/.config/hafs/config.toml`, legacy fallback). It is the source of truth for paths and plugin lists.

## 2. The Agent Layer (`hafs.agents`)

Agents are the workers. All agents inherit from `BaseAgent`, giving them shared DNA:
*   **Context Injection:** They automatically pull relevant "Verified Knowledge" via vector search.
*   **Tool Access:** They know which tools are available.
*   **Metric Logging:** Every action is recorded.
*   **Personas & Skills:** Persona profiles map roles to prompts, skills, and execution modes.

### Agent Archetypes
*   **Collectors:** Go out and get data (e.g., `BugCollector`, `CodeExplorer`).
*   **Specialists:** Perform complex reasoning (e.g., `SwarmStrategist`, `DeepDiveDocumenter`).
*   **Doers:** Execute changes (e.g., `CodeWriter`, `ShellAgent`).
*   **KnowledgeGraphAgent:** Merges verified/discovered docs with disassembly KB outputs
    into the shared `knowledge_graph.json` for the web dashboard.

## 3. The Orchestration Layer (`SwarmCouncil`)

The **Swarm Council** is the "manager."
*   **Dependency Injection:** It accepts a dictionary of *instantiated* agents at startup. This allows for easy mocking in tests and swapping of implementations via plugins.
*   **Execution:** It defines the multi-phase process (Planning -> Collection -> Refinement -> Synthesis).
*   **Pipeline Scaffold:** `OrchestrationPipeline` standardizes plan → execute → verify → summarize flows.
*   **Unified Entry:** `hafs.core.orchestration_entrypoint` provides a single entrypoint that can run either coordinator or swarm modes.

## 4. History & Memory Pipelines (`hafs.core.history`)

The history layer provides an immutable log plus semantic search and summaries.

*   **HistoryLogger + SessionManager:** Append-only JSONL logs and session metadata.
*   **HistoryEmbeddingIndex:** Embeds entry-level history records for semantic recall.
*   **HistorySessionSummaryIndex:** Generates session summaries + embeddings, auto-run on session completion.
*   **UI/CLI:** History search is available in the Logs screen and `hafs history` commands.

## 5. Project Registry & Tooling (`hafs.core.projects`, `hafs.core.tooling`)

Background agents now use a **Project Registry** to understand which repos to scan
and which tools they are allowed to run for each project.

*   **Project Catalog:** `projects` and `tracked_projects` in `hafs.toml` are loaded
    into a registry that normalizes names, paths, tags, and knowledge roots.
*   **Knowledge Indexing:** `EmbeddingService` syncs from the Project Registry to
    build per-repo embedding indexes with checkpointing.
*   **Tool Profiles:** `tool_profiles` define allow/deny lists for tool access
    (e.g., read-only search vs. test execution).
*   **Tool Runner:** Tools execute in a project root with explicit allowlists
    (no shell expansion), returning structured results for agents.
*   **Execution Policy:** `ExecutionPolicy` resolves tool profiles via project overrides
    plus execution mode (`HAFS_EXEC_MODE`).
*   **Execution Modes:** `execution_modes` select tool profiles (read-only vs build-only)
    and are enforced by ShellAgent/pipelines at runtime.
*   **AFS Policy Guard:** `.context/metadata.json` gating blocks write/build/test tools
    unless `tools` is marked executable.

## 6. The Plugin Layer (PluginLoader + IntegrationPlugin)

This is where HAFS becomes extensible.
*   Plugins are Python packages discovered via entry points (`hafs.plugins`) or `plugins.plugin_dirs`.
*   Recommended: implement a `Plugin` class (`HafsPlugin` + optional `IntegrationPlugin`, `BackendPlugin`, etc.).
*   Legacy: `register(registry)` functions are still supported for older plugins.
*   **IntegrationPlugin** adapters standardize external providers (issue tracker, code review, code search).

*Example:* `hafs-google-adapter` implements `IntegrationPlugin` and returns a review adapter class. When the core agents ask for `code_review`, they receive the Google implementation if the plugin is loaded.

## 7. The User Interface (`hafs.ui`)

The **Web Hub** is the cockpit.
*   **Registry-Based:** Pages are registered via `hafs.core.ui_registry`. Plugins can add their own views (e.g., "My Work").
*   **Live:** Connects to the shared `.context` state to show real-time agent updates.

## 8. Multi-Node Infrastructure (`hafs.core.nodes`)

HAFS can route work across multiple machines.
*   **NodeManager:** Loads `nodes.toml` and tracks multi-role nodes (compute, server, mobile).
*   **Health Checks:** Ollama nodes use `/api/tags`; other nodes can specify `health_url`.
*   **Routing Hooks:** UnifiedOrchestrator uses NodeManager to pick the best Ollama node.
*   **AFS Sync:** `AFSSyncService` consumes `sync.toml` profiles and runs guarded sync
    via ToolRunner (e.g., rsync/ssh).
