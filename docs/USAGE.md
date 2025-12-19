# HAFS (Agentic File System) Usage Guide

HAFS is a protocol and toolset for managing AI agent context. It ensures agents have access to the right "memory" and "knowledge" without overwhelming their token window.

## CLI Usage

### Initialize a project
Sets up the `.context` structure in the current directory.
```bash
hafs init
```

### Mount a Context
Injects specific files into the agent's working memory.
```bash
hafs mount <type> <source_path>
```
*Types:* `memory`, `knowledge`, `tools`, `scratchpad`, `history`.

### Clean Context
Removes stale or transient files from the scratchpad and history.
```bash
hafs clean
```

### History Search
Index and search AFS history embeddings.
```bash
hafs history index
hafs history summarize
hafs history search "query"
hafs history search "query" --mode all
```

### Orchestration Pipeline
Run the unified plan → execute → verify → summarize pipeline.
```bash
hafs orchestrate "Investigate ALTTP routine X"
hafs orchestrate "Daily infra health" --mode swarm
```

### Knowledge Indexing
Index configured projects for embeddings (uses `projects` from `hafs.toml`).
```bash
hafs embed index
hafs embed index oracle-code
```

### Nodes (Multi-Node Registry)
Manage distributed compute/AFS nodes.
```bash
hafs nodes list
hafs nodes status
hafs nodes show halext-server
hafs nodes discover
```

### AFS Sync
Run sync profiles defined in `sync.toml`.
```bash
hafs sync list
hafs sync show global
hafs sync run global --dry-run
```

## TUI Interaction
Launch the interactive explorer:
```bash
hafs tui
```
*   **Navigation:** Use Arrow keys or Vim keys (h/j/k/l).
*   **Mounting:** Press `m` on a file to mount it to the active context.
*   **Chat:** Press `c` to open the agent communication panel.
*   **History Search:** Use the Logs screen tab "AFS History" to run semantic search.

## Execution Modes
Use `HAFS_EXEC_MODE` to switch between tool profiles (e.g., `read_only`, `build_only`, `infra_ops`).

## The "Fears" Protocol
Located at `.context/memory/fears.json`.
This file contains JSON-encoded "Risk Patterns".
- **Format:**
  ```json
  [
    { "pattern": "blocking system call", "severity": "high", "remediation": "use non-blocking popen" }
  ]
  ```
- **Usage:** Agents are required to check this file before proposing code changes. If a proposed change matches a pattern, the agent must warn the user.

## IDE Integration (VS Code / OpenCode)
The `@halext/afs` package in `halext-code` automatically detects the `.context` folder.
- **Auto-Sync:** Changes made in the IDE to `scratchpad/` are immediately visible to HAFS agents.
- **Validation:** The IDE will run the `ContextEvaluator` to warn you if your code contradicts project goals stored in `memory/`.
