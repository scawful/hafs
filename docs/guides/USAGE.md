# HAFS (Agentic File System) Usage Guide

HAFS is a protocol and toolset for managing AI agent context. It ensures agents have access to the right "memory" and "knowledge" without overwhelming their token window.

## CLI Usage

### Initialize a project
Sets up the `.context` structure in the current directory.
```bash
hafs afs init
```

### Mount a Context
Injects specific files into the agent's working memory.
```bash
hafs afs mount <type> <source_path>
```
*Types:* `memory`, `knowledge`, `tools`, `scratchpad`, `history`.

### Clean Context
Removes stale or transient files from the scratchpad and history.
```bash
hafs afs clean
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
hafs orchestrate run "Investigate ALTTP routine X"
hafs orchestrate run "Daily infra health" --mode swarm
```

### Agent Chat
Start an interactive coordinator session.
```bash
hafs chat
```

### Agent Memory
Store and recall per-agent memories.
```bash
hafs memory status
hafs memory recall --agent Generalist "what did we decide?"
hafs memory remember --agent Generalist "Log key decision"
hafs memory cross-search "decision"
```

### Context Engineering Pipeline
Manage the context store and construct/evaluate task windows.
```bash
hafs context status
hafs context list --type fact
hafs context write "..." --type fact
hafs context search "query"
hafs context construct "task"
hafs context evaluate "task"
hafs context types
hafs context prune --dry-run
hafs context deep-dive --root .
hafs context ml-plan
```

### Protocol Lint
Validate cognitive protocol artifacts for missing files and schema issues.
```bash
hafs protocol lint
hafs protocol lint --path /path/to/project
hafs protocol lint --strict
```

### Knowledge Indexing
Index configured projects for embeddings (uses `projects` from the local config or `hafs.toml`).
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
hafs nodes models medical-mechanica
hafs nodes models medical-mechanica --details
hafs nodes pull medical-mechanica qwen3:14b
hafs nodes chat medical-mechanica --model qwen3:14b
hafs nodes probe medical-mechanica --model gemma3:12b --prompt "hello"
hafs nodes probe-suite --suite smoke --model qwen3:14b
hafs nodes probe-suite --suite tool-call --model qwen3:14b
```
Nodes load from `~/.config/hafs/nodes.toml` (or `~/.hafs/nodes.toml`). Use `nodes status`
to trigger a health check and see latency.

### AFS Sync
Run sync profiles defined in `sync.toml`.
```bash
hafs sync list
hafs sync show global
hafs sync run global --dry-run
```
Sync results are recorded to `~/.context/metrics/afs_sync_status.json` and summarized
in the Infrastructure UI panels.

### Services
Manage background daemons and dashboards.
```bash
hafs services list
hafs services start autonomy
hafs services start embedding
hafs services start context
hafs services start observability
```
Aliases map to the canonical service names: `autonomy-daemon`, `embedding-daemon`,
`context-agent-daemon`, and `observability-daemon`.

### Auth (Claude Max / Claude Code)
Use Claude CLI OAuth/token setup for the `claude` backend.
```bash
hafs auth claude
```
This runs `claude setup-token` using the configured Claude CLI command.
Run it from an interactive terminal (it needs a TTY).

To configure Claude CLI permissions/sandboxing for HAFS agents, set CLI flags
in your local `~/.config/hafs/config.toml` backend entry. Example:
```toml
[[backends]]
name = "claude"
enabled = true
# Use Claude Code with explicit permission mode.
command = ["claude", "--permission-mode", "default"]
```

For strict no-tool one-shot calls, configure `claude_oneshot`:
```toml
[[backends]]
name = "claude_oneshot"
enabled = true
command = ["claude", "--permission-mode", "plan", "--tools", ""]
```

### Observability + Scheduling
Use the context agent daemon to run scheduled syncs and reports.
```bash
python -m hafs.services.context_agent_daemon --status
python -m hafs.services.context_agent_daemon --install
```
To schedule sync runs, edit `~/.context/context_agent_daemon/scheduled_tasks.json`.

The observability daemon monitors endpoints, nodes, and sync status. It can
optionally perform allowlisted remediations from the `observability.remediation`
config.
```bash
python -m hafs.services.observability_daemon --status
python -m hafs.services.observability_daemon --once
```

If you need the daemon to run under a specific virtualenv, set `HAFS_PYTHON` or
`general.python_executable` in your config so launchd/systemd uses the right
interpreter.

### TUI: Infrastructure Panel
Run `hafs`, then go to the **Status** tab to view Nodes and AFS Sync status.

### Web Hub: Infrastructure
The Streamlit web dashboard exposes an **Infrastructure** page with node + sync
tables (run via the dashboard service).

## TUI Interaction
Launch the interactive explorer:
```bash
hafs
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
