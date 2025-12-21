# Configuration Guide

HAFS loads configuration from a layered TOML setup:

1. `~/.config/hafs/config.toml` (local overrides, auto-created)
2. `./hafs.toml` (project defaults)
3. `~/.context/hafs_config.toml` (legacy fallback)

Set `HAFS_PREFER_REPO_CONFIG=1` to prefer `hafs.toml` over the local config.

## File Location

If a file does not exist, HAFS will use safe defaults (or fail gracefully for critical keys like API tokens).
On first run, HAFS creates `~/.config/hafs/config.toml` and seeds it with
auto-discovered AFS projects so you can override repo defaults without
editing `hafs.toml`.

## Structure

### Current Config (preferred: ~/.config/hafs/config.toml)

```toml
[general]
context_root = "~/.context"
agent_workspaces_dir = "~/AgentWorkspaces"
refresh_interval = 5
default_editor = "nvim"
python_executable = "~/Code/hafs/.venv/bin/python"

[plugins]
enabled_plugins = ["hafs_plugin_github"]
plugin_dirs = ["~/Code/hafs_plugins"]

[[execution_modes]]
name = "read_only"
tool_profile = "read_only"

[[execution_modes]]
name = "build_only"
tool_profile = "build_only"

[[execution_modes]]
name = "infra_ops"
tool_profile = "infra_ops"

default_execution_mode = "read_only"

# Optional skills/personas for agent prompting and tool profile selection.
[[skills]]
name = "planning"
description = "Break down tasks into steps and constraints."
goals = ["Clarify scope", "Identify dependencies", "Define checkpoints"]

[[personas]]
name = "Coder"
role = "coder"
skills = ["coding"]
execution_mode = "build_only"
default_for_role = true

# --- Plugin Specific Configs ---
# Plugins can define their own configuration sections here.
# For example, an issue tracker plugin might use:
# [issue_tracker]
# api_url = "https://api.example.com"
# project_id = "my-project"
```

### Legacy Config (~/.context/hafs_config.toml)

```toml
[core]
context_root = "~/.context"
agent_workspaces = "~/AgentWorkspaces"
plugins = ["hafs_plugin_github"]

[llm]
aistudio_api_key = "AIza..."
```

## Environment Variables

You can override the API key using an environment variable, which is safer for CI/CD environments.

```bash
export AISTUDIO_API_KEY="your_key"
```

Set the active execution mode (overrides config):

```bash
export HAFS_EXEC_MODE="build_only"
```

## Provider Configuration (models.toml)

HAFS routes model requests using `~/.config/hafs/models.toml`. Each provider
entry declares which environment variable holds its API key, plus the model
IDs per task tier.

```toml
[providers.gemini]
enabled = true
api_key_env = "GEMINI_API_KEY"

  [providers.gemini.models]
  fast = "gemini-3-flash"
  coding = "gemini-3-flash"
  reasoning = "gemini-3-pro"

[providers.openai]
enabled = true
api_key_env = "OPENAI_API_KEY"

  [providers.openai.models]
  fast = "gpt-4o-mini"
  coding = "gpt-5.2-mini"
  reasoning = "o3"

[providers.anthropic]
enabled = true
api_key_env = "ANTHROPIC_API_KEY"

  [providers.anthropic.models]
  fast = "claude-3-haiku"
  coding = "claude-sonnet-4-5"
  reasoning = "claude-opus-4-5"

[providers.ollama]
enabled = true

  [providers.ollama.models]
  fast = "qwen2.5:7b"
  coding = "qwen2.5-coder:14b"
  reasoning = "deepseek-r1:8b"
```

Export your keys (or whatever you used in `api_key_env`):

```bash
export GEMINI_API_KEY="..."
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

For llama.cpp, configure `[llamacpp]` in your local config and set
`LLAMACPP_API_KEY` if your server requires one.

## Optional Plugin Hooks

Some UI actions can be wired to plugin commands via environment variables:

```bash
export HAFS_GARDENER_COMMAND="hafs-garden"
export HAFS_AGENT_MANAGER="my_plugin.agent_manager:AgentManager"
export HAFS_AGENT_MANAGER_PATH="~/Code/hafs-plugins/src"
```

### Python for Background Services

Background agents and daemons resolve the Python interpreter in this order:
1. `HAFS_PYTHON` environment variable
2. `general.python_executable` from config
3. Active virtualenv (`VIRTUAL_ENV`)
4. `.venv/bin/python` in the repo or current directory
5. `sys.executable`

To force a specific venv:

```bash
export HAFS_PYTHON="$HOME/Code/hafs/.venv/bin/python"
```

## Project Catalog (local config or hafs.toml)

Project discovery for background agents is configured in the local config
(`~/.config/hafs/config.toml`) or `hafs.toml` using `projects` and `tool_profiles`.
The embedding
indexer also uses this catalog; you can optionally add `knowledge_roots` to
target specific subdirectories for indexing.

HAFS seeds the local config with auto-discovered AFS projects on first run,
so most users only need to curate or rename entries.

```toml
[[projects]]
name = "halext-org"
path = "~/Code/halext-org"
kind = "backend"
tags = ["halext", "api"]
tooling_profile = "read_only"
knowledge_roots = ["docs", "src"]

[[tool_profiles]]
name = "read_only"
allow = ["rg", "rg_files", "rg_todos", "git_status", "git_branch", "git_log", "git_diff", "ls"]

[[tool_profiles]]
name = "infra_ops"
allow = [
  "rg", "rg_files", "rg_todos", "git_status", "git_branch", "git_log", "git_diff", "ls",
  "uname", "whoami", "uptime", "df", "du", "ps", "lsof", "tail", "journalctl", "log_show",
  "launchctl", "systemctl", "docker", "docker_compose", "kubectl", "ssh", "scp", "rsync",
  "curl", "ping",
]
```

## Model Routing and Registry (MoE)

HAFS reserves two optional files under `~/.context/models/` for expert routing
and model metadata. These are planning artifacts today; they lock naming and
routing intent ahead of training.

- `~/.context/models/routing.toml` (template: `docs/config/routing.toml`)
- `~/.context/models/registry.toml` (template: `docs/config/model_registry.toml`)

Optional overrides (useful for CI or experiments):
- `HAFS_MODEL_ROUTING_PATH`
- `HAFS_MODEL_REGISTRY_PATH`

## Node Registry (nodes.toml)

Distributed nodes are loaded from `~/.config/hafs/nodes.toml` (or `~/.hafs/nodes.toml`).
Use this to model compute nodes, servers, and mobile devices for Phase 5 autonomy.

```toml
[[nodes]]
name = "halext-server"
host = "100.100.100.10"
port = 11434
node_type = "compute"
platform = "linux"
capabilities = ["ollama", "afs", "tailscale"]
health_url = "https://halext.org/api/health"
afs_root = "/srv/afs"
sync_profiles = ["global", "halext-web"]
tags = ["server"]

[[nodes]]
name = "medical-mechanica"
host = "100.100.100.20"
node_type = "compute"
platform = "windows"
capabilities = ["ollama", "gpu"]
tags = ["workstation"]

[[nodes]]
name = "ios-cafe"
host = "ios.local"
node_type = "mobile"
platform = "ios"
capabilities = ["client"]
health_url = "https://ios.local/health"
tags = ["mobile"]
```

## Observability and Remediation

The observability daemon reads `observability` settings from your main config.
If `observability.endpoints` is empty, the built-in defaults are used.
Remediation actions are opt-in and controlled by allowlists.

```toml
[observability]
enabled = true
check_interval_seconds = 120
monitor_endpoints = true
monitor_nodes = true
monitor_local_services = true
monitor_sync = true
monitor_services = true

  [[observability.endpoints]]
  name = "halext-org"
  url = "https://halext.org/api/health"
  type = "api"
  enabled = true

[observability.remediation]
enabled = true
allowed_actions = ["restart_service", "start_service", "run_afs_sync", "context_burst"]
allowed_services = ["embedding-daemon", "context-agent-daemon", "observability-daemon"]
allowed_sync_profiles = ["global", "halext-web"]
max_actions_per_run = 2
cooldown_minutes = 30
trigger_context_burst_on_alerts = ["error", "critical"]
context_burst_force = false
```

## AFS Sync (sync.toml)

Sync profiles live in `~/.config/hafs/sync.toml` (or `~/.hafs/sync.toml`).

```toml
[[profiles]]
name = "global"
scope = "global"
direction = "bidirectional"
transport = "rsync"
source = "~/.context/global/shared"
exclude = [".DS_Store", "*.tmp"]
delete = false

  [[profiles.targets]]
  node = "halext-server"
  path = "/srv/afs/global/shared"
  user = "deploy"
  port = 22

[[profiles]]
name = "halext-web"
scope = "project"
direction = "push"
source = "~/Code/halext-org/.context"

  [[profiles.targets]]
  node = "halext-server"
  path = "/srv/afs/halext-org/.context"
```

## Context Agent Daemon Schedule

The daemon reads `~/.context/context_agent_daemon/scheduled_tasks.json`. Example:

```json
[
  {
    "name": "nightly_afs_sync",
    "task_type": "afs_sync",
    "interval_hours": 24,
    "enabled": true,
    "config": {
      "profiles": ["global", "halext-web"],
      "direction": "push",
      "dry_run": false
    }
  }
]
```
