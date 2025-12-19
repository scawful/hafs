# Configuration Guide

HAFS loads configuration from a layered TOML setup:

1. `./hafs.toml` (project-local, highest precedence)
2. `~/.config/hafs/config.toml` (user)
3. `~/.context/hafs_config.toml` (legacy fallback)

## File Location

If a file does not exist, HAFS will use safe defaults (or fail gracefully for critical keys like API tokens).

## Structure

### Current Config (hafs.toml or ~/.config/hafs/config.toml)

```toml
[general]
context_root = "~/.context"
agent_workspaces_dir = "~/AgentWorkspaces"
refresh_interval = 5
default_editor = "nvim"

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

## Project Catalog (hafs.toml)

Project discovery for background agents is configured in `hafs.toml` (or
`~/.config/hafs/config.toml`) using `projects` and `tool_profiles`. The embedding
indexer also uses this catalog; you can optionally add `knowledge_roots` to
target specific subdirectories for indexing.

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
