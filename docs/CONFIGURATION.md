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

## Project Catalog (hafs.toml)

Project discovery for background agents is configured in `hafs.toml` (or
`~/.config/hafs/config.toml`) using `projects` and `tool_profiles`.

```toml
[[projects]]
name = "halext-org"
path = "~/Code/halext-org"
kind = "backend"
tags = ["halext", "api"]
tooling_profile = "read_only"

[[tool_profiles]]
name = "read_only"
allow = ["rg", "rg_files", "rg_todos", "git_status", "git_branch", "git_log", "git_diff", "ls"]
```
