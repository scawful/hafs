# Configuration Guide

HAFS uses a central TOML configuration file located at `~/.context/hafs_config.toml`.

## File Location

If the file does not exist, HAFS will use safe defaults (or fail gracefully for critical keys like API tokens).

## Structure

```toml
[core]
# Where HAFS stores its brain (memory, knowledge, scratchpad)
context_root = "~/.context"

# Where HAFS manages its "fig" or git workspaces
agent_workspaces = "~/AgentWorkspaces"

[llm]
# Your Generative AI API Key
aistudio_api_key = "AIza..."

# Override default models if you have access to newer previews
default_fast_model = "gemini-3-flash-preview"
default_reasoning_model = "gemini-3-pro-preview"

[user_preferences]
# Your username (used for templating paths)
username = "your_username"

# --- Plugin Specific Configs ---
# Plugins can define their own configuration sections here.
# For example, an issue tracker plugin might use:
# [issue_tracker]
# api_url = "https://api.example.com"
# project_id = "my-project"
```

## Environment Variables

You can override the API key using an environment variable, which is safer for CI/CD environments.

```bash
export AISTUDIO_API_KEY="your_key"
```
