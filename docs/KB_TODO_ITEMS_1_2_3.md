# Knowledge Base TODOs (Items 1-3)

## 1) Local project config overrides (not repo-default)
- Keep `hafs.toml` in-repo as a base template only.
- Add a per-user config layer under `~/.config/hafs/` (or `~/.hafs/`) for overrides.
- Ensure project-specific settings (nodes, embeddings, paths) are resolved from local config first.

## 2) Multi-embedding support with per-profile settings
- Allow multiple embedding providers/models (e.g., Gemini + OpenAI) in parallel.
- Store each embedding set in its own namespace (separate index directory) to avoid collisions.
- Add CLI flags or config keys to select active embedding profile per task.

## 3) CLI help / agent orchestration audit
- Audit `hafs --help` and subcommands to ensure background agents and orchestration features are discoverable.
- Document agent council/swarm controls and background worker operations in `docs/USAGE.md`.
- Add examples for launching agents, reindexing, summarization, and model probes.
