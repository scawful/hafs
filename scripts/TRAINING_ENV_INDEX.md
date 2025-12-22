# Training Environment Index (Template)

This file is a generic index for setting up a Windows GPU training environment.
Move host-specific steps into your plugin repo (for example:
`~/Code/hafs_scawful/docs/TRAINING_ENV_INDEX.md`).

Use `~/Code/hafs_scawful/scripts/publish_plugin_configs.sh` to sync those
host-specific docs to halext-server and the Windows GPU host.

## Suggested Sections

- Host specs (GPU, RAM, OS)
- SSH and mount configuration
- Ollama endpoints and models
- Unsloth/finetuning setup
- Validation checklist

See `scripts/TRAINING_SETUP_README.md` for the generic setup overview.
