# Windows Models Setup (Template)

This document is a template for model setup on a Windows GPU host. Keep the
host-specific instructions in your plugin repo (for example:
`~/Code/hafs_scawful/docs/training/WINDOWS_MODELS_SETUP.md`).

Use `~/Code/hafs_scawful/scripts/publish_plugin_configs.sh` to sync those
host-specific docs to halext-server and the Windows GPU host.

## Suggested Contents

- SSH into GPU host
- Install Ollama and required models
- Validate model list
- Configure `config/training_medical_mechanica.toml` (template)
