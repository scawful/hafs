# GPU Acceleration (Template)

This document is a generic guide for using a GPU host for training and
inference. Move host-specific details to your plugin repo (for example:
`~/Code/hafs_scawful/docs/training/GPU_ACCELERATION.md`).

Use `~/Code/hafs_scawful/scripts/publish_plugin_configs.sh` to sync those
host-specific docs to halext-server and the Windows GPU host.

## Topics to Cover

- GPU host name and IP
- Ollama endpoints and models
- SSH access and health checks
- Data sync strategy
- Monitoring and logs

See `docs/windows/WINDOWS_SETUP.md` for the base Windows setup.
