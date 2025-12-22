# Training Setup (Template)

Generic guidance for setting up a Windows GPU training environment.
Keep host-specific values (hostnames, IPs, mount paths) in your plugin repo
(for example: `~/Code/hafs_scawful/docs/TRAINING_SETUP_README.md`).

Use `~/Code/hafs_scawful/scripts/publish_plugin_configs.sh` to sync those
host-specific docs to halext-server and the Windows GPU host.

## Overview

- Install Python 3.11+
- Create `C:\hafs` and `D:\hafs_training`
- Create a `.venv` and install dependencies
- Configure Ollama (local or remote)
- Validate GPU access (CUDA + nvidia-smi)

## Next Steps

- Copy `config/training_medical_mechanica.toml` into your plugin and customize
- Run `scripts/remote_install_training.sh` with your host
- Test with `scripts/test_training_setup.py`
