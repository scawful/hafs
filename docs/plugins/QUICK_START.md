# Plugin Quick Start

This is a generic quick start for user plugins.

## 1. Create a Plugin Repo

```bash
mkdir -p ~/Code/my_hafs_plugin/{config,scripts,docs}
```

## 2. Install (Symlink)

```bash
mkdir -p ~/.config/hafs/plugins
ln -s ~/Code/my_hafs_plugin ~/.config/hafs/plugins/my_hafs_plugin
```

## 3. Add Optional Aliases

```bash
# ~/.zshrc
source ~/.config/hafs/plugins/my_hafs_plugin/aliases.sh
```

## 4. Put Machine-Specific Files in the Plugin

- Hostnames, IPs, and SSH configs
- Deployment scripts
- Local paths and mounts
- Private runbooks
- Single plugin config (`config.toml`) for env + sync

For a concrete example, see your personal plugin repo (for example:
`~/.config/hafs/plugins/my_hafs_plugin`).

Sync your plugin to halext-server and the Windows GPU host:

```bash
~/.config/hafs/plugins/my_hafs_plugin/scripts/publish_plugin_configs.sh
```

## Optional: Single-File Sync Configuration

If you set `HAFS_PLUGIN_CONFIG` (exported by your plugin `aliases.sh`),
`scripts/sync_training_to_windows.sh` will read from your plugin config
instead of `~/.config/hafs/sync.toml`.
