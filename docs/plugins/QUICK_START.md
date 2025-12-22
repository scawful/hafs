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

For a concrete example, see your personal plugin repo (for example:
`~/Code/hafs_scawful`).

Sync your plugin to halext-server and the Windows GPU host:

```bash
~/Code/hafs_scawful/scripts/publish_plugin_configs.sh
```
