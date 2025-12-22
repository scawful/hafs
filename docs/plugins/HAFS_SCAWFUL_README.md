# User Plugin Example

This document is intentionally generic. User-specific plugin docs and configs
should live in your plugin repository (for example: `~/Code/hafs_scawful`) and
be installed to `~/.config/hafs/plugins/<your_plugin>`.

Use `~/Code/hafs_scawful/scripts/publish_plugin_configs.sh` to sync your
host-specific docs/configs to halext-server and the Windows GPU host.

## What Goes In a User Plugin

- Machine-specific hostnames, IPs, and credentials
- Deployment scripts and workflow aliases
- Local paths and mount points
- Private docs and operational checklists

## Suggested Layout

```
<your_plugin_repo>/
├── config/
├── scripts/
├── docs/
├── aliases.sh
└── install.sh
```

## Install Pattern

```bash
# Example install (symlink or copy)
mkdir -p ~/.config/hafs/plugins
ln -s ~/Code/your_plugin ~/.config/hafs/plugins/your_plugin
```

See `docs/plugins/PLUGIN_DEVELOPMENT.md` for plugin protocol details.
