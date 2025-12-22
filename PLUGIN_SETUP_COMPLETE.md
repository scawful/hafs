# Plugin Setup Checklist

This file is a generic checklist. User-specific plugin setup notes should live
in your personal plugin repository (for example: `~/.config/hafs/plugins/my_hafs_plugin`).

If your plugin includes a publish script, run:
`~/.config/hafs/plugins/my_hafs_plugin/scripts/publish_plugin_configs.sh` to sync
notes/configs to halext-server and the Windows GPU host.

## Checklist

1. Create a plugin repo with `config/`, `scripts/`, and `docs/`.
2. Add `aliases.sh` for workflow shortcuts.
3. Symlink into `~/.config/hafs/plugins/<plugin_name>`.
4. Add plugin paths to `~/.config/hafs/config.toml` if needed.
5. Keep machine-specific files out of the main repo.

See `docs/plugins/PLUGIN_DEVELOPMENT.md` and `docs/plugins/PLUGIN_ADAPTER_GUIDE.md`.
