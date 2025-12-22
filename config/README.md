# Configuration Directory

This directory contains HAFS configuration files. Configurations are organized into:

1. **Global configs** - Shared defaults (committed to git)
2. **Machine-specific configs** - Personal/host-specific (gitignored, stored in hafs_scawful)

## Global Configurations (Committed)

These files are committed to the main repo and shared across all users:

- `agent_personalities.toml` - Agent personality and cognitive protocol settings
- `cognitive_protocol.toml` - Cognitive loop and metacognition configuration
- `lsp.toml` - Language Server Protocol settings
- `prompt_templates.toml` - Prompt templates for training data generation
- `prompts.toml` - System prompts for various agents
- `sync.toml.example` - Example cross-platform sync configuration

## Machine-Specific Configurations (Not Committed)

These files contain personal paths, hostnames, and credentials. They are:
- **Gitignored** - Not tracked in the main hafs repo
- **Plugin-managed** - Stored in `hafs_scawful` plugin repo
- **Template-based** - `.example` files provided as templates

### Example Files (Templates)

Copy these to your `hafs_scawful/config/` directory and customize:

- `training_medical_mechanica.toml.example` → `training_medical_mechanica.toml`
  - Windows GPU training host configuration
  - Dataset paths, training settings, hardware config

- `windows_background_agents.toml.example` → `windows_background_agents.toml`
  - Background agents for Windows exploration and cataloging
  - Explorer, cataloger, context builder configs

- `windows_filesystem_agents.toml.example` → `windows_filesystem_agents.toml`
  - File sync monitoring between Mac and Windows
  - Change detection and conflict resolution

- `website_monitoring_agents.toml.example` → `website_monitoring_agents.toml`
  - Website health monitoring and content indexing
  - HTTP checks, SSL verification, uptime tracking

### Setup Instructions

1. **Create plugin directory:**
   ```bash
   mkdir -p ~/.config/hafs/plugins
   ln -s ~/Code/hafs_scawful ~/.config/hafs/plugins/hafs_scawful
   ```

2. **Copy example configs:**
   ```bash
   cd ~/Code/hafs_scawful/config
   cp ~/Code/hafs/config/training_medical_mechanica.toml.example training_medical_mechanica.toml
   cp ~/Code/hafs/config/windows_background_agents.toml.example windows_background_agents.toml
   # ... customize with your paths, hostnames, credentials
   ```

3. **Update paths:**
   - Replace `YOUR_USERNAME` with your actual username
   - Update drive letters (D:/, C:/) for your system
   - Set correct hostnames and IP addresses
   - Configure alert emails and credentials

4. **Sync to remote hosts:**
   ```bash
   ~/Code/hafs_scawful/scripts/publish_plugin_configs.sh
   ```

## Configuration Priority

When loading configs, HAFS checks locations in this order:

1. `hafs_scawful/config/` (plugin configs, highest priority)
2. `~/.config/hafs/` (user configs)
3. `~/Code/hafs/config/` (global defaults, lowest priority)

This allows plugin configs to override global defaults while maintaining shared base configurations.

## Best Practices

- ✅ **DO** commit changes to global configs (agent_personalities, cognitive_protocol, etc.)
- ✅ **DO** keep machine-specific paths and credentials in hafs_scawful
- ✅ **DO** update `.example` files when adding new config options
- ❌ **DON'T** commit files with hostnames, IPs, usernames, or passwords
- ❌ **DON'T** commit training paths specific to your machine
- ❌ **DON'T** commit website URLs or email addresses

## Gitignore Rules

The following patterns are gitignored to protect your privacy:

```gitignore
# Machine-specific configs
config/training_medical_mechanica.toml
config/windows_background_agents.toml
config/windows_filesystem_agents.toml
config/website_monitoring_agents.toml
config/models.toml

# User sync config
.config/hafs/sync.toml

# Plugin directories
.config/hafs/plugins/hafs_*/
```

## Questions?

See `docs/plugins/HAFS_SCAWFUL_README.md` for more information about the plugin system.
