# Plugin Adapter Pattern for hafs

How to keep user-specific code out of the main hafs repository using plugin adapters.

## Problem

The hafs repository contains machine-specific code that shouldn't be version controlled:
- Hardcoded hostnames (medical-mechanica)
- Hardcoded paths (/Users/scawful/...)
- User-specific configurations
- Deployment scripts for specific machines

This makes the repo harder to share and adapt for other users.

## Solution: Plugin Adapters

Each user creates their own plugin adapter in `~/.config/hafs/plugins/<username>/`:

```
~/.config/hafs/plugins/my_hafs_plugin/
├── config.toml          # Machine-specific settings
├── aliases.sh           # User-specific aliases
├── scripts/             # User-specific scripts
│   ├── deploy.sh
│   └── backup.sh
└── README.md            # Documentation
```

## What Goes in Plugins vs. Repo

### ✅ Keep in Main Repo (Generic)

**Configuration Templates:**
- `config/lsp.toml.example` - Example LSP config
- `config/sync.toml.example` - Example sync config
- `config/training.toml.example` - Example training config

**Generic Scripts:**
- `scripts/presubmit_training.sh` - Reads from `~/.config/hafs/sync.toml`
- `scripts/sync_training_to_windows.sh` - Uses config file, no hardcoded paths
- `scripts/create_fine_tuned_model.sh` - Generic, no machine-specific code

**Core Infrastructure:**
- All Python code in `src/`
- Generic documentation in `docs/`
- Test files in `tests/`

### ❌ Move to Plugin (User-Specific)

**Deployment Scripts with Hardcoded Hosts:**
- ~~`scripts/deploy_models_windows.sh`~~ → `~/.config/hafs/plugins/my_hafs_plugin/scripts/deploy_models.sh`
- ~~`scripts/deploy_training_medical_mechanica.sh`~~ → Plugin
- ~~`scripts/remote_install_training.sh`~~ → Plugin

**Machine-Specific Configs:**
- ~~`config/training_medical_mechanica.toml`~~ → `~/.config/hafs/plugins/my_hafs_plugin/config/training.toml`
- ~~`config/windows_background_agents.toml`~~ → Plugin
- ~~`config/windows_filesystem_agents.toml`~~ → Plugin
- ~~`config/models.toml`~~ → Plugin (if contains user paths)

**User-Specific Settings:**
- `~/.config/hafs/sync.toml` - Already user-specific
- `~/.config/hafs/plugins/my_hafs_plugin/config.toml` - User plugin config

## Creating a Plugin Adapter

### 1. Create Plugin Structure

```bash
mkdir -p ~/.config/hafs/plugins/hafs_$USER/{scripts,config}

cat > ~/.config/hafs/plugins/hafs_$USER/config.toml << 'EOF'
[plugin]
name = "hafs_$USER"
version = "1.0.0"

[machines]
dev_machine = "localhost"
gpu_server = "your-gpu-host"

[paths]
code = "/home/$USER/hafs"
context = "/home/$USER/.context"
EOF
```

### 2. Create Aliases

```bash
cat > ~/.config/hafs/plugins/hafs_$USER/aliases.sh << 'EOF'
#!/bin/bash
# User-specific hafs aliases

export HAFS_ROOT="$HOME/hafs"
alias hafs-dev="cd $HAFS_ROOT && source .venv/bin/activate"
EOF

chmod +x ~/.config/hafs/plugins/hafs_$USER/aliases.sh
```

### 3. Source in Shell

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# hafs plugin
if [ -f ~/.config/hafs/plugins/hafs_$USER/aliases.sh ]; then
    source ~/.config/hafs/plugins/hafs_$USER/aliases.sh
fi
```

## Migration Guide

### Step 1: Identify Machine-Specific Code

```bash
# Find files with machine-specific references
cd ~/Code/hafs
grep -r "your-hostname\|/home/youruser" scripts/ config/

# Find deployment scripts
ls scripts/ | grep -E "(deploy|setup|install)"
```

### Step 2: Move to Plugin

```bash
# Create plugin directory
mkdir -p ~/.config/hafs/plugins/hafs_$USER/scripts

# Move deployment scripts
mv scripts/deploy_*.sh ~/.config/hafs/plugins/hafs_$USER/scripts/

# Move machine-specific configs
mv config/*_$(hostname)_*.toml ~/.config/hafs/plugins/hafs_$USER/config/
```

### Step 3: Create Generic Versions

For scripts that need to stay in the repo, make them generic:

**Before (hardcoded):**
```bash
WINDOWS_HOST="medical-mechanica"
TRAINING_DIR="D:/hafs_training"
```

**After (reads from config):**
```bash
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/hafs/sync.toml"
WINDOWS_HOST=$(grep '^host' "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' "')
TRAINING_DIR=$(grep '^path' "$CONFIG_FILE" | cut -d'=' -f2 | tr -d ' "')
```

### Step 4: Add to .gitignore

```bash
cd ~/Code/hafs
cat >> .gitignore << 'EOF'

# User-specific plugin configs
.config/hafs/
plugins/hafs_*/

# Machine-specific configs (use .example templates instead)
config/*_$(hostname)_*.toml
config/sync.toml
EOF
```

## Example: my_hafs_plugin Plugin

Example layout:

```
~/.config/hafs/plugins/my_hafs_plugin/
├── config.toml              # Machines, mounts, paths
├── aliases.sh               # 30+ workflow aliases
├── scripts/
│   ├── deploy_gpu.sh        # Deploy to medical-mechanica
│   ├── backup_context.sh    # Backup to mm-d mount
│   └── monitor_training.sh  # Check training status
└── README.md                # Personal notes
```

**Key aliases:**
- `hafs-commit-sync` - Full dev cycle
- `hafs-train-dev` - Interactive training workflow
- `hafs-windows-status` - Check GPU server
- `hafs-analyze-latest` - Analyze recent dataset

## Benefits

1. **Clean Repo** - No machine-specific code in version control
2. **Portable** - Other users can clone and use immediately
3. **Flexible** - Each user customizes their own workflow
4. **Maintainable** - Generic scripts work for everyone

## Plugin Discovery

hafs can auto-discover plugins:

```python
# src/hafs/plugins/loader.py
def load_user_plugin():
    plugin_dir = Path.home() / ".config/hafs/plugins" / f"hafs_{os.getenv('USER')}"
    if plugin_dir.exists():
        config = toml.load(plugin_dir / "config.toml")
        return config
    return None
```

Users can then reference plugin settings:

```python
from hafs.plugins import get_user_config

config = get_user_config()
gpu_server = config['machines']['gpu_server']
```

## Templates for New Users

Provide example templates in the repo:

```
docs/plugins/examples/
├── hafs_example/
│   ├── config.toml.example
│   ├── aliases.sh.example
│   └── README.md
```

New users copy and customize:

```bash
cp -r docs/plugins/examples/hafs_example ~/.config/hafs/plugins/hafs_$USER
# Edit configs with your settings
```

## Integration with hafs CLI

```bash
# List available plugins
hafs plugin list

# Show plugin config
hafs plugin show my_hafs_plugin

# Validate plugin
hafs plugin validate my_hafs_plugin

# Create new plugin from template
hafs plugin init hafs_newuser
```

## Summary

**Main Repo:** Generic, portable, no machine-specific code
**User Plugin:** All machine-specific settings, scripts, aliases

This keeps hafs infrastructure clean while allowing deep customization per user.
