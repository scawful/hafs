# Files to Migrate from Main Repo

Files that contain machine-specific code and should be moved to user plugins.

## Status: 🔴 In Main Repo (Should be in Plugin)

### Deployment Scripts

These scripts contain hardcoded hostnames and should be user-specific:

#### scripts/deploy_models_windows.sh
- **Issue:** Hardcodes `WINDOWS_HOST="medical-mechanica"`
- **Lines:** 6, 17-18, 34, 55, 87, 96
- **Action:** Move to `~/.config/hafs/plugins/hafs_scawful/scripts/deploy_models.sh`
- **Replace with:** Generic template that reads from plugin config

#### scripts/deploy_training_medical_mechanica.sh
- **Issue:** Hardcodes medical-mechanica hostname
- **Action:** Move to plugin
- **Replace with:** Generic deployment script using config

#### scripts/remote_install_training.sh
- **Issue:** Contains specific remote host configuration
- **Action:** Move to plugin

#### scripts/launch_hybrid_training.sh
- **Issue:** May contain specific paths/hosts
- **Action:** Review and either generalize or move to plugin

### Configuration Files

#### config/training_medical_mechanica.toml
- **Issue:** Filename includes specific machine name
- **Action:** Move to `~/.config/hafs/plugins/hafs_scawful/config/training.toml`
- **Replace with:** `config/training.toml.example` template

#### config/windows_background_agents.toml
- **Issue:** Windows-specific configuration
- **Action:** Move to plugin or make generic

#### config/windows_filesystem_agents.toml
- **Issue:** Windows-specific filesystem config
- **Action:** Move to plugin

#### config/models.toml
- **Issue:** May contain user-specific model paths
- **Action:** Check for hardcoded paths, move to plugin if present
- **Replace with:** Template

### Setup Scripts

#### scripts/setup_background_agents.ps1
- **Issue:** PowerShell script for specific Windows setup
- **Action:** Review - may be generic enough to keep

#### scripts/setup_windows.ps1
- **Issue:** Windows-specific setup
- **Action:** Keep as template, move configured version to plugin

#### scripts/setup_windows_models.ps1
- **Issue:** Windows model setup
- **Action:** Keep as template

## Status: 🟢 Already Abstracted

These files correctly use configuration:

### ✓ scripts/sync_training_to_windows.sh
- Reads from `~/.config/hafs/sync.toml`
- No hardcoded paths

### ✓ scripts/presubmit_training.sh
- Generic, works for any user
- Uses relative paths

### ✓ config/lsp.toml
- User-specific, created during setup
- Never committed to repo

### ✓ ~/.config/hafs/sync.toml
- User-specific sync config
- In .gitignore

## Status: 🟡 Needs Review

### scripts/setup_hafs_lsp.sh
- **Review:** Check for hardcoded paths
- **Action:** TBD after review

### scripts/test_training_setup.py
- **Review:** Check for machine-specific assumptions
- **Action:** Should be generic test

## Migration Plan

### Phase 1: Create Plugin Infrastructure ✅ DONE

1. ✅ Create `~/.config/hafs/plugins/hafs_scawful/`
2. ✅ Create `config.toml` with machine settings
3. ✅ Create `aliases.sh` with workflow shortcuts

### Phase 2: Move Deployment Scripts

```bash
# Create scripts directory
mkdir -p ~/.config/hafs/plugins/hafs_scawful/scripts

# Move deployment scripts
mv ~/Code/hafs/scripts/deploy_models_windows.sh \
   ~/.config/hafs/plugins/hafs_scawful/scripts/deploy_models.sh

mv ~/Code/hafs/scripts/deploy_training_medical_mechanica.sh \
   ~/.config/hafs/plugins/hafs_scawful/scripts/deploy_training.sh

mv ~/Code/hafs/scripts/remote_install_training.sh \
   ~/.config/hafs/plugins/hafs_scawful/scripts/remote_install.sh

# Update scripts to read from plugin config
```

### Phase 3: Move Config Files

```bash
# Create config directory
mkdir -p ~/.config/hafs/plugins/hafs_scawful/config

# Move machine-specific configs
mv ~/Code/hafs/config/training_medical_mechanica.toml \
   ~/.config/hafs/plugins/hafs_scawful/config/training.toml

mv ~/Code/hafs/config/windows_background_agents.toml \
   ~/.config/hafs/plugins/hafs_scawful/config/background_agents.toml

mv ~/Code/hafs/config/windows_filesystem_agents.toml \
   ~/.config/hafs/plugins/hafs_scawful/config/filesystem_agents.toml

# Check models.toml for hardcoded paths
# If present, move to plugin
```

### Phase 4: Create Templates

```bash
# Create example templates in repo
cd ~/Code/hafs

# Training config template
cp ~/.config/hafs/plugins/hafs_scawful/config/training.toml \
   config/training.toml.example

# Remove machine-specific values from example
sed -i 's/medical-mechanica/YOUR_GPU_HOST/g' config/training.toml.example
sed -i 's/\/Users\/scawful/\/path\/to\/home/g' config/training.toml.example

# Deployment script template
cat > scripts/deploy_training.sh.example << 'EOF'
#!/bin/bash
# Template deployment script
# Copy to ~/.config/hafs/plugins/hafs_$USER/scripts/deploy_training.sh

# Load plugin config
PLUGIN_CONFIG="$HOME/.config/hafs/plugins/hafs_$USER/config.toml"
GPU_HOST=$(grep 'gpu_server' "$PLUGIN_CONFIG" | cut -d'=' -f2 | tr -d ' "')

# Your deployment logic here
ssh "$GPU_HOST" "your commands"
EOF
```

### Phase 5: Update .gitignore

```bash
cd ~/Code/hafs
cat >> .gitignore << 'EOF'

# Machine-specific deployment scripts
scripts/deploy_*_medical_mechanica.sh
scripts/deploy_*_windows.sh
scripts/*_$HOSTNAME_*.sh

# Machine-specific configs
config/*_medical_mechanica.toml
config/windows_*.toml
config/models.toml

# Use .example templates instead
# Users should copy .example files and customize
EOF

git add .gitignore
git commit -m "docs: update gitignore for machine-specific files"
```

### Phase 6: Create Plugin README

```bash
cat > ~/.config/hafs/plugins/hafs_scawful/README.md << 'EOF'
# hafs_scawful Plugin

Personal hafs configuration for scawful's development environment.

## Machines

- **Mac Dev:** localhost (M1 MacBook)
- **Windows GPU:** medical-mechanica (100.104.53.21)
  - RTX 4090
  - 64GB RAM
  - Training server

## Mounts

- mm-c, mm-d, mm-e: SMB mounts to Windows machines
- training: D:/hafs_training
- context: D:/.context

## Aliases

See `aliases.sh` - 30+ workflow shortcuts:
- `hafs-commit-sync` - Full dev cycle
- `hafs-train-dev` - Interactive training
- `hafs-windows-status` - Check GPU server

## Scripts

- `deploy_models.sh` - Deploy models to Windows GPU
- `deploy_training.sh` - Deploy training code
- `backup_context.sh` - Backup context to mount

## Configuration

All settings in `config.toml` - loaded by plugin loader.
EOF
```

## Verification Checklist

After migration, verify:

- [ ] No hardcoded hostnames in main repo
- [ ] No hardcoded user paths (/Users/scawful) in main repo
- [ ] All deployment scripts read from config
- [ ] .example templates exist for new users
- [ ] .gitignore excludes machine-specific files
- [ ] Plugin loads correctly
- [ ] Aliases work
- [ ] Scripts in plugin use plugin config

## Commands After Migration

```bash
# In main repo - should find NOTHING
cd ~/Code/hafs
grep -r "medical-mechanica" scripts/ config/
grep -r "/Users/scawful" scripts/ config/

# Should return 0 results

# In plugin - should find your settings
grep -r "medical-mechanica" ~/.config/hafs/plugins/hafs_scawful/
# Should show your configs
```

## Benefits After Migration

1. **Clean repo** - Ready to share with other users
2. **Portable** - Works on any machine after plugin setup
3. **Maintainable** - Generic scripts benefit everyone
4. **Customizable** - Deep customization in plugin
5. **Private** - Machine-specific settings never committed

## Next Steps

1. Execute Phase 2 (move deployment scripts)
2. Execute Phase 3 (move config files)
3. Execute Phase 4 (create templates)
4. Execute Phase 5 (update .gitignore)
5. Commit changes to main repo
6. Test workflow with plugin
