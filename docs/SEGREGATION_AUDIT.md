# hafs vs hafs_scawful Segregation Audit

**Date:** 2025-12-22
**Status:** ✅ PROPERLY SEGREGATED

## Summary

The codebase is **correctly segregated** between:
- **hafs** (public repository) - Generic templates and infrastructure
- **hafs_scawful** (user-specific plugin) - Personal configurations and data

## What Belongs Where

### hafs (Public Repository)

**Purpose:** Generic infrastructure that any user can use

**Contents:**
- Core application code (`src/`)
- Template configuration files (`config/*_agents.toml`, `config/training.toml`)
- Generic scripts with placeholders (`scripts/*.sh`, `scripts/*.ps1`)
- Documentation (`docs/`)
- Tests (`tests/`)

**Key Principle:** No hardcoded user-specific:
- Usernames (no "scawful", "Administrator" unless in TEMPLATE comment)
- Paths (no `/Users/scawful`, `C:/Users/Administrator` unless example)
- Hostnames (no "medical-mechanica", "GPU_HOST" unless placeholder)
- URLs (no halext.org, alttphacking.net unless example)
- Credentials

### hafs_scawful (User Plugin)

**Purpose:** User-specific customizations and configurations

**Contents:**
- Actual deployment configs (`config/*.toml`)
- User-specific resource paths (`config/training_resources.toml`)
- Personal data and scripts
- User-specific documentation

**Key Principle:** All actual:
- User credentials
- Server hostnames
- Website URLs
- Resource paths
- Personal preferences

## Current Files Analysis

### ✅ Correctly Placed in hafs

| File | Type | Notes |
|------|------|-------|
| `config/website_monitoring_agents.toml` | TEMPLATE | Has comment directing users to copy to plugin |
| `config/windows_background_agents.toml` | TEMPLATE | Generic `C:/hafs` paths, `Administrator` placeholder |
| `config/windows_filesystem_agents.toml` | TEMPLATE | Generic paths |
| `config/training_medical_mechanica.toml` | TEMPLATE | Generic `D:/hafs_training` paths |
| `config/training.toml` | CONFIG | Hardware profiles, expert definitions - generic |
| `scripts/deploy_training_*.sh` | TEMPLATE | Uses `GPU_HOST`, `Administrator` placeholders |
| `scripts/sync_training_to_windows.sh` | GENERIC | Reads config from `~/.config/hafs/sync.toml` |
| `src/agents/training/resource_discovery.py` | CODE | Loads paths from plugin config, no hardcoded paths |

### ✅ Correctly Placed in hafs_scawful

| File | Type | Notes |
|------|------|-------|
| `config/website_monitoring_agents.toml` | CONFIG | Actual URLs: halext.org, alttphacking.net, zeniea.com |
| `config/training_resources.toml` | CONFIG | Actual resource paths from user's filesystem |
| `config/windows_background_agents.toml` | CONFIG | Customized for medical-mechanica |
| `config/training.toml` | CONFIG | Medical-mechanica specific settings |

## Configuration Loading Pattern

The correct pattern used throughout hafs:

```python
# 1. Try plugin config first
plugin_config = Path.home() / "Code" / "hafs_scawful" / "config" / "training_resources.toml"
if plugin_config.exists():
    config = load_config(plugin_config)
else:
    # 2. Fall back to user config
    user_config = Path.home() / ".config" / "hafs" / "config.toml"
    if user_config.exists():
        config = load_config(user_config)
    else:
        # 3. Use defaults from hafs config
        default_config = hafs_root / "config" / "defaults.toml"
        config = load_config(default_config)
```

## Path Resolution Pattern

All scripts should use this pattern for cross-platform paths:

```bash
# Read config file for user-specific paths
CONFIG_FILE="${XDG_CONFIG_HOME:-$HOME/.config}/hafs/sync.toml"

# Use placeholders in templates
REMOTE_HOST="GPU_HOST"  # User replaces in their config
REMOTE_USER="Administrator"  # User replaces in their config
```

## Scripts Pattern

Scripts in hafs should:
1. Have placeholders for user-specific values
2. Read from config files (`~/.config/hafs/*.toml`)
3. Include usage instructions in comments
4. Fail gracefully with helpful error messages if config missing

## No Hardcoded Paths Found

✅ Audit complete - no problematic hardcoded paths in hafs repo.

The only references found:
- Template comments showing copy commands: `cp config/foo.toml ~/Code/hafs_scawful/config/`
- Generic `~/Code/hafs/` references in hafs code (expected)
- Placeholder paths in template configs

## Recommendations

### For Future Development

1. **Always use config files for user-specific values**
   ```python
   # ✅ GOOD
   config = load_plugin_config("training_resources.toml")
   paths = config.get("resource_roots", [])

   # ❌ BAD
   paths = [Path.home() / "Code" / "zelda3"]
   ```

2. **Use XDG config directories**
   ```bash
   # ✅ GOOD
   CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/hafs/sync.toml"

   # ❌ BAD
   CONFIG="/Users/scawful/.config/hafs/sync.toml"
   ```

3. **Template files should include migration instructions**
   ```toml
   # Template: Copy this to ~/Code/hafs_scawful/config/ and customize
   [example]
   value = "REPLACE_ME"
   ```

4. **Scripts should validate config**
   ```bash
   if [ ! -f "$CONFIG_FILE" ]; then
       echo "✗ Config file not found: $CONFIG_FILE"
       echo "Create it with: ..."
       exit 1
   fi
   ```

## Test Checklist

Before committing changes to hafs:

- [ ] No `/Users/scawful` references (except in docs as examples)
- [ ] No `~/Code/hafs_scawful` references (except in plugin loading code)
- [ ] No actual hostnames (only `GPU_HOST`, `REMOTE_HOST` placeholders)
- [ ] No actual URLs (only `example.com` placeholders)
- [ ] All user-specific values read from config files
- [ ] Config files are templates with placeholder values

## Conclusion

✅ **Segregation is correct.** No remediation needed.

The codebase follows best practices:
- hafs contains generic templates
- hafs_scawful contains actual user configurations
- Code loads user config from plugin directory
- Scripts read from `~/.config/hafs/*.toml`
