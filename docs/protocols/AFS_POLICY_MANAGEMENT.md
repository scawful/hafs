# AFS Policy Management Feature

## Overview

This feature adds the ability to modify AFS (Agentic File System) project configuration policies directly from the HAFS TUI. Users can now manage directory permissions (read_only, writable, executable) and persist changes to both global configuration and project-specific metadata.

## Implementation Details

### Files Created/Modified

1. **`src/hafs/config/saver.py`** (NEW)
   - Provides functions to save configuration to TOML files
   - Supports saving AFS policies to project metadata.json
   - Key functions:
     - `save_config(config, path)`: Save entire config to TOML file
     - `save_afs_policies(config, context_path)`: Save policies to project metadata.json

2. **`src/hafs/ui/screens/permissions_modal.py`** (MODIFIED)
   - Enhanced to support saving changes to multiple destinations
   - Added checkboxes for save destination selection:
     - Global config (`~/.config/hafs/config.toml`)
     - Project metadata (`.context/metadata.json`)
   - Automatically detects if a project has `.context` directory
   - Provides user feedback via notifications

3. **`src/hafs/ui/screens/settings.py`** (MODIFIED)
   - Added "Edit Policies" action (press `p`)
   - Displays current AFS policies with visual indicators
   - Opens PermissionsModal for editing
   - Refreshes display after policy updates

4. **`src/hafs/ui/screens/orchestrator.py`** (MODIFIED)
   - Updated to pass `context_path` to PermissionsModal
   - Enables project-specific policy editing from orchestrator screen

5. **`pyproject.toml`** (MODIFIED)
   - Added `tomli-w>=1.0.0` dependency for TOML writing support

## Usage

### From Settings Screen

1. Navigate to Settings screen (press `3` from main menu)
2. Scroll to "AFS DIRECTORY POLICIES" section
3. Press `p` to edit policies
4. Select a directory and press Enter to cycle through policies:
   - read_only → writable → executable → read_only
5. Choose save destinations:
   - ✓ Global config (affects all projects)
   - ✓ Project metadata (affects current project only)
6. Press "Save" to apply changes

### From Orchestrator Screen

1. In the orchestrator screen, press `Ctrl+P`
2. Follow the same editing process as above

## Policy Types

According to AFS philosophy:

- **read_only**: Long-term storage (memory, knowledge, history)
  - AI can read but not modify
  - Suitable for reference materials and archived content

- **writable**: Transient storage (scratchpad)
  - AI can both read and write
  - Suitable for working memory and temporary files

- **executable**: Executable scripts (tools)
  - AI can execute scripts and binaries
  - Suitable for automation tools

## Configuration Schema

### Global Config (TOML)

```toml
[[afs_directories]]
name = "memory"
policy = "read_only"
description = "Long-term docs and specs"

[[afs_directories]]
name = "scratchpad"
policy = "writable"
description = "AI reasoning space"

[[afs_directories]]
name = "tools"
policy = "executable"
description = "Executable scripts"
```

### Project Metadata (JSON)

```json
{
  "created_at": "2025-01-01T00:00:00",
  "agents": [],
  "description": "Project description",
  "policy": {
    "read_only": ["knowledge", "memory", "history"],
    "writable": ["scratchpad"],
    "executable": ["tools"]
  }
}
```

## Benefits

1. **Flexibility**: Different projects can have different policies
2. **Security**: Prevents accidental modifications to important files
3. **Convenience**: No need to manually edit config files
4. **Visual Feedback**: Color-coded policy indicators (blue=read_only, green=writable, red=executable) and a dashboard summary widget
5. **Dual Persistence**: Save to global config and/or project-specific metadata

## Technical Notes

- The PermissionsModal works on copies of the config to avoid accidental mutations
- Changes are only persisted when "Save" is clicked
- Project metadata checkbox is automatically disabled if no `.context` directory exists
- The modal validates the existence of `metadata.json` before attempting to save
- All config operations use Pydantic models for type safety
- TOML serialization handles Path objects automatically

## Future Enhancements

Possible improvements:
- Add ability to create custom directory types
- Support for fine-grained permissions (read/write/execute per file)
- Policy templates for common project types
- Validation warnings when changing critical directories
- Undo/redo for policy changes
- Policy history tracking
- Export policy summaries directly from the dashboard
