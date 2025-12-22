# Training Development Workflow

Standard workflow for working on training code that needs to run on Windows GPU server.

## Quick Reference

```bash
# Before committing
./scripts/presubmit_training.sh

# Commit and push
git add <files>
git commit -m "your message"
git push

# Sync to Windows
./scripts/sync_training_to_windows.sh
```

## Setup

### 1. Configure Sync Settings

Create `~/.config/hafs/sync.toml` (not in repo):

```toml
[windows]
# Mount path for Windows training server
mount = "/Users/scawful/Mounts/mm-d/hafs_training"

# SSH fallback (if mount unavailable)
host = "medical-mechanica"
path = "D:/hafs_training"
```

This file is user-specific and never committed to the repo.

## Development Workflow

### 1. Make Changes

Edit training code in `src/agents/training/`:
- `quality.py` - Quality scoring and filtering
- `curator.py` - Dataset management
- `generators/` - Sample generation
- `scripts/` - Training campaigns

### 2. Run Presubmit Checks

```bash
./scripts/presubmit_training.sh
```

This checks:
- [x] Python syntax
- [x] Imports work correctly
- [x] Type checking (if mypy installed)
- [x] Critical code quality (syntax errors, undefined names)
- [x] Unit tests (if they exist)
- [x] Integration checks (RejectedSample tracking, etc.)

**Note:** Presubmit allows style warnings (line length, unused imports) but fails on actual errors.

### 3. Commit and Push

```bash
git add src/agents/training/
git commit -m "feat: add rejected sample tracking"
git push
```

### 4. Sync to Windows

```bash
./scripts/sync_training_to_windows.sh
```

This syncs:
- Training code (`src/agents/training/`)
- Config files (`config/*training*.toml`)
- Scripts (`scripts/analyze_rejected_samples.py`, etc.)

The script:
- Uses mount if available (faster)
- Falls back to SSH if mount unavailable
- Verifies sync completed successfully

## What Gets Synced

| Source | Destination |
|--------|-------------|
| `~/Code/hafs/src/agents/training/` | `D:/hafs_training/src/agents/training/` |
| `~/Code/hafs/config/*training*.toml` | `D:/hafs_training/config/` |
| `~/Code/hafs/scripts/analyze_rejected_samples.py` | `D:/hafs_training/scripts/` |

## Presubmit Checks Explained

### 1. Python Syntax Check
Compiles all Python files to ensure no syntax errors.

### 2. Import Check
Tries to import main modules to ensure dependencies are correct.

**Non-blocking:** Import warnings are OK in venv (missing optional deps).

### 3. Type Checking (Optional)
Runs mypy if installed. Warnings are non-blocking.

### 4. Code Quality
Checks for:
- E9: Runtime syntax errors
- F821: Undefined names

**Ignored:**
- E501: Line too long
- F401: Unused imports
- F841: Unused variables

### 5. Unit Tests (If Present)
Runs pytest on `src/agents/training/tests/` if it exists.

### 6. Integration Checks
Verifies critical features:
- RejectedSample tracking in FilterStats
- Classes can be instantiated

## Troubleshooting

### "Cannot connect to Windows host"

```bash
# Check if mount is accessible
ls /Users/scawful/Mounts/mm-d/hafs_training

# Check if SSH works
ssh medical-mechanica "echo OK"

# Verify sync config
cat ~/.config/hafs/sync.toml
```

### "Presubmit failed"

```bash
# Run specific checks manually
PYTHONPATH=src .venv/bin/python -m py_compile src/agents/training/*.py
PYTHONPATH=src .venv/bin/python -c "from agents.training.quality import QualityScorer"
ruff check src/agents/training/ --select=E9,F821
```

### "Sync succeeded but files not updated on Windows"

SMB mount sometimes has caching issues. Force refresh:

```bash
# Unmount and remount
umount /Users/scawful/Mounts/mm-d
# Remount via Finder or mount command

# Or use SSH fallback
# Edit ~/.config/hafs/sync.toml to temporarily disable mount
```

## Best Practices

1. **Always run presubmit before committing**
   - Catches errors early
   - Ensures code works in venv

2. **Sync immediately after push**
   - Keeps Windows environment up to date
   - Avoids confusion about which version is deployed

3. **Use meaningful commit messages**
   - Makes it easier to track what changed
   - Helps when debugging training issues

4. **Test on Windows after sync**
   - SSH into Windows machine
   - Run a quick test to verify changes work

## Testing on Windows After Sync

```bash
# SSH to Windows
ssh medical-mechanica

# Navigate to training directory
cd D:/hafs_training

# Test imports
python -c "from agents.training.quality import RejectedSample; print('OK')"

# Run a test campaign (if configured)
python scripts/test_model_training.py
```

## Configuration Files

### Repo Files (Version Controlled)
- `scripts/presubmit_training.sh` - Presubmit checks
- `scripts/sync_training_to_windows.sh` - Sync script
- `config/training_medical_mechanica.toml` - Training config
- `docs/guides/TRAINING_DEVELOPMENT_WORKFLOW.md` - This file

### Local Files (Not in Repo)
- `~/.config/hafs/sync.toml` - User-specific sync settings
- `.venv/` - Python virtual environment
- `__pycache__/` - Python cache files

## Quick Troubleshooting Commands

```bash
# Check sync config
cat ~/.config/hafs/sync.toml

# Verify mount
ls /Users/scawful/Mounts/mm-d/hafs_training/src/agents/training/

# Check Windows via SSH
ssh medical-mechanica "ls D:/hafs_training/src/agents/training/"

# Verify RejectedSample tracking
grep -n "RejectedSample" ~/Code/hafs/src/agents/training/quality.py

# Run just syntax check
.venv/bin/python -m py_compile src/agents/training/*.py

# Run just import check
PYTHONPATH=src .venv/bin/python -c "from agents.training.quality import QualityScorer"
```
