# Git Sync Strategy: Mac ↔ Windows

**Purpose:** Keep hafs repos synchronized between Mac (development) and Windows (GPU server) using git instead of manual rsync/copy.

## Current Setup

### Mac (Primary Development)
- **Path:** `~/Code/hafs`
- **Branch:** `master`
- **Remote:** `origin` (GitHub)
- **Role:** Primary development machine

### Windows (medical-mechanica)
- **Path:** `C:\hafs`
- **Branch:** `master`
- **Remote:** `origin` (GitHub)
- **Role:** GPU training server + background agents

## Git-Based Sync Workflow

### Initial Setup (Already Done ✅)

1. **Mac:** Commit and push changes
   ```bash
   cd ~/Code/hafs
   git add .
   git commit -m "feat: your changes"
   git push origin master
   ```

2. **Windows:** Pull changes
   ```powershell
   cd C:\hafs
   git pull origin master
   ```

### Daily Workflow

#### When Developing on Mac

```bash
# 1. Make changes on Mac
cd ~/Code/hafs
# ... edit files ...

# 2. Test locally
python -m hafs.cli --help

# 3. Commit changes
git add .
git commit -m "feat: describe your changes"

# 4. Push to GitHub
git push origin master

# 5. Pull on Windows (via SSH from Mac)
ssh medical-mechanica 'cd C:\hafs && git pull origin master'

# Or via mounted drive
echo "git pull origin master" | ssh medical-mechanica 'cd C:\hafs && cmd'
```

#### When Testing on Windows

```powershell
# 1. Pull latest from Mac
cd C:\hafs
git pull origin master

# 2. Activate venv
.\.venv\Scripts\Activate.ps1

# 3. Test changes
python -m hafs.cli --help
python -m agents.training.health_check
```

## Platform-Specific Files

Some files should differ between platforms:

### Mac-Specific (not committed)
- `.venv/` - Python virtual environment
- `build/` - CMake build artifacts
- `__pycache__/` - Python cache
- `.DS_Store` - macOS metadata

### Windows-Specific (not committed)
- `.venv/` - Windows Python virtual environment
- `build/` - Windows build artifacts
- `*.pyc` - Python bytecode

### Shared Configuration Files (committed)
- `hafs.toml` - Edit per-platform as needed
- `config/windows_background_agents.toml` - Windows-specific (committed)
- `docs/windows/` - Windows documentation (committed)
- `scripts/*.ps1` - PowerShell scripts (committed)
- `scripts/*.sh` - Bash scripts (committed)

## .gitignore Configuration

Already configured in `.gitignore`:
```gitignore
# Python
__pycache__/
*.pyc
.venv/
build/

# OS
.DS_Store
*.swp

# Logs
logs/
*.log

# Sensitive
.env
secrets.toml
```

## Handling Conflicts

### If Mac and Windows Both Modified Same File

**On Windows:**
```powershell
cd C:\hafs

# Pull and see conflict
git pull origin master

# If conflict, resolve manually
git status
# Edit conflicted files
git add .
git commit -m "merge: resolve conflict"
git push origin master
```

**On Mac:**
```bash
cd ~/Code/hafs
git pull origin master
```

### Best Practice: Avoid Conflicts

1. **Always pull before editing** on either platform
2. **Commit frequently** on Mac (primary development)
3. **Windows is read-mostly** - use for testing, not development
4. **Use branches** for experimental features

## Automated Sync with Background Agents

The `repo_updater` background agent can auto-sync Windows:

**In `config/windows_background_agents.toml`:**
```toml
[agents.repo_updater]
enabled = true
schedule = "0 */4 * * *"  # Every 4 hours
auto_pull = true  # Enable automatic git pull
```

**Manual trigger:**
```powershell
cd C:\hafs
python -m agents.background.repo_updater
```

## Quick Reference Commands

### Mac → Windows Sync

```bash
# Method 1: Push + SSH pull (recommended)
cd ~/Code/hafs
git add . && git commit -m "your message" && git push
ssh medical-mechanica 'cd C:\hafs && git pull origin master'

# Method 2: Via mounted drive (if SSH unavailable)
cd ~/Code/hafs
git add . && git commit -m "your message" && git push
# Then on Windows via RDP:
cd C:\hafs && git pull origin master
```

### Windows → Mac Sync (rare, for testing)

```powershell
# On Windows
cd C:\hafs
git add .
git commit -m "test: Windows-specific fix"
git push origin master

# On Mac
cd ~/Code/hafs
git pull origin master
```

### Check Sync Status

**Mac:**
```bash
cd ~/Code/hafs
git status
git log --oneline -5
```

**Windows:**
```powershell
cd C:\hafs
git status
git log --oneline -5
```

**Compare:**
```bash
# On Mac
ssh medical-mechanica 'cd C:\hafs && git log --oneline -1'
cd ~/Code/hafs && git log --oneline -1

# Should show same commit hash
```

## Benefits Over rsync/tar

| Aspect | Git | rsync/tar |
|--------|-----|-----------|
| **Version control** | ✅ Full history | ❌ No history |
| **Selective sync** | ✅ Only changed files | ❌ All or nothing |
| **Conflict detection** | ✅ Automatic | ❌ Manual |
| **Rollback** | ✅ Easy (`git revert`) | ❌ Complex |
| **Branching** | ✅ Yes | ❌ No |
| **Collaboration** | ✅ Multi-developer | ❌ Single user |
| **Platform fixes** | ✅ Conditional code | ❌ Duplicate codebases |

## Platform-Specific Code Example

Thanks to git, we can maintain a single codebase with platform-specific code:

```python
# src/backends/cli/pty.py
import platform

_PLATFORM = platform.system()

if _PLATFORM != "Windows":
    import pty
    import termios
else:
    # Windows alternative
    import subprocess
```

Both Mac and Windows use the **same file**, but with conditional logic.

## Troubleshooting

### "Changes not staged for commit" on Windows

```powershell
# See what changed
git diff

# If changes are needed, commit them
git add .
git commit -m "fix: Windows-specific adjustment"
git push origin master

# If changes are unwanted (e.g., line endings), reset
git restore .
```

### File Permission Issues (Windows)

Windows may change file permissions. Add to `.gitattributes`:
```
*.sh text eol=lf
*.ps1 text eol=crlf
```

### Stale .venv on Windows After Pull

```powershell
# Reinstall dependencies if needed
cd C:\hafs
.\.venv\Scripts\Activate.ps1
pip install -e .[backends] --upgrade
```

## Commit Message Conventions

Use these prefixes for clarity:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code restructuring
- `test:` - Add/update tests
- `chore:` - Tooling, dependencies
- `platform:` - Windows/Mac specific fixes

**Example:**
```bash
git commit -m "platform: fix Windows PTY compatibility with subprocess fallback"
```

## Summary

**Before (rsync/tar):**
- Manual file copying
- No version control
- Risk of file conflicts
- Duplicate codebases diverging

**Now (git):**
- `git pull` to sync
- Full version history
- Automatic conflict detection
- Single codebase, platform-specific code via conditionals

**Workflow:**
1. Develop on Mac → commit → push
2. Pull on Windows (manual or automatic via background agent)
3. Test on Windows GPU
4. Report issues → fix on Mac → repeat

---

**Last Updated:** 2025-12-21
**Status:** ✅ Both repos in sync with Windows compatibility fixes
**Next Sync:** Automatic via repo_updater agent (every 4 hours)
