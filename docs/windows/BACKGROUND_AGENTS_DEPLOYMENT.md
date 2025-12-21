# Background Agents Deployment on Windows

**Last updated:** 2025-12-21
**Platform:** medical-mechanica (Windows 10 Pro, RTX 5060 Ti 16GB)
**Status:** ✅ Deployed and operational

## Overview

Three autonomous background agents are deployed on medical-mechanica using Windows Task Scheduler:

| Agent | Schedule | Purpose | Output |
|-------|----------|---------|--------|
| **hafs-explorer** | Every 6 hours | Scan codebase, track changes, analyze dependencies | `D:\.context\scratchpad\explorer\` |
| **hafs-cataloger** | Every 12 hours | Organize datasets/models, generate metadata, storage recommendations | `D:\.context\scratchpad\cataloger\` |
| **hafs-repo-updater** | Every 4 hours | Monitor git repository, check for updates, optional auto-pull | `D:\.context\scratchpad\repo_updater\` |

## Architecture

```
medical-mechanica (Windows)
├── C:\hafs\                              # Main codebase
│   ├── .venv\                           # Python virtual environment
│   ├── src\agents\background\           # Agent implementations
│   │   ├── base.py                      # Base class with TOML config
│   │   ├── explorer.py                  # Codebase scanner
│   │   ├── cataloger.py                 # Dataset organizer
│   │   └── repo_updater.py              # Git monitor
│   ├── config\windows_background_agents.toml  # Agent configuration
│   └── setup_hafs_tasks.ps1             # Task Scheduler setup script
│
└── D:\.context\                         # AFS data storage (1.56 TB)
    ├── scratchpad\                      # Agent outputs
    │   ├── explorer\                    # Exploration reports
    │   ├── cataloger\                   # Catalog reports
    │   └── repo_updater\                # Repo status reports
    ├── logs\                            # Agent logs
    │   └── task_scheduler\              # Scheduler logs
    ├── knowledge\                       # Knowledge bases
    ├── embeddings\                      # Vector embeddings
    └── training\                        # Training data & models
        ├── datasets\                    # Training datasets
        ├── checkpoints\                 # Model checkpoints
        └── models\                      # Trained models
```

## Deployment Steps

### 1. Prerequisites

Completed before deployment:
- ✅ Windows PTY compatibility fixes (`pty.py`, `quota.py`)
- ✅ Git repository synced from Mac
- ✅ API keys configured in User environment variables
- ✅ Python virtual environment created with dependencies
- ✅ D: drive structure created for AFS storage

### 2. Task Scheduler Setup

Tasks created using `setup_hafs_tasks.ps1`:

```powershell
# Run as Administrator
powershell -ExecutionPolicy Bypass -File C:\hafs\setup_hafs_tasks.ps1
```

**Task Details:**

**hafs-explorer:**
- **Trigger:** Once, then repeat every 6 hours
- **Action:** `C:\hafs\.venv\Scripts\python.exe -m agents.background.explorer --config config\windows_background_agents.toml`
- **Working Directory:** `C:\hafs`
- **Principal:** Current user with highest privileges
- **Settings:** Start if on batteries, don't stop on battery mode, start when available

**hafs-cataloger:**
- **Trigger:** Once, then repeat every 12 hours
- **Action:** `C:\hafs\.venv\Scripts\python.exe -m agents.background.cataloger --config config\windows_background_agents.toml`
- **Working Directory:** `C:\hafs`

**hafs-repo-updater:**
- **Trigger:** Once, then repeat every 4 hours
- **Action:** `C:\hafs\.venv\Scripts\python.exe -m agents.background.repo_updater --config config\windows_background_agents.toml`
- **Working Directory:** `C:\hafs`

### 3. Verification

Verified on 2025-12-21:

```powershell
# Check task status
Get-ScheduledTask -TaskName hafs-* | Select-Object TaskName, State

# Output:
# TaskName          State
# hafs-cataloger    Ready
# hafs-explorer     Ready
# hafs-repo-updater Ready

# Check next run times
Get-ScheduledTask -TaskName hafs-* | ForEach-Object {
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
    [PSCustomObject]@{
        Task=$_.TaskName
        NextRun=$info.NextRunTime
        LastRun=$info.LastRunTime
    }
} | Format-Table -AutoSize

# Manual test
Start-ScheduledTask -TaskName hafs-explorer
Get-ScheduledTaskInfo -TaskName hafs-explorer | Select-Object LastTaskResult, LastRunTime
# LastTaskResult: 0 (success)
```

**Test Results:**
- ✅ hafs-explorer: Scanned 556 files across 4 directories
- ✅ Output files created in `D:\.context\scratchpad\explorer\`
- ✅ JSON reports generated with metadata

## Agent Configuration

Configuration in `config/windows_background_agents.toml`:

```toml
[agents.explorer]
enabled = true
provider = "claude"
model = "claude-sonnet-4-5"
schedule = "0 */6 * * *"
description = "Explore codebase and track changes"

[agents.explorer.tasks]
scan_directories = [
    "C:/hafs/src",
    "C:/hafs/docs",
    "C:/hafs/config",
    "D:/hafs_training"
]
output_dir = "D:/.context/scratchpad/explorer"
report_dir = "D:/.context/logs/explorer"

[agents.cataloger]
enabled = true
provider = "claude"
model = "claude-sonnet-4-5"
schedule = "0 */12 * * *"

[agents.cataloger.tasks]
scan_directories = [
    "D:/.context/training/datasets",
    "D:/.context/training/models"
]
output_dir = "D:/.context/scratchpad/cataloger"

[agents.repo_updater]
enabled = true
provider = "claude"
model = "claude-haiku-4"
schedule = "0 */4 * * *"

[agents.repo_updater.tasks]
local_repo = "C:/hafs"
remote_origin = "origin"
auto_pull = false  # Manual review required
output_dir = "D:/.context/scratchpad/repo_updater"
```

## Management Commands

### View Tasks

```powershell
# List all hafs tasks
Get-ScheduledTask -TaskName hafs-*

# Get detailed info
Get-ScheduledTask -TaskName hafs-explorer | Format-List *

# Check recent runs
Get-ScheduledTaskInfo -TaskName hafs-explorer
```

### Control Tasks

```powershell
# Start manually
Start-ScheduledTask -TaskName hafs-explorer

# Stop running task
Stop-ScheduledTask -TaskName hafs-explorer

# Disable task
Disable-ScheduledTask -TaskName hafs-explorer

# Enable task
Enable-ScheduledTask -TaskName hafs-explorer

# Remove task
Unregister-ScheduledTask -TaskName hafs-explorer -Confirm:$false
```

### View Output

```powershell
# Latest explorer reports
Get-ChildItem D:\.context\scratchpad\explorer | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# View summary
Get-Content D:\.context\scratchpad\explorer\exploration_summary_*.json | ConvertFrom-Json

# Latest cataloger reports
Get-ChildItem D:\.context\scratchpad\cataloger | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Repo updater status
Get-Content D:\.context\scratchpad\repo_updater\repo_status_*.json | ConvertFrom-Json
```

## Agent Capabilities

### Explorer Agent

**Scans:**
- C:\hafs\src - Source code
- C:\hafs\docs - Documentation
- C:\hafs\config - Configuration files
- D:\hafs_training - Training artifacts

**Outputs:**
- File counts by type
- Recent git commits (last 7 days)
- Dependency analysis (pyproject.toml, requirements.txt)
- Directory structure summary

**Use cases:**
- Track code changes between Mac and Windows
- Monitor codebase growth
- Identify new dependencies
- Generate development activity reports

### Cataloger Agent

**Scans:**
- D:\.context\training\datasets - Training datasets (.jsonl)
- D:\.context\training\models - Model checkpoints

**Outputs:**
- Dataset metadata (sample count, size, checksum, modified date)
- Model metadata (architecture, size, file count)
- Storage usage statistics
- Cleanup recommendations

**Use cases:**
- Track dataset growth during training campaigns
- Identify duplicate or stale models
- Monitor disk usage
- Plan storage optimization

### Repo Updater Agent

**Monitors:**
- Current git branch
- Local uncommitted changes
- Commits behind remote
- Recent commit history

**Outputs:**
- Repository status report
- Branch information
- Pending updates notification
- Optional auto-pull results

**Use cases:**
- Keep Windows repo in sync with Mac development
- Alert on pending updates
- Track sync status
- Ensure code consistency across platforms

## Sync Workflow with Mac

Background agents work within the git-based sync strategy:

### Mac (Development)
```bash
cd ~/Code/hafs
# Make changes
git add .
git commit -m "feat: your changes"
git push origin master
```

### Windows (Auto-sync via repo_updater)
```powershell
# Runs every 4 hours automatically
# Checks for updates, reports commits behind
# Optional: Enable auto_pull in config for automatic updates
```

### Manual Sync
```bash
# From Mac, trigger Windows sync
ssh medical-mechanica 'cd C:\hafs && git pull origin master'
```

## Troubleshooting

### Task Not Running

**Check task state:**
```powershell
Get-ScheduledTask -TaskName hafs-explorer | Select-Object State
```

**Check last run result:**
```powershell
Get-ScheduledTaskInfo -TaskName hafs-explorer | Select-Object LastTaskResult
# 0 = Success
# Non-zero = Error code
```

**View task history:**
```powershell
Get-WinEvent -LogName Microsoft-Windows-TaskScheduler/Operational |
    Where-Object {$_.Message -like "*hafs-explorer*"} |
    Select-Object -First 10 TimeCreated, Message
```

### Python Import Errors

**Verify virtual environment:**
```powershell
C:\hafs\.venv\Scripts\python.exe --version
C:\hafs\.venv\Scripts\python.exe -m pip list
```

**Check working directory:**
```powershell
# Task must run with WorkingDirectory = "C:\hafs"
Get-ScheduledTask -TaskName hafs-explorer |
    Select-Object -ExpandProperty Actions |
    Select-Object WorkingDirectory
```

### API Key Errors

**Verify environment variables:**
```powershell
[Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
[Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')
[Environment]::GetEnvironmentVariable('GEMINI_API_KEY', 'User')
```

**Reload environment:**
```powershell
# Restart PowerShell or system after setting variables
```

### Output Files Not Created

**Check directory permissions:**
```powershell
Test-Path D:\.context\scratchpad\explorer
Get-Acl D:\.context\scratchpad\explorer | Format-List
```

**Verify config paths:**
```powershell
Get-Content C:\hafs\config\windows_background_agents.toml
```

### Git Sync Issues

**Check repo status:**
```powershell
cd C:\hafs
git status
git remote -v
git log -5 --oneline
```

**Verify network access:**
```powershell
git fetch origin
# Should complete without errors
```

## Monitoring

### Daily Health Check

```powershell
# Check all tasks are running
Get-ScheduledTask -TaskName hafs-* | Where-Object {$_.State -ne 'Ready'}
# Should return nothing

# Check recent execution
Get-ScheduledTask -TaskName hafs-* | ForEach-Object {
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName
    [PSCustomObject]@{
        Task=$_.TaskName
        LastResult=$info.LastTaskResult
        LastRun=$info.LastRunTime
        NextRun=$info.NextRunTime
    }
} | Format-Table -AutoSize

# Check output freshness
Get-ChildItem D:\.context\scratchpad\*\* -File |
    Where-Object {$_.LastWriteTime -gt (Get-Date).AddHours(-24)} |
    Select-Object Directory, Name, LastWriteTime
```

### Weekly Review

```powershell
# Storage usage trends
Get-ChildItem D:\.context -Recurse -File |
    Measure-Object -Property Length -Sum |
    Select-Object @{N='SizeGB';E={[math]::Round($_.Sum/1GB, 2)}}, Count

# Dataset growth
Get-ChildItem D:\.context\training\datasets\*.jsonl |
    Select-Object Name, @{N='SizeMB';E={[math]::Round($_.Length/1MB, 2)}}, LastWriteTime |
    Sort-Object LastWriteTime -Descending

# Model checkpoints
Get-ChildItem D:\.context\training\models -Directory |
    ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum
        [PSCustomObject]@{
            Model=$_.Name
            SizeGB=[math]::Round($size/1GB, 2)
            Modified=$_.LastWriteTime
        }
    }
```

## Future Enhancements

### Planned Features

1. **Context Builder Agent** (daily at 2 AM)
   - Update knowledge bases
   - Rebuild embeddings
   - Sync context between Mac and Windows

2. **Mac Sync Agents** (daily)
   - Pull from Mac at 1 AM
   - Push to Mac at 11 PM
   - Bidirectional context synchronization

3. **Training Monitor Agent** (hourly)
   - Monitor training runs
   - Track GPU utilization
   - Alert on errors or completion

4. **Email Notifications**
   - Task failures
   - Storage warnings
   - Training completion alerts

### Configuration Improvements

- [ ] Centralized logging to D:\.context\logs
- [ ] Log rotation and archival
- [ ] Prometheus metrics export
- [ ] Dashboard integration (ImGui C++ app)

## References

- Main documentation: `docs/windows/WINDOWS_SETUP.md`
- Git sync strategy: `docs/windows/GIT_SYNC_STRATEGY.md`
- Agent implementations: `src/agents/background/`
- Configuration: `config/windows_background_agents.toml`
- Setup script: `setup_hafs_tasks.ps1`

## Support

For issues or questions:
1. Check task logs: `D:\.context\logs\task_scheduler\`
2. Review agent output: `D:\.context\scratchpad\{agent}\`
3. Verify configuration: `config/windows_background_agents.toml`
4. Test manually: `Start-ScheduledTask -TaskName hafs-{agent}`
5. Check git sync: Ensure Mac and Windows repos are aligned
