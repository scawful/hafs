# Windows Background Agents Guide

**System:** GPU host (Windows 10/11 Pro 64-bit)
**Purpose:** Automated exploration, cataloging, and context building for hafs
**Providers:** Claude (primary), OpenAI (secondary) - **Gemini reserved for training data generation**

## Overview

Background agents run automatically on the GPU host to:
1. **Explore** - Scan codebase for changes and structure
2. **Catalog** - Organize training datasets and artifacts
3. **Build Context** - Update knowledge bases and embeddings
4. **Sync** - Coordinate .context between Mac and Windows
5. **Monitor** - Track repo updates and health

## Configuration

All agents are configured in `config/windows_background_agents.toml`.
Treat it as a template and copy into your user plugin repo for edits:
`~/Code/hafs_scawful/config/windows_background_agents.toml`.
Use `~/Code/hafs_scawful/scripts/publish_plugin_configs.sh` to sync those
configs/docs to halext-server and the Windows GPU host.

### Agent Roster

| Agent | Schedule | Provider | Purpose |
|-------|----------|----------|---------|
| **explorer** | Every 6 hours | Claude Sonnet 4.5 | Scan codebase, track changes |
| **cataloger** | Every 12 hours | Claude Sonnet 4.5 | Organize datasets and models |
| **context_builder** | Daily at 2 AM | GPT-4o | Update knowledge bases |
| **repo_updater** | Every 4 hours | Claude Haiku 4 | Monitor repo changes |
| **mac_sync_pull** | Daily at 1 AM | - | Pull context from Mac |
| **mac_sync_push** | Daily at 11 PM | - | Push context to Mac |

### Why Not Gemini?

**Gemini is reserved exclusively for training data generation** (teacher model in agent.training.generators). Background agents use Claude/OpenAI to:
- Conserve Gemini API quotas for high-value training workloads
- Avoid rate limit conflicts during campaigns
- Reduce costs (Claude Haiku is cheaper for simple tasks)

## Setup

### 1. Prerequisites

```powershell
# Verify hafs is installed
cd C:\hafs
.\.venv\Scripts\python.exe -c "import hafs; print('OK')"

# Check .context directories exist
Test-Path D:\.context
Test-Path D:\hafs_training
```

### 2. Set API Keys

```powershell
# Required: Set in User environment variables (persists across reboots)
[Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-...', 'User')
[Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-...', 'User')
[Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIza...', 'User')  # For training only

# Verify
$env:ANTHROPIC_API_KEY  # Should show your key
$env:OPENAI_API_KEY     # Should show your key
```

### 3. Copy Configuration

```powershell
# Copy background agents config to hafs
Copy-Item config\windows_background_agents.toml C:\hafs\config\
```

### 4. Create Scheduled Tasks

Run PowerShell as Administrator:

```powershell
cd C:\hafs
.\scripts\setup_background_agents.ps1
```

Expected output:
```
[1/6] Creating Explorer Agent task...
  ✓ hafs-explorer created (every 6 hours)
[2/6] Creating Cataloger Agent task...
  ✓ hafs-cataloger created (every 12 hours)
...
SETUP COMPLETE
```

### 5. Verify Tasks

```powershell
# List all hafs tasks
Get-ScheduledTask -TaskName 'hafs-*' | Select-Object TaskName, State, LastRunTime, NextRunTime

# Example output:
# TaskName                State   LastRunTime        NextRunTime
# --------                -----   -----------        -----------
# hafs-explorer          Ready                      12/21/2025 3:00 PM
# hafs-cataloger         Ready                      12/21/2025 9:00 PM
# hafs-context-builder   Ready                      12/22/2025 2:00 AM
```

## Agent Details

### Explorer Agent

**Schedule:** Every 6 hours
**Provider:** Claude Sonnet 4.5

Scans directories:
- `C:/hafs/src` - Source code
- `C:/hafs/docs` - Documentation
- `C:/hafs/config` - Configuration files
- `D:/hafs_training` - Training artifacts

Outputs:
- `D:/.context/scratchpad/explorer/` - Working notes
- `D:/.context/logs/explorer/` - Activity logs

Capabilities:
- File change detection
- Dependency tracking
- Code structure mapping
- Documentation indexing
- Automatic changelog generation

### Cataloger Agent

**Schedule:** Every 12 hours
**Provider:** Claude Sonnet 4.5

Organizes:
- Training datasets (`D:/hafs_training/datasets`)
- Trained models (`D:/hafs_training/models`)
- Knowledge bases (`D:/.context/knowledge`)

Outputs:
- Dataset inventory reports
- Model metadata
- Quality assessments
- Storage optimization suggestions

### Context Builder Agent

**Schedule:** Daily at 2 AM
**Provider:** GPT-4o

Updates knowledge bases:
- `oracle-of-secrets` - ALTTP ROM hacking
- `gigaleak` - Nintendo source code
- `yaze-docs` - YAZE emulator

Can sync from Mac:
- Pulls from `~/Mounts/your-mac/.context` (if `sync_from_mac = true`)
- Updates embeddings
- Cross-references knowledge

### Repo Updater Agent

**Schedule:** Every 4 hours
**Provider:** Claude Haiku 4 (cheaper for simple tasks)

Monitors:
- Local repo: `C:/hafs`
- Remote origin: `https://github.com/youruser/hafs.git`
- Mac repo: `~/Mounts/your-mac/Code/hafs` (via network mount)

Actions:
- Checks for updates
- Detects conflicts
- Tracks changelog
- **Does NOT auto-pull** (set `auto_pull = true` to enable)

### Mac Sync Agents

**Pull Schedule:** Daily at 1 AM
**Push Schedule:** Daily at 11 PM

Syncs between:
- Windows: `D:/.context`
- Mac: `/Users/youruser/.context` (via network mount)

Synced directories:
- `knowledge/` - Knowledge bases
- `scratchpad/` - Working notes
- `hivemind/` - Multi-agent coordination

NOT synced (too large):
- `embeddings/` - Generated locally
- `models/` - Generated locally
- `datasets/` - Generated locally
- `logs/` - Kept separate

## Management

### Start All Agents

```powershell
Start-ScheduledTask -TaskName 'hafs-explorer'
Start-ScheduledTask -TaskName 'hafs-cataloger'
Start-ScheduledTask -TaskName 'hafs-context-builder'
Start-ScheduledTask -TaskName 'hafs-repo-updater'
```

### Stop All Agents

```powershell
Get-ScheduledTask -TaskName 'hafs-*' | Stop-ScheduledTask
```

### Run Manually (Test)

```powershell
cd C:\hafs

# Activate venv
.\.venv\Scripts\Activate.ps1

# Run explorer manually
python -m agents.background.explorer --config config\windows_background_agents.toml

# Run with verbose logging
python -m agents.background.explorer --config config\windows_background_agents.toml --verbose
```

### View Logs

```powershell
# Explorer logs
Get-Content D:\.context\logs\explorer\latest.log -Wait -Tail 50

# Task Scheduler logs
Get-Content D:\.context\logs\task_scheduler\hafs-explorer.log -Wait -Tail 50

# All agent logs
Get-ChildItem D:\.context\logs\*\*.log -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 10
```

### Remove All Tasks

```powershell
cd C:\hafs
.\scripts\setup_background_agents.ps1 -Remove
```

## Monitoring

### Health Check

Create a monitoring script:

```powershell
# health_check.ps1
$tasks = Get-ScheduledTask -TaskName 'hafs-*'

foreach ($task in $tasks) {
    $info = Get-ScheduledTaskInfo -TaskName $task.TaskName
    $status = if ($info.LastTaskResult -eq 0) { "✓ OK" } else { "✗ FAILED" }

    Write-Host "$($task.TaskName.PadRight(30)) $status  Last: $($info.LastRunTime)" -ForegroundColor $(if ($info.LastTaskResult -eq 0) { "Green" } else { "Red" })
}

# Check disk space
$drive = Get-PSDrive D
$freeGB = [math]::Round($drive.Free / 1GB, 2)
$usedPercent = [math]::Round(($drive.Used / ($drive.Free + $drive.Used)) * 100, 1)

Write-Host ""
Write-Host "D Drive: $freeGB GB free ($usedPercent% used)" -ForegroundColor $(if ($freeGB -lt 100) { "Yellow" } else { "Green" })
```

Run:
```powershell
.\health_check.ps1
```

### Alerts

Configure alerts in `config/windows_background_agents.toml`:

```toml
[monitoring.disk]
warn_threshold_gb = 200
critical_threshold_gb = 100

[monitoring.alerts]
log_file = "D:/.context/logs/monitoring/alerts.log"
```

## Troubleshooting

### Agent Not Running

```powershell
# Check task state
Get-ScheduledTask -TaskName 'hafs-explorer' | Select-Object State, LastRunTime, LastTaskResult

# Check task history (Event Viewer)
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-TaskScheduler/Operational'; Id=200,201} -MaxEvents 10 |
    Where-Object { $_.Message -like '*hafs*' }
```

### API Key Errors

```powershell
# Verify environment variables
[Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
[Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')

# Test API access
cd C:\hafs
.\.venv\Scripts\python.exe -c "from anthropic import Anthropic; print(Anthropic().models.list())"
```

### Permission Errors

Run Task Scheduler as Administrator:
```powershell
Start-Process taskschd.msc -Verb RunAs
```

### Mac Sync Failing

Check network mount:
```powershell
# Test SSH connection
ssh youruser@your-mac-host echo "OK"

# Test mount access (if using SMB)
Test-Path \\your-mac-host\Code\hafs
```

## Best Practices

1. **Monitor Disk Space**: Check D drive weekly
   ```powershell
   Get-PSDrive D | Select-Object Used,Free
   ```

2. **Review Logs Periodically**: Check for errors
   ```powershell
   Get-Content D:\.context\logs\monitoring\alerts.log
   ```

3. **Update Agent Config**: Adjust schedules based on usage
   ```toml
   # Reduce frequency if not needed
   schedule = "0 */12 * * *"  # Every 12 hours instead of 6
   ```

4. **Test Before Deploying**: Run agents manually first
   ```powershell
   python -m agents.background.explorer --config config\windows_background_agents.toml --dry-run
   ```

5. **Keep API Quotas Separate**: Never use Gemini for background work

## Remote Management

### From Mac via SSH

```bash
# Check agent status
ssh Administrator@GPU_HOST 'powershell Get-ScheduledTask -TaskName "hafs-*"'

# View logs
ssh Administrator@GPU_HOST 'powershell Get-Content D:\.context\logs\explorer\latest.log -Tail 50'

# Start/stop agents
ssh Administrator@GPU_HOST 'powershell Start-ScheduledTask -TaskName "hafs-explorer"'
ssh Administrator@GPU_HOST 'powershell Stop-ScheduledTask -TaskName "hafs-explorer"'
```

### From Mac via Mounted Drives

```bash
# View logs directly
tail -f ~/Mounts/your-mac/.context/logs/explorer/latest.log

# Check agent outputs
ls ~/Mounts/your-mac/.context/scratchpad/explorer/
```

## Integration with Training

Background agents work alongside training workflows:

1. **Explorer** - Detects new training datasets in `D:/hafs_training/datasets`
2. **Cataloger** - Organizes and indexes datasets for easy discovery
3. **Context Builder** - Updates knowledge bases with training results
4. **Repo Updater** - Ensures hafs code is in sync for running campaigns

**Key Principle:** Background agents use Claude/OpenAI, leaving Gemini quota free for training campaigns.

---

**Last Updated**: 2025-12-21
**System**: GPU host (Windows 10/11 Pro 64-bit)
**Configuration**: `C:/hafs/config/windows_background_agents.toml`
