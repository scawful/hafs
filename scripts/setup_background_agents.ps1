# Setup Background Agents on Windows (medical-mechanica)
# Creates Windows Task Scheduler tasks for automated exploration, cataloging, and context building

param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

$HafsRoot = "C:\hafs"
$PythonExe = "$HafsRoot\.venv\Scripts\python.exe"
$LogDir = "D:\.context\logs\task_scheduler"

# Create log directory
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "HAFS BACKGROUND AGENTS SETUP" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

if ($Remove) {
    Write-Host "Removing background agent tasks..." -ForegroundColor Yellow

    $tasks = @(
        "hafs-explorer",
        "hafs-cataloger",
        "hafs-context-builder",
        "hafs-repo-updater",
        "hafs-mac-sync-pull",
        "hafs-mac-sync-push"
    )

    foreach ($task in $tasks) {
        try {
            Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue
            Write-Host "  ✓ Removed: $task" -ForegroundColor Green
        } catch {
            Write-Host "  - Not found: $task" -ForegroundColor Gray
        }
    }

    Write-Host ""
    Write-Host "All tasks removed." -ForegroundColor Green
    exit 0
}

Write-Host "Creating background agent tasks..." -ForegroundColor Green
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (!$isAdmin) {
    Write-Host "ERROR: Must run as Administrator to create scheduled tasks" -ForegroundColor Red
    exit 1
}

# Principal for running tasks
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -RunLevel Highest

# [1] Explorer Agent (every 6 hours)
Write-Host "[1/6] Creating Explorer Agent task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.explorer --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "hafs-explorer" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Explorer Agent - Scan codebase every 6 hours" | Out-Null

Write-Host "  ✓ hafs-explorer created (every 6 hours)" -ForegroundColor Green

# [2] Cataloger Agent (every 12 hours)
Write-Host "[2/6] Creating Cataloger Agent task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.cataloger --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 12)

Register-ScheduledTask `
    -TaskName "hafs-cataloger" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Cataloger Agent - Organize datasets every 12 hours" | Out-Null

Write-Host "  ✓ hafs-cataloger created (every 12 hours)" -ForegroundColor Green

# [3] Context Builder Agent (daily at 2 AM)
Write-Host "[3/6] Creating Context Builder Agent task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.context_builder --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Daily -At 2am

Register-ScheduledTask `
    -TaskName "hafs-context-builder" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Context Builder - Update knowledge bases daily at 2 AM" | Out-Null

Write-Host "  ✓ hafs-context-builder created (daily at 2 AM)" -ForegroundColor Green

# [4] Repo Updater Agent (every 4 hours)
Write-Host "[4/6] Creating Repo Updater Agent task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.repo_updater --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 4)

Register-ScheduledTask `
    -TaskName "hafs-repo-updater" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Repo Updater - Check for updates every 4 hours" | Out-Null

Write-Host "  ✓ hafs-repo-updater created (every 4 hours)" -ForegroundColor Green

# [5] Mac Sync Pull (daily at 1 AM)
Write-Host "[5/6] Creating Mac Sync Pull task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.mac_sync --direction pull --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Daily -At 1am

Register-ScheduledTask `
    -TaskName "hafs-mac-sync-pull" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Mac Sync - Pull context from Mac daily at 1 AM" | Out-Null

Write-Host "  ✓ hafs-mac-sync-pull created (daily at 1 AM)" -ForegroundColor Green

# [6] Mac Sync Push (daily at 11 PM)
Write-Host "[6/6] Creating Mac Sync Push task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.mac_sync --direction push --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Daily -At 11pm

Register-ScheduledTask `
    -TaskName "hafs-mac-sync-push" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Mac Sync - Push context to Mac daily at 11 PM" | Out-Null

Write-Host "  ✓ hafs-mac-sync-push created (daily at 11 PM)" -ForegroundColor Green

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Background agents configured:" -ForegroundColor Yellow
Write-Host "  1. Explorer (every 6 hours) - Scan codebase"
Write-Host "  2. Cataloger (every 12 hours) - Organize datasets"
Write-Host "  3. Context Builder (daily 2 AM) - Update knowledge bases"
Write-Host "  4. Repo Updater (every 4 hours) - Monitor repo changes"
Write-Host "  5. Mac Sync Pull (daily 1 AM) - Pull context from Mac"
Write-Host "  6. Mac Sync Push (daily 11 PM) - Push context to Mac"
Write-Host ""
Write-Host "Manage tasks:" -ForegroundColor Yellow
Write-Host "  View: Get-ScheduledTask -TaskName 'hafs-*'"
Write-Host "  Start: Start-ScheduledTask -TaskName 'hafs-explorer'"
Write-Host "  Stop: Stop-ScheduledTask -TaskName 'hafs-explorer'"
Write-Host "  Remove all: .\setup_background_agents.ps1 -Remove"
Write-Host ""
Write-Host "Logs: $LogDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "NOTE: Set API keys before running:" -ForegroundColor Red
Write-Host "  [Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-...', 'User')"
Write-Host "  [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-...', 'User')"
Write-Host "  [Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIza...', 'User')"
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
