# Setup hafs Background Agent Tasks on Windows
# Run as Administrator

$ErrorActionPreference = "Stop"

Write-Host "========================================================================"
Write-Host "HAFS BACKGROUND AGENTS SETUP"
Write-Host "========================================================================"
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (!$isAdmin) {
    Write-Host "ERROR: Must run as Administrator to create scheduled tasks" -ForegroundColor Red
    exit 1
}

$HafsRoot = "C:\hafs"
$PythonExe = "$HafsRoot\.venv\Scripts\python.exe"
$LogDir = "D:\.context\logs\task_scheduler"

# Create log directory
if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Principal for running tasks (use current user)
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -RunLevel Highest

# Task settings
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

# [1] Explorer Agent (every 6 hours)
Write-Host "[1/3] Creating Explorer Agent task..." -ForegroundColor Yellow

$action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "-m agents.background.explorer --config config\windows_background_agents.toml" `
    -WorkingDirectory $HafsRoot

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 6)

Register-ScheduledTask `
    -TaskName "hafs-explorer" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "hafs Explorer Agent - Scan codebase every 6 hours" `
    -Force | Out-Null

Write-Host "  Created: hafs-explorer (every 6 hours)" -ForegroundColor Green

# [2] Cataloger Agent (every 12 hours)
Write-Host "[2/3] Creating Cataloger Agent task..." -ForegroundColor Yellow

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
    -Description "hafs Cataloger Agent - Organize datasets every 12 hours" `
    -Force | Out-Null

Write-Host "  Created: hafs-cataloger (every 12 hours)" -ForegroundColor Green

# [3] Repo Updater Agent (every 4 hours)
Write-Host "[3/3] Creating Repo Updater Agent task..." -ForegroundColor Yellow

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
    -Description "hafs Repo Updater - Check for updates every 4 hours" `
    -Force | Out-Null

Write-Host "  Created: hafs-repo-updater (every 4 hours)" -ForegroundColor Green

Write-Host ""
Write-Host "========================================================================"
Write-Host "SETUP COMPLETE"
Write-Host "========================================================================"
Write-Host ""
Write-Host "Background agents configured:" -ForegroundColor Yellow
Write-Host "  1. Explorer (every 6 hours) - Scan codebase"
Write-Host "  2. Cataloger (every 12 hours) - Organize datasets"
Write-Host "  3. Repo Updater (every 4 hours) - Monitor repo changes"
Write-Host ""
Write-Host "Manage tasks:" -ForegroundColor Yellow
Write-Host "  View: Get-ScheduledTask -TaskName 'hafs-*'"
Write-Host "  Start: Start-ScheduledTask -TaskName 'hafs-explorer'"
Write-Host "  Stop: Stop-ScheduledTask -TaskName 'hafs-explorer'"
Write-Host ""
Write-Host "Logs: $LogDir" -ForegroundColor Yellow
Write-Host ""
