# PowerShell script to run training campaign on medical-mechanica
# Use D drive for all large data storage
#
# Usage:
#   .\scripts\run_training_campaign.ps1 -Target 34500
#   .\scripts\run_training_campaign.ps1 -Target 100000 -Resume

param(
    [int]$Target = 34500,
    [switch]$Resume,
    [switch]$Export,
    [string]$OutputName = "",
    [switch]$Pilot
)

$ErrorActionPreference = "Stop"

# Paths
$HafsRoot = "C:\hafs"
$DrivePaths = @{
    Datasets = "D:\hafs_training\datasets"
    Checkpoints = "D:\hafs_training\checkpoints"
    Logs = "D:\hafs_training\logs"
    Models = "D:\hafs_training\models"
    Temp = "D:\hafs_training\temp"
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "TRAINING CAMPAIGN LAUNCHER (medical-mechanica)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Target samples: $Target"
Write-Host "  Resume: $Resume"
Write-Host "  Export: $Export"
Write-Host "  Pilot mode: $Pilot"
Write-Host ""
Write-Host "Storage (D Drive):" -ForegroundColor Yellow
foreach ($key in $DrivePaths.Keys) {
    Write-Host "  $key : $($DrivePaths[$key])"
}
Write-Host ""

# Check D drive space
Write-Host "[1/5] Checking D drive space..." -ForegroundColor Green
$drive = Get-PSDrive -Name D
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
$totalSpaceGB = [math]::Round(($drive.Free + $drive.Used) / 1GB, 2)
$usedPercent = [math]::Round(($drive.Used / ($drive.Free + $drive.Used)) * 100, 1)

Write-Host "  Total: $totalSpaceGB GB"
Write-Host "  Free: $freeSpaceGB GB ($usedPercent% used)"

if ($freeSpaceGB -lt 10) {
    Write-Host "  ERROR: Less than 10 GB free on D drive!" -ForegroundColor Red
    exit 1
}
Write-Host "  OK: $freeSpaceGB GB available" -ForegroundColor Green
Write-Host ""

# Create directories
Write-Host "[2/5] Creating D drive directories..." -ForegroundColor Green
foreach ($path in $DrivePaths.Values) {
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
        Write-Host "  Created: $path" -ForegroundColor Gray
    } else {
        Write-Host "  Exists: $path" -ForegroundColor Gray
    }
}
Write-Host ""

# Set environment variables
Write-Host "[3/5] Setting environment variables..." -ForegroundColor Green
$env:PYTHONPATH = "$HafsRoot\src"
$env:TRAINING_OUTPUT_DIR = $DrivePaths.Datasets
$env:TRAINING_CHECKPOINT_DIR = $DrivePaths.Checkpoints
$env:TRAINING_LOG_DIR = $DrivePaths.Logs
Write-Host "  PYTHONPATH=$env:PYTHONPATH"
Write-Host "  TRAINING_OUTPUT_DIR=$env:TRAINING_OUTPUT_DIR"
Write-Host ""

# Build command
Write-Host "[4/5] Building command..." -ForegroundColor Green
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $DrivePaths.Logs "campaign_${Target}_${timestamp}.log"

$pythonExe = "$HafsRoot\.venv\Scripts\python.exe"
$arguments = "-m hafs_scawful.scripts.training.generate_campaign --target $Target"

if ($Resume) { $arguments += " --resume" }
if ($Export) { $arguments += " --export" }
if ($Pilot) { $arguments += " --pilot" }
if ($OutputName) { $arguments += " --output-name $OutputName" }

Write-Host "  Command: $pythonExe $arguments"
Write-Host "  Log: $logFile"
Write-Host ""

# Launch campaign
Write-Host "[5/5] Launching campaign..." -ForegroundColor Green
Write-Host "  Starting in background..."

# Start process in background and log output
$process = Start-Process `
    -FilePath $pythonExe `
    -ArgumentList $arguments `
    -WorkingDirectory $HafsRoot `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $logFile `
    -WindowStyle Hidden `
    -PassThru
$pid = $process.Id

Write-Host "  Campaign started (PID: $pid)" -ForegroundColor Green
Write-Host ""

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "CAMPAIGN RUNNING" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Process ID: $pid" -ForegroundColor Yellow
Write-Host "Log file: $logFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "Monitoring:" -ForegroundColor Yellow
Write-Host "  Live log: Get-Content '$logFile' -Wait"
Write-Host "  Status: .\scripts\check_campaign_status.ps1"
Write-Host "  Stop: Stop-Process -Id $pid"
Write-Host ""
Write-Host "Estimated completion: 8-12 hours" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan

# Save PID for later reference
$pidFile = Join-Path $DrivePaths.Checkpoints "campaign.pid"
$pid | Out-File -FilePath $pidFile -Encoding ASCII

Write-Host "PID saved to: $pidFile" -ForegroundColor Gray
