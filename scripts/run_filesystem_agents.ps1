# Run Filesystem Analysis Agents on Windows
# Explores local drives, network mounts, and remote systems to generate
# consolidation reports. All operations are READ-ONLY.

param(
    [switch]$ExplorerOnly,
    [switch]$AnalyzerOnly,
    [switch]$NetworkOnly,
    [switch]$All,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

$HafsRoot = "C:\hafs"
$PythonExe = "$HafsRoot\.venv\Scripts\python.exe"
$ConfigFile = "$HafsRoot\config\windows_filesystem_agents.toml"

# Check prerequisites
if (!(Test-Path $PythonExe)) {
    Write-Host "ERROR: Python not found at $PythonExe" -ForegroundColor Red
    exit 1
}

if (!(Test-Path $ConfigFile)) {
    Write-Host "ERROR: Config not found at $ConfigFile" -ForegroundColor Red
    exit 1
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "HAFS FILESYSTEM ANALYSIS AGENTS" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Mode: READ-ONLY filesystem exploration and reporting" -ForegroundColor Green
Write-Host "Config: $ConfigFile" -ForegroundColor Gray
Write-Host ""

$verboseFlag = if ($Verbose) { "-v" } else { "" }

# Run filesystem explorer
if ($ExplorerOnly -or $All -or (-not $AnalyzerOnly -and -not $NetworkOnly)) {
    Write-Host "[1/3] Running Filesystem Explorer..." -ForegroundColor Yellow
    Write-Host "      Scanning local drives for files, duplicates, and organization opportunities" -ForegroundColor Gray
    Write-Host ""

    $startTime = Get-Date

    & $PythonExe -m agents.background.filesystem_explorer `
        --config $ConfigFile `
        $verboseFlag

    $duration = (Get-Date) - $startTime
    Write-Host ""
    Write-Host "      Completed in $($duration.TotalSeconds.ToString('0.0')) seconds" -ForegroundColor Green
    Write-Host ""
}

# Run consolidation analyzer
if ($AnalyzerOnly -or $All) {
    Write-Host "[2/3] Running Consolidation Analyzer..." -ForegroundColor Yellow
    Write-Host "      Analyzing filesystem inventory and generating recommendations" -ForegroundColor Gray
    Write-Host ""

    $startTime = Get-Date

    & $PythonExe -m agents.background.consolidation_analyzer `
        --config $ConfigFile `
        $verboseFlag

    $duration = (Get-Date) - $startTime
    Write-Host ""
    Write-Host "      Completed in $($duration.TotalSeconds.ToString('0.0')) seconds" -ForegroundColor Green
    Write-Host ""
}

# Run network inventory
if ($NetworkOnly -or $All) {
    Write-Host "[3/3] Running Network Inventory..." -ForegroundColor Yellow
    Write-Host "      Querying remote systems (Mac, halext-server) via SSH" -ForegroundColor Gray
    Write-Host ""

    $startTime = Get-Date

    & $PythonExe -m agents.background.network_inventory `
        --config $ConfigFile `
        $verboseFlag

    $duration = (Get-Date) - $startTime
    Write-Host ""
    Write-Host "      Completed in $($duration.TotalSeconds.ToString('0.0')) seconds" -ForegroundColor Green
    Write-Host ""
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "ANALYSIS COMPLETE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Reports saved to:" -ForegroundColor Yellow
Write-Host "  D:\.context\scratchpad\filesystem_explorer\" -ForegroundColor White
Write-Host "  D:\.context\scratchpad\consolidation_analyzer\" -ForegroundColor White
Write-Host "  D:\.context\scratchpad\network_inventory\" -ForegroundColor White
Write-Host ""
Write-Host "Logs saved to:" -ForegroundColor Yellow
Write-Host "  D:\.context\logs\filesystem_explorer\" -ForegroundColor White
Write-Host "  D:\.context\logs\consolidation_analyzer\" -ForegroundColor White
Write-Host "  D:\.context\logs\network_inventory\" -ForegroundColor White
Write-Host ""

# Show latest reports
Write-Host "Latest Reports:" -ForegroundColor Yellow
Write-Host ""

# Filesystem explorer summary
$explorerSummary = Get-ChildItem "D:\.context\scratchpad\filesystem_explorer\filesystem_summary_*.json" `
    -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($explorerSummary) {
    Write-Host "Filesystem Summary:" -ForegroundColor Cyan
    $summary = Get-Content $explorerSummary.FullName | ConvertFrom-Json
    Write-Host "  Files scanned: $($summary.total_files)" -ForegroundColor White
    Write-Host "  Total size: $($summary.total_size_gb) GB" -ForegroundColor White
    Write-Host "  Duplicate groups: $($summary.duplicate_groups)" -ForegroundColor White
    Write-Host "  Potential savings: $($summary.potential_savings_gb) GB" -ForegroundColor White
    Write-Host ""
}

# Consolidation analysis summary
$consolidationReport = Get-ChildItem "D:\.context\scratchpad\consolidation_analyzer\consolidation_report_*.md" `
    -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($consolidationReport) {
    Write-Host "Consolidation Report: $($consolidationReport.FullName)" -ForegroundColor Cyan
    Write-Host ""
    Get-Content $consolidationReport.FullName | Select-Object -First 20
    Write-Host ""
    Write-Host "  [Full report at: $($consolidationReport.FullName)]" -ForegroundColor Gray
    Write-Host ""
}

# Network inventory summary
$networkSummary = Get-ChildItem "D:\.context\scratchpad\network_inventory\network_summary_*.json" `
    -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($networkSummary) {
    Write-Host "Network Inventory:" -ForegroundColor Cyan
    $summary = Get-Content $networkSummary.FullName | ConvertFrom-Json
    Write-Host "  Remote systems: $($summary.remote_systems_total)" -ForegroundColor White
    Write-Host "  Reachable: $($summary.reachable_systems)" -ForegroundColor White
    if ($summary.reachable_system_names.Count -gt 0) {
        Write-Host "  Systems: $($summary.reachable_system_names -join ', ')" -ForegroundColor White
    }
    Write-Host "  Remote size: $($summary.total_remote_size_gb) GB" -ForegroundColor White
    Write-Host ""
}

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Review consolidation report for recommended actions" -ForegroundColor White
Write-Host "  2. Check duplicate candidates and verify before deletion" -ForegroundColor White
Write-Host "  3. Plan file organization based on suggestions" -ForegroundColor White
Write-Host ""
