# Setup hafs on Windows (medical-mechanica)
# This script:
# 1. Clones/syncs hafs to C:\hafs
# 2. Sets up Python virtual environment
# 3. Installs dependencies
# 4. Creates .context structure on D drive
# 5. Sets up Windows services for background agents

param(
    [switch]$SkipClone,
    [switch]$UseDDrive = $true
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "HAFS WINDOWS SETUP (medical-mechanica)" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (!$isAdmin) {
    Write-Host "⚠️  Not running as Administrator. Some features may be limited." -ForegroundColor Yellow
    Write-Host ""
}

# Paths
$HafsRoot = "C:\hafs"
$ContextRoot = if ($UseDDrive) { "D:\.context" } else { "$env:USERPROFILE\.context" }

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  hafs root: $HafsRoot"
Write-Host "  .context: $ContextRoot"
Write-Host ""

# [1] Check Python
Write-Host "[1/7] Checking Python..." -ForegroundColor Green
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green

    # Check if Python 3.11+
    $versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
    if ($versionMatch) {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
            Write-Host "  ⚠️  Python 3.11+ recommended (found $major.$minor)" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "  ❌ Python not found!" -ForegroundColor Red
    Write-Host "  Please install Python 3.11+ from https://www.python.org/" -ForegroundColor Red
    exit 1
}
Write-Host ""

# [2] Setup hafs directory
Write-Host "[2/7] Setting up hafs directory..." -ForegroundColor Green
if (!(Test-Path $HafsRoot)) {
    Write-Host "  Creating $HafsRoot..."
    New-Item -ItemType Directory -Path $HafsRoot -Force | Out-Null
}

if (!$SkipClone) {
    Write-Host "  Syncing code from Mac..."
    Write-Host "  (Run from Mac: rsync -avz ~/Code/hafs/ medical-mechanica:C:/hafs/)"
    Write-Host "  Waiting for manual sync..."
    Write-Host "  Press Enter when sync complete..."
    Read-Host
}

# Check for key files
$requiredFiles = @("src")
$missing = @()
foreach ($file in $requiredFiles) {
    if (!(Test-Path (Join-Path $HafsRoot $file))) {
        $missing += $file
    }
}

$hasRequirements = Test-Path (Join-Path $HafsRoot "requirements.txt")
$hasPyproject = Test-Path (Join-Path $HafsRoot "pyproject.toml")
if (-not ($hasRequirements -or $hasPyproject)) {
    $missing += "requirements.txt or pyproject.toml"
}

if ($missing.Count -gt 0) {
    Write-Host "  ⚠️  Missing files: $($missing -join ', ')" -ForegroundColor Yellow
    Write-Host "  Please sync hafs code first" -ForegroundColor Yellow
}
Write-Host ""

# [3] Setup virtual environment
Write-Host "[3/7] Setting up Python virtual environment..." -ForegroundColor Green
$venvPath = Join-Path $HafsRoot ".venv"

if (!(Test-Path $venvPath)) {
    Write-Host "  Creating virtual environment..."
    python -m venv $venvPath
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  Virtual environment exists" -ForegroundColor Gray
}

# Activate venv
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
Write-Host "  Activating virtual environment..."
& $activateScript

# Upgrade pip
Write-Host "  Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "  Installing dependencies..."
$requirementsPath = Join-Path $HafsRoot "requirements.txt"
$pyprojectPath = Join-Path $HafsRoot "pyproject.toml"
if (Test-Path $requirementsPath) {
    pip install -r $requirementsPath --quiet
    Write-Host "  ✓ Dependencies installed (requirements.txt)" -ForegroundColor Green
} elseif (Test-Path $pyprojectPath) {
    pip install -e $HafsRoot --quiet
    Write-Host "  ✓ Dependencies installed (pyproject.toml)" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  requirements.txt or pyproject.toml not found" -ForegroundColor Yellow
}
Write-Host ""

# [4] Create .context structure
Write-Host "[4/7] Setting up .context directory..." -ForegroundColor Green
$contextDirs = @(
    "knowledge",
    "embeddings",
    "logs",
    "training\datasets",
    "training\checkpoints",
    "training\models",
    "training\temp",
    "history\agents",
    "scratchpad",
    "memory",
    "hivemind"
)

foreach ($dir in $contextDirs) {
    $fullPath = Join-Path $ContextRoot $dir
    if (!(Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

Write-Host "  ✓ .context structure created at: $ContextRoot" -ForegroundColor Green

# Create symlink if using D drive
if ($UseDDrive -and $isAdmin) {
    $symlinkTarget = "$env:USERPROFILE\.context"
    if (!(Test-Path $symlinkTarget)) {
        Write-Host "  Creating symlink: $symlinkTarget -> $ContextRoot"
        New-Item -ItemType SymbolicLink -Path $symlinkTarget -Target $ContextRoot -Force | Out-Null
        Write-Host "  ✓ Symlink created" -ForegroundColor Green
    }
} elseif ($UseDDrive -and !$isAdmin) {
    Write-Host "  ⚠️  Run as Administrator to create symlink" -ForegroundColor Yellow
}
Write-Host ""

# [5] Create config files
Write-Host "[5/7] Creating configuration files..." -ForegroundColor Green

# hafs.toml
$hafsConfig = @"
# hafs Configuration for Windows (medical-mechanica)

[paths]
context_root = "$($ContextRoot.Replace('\', '/'))"
training_datasets = "$($ContextRoot.Replace('\', '/'))/training/datasets"
training_checkpoints = "$($ContextRoot.Replace('\', '/'))/training/checkpoints"
training_logs = "$($ContextRoot.Replace('\', '/'))/training/logs"
training_models = "$($ContextRoot.Replace('\', '/'))/training/models"

[gpu]
enabled = true
device = "cuda"
memory_fraction = 0.9

[services]
# Background services (Windows-specific)
embedding_service = true
context_agent = true
autonomy_daemon = false  # Set to true to enable autonomous agents

[api]
# API keys (set in environment variables)
# ANTHROPIC_API_KEY
# GEMINI_API_KEY
# OPENAI_API_KEY
"@

$configPath = Join-Path $HafsRoot "hafs.toml"
$hafsConfig | Out-File -FilePath $configPath -Encoding UTF8
Write-Host "  ✓ hafs.toml created" -ForegroundColor Green
Write-Host ""

# [6] Test installation
Write-Host "[6/7] Testing installation..." -ForegroundColor Green
try {
    $env:PYTHONPATH = Join-Path $HafsRoot "src"
    $testResult = python -c "import hafs; print('hafs import successful')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ hafs imports successfully" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  Import test failed: $testResult" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ⚠️  Import test error: $_" -ForegroundColor Yellow
}
Write-Host ""

# [7] Setup Windows services (optional)
Write-Host "[7/7] Windows service setup..." -ForegroundColor Green
Write-Host "  To run hafs agents as Windows services, see:" -ForegroundColor Gray
Write-Host "  docs/windows/WINDOWS_SERVICES.md" -ForegroundColor Gray
Write-Host ""

# Summary
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Installation Summary:" -ForegroundColor Yellow
Write-Host "  hafs: $HafsRoot"
Write-Host "  .context: $ContextRoot"
if ($UseDDrive) {
    $dDrive = Get-PSDrive -Name D
    $freeGB = [math]::Round($dDrive.Free / 1GB, 2)
    Write-Host "  D drive free: $freeGB GB" -ForegroundColor Green
}
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Set environment variables (API keys):"
Write-Host "     [Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'your-key', 'User')"
Write-Host "     [Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-key', 'User')"
Write-Host ""
Write-Host "  2. Test hafs CLI:"
Write-Host "     cd $HafsRoot"
Write-Host "     .\.venv\Scripts\python.exe -m hafs.cli --help"
Write-Host ""
Write-Host "  3. Run training campaign on D drive:"
Write-Host "     C:\\hafs_scawful\\scripts\\run_training_campaign.ps1 -Target 34500"
Write-Host ""
Write-Host "  4. Setup background services (optional):"
Write-Host "     See docs\windows\WINDOWS_SERVICES.md"
Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
