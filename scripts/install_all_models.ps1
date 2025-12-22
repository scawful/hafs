# Install all recommended models + latest Gemma models
# Ensures D drive is used for storage

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing All Models on D Drive" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Configure D drive
Write-Host "[1/3] Configuring Ollama to use D drive..." -ForegroundColor Green
$ModelsDir = "D:\ollama\models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
$env:OLLAMA_MODELS = $ModelsDir

# Set permanently
try {
    [System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $ModelsDir, 'Machine')
} catch {
    [System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $ModelsDir, 'User')
}

Write-Host "Models will be stored in: $ModelsDir" -ForegroundColor Yellow
Write-Host ""

# Step 2: Check disk space
Write-Host "[2/3] Checking disk space..." -ForegroundColor Green
$disk = Get-PSDrive D
$freeGB = [math]::Round($disk.Free / 1GB, 2)
Write-Host "Available space: ${freeGB}GB" -ForegroundColor Yellow

if ($freeGB -lt 100) {
    Write-Host "WARNING: Less than 100GB free. May need to free up space." -ForegroundColor Red
}
Write-Host ""

# Step 3: Install models
Write-Host "[3/3] Installing models..." -ForegroundColor Green
Write-Host ""

# Code specialists (PRIORITY)
Write-Host "=== Code Specialists ===" -ForegroundColor Cyan
$codeModels = @(
    @{name="qwen2.5-coder:14b"; size="9GB"; desc="Best for C++/Python code"},
    @{name="qwen2.5-coder:7b"; size="5GB"; desc="Faster alternative"},
    @{name="qwen2.5-coder:32b"; size="19GB"; desc="Highest quality (optional)"},
    @{name="deepseek-coder:6.7b"; size="4GB"; desc="ASM specialist"},
    @{name="deepseek-coder:33b"; size="19GB"; desc="Best for ASM (optional)"}
)

foreach ($model in $codeModels) {
    Write-Host "Pulling $($model.name) ($($model.size)) - $($model.desc)" -ForegroundColor White
    ollama pull $model.name
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Installed $($model.name)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to install $($model.name)" -ForegroundColor Red
    }
}
Write-Host ""

# Fast/efficient models
Write-Host "=== Fast Alternatives ===" -ForegroundColor Cyan
$fastModels = @(
    @{name="phi3.5:latest"; size="2GB"; desc="Microsoft efficient model"},
    @{name="mistral:7b"; size="4GB"; desc="Fast general purpose"},
    @{name="llama3.2:3b"; size="2GB"; desc="Very fast"}
)

foreach ($model in $fastModels) {
    Write-Host "Pulling $($model.name) ($($model.size)) - $($model.desc)" -ForegroundColor White
    ollama pull $model.name
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Installed $($model.name)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to install $($model.name)" -ForegroundColor Red
    }
}
Write-Host ""

# Latest Gemma models (ALL OF THEM)
Write-Host "=== Latest Gemma Models ===" -ForegroundColor Cyan
$gemmaModels = @(
    @{name="gemma2:9b"; size="5GB"; desc="Latest Gemma 2 (9B)"},
    @{name="gemma2:27b"; size="16GB"; desc="Latest Gemma 2 (27B)"},
    @{name="gemma2:2b"; size="2GB"; desc="Latest Gemma 2 (2B tiny)"},
    @{name="codegemma:7b"; size="5GB"; desc="Code-specialized Gemma"},
    @{name="codegemma:2b"; size="2GB"; desc="Tiny code Gemma"},
    @{name="embeddinggemma:latest"; size="600MB"; desc="Embedding model"},
    @{name="functiongemma:latest"; size="300MB"; desc="Function calling"}
)

foreach ($model in $gemmaModels) {
    Write-Host "Pulling $($model.name) ($($model.size)) - $($model.desc)" -ForegroundColor White
    ollama pull $model.name
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Installed $($model.name)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to install $($model.name)" -ForegroundColor Red
    }
}
Write-Host ""

# Reasoning models
Write-Host "=== Reasoning Models ===" -ForegroundColor Cyan
$reasoningModels = @(
    @{name="qwen2.5:14b"; size="9GB"; desc="General reasoning"},
    @{name="deepseek-r1:8b"; size="5GB"; desc="Fast reasoning"},
    @{name="deepseek-r1:32b"; size="19GB"; desc="Best reasoning (optional)"}
)

foreach ($model in $reasoningModels) {
    Write-Host "Pulling $($model.name) ($($model.size)) - $($model.desc)" -ForegroundColor White
    ollama pull $model.name
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Installed $($model.name)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to install $($model.name)" -ForegroundColor Red
    }
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Installed models:" -ForegroundColor Green
ollama list

Write-Host "`nDisk usage:" -ForegroundColor Yellow
$disk = Get-PSDrive D
$usedGB = [math]::Round($disk.Used / 1GB, 2)
$freeGB = [math]::Round($disk.Free / 1GB, 2)
Write-Host "D: drive - Used: ${usedGB}GB | Free: ${freeGB}GB"

Write-Host "`nModels location: $env:OLLAMA_MODELS" -ForegroundColor Cyan
if (Test-Path $ModelsDir) {
    $modelsDirSize = [math]::Round((Get-ChildItem $ModelsDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
    Write-Host "Models directory size: ${modelsDirSize}GB" -ForegroundColor Yellow
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Test models: curl http://localhost:11435/api/tags"
Write-Host "  2. Run hybrid campaign: python -m agents.training.scripts.hybrid_campaign --pilot --target 100"
Write-Host "  3. Configure domain routing in config/training_medical_mechanica.toml"
Write-Host ""
