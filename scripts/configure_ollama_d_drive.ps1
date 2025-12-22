# Configure Ollama to use D drive for models
# This saves C drive space and puts models on the larger D drive

Write-Host "Configuring Ollama to use D drive..." -ForegroundColor Cyan

# Create models directory on D drive
$ModelsDir = "D:\ollama\models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
Write-Host "Created directory: $ModelsDir" -ForegroundColor Green

# Set environment variable for current session
$env:OLLAMA_MODELS = $ModelsDir
Write-Host "Set OLLAMA_MODELS=$ModelsDir for current session" -ForegroundColor Green

# Set environment variable system-wide (requires admin)
try {
    [System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $ModelsDir, 'Machine')
    Write-Host "Set OLLAMA_MODELS system-wide (permanent)" -ForegroundColor Green
} catch {
    Write-Host "Warning: Could not set system-wide (may need admin). Setting user-level..." -ForegroundColor Yellow
    [System.Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $ModelsDir, 'User')
}

# Check if Ollama service exists and restart it
Write-Host "`nRestarting Ollama service..." -ForegroundColor Cyan
try {
    $service = Get-Service -Name "Ollama" -ErrorAction SilentlyContinue
    if ($service) {
        Restart-Service -Name "Ollama" -Force
        Write-Host "Ollama service restarted" -ForegroundColor Green
        Start-Sleep -Seconds 3
    } else {
        Write-Host "Ollama service not found. Restart Ollama manually." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not restart service. Restart Ollama manually." -ForegroundColor Yellow
}

# Verify configuration
Write-Host "`nConfiguration:" -ForegroundColor Cyan
Write-Host "OLLAMA_MODELS = $env:OLLAMA_MODELS"
Write-Host "`nDisk space on D:" -ForegroundColor Cyan
Get-PSDrive D | Select-Object Used,Free | Format-Table -AutoSize

Write-Host "`nNOTE: Existing models on C drive will need to be re-downloaded." -ForegroundColor Yellow
Write-Host "Or manually move from C:\Users\$env:USERNAME\.ollama\models to $ModelsDir" -ForegroundColor Yellow
