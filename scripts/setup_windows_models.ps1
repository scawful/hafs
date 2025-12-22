# Setup Additional Models on medical-mechanica (Windows GPU)
# Run this via: ssh medical-mechanica 'powershell -File D:\hafs_training\scripts\setup_windows_models.ps1'

Write-Host "Setting up additional models on medical-mechanica..." -ForegroundColor Cyan

# Check available disk space
$disk = Get-PSDrive D
$freeGB = [math]::Round($disk.Free / 1GB, 2)
Write-Host "Available space on D: ${freeGB}GB" -ForegroundColor Yellow

# Coding specialists (HIGHEST PRIORITY for your use case)
Write-Host "`n=== Installing Code-Specialized Models ===" -ForegroundColor Green

ollama pull qwen2.5-coder:14b       # 9GB - Best for general code
ollama pull qwen2.5-coder:7b        # 5GB - Faster alternative
ollama pull qwen2.5-coder:32b       # 19GB - Highest quality (if space allows)

ollama pull deepseek-coder:6.7b     # 4GB - Fast, good for ASM
ollama pull deepseek-coder:33b      # 19GB - Excellent for low-level code

# Reasoning models (for complex logic)
Write-Host "`n=== Installing Reasoning Models ===" -ForegroundColor Green

ollama pull qwen2.5:14b             # 9GB - General purpose with reasoning
ollama pull deepseek-r1:8b          # 5GB - Faster reasoning (you have 14b already)
ollama pull deepseek-r1:32b         # 19GB - Best reasoning

# Efficient alternatives (for speed)
Write-Host "`n=== Installing Fast Alternatives ===" -ForegroundColor Green

ollama pull phi3.5:latest           # 2GB - Microsoft's efficient model
ollama pull mistral:7b              # 4GB - Fast, general purpose
ollama pull llama3.2:3b             # 2GB - Very fast for simple tasks

# Large context models (for analyzing big files)
Write-Host "`n=== Installing Large Context Models ===" -ForegroundColor Green

ollama pull qwen2.5:72b             # 41GB - HUGE, best quality (OPTIONAL)
ollama pull llama3.3:70b            # 40GB - Meta's flagship (OPTIONAL)

Write-Host "`n=== Current Models ===" -ForegroundColor Cyan
ollama list

Write-Host "`n=== Disk Usage ===" -ForegroundColor Yellow
$disk = Get-PSDrive D
$usedGB = [math]::Round($disk.Used / 1GB, 2)
$freeGB = [math]::Round($disk.Free / 1GB, 2)
Write-Host "Used: ${usedGB}GB | Free: ${freeGB}GB"

Write-Host "`nSetup complete!" -ForegroundColor Green
