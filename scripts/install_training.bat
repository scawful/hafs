@echo off
REM Quick installer batch file for Windows
REM Runs the PowerShell installation script
REM Usage: install_training.bat

echo ================================================================================
echo PyTorch/Unsloth Training Environment Installation
echo ================================================================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: PowerShell not found
    pause
    exit /b 1
)

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

echo Running installation script...
echo Script location: %SCRIPT_DIR%
echo.

REM Run the PowerShell script with execution policy bypass
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%install_training_env.ps1"

echo.
echo ================================================================================
echo Installation script completed
echo ================================================================================
echo.

pause
