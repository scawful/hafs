#!/bin/bash
# Remote Training Environment Installation Helper
# Transfers scripts to a Windows GPU host and executes installation
# Usage: ./remote_install_training.sh [host]

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
REMOTE_HOST="${1:-mm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_TEMP_DIR="C:/Users/\$env:USERNAME/AppData/Local/Temp/hafs_training_setup"
REMOTE_INSTALL_PATH="D:/training"

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}Remote Training Environment Setup${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}Target: ${REMOTE_HOST}${NC}"
echo -e "${GREEN}Scripts: ${SCRIPT_DIR}${NC}\n"

# Check if SSH is available
if ! command -v ssh &> /dev/null; then
    echo -e "${RED}ERROR: ssh command not found${NC}"
    exit 1
fi

# Test connection
echo -e "${CYAN}[1/5] Testing SSH connection to ${REMOTE_HOST}...${NC}"
if ssh -o ConnectTimeout=10 -q "${REMOTE_HOST}" exit; then
    echo -e "${GREEN}✓ Connection successful${NC}\n"
else
    echo -e "${RED}✗ Connection failed${NC}"
    echo -e "${YELLOW}Make sure Tailscale is running and ${REMOTE_HOST} is accessible${NC}"
    exit 1
fi

# Create remote temp directory
echo -e "${CYAN}[2/5] Creating remote temporary directory...${NC}"
ssh "${REMOTE_HOST}" "powershell -Command \"New-Item -ItemType Directory -Force -Path '${REMOTE_TEMP_DIR}' | Out-Null\""
echo -e "${GREEN}✓ Directory created${NC}\n"

# Transfer installation script
echo -e "${CYAN}[3/5] Transferring installation scripts...${NC}"
scp "${SCRIPT_DIR}/install_training_env.ps1" \
    "${REMOTE_HOST}:${REMOTE_TEMP_DIR}/install_training_env.ps1"
echo -e "${GREEN}✓ install_training_env.ps1 transferred${NC}"

scp "${SCRIPT_DIR}/test_training_setup.py" \
    "${REMOTE_HOST}:${REMOTE_TEMP_DIR}/test_training_setup.py"
echo -e "${GREEN}✓ test_training_setup.py transferred${NC}\n"

# Display pre-installation info
echo -e "${CYAN}[4/5] Checking remote system...${NC}"
ssh "${REMOTE_HOST}" "powershell -Command \"
    Write-Host 'Hostname: ' -NoNewline; hostname
    Write-Host 'OS: ' -NoNewline; (Get-WmiObject Win32_OperatingSystem).Caption
    Write-Host 'Python: ' -NoNewline; python --version 2>&1
    Write-Host 'Free Space (D:): ' -NoNewline; (Get-PSDrive D).Free / 1GB | ForEach-Object { '{0:N2} GB' -f \$_ }
\""
echo ""

# Ask for confirmation
echo -e "${YELLOW}Ready to install PyTorch/Unsloth training environment${NC}"
echo -e "${YELLOW}Installation path: ${REMOTE_INSTALL_PATH}${NC}"
read -p "Continue with installation? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Installation cancelled${NC}"
    exit 0
fi

# Run installation
echo -e "\n${CYAN}[5/5] Running installation on ${REMOTE_HOST}...${NC}"
echo -e "${YELLOW}This may take 10-20 minutes depending on internet speed...${NC}\n"

ssh "${REMOTE_HOST}" "powershell -ExecutionPolicy Bypass -File '${REMOTE_TEMP_DIR}/install_training_env.ps1' -InstallPath '${REMOTE_INSTALL_PATH}'"

# Check installation status
INSTALL_STATUS=$?

if [ $INSTALL_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    echo -e "${CYAN}Next steps:${NC}"
    echo -e "1. SSH to the machine: ${GREEN}ssh ${REMOTE_HOST}${NC}"
    echo -e "2. Validate installation: ${GREEN}python ${REMOTE_TEMP_DIR}/test_training_setup.py${NC}"
    echo -e "3. Check GPU: ${GREEN}nvidia-smi${NC}"
    echo -e "4. View quick start: ${GREEN}type ${REMOTE_INSTALL_PATH}\\QUICK_START.txt${NC}\n"

    echo -e "${CYAN}Training directories created at:${NC}"
    echo -e "  ${GREEN}${REMOTE_INSTALL_PATH}${NC}\n"

else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}Installation failed!${NC}"
    echo -e "${RED}========================================${NC}\n"

    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "1. Check the error messages above"
    echo -e "2. Verify Python is installed: ${GREEN}ssh ${REMOTE_HOST} python --version${NC}"
    echo -e "3. Check internet connection on remote machine"
    echo -e "4. Try running manually: ${GREEN}ssh ${REMOTE_HOST}${NC}"
    echo -e "   Then: ${GREEN}powershell -ExecutionPolicy Bypass -File '${REMOTE_TEMP_DIR}/install_training_env.ps1'${NC}\n"
fi

# Optional: Keep temp directory or clean up
read -p "Clean up temporary files on remote? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    ssh "${REMOTE_HOST}" "powershell -Command \"Remove-Item -Recurse -Force '${REMOTE_TEMP_DIR}' -ErrorAction SilentlyContinue\""
    echo -e "${GREEN}✓ Temporary files cleaned up${NC}"
else
    echo -e "${YELLOW}Temporary files kept at: ${REMOTE_TEMP_DIR}${NC}"
fi

echo -e "\n${CYAN}Installation script completed${NC}"
exit $INSTALL_STATUS
