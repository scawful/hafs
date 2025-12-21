# medical-mechanica Setup Summary

**Date:** 2025-12-21
**System:** Windows 10 Pro 64-bit
**GPU:** NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
**CUDA:** 11.2

## ✅ Completed Setup

### 1. hafs Installation (C:/hafs)

- **Status:** ✅ Code deployed successfully
- **Size:** 28 MB (compressed tarball)
- **Method:** Rsync via mounted drives (~/Mounts/mm-c)
- **Python:** 3.14.0 with virtual environment at `.venv`
- **Dependencies:** Installed via `pip install -e .[backends]`

**Directory Structure:**
```
C:\hafs\
├── src\              # Python source code
├── config\           # Configuration files
├── scripts\          # Utility and setup scripts
├── docs\             # Documentation
├── hafs.toml         # Windows-specific config
└── .venv\            # Python virtual environment
```

### 2. .context Directory (D:/.context)

- **Status:** ✅ Created with full structure
- **Location:** D drive (1.56 TB free space)
- **Purpose:** AFS context storage for agents

**Structure:**
```
D:\.context\
├── knowledge\        # Knowledge bases
├── embeddings\       # Vector embeddings
├── logs\             # Agent logs
├── training\         # Training artifacts
│   ├── datasets\
│   ├── checkpoints\
│   └── models\
├── history\          # Agent history
├── scratchpad\       # Working memory
├── memory\           # Persistent memory
├── hivemind\         # Multi-agent coordination
└── README.md         # Documentation
```

### 3. Training Directory (D:/hafs_training)

- **Status:** ✅ Created for training workflows
- **Purpose:** Campaign datasets and model training

**Structure:**
```
D:\hafs_training\
├── datasets\         # Generated training datasets
├── checkpoints\      # Campaign checkpoints
├── logs\             # Training logs
├── models\           # Trained models
└── temp\             # Temporary files
```

### 4. Configuration Files

#### `hafs.toml` (Windows-specific)
✅ Created with D drive paths:
```toml
[paths]
context_root = "D:/.context"
training_datasets = "D:/hafs_training/datasets"
training_models = "D:/hafs_training/models"

[gpu]
enabled = true
device = "cuda"
memory_fraction = 0.9
```

#### `config/windows_background_agents.toml`
✅ Created background agent configuration:
- **Explorer** - Every 6 hours (Claude Sonnet)
- **Cataloger** - Every 12 hours (Claude Sonnet)
- **Context Builder** - Daily at 2 AM (GPT-4o)
- **Repo Updater** - Every 4 hours (Claude Haiku)
- **Mac Sync** - Daily at 1 AM (pull) and 11 PM (push)

**Key Design:** Uses Claude/OpenAI, reserves Gemini for training data generation

### 5. Documentation

✅ Created comprehensive Windows docs:

| File | Purpose |
|------|---------|
| `docs/windows/WINDOWS_SETUP.md` | Complete setup guide |
| `docs/windows/BACKGROUND_AGENTS.md` | Agent configuration and management |
| `D:/.context/README.md` | .context directory guide |

### 6. Scripts

✅ Created PowerShell automation:

| Script | Purpose |
|--------|---------|
| `scripts/setup_windows.ps1` | Initial hafs installation |
| `scripts/run_training_campaign.ps1` | Launch training campaigns |
| `scripts/setup_background_agents.ps1` | Configure Task Scheduler |
| `scripts/deploy_training_medical_mechanica.sh` | Deploy from Mac (bash) |

## ⚠️ Known Issues

### 1. PTY Module Error (Windows Incompatibility)

**Issue:** hafs CLI fails on Windows due to `termios` import (Unix-only module)

```
ModuleNotFoundError: No module named 'termios'
```

**Impact:**
- hafs CLI (`python -m hafs.cli`) not functional on Windows
- Background agents may have similar issues if they use PTY code
- Training scripts might work if they don't import CLI code

**Workaround Options:**

A) **Conditional Imports** (recommended):
```python
# In backends/cli/pty.py
import platform
if platform.system() != "Windows":
    import pty
    import termios
else:
    # Windows-specific alternatives or stub
    pty = None
    termios = None
```

B) **Windows-Specific Backend**:
- Create `backends/cli/windows_pty.py` using `winpty` or `pywinpty`
- Install: `pip install pywinpty`

C) **Skip CLI Mode on Windows**:
- Use API-based backends only (Anthropic, OpenAI, Gemini)
- Avoid Claude CLI backend on Windows

**Status:** 🔴 Needs fix before background agents can run

### 2. Missing Agent Implementation Files

Background agents configured but implementation files need to be created:
- `agents/background/explorer.py`
- `agents/background/cataloger.py`
- `agents/background/context_builder.py`
- `agents/background/repo_updater.py`
- `agents/background/mac_sync.py`

**Status:** 🟡 Implementation needed

## 📋 Next Steps (Priority Order)

### High Priority

1. **Fix PTY Import Issue**
   ```bash
   # Option 1: Conditional imports in pty.py
   # Option 2: Install pywinpty for Windows PTY support
   ```

2. **Set API Keys** (Required for agents)
   ```powershell
   [Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', 'sk-ant-...', 'User')
   [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-...', 'User')
   [Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'AIza...', 'User')
   ```

3. **Implement Background Agents**
   - Create agent implementation files
   - Test manually before scheduling
   - Deploy to Task Scheduler

### Medium Priority

4. **Test Training Campaign on Windows**
   ```powershell
   cd C:\hafs
   .\.venv\Scripts\Activate.ps1
   python -m agents.training.scripts.generate_campaign --target 100  # Small test
   ```

5. **Setup Mac-Windows Sync**
   - Configure SSH keys for passwordless access
   - Test rsync over SSH
   - Verify mounted drives remain connected

6. **Monitor Mac Campaign**
   - Check PID 13516 progress (34.5K samples)
   - Expected completion: ~2025-12-23
   - Export datasets when complete

### Low Priority

7. **Deploy Training to GPU**
   - After datasets exported from Mac campaign
   - Train `oracle-rauru-assembler` on RTX 5060 Ti
   - Train `oracle-yaze-expert`

8. **Background Service Optimization**
   - Tune agent schedules based on usage
   - Configure disk space alerts
   - Setup email notifications (optional)

## 🔧 Quick Commands

### Verify Installation

```powershell
# Check hafs directory
dir C:\hafs

# Check .context
dir D:\.context

# Check Python environment
C:\hafs\.venv\Scripts\python.exe --version

# Test imports (will fail due to PTY issue)
C:\hafs\.venv\Scripts\python.exe -c "import hafs"
```

### Access from Mac

```bash
# Via SSH
ssh Administrator@medical-mechanica

# Via mounted drives
ls ~/Mounts/mm-c/hafs
ls ~/Mounts/mm-d/.context
ls ~/Mounts/mm-d/hafs_training

# Copy files
cp ~/Code/hafs/new_file.py ~/Mounts/mm-c/hafs/src/
```

### Future Training Workflow

```powershell
# 1. Generate dataset (on Windows or Mac)
python -m agents.training.scripts.generate_campaign --target 34500

# 2. Train model (on Windows GPU)
python -m agents.training.scripts.train_model `
    --dataset D:\hafs_training\datasets\alttp_yaze_full_34500_asm.jsonl `
    --model-name oracle-rauru-assembler `
    --output-dir D:\hafs_training\models\oracle-rauru-assembler

# 3. Test model
python -m hafs.cli chat --model D:\hafs_training\models\oracle-rauru-assembler
```

## 📊 Storage Summary

| Location | Purpose | Size | Available |
|----------|---------|------|-----------|
| C:/hafs | Code + venv | ~200 MB | 149 GB free |
| D:/.context | AFS context | ~10 MB | 1.56 TB free |
| D:/hafs_training | Training artifacts | ~10 MB | 1.56 TB free |

**Estimated Capacity:**
- Training datasets: ~4,100 campaigns (380 MB each)
- Model checkpoints: ~150 models (10 GB each)
- Years of agent logs and history

## 🔗 Related Systems

### Mac Development Machine

- **Campaign Status:** Running (PID 13516)
- **Target:** 34,500 samples
- **Progress:** ~32/6900 samples (ASM domain)
- **ETA:** 8-12 hours
- **Log:** `~/.context/logs/campaign_34500_20251221_144848.log`

### Shared Context

- **Mac .context:** `/Users/scawful/.context`
- **Windows .context:** `D:/.context`
- **Sync:** Bidirectional via background agents
- **Mounted:** `~/Mounts/halext/.context` (from Windows perspective)

## 📝 Notes

1. **Provider Strategy:**
   - **Gemini:** Reserved for training data generation (teacher model)
   - **Claude:** Primary for background agents (exploration, cataloging)
   - **OpenAI:** Secondary for context building
   - **Rationale:** Conserve Gemini quota for high-value training work

2. **Windows vs Unix:**
   - Some hafs code assumes Unix environment (PTY, termios)
   - Need conditional imports or Windows-specific backends
   - Background agents should work if implemented without PTY deps

3. **GPU Utilization:**
   - RTX 5060 Ti 16GB ideal for 14B parameter models with LoRA
   - Can run inference and training locally
   - Consider fine-tuning Qwen 2.5 Coder 14B for ROM hacking

4. **Security:**
   - API keys in environment variables (User scope, persists)
   - Never commit .env files or credentials to git
   - Use Windows Credential Manager for sensitive data

---

## Summary

**What Works:**
- ✅ hafs code deployed to C:/hafs
- ✅ Python 3.14 + venv + dependencies installed
- ✅ .context structure created on D drive
- ✅ Training directories ready
- ✅ Configuration files in place
- ✅ Comprehensive documentation
- ✅ Mounted drives for Mac access

**What Needs Attention:**
- 🔴 Fix PTY import for Windows compatibility
- 🟡 Implement background agent scripts
- 🟡 Set API keys
- 🟡 Test training pipeline on Windows
- 🟡 Setup Task Scheduler

**Next Session:**
1. Fix Windows PTY compatibility issue
2. Implement background agents
3. Set API keys and test agents manually
4. Monitor Mac campaign progress
5. Deploy training to GPU after dataset export

**Estimated Time to Full Operational:**
- PTY fix: 30-60 minutes
- Agent implementation: 2-4 hours
- Testing and deployment: 1-2 hours
- **Total:** 4-7 hours of focused work

---

**Last Updated:** 2025-12-21 15:20 PST
**Setup By:** Claude Sonnet 4.5 via ~/Code/hafs
**System:** medical-mechanica (Windows 10 Pro, RTX 5060 Ti 16GB)
