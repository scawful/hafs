# GPU Limitations & Workarounds

**Last Updated**: 2025-12-22

## ✓ RTX 50-Series (sm_120) NOW SUPPORTED

### Solution: PyTorch 2.7+ with CUDA 12.8

NVIDIA RTX 50-series GPUs (RTX 5060 Ti, RTX 5070, RTX 5080, RTX 5090) have CUDA compute capability **sm_120** (Blackwell architecture), which **IS NOW SUPPORTED** by PyTorch 2.7+ with CUDA 12.8.

**Working Configuration**:
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB
- OS: Windows 11
- Python: 3.12.8 (recommended for wheel availability)
- PyTorch: 2.9.1+cu128 (stable release)
- CUDA: 12.8

### Installation

**1. Install Python 3.12** (recommended for better package compatibility):
```bash
# Download and install Python 3.12.8
curl -o python-3.12-installer.exe https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe
python-3.12-installer.exe /quiet InstallAllUsers=0 PrependPath=1
```

**2. Create Virtual Environment**:
```bash
python -m venv pytorch_env

# Windows
.\pytorch_env\Scripts\activate.bat

# Linux/macOS
source pytorch_env/bin/activate
```

**3. Install PyTorch with CUDA 12.8**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**4. Verify Installation**:
```python
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")  # Should be (12, 0)
```

Expected output:
```
PyTorch Version: 2.9.1+cu128
CUDA available: True
GPU Name: NVIDIA GeForce RTX 5060 Ti
GPU Capability: (12, 0)
```

### Alternative: PyTorch Nightly

For bleeding-edge features:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade --force-reinstall
```

### ⚠️ Previous Issues (RESOLVED)

**Old setup that DIDN'T work:**
- PyTorch 2.7.0.dev20250310+cu124 (nightly with CUDA 12.4)
- Error: `CUDA error: no kernel image is available for execution on the device`
- Reason: Required CUDA 12.8+ for sm_120 support

**Solution:** Use cu128 wheels (CUDA 12.8+) instead of cu124

### Unsloth Compatibility

Unsloth also fails on RTX 50-series:
```python
NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.
```

This is because Unsloth relies on PyTorch's CUDA support, which doesn't recognize sm_120.

## Mac M1/M2/M3 MPS Limitations

### Memory Constraints

Mac MPS uses unified memory shared with CPU:
- M1 Max: 32GB total, ~20GB available for MPS
- M1 Pro: 16GB total, ~13GB available for MPS
- M2 Max: 96GB total, ~80GB available for MPS

**Out of Memory Error** (Qwen2.5-Coder-1.5B with LoRA):
```
RuntimeError: MPS backend out of memory (MPS allocated: 19.17 GiB, other allocations: 204.69 MiB, max allowed: 20.13 GiB).
Tried to allocate 1.16 GiB on private pool.
```

### Solutions for Mac OOM

1. **Reduce batch size**:
   ```python
   per_device_train_batch_size=1  # Already at minimum
   gradient_accumulation_steps=4  # Reduce to 2
   ```

2. **Reduce sequence length**:
   ```python
   max_length=1024  # Instead of 2048
   ```

3. **Disable padding**:
   ```python
   padding=False  # Dynamic padding instead of max_length
   ```

4. **Use smaller LoRA rank**:
   ```python
   r=8  # Instead of 16
   ```

5. **Use 8-bit training**:
   ```python
   load_in_8bit=True  # Requires bitsandbytes
   ```

## Recommended Training Strategy

### For Small Models (< 3B parameters)

**Option 1: Mac M1/M2 with optimized settings**
```python
# Optimized for 20GB MPS
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,  # Effective batch = 2
    max_seq_length=1024,  # Reduced from 2048
    fp16=True,
    gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=8,  # Reduced from 16
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

**Option 2: Cloud GPU (RunPod RTX 4090)**
- Faster training (~15 min)
- Cost: ~$0.40-0.80 total
- Full LoRA rank and sequence length

### For Medium Models (3-7B parameters)

**Required: Cloud GPU**
- RTX 4090 (24GB) for 7B models
- A100 (40GB) for larger LoRA configurations

### For Large Models (7B+ parameters)

**Required: Multi-GPU or large single GPU**
- A100 80GB
- H100
- Multi-GPU setup with DeepSpeed

## Status Tracking

| GPU | CUDA Cap | PyTorch Support | Training Ready | Notes |
|-----|----------|-----------------|----------------|-------|
| **RTX 5060 Ti** | sm_120 | ✓ Yes (2.9.1+cu128) | ✓ **Ready** | Requires PyTorch with CUDA 12.8+ |
| RTX 4090 | sm_89 | ✓ Yes | ✓ Ready | Stable |
| RTX 3090 | sm_86 | ✓ Yes | ✓ Ready | Stable |
| Mac M1 MPS | N/A | ✓ Yes | ✓ Limited | 20GB memory constraint |
| Mac M2 MPS | N/A | ✓ Yes | ✓ Limited | Memory varies by config |

## Future Updates

Check this document for updates when:
1. PyTorch adds sm_120 support
2. Unsloth adds Mac MPS support
3. New workarounds are discovered

**Last Checked**: 2025-12-22
