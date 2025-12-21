# GPU-Accelerated Training Data Generation

Use your local GPU (medical-mechanica 5060TI 16GB) for training data generation with intelligent cost optimization.

## Quick Start

### Option 1: Hybrid Mode - GPU + API (RECOMMENDED ⭐)

**Intelligently routes between GPU (free) and Gemini API (paid) based on GPU load.**

```bash
cd ~/Code/hafs
./scripts/launch_hybrid_training.sh 34500
```

**How it works:**
- 🟢 **GPU <70% utilized** → Route to GPU (FREE)
- 🟡 **GPU 70-90% utilized** → Mix GPU + API (optimize cost)
- 🔴 **GPU >90% utilized** → Route to API only (prevent overload)

**Benefits:**
- ✅ **50-80% cost savings** vs API-only
- ✅ Maximizes FREE GPU usage
- ✅ Prevents GPU overload
- ✅ Maintains high throughput
- ✅ Automatic failover if GPU unavailable

### Option 2: GPU-Only Acceleration (FREE, slower startup)

```bash
cd ~/Code/hafs
PYTHONPATH=src .venv/bin/python -m agents.training.scripts.medical_mechanica_accelerator \
  --target 34500
```

**Benefits:**
- ✅ FREE (no API costs)
- ✅ Uses GPU on medical-mechanica
- ✅ Models: qwen3-14b, deepseek-r1:8b, gemma3:4b
- ⚠️ Requires Tailscale connection to medical-mechanica
- ⚠️ Requires Ollama running on Windows

### Option 2: Gemini API (PAID, faster startup)

```bash
cd ~/Code/hafs
PYTHONPATH=src .venv/bin/python -m agents.training.scripts.generate_campaign \
  --target 34500 \
  --resume
```

**Benefits:**
- ✅ Fast startup (no GPU warmup)
- ✅ Reliable (cloud-based)
- ❌ Costs money (API usage)
- ✅ Parallelism limited by API quota (1000 RPM, 4M TPM)

## Configuration

**File:** `config/training_medical_mechanica.toml`

```toml
[campaign]
parallel_workers = 100  # Increased for maximum throughput
max_concurrent_generations = 100  # Parallel streams

[gpu]
enabled = true
use_local_models = true
ollama_host = "100.104.53.21"  # Tailscale IP
ollama_port = 11435
models = ["qwen3-14b"]
fallback_to_gemini = true  # Fallback if GPU unavailable

[api]
gemini_rpm = 1000  # Max requests per minute
gemini_tpm = 4000000  # Max tokens per minute
```

## How Hybrid Mode Works

The hybrid orchestrator monitors GPU utilization every 5 seconds and makes intelligent routing decisions:

```python
# Routing Logic
if gpu_utilization < 70%:
    route_to_gpu()  # FREE!
elif gpu_utilization < 90%:
    # Probabilistic routing based on load
    # 70% util → 50% GPU, 50% API
    # 80% util → 25% GPU, 75% API
    # 90% util → 0% GPU, 100% API
    route_based_on_probability()
else:
    route_to_api()  # Prevent GPU overload
```

**Real-world example:**
- GPU at 60% → 100% requests to GPU (saving 100%)
- GPU at 80% → 50% GPU, 50% API (saving 50%)
- GPU at 95% → 0% GPU, 100% API (protecting hardware)

## Performance Comparison

| Method | Speed | Cost | GPU Usage | Requires |
|--------|-------|------|-----------|----------|
| **Hybrid (RECOMMENDED)** | ~40-50 samples/min | $ | 60-80% avg | Tailscale + Ollama + API |
| **GPU-Only** | ~10-15 samples/min | $0 | 100% | Tailscale + Ollama |
| **Gemini API (10 workers)** | ~6 samples/min | $$$ | 0% | API key |
| **Gemini API (100 workers)** | ~60 samples/min | $$$$ | 0% | API key + high quota |

## Checking GPU Connection

```bash
# Test connection to medical-mechanica
cd ~/Code/hafs
PYTHONPATH=src .venv/bin/python -m agents.training.scripts.medical_mechanica_accelerator --test

# Check Ollama status
hafs llamacpp status

# Check medical-mechanica health
hafs nodes status medical-mechanica
```

## Troubleshooting

### "NodeManager not initialized"
- Ensure `hafs.toml` has the `[[training.nodes]]` section configured
- Check that medical-mechanica is listed

### "Connection failed"
1. Ensure Tailscale is connected: `tailscale status`
2. Ensure Ollama is running on Windows: `ssh medical-mechanica 'ollama serve'`
3. Test connection: `curl http://100.104.53.21:11435/api/tags`

### "Models not found"
Pull models on medical-mechanica:
```bash
ssh medical-mechanica
ollama pull qwen3:14b
ollama pull deepseek-r1:8b
ollama pull gemma3:4b
```

## Monitoring

```bash
# Campaign status
hafs training status

# Watch logs
hafs training logs -f

# GPU usage (on medical-mechanica)
ssh medical-mechanica 'nvidia-smi -l 1'

# Check dataset output
ls -lh ~/.context/training/datasets/
```

## Domain-Specific Quality Thresholds

The campaign now uses domain-aware quality thresholds (fixed 2025-12-21):

```toml
[campaign.thresholds]
asm = 0.4        # ASM is hard - lower threshold
gigaleak = 0.45  # Original source
oracle = 0.4     # ROM hack
yaze = 0.5       # C++ code
cpp = 0.5
errors = 0.3     # Lowest
text = 0.6       # Natural language - highest
```

## Expected Timeline

**34,500 samples:**
- **GPU Acceleration:** ~38-57 hours
- **Gemini (10 workers):** ~96 hours
- **Gemini (100 workers):** ~10 hours (if API quota allows)

**Checkpoints:** Every 100 samples
**Resume:** Use `--resume` to continue from last checkpoint

## Next Steps After Generation

Once dataset is generated:

1. **Export to training format**
   ```bash
   hafs training export --dataset ~/.context/training/datasets/alttp_yaze_full_34500_*
   ```

2. **Start model training on GPU**
   ```bash
   ./scripts/deploy_training_medical_mechanica.sh train
   ```

3. **Monitor training**
   ```bash
   ssh medical-mechanica 'tail -f D:/hafs_training/logs/training_*.log'
   ```

## Config Files

- `config/training_medical_mechanica.toml` - GPU server config
- `hafs.toml` - Main config with node definitions
- `~/.context/training/quality_feedback.json` - Quality tracking
- `~/.context/training/*_checkpoint.json` - Resume points

---

**Last Updated:** 2025-12-21
**Status:** GPU acceleration script available, config updated for 100x parallelism
**Running:** Campaign active with domain-specific quality thresholds (0.4 for ASM)
