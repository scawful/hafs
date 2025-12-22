# Switching Models for Training

Quick guide to test and switch models for training data generation.

## Available New Models (Installed)

### Reasoning Models
- **deepseek-r1:32b** - Best reasoning, slowest (19 GB)
- **deepseek-r1:14b** - Balanced reasoning (9 GB)
- **deepseek-r1:8b** - Fast reasoning (5.2 GB)

### Latest Generation
- **qwen3:14b** - Newest Qwen (9.3 GB)
- **gemma3:27b** - Largest Gemma (17 GB)
- **gemma3:12b** - Balanced Gemma (8.1 GB)

### Code Specialists
- **deepseek-coder:33b** - Best for code (18 GB)
- **qwen2.5-coder:32b** - Qwen code specialist (19 GB)
- **qwen2.5-coder:14b** - Current baseline (9 GB) ✅

## Quick Model Test

Test a model before committing to training:

```bash
cd ~/Code/hafs

# Quick shell test
./scripts/quick_model_test.sh

# Comprehensive comparison test
PYTHONPATH=src .venv/bin/python scripts/test_new_models.py
```

## Switch GPU Training Model

Update `src/agents/training/scripts/hybrid_campaign.py`:

```python
# Line 67 - Change the model
self._gpu_backend = OllamaBackend(
    host="100.104.53.21",
    port=11434,
    model="deepseek-r1:14b",  # <-- CHANGE HERE
    timeout=60.0,
)
```

**Model Recommendations:**
- **Speed:** `qwen2.5-coder:14b` (current) or `deepseek-r1:8b`
- **Quality:** `deepseek-r1:14b` or `deepseek-coder:33b`
- **Balance:** `qwen3:14b` or `deepseek-r1:14b`

## Switch Gemini API Model

Update `src/agents/training/scripts/hybrid_campaign.py`:

```python
# Line 88 - Change preferred model
all_models = [
    "gemini-3-flash-preview",    # Current (fastest)
    "gemini-2.0-flash-thinking-exp",  # Reasoning (new)
    "gemini-3-pro-preview",      # Most capable
    "gemini-2.5-flash",          # Balanced
]
```

Or change at launch:

```bash
# Edit hybrid_campaign.py line 217
hybrid_orch = HybridOrchestrator(
    gpu_monitor,
    load_balancer,
    preferred_model="gemini-2.0-flash-thinking-exp"  # <-- CHANGE HERE
)
```

## Model Selection Guide

### For ASM Training Data

**Best Quality:**
1. `deepseek-r1:14b` (reasoning about code logic)
2. `deepseek-coder:33b` (code specialist)
3. `qwen2.5-coder:32b` (current best alternative)

**Best Speed:**
1. `qwen2.5-coder:14b` (current baseline)
2. `deepseek-r1:8b` (fast reasoning)
3. `qwen3:14b` (latest, fast)

**Best Balance:**
1. `deepseek-r1:14b` (⭐ recommended)
2. `qwen3:14b`
3. `deepseek-coder:33b`

### For Oracle Training Data

**Best Quality:**
1. `gemma3:27b` (general knowledge)
2. `qwen3:14b` (balanced)
3. `deepseek-r1:14b`

**Best Speed:**
1. `gemma3:12b`
2. `qwen3:14b`
3. `qwen2.5:14b`

## Testing Methodology

1. **Run comparison test:**
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/test_new_models.py
   ```

2. **Check output for:**
   - Response quality (technical accuracy)
   - Speed (tokens/sec)
   - Detail level (words generated)

3. **Run mini A/B test:**
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/run_ab_test.py \
     --domain asm \
     --num-samples 10 \
     --model1 qwen2.5-coder:14b \
     --model2 deepseek-r1:14b
   ```

## Recommended Next Steps

### Test DeepSeek-R1 Reasoning
DeepSeek-R1 is designed to "think through" problems. For ASM code:

```bash
# Test if reasoning helps with code understanding
PYTHONPATH=src .venv/bin/python -c "
import asyncio
from hafs.services.local_ai_orchestrator import *

async def test():
    orch = LocalAIOrchestrator(
        ollama_url='http://100.104.53.21:11434',
        default_model='deepseek-r1:14b'
    )
    await orch.start()

    req = InferenceRequest(
        id='test',
        priority=RequestPriority.INTERACTIVE,
        prompt='Explain the 65816 REP instruction and why it is used before 16-bit operations.',
        model='deepseek-r1:14b',
    )

    result = await orch.submit_request(req)
    print(result.response)
    await orch.stop()

asyncio.run(test())
"
```

### Try Qwen3 (Latest)
Qwen3 just released - test if it's better than Qwen 2.5:

```python
# Update hybrid_campaign.py line 67
model="qwen3:14b",  # Latest Qwen
```

### Use DeepSeek-Coder for Code
For pure code generation (not explanations):

```python
model="deepseek-coder:33b",  # Best code specialist
```

## Rollback

If new model performs worse, revert to baseline:

```python
# src/agents/training/scripts/hybrid_campaign.py line 67
model="qwen2.5-coder:14b",  # Original baseline
```

## Notes

- All models use **baseline prompts** (from A/B test winner)
- GPU models run on Windows (100.104.53.21:11434)
- API models use Gemini (Google Cloud)
- Test before committing to full 34.5K sample run
