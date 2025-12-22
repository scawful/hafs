# Model Testing Guide

Comprehensive testing workflow for newly trained Oracle expert models.

## Quick Reference

```bash
# 1. Register model
hafs models register oracle-farore-general-qwen25-coder-15b-20251222 \
  --role general --dataset oracle_farore_improved \
  --location windows --loss [FINAL_LOSS]

# 2. Quick test (PyTorch)
python3 scripts/test_trained_model.py \
  --model-path "D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222" \
  --prompt "Explain the secrets system in Oracle of Secrets"

# 3. Deploy to Ollama
hafs models deploy oracle-farore-general-qwen25-coder-15b-20251222 ollama \
  --name oracle-farore --quantization Q4_K_M

# 4. Test via Ollama
hafs models test oracle-farore-general-qwen25-coder-15b-20251222 ollama
```

---

## Phase 1: Registration & Basic Testing

### 1.1 Register the Model

First, extract training metrics from the log:

```bash
# Get final loss from training log
tail -50 /tmp/oracle_farore_training.log | grep -E "loss|perplexity"

# Register model
hafs models register oracle-farore-general-qwen25-coder-15b-20251222 \
  --display-name "Oracle: Farore Secrets" \
  --role general \
  --base-model "Qwen/Qwen2.5-Coder-1.5B" \
  --dataset oracle_farore_improved_20251222_040629 \
  --location windows \
  --path "D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222" \
  --loss [FINAL_LOSS] \
  --format pytorch
```

### 1.2 Quick PyTorch Test

Test directly from the checkpoint (fastest, no conversion):

```python
# scripts/test_trained_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "D:/hafs_training/models/oracle-farore-general-qwen25-coder-15b-20251222"
base_model = "Qwen/Qwen2.5-Coder-1.5B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load base model + LoRA adapter
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base, model_path)

# Test prompts
prompts = [
    "Explain the secrets system in Oracle of Secrets",
    "How does the ring menu work in Oracle ROM hacks?",
    "Write ASM code to add a new item to the inventory",
    "What are the main differences between Oracle of Ages and Oracle of Secrets?",
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}\n{'-'*80}")
```

**Run:**
```bash
ssh medical-mechanica "cd C:/hafs && python scripts/test_trained_model.py"
```

---

## Phase 2: Deployment Testing

### 2.1 Deploy to Ollama (Recommended)

Ollama is easiest for interactive testing:

```bash
# Pull model to Mac (via mount or SSH)
hafs models pull oracle-farore-general-qwen25-coder-15b-20251222 \
  --source windows --dest mac

# Convert to GGUF + deploy
hafs models deploy oracle-farore-general-qwen25-coder-15b-20251222 ollama \
  --name oracle-farore \
  --quantization Q4_K_M
```

**Test interactively:**
```bash
ollama run oracle-farore "Explain how to add a new secret to Oracle of Secrets"
```

### 2.2 Deploy to llama.cpp (Performance Testing)

For benchmarking inference speed:

```bash
# Convert to GGUF
hafs models convert oracle-farore-general-qwen25-coder-15b-20251222 \
  --format gguf --quantization Q4_K_M

# Test with llama.cpp
llama-cli -m ~/.context/models/oracle-farore-general-qwen25-coder-15b-20251222-Q4_K_M.gguf \
  -p "Explain the secrets system in Oracle of Secrets" \
  --temp 0.7 -n 256
```

---

## Phase 3: Quality Evaluation

### 3.1 Domain-Specific Test Suite

Test across all training domains:

```bash
# Create test suite
cat > /tmp/oracle_farore_test_suite.txt <<'EOF'
# General Knowledge
1. What is Oracle of Secrets?
2. How does the secrets system work?
3. What are the main features of the Oracle ROM hacks?

# Game Design
4. Explain the ring menu system
5. How do you design a new dungeon for Oracle of Secrets?
6. What are the key differences between vanilla ALTTP and Oracle hacks?

# Technical/ASM
7. Write ASM code to hook the item collection routine
8. How do you modify the ROM to add a new secret?
9. Explain the memory layout for secrets storage

# YAZE Integration
10. How do you use YAZE to edit Oracle of Secrets?
11. What's the workflow for adding custom sprites?
12. Explain the tilemap format for Oracle dungeons
EOF

# Run test suite
python3 scripts/run_test_suite.py \
  --model oracle-farore \
  --suite /tmp/oracle_farore_test_suite.txt \
  --output oracle_farore_eval_results.json
```

### 3.2 Comparison Testing

Compare oracle-farore vs oracle-rauru vs base model:

```python
# scripts/compare_models.py
models = [
    "Qwen/Qwen2.5-Coder-1.5B",  # Base
    "oracle-rauru",              # ASM specialist
    "oracle-farore",             # General Oracle expert
]

test_prompts = [
    "Write ASM code for a JSR instruction",  # ASM (rauru should excel)
    "Explain the Oracle secrets system",      # General (farore should excel)
    "How do you add a new item?",            # Both should do well
]

for prompt in test_prompts:
    print(f"\n{'='*80}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*80}\n")

    for model in models:
        response = query_model(model, prompt)
        print(f"\n[{model}]:")
        print(response)
        print(f"\nScore: {score_response(response, prompt)}")
```

### 3.3 Perplexity Evaluation

Measure model confidence on test data:

```bash
# Evaluate on held-out test set
python3 scripts/evaluate_perplexity.py \
  --model oracle-farore \
  --test-data ~/.context/training/datasets/oracle_farore_test.jsonl \
  --output oracle_farore_perplexity.json
```

---

## Phase 4: Regression Testing

### 4.1 Test Against Known Issues

If oracle-rauru had issues, test if oracle-farore avoids them:

```bash
# Test prompts that exposed previous model weaknesses
python3 scripts/regression_tests.py \
  --model oracle-farore \
  --known-issues docs/training/known_issues.json
```

### 4.2 Diversity Check

Ensure responses are diverse (not memorized):

```python
# Test same prompt 5 times with different temperatures
prompt = "Explain the secrets system"
temperatures = [0.3, 0.5, 0.7, 0.9, 1.0]

responses = []
for temp in temperatures:
    response = query_model("oracle-farore", prompt, temperature=temp)
    responses.append(response)

# Calculate diversity score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(responses)
similarity_matrix = cosine_similarity(vectors)

avg_similarity = similarity_matrix.sum() / (len(responses) * (len(responses) - 1))
diversity_score = 1 - avg_similarity

print(f"Diversity Score: {diversity_score:.3f} (higher is better)")
```

---

## Phase 5: Integration Testing

### 5.1 Test with hafs TUI

```bash
# Test in interactive chat mode
hafs chat --model oracle-farore

# Example conversation:
# User: "I want to add a new secret to Oracle of Secrets. How do I start?"
# oracle-farore: [should give comprehensive guide with ASM hooks, YAZE steps, etc.]
```

### 5.2 Test with Agent Workflow

```bash
# Test as part of agent workflow
hafs nodes chat --node oracle-farore --task "Design a new dungeon secret"
```

---

## Success Criteria

### Minimum Viable (Pass)
- ✓ Model loads without errors
- ✓ Generates coherent responses (not gibberish)
- ✓ Responds to Oracle-specific prompts correctly
- ✓ Perplexity < 3.0 on test set

### Target Quality (Good)
- ✓ Better than base model on Oracle prompts (subjective evaluation)
- ✓ Diverse responses (diversity score > 0.6)
- ✓ Correct ASM code generation
- ✓ No hallucination of non-existent Oracle features

### Stretch Goal (Excellent)
- ✓ Comparable to oracle-rauru on ASM tasks
- ✓ Superior to oracle-rauru on game design tasks
- ✓ Perplexity < 2.0 on test set
- ✓ Passes all regression tests

---

## Quick Start Testing Script

Here's a complete script to run basic tests:

```bash
#!/bin/bash
# scripts/quick_model_test.sh

MODEL_ID="oracle-farore-general-qwen25-coder-15b-20251222"
MODEL_NAME="oracle-farore"

echo "=== Quick Model Test: $MODEL_NAME ==="

# 1. Register (if not already)
hafs models info $MODEL_ID 2>/dev/null || {
    echo "Registering model..."
    hafs models register $MODEL_ID \
        --role general \
        --location windows
}

# 2. Deploy to Ollama
echo "Deploying to Ollama..."
hafs models deploy $MODEL_ID ollama --name $MODEL_NAME

# 3. Run test prompts
echo "Running test prompts..."
cat <<'PROMPTS' | while read prompt; do
    echo -e "\n=== PROMPT: $prompt ===\n"
    ollama run $MODEL_NAME "$prompt"
done
Explain the secrets system in Oracle of Secrets
Write ASM code to add a new item
How do you use YAZE to edit Oracle ROM hacks?
What makes Oracle of Secrets different from ALTTP?
PROMPTS

echo "=== Test Complete ==="
```

**Run:**
```bash
chmod +x scripts/quick_model_test.sh
./scripts/quick_model_test.sh
```

---

## Troubleshooting

### Model Won't Load
```bash
# Check adapter files exist
ls -la ~/Mounts/mm-d/hafs_training/models/oracle-farore-*/

# Verify adapter_config.json
cat ~/Mounts/mm-d/hafs_training/models/oracle-farore-*/adapter_config.json
```

### Poor Quality Responses

1. **Check training loss** - Should be < 1.0
2. **Verify dataset quality** - Review rejected samples
3. **Test with lower temperature** - Try 0.3 instead of 0.7
4. **Check for overfitting** - Compare train vs eval loss

### Conversion Fails

```bash
# Manual GGUF conversion
python3 ~/.local/bin/llama-cpp-python-convert.py \
  ~/Mounts/mm-d/hafs_training/models/oracle-farore-* \
  --outtype f16 \
  --outfile oracle-farore-f16.gguf

# Then quantize
llama-quantize oracle-farore-f16.gguf oracle-farore-Q4_K_M.gguf Q4_K_M
```

---

## Next Steps After Testing

**If model passes tests:**
1. Update model registry with eval metrics
2. Deploy to production (halext nodes)
3. Document performance characteristics
4. Add to agent tool selection

**If model needs improvement:**
1. Analyze failure modes
2. Improve dataset quality
3. Adjust training hyperparameters
4. Retrain with refined data
