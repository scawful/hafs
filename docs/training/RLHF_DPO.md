# RLHF and DPO Training Guide

## Overview

After supervised fine-tuning (SFT) with LoRA, you can further improve the model using human preferences:

- **RLHF:** Reinforcement Learning from Human Feedback (complex, traditional approach)
- **DPO:** Direct Preference Optimization (simpler, modern approach) ← **Recommended**

Both require preference data: pairs of (good, bad) outputs for the same input.

## DPO vs RLHF Comparison

| Aspect | RLHF (PPO) | DPO |
|--------|------------|-----|
| Complexity | High (reward model + RL) | Low (direct optimization) |
| Training Time | 2-3x longer | Same as SFT |
| Memory | 2x (reward model + policy) | 1x (just model) |
| Stability | Can diverge, needs tuning | Stable |
| Results | Slightly better at scale | Nearly equivalent |
| **Recommendation** | Large teams, huge scale | **Solo dev, practical** |

**TL;DR:** Use DPO unless you have a specific reason for RLHF.

## DPO: How It Works

### Intuition

Instead of training a separate reward model and using RL (RLHF), DPO directly optimizes the model to prefer good outputs over bad ones.

Given:
- Prompt: "Write a sprite DMA routine"
- Good output: Correct, efficient ASM code
- Bad output: Buggy or inefficient code

DPO increases probability of good output, decreases probability of bad output.

### Mathematical Formulation

DPO loss function:
```
L_DPO(θ) = -E [ log σ( β log π_θ(y_w|x) / π_ref(y_w|x) - β log π_θ(y_l|x) / π_ref(y_l|x) ) ]
```

Where:
- `y_w` = "winning" (preferred) output
- `y_l` = "losing" (rejected) output
- `π_θ` = your model being trained
- `π_ref` = reference model (frozen SFT model)
- `β` = temperature parameter (typically 0.1-0.5)

**In plain English:** Increase log-probability of winners, decrease log-probability of losers, relative to the reference model.

## Generating Preference Data

You need pairs of (chosen, rejected) outputs. Three approaches:

### 1. Best-of-N Sampling (Automatic)

Generate N completions, pick best via automatic scoring:

```python
def generate_preference_pairs(model, prompts, n=4):
    """Generate N outputs, pick best via automatic metrics."""
    pairs = []

    for prompt in prompts:
        # Generate N candidates
        candidates = []
        for _ in range(n):
            output = model.generate(prompt, temperature=0.8)
            score = evaluate_output(output)  # Automatic scoring
            candidates.append((output, score))

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Create pairs: best vs worst, 2nd best vs 2nd worst, etc.
        for i in range(n // 2):
            chosen = candidates[i][0]
            rejected = candidates[-(i+1)][0]
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    return pairs
```

Automatic scoring methods:
- **Code:** Run linter/formatter, check for errors
- **ASM:** Validate syntax, check instruction usage
- **General:** Perplexity, length heuristics

### 2. LLM-as-Judge (Semi-Automatic)

Use frontier model (GPT-4, Claude Opus) to rank outputs:

```python
import anthropic

def llm_rank_outputs(prompt, outputs):
    """Use Claude Opus to rank multiple outputs."""
    client = anthropic.Anthropic()

    ranking_prompt = f"""Rank these {len(outputs)} coding assistant responses from best to worst.

USER REQUEST: {prompt}

RESPONSES:
{chr(10).join(f"[{i}] {out}" for i, out in enumerate(outputs))}

Respond with just the ranking as comma-separated indices (best first):
Example: 2,0,3,1
"""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=100,
        messages=[{"role": "user", "content": ranking_prompt}]
    )

    ranking = [int(x.strip()) for x in response.content[0].text.split(",")]
    return ranking
```

### 3. Human Annotation (Gold Standard)

Build annotation UI:

```bash
hafs annotate start --model hafs-coder:14b --num-candidates 4
```

Shows 4 model outputs side-by-side:
```
Prompt: Write a routine to load player state from WRAM

[A] LoadPlayerState:          [B] LoadPlayerState:
    LDA.w $0E20                   LDA $0E20
    AND.b #$1F                    STA $00
    STA.b $00                     RTS
    RTS

[C] LoadPlayerState:          [D] LoadPlayerState:
    ; Load Link's state           LDA.b #$0E20  ; WRONG: immediate mode!
    LDA.w $0E20                   STA.b $00
    AND.b #$1F                    RTS
    STA.b $00
    RTS

Rank from best to worst (e.g., "C A B D"):
> _
```

Human provides ranking → generate all pairs:
- Chosen: C, Rejected: A
- Chosen: C, Rejected: B
- Chosen: C, Rejected: D
- Chosen: A, Rejected: B
- Chosen: A, Rejected: D
- Chosen: B, Rejected: D

### Hybrid Approach (Recommended)

1. Generate 4-8 candidates per prompt (best-of-N)
2. Use LLM judge to pre-rank (filter obviously bad)
3. Human annotates top 50-100 examples for calibration
4. Use calibrated LLM judge for rest

## DPO Training

### 1. Install DPO Library

```bash
pip install trl  # Transformer Reinforcement Learning
```

### 2. Prepare Preference Dataset

Format preference pairs:

```python
from datasets import Dataset

def format_dpo_dataset(preference_pairs):
    """Convert to DPO format."""
    data = []

    for pair in preference_pairs:
        data.append({
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    return Dataset.from_list(data)

# Load your pairs
pairs = load_preference_data("~/.context/training/preferences.jsonl")
dataset = format_dpo_dataset(pairs)

# Split train/val
train_test = dataset.train_test_split(test_size=0.1)
train_dataset = train_test["train"]
val_dataset = train_test["test"]
```

### 3. DPO Training Script

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
import torch

# Load SFT model as starting point
model = AutoModelForCausalLM.from_pretrained(
    "~/.context/training/checkpoints/hafs-coder-sft",  # Your SFT model
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("~/.context/training/checkpoints/hafs-coder-sft")

# Load reference model (frozen SFT model)
ref_model = AutoModelForCausalLM.from_pretrained(
    "~/.context/training/checkpoints/hafs-coder-sft",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Optional: Add LoRA for DPO (parameter-efficient)
lora_config = LoraConfig(
    r=32,  # Smaller rank for DPO
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# DPO training config
training_args = DPOConfig(
    output_dir="~/.context/training/checkpoints/hafs-coder-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,  # Lower than SFT!
    beta=0.1,  # DPO temperature (0.1-0.5)
    max_length=2048,
    max_prompt_length=512,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    eval_steps=50,
)

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Train!
trainer.train()

# Save final model
trainer.save_model("~/.context/training/checkpoints/hafs-coder-dpo-final")
```

### 4. Run DPO Training

```bash
cd ~/Code/hafs
python -m agents.training.dpo_trainer \
  --sft_model ~/.context/training/checkpoints/hafs-coder-sft \
  --preference_data ~/.context/training/preferences.jsonl \
  --output_dir ~/.context/training/checkpoints/hafs-coder-dpo \
  --beta 0.1 \
  --learning_rate 5e-6 \
  --num_epochs 1 \
  --batch_size 2 \
  --gradient_accumulation_steps 8
```

## DPO Hyperparameters

### Beta (Temperature)

Controls strength of preference:
- **β=0.1:** Gentle, conservative (safe default)
- **β=0.3:** Moderate, noticeable improvement
- **β=0.5:** Aggressive, may overfit

Start with 0.1, increase if improvements are weak.

### Learning Rate

Much lower than SFT:
- SFT: 2e-4
- DPO: **5e-6 to 1e-5** (10-40x lower)

### Epochs

DPO overfits quickly:
- **1 epoch** usually sufficient
- **2-3 epochs** if you have very large preference dataset (10K+ pairs)
- Monitor val loss closely

### Batch Size

Effective batch size should be 16-32:
```python
# Example:
batch_size = 2
gradient_accumulation_steps = 8
# Effective batch size = 2 × 8 = 16
```

## RLHF (Full Pipeline)

If you need traditional RLHF with PPO:

### 1. Train Reward Model

```python
from transformers import AutoModelForSequenceClassification

# Start from SFT model
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "~/.context/training/checkpoints/hafs-coder-sft",
    num_labels=1,  # Single scalar reward
)

# Train on preference pairs
for pair in preference_pairs:
    # Encode chosen and rejected
    chosen_ids = tokenizer(pair["prompt"] + pair["chosen"])
    rejected_ids = tokenizer(pair["prompt"] + pair["rejected"])

    # Reward model should score chosen > rejected
    chosen_reward = reward_model(chosen_ids).logits
    rejected_reward = reward_model(rejected_ids).logits

    # Loss: encourage chosen_reward > rejected_reward
    loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward))
    loss.backward()
    optimizer.step()
```

### 2. PPO Training

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Wrap model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "~/.context/training/checkpoints/hafs-coder-sft"
)

# PPO config
ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1.41e-5,
    ppo_epochs=4,
    mini_batch_size=4,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# Training loop
for batch in dataloader:
    queries = batch["prompt"]
    responses = ppo_trainer.generate(queries)

    # Get rewards from reward model
    rewards = [reward_model(q + r) for q, r in zip(queries, responses)]

    # PPO update
    stats = ppo_trainer.step(queries, responses, rewards)
```

**Why this is harder:**
- Need to train reward model separately (2x memory)
- PPO is unstable, needs careful hyperparameter tuning
- Can diverge from reference model (mode collapse)
- 2-3x slower than DPO

**When to use RLHF:**
- You have a very large team and compute budget
- You're doing research on RL methods
- DPO didn't work for your use case (rare)

## Evaluation: SFT vs DPO

Compare models:

```python
# Baseline: Base model (Qwen2.5-Coder-14B)
# SFT: After supervised fine-tuning
# DPO: After DPO on top of SFT

models = {
    "baseline": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "sft": "~/.context/training/checkpoints/hafs-coder-sft",
    "dpo": "~/.context/training/checkpoints/hafs-coder-dpo",
}

results = evaluate_models(models, test_set)

print_comparison_table(results)
```

Expected improvements:
```
| Metric              | Baseline | SFT   | DPO   |
|---------------------|----------|-------|-------|
| ASM Benchmark       | 48%      | 75%   | 82%   |
| ROM Hack Benchmark  | 40%      | 75%   | 79%   |
| Human Preference    | -        | 62%   | 74%   |
| Win Rate vs SFT     | -        | -     | 68%   |
```

DPO typically gives 5-10% absolute improvement over SFT on preference-based metrics.

## Preference Data Collection Strategy

### Phase 1: Automatic (0 human hours)

Generate 10K pairs using best-of-4 + automatic scoring:
```bash
python -m agents.training.generate_preferences \
  --model hafs-coder-sft:14b \
  --num-samples 10000 \
  --candidates-per-prompt 4 \
  --scorer automatic \
  --output ~/.context/training/preferences_auto.jsonl
```

### Phase 2: LLM Judge (1-2 hours)

Refine with Claude Opus judging:
```bash
python -m agents.training.generate_preferences \
  --model hafs-coder-sft:14b \
  --num-samples 2000 \
  --candidates-per-prompt 4 \
  --scorer claude-opus-4-5 \
  --output ~/.context/training/preferences_llm.jsonl
```

Cost: ~$50-100 for 2K samples at Opus pricing.

### Phase 3: Human Calibration (2-4 hours)

Manually annotate 100-200 challenging examples:
```bash
hafs annotate start \
  --model hafs-coder-sft:14b \
  --num-samples 200 \
  --candidates 4 \
  --output ~/.context/training/preferences_human.jsonl
```

Human time: ~1-2 min per sample = 3-4 hours total.

### Combined Dataset

```python
# Merge all sources
preferences = []
preferences.extend(load_jsonl("preferences_auto.jsonl"))      # 10K
preferences.extend(load_jsonl("preferences_llm.jsonl"))       # 2K
preferences.extend(load_jsonl("preferences_human.jsonl"))     # 200

# Weight human data higher (optional)
for pref in preferences:
    if pref["source"] == "human":
        # Duplicate human samples to increase weight
        preferences.extend([pref] * 5)

save_jsonl(preferences, "preferences_combined.jsonl")
# Total: ~12K pairs (200 human-equivalent with weighting)
```

## Iterative Improvement Loop

```
1. SFT Training (LoRA)
   ↓
2. Evaluate on benchmarks
   ↓
3. Identify failure modes
   ↓
4. Generate preference pairs (focus on failures)
   ↓
5. DPO Training
   ↓
6. Evaluate again
   ↓
7. If not satisfied, go to step 3
```

Each iteration:
- SFT: Major improvement (40% → 75%)
- DPO Round 1: Moderate improvement (75% → 82%)
- DPO Round 2: Diminishing returns (82% → 85%)
- DPO Round 3+: Minimal gains

Stop when:
- Performance plateaus
- Passing your target benchmarks
- Starts overfitting (val metrics degrade)

## Next Steps

1. **Complete SFT first:** Don't start DPO until you have a solid SFT model
2. **Generate preferences:** Use hybrid approach (auto + LLM + human)
3. **Train DPO:** Start with conservative settings (β=0.1, lr=5e-6, 1 epoch)
4. **Evaluate:** Compare SFT vs DPO on benchmarks
5. **Iterate:** Generate more preferences for weak areas, run DPO round 2

## Resources

- [DPO Paper (Rafailov et al. 2023)](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Zephyr-7B DPO Case Study](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
