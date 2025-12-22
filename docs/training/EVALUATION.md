# Model Evaluation Guide

## Overview

Comprehensive evaluation framework for fine-tuned models across ASM, ROM hacking, and coding domains.

## Evaluation Strategy

### 1. Automatic Metrics

**Perplexity:** How confident is the model?
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def compute_perplexity(model, tokenizer, text):
    """Compute perplexity on test set."""
    encodings = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**encodings, labels=encodings.input_ids)
        loss = outputs.loss

    return torch.exp(loss).item()
```

Lower perplexity = better understanding of the domain.

**BLEU/ROUGE:** Text overlap metrics
```python
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")

def evaluate_generation(predictions, references):
    """Compare generated text to ground truth."""
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return {
        "bleu": bleu_score["bleu"],
        "rouge-L": rouge_score["rougeL"],
    }
```

**Exact Match:** For code/ASM where precision matters
```python
def exact_match_score(predictions, references):
    """Percentage of exactly correct outputs."""
    matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    return matches / len(predictions)
```

### 2. Domain-Specific Benchmarks

#### ASM Benchmark

Test set of 100 hand-verified ASM routines:

```python
# Test categories:
asm_benchmark = {
    "basic_operations": 20,      # LDA, STA, simple logic
    "memory_access": 15,          # Direct page, absolute, long
    "control_flow": 15,           # Branches, jumps, subroutines
    "hardware_interaction": 20,   # PPU, APU, DMA
    "optimization": 15,           # Fast paths, efficient code
    "complex_routines": 15,       # Multi-function, state machines
}
```

Evaluation criteria:
- **Correctness:** Does the code work?
- **Addressing modes:** Proper .b/.w usage?
- **Comments:** Clear explanations?
- **Idioms:** Uses ALTTP conventions?

#### ROM Hacking Benchmark

Test set for Oracle-of-Secrets techniques:

```python
rom_hack_benchmark = {
    "hook_implementation": 25,    # JSL redirects, org directives
    "bank_allocation": 15,        # Expanded ROM banks
    "vanilla_analysis": 20,       # Understanding original code
    "feature_implementation": 25, # Custom game mechanics
    "integration": 15,            # Working with existing systems
}
```

Evaluation:
- **Technique accuracy:** Correct ROM hacking method?
- **Address precision:** Exact $XX:XXXX addresses?
- **Risk assessment:** Identifies pitfalls?

#### Code Understanding Benchmark

Test comprehension vs generation:

```python
understanding_benchmark = {
    "code_explanation": 30,       # "What does this routine do?"
    "bug_diagnosis": 25,          # "Why doesn't this work?"
    "optimization_suggestions": 20, # "How to make this faster?"
    "api_usage": 15,              # "How to use this function?"
    "debugging": 10,              # "Add debug output here"
}
```

### 3. LLM-as-Judge Evaluation

Use frontier models (GPT-4, Claude Opus, Gemini Pro) to judge quality:

```python
import anthropic

def llm_judge_quality(instruction, ground_truth, prediction):
    """Use Claude Opus to judge response quality."""
    client = anthropic.Anthropic()

    prompt = f"""You are evaluating the quality of a coding assistant response.

INSTRUCTION: {instruction}

GROUND TRUTH ANSWER:
{ground_truth}

MODEL PREDICTION:
{prediction}

Rate the prediction on a scale of 1-10 for:
1. Correctness: Is the technical information accurate?
2. Completeness: Does it fully answer the question?
3. Clarity: Is the explanation clear and well-structured?
4. Code Quality: If code is present, is it correct and idiomatic?

Respond with JSON:
{{
  "correctness": 1-10,
  "completeness": 1-10,
  "clarity": 1-10,
  "code_quality": 1-10,
  "overall": 1-10,
  "reasoning": "Brief explanation"
}}"""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)
```

### 4. Human Evaluation

Gold standard - manual review:

**Evaluation UI:** `src/hafs/cli/commands/evaluate.py`

```bash
hafs evaluate start --model hafs-coder:14b --benchmark asm
```

Presents samples one-by-one:
```
[1/100] ASM - Basic Operations

Instruction: Write a routine to clear the OAM buffer

Model Output:
ClearOAMBuffer:
    LDA.b #$F0           ; Load Y-coordinate for off-screen
    LDX.b #$00           ; Start at OAM entry 0
.loop:
    STA.w $0800,X        ; Write to OAM Y position
    INX #4               ; Next OAM entry (4 bytes each)
    CPX.b #$80           ; 128 sprites * 4 = 512 bytes
    BNE .loop
    RTL

Rate this response:
[1] Poor  [2] Fair  [3] Good  [4] Excellent  [s] Skip  [c] Comment
> _
```

Stores ratings in `~/.context/training/evaluations/`:
```json
{
  "model": "hafs-coder:14b",
  "benchmark": "asm",
  "timestamp": "2025-12-21T17:30:00",
  "samples": [
    {
      "id": "asm_001",
      "rating": 4,
      "comments": "Correct, efficient, well-commented",
      "time_to_rate": 12.5
    }
  ],
  "summary": {
    "mean_rating": 3.4,
    "excellent_pct": 0.42,
    "poor_pct": 0.08
  }
}
```

## Evaluation Pipeline

### Full Evaluation Script

```bash
#!/bin/bash
# evaluate_model.sh

MODEL=$1  # e.g., hafs-coder:14b
OUTPUT_DIR=~/.context/training/evaluations/$(date +%Y%m%d_%H%M%S)

echo "Evaluating model: $MODEL"
mkdir -p $OUTPUT_DIR

# 1. Automatic metrics on test set
echo "Computing perplexity..."
python -m agents.training.eval.perplexity \
  --model $MODEL \
  --test-set ~/.context/training/test_set.jsonl \
  --output $OUTPUT_DIR/perplexity.json

# 2. Generation quality (BLEU/ROUGE)
echo "Evaluating generation quality..."
python -m agents.training.eval.generation \
  --model $MODEL \
  --test-set ~/.context/training/test_set.jsonl \
  --output $OUTPUT_DIR/generation.json

# 3. Domain benchmarks
echo "Running ASM benchmark..."
python -m agents.training.eval.benchmark \
  --model $MODEL \
  --domain asm \
  --output $OUTPUT_DIR/asm_benchmark.json

echo "Running ROM hack benchmark..."
python -m agents.training.eval.benchmark \
  --model $MODEL \
  --domain rom_hack \
  --output $OUTPUT_DIR/rom_hack_benchmark.json

# 4. LLM-as-judge
echo "Running LLM judge evaluation..."
python -m agents.training.eval.llm_judge \
  --model $MODEL \
  --test-set ~/.context/training/test_set.jsonl \
  --judge claude-opus-4-5 \
  --output $OUTPUT_DIR/llm_judge.json

# 5. Generate report
python -m agents.training.eval.report \
  --eval-dir $OUTPUT_DIR \
  --output $OUTPUT_DIR/report.md

echo "✓ Evaluation complete: $OUTPUT_DIR/report.md"
```

### Comparison Report

```markdown
# Model Evaluation Report

**Date:** 2025-12-21
**Model:** hafs-coder:14b (LoRA fine-tune on Qwen2.5-Coder-14B)
**Training Data:** 10,000 samples (ASM, Gigaleak, Oracle, YAZE, errors, text)

## Automatic Metrics

| Metric | Base Model | Fine-Tuned | Δ |
|--------|------------|------------|---|
| Perplexity (ASM) | 12.4 | 6.8 | ↓ 45% |
| Perplexity (ROM hack) | 18.2 | 9.1 | ↓ 50% |
| BLEU (overall) | 0.32 | 0.51 | ↑ 59% |
| ROUGE-L (overall) | 0.41 | 0.63 | ↑ 54% |
| Exact Match (ASM) | 0.08 | 0.23 | ↑ 188% |

## Benchmark Results

### ASM Benchmark (100 samples)

| Category | Score | vs Base |
|----------|-------|---------|
| Basic Operations | 85% | +32% |
| Memory Access | 78% | +28% |
| Control Flow | 72% | +25% |
| Hardware Interaction | 81% | +35% |
| Optimization | 68% | +22% |
| Complex Routines | 64% | +18% |
| **Overall** | **75%** | **+27%** |

### ROM Hacking Benchmark (100 samples)

| Category | Score | vs Base |
|----------|-------|---------|
| Hook Implementation | 79% | +41% |
| Bank Allocation | 71% | +38% |
| Vanilla Analysis | 82% | +29% |
| Feature Implementation | 74% | +35% |
| Integration | 68% | +31% |
| **Overall** | **75%** | **+35%** |

## LLM Judge Scores (Claude Opus 4.5)

| Dimension | Score (1-10) | vs Base |
|-----------|--------------|---------|
| Correctness | 8.2 | +1.8 |
| Completeness | 7.9 | +1.6 |
| Clarity | 8.4 | +0.9 |
| Code Quality | 8.1 | +2.1 |
| **Overall** | **8.2** | **+1.6** |

## Human Evaluation (20 samples)

- Mean Rating: 3.4 / 4.0
- Excellent (4): 42%
- Good (3): 46%
- Fair (2): 10%
- Poor (1): 2%

## Qualitative Analysis

### Strengths
- Strong understanding of 65816 assembly idioms
- Accurate hardware register usage ($21XX, $42XX)
- Clear, pedagogical explanations
- Good ROM hacking technique knowledge

### Weaknesses
- Occasional hallucination of non-existent ALTTP routines
- Less confident on rare/obscure opcodes
- Sometimes verbose in explanations
- Struggles with very complex multi-file interactions

## Failure Analysis

Top 10 failure modes (from 50 incorrect samples):
1. Incorrect memory address (18 cases)
2. Wrong addressing mode (.b vs .w) (12 cases)
3. Hallucinated routine name (8 cases)
4. Missing edge case handling (6 cases)
5. Inefficient code pattern (4 cases)
6. Unclear explanation (2 cases)

## Recommendations

1. **Generate more data for:**
   - Complex multi-routine interactions
   - Edge cases and error handling
   - Rare opcodes (MVN, MVP, PER, etc.)

2. **Add grounding:**
   - Include ALTTP symbol table as context
   - Add routine cross-reference checks
   - Validate addresses against known ROM layout

3. **Training adjustments:**
   - Increase LoRA rank to 128 for more capacity
   - Add 1-2 more epochs on weak categories
   - Use DPO to penalize hallucinations

4. **Next iteration target:** 85% overall benchmark score
```

## A/B Testing

Compare multiple model versions:

```python
def ab_test_models(models, test_set, num_samples=100):
    """Run head-to-head comparison."""
    results = {model: {"wins": 0, "losses": 0, "ties": 0} for model in models}

    for sample in test_set[:num_samples]:
        outputs = {}
        for model in models:
            outputs[model] = generate(model, sample["instruction"])

        # LLM judge picks winner
        winner = llm_pick_best(sample["instruction"], outputs)

        if winner == "tie":
            for model in models:
                results[model]["ties"] += 1
        else:
            results[winner]["wins"] += 1
            for model in models:
                if model != winner:
                    results[model]["losses"] += 1

    return results
```

## Continuous Evaluation

Set up automated evaluation on each training run:

```toml
# config/training_medical_mechanica.toml

[evaluation]
enabled = true
test_set = "~/.context/training/test_set_v2.jsonl"
benchmarks = ["asm", "rom_hack", "code_understanding"]
run_on_checkpoint = true  # Evaluate every N steps
checkpoint_interval = 500

# Auto-compare to baseline
baseline_model = "Qwen/Qwen2.5-Coder-14B-Instruct"
report_improvements = true
```

## Next Steps

1. **Build test sets:** Create gold-standard benchmarks (see `create_test_set.py`)
2. **Implement eval scripts:** Full pipeline in `src/agents/training/eval/`
3. **Run baseline:** Evaluate base model before fine-tuning
4. **Iterate:** Train → Eval → Analyze → Improve data → Repeat
5. **Deploy best model:** Export to Ollama when benchmarks pass

## Resources

- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [HuggingFace Evaluate](https://huggingface.co/docs/evaluate)
- [Stanford HELM Benchmarks](https://crfm.stanford.edu/helm/)
