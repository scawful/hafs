# A/B Testing Guide for Training Data Generators

**Last updated:** 2025-12-21
**Purpose:** Compare prompt versions to guide quality improvements

## Overview

The A/B testing framework allows you to compare different prompt versions side-by-side to measure quality improvements. This enables data-driven prompt engineering with quantifiable metrics.

## Quick Start

### Run a Quick Test (50 samples)

```bash
cd ~/Code/hafs

# ASM generator
python scripts/run_ab_test.py --domain asm --quick

# Oracle generator
python scripts/run_ab_test.py --domain oracle --quick
```

### Run a Standard Test (100 samples)

```bash
# ASM
python scripts/run_ab_test.py --domain asm --samples 100

# Oracle
python scripts/run_ab_test.py --domain oracle --samples 100
```

### Run a Full Test (1000 samples)

```bash
# This will take ~30-60 minutes depending on API speed
python scripts/run_ab_test.py --domain asm --samples 1000
```

## How It Works

### 1. Test Setup

The test runner:
1. Initializes two generators: baseline (old prompts) and enhanced (new prompts)
2. Extracts the same source items for both
3. Generates samples using each prompt version
4. Runs quality filtering pipeline on both sets
5. Compares metrics

### 2. Metrics Compared

**Primary Metrics:**
- **Pass Rate:** Percentage of samples passing quality filters
- **Avg Quality Score:** Overall quality (0.0-1.0)
- **Avg Diversity:** Embedding-based uniqueness
- **Avg Coherence:** Instruction/output alignment
- **Avg Hallucination:** Risk score (lower is better)

**Rejection Breakdown:**
- Validation failures (syntax, format errors)
- Quality too low (below domain threshold)
- Duplicates (similarity > 0.95)

### 3. Statistical Significance

Tests are considered significant if:
- **Pass rate improvement:** > 5% (absolute)
- **Sample size:** ≥ 100 samples per version

Confidence level: 95% if significant

### 4. Output

Results are saved to:
```
~/.context/training/ab_tests/
└── ab_test_asm_YYYYMMDD_HHMMSS.json
```

## Test Output Example

```
================================================================================
A/B TEST RESULTS: ASM
================================================================================
Test ID: ab_test_asm_20251221_140530
Timestamp: 2025-12-21T14:05:30.123456
Target Samples: 1000

BASELINE vs ENHANCED
────────────────────────────────────────────────────────────────────────────────
Metric                         Baseline        Enhanced     Improvement
────────────────────────────────────────────────────────────────────────────────
Samples Generated                  1000            1000
Samples Passed                      600             820
Pass Rate                          60.0%           82.0%          +22.0%
Avg Quality Score                  0.521           0.712          +0.191
Avg Diversity                      0.680           0.730          +0.050
Avg Coherence                      0.480           0.710          +0.230
Avg Hallucination                  0.320           0.180          -0.140

REJECTION BREAKDOWN
────────────────────────────────────────────────────────────────────────────────
Reason                         Baseline        Enhanced
────────────────────────────────────────────────────────────────────────────────
Validation Failed                    50              20
Quality Too Low                     320             150
Duplicates                           30              10

VERDICT
────────────────────────────────────────────────────────────────────────────────
Winner: ENHANCED
Significant: YES (confidence: 95%)

STRONG RECOMMENDATION: Use enhanced prompts (+22.0% pass rate)
================================================================================
```

## Interpreting Results

### Pass Rate Improvement

- **+20% or more:** 🎯 Excellent improvement - use enhanced prompts
- **+10% to +20%:** ✅ Strong improvement - recommended
- **+5% to +10%:** 👍 Moderate improvement - beneficial
- **0% to +5%:** 🤷 Marginal - consider more iterations
- **Negative:** ⚠️ Enhanced prompts are worse - iterate further

### Quality Score Improvement

- **+0.15 or more:** Major quality jump
- **+0.10 to +0.15:** Significant improvement
- **+0.05 to +0.10:** Moderate improvement
- **0.00 to +0.05:** Slight improvement
- **Negative:** Quality regression

### Coherence Improvement

Most important for ASM/Oracle domains:
- **+0.20 or more:** Instructions and outputs are much better aligned
- **+0.10 to +0.20:** Good alignment improvement
- **+0.05 to +0.10:** Some improvement

### Hallucination Reduction

Lower is better:
- **-0.15 or more:** Much less uncertain language
- **-0.10 to -0.15:** Good reduction in hallucinations
- **-0.05 to -0.10:** Moderate reduction

## Running Custom Tests

### Python API

```python
from agents.training.ab_testing import ABTestRunner, PromptVersion
from agents.training.generators.asm_generator import AsmDataGenerator

# Create runner
runner = ABTestRunner()

# Define versions
baseline = PromptVersion(
    name="baseline",
    generator_cls=AsmDataGenerator,
    use_enhanced=False,
)

enhanced = PromptVersion(
    name="enhanced_v1",
    generator_cls=AsmDataGenerator,
    use_enhanced=True,
)

# Run test
comparison = await runner.run_test(
    versions=[baseline, enhanced],
    num_samples=1000,
    source_limit=200,  # Use first 200 source items
    domain="asm",
)

# Access results
print(f"Pass rate improvement: {comparison.pass_rate_improvement:+.1%}")
print(f"Quality improvement: {comparison.quality_improvement:+.3f}")
print(f"Winner: {comparison.winner}")
```

### Testing New Prompt Variations

To test a new prompt variation:

1. **Create new prompt function** in `src/agents/training/generators/enhanced_prompts.py`:
   ```python
   def get_enhanced_asm_prompt_v2(...):
       # Your new prompt with different structure
       pass
   ```

2. **Add version flag** to generator:
   ```python
   class AsmDataGenerator:
       def __init__(self, prompt_version: str = "baseline"):
           self.prompt_version = prompt_version
   ```

3. **Run A/B test:**
   ```python
   v1 = PromptVersion(name="v1", use_enhanced=True)  # Old enhanced
   v2 = PromptVersion(name="v2", config={"version": "v2"})  # New

   comparison = await runner.run_test([v1, v2], ...)
   ```

## Best Practices

### Sample Size Recommendations

| Test Type | Samples | Source Limit | Duration | Use Case |
|-----------|---------|--------------|----------|----------|
| Quick | 50 | 25 | ~5 min | Initial validation |
| Standard | 100 | 50 | ~10 min | Iterative testing |
| Thorough | 500 | 200 | ~30 min | Pre-deployment |
| Full | 1000 | None | ~60 min | Final validation |

### Testing Workflow

1. **Quick test first** (50 samples)
   - Verify enhanced prompts don't break generation
   - Check if improvement direction is correct

2. **Standard test** (100 samples)
   - More reliable metrics
   - Enough for statistical significance

3. **Thorough test** (500-1000 samples)
   - Before committing to production use
   - Validates consistency across more samples

### Iterative Improvement

```
Iteration 1: Baseline vs Enhanced v1
├─ Result: +15% pass rate
├─ Action: Identify remaining failure patterns
└─ Create: Enhanced v2 with targeted fixes

Iteration 2: Enhanced v1 vs Enhanced v2
├─ Result: +8% additional improvement
├─ Action: Analyze what worked
└─ Create: Enhanced v3

Iteration 3: Enhanced v2 vs Enhanced v3
├─ Result: +2% (diminishing returns)
└─ Decision: Deploy Enhanced v2 (best ROI)
```

### When to Stop Iterating

Stop when:
- **Improvement < 2%:** Diminishing returns
- **Pass rate > 85%:** Excellent quality achieved
- **Quality plateau:** No improvement for 2-3 iterations
- **Cost/benefit:** Improvement doesn't justify effort

## Cost Estimation

### Gemini 2.0 Flash Pricing

- **Input:** $0.075 per 1M tokens
- **Output:** $0.30 per 1M tokens

**Estimated costs per test:**

| Samples | Tokens (est) | Cost (est) |
|---------|--------------|------------|
| 50 | ~1.5M | ~$0.50 |
| 100 | ~3M | ~$1.00 |
| 500 | ~15M | ~$5.00 |
| 1000 | ~30M | ~$10.00 |

**Note:** Running both baseline AND enhanced doubles the cost (need both for comparison).

### Budget-Friendly Testing

1. **Use quick tests** for rapid iteration (50 samples = $1)
2. **Limit source items** (--source-limit 25)
3. **Run full tests** only for final validation
4. **Share results** across team to avoid duplicate tests

## Analyzing Results

### Quality Distribution

Check the quality distribution bins to understand the shift:

```json
{
  "baseline": {
    "0.0-0.2": 80,   // Very poor
    "0.2-0.4": 250,  // Below threshold
    "0.4-0.6": 400,  // Passing
    "0.6-0.8": 200,  // Good
    "0.8-1.0": 70    // Excellent
  },
  "enhanced": {
    "0.0-0.2": 20,   // ↓ Much fewer very poor
    "0.2-0.4": 100,  // ↓ Fewer rejections
    "0.4-0.6": 350,  // ~ Similar passing
    "0.6-0.8": 380,  // ↑ More good samples
    "0.8-1.0": 150   // ↑ More excellent samples
  }
}
```

**Good sign:** Distribution shifts right (toward higher quality)

### Rejection Reasons

Examine why samples are rejected:

```json
{
  "baseline": {
    "low_coherence": 45%,
    "missing_addressing_modes": 30%,
    "incomplete_code": 25%
  },
  "enhanced": {
    "low_coherence": 15%,      // ↓ Big improvement
    "missing_addressing_modes": 10%,  // ↓ Much better
    "incomplete_code": 5%      // ↓ Rare now
  }
}
```

**Action:** Focus next iteration on remaining issues

### Top vs Failed Samples

Review example samples:

**Top samples (quality > 0.8):**
- What makes them excellent?
- Can we incorporate patterns into prompt?

**Failed samples (quality < 0.4):**
- What common mistakes?
- Add anti-patterns to prompt

## Troubleshooting

### Test Fails to Run

**Error:** `ModuleNotFoundError: No module named 'agents.training.ab_testing'`

**Fix:**
```bash
cd ~/Code/hafs
export PYTHONPATH=$PWD/src:$PYTHONPATH
python scripts/run_ab_test.py --domain asm --quick
```

### Low Sample Generation

**Issue:** Only generated 50 samples instead of 1000

**Causes:**
1. Limited source items (use `--source-limit None`)
2. Generator errors (check logs)
3. API rate limiting (add delays)

### No Quality Difference

**Issue:** Both versions show similar pass rates

**Causes:**
1. Prompts are too similar
2. Sample size too small (increase to 500+)
3. Quality pipeline too lenient (check thresholds)

### API Timeout Errors

**Issue:** `TimeoutError` during generation

**Fixes:**
1. Reduce `--samples` (try 50 first)
2. Increase timeout in generator
3. Check network/API status

## Integration with Production

After A/B test shows improvement:

### 1. Update Generator Default

```python
# src/agents/training/generators/asm_generator.py
class AsmDataGenerator(DataGenerator):
    def __init__(self, use_enhanced_prompts: bool = True):  # Changed default
        # ...
```

### 2. Launch Production Campaign

```bash
# Launch full campaign with enhanced prompts
python scripts/run_training_campaign.py \
  --generator asm \
  --target 24000 \
  --use-enhanced \
  --checkpoint-interval 500
```

### 3. Monitor Quality Metrics

```bash
# Track pass rate during campaign
hafs logs campaign --follow | grep "Pass rate"

# Check quality trends
hafs agents quality-report
```

### 4. Rollback if Needed

If production quality degrades:

```python
# Quick rollback to baseline
class AsmDataGenerator:
    def __init__(self, use_enhanced_prompts: bool = False):  # Rollback
        # ...
```

## Future Enhancements

### Multi-Variant Testing (A/B/C)

Test 3+ variants simultaneously:

```python
variants = [
    PromptVersion("baseline", use_enhanced=False),
    PromptVersion("enhanced_v1", use_enhanced=True),
    PromptVersion("enhanced_v2", config={"version": "v2"}),
]

# Run all pairwise comparisons
for i in range(len(variants)):
    for j in range(i+1, len(variants)):
        comparison = await runner.run_test([variants[i], variants[j]], ...)
```

### Automated Iteration

Use feedback to automatically improve prompts:

```python
# Analyze failures
patterns = analyze_rejection_patterns(failed_samples)

# Generate new prompt
new_prompt = auto_improve_prompt(
    current_prompt=enhanced_v1,
    failure_patterns=patterns,
)

# Test automatically
comparison = await runner.run_test([enhanced_v1, new_prompt], ...)
```

### Continuous Testing

Run A/B tests on every prompt change:

```bash
# CI/CD hook
git push
  → Trigger A/B test
  → Compare against baseline
  → Block merge if regression
  → Auto-approve if improvement > 5%
```

## References

- A/B testing framework: `src/agents/training/ab_testing.py`
- Enhanced prompts: `src/agents/training/generators/enhanced_prompts.py`
- Test runner script: `scripts/run_ab_test.py`
- Quality pipeline: `src/agents/training/quality.py`
