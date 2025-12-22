# Rejected Samples Analysis Guide

## Overview

The training data generation system now saves ALL samples, including those rejected by quality filtering, so you can analyze why they're being rejected and tune the quality thresholds accordingly.

## What Changed

### Quality Pipeline (`src/agents/training/quality.py`)

- Added `RejectedSample` dataclass to track rejected samples with reasons
- Modified `FilterStats` to include `rejected_samples` list
- Updated `filter_samples()` to collect rejected samples with detailed metadata:
  - Validation failures
  - Low quality scores (with component breakdown)
  - Duplicates

### Data Curator (`src/agents/training/curator.py`)

- Added `_save_rejected_samples()` method
- Automatically saves rejected samples to `rejected.jsonl` in dataset directory
- Creates `rejection_summary.json` with statistics

## Output Files

When you run a training campaign, each dataset directory now contains:

```
~/.context/training/datasets/<campaign_name>/
├── train.jsonl              # Accepted samples
├── val.jsonl                # Accepted samples
├── test.jsonl               # Accepted samples
├── rejected.jsonl           # NEW: All rejected samples
├── rejection_summary.json   # NEW: Rejection statistics
├── metadata.json
└── stats.json
```

### rejected.jsonl Format

Each line contains:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "domain": "asm",
  "source": "alttp/bank_00.asm",
  "rejection_reason": "low_quality",
  "quality_score": 0.35,
  "rejection_details": {
    "threshold": 0.4,
    "diversity": 0.42,
    "kg_consistency": 0.38,
    "hallucination_risk": 0.25,
    "coherence": 0.32
  }
}
```

### rejection_summary.json Format

```json
{
  "total_rejected": 5432,
  "by_reason": {
    "low_quality": 4521,
    "low_diversity": 523,
    "validation_failed": 221,
    "duplicate": 167
  },
  "by_domain": {
    "asm": 3421,
    "oracle": 1234,
    "gigaleak": 777
  },
  "avg_quality_score": 0.342,
  "min_quality_score": 0.103,
  "max_quality_score": 0.398
}
```

## Analyzing Rejected Samples

### Quick Analysis

```bash
# Analyze the most recent dataset
python scripts/analyze_rejected_samples.py --latest

# Analyze a specific dataset
python scripts/analyze_rejected_samples.py ~/.context/training/datasets/my_campaign_20251221_123456
```

### Output

The analysis script shows:

1. **Summary**: Total rejections, score distribution
2. **Rejection Reasons**: Breakdown by reason with percentages
3. **By Domain**: Which domains have most rejections
4. **Closest to Passing**: Top 10 samples that almost made it
5. **Quality Score Distribution**: Histogram of scores
6. **Recommendations**: Specific suggestions based on patterns

### Example Output

```
================================================================================
Analyzing Rejected Samples: hybrid_34500_20251221_155838
================================================================================

SUMMARY
--------------------------------------------------------------------------------
Total rejected: 5597
Average quality score: 0.342
Min quality score: 0.103
Max quality score: 0.398

REJECTION REASONS
--------------------------------------------------------------------------------
  low_quality                 4521 ( 80.8%)
  low_diversity                523 (  9.3%)
  validation_failed            221 (  3.9%)
  duplicate                    167 (  3.0%)

CLOSEST TO PASSING (Top 10)
--------------------------------------------------------------------------------
1. [asm] Score: 0.398 | Reason: low_quality
   Threshold: 0.400
   Diversity: 0.42
   KG Consistency: 0.38
   Hallucination Risk: 0.25
   Coherence: 0.35
   Instruction: Explain the purpose of the REP #$30 instruction in 65816...

2. [oracle] Score: 0.395 | Reason: low_quality
   ...
```

## Common Issues and Solutions

### Issue 1: All Samples Rejected for Low Quality

**Symptom**: Most samples have scores just below threshold (e.g., 0.35-0.39 when threshold is 0.4)

**Solutions**:
1. Lower domain-specific thresholds in `src/agents/training/quality.py`:
   ```python
   DOMAIN_THRESHOLDS = {
       "asm": 0.35,  # Lower from 0.4
       "oracle": 0.35,  # Lower from 0.4
       # ...
   }
   ```

2. Review quality component weights - one low component may be dragging down overall score

### Issue 2: Low Diversity

**Symptom**: Many samples rejected for `low_diversity`

**Solutions**:
1. Increase prompt variety in generators
2. Add more diverse source material
3. Use different base models for generation

### Issue 3: KG Inconsistent

**Symptom**: High rejection rate for `kg_inconsistent`

**Solutions**:
1. Update knowledge graph with correct information
2. Review entity extraction logic
3. Consider lowering KG consistency weight if it's too strict

### Issue 4: High Hallucination Risk

**Symptom**: Samples rejected for `high_hallucination`

**Solutions**:
1. Add more source context to generation prompts
2. Use more grounded generation approaches
3. Review hallucination detection thresholds

## Recovering Good Samples

If you find that many rejected samples are actually good quality:

### Option 1: Lower Thresholds and Re-run

Edit `src/agents/training/quality.py` and restart campaign:

```python
DOMAIN_THRESHOLDS = {
    "asm": 0.30,  # Lower threshold
    "oracle": 0.30,
    # ...
}
```

### Option 2: Manually Review and Accept

Create a script to filter rejected samples above a certain score:

```python
#!/usr/bin/env python3
import json
from pathlib import Path

rejected_file = Path("~/.context/training/datasets/my_dataset/rejected.jsonl").expanduser()
accepted_file = rejected_file.parent / "manually_accepted.jsonl"

min_score = 0.35  # Adjust threshold

with open(rejected_file) as f_in, open(accepted_file, "w") as f_out:
    for line in f_in:
        sample = json.loads(line)
        if sample.get('quality_score', 0) >= min_score:
            # Remove rejection metadata
            cleaned = {
                "instruction": sample['instruction'],
                "input": sample['input'],
                "output": sample['output']
            }
            f_out.write(json.dumps(cleaned) + "\n")
```

### Option 3: Merge Rejected Samples into Training Data

If rejected samples are close to threshold, you can merge them:

```bash
# Combine rejected.jsonl with train.jsonl
cd ~/.context/training/datasets/my_dataset/
cat rejected.jsonl >> train.jsonl
```

## Monitoring Quality Over Time

Track rejection rates across campaigns to see if quality is improving:

```bash
# Compare rejection rates
for dir in ~/.context/training/datasets/*/; do
  name=$(basename "$dir")
  if [ -f "$dir/rejection_summary.json" ]; then
    total=$(jq '.total_rejected' "$dir/rejection_summary.json")
    avg=$(jq '.avg_quality_score' "$dir/rejection_summary.json")
    echo "$name: $total rejected (avg: $avg)"
  fi
done
```

## Next Steps

1. **Run a new campaign** to generate datasets with rejection tracking
2. **Analyze rejections** using the analysis script
3. **Tune thresholds** based on findings
4. **Re-run campaign** with adjusted thresholds
5. **Iterate** until you get acceptable acceptance rate (aim for 60-80%)

## Testing Model Training

Once you have a dataset with acceptable samples, you can:

1. **Check if you have training data**:
   ```bash
   python scripts/test_model_training.py --check-data
   ```

2. **Run demo training** (tests GPU setup):
   ```bash
   python scripts/test_model_training.py --train-demo
   ```

3. **Train on your data**: Modify `scripts/example_unsloth_training.py` to use your dataset

4. **Test trained model**:
   ```bash
   python scripts/test_model_training.py --test-model D:/training/models/my_model
   ```

## References

- `src/agents/training/quality.py` - Quality filtering logic
- `src/agents/training/curator.py` - Dataset curation and saving
- `scripts/analyze_rejected_samples.py` - Analysis tool
- `scripts/test_model_training.py` - Training and testing tool
