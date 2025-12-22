# Distributed Generation Campaign - Ready to Deploy

## Current Status

### ✅ Alpha Pilot Complete (20 samples)
- **100% pass rate** (20/20)
- Quality fixes validated
- Domain-specific thresholds working

### 🔄 Aggressive Pilot Running (1000 samples)
- **ETA:** 10-15 minutes
- **Parallelism:** 10x concurrent on Gemini Flash
- **Domains:** ASM, Gigaleak, Oracle, YAZE, Errors
- **Task ID:** b87a73d

### ✅ Distributed Campaign Scripts Ready (34.5K samples)

## Architecture

### Generation Distribution
```
Gemini Flash (Primary - 70%):     ~24,000 samples
  └─ Fast, high quality, proven reliable

medical-mechanica (Offload - 30%): ~10,500 samples
  ├─ qwen3:14b (14.8B)       - Code generation
  ├─ deepseek-r1:14b (14.8B) - Reasoning tasks
  └─ magistral:24b (23.6B)   - Complex validation
```

### Node Capabilities

**medical-mechanica (100.104.53.21:11434)**
- ✅ `qwen3:14b` - Fast coding specialist
- ✅ `deepseek-r1:14b` - Reasoning engine
- ✅ `magistral:24b` - Heavy analysis
- ✅ `gemma3:27b` - Heavyweight tasks
- ✅ `embeddinggemma` - Fast embeddings

**Gemini Flash (Primary)**
- ✅ Main teacher model
- ✅ 70% of generation load
- ✅ Proven quality in alpha pilot

## Performance Estimates

| Approach | Samples | Duration | Throughput |
|----------|---------|----------|------------|
| Sequential | 34,500 | 30-40 hrs | ~1,000/hr |
| Parallel (10x) | 34,500 | 15-20 hrs | ~2,000/hr |
| **Distributed** | **34,500** | **8-12 hrs** | **~3,500/hr** |

**Speedup:** 2.5-4x faster than sequential

## Domain Allocation (Balanced)

| Domain | Target | % | Generator |
|--------|--------|---|-----------|
| ASM | 15,000 | 43.5% | AsmDataGenerator |
| Gigaleak | 8,000 | 23.2% | GigaleakDataGenerator |
| Oracle | 4,000 | 11.6% | OracleDataGenerator |
| YAZE | 6,000 | 17.4% | CppDataGenerator |
| Errors | 1,500 | 4.3% | ErrorSampleGenerator |
| **Total** | **34,500** | **100%** | |

## Quality Thresholds (Domain-Specific)

```python
{
    "asm": 0.4,       # ASM is hard - lower threshold
    "gigaleak": 0.5,  # Original source - medium
    "oracle": 0.4,    # ROM hack - lower
    "yaze": 0.5,      # C++ code - medium
    "errors": 0.3,    # Diagnostics - lowest
    "text": 0.6,      # Natural language - higher
}
```

## Generated Files

### Core Scripts
- ✅ `src/agents/training/distributed_generator.py` - Distributed generation logic
- ✅ `run_distributed_campaign.py` - Main campaign launcher
- ✅ `src/agents/training/parallel_generator.py` - 10x parallelism

### Supporting
- ✅ `src/agents/training/quality.py` - Fixed quality pipeline
- ✅ `src/agents/training/tests/test_quality_validation.py` - Regression tests
- ✅ `test_alpha_pilot.py` - Alpha validation (passed)
- ✅ `run_aggressive_pilot.py` - 1000-sample pilot (running)

## Next Steps

### 1. Wait for Aggressive Pilot (~10 min)
```bash
# Monitor progress
tail -f /tmp/claude/-Users-scawful-Code-hafs/tasks/b87a73d.output
```

### 2. Validate Pilot Results
- Check pass rate (expect >50%)
- Verify domain distribution
- Confirm quality scores

### 3. Launch Full Campaign (if pilot passes)
```bash
cd ~/Code/hafs
PYTHONPATH=src .venv/bin/python run_distributed_campaign.py
```

**Duration:** 8-12 hours
**Output:** `~/.context/training/datasets/alttp_yaze_full_distributed/`

### 4. Export Datasets
After campaign completes:
- **ALTTP ASM Dataset:** ~24,000 samples (asm + gigaleak + oracle + errors)
- **YAZE Tool Dataset:** ~7,000 samples (yaze + errors)

### 5. Deploy Training to medical-mechanica
```bash
# Copy datasets to medical-mechanica
scp -r ~/.context/training/datasets/alttp_yaze_full_distributed \
    scawful@100.104.53.21:D:/training/

# Launch training (12-16 hours on RTX 5060 Ti)
ssh scawful@100.104.53.21
cd D:/training
python train_qwen2.5_coder.py --dataset alttp_yaze_full_distributed
```

## Monitoring

### Check Pilot Progress
```bash
# Watch output
tail -f /tmp/claude/-Users-scawful-Code-hafs/tasks/b87a73d.output | grep -E "Progress|Checkpoint|Quality"

# Check files
ls -lh ~/.context/training/datasets/aggressive_pilot_1000*/
```

### Check Campaign Progress (when running)
```bash
# Watch checkpoints
watch -n 10 'ls -lh ~/.context/training/checkpoints/'

# Monitor output
tail -f campaign.log | grep -E "Checkpoint|samples|Quality"
```

## Expected Output Structure

```
~/.context/training/datasets/alttp_yaze_full_distributed/
├── train.jsonl          # 27,600 samples (80%)
├── val.jsonl            # 3,450 samples (10%)
├── test.jsonl           # 3,450 samples (10%)
├── metadata.json        # Dataset info
└── stats.json           # Quality metrics

alttp_yaze_full_distributed_asm/
├── train.jsonl          # ~19,200 samples
├── val.jsonl            # ~2,400 samples
├── test.jsonl           # ~2,400 samples
└── metadata.json

alttp_yaze_full_distributed_yaze/
├── train.jsonl          # ~5,600 samples
├── val.jsonl            # ~700 samples
├── test.jsonl           # ~700 samples
└── metadata.json
```

## Fallback Plans

### If medical-mechanica is unavailable:
- Falls back to Gemini-only (still 10x parallel)
- Duration increases to ~15-20 hours
- Still 2x faster than sequential

### If Gemini quota is hit:
- Can route more to medical-mechanica
- Adjust distribution in `distributed_generator.py`
- Slower but will complete

## Quality Validation

All samples validated with:
- ✅ Domain-specific validators (ASM, C++, KG)
- ✅ Hallucination detection (code-aware)
- ✅ Coherence scoring (pattern-based for code)
- ✅ KG consistency (hardware register exemptions)
- ✅ Deduplication (>0.95 similarity threshold)
- ✅ Domain-specific quality thresholds

**Proven:** Alpha pilot achieved 100% pass rate (20/20)
