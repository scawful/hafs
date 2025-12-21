# Model Training Plan - Review

**Reviewed**: 2025-12-21
**Based on**: docs/architecture/MOE_SYSTEM.md + docs/training/TRAINING_PREPARATION.md

---

## Model Names (LOCKED)

### Primary Models (Oracle Naming Schema)

**ASM Expert:**
```
Full name:    oracle-rauru-assembler-qwen3-coder-14b-20251221
Short name:   oracle-rauru-assembler
Legacy alias: hyrule-asm-v1

Purpose:      65816 assembly routines, hooks, patches
Base model:   qwen2.5-coder:14b (14.8B parameters)
Dataset:      ALTTP ASM (24K samples)
Training:     8-12 hours on medical-mechanica
```

**YAZE Expert:**
```
Full name:    oracle-yaze-expert-qwen3-coder-14b-20251221
Short name:   oracle-yaze-expert
Legacy alias: yaze-sage-v1

Purpose:      YAZE-specific workflows and C++ API usage
Base model:   qwen2.5-coder:14b (14.8B parameters)
Dataset:      YAZE Tools (7K samples)
Training:     3-5 hours on medical-mechanica
```

**MoE System:**
```
Full name:    oracle-council-moe-v1
Short names:  triforce-moe-v1, hyrule-council-v1

Purpose:      Mixture of Experts orchestrator
Experts:      [rauru-assembler, yaze-expert, sheik-debugger]
Classifier:   task-router-v1 (Gemini Flash for routing)
```

### Future Oracle Experts (Planned)

**Oracle of Secrets Story Experts:**
- `oracle-nayru-canon` - Lore bible, continuity, timeline (gemma3-12b-it)
- `oracle-zelda-plotweaver` - Plot arcs, reveals, pacing (gemma3-12b-it)
- `oracle-farore-pathfinder` - Quests, world layout (gemma3-12b-it)
- `oracle-din-forge` - Combat balance, item tuning (gemma3-12b-it)
- `oracle-saria-voice` - Dialogue, character voice (gemma3-12b-it)
- `oracle-impa-archivist` - Consistency checks, citations (gemma3-12b-it)

**ROM Tooling Experts:**
- `oracle-rauru-assembler` - ✅ TRAINING NOW (65816 routines)
- `oracle-sheik-debugger` - Crash triage, trace reading
- `oracle-purah-profiler` - Performance, WRAM/VRAM/ROM map
- `oracle-kaepora-banker` - Bank layout, freespace strategy
- `oracle-robbie-toolsmith` - Core ROM tooling workflows
- `oracle-yaze-expert` - ✅ TRAINING NOW (YAZE C++ API)

---

## Training Configuration

### Hardware: medical-mechanica
- **GPU**: RTX 5060 Ti (16GB VRAM)
- **IP**: 100.104.53.21:11434
- **Platform**: Windows + Ollama + WSL2
- **Available models**: qwen3:14b, deepseek-r1:14b, magistral:24b

### Training Specs

**Model**: qwen2.5-coder:14b (14.8B parameters)
**Quantization**: 4-bit (fits in 16GB)
**Adapter**: LoRA (rank 16)
**Batch size**: 2 (with gradient accumulation 4 → effective batch 8)
**Optimizer**: AdamW with cosine schedule
**Epochs**: 3

**VRAM Budget:**
```
Model (4-bit):        ~8GB
LoRA adapters:        ~2GB
Optimizer states:     ~3GB
Batch processing:     ~2GB
─────────────────────────
TOTAL:                ~15GB (fits in 16GB with 1GB headroom) ✓
```

---

## Datasets

### oracle-rauru-assembler (ASM Expert)

**Training Data:**
- ALTTP ASM: 15,000 samples (vanilla disassembly)
- Gigaleak: 8,000 samples (Nintendo original source)
- Oracle: 4,000 samples (ROM hack code)
- Errors: 1,500 samples (debugging, error handling)

**Total**: ~28,000 samples → split to 24K train / 2K val / 2K test

**Domains**:
- 65816 assembly routines
- Memory mapping (WRAM, ROM, SRAM)
- Bank allocation strategies
- Hook/patch generation
- Code optimization

**Training Time**: 8-12 hours

### oracle-yaze-expert (YAZE Tools Expert)

**Training Data:**
- YAZE Tools: 6,000 samples (C++ API calls, workflows)
- Errors: 1,000 samples (tool-specific debugging)

**Total**: ~7,000 samples → split to 5.6K train / 700 val / 700 test

**Domains**:
- YAZE C++ API usage
- ROM file manipulation
- Graphics format conversion (2bpp, 3bpp, 4bpp, 8bpp)
- Compression/decompression (LC_LZ2, Hyrule Magic)
- Map and dungeon editing
- Tile systems (8x8 → 16x16 → 32x32)

**Training Time**: 3-5 hours

---

## Quality Thresholds (Fixed)

```python
DOMAIN_THRESHOLDS = {
    "asm": 0.4,       # ASM is hard - lower threshold
    "gigaleak": 0.5,  # Original source - medium
    "oracle": 0.4,    # ROM hack - lower
    "yaze": 0.5,      # C++ code - medium
    "errors": 0.3,    # Diagnostics - lowest
    "text": 0.6,      # Natural language - higher
}
```

**Quality Pipeline (Fixed)**:
- ✅ Domain-aware coherence (code pattern detection, not word overlap)
- ✅ Hallucination check skips LLM for code domains
- ✅ Hardware registers exempt from KG validation
- ✅ Domain-specific thresholds applied automatically

**Proven**: Alpha pilot achieved 100% pass rate (20/20 samples)

---

## Training Pipeline

### Phase 1: Data Generation (IN PROGRESS)
```bash
# Aggressive pilot (1000 samples) - RUNNING
# ETA: 10-15 minutes
# Status: Validating quality fixes at scale

# Full campaign (34,500 samples) - READY
# Run: python run_distributed_campaign.py
# Duration: 8-12 hours (Gemini 70% + medical-mechanica 30%)
```

### Phase 2: Dataset Export
```bash
# After generation completes, export to training format

python -m agents.training.scripts.export_datasets \
    --source ~/.context/training/datasets/alttp_yaze_full_distributed \
    --output ~/training_data/ \
    --format unsloth

# Creates:
# - oracle_rauru_assembler_24k/ (train.jsonl, val.jsonl, test.jsonl)
# - oracle_yaze_expert_7k/ (train.jsonl, val.jsonl, test.jsonl)
```

### Phase 3: Transfer to medical-mechanica
```bash
# Copy datasets to Windows machine
scp -r ~/training_data/ scawful@100.104.53.21:D:/training/

# Verify transfer
ssh scawful@100.104.53.21 "dir D:\training"
```

### Phase 4: Training (Sequential)
```bash
# SSH into medical-mechanica
ssh scawful@100.104.53.21

# Train ASM expert (8-12 hours)
cd D:\training
python train_oracle_rauru_assembler.py \
    --dataset oracle_rauru_assembler_24k \
    --base-model qwen2.5-coder:14b \
    --output oracle-rauru-assembler-20251221 \
    --lora-rank 16 \
    --batch-size 2 \
    --grad-accum 4 \
    --epochs 3

# Train YAZE expert (3-5 hours)
python train_oracle_yaze_expert.py \
    --dataset oracle_yaze_expert_7k \
    --base-model qwen2.5-coder:14b \
    --output oracle-yaze-expert-20251221 \
    --lora-rank 16 \
    --batch-size 2 \
    --grad-accum 4 \
    --epochs 3

# Total: 12-16 hours
```

### Phase 5: Validation
```bash
# Test trained models on held-out test sets
python validate_oracle_models.py \
    --models oracle-rauru-assembler-20251221,oracle-yaze-expert-20251221 \
    --test-sets oracle_rauru_assembler_24k/test.jsonl,oracle_yaze_expert_7k/test.jsonl

# Expected metrics:
# - Perplexity: < 3.5 (lower is better)
# - Accuracy: > 0.35 (on code generation)
# - BLEU score: > 0.25 (on exact matches)
```

### Phase 6: Deployment
```bash
# Copy LoRA adapters back to Mac
scp -r scawful@100.104.53.21:D:/training/oracle-*-20251221/ \
    ~/.context/models/

# Update model registry
python -m hafs.scripts.register_model \
    --name oracle-rauru-assembler \
    --version 20251221 \
    --adapter ~/.context/models/oracle-rauru-assembler-20251221/

python -m hafs.scripts.register_model \
    --name oracle-yaze-expert \
    --version 20251221 \
    --adapter ~/.context/models/oracle-yaze-expert-20251221/

# Test MoE system with trained models
python -m hafs.agents.moe.test_moe --use-trained
```

---

## MoE Routing Table

### Task Classification Keywords

**oracle-rauru-assembler** (ASM Expert):
- Keywords: asm, assembly, routine, hook, patch, bank, memory, optimization, 65816
- Confidence threshold: 0.75
- Example: "Write a routine that checks if Link has the Master Sword"

**oracle-yaze-expert** (YAZE Tools):
- Keywords: yaze, rom, graphics, sprite, tile, map, editor, tool, compression
- Confidence threshold: 0.70
- Example: "Load custom sprite graphics at offset 0x80000 using YAZE"

**oracle-sheik-debugger** (Debug - Future):
- Keywords: error, bug, crash, fix, debug, problem, trace, diagnostic
- Confidence threshold: 0.80
- Example: "My ROM hack crashes when entering room $45 in dungeon 3"

### Multi-Expert Tasks

For complex tasks requiring multiple experts:
```
User: "Create a new boss enemy with custom sprite graphics and AI routine"

Classification:
  - oracle-yaze-expert (0.85) - handles graphics loading
  - oracle-rauru-assembler (0.90) - handles AI routine

Synthesis:
  → Integrated solution with both graphics and code
```

---

## Performance Expectations

### Training Performance
- ASM Expert: ~2-3 tokens/sec on RTX 5060 Ti
- YAZE Expert: ~2-3 tokens/sec on RTX 5060 Ti
- Total time: 12-16 hours for both models

### Inference Performance (with LoRA adapters)
- Single expert call: ~1-2 seconds
- Multi-expert (parallel): ~2-3 seconds
- MoE synthesis: ~4-5 seconds total

### Quality Targets
- Perplexity: < 3.5 (code generation)
- Accuracy: > 0.35 (exact match on test set)
- User satisfaction: > 80% (qualitative)
- Better than base model: Yes (measured on ROM hacking benchmarks)

---

## Checkpointing Strategy

**Every 100 steps:**
```python
checkpoint = {
    "epoch": current_epoch,
    "step": current_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": current_loss,
    "best_loss": best_loss,
    "timestamp": time.time(),
}
torch.save(checkpoint, f"checkpoint_epoch{epoch}_step{step}.pt")
```

**Resume from failure:**
```bash
# Automatically resumes from latest checkpoint
python train_oracle_rauru_assembler.py \
    --resume-from-latest \
    --checkpoint-dir D:/training/checkpoints/
```

---

## Summary

**Models to Train:**
1. ✅ `oracle-rauru-assembler` - ASM expert (24K samples, 8-12 hrs)
2. ✅ `oracle-yaze-expert` - YAZE expert (7K samples, 3-5 hrs)

**Training Platform:**
- medical-mechanica (RTX 5060 Ti 16GB)
- qwen2.5-coder:14b base model
- 4-bit quantization + LoRA (rank 16)

**Expected Duration:**
- Total: 12-16 hours (can run overnight)

**Next Steps:**
1. ✅ Wait for aggressive pilot to validate (1000 samples)
2. Launch full campaign (34.5K samples, 8-12 hours)
3. Export datasets to Unsloth format
4. Transfer to medical-mechanica
5. Train both experts sequentially
6. Validate on test sets
7. Deploy to production MoE system

**Status**: Ready to proceed once pilot validates quality fixes.
