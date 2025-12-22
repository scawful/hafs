# Oracle-Farore-Secrets: The North Star Model

**Created**: 2025-12-22
**Status**: PLANNED
**Purpose**: Ultimate integration of Oracle of Secrets storytelling + YAZE ROM hacking

---

## Vision

**oracle-farore-secrets** is the pinnacle model - a unified expert that combines:
- **Narrative Design** (Oracle of Secrets trilogy lore, quests, dialogue)
- **ROM Hacking** (YAZE tools, 65816 ASM, SNES hardware)
- **Multi-Modal Synthesis** (story → implementation → validation)

This model represents the **north star** for our training efforts - a single model that can:
1. Design quest narratives with proper pacing and reveals
2. Implement them in ROM using YAZE and ASM
3. Balance mechanics and difficulty
4. Validate against canon and technical constraints

---

## Why This Is The Goal

### Current State: Fragmented Expertise

**Problem**: We have separate experts:
- Story experts understand lore but not ROM limitations
- ROM experts understand tools but not narrative design
- MoE synthesis adds latency and complexity

**oracle-farore-secrets solves this** by embedding **both domains** in a single model.

### Advantages Over MoE

**MoE** (Mixture of Experts):
- ✅ Modular, easier to train individual experts
- ✅ Can swap experts independently
- ❌ Requires routing/classification overhead
- ❌ Synthesis step can lose context
- ❌ Higher inference latency (3-5 seconds)
- ❌ Harder to maintain cross-domain reasoning

**oracle-farore-secrets** (Unified Model):
- ✅ Single-shot reasoning across domains
- ✅ Lower inference latency (1-2 seconds)
- ✅ Natural cross-domain synthesis
- ❌ Harder to train (needs diverse dataset)
- ❌ Larger model required (14B-34B parameters)
- ❌ More expensive to fine-tune

### When to Use MoE vs Unified

**Use MoE When**:
- Tasks are clearly separable (pure ASM vs pure lore)
- Need to update one domain without retraining everything
- Smaller models (7B each) are sufficient

**Use oracle-farore-secrets When**:
- Tasks require cross-domain reasoning (quest design → ROM implementation)
- Latency matters (real-time assistance)
- Want seamless integration without synthesis step

**Recommendation**: Train **both**:
1. MoE experts (oracle-rauru-assembler, oracle-yaze-expert, oracle-nayru-canon) for focused tasks
2. oracle-farore-secrets for integrated workflows

---

## Model Specifications

### Base Model Options

**Option 1: Qwen2.5-Coder-32B** (RECOMMENDED)
- **Parameters**: 32B
- **Context**: 32K tokens
- **Strengths**: Code + reasoning, multilingual
- **VRAM**: ~20GB (4-bit quantization)
- **Training**: Possible on medical-mechanica with DeepSpeed ZeRO-3 offloading
- **Why**: Best balance of code and natural language

**Option 2: Magistral-24B**
- **Parameters**: 24B
- **Context**: 32K tokens
- **Strengths**: Creative writing, synthesis
- **VRAM**: ~15GB (4-bit)
- **Training**: Easier fit on medical-mechanica
- **Why**: Excellent for narrative + code fusion

**Option 3: DeepSeek-R1-32B**
- **Parameters**: 32B
- **Context**: 64K tokens
- **Strengths**: Reasoning, planning
- **VRAM**: ~20GB (4-bit)
- **Why**: Strong reasoning chains for complex tasks

**Chosen**: **Qwen2.5-Coder-32B** (code-first with strong reasoning)

### Training Configuration

```python
model = "qwen2.5-coder:32b"
quantization = "4-bit"  # Fits in 16GB with ZeRO-3
lora_rank = 32          # Higher rank for complex task
lora_alpha = 64
batch_size = 1
gradient_accumulation = 8  # Effective batch = 8
epochs = 3
learning_rate = 2e-5
warmup_steps = 100
optimizer = "AdamW"
scheduler = "cosine"
```

**VRAM Budget**:
```
Model (4-bit + ZeRO-3):    ~10GB (offloaded)
LoRA adapters:             ~4GB
Optimizer states:          ~2GB (offloaded)
Batch processing:          ~2GB
──────────────────────────────
TOTAL:                     ~18GB (fits in 16GB with offloading)
```

---

## Dataset Construction

### Total Dataset Size: ~60,000 Samples

#### Narrative Design (20,000 samples)

**Sources**:
1. **Oracle of Seasons/Ages Analysis** (5,000 samples)
   - Quest breakdowns
   - Dialogue patterns
   - Pacing analysis
   - Difficulty: MEDIUM (requires manual annotation)
   - HITL: HIGH (need expert validation of quest structure)

2. **ALTTP Story Beats** (3,000 samples)
   - Plot arcs
   - Character development
   - Narrative pacing
   - Difficulty: MEDIUM (analyze existing game)
   - HITL: MEDIUM (validate against canon)

3. **Zelda Lore Synthesis** (5,000 samples)
   - Timeline continuity
   - Mythos consistency
   - Canon verification
   - Difficulty: HIGH (requires deep lore knowledge)
   - HITL: VERY HIGH (expert validation critical)

4. **Synthetic Quest Generation** (7,000 samples)
   - Gemini 3.0 Flash generates quest outlines
   - Validated against templates
   - Difficulty: LOW (LLM generation)
   - HITL: MEDIUM (quality filtering)

#### ROM Hacking (30,000 samples)

**Sources**:
1. **ALTTP ASM** (15,000 samples) ✅ READY
   - 65816 routines
   - Memory mapping
   - Hook generation
   - Difficulty: LOW (automated extraction)
   - HITL: LOW (verified via assembly)

2. **Gigaleak Nintendo Source** (8,000 samples) ✅ READY
   - Original ALTTP code
   - Hardware routines
   - Difficulty: LOW (automated)
   - HITL: LOW (canonical source)

3. **YAZE Tools Workflows** (7,000 samples) ✅ READY
   - C++ API usage
   - Graphics pipelines
   - Difficulty: LOW (API documentation)
   - HITL: LOW (executable validation)

#### Cross-Domain Integration (10,000 samples)

**Sources**:
1. **Quest → Implementation Pairs** (5,000 samples)
   - Quest description → ASM hooks + YAZE edits
   - Narrative goal → Technical implementation
   - Difficulty: VERY HIGH (manual pairing required)
   - HITL: VERY HIGH (expert validation)
   - **This is the critical dataset for integration**

2. **ROM Hack Case Studies** (3,000 samples)
   - Parallel Worlds analysis
   - Goddess of Wisdom analysis
   - Ancient Stone Tablets analysis
   - Difficulty: HIGH (reverse engineering)
   - HITL: HIGH (validate interpretations)

3. **Error Correction Chains** (2,000 samples)
   - Story proposal → Technical constraint → Revised story
   - ASM implementation → Narrative issue → Revised code
   - Difficulty: VERY HIGH (requires expert iteration)
   - HITL: VERY HIGH (multi-turn validation)

---

## Data Generation Difficulty Assessment

### Easy (Automated, Low HITL)

**Samples**: 30,000 (50% of dataset)
**Effort**: 1-2 weeks

| Source | Count | Method | HITL |
|--------|-------|--------|------|
| ALTTP ASM | 15,000 | Existing generators | Validation only |
| Gigaleak | 8,000 | Existing generators | Validation only |
| YAZE Tools | 7,000 | Existing generators | Validation only |

**Status**: ✅ READY (already generated for oracle-rauru-assembler / oracle-yaze-expert)

---

### Medium (Semi-Automated, Moderate HITL)

**Samples**: 15,000 (25% of dataset)
**Effort**: 2-4 weeks

| Source | Count | Method | HITL |
|--------|-------|--------|------|
| Oracle of Seasons/Ages | 5,000 | Script analysis + LLM annotation | Validation + filtering |
| ALTTP Story Beats | 3,000 | Game script extraction + analysis | Validation |
| Synthetic Quests | 7,000 | Gemini 3.0 Flash/Pro generation | Quality filtering (30% reject rate) |

**Difficulty**:
- ⚠️ Requires building quest extraction pipelines
- ⚠️ LLM-generated quests need quality validation
- ⚠️ Story beat analysis needs canonical verification

**Recommended Approach**:
1. Build quest extraction pipeline (1 week)
2. Run Gemini 3.0 Flash/Pro to generate synthetic quests (2 days)
3. Manual filtering and validation (1 week, 2-3 hours/day)

---

### Hard (Manual, High HITL)

**Samples**: 15,000 (25% of dataset)
**Effort**: 4-8 weeks

| Source | Count | Method | HITL |
|--------|-------|--------|------|
| Zelda Lore Synthesis | 5,000 | Expert annotation | Expert validation (100%) |
| ROM Hack Case Studies | 3,000 | Manual reverse engineering | Expert validation |
| Quest → Implementation | 5,000 | Manual pairing | Expert validation (100%) |
| Error Correction Chains | 2,000 | Expert iteration | Expert validation (100%) |

**Difficulty**:
- 🔴 **Quest → Implementation Pairs** is the hardest
  - Requires expert to design quest AND implement in ROM
  - Each sample = 30-60 minutes of work
  - 5,000 samples = 2,500-5,000 hours (312-625 days at 8hr/day)
  - **CRITICAL BOTTLENECK**

**Mitigation Strategies**:

**Strategy 1: Reduce Sample Count**
- Target 1,000 high-quality pairs instead of 5,000
- Focus on diverse patterns (tutorial, dungeon, overworld, boss)
- Effort: 500-1,000 hours (62-125 days)

**Strategy 2: Synthetic Generation with Expert Validation**
- LLM generates quest → implementation pairs
- Expert validates and corrects (10-15 min/sample)
- Reject rate: ~50%
- Effort: 1,250-1,875 hours (156-234 days) for 5,000 samples

**Strategy 3: Community Sourcing**
- Recruit ROM hacking community to contribute samples
- Provide templates and guidelines
- Quality control via peer review
- Effort: Unknown, depends on participation

**Strategy 4: Iterative Refinement**
- Start with 500 expert pairs
- Train intermediate model
- Use model to generate more pairs (with expert correction)
- Bootstrap to 5,000 samples
- Effort: 250-500 hours initial + validation overhead

**RECOMMENDED**: **Strategy 4** (Iterative Refinement)
- Most realistic for solo/small team
- Leverages model to accelerate data generation
- Quality improves over iterations

---

## Training Timeline

### Phase 1: Easy Data (Weeks 1-2)

**Goal**: Validate training pipeline with existing data

**Tasks**:
1. Combine existing ASM + YAZE datasets (30K samples)
2. Train baseline oracle-farore-secrets model
3. Evaluate on pure ASM and pure YAZE tasks
4. Verify medical-mechanica can handle 32B model with ZeRO-3

**Outcome**: Working training pipeline + baseline model

---

### Phase 2: Medium Data (Weeks 3-6)

**Goal**: Add narrative capabilities

**Tasks**:
1. Build quest extraction pipeline (Week 3)
2. Extract Oracle of Seasons/Ages quests (Week 4)
3. Extract ALTTP story beats (Week 4)
4. Generate synthetic quests with Gemini (Week 5)
5. Quality filtering and validation (Week 6)
6. Train oracle-farore-secrets-v2 (Week 6)

**Outcome**: Model with narrative + ROM capabilities (separate domains)

---

### Phase 3: Hard Data - Iterative Refinement (Weeks 7-16)

**Goal**: Cross-domain integration

**Iteration 1 (Weeks 7-8):**
- Create 100 expert quest → implementation pairs
- Train oracle-farore-secrets-v3
- Test on quest implementation tasks

**Iteration 2 (Weeks 9-10):**
- Use v3 to generate 400 more pairs
- Expert correction (10 min/sample = 67 hours)
- Train oracle-farore-secrets-v4 with 500 total pairs

**Iteration 3 (Weeks 11-12):**
- Use v4 to generate 500 more pairs
- Expert correction
- Train oracle-farore-secrets-v5 with 1,000 total pairs

**Iteration 4 (Weeks 13-14):**
- Use v5 to generate 1,000 more pairs
- Expert correction
- Train oracle-farore-secrets-v6 with 2,000 total pairs

**Iteration 5 (Weeks 15-16):**
- Use v6 to generate 1,000 more pairs
- Expert correction
- Train oracle-farore-secrets-v7 with 3,000 total pairs

**Outcome**: Production-ready integrated model

---

### Phase 4: Error Correction & Polish (Weeks 17-20)

**Goal**: Refine cross-domain reasoning

**Tasks**:
1. Generate error correction chains (2,000 samples)
2. ROM hack case study analysis (3,000 samples)
3. Final training run: oracle-farore-secrets-v8
4. Comprehensive evaluation

**Outcome**: **oracle-farore-secrets-v1.0** - production release

---

## Evaluation Plan

### Benchmark Categories

**1. Pure ASM Tasks**
- Baseline: oracle-rauru-assembler performance
- Target: ≥95% of specialist performance
- Metrics: Assembly correctness, hook placement

**2. Pure YAZE Tasks**
- Baseline: oracle-yaze-expert performance
- Target: ≥95% of specialist performance
- Metrics: API call correctness, workflow validity

**3. Pure Narrative Tasks**
- Baseline: oracle-nayru-canon performance
- Target: ≥90% of specialist performance
- Metrics: Canon consistency, pacing quality

**4. Cross-Domain Integration** (CRITICAL)
- Baseline: MoE synthesis performance
- Target: ≥110% of MoE (better due to unified reasoning)
- Metrics:
  - Quest → Implementation correctness
  - Technical constraint awareness in narratives
  - Narrative quality in technical implementations

**5. Latency**
- Baseline: MoE system (3-5 seconds)
- Target: <2 seconds (single model inference)

---

## Exceeding MoE Performance

### Where Unified Model Wins

**1. Cross-Domain Reasoning**
- MoE: Story expert doesn't know ROM limits → revision loop
- Unified: Knows both → correct first time

**2. Context Retention**
- MoE: Context lost in synthesis step
- Unified: Full context throughout

**3. Latency**
- MoE: Route (0.5s) + Expert (1.5s) + Synth (2s) = 4s
- Unified: Single inference (1.5s)

**4. Coherence**
- MoE: Potential conflicts between expert outputs
- Unified: Single coherent voice

### Where MoE Wins

**1. Modularity**
- MoE: Update ASM expert without retraining story expert
- Unified: Must retrain entire 32B model

**2. Specialization**
- MoE: Each 14B expert highly specialized
- Unified: 32B model split across domains

**3. Training Cost**
- MoE: Train 3×14B models separately (cheaper)
- Unified: Train 1×32B model (more expensive)

### Hybrid Strategy (RECOMMENDED)

**Use Both**:
- **MoE** for pure tasks (ASM-only, story-only)
- **oracle-farore-secrets** for integrated workflows
- Route based on task complexity

**Routing Logic**:
```
if task.is_pure_asm or task.is_pure_story:
    use MoE (faster, cheaper)
elif task.requires_cross_domain_reasoning:
    use oracle-farore-secrets (better quality)
else:
    default to oracle-farore-secrets
```

---

## Success Criteria

### Minimum Viable Model (MVP)

**Requirements**:
1. ≥90% ASM task accuracy (vs oracle-rauru-assembler)
2. ≥90% YAZE task accuracy (vs oracle-yaze-expert)
3. ≥80% narrative task accuracy (vs oracle-nayru-canon)
4. ≥70% quest → implementation accuracy (vs expert)
5. <2 seconds inference latency

**Training Data Needed**:
- 30K ASM/YAZE samples (READY)
- 10K narrative samples (Medium difficulty)
- 500 quest → implementation pairs (Iterative refinement)

**Timeline**: 12 weeks

---

### Production Model (v1.0)

**Requirements**:
1. ≥95% ASM task accuracy
2. ≥95% YAZE task accuracy
3. ≥90% narrative task accuracy
4. ≥85% quest → implementation accuracy
5. Outperforms MoE on integrated tasks
6. <1.5 seconds inference latency

**Training Data Needed**:
- All MVP data +
- 15K narrative samples
- 3,000 quest → implementation pairs
- 2,000 error correction chains

**Timeline**: 20 weeks

---

## Risk Mitigation

### Risk 1: Quest → Implementation Pairs Too Hard to Generate

**Mitigation**:
- **Plan A**: Iterative refinement (use model to help generate)
- **Plan B**: Reduce target to 1,000 pairs
- **Plan C**: Focus on simpler quest types (fetch, talk, trigger)
- **Plan D**: Community sourcing via ROM hacking forums

**Fallback**: Train separate experts, use MoE for complex tasks

---

### Risk 2: 32B Model Won't Fit on medical-mechanica

**Mitigation**:
- **Plan A**: DeepSpeed ZeRO-3 with CPU offloading
- **Plan B**: Use Magistral-24B instead
- **Plan C**: Rent cloud GPU (A100 80GB) for training
- **Plan D**: Multi-GPU setup (add second GPU)

**Fallback**: Train 14B unified model (reduced capacity)

---

### Risk 3: Cross-Domain Performance Degrades

**Mitigation**:
- **Plan A**: Adjust dataset ratios (more cross-domain samples)
- **Plan B**: Curriculum learning (pure tasks first, then integration)
- **Plan C**: Multi-task learning with task-specific LoRA
- **Plan D**: Stick with MoE for production

**Fallback**: Use oracle-farore-secrets for integrated tasks only

---

## Summary

**oracle-farore-secrets** is the **north star** - a unified 32B model combining narrative design and ROM hacking.

**Key Insights**:
- **Easy data** (50%): ✅ Ready now
- **Medium data** (25%): 2-4 weeks of pipeline building
- **Hard data** (25%): 12-16 weeks of iterative refinement
- **Critical bottleneck**: Quest → Implementation pairs

**Recommended Approach**:
1. Train MVP with 500 cross-domain pairs (12 weeks)
2. Evaluate vs MoE
3. If promising, continue to v1.0 (20 weeks total)
4. If not, use MoE for complex tasks

**Hybrid Strategy**:
- Keep MoE experts for pure tasks
- Use oracle-farore-secrets for integrated workflows
- Route intelligently based on task complexity

**Next Steps**:
1. Validate 32B model fits on medical-mechanica
2. Create 100 quest → implementation pairs (pilot)
3. Train oracle-farore-secrets-pilot
4. Evaluate and decide: MVP or MoE-only

---

**This is an ambitious goal, but achievable with iterative refinement and realistic scoping.**
