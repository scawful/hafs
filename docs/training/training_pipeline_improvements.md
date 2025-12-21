# Training Data Pipeline - Improvement Roadmap

> Generated: 2025-12-21
> Status: Active Development
> Models: Gemini 3 Pro/Flash, Claude Opus 4.5, GPT-5.2, Ollama (medical-mechanica node)

This document captures ideas and strategies for improving training data creation in hafs.

---

## Executive Summary

The hafs codebase has infrastructure to become a **self-improving training data factory**. The key is closing feedback loops between:

1. **Failure detection** → **Sample generation** → **Quality feedback**
2. **History logging** → **Pattern extraction** → **Workflow templates**
3. **Knowledge graph** → **Entity validation** → **Semantic alignment**
4. **Agent memory** → **Experience sharing** → **Collective learning**

**Estimated potential:** ~8,400 high-quality training samples per month (vs. ~300 baseline)

---

## 1. Existing Agents to Leverage

### 1.1 SelfImprovementAgent
**File:** `src/agents/autonomy/self_improvement.py`

**Current Capability:**
- Tracks tool failures and system friction
- Aggregates failure metrics (top failing tools, error messages, affected files)
- Generates improvement recommendations

**Training Data Opportunity:**
- Generate failure-recovery sample pairs
- Extract system health patterns
- Create error classification data
- **Output:** ~50-100 samples/week from real failures

### 1.2 HallucinationWatcher
**File:** `src/agents/autonomy/hallucination.py`

**Current Capability:**
- Flags unsubstantiated claims
- Checks for tool evidence and file references
- Detects claim-without-action patterns

**Training Data Opportunity:**
- Generate hallucination detection samples
- Create properly grounded examples
- Build claim-evidence alignment data
- **Output:** ~200-300 samples/month

### 1.3 SelfHealingAgent
**File:** `src/agents/autonomy/self_healing.py`

**Current Capability:**
- Scans logs for errors and exceptions
- Detects service crashes
- Logs error patterns with recovery actions

**Training Data Opportunity:**
- Generate troubleshooting samples (error → diagnosis → fix)
- Create operational runbooks
- Extract recurring error patterns with resolutions
- **Output:** ~100-150 samples/month

### 1.4 CuriosityExplorer
**File:** `src/agents/autonomy/curiosity.py`

**Current Capability:**
- Analyzes git history and TODOs
- Generates exploration topics
- Proposes research directions

**Training Data Opportunity:**
- Generate exploration prompts
- Create research planning samples
- Build topic modeling data
- **Output:** ~50-100 samples/month

### 1.5 Mission Agents
**File:** `src/agents/mission/mission_agents.py`

**Current Capability:**
- Deep research on specific domains
- Generate semantic search queries
- Discover cross-references and patterns
- Create discovery records with evidence

**Training Data Opportunity:**
- Generate research instruction samples
- Create pattern recognition samples
- Build cross-reference discovery data
- **Output:** ~300-500 samples/week (highly valuable for reasoning)

---

## 2. Untapped Data Sources

### 2.1 History Logger & Sessions
**Files:**
- `src/hafs/core/history/logger.py` - ULID-based operation logging
- `src/hafs/core/history/session.py` - Session management

**Potential:**
- Tool call sequences → workflow templates
- Operation success patterns → precondition learning
- Error-to-recovery mapping → diagnostic samples
- Duration patterns → performance expectations
- Provenance chains → full traceability

**Estimated output:** 500-1000 workflow samples/month

### 2.2 Agent Memory (Episodic)
**Files:**
- `src/hafs/core/history/agent_memory.py` - Temporal memory buckets
- `src/agents/utility/episodic.py` - Episodic indexing

**Potential:**
- Session summaries with outcomes
- Memory type distribution patterns
- Cross-agent learning signals
- Context reconstruction samples

**Estimated output:** 100-200 contextual samples/week

### 2.3 Shadow Observer & Trend Watcher
**Files:**
- `src/agents/utility/shadow_observer.py` - Shell command observation
- `src/agents/utility/trend_watcher.py` - Activity stream analysis

**Potential:**
- Command intent patterns ("cd repo" + "git status" → diagnosis)
- Workflow detection from command sequences
- Topic evolution from work history
- Context switching patterns

**Estimated output:** 300-500 samples/month

### 2.4 Streaming Index & Embeddings
**Files:**
- `src/hafs/core/streaming_index.py` - Real-time embedding index
- `src/agents/training/quality.py` - Quality scoring

**Potential:**
- Similarity-based deduplication learning
- Embedding quality feedback loops
- Diversity scoring patterns
- Semantic clustering for balanced datasets

**Estimated output:** 200-400 samples/week

### 2.5 Knowledge Graph
**File:** `src/agents/knowledge/graph.py`

**Potential:**
- Entity extraction from code
- Relationship inference
- Code-to-NL alignment
- Domain-specific terminology
- Cross-KB validation

**Estimated output:** 500-1000 alignment samples/week

---

## 3. Quality Assurance Gaps

### Current QA (in `quality.py`)
- ✅ Diversity scoring (embedding distance)
- ✅ KG consistency validation
- ✅ Hallucination risk detection
- ✅ Semantic coherence scoring

### Missing Components

#### 3.1 Feedback Loop Integration
**Problem:** Currently one-way (generate → score → filter)
**Solution:** Bidirectional feedback
- Track which samples correlate with model failures
- Adjust quality thresholds based on downstream impact
- Store in `QualityFeedbackTracker`

#### 3.2 Failure Categorization
**Problem:** No distinction between fixable vs. systemic issues
**Solution:** Failure taxonomy
- Tag rejection reasons in memory
- Feed patterns back to generators
- New file: `quality_feedback.py`

#### 3.3 Quality Drift Detection
**Problem:** Can't detect quality degradation over time
**Solution:** Temporal quality analysis
- Track metrics with timestamps
- Alert when distributions shift
- Enhanced `CurationStats`

#### 3.4 Domain-Specific Validation
**Problem:** Generic coherence doesn't catch domain errors
**Solution:** Pluggable validators
- ASM: instruction validity, addressing modes
- C++: syntax, compile check
- Knowledge: entity presence, link validity

---

## 4. Proposed New Systems

### 4.1 Error-to-Sample Pipeline (HIGH PRIORITY)
**Status:** Implementing

Generate training samples from system failures:
```
SelfHealingAgent → Error detected
                 → Extract context
                 → Generate "how to fix" sample
                 → Validate with KG
                 → Store with high importance
```

### 4.2 History Pattern Miner
Extract workflow patterns from logs:
```
HistoryLogger → Successful tool chains
             → Session summaries
             → Failed → recovered sequences
             → Generate workflow samples
```

### 4.3 Multi-Teacher Validation
Cross-validate with multiple models:
```python
teachers = ["gemini-2.0-flash", "claude-3-haiku", "qwen2.5-coder"]
sample = await generate_with_consensus(teachers, threshold=2)
```

### 4.4 Active Learning Sampler
Target generation at embedding gaps:
```
StreamingIndex → Find sparse regions
              → Prioritize generation there
              → Improve coverage
```

### 4.5 Synthetic Augmentation
Generate variations of high-quality samples:
- Paraphrase instructions
- Code mutations (refactored versions)
- Difficulty scaling (simple → complex)

---

## 5. Implementation Roadmap

### Phase 1: Immediate (1-2 weeks)
- [x] Core training pipeline (`base.py`, `curator.py`, `quality.py`)
- [x] Domain generators (ASM, C++, text)
- [x] Training daemon with scheduling
- [x] **Error-to-sample generator** (`error_generator.py`)
- [x] History pattern miner (`history_miner.py`)
- [x] Domain validators (`validators/`)

### Phase 2: Smart Feedback (2-3 weeks)
- [x] Quality feedback tracker (`feedback/quality_tracker.py`)
- [x] Training feedback analysis (`feedback/training_feedback.py`)
- [x] Active learning sampler (`active_learning.py`)
- [ ] SelfImprovement signal extraction
- [ ] Mission agent discovery export
- [ ] Embedding quality feedback loop

### Phase 3: Advanced (3-4 weeks)
- [x] Multi-teacher consensus (`generators/error_generator.py:MultiTeacherGenerator`)
- [x] Active learning sampler (`active_learning.py`)
- [ ] Synthetic augmentation
- [ ] Real-time streaming generator

---

## 6. File Structure

```
src/agents/training/
├── __init__.py                 # ✅ Package exports
├── base.py                     # ✅ DataGenerator ABC, TrainingSample
├── curator.py                  # ✅ DataCurator coordinator
├── quality.py                  # ✅ QualityPipeline (+ validators, feedback, active learning)
├── exporter.py                 # ✅ Unsloth export formats
├── active_learning.py          # ✅ Coverage optimization, sparse region detection
├── generators/
│   ├── __init__.py             # ✅ Generator exports
│   ├── asm_generator.py        # ✅ 65816 ASM
│   ├── cpp_generator.py        # ✅ C++ (yaze)
│   ├── text_generator.py       # ✅ Markdown/text
│   ├── error_generator.py      # ✅ FROM FAILURES + MultiTeacherGenerator
│   └── history_miner.py        # ✅ FROM LOGS (workflow patterns)
├── validators/                  # ✅ Domain-specific validation
│   ├── __init__.py             # ✅ Validator exports
│   ├── base.py                 # ✅ ValidationResult, Validator ABC, CompositeValidator
│   ├── asm_validator.py        # ✅ 65816 instruction validation (mnemonics, addressing modes)
│   ├── cpp_validator.py        # ✅ C++ syntax checks (brackets, compile check)
│   └── kg_validator.py         # ✅ Knowledge graph consistency (entity validation)
├── feedback/                    # ✅ Quality feedback loops
│   ├── __init__.py             # ✅ Feedback exports
│   ├── quality_tracker.py      # ✅ Track quality over time, auto-adjust thresholds
│   └── training_feedback.py    # ✅ Post-training analysis, sample correlation
├── nodes/
│   ├── __init__.py             # ✅ Node exports
│   ├── node_client.py          # ✅ Client for remote training nodes
│   └── node_server.py          # ✅ FastAPI server for Windows GPU node
└── augmentation.py             # ⏳ Synthetic variations (future)
```

---

## 7. Integration Points

| Data Source | File | Output Type | Frequency |
|-------------|------|-------------|-----------|
| Tool Failures | `self_improvement.py` | error-recovery pairs | Weekly |
| Hallucinations | `hallucination.py` | grounding samples | Weekly |
| Service Errors | `self_healing.py` | troubleshooting guides | Daily |
| Research | `mission_agents.py` | discovery samples | Weekly |
| Session History | `logger.py` | workflow templates | Daily |
| Agent Memory | `agent_memory.py` | contextual samples | Weekly |
| Embeddings | `streaming_index.py` | diversity-aware pairs | Real-time |
| Knowledge Graph | `graph.py` | entity-relation samples | Weekly |
| Shell Activity | `shadow_observer.py` | intent inference | Daily |

---

## 8. Metrics to Track

### Data Volume
- Samples generated per domain per day
- Quality pass rate by domain
- Deduplication ratio

### Quality Indicators
- Average quality score trend
- Diversity score distribution
- KG validation pass rate
- Hallucination risk distribution

### Feedback Metrics
- Samples → training run → model improvement correlation
- Rejection reason distribution
- Generator adjustment frequency

### System Health
- Generation latency
- Checkpoint frequency
- Error rate by generator

---

## 9. 2025 Model Configuration

### Cloud Models (Teacher LLMs)

| Provider | Model | Best For | Context |
|----------|-------|----------|---------|
| Gemini | `gemini-3-pro-preview` | Reasoning, Research | 1M tokens |
| Gemini | `gemini-3-flash-preview` | Coding, Fast tasks | 1M tokens |
| Anthropic | `claude-opus-4-5-20251101` | Complex code, Creative | 200k tokens |
| OpenAI | `gpt-5.2` | General, Research | 256k tokens |
| OpenAI | `gpt-5.2-mini` | Fast responses | 128k tokens |

### Local Models (medical-mechanica node - 5060TI 16GB)

| Model | Best For | VRAM |
|-------|----------|------|
| `qwen2.5-coder:14b` | Code generation, review | 12GB |
| `qwen2.5-coder:7b` | Smaller code tasks | 6GB |
| `deepseek-coder-v2-lite` | MoE coding | 8GB |
| `deepseek-r1:8b` | Local reasoning | 8GB |
| `qwen3:8b` | General tasks | 8GB |
| `mistral:7b` | Efficient general | 6GB |
| `gemma3:4b` | Fastest local | 4GB |

### Embedding Models

| Model | Dimensions | Use Case |
|-------|------------|----------|
| `embeddinggemma` | 768 | Primary embeddings |
| `nomic-embed-text` | 768 | Fallback embeddings |

### Multi-Teacher Configuration

For premium quality samples, use consensus from multiple teachers:

```python
from agents.training.generators.error_generator import MultiTeacherGenerator

# Uses Gemini 3, Opus 4.5, GPT-5.2 for consensus
multi_gen = MultiTeacherGenerator(
    base_generator=asm_generator,
    consensus_threshold=2,  # 2 of 3 must agree
    validate_locally=True,  # Validate with Ollama
)
```

## 10. Configuration (`hafs.toml`)

Current training config in `hafs.toml`:

```toml
[training]
enabled = true
output_dir = "~/.context/training"
checkpoint_interval = 100
teacher_provider = "gemini"
teacher_tier = "coding"
min_quality_score = 0.7
diversity_threshold = 0.95

# Future additions:
# feedback_enabled = true
# multi_teacher = ["gemini", "claude", "qwen"]
# active_learning = true
# augmentation_ratio = 0.3
```

---

## 10. References

- Training Pipeline Plan: `~/.claude/plans/vectorized-sniffing-giraffe.md`
- Quality Pipeline: `src/agents/training/quality.py`
- Training Daemon: `src/hafs/services/training_daemon.py`
- Autonomy Agents: `src/agents/autonomy/`
- Mission Agents: `src/agents/mission/mission_agents.py`
