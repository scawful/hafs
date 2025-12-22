# Next Steps and Future Improvements

**Created:** 2025-12-21
**Status:** Planning
**Priority:** High-impact improvements for hafs

## Current Status

### ✅ Completed Recently
- A/B testing infrastructure for prompt evaluation
- 50-sample A/B test (baseline prompts won: 0.5182 vs 0.5083)
- Filesystem exploration agents for Windows drives
- OllamaClient with async/sync support and tool calling
- ToolExecutor with 7 safe tools for LLM function calling
- Comprehensive local AI orchestration design

### 🔄 In Progress
- Filesystem scan: 31,000+ directories, 161,786+ files (C:/D:/E: drives)
- Hybrid training campaign using baseline prompts (GPU + Gemini)

## 1. Finish Local AI Orchestration System ⭐ HIGH PRIORITY

**Time:** 2-3 hours
**Status:** 70% complete (design + foundation done)
**Impact:** Enable intelligent local AI across all hafs agents

### What's Done
- ✅ Full architecture design (`docs/plans/LOCAL_AI_ORCHESTRATION.md`)
- ✅ OllamaClient (`src/hafs/services/ollama_client.py`)
- ✅ ToolExecutor (`src/hafs/services/tool_executor.py`)

### What's Needed
```python
# src/hafs/services/local_ai_orchestrator.py
class LocalAIOrchestrator:
    """Priority-based inference orchestration"""
    - Priority queue (Training > Interactive > Analysis > Scheduled)
    - Context window management (auto-retrieval from hafs context)
    - Tool execution integration
    - Resource monitoring (don't interfere with training)
    - Async request handling
```

### Integration Points
1. **ConsolidationAnalyzerAgent**: AI-powered filesystem recommendations
2. **Training Quality Pipeline**: Local quality scoring
3. **Chat Mode**: Local AI responses when API unavailable
4. **Background Agents**: Scheduled analysis tasks

### Implementation Steps
1. Create `LocalAIOrchestrator` class (1 hour)
2. Integrate with `consolidation_analyzer.py` (30 min)
3. Add configuration to `hafs.toml` (15 min)
4. Test with filesystem scan results (30 min)
5. Deploy to Windows for live testing (15 min)

### Success Metrics
- ✅ Requests queued by priority
- ✅ Training never blocked by analysis
- ✅ Tools execute successfully
- ✅ Context retrieved from hafs
- ✅ Recommendations reference actual files
- ✅ Zero GPU interference

---

## 2. Training Data Quality Improvements

**Time:** 3-4 hours
**Impact:** Increase high-quality sample rate (>0.6 quality)

### 2.1 Investigate the 0.76 Outlier

The baseline A/B test produced one 0.76 quality sample (hardware reset routine). Why was it exceptional?

**Tasks:**
- Load the 0.76 sample and analyze characteristics
- Compare to other hardware-related routines
- Identify patterns: complexity, structure, naming conventions
- Extract "golden template" features
- Test if we can reliably generate similar quality

**Hypothesis:** Hardware routines have clearer intent than gameplay logic, making them better training targets.

### 2.2 Quality Component Analysis

Current quality score is composite: `0.3*diversity + 0.25*kg + 0.25*halluc + 0.2*coherence`

**Questions:**
- Which components correlate most with usefulness?
- Are we over-weighting diversity?
- Should we add new components (novelty, technical accuracy)?

**Tasks:**
- Analyze 50 A/B samples by component scores
- Plot component distributions
- Test alternative weighting schemes
- Implement configurable weights per domain

### 2.3 Source Filtering

Not all ALTTP source files are equal for training.

**Tasks:**
- Analyze quality by source bank (`$00-$3F`)
- Identify high-quality modules (AI, physics, sprite logic)
- Filter out low-quality sources (simple loops, data tables)
- Create source quality rankings
- Bias sampling toward high-quality sources

### 2.4 Rejection Pattern Analysis

**Tasks:**
- Load rejection logs from quality pipeline
- Categorize rejection reasons
- Find patterns in rejected samples
- Update prompts to avoid common rejection causes
- Add pre-generation filters

---

## 3. ML Visualization App Enhancements

**Time:** 2-3 hours
**Status:** C++/ImGui viz app exists (`src/cc/viz/`)
**Impact:** Real-time visibility into training and quality

### 3.1 Real-Time Training Metrics

**Add to ImGui dashboard:**
- Live sample generation rate (samples/min)
- Quality distribution histogram (0.0-1.0)
- Pass/fail ratio over time
- Cost tracking (API calls vs GPU)
- Domain breakdown (ASM, Oracle, YAZE, etc.)

**Implementation:**
- WebSocket connection to training daemon
- Circular buffer for time-series data
- ImPlot for charts and graphs

### 3.2 GPU vs API Routing Visualization

**Show hybrid orchestrator decisions:**
- GPU utilization % over time
- Routing decisions (GPU/API) with rationale
- Cost savings vs API-only
- Request queue depth

### 3.3 Filesystem Scan Progress

**Live dashboard for filesystem exploration:**
- Directories scanned
- Files cataloged
- Duplicates found (with savings)
- Largest files discovered
- Scan speed (files/sec)

### 3.4 Quality Inspector

**Interactive sample viewer:**
- Browse generated samples
- See quality breakdown per component
- Compare baseline vs enhanced samples
- Flag samples for manual review
- Export high-quality samples for fine-tuning

---

## 4. Knowledge Graph Expansion

**Time:** 4-5 hours
**Impact:** Enable semantic search and intelligent retrieval

### 4.1 Filesystem Scan Embeddings

Once scan completes, generate embeddings for:
- Directory structures
- File organization patterns
- Code repositories found
- Duplicate file groups

**Use cases:**
- "Find similar projects to ALTTP"
- "Where are all my machine learning datasets?"
- "Show me abandoned projects from 2023"

### 4.2 ALTTP → Gigaleak Linking

**Build knowledge graph connections:**
- ALTTP routine → Gigaleak equivalent
- Shared variable names
- Common patterns across games
- Evolution of algorithms (ALTTP → SMW → Yoshi's Island)

### 4.3 Cross-Codebase Semantic Search

**Enable queries like:**
- "How does sprite collision work?"
  - Returns: ALTTP code, Gigaleak equivalents, YAZE emulator implementation
- "Find all DMA transfer routines"
  - Searches across all indexed codebases

### 4.4 Code Pattern Extraction

**Automatic pattern discovery:**
- Common assembly idioms (e.g., "16-bit add with carry")
- Optimization patterns
- Anti-patterns and bugs
- Hardware interaction sequences

---

## 5. Autonomous Training Improvements

**Time:** 3-4 hours
**Impact:** More robust, higher-quality training

### 5.1 Checkpoint Validation

**Problem:** Bad checkpoints can corrupt training runs

**Solution:**
- Validate checkpoint before saving
- Test sample generation from checkpoint
- Compare quality to previous checkpoint
- Auto-rollback if quality degrades >10%

### 5.2 Curriculum Learning

**Start easy, increase difficulty:**

**Phase 1 (0-5k samples):** Simple routines
- Single-purpose functions
- Clear variable names
- Minimal branching

**Phase 2 (5k-15k samples):** Moderate complexity
- Multi-step algorithms
- State machines
- Hardware interaction

**Phase 3 (15k-35k samples):** Advanced
- Complex AI routines
- Interrupt handlers
- Optimization-heavy code

### 5.3 Multi-Objective Optimization

**Current:** Single quality score
**Proposed:** Optimize multiple objectives

**Objectives:**
1. Quality (existing metric)
2. Diversity (from other samples)
3. Novelty (unique patterns)
4. Coverage (hit all source areas)

**Implementation:** Pareto frontier optimization

### 5.4 Safety Rails

**Detect and reject:**
- Hallucinated instruction names
- Impossible register combinations
- Nonsensical explanations
- Circular reasoning

**Tools:**
- ASM syntax validator
- 65816 instruction set verifier
- Semantic consistency checker

---

## 6. Cross-System Sync

**Time:** 2 hours
**Impact:** Seamless multi-machine workflow

### 6.1 Automated Sync Scripts

**Mac → Windows:**
```bash
rsync -avz ~/Code/hafs/ medical-mechanica:C:/hafs/ \
  --exclude .venv --exclude .git
```

**Windows → Mac:**
```bash
rsync -avz medical-mechanica:D:/.context/training/datasets/ \
  ~/.context/training/datasets/
```

### 6.2 Git Hooks

**Pre-commit hook:**
- Sync configs to Windows
- Validate TOML syntax
- Check for API key leaks

**Post-training hook:**
- Pull latest datasets
- Update embeddings
- Sync reports

### 6.3 Tailscale Integration

**Automatic mounts:**
- D:/.context → ~/Mounts/mm-d/.context
- Halext-server:/data → ~/Mounts/halext/data

### 6.4 Backup Strategy

**Daily backups:**
- Knowledge graphs → halext-server
- Training datasets → Mac + cloud
- Configs → Git + halext-server

---

## 7. Wait for Scan & Analyze

**Time:** 10-15 min wait + 30 min analysis
**Status:** Scan in progress (31,000+ dirs)

### 7.1 When Scan Completes

**Automatic tasks:**
1. Run consolidation_analyzer
2. Generate AI recommendations (using Ollama)
3. Create actionable cleanup plan
4. Identify largest savings opportunities

### 7.2 Analysis Targets

**Duplicates:**
- Group by file type
- Prioritize large duplicates (>100 MB)
- Suggest keep/delete for each group

**Organization:**
- Scattered projects (same type across drives)
- Temp files and caches
- Old downloads and archives
- Abandoned experiments

### 7.3 AI Recommendations

**Using local Ollama:**
- Analyze file patterns
- Suggest directory structures
- Identify archival candidates
- Propose automation scripts

### 7.4 Cleanup Execution

**Semi-automated cleanup:**
```bash
# Review recommendations
cat ~/Mounts/mm-d/.context/scratchpad/consolidation_analyzer/ai_recommendations.md

# Execute safe cleanups (user confirmation required)
./scripts/cleanup_duplicates.sh --dry-run
./scripts/organize_by_category.sh --preview
```

---

## Priority Ranking

### Immediate (This Week)
1. ⭐ **Local AI Orchestration** - Foundation for all AI features
2. **Filesystem Analysis** - Once scan completes
3. **Visualization Dashboard** - Real-time training visibility

### Short-Term (Next 2 Weeks)
4. **Training Quality** - Increase >0.6 sample rate
5. **Knowledge Graph** - Semantic search and linking
6. **Cross-System Sync** - Automation and reliability

### Medium-Term (Next Month)
7. **Autonomous Training** - Curriculum learning, safety rails
8. **Advanced Prompting** - Learn from 0.76 outlier

---

## Resource Requirements

### Compute
- **Mac**: Development, orchestration
- **Windows GPU**: Training, local AI inference
- **Halext-server**: Backups, long-term storage

### Storage
- **Current**: ~3.4 TB (D: + E: drives)
- **After cleanup**: Est. 2.8-3.0 TB (10-15% savings)
- **Training datasets**: ~50 GB and growing

### API Costs
- **Current**: Hybrid GPU + Gemini (~50% savings)
- **With local AI**: 70-80% cost reduction
- **Target**: <$10/month for training

---

## Success Metrics

### Local AI Orchestration
- [ ] 90% of consolidation queries answered locally
- [ ] Zero training interference
- [ ] <2 sec response time for file queries

### Training Quality
- [ ] 5% of samples >0.6 quality (vs current ~2%)
- [ ] Pass rate >95% (currently 98-100%)
- [ ] Reduce API costs by 20%

### Filesystem Organization
- [ ] 10-15 GB freed from duplicates
- [ ] 80% of code projects in organized structure
- [ ] <5 min to find any file semantically

### Development Velocity
- [ ] Cross-system sync <1 minute
- [ ] Real-time training metrics visible
- [ ] Knowledge graph answers 80% of questions

---

## Notes

- All improvements designed to work together
- Focus on automation and intelligence
- Minimize manual intervention
- Maximize use of local resources (GPU, Ollama)
- Reduce cloud API costs
