# Sample Quality Improvement Guide

When model performance is suboptimal, the root cause is often training data quality. This guide covers strategies to improve sample generation from varied-quality sources.

---

## Problem: Varied Source Quality

Different codebases have different characteristics:

| Source | Quality | Challenge |
|--------|---------|-----------|
| **zelda3 disassembly** | High | Well-documented but vanilla-only, needs Oracle context |
| **Oracle-of-Secrets** | Medium | ROM hack code, inconsistent comments, mixed quality |
| **alttp-gigaleak** | Very High | Nintendo source with Japanese comments, needs translation context |
| **YAZE C++ code** | Medium | Tool code, not game logic - needs relevance filtering |
| **book-of-mudora** | High | Documentation, but needs code examples to be actionable |
| **hyrule-historian** | Medium | Community guides, varying technical depth |

**Core Issue**: The model can't distinguish between:
- Relevant context (Oracle secrets system) vs boilerplate (YAZE UI code)
- High-value samples (detailed explanations) vs low-value (auto-generated comments)
- Accurate information vs outdated community guesses

---

## Phase 1: Source Quality Assessment

### 1.1 Analyze Current Rejection Patterns

First, understand **why** samples are being rejected:

```bash
# Extract rejection reasons from all campaigns
for dir in ~/.context/training/datasets/*/; do
    if [ -f "$dir/rejected.jsonl" ]; then
        echo "=== $(basename $dir) ==="
        jq -r '.rejection_reason' "$dir/rejected.jsonl" | sort | uniq -c | sort -rn
        echo ""
    fi
done
```

**Expected output:**
```
=== oracle_farore_improved_20251222_040629 ===
    144 diversity_too_low (similarity > 0.70)
     12 quality_too_low (< 0.4)
      3 coherence_poor
      2 llm_low_confidence

=== alttp_yaze_full_1000_20251221_195746 ===
     89 diversity_too_low
     23 quality_too_low
      8 irrelevant_content
```

### 1.2 Sample Quality Distribution by Source

Analyze which sources produce the best samples:

```python
# scripts/analyze_sample_sources.py
import json
from pathlib import Path
from collections import defaultdict

def analyze_sources(dataset_path):
    """Analyze sample quality by source."""

    accepted = Path(dataset_path) / "accepted.jsonl"
    rejected = Path(dataset_path) / "rejected.jsonl"

    source_stats = defaultdict(lambda: {
        'accepted': 0,
        'rejected': 0,
        'avg_quality': [],
        'avg_diversity': []
    })

    # Process accepted samples
    with open(accepted) as f:
        for line in f:
            sample = json.loads(line)
            source = sample.get('metadata', {}).get('source_file', 'unknown')
            source_stats[source]['accepted'] += 1
            source_stats[source]['avg_quality'].append(sample.get('quality_score', 0))
            source_stats[source]['avg_diversity'].append(sample.get('diversity_score', 0))

    # Process rejected samples
    with open(rejected) as f:
        for line in f:
            sample = json.loads(line)
            source = sample.get('metadata', {}).get('source_file', 'unknown')
            source_stats[source]['rejected'] += 1

    # Calculate acceptance rates
    for source, stats in source_stats.items():
        total = stats['accepted'] + stats['rejected']
        acceptance_rate = stats['accepted'] / total if total > 0 else 0
        avg_quality = sum(stats['avg_quality']) / len(stats['avg_quality']) if stats['avg_quality'] else 0
        avg_diversity = sum(stats['avg_diversity']) / len(stats['avg_diversity']) if stats['avg_diversity'] else 0

        print(f"\n{source}")
        print(f"  Acceptance: {acceptance_rate:.1%} ({stats['accepted']}/{total})")
        print(f"  Avg Quality: {avg_quality:.3f}")
        print(f"  Avg Diversity: {avg_diversity:.3f}")
```

**Run:**
```bash
python scripts/analyze_sample_sources.py \
  ~/.context/training/datasets/oracle_farore_improved_20251222_040629
```

**Expected insights:**
```
~/Code/zelda3/bank_00.asm
  Acceptance: 82.5% (33/40)
  Avg Quality: 0.71
  Avg Diversity: 0.58
  → High quality, well-documented vanilla code

~/Code/Oracle-of-Secrets/src/asm/secrets_system.asm
  Acceptance: 45.2% (19/42)
  Avg Quality: 0.52
  Avg Diversity: 0.68
  → Medium quality, inconsistent comments, but diverse

~/Code/hafs/src/cc/yaze/zelda3_editor.cc
  Acceptance: 23.1% (6/26)
  Avg Quality: 0.38
  Avg Diversity: 0.42
  → Low quality, tool code not game logic, low diversity
```

---

## Phase 2: Source-Specific Improvement Strategies

### 2.1 High-Quality Sources (zelda3, gigaleak, book-of-mudora)

**Strategy**: Extract more samples with richer context

**Before (vanilla zelda3):**
```
Instruction: What does this code do?
Input: LDA $0AC0 BEQ .return STA $7E0FC6 .return: RTS
Output: Loads value from $0AC0, if zero returns, otherwise stores to RAM
```

**Problem**: Missing Oracle context, too literal

**After (with Oracle enrichment):**
```
Instruction: Explain this vanilla ALTTP code and how Oracle of Secrets modifies it
Input:
; Vanilla ALTTP: Simple item check
LDA $0AC0      ; Check if player has item
BEQ .return
STA $7E0FC6    ; Store to RAM
.return: RTS

; Oracle modification: Add secret requirements
Context: Oracle of Secrets adds a secrets system that requires multiple conditions

Output: In vanilla ALTTP, this code simply checks if the player has an item ($0AC0)
and stores it to RAM. Oracle of Secrets extends this by adding a secrets verification
layer - instead of just checking the item, it also verifies if the player has unlocked
the corresponding secret. This requires hooking the code to call a secrets_check
subroutine before the STA instruction.

To implement in Oracle:
1. Hook at $[original address]
2. JSR secrets_check  ; New subroutine
3. BCC .return         ; Failed secret check
4. Original code continues...
```

**Implementation:**

```python
# src/agents/training/generators/zelda3_enhanced_generator.py

class Zelda3EnhancedGenerator(DataGenerator):
    """Extract from zelda3 with Oracle context enrichment."""

    async def generate_samples(self, count: int) -> list[TrainingSample]:
        """Generate samples with Oracle comparisons."""

        samples = []
        zelda3_routines = self._extract_routines("~/Code/zelda3/")
        oracle_routines = self._extract_routines("~/Code/Oracle-of-Secrets/")

        for z3_routine in zelda3_routines[:count]:
            # Find similar Oracle routine (if exists)
            oracle_match = self._find_similar_routine(z3_routine, oracle_routines)

            # Generate comparison sample
            if oracle_match:
                prompt = self._build_comparison_prompt(z3_routine, oracle_match)
            else:
                prompt = self._build_oracle_extension_prompt(z3_routine)

            sample = await self._generate_with_teacher(prompt)
            samples.append(sample)

        return samples

    def _build_comparison_prompt(self, vanilla, oracle):
        """Build prompt comparing vanilla vs Oracle implementation."""
        return f"""Explain this ALTTP code and how Oracle of Secrets modifies it.

Vanilla ALTTP code:
```asm
{vanilla['code']}
```

Oracle of Secrets modification:
```asm
{oracle['code']}
```

Context:
- Vanilla address: {vanilla['address']}
- Oracle changes: {oracle.get('comments', 'See code')}
- Purpose: {vanilla['purpose']}

Explain:
1. What the vanilla code does
2. Why Oracle modifies it
3. How to implement similar modifications
4. Technical details (memory layout, hooks, etc.)
"""

    def _build_oracle_extension_prompt(self, vanilla):
        """Build prompt for extending vanilla code with Oracle features."""
        return f"""This is vanilla ALTTP code. Explain how to extend it for Oracle of Secrets.

Vanilla code:
```asm
{vanilla['code']}
```

Context:
- Address: {vanilla['address']}
- Purpose: {vanilla['purpose']}
- Oracle feature to add: {self._suggest_oracle_feature(vanilla)}

Explain:
1. What the vanilla code does
2. What Oracle feature would enhance it
3. How to implement the modification (detailed ASM)
4. Potential challenges and solutions
"""
```

### 2.2 Medium-Quality Sources (Oracle-of-Secrets, hyrule-historian)

**Strategy**: Add validation and fact-checking

**Before:**
```
Instruction: How does the secrets system work?
Input: [Extract from Oracle README]
Output: [Direct copy of README - may be outdated]
```

**Problem**: Community docs can be outdated, incomplete, or wrong

**After:**
```
Instruction: Explain the secrets system implementation
Input:
Documentation claim: "Secrets are stored at $7E1000-$7E1100"
Code evidence:
  secrets_base = $7EF000  ; Actually at $7EF000
  secrets_count = 256     ; 256 bytes total

Output: The secrets system uses a 256-byte array at $7EF000 (not $7E1000 as some
older documentation states - that was changed in v0.3). Each byte represents a
secret flag, with bit 0 = discovered, bit 1 = completed, bits 2-7 reserved for
future use...
```

**Implementation:**

```python
# src/agents/training/generators/oracle_validated_generator.py

class OracleValidatedGenerator(DataGenerator):
    """Generate Oracle samples with code validation."""

    async def generate_samples(self, count: int) -> list[TrainingSample]:
        samples = []

        # Load documentation
        docs = self._load_documentation("~/Code/book-of-mudora/")

        # Load actual code
        code = self._load_code("~/Code/Oracle-of-Secrets/")

        for doc_item in docs[:count]:
            # Validate doc claims against code
            validation = self._validate_against_code(doc_item, code)

            if validation['conflicts']:
                # Documentation disagrees with code - create corrective sample
                sample = await self._generate_correction_sample(
                    doc_claim=doc_item,
                    actual_code=validation['evidence'],
                    conflicts=validation['conflicts']
                )
            else:
                # Documentation matches code - create explanatory sample
                sample = await self._generate_validated_sample(
                    doc_item=doc_item,
                    code_evidence=validation['evidence']
                )

            samples.append(sample)

        return samples

    def _validate_against_code(self, doc_item, code):
        """Check if documentation matches actual code."""

        conflicts = []
        evidence = []

        # Extract claimed addresses from docs
        doc_addresses = self._extract_addresses(doc_item['text'])

        # Search for those addresses in code
        for addr in doc_addresses:
            code_usage = self._find_address_usage(addr, code)

            if not code_usage:
                conflicts.append(f"Address {addr} not found in code")
            elif code_usage['usage'] != doc_item.get('claimed_usage'):
                conflicts.append(f"Address {addr} used differently than documented")
                evidence.append(code_usage)

        return {'conflicts': conflicts, 'evidence': evidence}
```

### 2.3 Low-Quality Sources (YAZE UI code, auto-generated comments)

**Strategy**: Filter aggressively + add relevance scoring

**Before:**
```
Instruction: Explain this code
Input:
// QT UI boilerplate
void ZeldaEditor::on_pushButton_clicked() {
    QMessageBox::information(this, "Info", "Button clicked");
}

Output: This function shows a message box when the button is clicked...
```

**Problem**: Completely irrelevant to Oracle ROM hacking

**After (filtered out):**
```python
# src/agents/training/generators/yaze_filtered_generator.py

class YAZEFilteredGenerator(DataGenerator):
    """Extract only relevant YAZE code (game logic, not UI)."""

    IRRELEVANT_PATTERNS = [
        r'void.*on_.*_clicked',           # QT signals
        r'#include <Q.*>',                 # QT headers
        r'ui->.*->setText',                # UI updates
        r'QMessageBox',                    # Dialogs
        r'class.*Widget',                  # Widget classes
    ]

    RELEVANT_PATTERNS = [
        r'class.*Tilemap',                 # Tilemap logic
        r'void.*decompress.*\(',           # Compression
        r'struct.*Header',                 # ROM structures
        r'uint8_t.*Read.*\(',              # ROM reading
        r'class.*ROM',                     # ROM handling
    ]

    def _is_relevant_code(self, code_block):
        """Check if code is relevant to ROM hacking."""

        # Reject if matches irrelevant patterns
        for pattern in self.IRRELEVANT_PATTERNS:
            if re.search(pattern, code_block):
                return False

        # Accept if matches relevant patterns
        for pattern in self.RELEVANT_PATTERNS:
            if re.search(pattern, code_block):
                return True

        # Use LLM for borderline cases
        return self._llm_relevance_check(code_block)

    async def _llm_relevance_check(self, code_block):
        """Ask LLM if code is relevant to Oracle ROM hacking."""

        prompt = f"""Is this code relevant to SNES ROM hacking and Oracle of Secrets development?

Code:
```cpp
{code_block}
```

Relevant topics: ROM structure, compression, graphics, game logic, memory layout
Irrelevant topics: UI code, file dialogs, menu handling, widget setup

Answer: YES or NO, then briefly explain why.
"""

        response = await self.teacher_model.query(prompt)
        return response.startswith("YES")
```

---

## Phase 3: Prompt Engineering Improvements

### 3.1 Template Diversity (Already Planned)

**Problem**: All samples use same instruction format → similar embeddings

**Solution**: Rotate through 15-20 templates per domain

```toml
# config/prompt_templates_enhanced.toml

[templates.asm.perspectives]
expert = "As an experienced 65816 assembly programmer, explain {topic}"
beginner = "I'm new to SNES assembly. Can you explain {topic} in simple terms?"
reference = "Provide a technical reference for {topic}"
tutorial = "Write a step-by-step tutorial for {topic}"
comparison = "Compare vanilla ALTTP's {topic} with Oracle's implementation"
debugging = "I'm debugging {topic}. What should I check?"
optimization = "How can I optimize this code for {topic}?"
porting = "How would I port this {topic} from ALTTP to Oracle?"

[templates.asm.tones]
formal = "Explain the technical details of {topic}"
conversational = "Hey, quick question about {topic} - how does it work?"
terse = "Briefly: what is {topic}?"
verbose = "Provide a comprehensive explanation of {topic} including history, implementation, and best practices"
socratic = "What are the key concepts needed to understand {topic}?"
problem_solving = "I need to implement {topic}. Walk me through the approach."

[templates.asm.contexts]
dungeon = "In the context of dungeon development, explain {topic}"
overworld = "For overworld modifications, explain {topic}"
menu = "Regarding menu systems, explain {topic}"
hardware = "From a hardware/timing perspective, explain {topic}"
secrets = "In relation to the secrets system, explain {topic}"
items = "For item handling, explain {topic}"
```

### 3.2 Context Enrichment

**Add more context to every sample:**

```python
# src/agents/training/context_enricher.py

class ContextEnricher:
    """Add relevant context to samples."""

    def enrich_sample(self, sample: dict) -> dict:
        """Add context from multiple sources."""

        enriched = sample.copy()

        # 1. Add cross-references
        enriched['cross_references'] = self._find_related_code(sample)

        # 2. Add historical context
        enriched['version_info'] = self._get_version_history(sample)

        # 3. Add usage examples
        enriched['examples'] = self._find_usage_examples(sample)

        # 4. Add warnings/gotchas
        enriched['warnings'] = self._extract_warnings(sample)

        return enriched

    def _find_related_code(self, sample):
        """Find related code in other files."""

        # Extract symbols from sample
        symbols = self._extract_symbols(sample['code'])

        # Search for those symbols in other files
        related = []
        for symbol in symbols:
            usages = self._search_symbol(symbol)
            related.extend(usages)

        return related

    def _get_version_history(self, sample):
        """Get git history for this code."""

        file_path = sample['metadata']['source_file']
        line_num = sample['metadata'].get('line_number')

        # Get git blame
        history = subprocess.run(
            ['git', 'log', '-L', f'{line_num},{line_num+10}:{file_path}'],
            capture_output=True,
            text=True
        ).stdout

        return self._parse_git_history(history)

    def _find_usage_examples(self, sample):
        """Find real examples of this code being used."""

        # If this is a function definition, find calls
        if self._is_function_def(sample):
            func_name = self._extract_function_name(sample)
            return self._find_function_calls(func_name)

        # If this is a data structure, find usage
        if self._is_struct_def(sample):
            struct_name = self._extract_struct_name(sample)
            return self._find_struct_usage(struct_name)

        return []
```

### 3.3 Multi-Stage Generation

**Instead of single-shot generation, use multi-stage refinement:**

```python
# src/agents/training/multi_stage_generator.py

class MultiStageGenerator(DataGenerator):
    """Generate samples in multiple refinement stages."""

    async def generate_sample(self, topic) -> TrainingSample:
        """Multi-stage generation process."""

        # Stage 1: Draft generation
        draft_prompt = self._build_draft_prompt(topic)
        draft = await self.teacher_model.query(draft_prompt)

        # Stage 2: Fact checking
        facts_prompt = self._build_fact_check_prompt(draft, topic)
        fact_check = await self.teacher_model.query(facts_prompt)

        if fact_check['has_errors']:
            # Stage 3: Correction
            correction_prompt = self._build_correction_prompt(
                draft,
                fact_check['errors']
            )
            corrected = await self.teacher_model.query(correction_prompt)
        else:
            corrected = draft

        # Stage 4: Enrichment
        enrichment_prompt = self._build_enrichment_prompt(corrected, topic)
        enriched = await self.teacher_model.query(enrichment_prompt)

        # Stage 5: Diversity check
        if self._is_too_similar_to_existing(enriched):
            # Stage 6: Perspective shift
            perspective_prompt = self._build_perspective_shift_prompt(
                enriched,
                perspective=random.choice(['beginner', 'expert', 'tutorial'])
            )
            final = await self.teacher_model.query(perspective_prompt)
        else:
            final = enriched

        return self._build_training_sample(final, topic)
```

---

## Phase 4: Quality Scoring Improvements

### 4.1 Multi-Dimensional Quality

Current quality score is single-dimensional. Make it multi-dimensional:

```python
# src/agents/training/quality_multidim.py

@dataclass
class QualityMetrics:
    """Multi-dimensional quality assessment."""

    # Content quality
    technical_accuracy: float      # Is the code/info correct?
    completeness: float            # Are all details covered?
    clarity: float                 # Is it easy to understand?

    # Relevance
    domain_relevance: float        # Relevant to Oracle hacking?
    actionability: float           # Can someone use this info?
    uniqueness: float              # Novel information?

    # Structure
    coherence: float               # Logical flow?
    code_quality: float            # If contains code, is it good?
    examples_quality: float        # Good examples?

    # Meta
    llm_confidence: float          # Teacher model confidence
    source_authority: float        # Is source authoritative?

    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        return (
            0.25 * self.technical_accuracy +
            0.20 * self.domain_relevance +
            0.15 * self.completeness +
            0.15 * self.actionability +
            0.10 * self.coherence +
            0.10 * self.uniqueness +
            0.05 * self.source_authority
        )


class MultiDimensionalQualityScorer:
    """Score samples across multiple dimensions."""

    async def score_sample(self, sample: dict) -> QualityMetrics:
        """Score sample on all dimensions."""

        return QualityMetrics(
            technical_accuracy=await self._score_accuracy(sample),
            completeness=await self._score_completeness(sample),
            clarity=self._score_clarity(sample),
            domain_relevance=self._score_domain_relevance(sample),
            actionability=self._score_actionability(sample),
            uniqueness=self._score_uniqueness(sample),
            coherence=self._score_coherence(sample),
            code_quality=self._score_code_quality(sample),
            examples_quality=self._score_examples(sample),
            llm_confidence=sample.get('llm_confidence', 0.5),
            source_authority=self._get_source_authority(sample),
        )

    async def _score_accuracy(self, sample):
        """Score technical accuracy using validation."""

        # Extract claims from sample
        claims = self._extract_technical_claims(sample)

        # Validate each claim
        validations = []
        for claim in claims:
            is_valid = await self._validate_claim(claim)
            validations.append(is_valid)

        # Return percentage of valid claims
        return sum(validations) / len(validations) if validations else 0.5

    def _score_domain_relevance(self, sample):
        """Score relevance to Oracle ROM hacking."""

        oracle_keywords = [
            'secrets', 'oracle', 'rom hack', 'farore', 'nayru', 'din',
            'ring menu', 'yaze', 'dungeon', 'vanilla vs oracle'
        ]

        irrelevant_keywords = [
            'qt', 'widget', 'gui', 'button', 'dialog', 'menu bar',
            'file open', 'save as', 'preferences'
        ]

        text = sample['output'].lower()

        oracle_score = sum(kw in text for kw in oracle_keywords) / len(oracle_keywords)
        irrelevant_score = sum(kw in text for kw in irrelevant_keywords) / len(irrelevant_keywords)

        return max(0, oracle_score - irrelevant_score)

    def _score_actionability(self, sample):
        """Score whether sample provides actionable information."""

        # Check for concrete elements
        has_code = '```' in sample['output']
        has_steps = any(marker in sample['output'] for marker in ['1.', '2.', 'Step', 'First'])
        has_examples = 'example' in sample['output'].lower() or 'e.g.' in sample['output']
        has_addresses = re.search(r'\$[0-9A-Fa-f]{4,6}', sample['output'])
        has_instructions = re.search(r'(LDA|STA|JSR|RTS|BEQ)', sample['output'])

        actionable_elements = [
            has_code,
            has_steps,
            has_examples,
            has_addresses,
            has_instructions
        ]

        return sum(actionable_elements) / len(actionable_elements)
```

### 4.2 Source Authority Ranking

Rank sources by authority:

```python
# src/agents/training/source_authority.py

SOURCE_AUTHORITY = {
    # Tier 1: Official/High Authority (0.9-1.0)
    'alttp-gigaleak': 1.0,          # Nintendo source code
    'zelda3/': 0.95,                 # Community-verified disassembly

    # Tier 2: Reliable Community (0.7-0.89)
    'Oracle-of-Secrets/src/': 0.85,  # Main Oracle codebase
    'book-of-mudora/': 0.80,         # Curated documentation
    'asar/docs/': 0.75,              # Assembler docs

    # Tier 3: Medium Reliability (0.5-0.69)
    'hyrule-historian/': 0.65,       # Community guides
    'alttp-hacker-workspace/': 0.60, # User experiments

    # Tier 4: Low Reliability (0.3-0.49)
    'yaze/ui/': 0.40,                # Tool UI code
    'experimental/': 0.35,           # Experimental code

    # Tier 5: Unknown (0.5 default)
    'unknown': 0.50,
}

def get_source_authority(file_path: str) -> float:
    """Get authority score for a source file."""

    for source_pattern, authority in SOURCE_AUTHORITY.items():
        if source_pattern in file_path:
            return authority

    return 0.50  # Default for unknown sources
```

---

## Phase 5: Automated Quality Analysis

### 5.1 Sample Quality Dashboard

Create a dashboard to visualize quality metrics:

```python
# scripts/quality_dashboard.py

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def create_quality_dashboard(dataset_path):
    """Create visualization of sample quality metrics."""

    accepted = Path(dataset_path) / "accepted.jsonl"
    rejected = Path(dataset_path) / "rejected.jsonl"

    # Load samples
    accepted_samples = [json.loads(line) for line in open(accepted)]
    rejected_samples = [json.loads(line) for line in open(rejected)]

    # Create multi-panel figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: Quality score distribution
    plot_quality_distribution(accepted_samples, rejected_samples, axes[0, 0])

    # Panel 2: Diversity score distribution
    plot_diversity_distribution(accepted_samples, rejected_samples, axes[0, 1])

    # Panel 3: Rejection reasons pie chart
    plot_rejection_reasons(rejected_samples, axes[0, 2])

    # Panel 4: Quality by source
    plot_quality_by_source(accepted_samples, rejected_samples, axes[1, 0])

    # Panel 5: Quality vs diversity scatter
    plot_quality_diversity_scatter(accepted_samples, axes[1, 1])

    # Panel 6: Domain distribution
    plot_domain_distribution(accepted_samples, axes[1, 2])

    plt.tight_layout()
    plt.savefig(Path(dataset_path) / "quality_dashboard.png", dpi=150)
    print(f"Dashboard saved to {dataset_path}/quality_dashboard.png")
```

### 5.2 Automated Quality Report

Generate detailed quality report:

```bash
# scripts/generate_quality_report.sh

DATASET_PATH="$1"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║              Dataset Quality Report                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Overall stats
python scripts/quality_stats.py "$DATASET_PATH" --section overall

# Source quality breakdown
python scripts/quality_stats.py "$DATASET_PATH" --section sources

# Rejection analysis
python scripts/quality_stats.py "$DATASET_PATH" --section rejections

# Recommendations
python scripts/quality_stats.py "$DATASET_PATH" --section recommendations

# Generate visualizations
python scripts/quality_dashboard.py "$DATASET_PATH"
```

---

## Implementation Priority

### Week 1: Quick Wins
1. ✅ Source authority ranking (1 day)
2. ✅ YAZE irrelevance filtering (1 day)
3. ✅ Quality analysis scripts (2 days)
4. ✅ Multi-dimensional quality scoring (2 days)

### Week 2: Context Enrichment
1. ✅ Zelda3 Oracle comparison generator (2 days)
2. ✅ Oracle validation generator (2 days)
3. ✅ Context enricher (cross-refs, examples) (2 days)

### Week 3: Advanced Techniques
1. ✅ Multi-stage generation (3 days)
2. ✅ Prompt template rotation (already planned)
3. ✅ Quality dashboard (2 days)

---

## Success Metrics

**Before Improvements:**
- Acceptance rate: 77.8% (504/648)
- Diversity rejection: 85% of rejections
- Average quality: 0.65
- Source variation: Minimal

**Target After Improvements:**
- Acceptance rate: >85% (fewer low-quality samples generated)
- Diversity rejection: <50% of rejections
- Average quality: >0.75
- High-value sources: >60% of samples

---

## Next Steps

1. **Immediate**: Run quality analysis on oracle_farore dataset
   ```bash
   python scripts/analyze_sample_sources.py ~/.context/training/datasets/oracle_farore_improved_*
   ```

2. **Short-term**: Implement source authority + YAZE filtering
3. **Medium-term**: Build enhanced generators for high-quality sources
4. **Long-term**: Multi-stage generation for all domains

The key insight: **Quality improvements happen at generation time, not filtering time**. Better to generate 800 high-quality samples than 2000 mediocre ones and filter down.
