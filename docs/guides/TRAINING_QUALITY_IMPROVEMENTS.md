# Training Pipeline Quality Improvements

**Date:** 2025-12-22
**Status:** Proposal - Complements Multi-Model Infrastructure

---

## Executive Summary

Based on comprehensive audit of the training pipeline, here are prioritized improvements to increase dataset quality and leverage the new multi-model infrastructure.

---

## Critical Issues (High Priority)

### 1. Multi-Model Cross-Validation for Code

**Problem:** Hallucination detection is bypassed for code domains ("too slow").

**Solution:** Use multi-model consensus for code validation.

```python
# In quality.py - new method
async def validate_code_multi_model(self, sample: TrainingSample) -> float:
    """Use 2+ models to validate code correctness."""
    from agents.training.provider_rotation import ProviderRotation, ProviderWeight

    rotation = ProviderRotation.from_preset(ProviderWeight.BALANCED)

    validation_prompt = f"""
    Validate this 65816 assembly code for correctness.
    Score 0.0-1.0 based on:
    - Correct opcode usage
    - Valid addressing modes
    - Logical register flow
    - No obvious bugs

    Code:
    {sample.output}

    Return JSON: {{"score": 0.X, "issues": ["..."]}}
    """

    scores = []
    for _ in range(2):  # Get 2 opinions
        provider = rotation.select_provider()
        response, model = await self.generate_with_rotation(validation_prompt)
        if response:
            scores.append(extract_score(response))

    # Require consensus - both must agree it's valid
    if len(scores) >= 2:
        return min(scores)  # Conservative: take worst score
    return 0.5  # Uncertain
```

**Impact:** Catches logically incorrect but syntactically valid code.

---

### 2. Tighten Deduplication Threshold

**Problem:** 0.95 similarity threshold allows near-duplicates.

**Current:** `if similarity > 0.95: reject`

**Proposed:**
```python
DEDUP_THRESHOLDS = {
    "asm": 0.88,      # ASM is repetitive - stricter
    "oracle": 0.88,   # Same
    "text": 0.85,     # Text should be more diverse
    "cpp": 0.90,      # Code patterns repeat
    "default": 0.88,
}
```

**Location:** `quality.py` line ~180

---

### 3. Increase KG Coverage Requirements

**Problem:** Only 30% of entities need to exist in KG.

**Current:** `min_coverage = 0.3`

**Proposed:**
```python
KG_COVERAGE = {
    "asm": 0.5,       # Half of mentioned routines should be real
    "oracle": 0.5,
    "gigaleak": 0.4,  # More lenient - less complete KB
    "text": 0.6,      # Higher for text
    "default": 0.5,
}
```

**Location:** `validators/kg_validator.py`

---

## Embedding Coverage (Critical Gap)

### 4. Complete Embedding Generation

**Current Coverage:**
| KB | Embedded | Total | Coverage |
|----|----------|-------|----------|
| Oracle | 8,438 | 8,438 | 100% |
| ALTTP | 500 | 6,591 | 7.6% |
| Gigaleak | 0 | 129,899 | 0% |

**Action Plan:**
```bash
# Priority 1: Finish ALTTP (6,000 remaining)
hafs embeddings generate --kb alttp --batch-size 200 --priority high

# Priority 2: Gigaleak core symbols (top 10K by reference count)
hafs embeddings generate --kb gigaleak --limit 10000 --sort-by refs
```

**Script to add:** `scripts/complete_embedding_coverage.py`

---

## Multi-Model Improvements

### 5. Model-Aware Sample Attribution

**Problem:** All samples use same teacher model - no diversity tracking.

**Solution:** Track and balance model attribution:

```python
@dataclass
class TrainingSample:
    # ... existing fields ...
    teacher_model: str = ""        # Already exists
    teacher_provider: str = ""     # NEW: "gemini", "anthropic", "openai"
    validation_models: list[str]   # NEW: Models that validated this sample
```

**Config target:**
```toml
[generation.model_balance]
gemini = 0.4      # 40% of samples
anthropic = 0.3   # 30% of samples
openai = 0.2      # 20% of samples
local = 0.1       # 10% local model
```

---

### 6. Semantic Cross-Domain Pairing

**Problem:** Cross-domain samples use simple boolean filtering (`is_hook`).

**Solution:** Use embeddings to find semantically related pairs:

```python
async def find_related_pairs(
    vanilla_items: list[SourceItem],
    hack_items: list[SourceItem],
    similarity_threshold: float = 0.7
) -> list[tuple[SourceItem, SourceItem]]:
    """Find vanilla+hack pairs that modify the same functionality."""
    pairs = []

    for hack in hack_items:
        if not hack.metadata.get("is_hook"):
            continue

        # Get embedding for hack
        hack_emb = await get_embedding(hack.content)

        # Find most similar vanilla routine
        best_match = None
        best_score = 0.0

        for vanilla in vanilla_items:
            vanilla_emb = await get_embedding(vanilla.content)
            score = cosine_similarity(hack_emb, vanilla_emb)

            if score > best_score and score > similarity_threshold:
                best_score = score
                best_match = vanilla

        if best_match:
            pairs.append((best_match, hack))

    return pairs
```

**Impact:** Cross-domain samples show genuinely related code, not random pairings.

---

### 7. Template Diversity Tracking

**Problem:** Template rotation tracks usage but not diversity impact.

**Solution:** Score templates by embedding variance:

```python
class DiversityAwareTemplateRotator:
    def __init__(self):
        self.template_diversity_scores = {}  # template_id -> avg diversity

    def select_template(self, domain: str) -> str:
        """Select template that historically produces diverse samples."""
        templates = self.get_templates(domain)

        # Weight by: (1 / usage_count) * diversity_score
        weights = []
        for t in templates:
            usage = self.usage_counts.get(t.id, 1)
            diversity = self.template_diversity_scores.get(t.id, 0.5)
            weight = (1.0 / usage) * diversity
            weights.append(weight)

        return random.choices(templates, weights=weights)[0]

    def record_diversity(self, template_id: str, sample_diversity: float):
        """Update template's diversity score based on generated sample."""
        current = self.template_diversity_scores.get(template_id, 0.5)
        # Exponential moving average
        self.template_diversity_scores[template_id] = 0.9 * current + 0.1 * sample_diversity
```

---

### 8. Domain-Aware Active Learning

**Problem:** Active learning treats all domains equally in embedding space.

**Solution:** Separate embedding spaces per domain:

```python
class DomainAwareActiveLearning:
    def __init__(self):
        self.domain_samplers = {}  # domain -> RegionSampler

    def add_sample(self, sample: TrainingSample):
        domain = sample.domain
        if domain not in self.domain_samplers:
            self.domain_samplers[domain] = RegionSampler(num_regions=50)

        self.domain_samplers[domain].add(sample)

    def get_sparse_regions(self, domain: str) -> list[int]:
        """Get under-sampled regions for a specific domain."""
        if domain not in self.domain_samplers:
            return []
        return self.domain_samplers[domain].get_sparse_regions()
```

---

## Feedback Loop Improvements

### 9. Retry Failed Samples with Alternative Prompts

**Problem:** Rejected samples are logged but not retried.

**Solution:** Automatic retry with template variation:

```python
async def generate_with_retry(
    self,
    item: SourceItem,
    max_retries: int = 3
) -> Optional[TrainingSample]:
    """Generate sample, retrying with different templates on failure."""

    used_templates = set()

    for attempt in range(max_retries):
        # Get unused template
        template = self.template_rotator.select_template(
            self.domain,
            exclude=used_templates
        )
        used_templates.add(template.id)

        sample = await self.generate_sample(item, template=template)

        if sample and sample.quality_score >= self.threshold:
            return sample

        # Log retry
        logger.info(f"Retry {attempt+1}/{max_retries} for {item.name}")

    return None  # All retries failed
```

---

### 10. Adaptive Quality Thresholds

**Problem:** Thresholds are static despite acceptance rate data.

**Solution:** Auto-adjust based on rolling acceptance rate:

```python
class AdaptiveThresholds:
    def __init__(self, target_acceptance: float = 0.85):
        self.target = target_acceptance
        self.thresholds = dict(DOMAIN_THRESHOLDS)
        self.window = deque(maxlen=100)  # Last 100 samples

    def update(self, domain: str, accepted: bool):
        self.window.append((domain, accepted))

        # Calculate domain-specific acceptance rate
        domain_samples = [(d, a) for d, a in self.window if d == domain]
        if len(domain_samples) < 20:
            return  # Not enough data

        rate = sum(a for _, a in domain_samples) / len(domain_samples)

        # Adjust threshold
        if rate < self.target - 0.1:
            # Too many rejections - lower threshold
            self.thresholds[domain] *= 0.95
        elif rate > self.target + 0.1:
            # Too lenient - raise threshold
            self.thresholds[domain] *= 1.05

        # Clamp to reasonable range
        self.thresholds[domain] = max(0.3, min(0.8, self.thresholds[domain]))
```

---

## Data Source Improvements

### 11. Gigaleak Integration

**Problem:** 129,899 symbols completely unused.

**Solution:** Create dedicated generator:

```python
class GigaleakGenerator(DataGenerator):
    """Generate training data from Nintendo original ALTTP source."""

    def __init__(self):
        super().__init__(
            name="GigaleakGenerator",
            domain="gigaleak",
            teacher_tier="coding",
        )

    async def extract_source_items(self) -> list[GigaleakSourceItem]:
        symbols = load_json("~/.context/knowledge/gigaleak/symbols.json")

        items = []
        for sym in symbols:
            # Prioritize symbols with Japanese comments
            if sym.get("japanese_comment"):
                items.append(GigaleakSourceItem(
                    name=sym["name"],
                    content=sym.get("code_context", ""),
                    japanese=sym["japanese_comment"],
                    english=sym.get("english_translation", ""),
                    source="gigaleak",
                ))

        return items
```

---

### 12. Cross-KB Linking

**Problem:** No links between Oracle hooks and ALTTP vanilla routines.

**Solution:** Build relationship map:

```python
async def build_cross_kb_links():
    """Map Oracle hooks to their ALTTP vanilla targets."""

    oracle_routines = load_oracle_routines()
    alttp_routines = load_alttp_routines()

    links = {}

    for oracle in oracle_routines:
        if not oracle.get("is_hook"):
            continue

        # Many hooks have vanilla_target in metadata
        vanilla_name = oracle.get("hooks_routine") or oracle.get("vanilla_target")

        if vanilla_name and vanilla_name in alttp_routines:
            links[oracle["name"]] = {
                "oracle": oracle,
                "vanilla": alttp_routines[vanilla_name],
                "relationship": "hooks",
            }

    save_json("~/.context/knowledge/cross_kb_links.json", links)
    return len(links)
```

---

## Implementation Priority

| # | Improvement | Effort | Impact | Priority |
|---|-------------|--------|--------|----------|
| 1 | Multi-model code validation | Medium | High | P0 |
| 2 | Tighten dedup threshold | Low | Medium | P0 |
| 3 | Increase KG coverage | Low | Medium | P0 |
| 4 | Complete embeddings | High | High | P1 |
| 5 | Model attribution | Low | Medium | P1 |
| 6 | Semantic cross-domain | Medium | High | P1 |
| 7 | Template diversity | Medium | Medium | P2 |
| 8 | Domain-aware active learning | Medium | Medium | P2 |
| 9 | Retry with alternatives | Low | Medium | P2 |
| 10 | Adaptive thresholds | Medium | Medium | P2 |
| 11 | Gigaleak generator | High | High | P1 |
| 12 | Cross-KB linking | Medium | High | P1 |

---

## Quick Wins (Can Do Now)

```python
# 1. Tighten dedup in quality.py
SIMILARITY_THRESHOLD = 0.88  # was 0.95

# 2. Increase KG coverage in kg_validator.py
MIN_KG_COVERAGE = 0.5  # was 0.3

# 3. Add provider tracking to TrainingSample
teacher_provider: str = ""  # Add field
```

---

## Metrics to Track

After improvements:
- **Acceptance rate per domain** (target: 85%)
- **Model diversity** (% from each provider)
- **Embedding coverage** (target: 80%+ for all KBs)
- **Cross-domain pair quality** (semantic similarity score)
- **Template diversity impact** (avg diversity per template)
