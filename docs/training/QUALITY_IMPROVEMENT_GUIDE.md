# Training Data Quality Improvement Guide

**Last updated:** 2025-12-21
**Status:** 🎯 Active Development

## Current Quality Baseline

### Validation Pilot Results (2025-12-21)

**Before Improvements:**
- Pass rate: 1% (99% rejected)
- Main issues: Format errors, hallucinations, low coherence

**After Improvements (v1):**
- Pass rate: 62.7% (37.3% rejected)
- Improvement: +61.7 percentage points
- Quality thresholds: Domain-specific (ASM: 0.4, Oracle: 0.4, YAZE: 0.5)

**Remaining Challenges:**
- ASM instruction/output alignment
- Oracle hook explanation clarity
- YAZE C++ code structure quality

## Domain-Specific Quality Metrics

### ASM (65816 Assembly)

**Threshold:** 0.4 (40% minimum quality score)

**Quality Components:**
```python
# From quality.py scoring
diversity_score = 0.0-1.0      # Embedding uniqueness
kg_consistency = 0.0-1.0       # Entity validation (SNES registers, mnemonics)
hallucination_risk = 0.0-1.0   # Lower is better
semantic_coherence = 0.0-1.0   # Instruction/output alignment

overall_quality = (diversity + kg_consistency + (1 - hallucination) + coherence) / 4
```

**Common Issues:**
1. **Low Coherence (< 0.4):**
   - Instruction asks for one thing, output does another
   - Missing register explanations
   - Incomplete code snippets

2. **KG Inconsistency (< 0.5):**
   - Using non-existent RAM addresses
   - Incorrect hardware register names
   - Fictional SNES capabilities

3. **High Hallucination (> 0.5):**
   - "I think", "maybe", "possibly" language
   - Vague or uncertain explanations
   - Output much longer/shorter than instruction warrants

**Reference Example (High Quality - Score: 0.85):**

```json
{
  "instruction": "Write an assembly routine that loads Link's current state from WRAM and checks if he's in a specific animation state",
  "input": "The player state byte is stored at $7E:0E20. Animation states range from 0x00-0x1F. We need to check if Link is in state 0x0C (sword swinging). Preserve all registers except accumulator.",
  "output": "```asm\nCheckPlayerSwinging:\n    ; Load Link's current state from WRAM\n    PHP                 ; Preserve processor status\n    REP #$20            ; Set A to 16-bit mode\n    LDA.w $0E20         ; Load player state word ($7E:0E20-$7E:0E21)\n    AND.w #$001F        ; Mask lower 5 bits (state index 0-31)\n    CMP.w #$000C        ; Compare with state 0x0C (sword swing)\n    PLP                 ; Restore processor status\n    BEQ .IsSwinging     ; Branch if equal (Z flag set)\n    ; Not swinging\n    CLC                 ; Clear carry (return false)\n    RTS\n.IsSwinging:\n    SEC                 ; Set carry (return true)\n    RTS\n```\n\nThis routine loads the player state byte from $7E:0E20 (Link's current animation/action state). The AND operation masks the lower 5 bits because only bits 0-4 contain the state index (0x00-0x1F, 32 possible states). State 0x0C specifically represents Link's sword-swinging animation. The routine returns with carry set if swinging, clear otherwise.",
  "domain": "asm",
  "quality_score": 0.85
}
```

**Anti-Pattern Example (Low Quality - Score: 0.32):**

```json
{
  "instruction": "Create a routine to check player state",
  "input": "Player data is in memory somewhere",
  "output": "```asm\nCheckState:\n    LDA $0E20  ; I think this loads the state?\n    CMP #$0C   ; Maybe compare to 12?\n    RTS        ; Return\n```\n\nThis routine checks if the player is doing something. The LDA instruction loads from memory and CMP compares it to a value.",
  "domain": "asm",
  "quality_score": 0.32,
  "issues": [
    "Vague instruction (\"check player state\" - check what?)",
    "Uncertain language (\"I think\", \"maybe\")",
    "Missing register context (8-bit vs 16-bit)",
    "Incomplete explanation (what does $0E20 contain?)",
    "No addressing mode suffix (.w vs .b)",
    "Missing preservation (doesn't save/restore P register)"
  ]
}
```

### Oracle (ROM Hack Modifications)

**Threshold:** 0.4 (40% minimum quality score)

**Common Issues:**
1. **Missing Vanilla Context:**
   - Doesn't explain original behavior before modification
   - No comparison of vanilla vs hack code
   - Missing hook addresses

2. **Incomplete ROM Hacking Technique:**
   - Doesn't explain WHY this approach was chosen
   - Missing bank allocation strategy
   - No code injection method details (org vs pushpc/pullpc)

3. **Poor Integration Explanation:**
   - Doesn't show how it connects to other hack components
   - Missing call graph context
   - No testing guidance

**Reference Example (High Quality - Score: 0.78):**

```json
{
  "instruction": "Explain how Oracle-of-Secrets implements custom sprite loading to expand sprite variety beyond vanilla ALTTP's 128 sprite limit",
  "input": "Vanilla ALTTP loads sprite graphics from banks $09-$0B using a simple lookup table indexed by sprite ID (0x00-0x7F). The Oracle ROM hack needs to support 256 sprites (0x00-0xFF) and uses custom sprite banks in expanded ROM space ($32-$35).",
  "output": "**Vanilla Behavior:**\nIn the original ALTTP, sprite graphics loading occurs at $0D:B4E0 in the LoadSpriteGfx routine. The game uses sprite ID as a direct index into a table at $0D:B500, which maps to graphics banks $09-$0B. This limits the game to 128 unique sprites (7-bit ID).\n\n**Hook Implementation:**\nOracle redirects this routine using a JSL hook:\n```asm\n; Vanilla code at $0D:B4E0 (5 bytes)\norg $0DB4E0\n    JSL OracleCustomSpriteLoader  ; Jump to bank $32 (4 bytes)\n    NOP                           ; Fill remaining byte\n```\n\n**Custom Sprite Loader (Bank $32):**\n```asm\norg $32F000\nOracleCustomSpriteLoader:\n    ; Expand sprite ID from 7-bit to 8-bit\n    PHP                  ; Save processor status\n    REP #$20             ; 16-bit A\n    LDA.b $00            ; Load sprite ID from DP\n    AND.w #$00FF         ; Mask to 8 bits (0x00-0xFF)\n    \n    ; Use room number as bank selector\n    XBA                  ; Swap bytes\n    LDA.b $A0            ; Load current room number\n    LSR A                ; Divide by 4 (bank groups)\n    LSR A\n    AND.w #$0003         ; Limit to 4 banks ($32-$35)\n    XBA                  ; Restore sprite ID\n    \n    ; Calculate graphics address\n    ASL A                ; Multiply ID by 2 (word index)\n    TAX                  ; Transfer to index\n    LDA.l SpriteGfxTable,x  ; Load address from expanded table\n    STA.b $02            ; Store target address\n    \n    PLP                  ; Restore status\n    RTL                  ; Return to vanilla code\n\nSpriteGfxTable:\n    ; 256 entries x 2 bytes = 512 bytes\n    .dw $3200, $3210, $3220, ...  ; Graphics addresses in banks $32-$35\n```\n\n**Why This Approach:**\n1. **JSL Hook:** Allows jumping to expanded ROM (bank $32+) while preserving return address\n2. **Room-Based Banking:** Distributes sprites across 4 banks based on room number, avoiding bank overflow\n3. **Backward Compatible:** Sprite IDs 0x00-0x7F still map to vanilla graphics for compatibility\n4. **Expandable:** Can easily add more banks ($36+) by changing the AND mask\n\n**Integration:**\n- Called by: Sprite initialization ($0D:8000), dynamic sprite spawner ($0D:8500)\n- Requires: Custom graphics data in banks $32-$35 (must be included in ROM build)\n- Tested: Load each sprite ID in test rooms, verify graphics display correctly\n\n**Common Pitfalls:**\n- Forgetting to include graphics data in banks $32-$35 (causes glitched sprites)\n- Not preserving processor status (P register) - causes A/X/Y width mismatches\n- Using LDA.w instead of LDA.l for long addressing (wrong bank)\n",
  "domain": "oracle",
  "quality_score": 0.78
}
```

### YAZE (C++ Code Analysis)

**Threshold:** 0.5 (50% minimum quality score)

**Common Issues:**
1. **Missing Context:**
   - Doesn't explain what YAZE is (Zelda 3 editor)
   - No explanation of data structures
   - Missing dependencies/includes

2. **Poor Code Quality:**
   - Inconsistent naming conventions
   - Missing error handling
   - No const correctness

3. **Insufficient Explanation:**
   - Doesn't explain algorithm complexity
   - Missing edge case handling
   - No testing examples

## Improvement Strategies

### 1. Enhanced Generator Prompts

**Current ASM Prompt Issues:**
- Too generic ("Generate high-quality training data")
- Missing concrete examples of good vs bad output
- No explicit anti-patterns to avoid

**Improved Prompt Structure:**
```
You are an expert SNES 65816 assembly programmer.

TASK: Generate training data for this assembly routine.
[Routine details]

REQUIRED OUTPUT QUALITY:
✅ DO:
- Use precise register modes (.b for 8-bit, .w for 16-bit)
- Explain WHY registers are preserved (PHP/PLP)
- Include full addresses ($7E:0E20, not just $0E20)
- Add comments explaining INTENT, not just actions
- Show complete, working code (no TODO or ellipsis)

❌ DON'T:
- Use uncertain language ("I think", "maybe", "probably")
- Leave incomplete code snippets (..., etc.)
- Forget addressing modes (LDA without .w/.b)
- Omit register preservation when needed
- Use fictional registers or addresses

REFERENCE EXAMPLE:
[Show a high-quality example]

ANTI-PATTERN EXAMPLE:
[Show what NOT to do]

Generate JSON:
{
  "instruction": "Specific, actionable request",
  "input": "Technical context (addresses, registers, constraints)",
  "output": "Complete code with explanatory comments"
}
```

### 2. Reference Examples Library

**Create:** `data/training_reference_examples.json`

```json
{
  "asm": {
    "high_quality": [
      {
        "instruction": "...",
        "quality_score": 0.85,
        "why_good": "Complete code, precise addressing, clear explanations"
      }
    ],
    "low_quality": [
      {
        "instruction": "...",
        "quality_score": 0.28,
        "why_bad": "Uncertain language, missing context, incomplete code"
      }
    ]
  },
  "oracle": { ... },
  "yaze": { ... }
}
```

### 3. Iterative Refinement Loop

```python
# Pseudo-code for quality improvement
for iteration in range(max_iterations):
    # Generate samples with current prompts
    samples = await generator.generate(n=1000)

    # Filter and analyze
    passed, failed = quality_pipeline.filter_samples(samples)

    # Analyze failure patterns
    patterns = quality_feedback.analyze_rejections(failed)

    # Update prompts
    if patterns["high_hallucination"] > 0.3:
        prompts.add_anti_pattern("uncertain_language")

    if patterns["low_coherence"] > 0.3:
        prompts.add_reference_example("coherent_asm")

    # Re-generate failed samples with improved prompts
    improved = await generator.regenerate(failed, updated_prompts)

    # Track improvement
    improvement = len(passed_improved) / len(failed)
    if improvement < 0.1:
        break  # Diminishing returns
```

### 4. Active Learning Coverage

**Goal:** Balance dataset across embedding space

```python
# Check coverage
coverage_report = quality_pipeline.get_coverage_report()

# Identify sparse regions
sparse_regions = coverage_report["sparse_regions"]
# Output: [(region_id, sample_count, avg_quality), ...]

# Generate targeted samples for sparse regions
for region in sparse_regions[:10]:  # Top 10 sparse
    # Find representative sample in region
    rep_sample = active_learner.get_representative(region)

    # Generate variations
    variations = await generator.generate_similar(
        base_sample=rep_sample,
        num_variations=50,
        diversity_target=0.3  # Not too different
    )
```

### 5. Human-in-the-Loop Validation

**Process:**
1. Generate 1000 samples with improved prompts
2. Quality pipeline filters to ~400-600 (40-60%)
3. **Manual review:** Randomly sample 50 (10% of passed)
4. Annotate quality issues:
   - Technical errors
   - Coherence problems
   - Missing explanations
5. Use annotations to further refine prompts

**Annotation Interface:**
```json
{
  "sample_id": "asm_001234",
  "human_rating": 0.9,
  "human_feedback": "Excellent explanation of register preservation",
  "issues_found": [],
  "suggestions": "Could mention timing (cycle count)"
}
```

## Prompt Engineering Best Practices

### Specificity Spectrum

**❌ Too Generic:**
```
Generate training data for this code.
```

**⚠️ Somewhat Specific:**
```
Generate an instruction and output for this assembly routine.
Include comments explaining what it does.
```

**✅ Highly Specific:**
```
Generate training data for this 65816 assembly routine.

INSTRUCTION: A specific, technical question about:
- Game mechanic implementation
- Hardware interaction
- Optimization technique
- RAM/register manipulation

INPUT: Technical context (2-3 sentences):
- RAM addresses used ($7E:XXXX format)
- Hardware registers ($21XX for PPU, $42XX for CPU)
- Variables and constants
- Constraints (timing, preservation)

OUTPUT: Complete assembly routine with:
- Function header comment (what it does)
- Line-by-line comments (what AND why)
- Proper addressing modes (.b/.w)
- Register preservation (PHP/PLP when needed)
- No placeholders or incomplete code

QUALITY CHECKS:
✅ Uses real SNES registers ($2100-$21FF, $4200-$43FF)
✅ Includes addressing mode suffixes
✅ Explains INTENT, not just actions
✅ Shows complete, working code
❌ No uncertain language ("maybe", "I think")
❌ No placeholders (..., TODO, etc.)
```

### Few-Shot Examples

**Include 2-3 reference examples in prompt:**

```
EXAMPLE 1 (High Quality):
{
  "instruction": "Write a routine to check if Link is swimming",
  "input": "Player state is at $7E:0E20. Swimming states are 0x04-0x05. Return carry set if swimming.",
  "output": "```asm\nCheckPlayerSwimming:\n    PHP\n    REP #$20\n    LDA.w $0E20\n    AND.w #$00FF\n    CMP.w #$0004\n    BCC .NotSwimming\n    CMP.w #$0006\n    BCS .NotSwimming\n    PLP\n    SEC\n    RTS\n.NotSwimming:\n    PLP\n    CLC\n    RTS\n```\n..."
}

EXAMPLE 2 (Anti-Pattern - DON'T DO THIS):
{
  "instruction": "Check player state",
  "input": "State is in memory",
  "output": "LDA $0E20  ; I think this works?\nCMP #$04\n..."
}
```

### Domain-Specific Constraints

**ASM:**
- MUST use proper addressing modes (.b, .w, .l)
- MUST include register preservation when modifying A/X/Y/P
- MUST use real SNES addresses ($0000-$1FFF WRAM, $7E:0000-$7F:FFFF RAM, $00:0000-$FF:FFFF ROM)
- MUST explain hardware timing when relevant (NMI, VBlank)

**Oracle:**
- MUST explain vanilla behavior BEFORE hack modifications
- MUST show exact hook address and injection method
- MUST explain bank allocation strategy
- MUST include integration with other hack components

**YAZE:**
- MUST explain YAZE context (Zelda 3 editor/emulator)
- MUST use proper C++ style (const correctness, RAII)
- MUST include error handling
- MUST explain algorithm complexity

## Quality Monitoring Dashboard

**Metrics to Track:**
```json
{
  "campaign_id": "alpha_pilot_20_20251221",
  "total_generated": 34500,
  "total_passed": 21632,
  "pass_rate": 0.627,

  "by_domain": {
    "asm": {
      "generated": 24000,
      "passed": 14400,
      "pass_rate": 0.60,
      "avg_quality": 0.52,
      "common_issues": [
        "low_coherence: 45%",
        "missing_addressing_modes: 30%",
        "incomplete_code: 25%"
      ]
    },
    "oracle": {
      "generated": 7000,
      "passed": 4900,
      "pass_rate": 0.70,
      "avg_quality": 0.58,
      "common_issues": [
        "missing_vanilla_context: 40%",
        "incomplete_hook_explanation: 35%"
      ]
    }
  },

  "quality_trends": {
    "diversity": [0.62, 0.65, 0.68, 0.71],  # Improving
    "kg_consistency": [0.75, 0.76, 0.77, 0.77],  # Stable
    "hallucination": [0.35, 0.32, 0.28, 0.25],  # Improving (lower is better)
    "coherence": [0.48, 0.51, 0.54, 0.57]  # Improving
  },

  "generator_performance": {
    "gemini-2.0-flash": {
      "samples_generated": 34500,
      "avg_quality": 0.55,
      "cost_usd": 38.50,
      "samples_per_dollar": 896
    }
  }
}
```

**Visualization:**
```
Quality Score Distribution (ASM Domain)
───────────────────────────────────────
0.0-0.2: ███░░░░░░░ (8%)   ← Rejected
0.2-0.4: ████████░░ (32%)  ← Rejected
0.4-0.6: ██████████ (40%)  ← Borderline
0.6-0.8: ████░░░░░░ (15%)  ← Good
0.8-1.0: ██░░░░░░░░ (5%)   ← Excellent

Pass Rate Trend
───────────────
60% ████████████████░░░░  ← Current
50% ██████████████░░░░░░
40% ████████████░░░░░░░░  ← v0 baseline
30% ██████████░░░░░░░░░░
    Week 1  Week 2  Week 3
```

## Next Steps

1. **Implement Enhanced Prompts** (this PR)
   - Update `asm_generator.py` with reference examples
   - Update `oracle_generator.py` with vanilla/hack structure
   - Add anti-patterns and quality checks

2. **Create Reference Library**
   - Curate 10 high-quality examples per domain
   - Curate 10 anti-pattern examples per domain
   - Store in `data/training_reference_examples.json`

3. **A/B Test Prompts**
   - Generate 1000 samples with old prompts (baseline)
   - Generate 1000 samples with new prompts (v2)
   - Compare pass rates and quality scores

4. **Iterative Refinement**
   - Run 3-5 iteration cycles
   - Track improvement per iteration
   - Stop when diminishing returns (< 5% improvement)

5. **Deploy to Production**
   - Launch full 43K sample campaign with v2 prompts
   - Monitor quality metrics in real-time
   - Adjust thresholds based on results

## References

- Quality pipeline: `src/agents/training/quality.py`
- ASM generator: `src/agents/training/generators/asm_generator.py`
- Oracle generator: `src/agents/training/generators/oracle_generator.py`
- Feedback tracker: `src/agents/training/feedback/quality_tracker.py`
- Active learning: `src/agents/training/active_learning.py`
