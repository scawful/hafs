"""Enhanced prompt templates for training data generators.

These prompts incorporate:
- Reference examples (high quality and anti-patterns)
- Explicit quality requirements
- Domain-specific constraints
- Clear formatting guidelines

Usage:
    from agents.training.generators.enhanced_prompts import get_enhanced_asm_prompt

    prompt = get_enhanced_asm_prompt(routine_name, code, context)
"""

from __future__ import annotations


def get_enhanced_asm_prompt(
    routine_name: str,
    code: str,
    bank: str = "",
    description: str = "",
    memory_access: list[str] | None = None,
    address: str = "",
) -> str:
    """Enhanced ASM generator prompt with quality improvements.

    Args:
        routine_name: Name of the assembly routine
        code: Assembly code to generate training data from
        bank: ROM bank (e.g., "$0D")
        description: Human-readable description
        memory_access: List of memory addresses accessed
        address: ROM address (e.g., "$0D:8000")

    Returns:
        Enhanced prompt string
    """
    memory_context = ", ".join(memory_access) if memory_access else "None specified"

    # Truncate code if too long (keep first 80 lines)
    code_lines = code.split("\n")
    if len(code_lines) > 80:
        code = "\n".join(code_lines[:80]) + "\n... (truncated)"

    return f"""You are an expert SNES 65816 assembly programmer specializing in Zelda: A Link to the Past.

Generate high-quality training data for this assembly routine.

ROUTINE: {routine_name}
BANK: {bank}
ADDRESS: {address}
DESCRIPTION: {description}
MEMORY ACCESS: {memory_context}

CODE:
```asm
{code}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Generate a JSON object with THREE fields
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. "instruction": A clear, specific, technical request for this code.
   Make it varied and realistic:
   • Request implementation of a specific game mechanic
   • Ask for hardware interaction (PPU, DMA, interrupts)
   • Request optimization of a routine
   • Ask for code that manipulates specific RAM/registers

   Examples:
   ✅ "Write a routine to check if Link is swimming using the player state flags"
   ✅ "Implement a function that waits for VBlank before writing to VRAM"
   ❌ "Create a routine"  (too vague)
   ❌ "Write some code to check player state"  (not specific)

2. "input": Technical context (2-4 sentences) to help write this code:
   • RAM addresses used (format: $7E:XXXX for WRAM, $0000-$1FFF for low RAM)
   • Hardware registers (PPU: $2100-$21FF, CPU: $4200-$43FF, APU: $2140-$2143)
   • Key variables, constants, or flags
   • Constraints (timing, register preservation, bank restrictions)

   Example:
   "The player state byte is at $7E:0E20. Bit 7 indicates swimming (1 = swimming).
   The routine must preserve X and Y registers. Return with carry set if swimming."

3. "output": Complete assembly routine with detailed inline comments.

   FORMAT:
   ```asm
   RoutineName:
       ; [Brief overview - what this routine does in 1-2 sentences]

       [Complete code with line-by-line comments]
   ```

   COMMENT REQUIREMENTS:
   • Explain INTENT, not just actions
     ✅ "Load player state (swimming/walking/climbing)"
     ❌ "Load from $0E20"

   • Explain register preservation
     ✅ "PHP    ; Preserve processor status (A/X/Y width modes)"
     ❌ "PHP    ; Push processor status"

   • Document memory addresses
     ✅ "LDA.w $0E20    ; Load player state ($7E:0E20)"
     ❌ "LDA.w $0E20"

   • Explain hardware timing when relevant
     ✅ "LDA.b $4212    ; Read HVBJOY (wait for VBlank bit 7)"
     ❌ "LDA.b $4212    ; Read register"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY REQUIREMENTS - MUST FOLLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ DO:
  • Use proper addressing mode suffixes (.b for 8-bit, .w for 16-bit, .l for long)
  • Include register preservation (PHP/PLP) when modifying A/X/Y/P registers
  • Use full address format ($7E:XXXX for RAM, $XX:XXXX for ROM)
  • Add comments explaining INTENT and WHY, not just WHAT
  • Show complete, working code (no placeholders like "...", "TODO", "etc.")
  • Use real SNES addresses:
    - WRAM: $0000-$1FFF (direct page, stack, low RAM)
    - RAM: $7E:0000-$7F:FFFF (64KB work RAM)
    - ROM: $00:8000-$FF:FFFF (depends on mapping mode)
    - PPU: $2100-$21FF (graphics, sprites, palettes)
    - CPU: $4200-$43FF (DMA, interrupts, multiplication)
    - APU: $2140-$2143 (sound)
  • Explain hardware timing (VBlank, NMI, DMA) when relevant

❌ DON'T:
  • Use uncertain language ("I think", "maybe", "probably", "might")
  • Leave incomplete code (..., TODO, etc., [code here])
  • Forget addressing mode suffixes (LDA instead of LDA.w or LDA.b)
  • Omit register preservation when caller expects it
  • Use fictional registers or addresses
  • Give vague explanations ("does stuff", "handles things")
  • Write comments that just repeat the instruction
    ❌ "LDA.w $0E20    ; Load accumulator with $0E20"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1: HIGH QUALITY (Score: 0.85)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instruction: "Write an assembly routine that loads Link's current state from WRAM and checks if he's in a specific animation state"

Input: "The player state byte is stored at $7E:0E20. Animation states range from 0x00-0x1F. We need to check if Link is in state 0x0C (sword swinging). Preserve all registers except accumulator."

Output:
```asm
CheckPlayerSwinging:
    ; Load Link's current state from WRAM and check for sword swing
    PHP                 ; Preserve processor status (8/16-bit modes)
    REP #$20            ; Set A to 16-bit mode
    LDA.w $0E20         ; Load player state word ($7E:0E20-$7E:0E21)
    AND.w #$001F        ; Mask lower 5 bits (state index 0-31)
    CMP.w #$000C        ; Compare with state 0x0C (sword swing)
    PLP                 ; Restore processor status
    BEQ .IsSwinging     ; Branch if equal (Z flag set)
    ; Not swinging
    CLC                 ; Clear carry (return false)
    RTS
.IsSwinging:
    SEC                 ; Set carry (return true)
    RTS
```

This routine loads the player state byte from $7E:0E20 (Link's current animation/action state in WRAM). The AND operation masks the lower 5 bits because only bits 0-4 contain the state index (0x00-0x1F, 32 possible states). State 0x0C specifically represents Link's sword-swinging animation. The routine returns with carry set if swinging, clear otherwise. We preserve the processor status register (P) because the caller may be in 8-bit or 16-bit mode.

WHY THIS IS GOOD:
• Complete code with proper addressing modes (.w)
• Register preservation (PHP/PLP) with explanation
• Full address format ($7E:0E20)
• Clear explanations of INTENT (why mask, why state 0x0C)
• Technical precision (bit ranges, flag meanings)
• Explains return value (carry flag convention)


ANTI-PATTERN: LOW QUALITY (Score: 0.28) - DON'T DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Instruction: "Create a routine to check player state"

Input: "Player data is in memory somewhere"

Output:
```asm
CheckState:
    LDA $0E20  ; I think this loads the state?
    CMP #$0C   ; Maybe compare to 12?
    RTS        ; Return
```

This routine checks if the player is doing something. The LDA instruction loads from memory and CMP compares it to a value.

WHY THIS IS BAD:
❌ Vague instruction ("check player state" - check WHAT?)
❌ Uncertain language ("I think", "maybe")
❌ Missing addressing modes (.w or .b)
❌ Incomplete address ($0E20 instead of $7E:0E20)
❌ No register preservation
❌ Minimal explanation (doesn't explain WHY)
❌ Vague input ("memory somewhere")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate your JSON response now. Follow the HIGH QUALITY example pattern.

JSON FORMAT:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""


def get_enhanced_oracle_prompt(
    routine_name: str,
    code_snippet: str,
    address: str = "",
    file_path: str = "",
    description: str = "",
    category: str = "",
    is_hook: bool = False,
    hooks_vanilla: str | None = None,
    calls: list[str] | None = None,
    called_by: list[str] | None = None,
) -> str:
    """Enhanced Oracle ROM hack generator prompt.

    Args:
        routine_name: Name of the ROM hack routine
        code_snippet: Assembly code from Oracle-of-Secrets
        address: ROM address
        file_path: Source file path
        description: Human description
        category: Category (sprites, items, dungeons, etc.)
        is_hook: Whether this hooks vanilla code
        hooks_vanilla: Name of vanilla routine being hooked
        calls: List of routines this calls
        called_by: List of routines that call this

    Returns:
        Enhanced prompt string
    """
    # Truncate code if too long
    code_lines = code_snippet.split("\n")
    if len(code_lines) > 60:
        code_snippet = "\n".join(code_lines[:60]) + "\n... (truncated)"

    # Build context sections
    context_parts = []
    if file_path:
        context_parts.append(f"Source: {file_path}")
    if category:
        context_parts.append(f"Category: {category}")
    if is_hook and hooks_vanilla:
        context_parts.append(f"⚠️  HOOK - Modifies vanilla routine: {hooks_vanilla}")
    if address:
        context_parts.append(f"ROM Address: {address}")
    if calls:
        context_parts.append(f"Calls: {', '.join(calls[:5])}")
    if called_by:
        context_parts.append(f"Called by: {', '.join(called_by[:5])}")

    context = "\n".join(context_parts) if context_parts else "No additional context"

    hook_type = "HOOK (modifies vanilla)" if is_hook else "NEW CODE (custom addition)"

    return f"""You are an expert ROM hacker specializing in SNES and ALTTP modifications.

Generate high-quality training data for this Oracle-of-Secrets ROM hack routine.

ROUTINE: {routine_name}
TYPE: {hook_type}
{f"VANILLA ROUTINE HOOKED: {hooks_vanilla}" if hooks_vanilla else ""}
CATEGORY: {category}
DESCRIPTION: {description}

CONTEXT:
{context}

CODE:
```asm
{code_snippet}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK: Generate a JSON object with THREE fields
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. "instruction": A clear, pedagogical question about this ROM hack.
   Focus on teaching ROM hacking techniques:
   • Implementation of specific features/modifications
   • Explanation of hooking/patching techniques
   • Vanilla vs hack behavior comparisons
   • Guidance on adding similar custom content

   Examples:
   ✅ "Explain how Oracle implements custom sprite loading to expand beyond vanilla's 128 sprite limit"
   ✅ "How does Oracle hook the damage calculation routine to add new item effects?"
   ❌ "How does Oracle work?"  (too vague)
   ❌ "Describe this code"  (not pedagogical)

2. "input": Technical context (2-4 sentences):
   • Vanilla behavior being modified (if hook)
   • ROM bank and address information
   • Related routines in call graph
   • Technical constraints or requirements

   Example:
   "Vanilla ALTTP loads sprites from banks $09-$0B with 7-bit IDs (128 max).
   Oracle needs 256 sprites and uses expanded ROM banks $32-$35. The hook
   must preserve compatibility with vanilla sprite IDs 0x00-0x7F."

3. "output": Comprehensive ROM hacking tutorial (200-400 words).

   REQUIRED STRUCTURE:

   **Vanilla Behavior:**  (REQUIRED for hooks)
   • What the original game does at this address
   • Original code/algorithm
   • Why it's limited or needs modification

   **Hook Implementation:**  (if applicable)
   • Exact vanilla address being hooked ($XX:XXXX)
   • Code injection method (org directive, JSL redirect, etc.)
   • Why this hooking approach was chosen

   **Modified Behavior:**
   • What the hack does differently
   • Line-by-line code explanation with assembly details
   • Bank allocation strategy (using $20-$FF expanded ROM)

   **ROM Hacking Technique:**  (CRITICAL - explain the technique!)
   • WHY this approach works
   • Alternative approaches (and why they weren't used)
   • How it integrates with existing game systems
   • Technical trade-offs

   **Integration & Testing:**
   • What other routines call this / are called by this
   • How to test the modification
   • Common pitfalls when implementing similar features

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUALITY REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ DO:
  • Explain vanilla behavior FIRST (before modifications)
  • Show exact hook addresses ($XX:XXXX format)
  • Explain the ROM hacking TECHNIQUE, not just the result
  • Include concrete code examples with assembly syntax
  • Explain bank allocation strategy (why bank $32 vs $0D?)
  • Show integration (what calls this? what does it call?)
  • Include testing approach
  • List common pitfalls

❌ DON'T:
  • Skip vanilla behavior explanation
  • Just describe what the code does (explain HOW and WHY)
  • Use vague addresses ("somewhere in bank $0D")
  • Omit the hooking method (org? JSL? pushpc/pullpc?)
  • Forget to explain integration
  • Use uncertain language ("probably", "I think", "maybe")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REFERENCE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HIGH QUALITY OUTPUT STRUCTURE:

**Vanilla Behavior:**
In the original ALTTP, [routine name] at [address] performs [function].
The vanilla code uses [approach], which limits [capability] because [reason].

[Show vanilla code snippet if relevant]

**Hook Implementation:**
Oracle redirects this routine using a [hook type]:
```asm
; Original vanilla code at $XX:YYYY
org $XXYYYY
    JSL CustomRoutine  ; Jump to bank $ZZ
    NOP #N             ; Fill remaining bytes
```

**Modified Behavior:**
The custom routine (shown above) [does what differently]. It works by:

```asm
[Commented code with line-by-line explanations]
```

**Why This Approach:**
1. [Technique 1]: Explanation of why this works
2. [Technique 2]: Alternative considered and why it wasn't used
3. [Technique 3]: Integration benefit

**Integration:**
- Called by: [List of callers]
- Requires: [Dependencies, resources, data]
- Build: [How it's included in ROM]

**Testing:**
- [Test procedure 1]
- [Test procedure 2]
- [Verification method]

**Common Pitfalls:**
1. [Pitfall 1]: What happens if you forget X
2. [Pitfall 2]: Error if you use Y instead of Z

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Generate your JSON response now. Follow the structure above.

JSON FORMAT:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""
