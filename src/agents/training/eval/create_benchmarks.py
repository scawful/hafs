#!/usr/bin/env python3
"""Create benchmark test sets for model evaluation.

Generates curated test sets for ASM, ROM hacking, and code understanding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def create_asm_benchmark() -> list[dict]:
    """Create ASM benchmark test set."""
    return [
        # Basic Operations
        {
            "instruction": "Write a routine to clear the OAM buffer (128 sprites).",
            "input": "Use efficient 65816 assembly. The OAM buffer starts at $0800.",
            "output": """ClearOAMBuffer:
    LDA.b #$F0           ; Load Y-coordinate for off-screen sprites
    LDX.b #$00           ; Start at OAM entry 0
.loop:
    STA.w $0800,X        ; Write to OAM Y position
    INX
    INX
    INX
    INX                  ; Next OAM entry (4 bytes each)
    CPX.b #$00           ; Loop until X wraps (256 bytes = 64 sprites * 4)
    BNE .loop
    RTL""",
            "category": "basic_operations",
        },
        {
            "instruction": "Explain what this instruction does: LDA.w $7E0E20",
            "input": "Context: ALTTP player state management",
            "output": "This instruction loads the Accumulator (A) with a 16-bit word from RAM address $7E:0E20, which stores Link's current state in ALTTP. The .w suffix indicates a word (16-bit) operation. This address is commonly checked to determine Link's animation state (walking, attacking, damaged, etc.).",
            "category": "memory_access",
        },
        # Hardware Interaction
        {
            "instruction": "Write a routine to wait for VBlank (vertical blank period).",
            "input": "Poll the NMI flag at $4210. Clear it after detecting.",
            "output": """WaitForVBlank:
    LDA.w $4210          ; Read NMI flag (PPU register)
    AND.b #$80           ; Check bit 7 (NMI occurred)
    BEQ WaitForVBlank    ; If not set, keep waiting
    ; NMI flag auto-clears on read
    RTL""",
            "category": "hardware_interaction",
        },
        # Control Flow
        {
            "instruction": "Convert this C code to 65816 assembly: if (playerHealth < 3) { playLowHealthBeep(); }",
            "input": "playerHealth is at $7E:0F6C (8-bit). playLowHealthBeep is at $00:8B4A.",
            "output": """    LDA.w $0F6C          ; Load player health
    CMP.b #$03           ; Compare with 3
    BCS .skip            ; Branch if >= 3 (carry set)
    JSL $008B4A          ; Call playLowHealthBeep
.skip:
    ; Continue execution""",
            "category": "control_flow",
        },
    ]


def create_rom_hack_benchmark() -> list[dict]:
    """Create ROM hacking benchmark test set."""
    return [
        # Hook Implementation
        {
            "instruction": "How do you hook into the sprite loading routine at $0D:B4E0 to add custom sprites?",
            "input": "You want to redirect to your custom loader in bank $32.",
            "output": """To hook the sprite loading routine:

1. **Modify original routine** ($0D:B4E0):
```asm
org $0DB4E0
    JSL CustomSpriteLoader  ; Jump to bank $32
    NOP
    NOP                     ; Fill remaining bytes
```

2. **Implement custom loader** (bank $32):
```asm
org $328000
CustomSpriteLoader:
    ; Your custom sprite loading logic
    ; Check sprite ID, load from expanded banks, etc.

    ; Call original logic if needed
    ; Or RTL to return to caller

    RTL
```

3. **Update ROM header** to expand ROM size if needed:
```asm
org $00FFD7
    db $0C  ; ROM size: 4MB (if using banks $20-$3F)
```

This technique allows you to intercept vanilla sprite loads and add custom graphics from expanded ROM space.""",
            "category": "hook_implementation",
        },
        # Bank Allocation
        {
            "instruction": "You have 50KB of custom code. Which ROM banks should you use and why?",
            "input": "This is for Oracle of Secrets ROM hack on ALTTP.",
            "output": """For 50KB of custom code, use banks $20-$23:

**Why these banks:**
- Banks $00-$1F: Original ALTTP code/data (don't touch)
- Banks $20-$3F: Expanded ROM space (safe for hacks)
- Banks $40+: Available but further from original code

**Allocation strategy:**
```
Bank $20: Main custom code (16KB)
Bank $21: Additional routines (16KB)
Bank $22: Data tables and graphics (16KB)
Bank $23: Remaining code (~2KB needed)
```

**Access pattern:**
- Use `JSL $20XXXX` to call from bank $00-$1F
- Use `JSR` for within-bank calls (faster)
- Keep frequently-called code in bank $20 (fewer bank switches)

**ROM size:** Set to 2MB minimum ($0B) or 4MB ($0C) in header at $00FFD7.""",
            "category": "bank_allocation",
        },
    ]


def create_code_understanding_benchmark() -> list[dict]:
    """Create code understanding benchmark."""
    return [
        {
            "instruction": "What's wrong with this code and how do you fix it?",
            "input": """```asm
LoadData:
    LDA #$1234      ; Trying to load 16-bit value
    STA $0000       ; Store to direct page
    RTS
```""",
            "output": """**Problem:** Using 8-bit immediate mode (`#$1234`) will only load `$34`, truncating the high byte.

**Fix:**
```asm
LoadData:
    REP #$20        ; Set A to 16-bit mode
    LDA.w #$1234    ; Load 16-bit value (note .w suffix)
    STA.w $0000     ; Store 16-bit value
    SEP #$20        ; Return A to 8-bit mode (if needed)
    RTS
```

**Key points:**
- Must set processor flags with `REP #$20` for 16-bit accumulator
- Use `.w` suffix to clarify intent
- Restore 8-bit mode with `SEP #$20` if needed
- Direct page address `$0000` is fine, but `.w` makes it explicit""",
            "category": "debugging",
        },
        {
            "instruction": "Explain what this ALTTP routine does:",
            "input": """```asm
Module_MainRouting:
    LDA.w $0100
    ASL A
    TAX
    JSR (.ptr,X)
    RTL
.ptr:
    dw Module_Intro
    dw Module_FileSelect
    dw Module_GameLoop
```""",
            "output": """This is ALTTP's main module dispatcher - the core game loop router.

**Step-by-step:**
1. `LDA.w $0100` - Load current module ID (0=Intro, 1=FileSelect, 2=Game, etc.)
2. `ASL A` - Multiply by 2 (because pointers are 2 bytes)
3. `TAX` - Transfer to X register (index into table)
4. `JSR (.ptr,X)` - Indirect indexed jump - calls function from pointer table
5. `RTL` - Return from long call (back to main loop)

**Pointer table (`.ptr`):**
- Offset 0: Module_Intro (title screen)
- Offset 2: Module_FileSelect (save file selection)
- Offset 4: Module_GameLoop (actual gameplay)

This pattern is used throughout ALTTP for state machines. To add a custom module, you'd expand this table and increment the module ID at $0100.""",
            "category": "code_explanation",
        },
    ]


def main():
    """Create and save all benchmark test sets."""
    benchmark_dir = Path.home() / ".context" / "training" / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # Create benchmarks
    benchmarks = {
        "asm": create_asm_benchmark(),
        "rom_hack": create_rom_hack_benchmark(),
        "code_understanding": create_code_understanding_benchmark(),
    }

    # Save to JSONL
    for name, samples in benchmarks.items():
        output_file = benchmark_dir / f"{name}_benchmark.jsonl"
        with open(output_file, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        logger.info(f"✓ Created {name} benchmark: {output_file} ({len(samples)} samples)")

    logger.info(f"\n✓ All benchmarks created in {benchmark_dir}")
    logger.info("\nTo run benchmarks:")
    logger.info("  python -m agents.training.eval.benchmark --model <model_path> --benchmark asm")


if __name__ == "__main__":
    main()
