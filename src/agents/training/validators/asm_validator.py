"""ASM Validator for 65816 assembly training samples.

Validates:
- Instruction mnemonics
- Addressing modes
- Register usage
- Memory addressing patterns
- SNES-specific constructs
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agents.training.base import TrainingSample
from agents.training.validators.base import ValidationResult, Validator


@dataclass
class InstructionInfo:
    """Information about a 65816 instruction."""

    mnemonic: str
    addressing_modes: list[str]
    description: str


class AsmValidator(Validator):
    """Validator for 65816 assembly code in training samples."""

    # Valid 65816 instruction mnemonics
    VALID_MNEMONICS = {
        # Load/Store
        "LDA", "LDX", "LDY", "STA", "STX", "STY", "STZ",
        # Transfer
        "TAX", "TAY", "TXA", "TYA", "TXS", "TSX", "TCD", "TDC", "TCS", "TSC", "TXY", "TYX",
        # Stack
        "PHA", "PHP", "PHX", "PHY", "PHB", "PHD", "PHK",
        "PLA", "PLP", "PLX", "PLY", "PLB", "PLD",
        "PEA", "PEI", "PER",
        # Arithmetic
        "ADC", "SBC", "INC", "INX", "INY", "DEC", "DEX", "DEY",
        # Comparison
        "CMP", "CPX", "CPY",
        # Logical
        "AND", "ORA", "EOR", "BIT",
        # Shift/Rotate
        "ASL", "LSR", "ROL", "ROR",
        # Branch
        "BCC", "BCS", "BEQ", "BMI", "BNE", "BPL", "BVC", "BVS", "BRA", "BRL",
        # Jump
        "JMP", "JML", "JSR", "JSL", "RTS", "RTL", "RTI",
        # Flags
        "CLC", "CLD", "CLI", "CLV", "SEC", "SED", "SEI",
        "REP", "SEP",
        # Processor
        "NOP", "WDM", "STP", "WAI", "XBA", "XCE",
        # Block Move
        "MVP", "MVN",
        # Misc
        "BRK", "COP", "WDM",
        # 65C816 specific
        "TRB", "TSB",
    }

    # Valid addressing mode patterns
    ADDRESSING_PATTERNS = {
        "immediate_8": r"#\$[0-9A-Fa-f]{1,2}",  # #$XX
        "immediate_16": r"#\$[0-9A-Fa-f]{3,4}",  # #$XXXX
        "immediate_symbol": r"#[A-Za-z_]\w*",  # #SYMBOL
        "direct_page": r"\$[0-9A-Fa-f]{1,2}(?!\w)",  # $XX (not followed by more hex)
        "absolute": r"\$[0-9A-Fa-f]{4}(?!\w)",  # $XXXX
        "long": r"\$[0-9A-Fa-f]{6}",  # $XXXXXX
        "indexed_x": r",\s*[Xx]",  # ,X
        "indexed_y": r",\s*[Yy]",  # ,Y
        "indirect": r"\([^)]+\)",  # (...)
        "stack_relative": r"\$[0-9A-Fa-f]{1,2},\s*[Ss]",  # $XX,S
        "accumulator": r"[Aa](?:\s|$)",  # A
        "label": r"[A-Za-z_]\w*",  # Labels
    }

    # SNES-specific registers and addresses
    SNES_REGISTERS = {
        # PPU Registers
        "INIDISP", "OBSEL", "OAMADDL", "OAMADDH", "OAMDATA",
        "BGMODE", "MOSAIC", "BG1SC", "BG2SC", "BG3SC", "BG4SC",
        "BG12NBA", "BG34NBA", "BG1HOFS", "BG1VOFS", "BG2HOFS", "BG2VOFS",
        "BG3HOFS", "BG3VOFS", "BG4HOFS", "BG4VOFS",
        "VMAIN", "VMADDL", "VMADDH", "VMDATAL", "VMDATAH",
        "M7SEL", "M7A", "M7B", "M7C", "M7D", "M7X", "M7Y",
        "CGADD", "CGDATA", "W12SEL", "W34SEL", "WOBJSEL",
        "WH0", "WH1", "WH2", "WH3", "WBGLOG", "WOBJLOG",
        "TM", "TS", "TMW", "TSW", "CGWSEL", "CGADSUB",
        "COLDATA", "SETINI",
        # APU Registers
        "APUIO0", "APUIO1", "APUIO2", "APUIO3",
        # DMA Registers
        "MDMAEN", "HDMAEN", "MEMSEL",
        # CPU Registers
        "NMITIMEN", "WRIO", "WRMPYA", "WRMPYB", "WRDIVL", "WRDIVH",
        "WRDIVB", "HTIMEL", "HTIMEH", "VTIMEL", "VTIMEH",
        "RDNMI", "TIMEUP", "HVBJOY", "RDIO", "RDDIVL", "RDDIVH",
        "RDMPYL", "RDMPYH", "JOY1L", "JOY1H", "JOY2L", "JOY2H",
        "JOY3L", "JOY3H", "JOY4L", "JOY4H",
    }

    # Common ALTTP-specific labels
    ALTTP_LABELS = {
        "Module", "Submodule", "Link", "Player", "Sprite",
        "WRAM", "SRAM", "VRAM", "OAM", "CGRAM",
    }

    VALID_DOMAINS = {"asm", "hack_curated"}

    def __init__(self, strict: bool = False):
        """Initialize ASM validator.

        Args:
            strict: If True, apply stricter validation rules
        """
        super().__init__("AsmValidator", "asm")
        self.strict = strict

    def can_validate(self, sample: TrainingSample) -> bool:
        """Allow ASM validation for curated hack samples too."""
        return sample.domain in self.VALID_DOMAINS

    async def validate(self, sample: TrainingSample) -> ValidationResult:
        """Validate 65816 assembly in the sample output."""
        errors: list[str] = []
        warnings: list[str] = []
        details: dict = {
            "instructions_found": 0,
            "valid_instructions": 0,
            "invalid_instructions": [],
            "snes_registers_used": [],
            "addressing_modes": [],
        }

        # Extract code from output
        code = sample.output

        # Parse instructions
        instructions = self._extract_instructions(code)
        details["instructions_found"] = len(instructions)

        if len(instructions) == 0:
            warnings.append("No assembly instructions found in output")
            return ValidationResult(
                valid=True,
                score=0.5,
                warnings=warnings,
                details=details,
            )

        # Validate each instruction
        for line_num, instr in instructions:
            result = self._validate_instruction(instr)
            if result.valid:
                details["valid_instructions"] += 1
                if result.addressing_mode:
                    details["addressing_modes"].append(result.addressing_mode)
            else:
                details["invalid_instructions"].append({
                    "line": line_num,
                    "instruction": instr,
                    "error": result.error,
                })
                if self.strict:
                    errors.append(f"Line {line_num}: {result.error}")
                else:
                    warnings.append(f"Line {line_num}: {result.error}")

        # Check for SNES registers
        for reg in self.SNES_REGISTERS:
            if reg in code:
                details["snes_registers_used"].append(reg)

        # Calculate score
        if details["instructions_found"] > 0:
            score = details["valid_instructions"] / details["instructions_found"]
        else:
            score = 0.5

        # Boost score if SNES-specific content found
        if details["snes_registers_used"]:
            score = min(1.0, score + 0.1)

        return ValidationResult(
            valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details,
        )

    def _extract_instructions(self, code: str) -> list[tuple[int, str]]:
        """Extract assembly instructions from code.

        Returns:
            List of (line_number, instruction) tuples
        """
        instructions = []
        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Remove comments
            if ";" in line:
                line = line[:line.index(";")]

            # Remove labels (lines ending with :)
            if ":" in line:
                # Check if it's a label definition
                parts = line.split(":")
                if len(parts) > 1:
                    line = parts[-1]

            # Remove address prefixes like #_008000:
            line = re.sub(r"#_[0-9A-Fa-f]+:\s*", "", line)

            line = line.strip()

            if not line:
                continue

            # Check if line starts with a valid mnemonic
            parts = line.split()
            if parts:
                mnemonic = parts[0].upper()
                if mnemonic in self.VALID_MNEMONICS:
                    instructions.append((i, line))
                elif re.match(r"[A-Za-z]{2,4}", mnemonic):
                    # Might be an instruction-like thing
                    instructions.append((i, line))

        return instructions

    def _validate_instruction(self, instruction: str) -> "_InstructionValidation":
        """Validate a single instruction."""
        parts = instruction.split(None, 1)
        if not parts:
            return _InstructionValidation(False, "Empty instruction")

        mnemonic = parts[0].upper()
        operand = parts[1] if len(parts) > 1 else ""

        # Check mnemonic
        if mnemonic not in self.VALID_MNEMONICS:
            # Check if it's close to a valid mnemonic (typo detection)
            close_matches = [m for m in self.VALID_MNEMONICS
                           if self._levenshtein_distance(mnemonic, m) <= 1]
            if close_matches:
                return _InstructionValidation(
                    False,
                    f"Unknown mnemonic '{mnemonic}' (did you mean {close_matches[0]}?)"
                )
            return _InstructionValidation(False, f"Unknown mnemonic '{mnemonic}'")

        # Validate operand if present
        addressing_mode = None
        if operand:
            addressing_mode = self._detect_addressing_mode(operand)

        return _InstructionValidation(True, None, addressing_mode)

    def _detect_addressing_mode(self, operand: str) -> Optional[str]:
        """Detect the addressing mode from the operand."""
        operand = operand.strip()

        # Check patterns in order of specificity
        if re.match(r"#", operand):
            if re.search(r"#\$[0-9A-Fa-f]{3,4}", operand):
                return "immediate_16"
            elif re.search(r"#\$[0-9A-Fa-f]{1,2}", operand):
                return "immediate_8"
            else:
                return "immediate_symbol"

        if re.search(r",\s*[Ss]", operand):
            return "stack_relative"

        if re.match(r"\([^)]+\)", operand):
            if ",X" in operand.upper():
                return "indexed_indirect_x"
            elif ",Y" in operand.upper():
                return "indirect_indexed_y"
            else:
                return "indirect"

        if re.search(r",\s*[Xx]", operand):
            return "indexed_x"

        if re.search(r",\s*[Yy]", operand):
            return "indexed_y"

        if re.match(r"\$[0-9A-Fa-f]{6}", operand):
            return "long"

        if re.match(r"\$[0-9A-Fa-f]{4}", operand):
            return "absolute"

        if re.match(r"\$[0-9A-Fa-f]{1,2}(?!\w)", operand):
            return "direct_page"

        if re.match(r"[Aa]$", operand):
            return "accumulator"

        if re.match(r"[A-Za-z_]\w*", operand):
            return "label"

        return None

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


@dataclass
class _InstructionValidation:
    """Internal result of validating a single instruction."""

    valid: bool
    error: Optional[str] = None
    addressing_mode: Optional[str] = None
