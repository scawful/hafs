#!/usr/bin/env python3
"""
LLVM-based 65816 ASM parser for semantic analysis

Provides:
- Instruction parsing and validation
- Addressing mode detection
- Control flow analysis
- Symbol resolution
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Set


class AddressingMode(Enum):
    """65816 addressing modes."""
    IMMEDIATE = "immediate"
    ABSOLUTE = "absolute"
    ABSOLUTE_X = "absolute_x"
    ABSOLUTE_Y = "absolute_y"
    ABSOLUTE_LONG = "absolute_long"
    DIRECT = "direct"
    DIRECT_X = "direct_x"
    DIRECT_Y = "direct_y"
    INDIRECT = "indirect"
    INDIRECT_X = "indirect_x"
    INDIRECT_Y = "indirect_y"
    STACK_RELATIVE = "stack_relative"
    IMPLIED = "implied"


@dataclass
class Instruction:
    """Parsed 65816 instruction."""
    mnemonic: str
    mode: AddressingMode
    operand: Optional[str] = None
    size: int = 1  # Instruction size in bytes
    cycles: int = 0

    # Semantic info
    reads_memory: bool = False
    writes_memory: bool = False
    affects_flags: Set[str] = None
    is_branch: bool = False
    is_call: bool = False
    is_return: bool = False


class LLVMAsmParser:
    """LLVM-powered ASM parser for 65816."""

    # 65816 instruction set
    INSTRUCTIONS = {
        "LDA": {"reads": True, "flags": {"N", "Z"}},
        "LDX": {"reads": True, "flags": {"N", "Z"}},
        "LDY": {"reads": True, "flags": {"N", "Z"}},
        "STA": {"writes": True},
        "STX": {"writes": True},
        "STY": {"writes": True},
        "JMP": {"branch": True},
        "JSR": {"call": True},
        "JSL": {"call": True},
        "RTL": {"return": True},
        "RTS": {"return": True},
        # ... full instruction set
    }

    def __init__(self):
        self.symbols: dict[str, int] = {}

    def parse_instruction(self, line: str) -> Optional[Instruction]:
        """Parse a single ASM line."""

        # Remove comments
        line = line.split(';')[0].strip()

        if not line:
            return None

        # Parse mnemonic and operand
        parts = line.split(None, 1)
        if not parts:
            return None

        mnemonic = parts[0].upper()
        operand = parts[1] if len(parts) > 1 else None

        # Detect addressing mode
        mode = self._detect_addressing_mode(mnemonic, operand)

        # Get instruction info
        info = self.INSTRUCTIONS.get(mnemonic, {})

        return Instruction(
            mnemonic=mnemonic,
            mode=mode,
            operand=operand,
            reads_memory=info.get("reads", False),
            writes_memory=info.get("writes", False),
            affects_flags=info.get("flags", set()),
            is_branch=info.get("branch", False),
            is_call=info.get("call", False),
            is_return=info.get("return", False),
        )

    def _detect_addressing_mode(
        self,
        mnemonic: str,
        operand: Optional[str],
    ) -> AddressingMode:
        """Detect addressing mode from operand syntax."""

        if not operand:
            return AddressingMode.IMPLIED

        operand = operand.strip()

        # Immediate: #$xx
        if operand.startswith('#'):
            return AddressingMode.IMMEDIATE

        # Indirect: ($xx)
        if operand.startswith('(') and operand.endswith(')'):
            inner = operand[1:-1]
            if ',X' in inner:
                return AddressingMode.INDIRECT_X
            elif ',Y' in inner:
                return AddressingMode.INDIRECT_Y
            return AddressingMode.INDIRECT

        # Indexed
        if ',X' in operand:
            return AddressingMode.ABSOLUTE_X
        elif ',Y' in operand:
            return AddressingMode.ABSOLUTE_Y

        # Long addressing: $xxxxxx
        if operand.startswith('$') and len(operand) == 7:
            return AddressingMode.ABSOLUTE_LONG

        # Direct page: $xx (2 hex digits)
        if operand.startswith('$') and len(operand) == 3:
            return AddressingMode.DIRECT

        # Default to absolute
        return AddressingMode.ABSOLUTE

    def validate_instruction(self, instr: Instruction) -> List[str]:
        """Validate instruction and return warnings."""
        warnings = []

        # Check valid mnemonic
        if instr.mnemonic not in self.INSTRUCTIONS:
            warnings.append(f"Unknown instruction: {instr.mnemonic}")

        # Check addressing mode compatibility
        # TODO: Add mode validation per instruction

        return warnings

    def analyze_control_flow(self, code: str) -> dict[str, List[str]]:
        """Analyze control flow and find reachable blocks."""

        # Parse all instructions
        instructions = []
        for line in code.split('\n'):
            instr = self.parse_instruction(line)
            if instr:
                instructions.append(instr)

        # Build CFG
        cfg = {}
        current_block = []

        for instr in instructions:
            current_block.append(instr)

            # Block ends on branch/call/return
            if instr.is_branch or instr.is_call or instr.is_return:
                # TODO: Store block in CFG
                current_block = []

        return cfg


# TODO: Integrate with LLVM's MC layer for:
# - Disassembly
# - Encoding validation
# - Optimization passes
