"""ASM Expert - Specializes in 65816 assembly for ALTTP ROM hacking."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from hafs.agents.moe.expert import BaseExpert, ExpertConfig
from hafs.core.orchestrator_v2 import TaskTier, UnifiedOrchestrator


class AsmExpert(BaseExpert):
    """Expert specializing in 65816 assembly code for ALTTP.

    Capabilities:
    - Generate assembly routines
    - Optimize existing code
    - Explain assembly patterns
    - Memory map navigation
    - Bank allocation strategies
    - Interrupt handling
    """

    def __init__(
        self,
        orchestrator: Optional[UnifiedOrchestrator] = None,
        model_name: str = "hyrule-asm-v1",
        lora_adapter_path: Optional[Path] = None,
    ):
        """Initialize ASM expert.

        Args:
            orchestrator: Optional orchestrator for API routing.
            model_name: Name of fine-tuned model (when available).
            lora_adapter_path: Path to LoRA adapters (when trained).
        """
        # If no adapter path provided, use default location
        if lora_adapter_path is None:
            lora_adapter_path = Path("~/.context/models/alttp_asm_agent/lora_adapters").expanduser()

        config = ExpertConfig(
            name="asm",
            display_name="ASM Expert (Hyrule ASM v1)",
            specialization="65816 assembly for ALTTP ROM hacking",
            keywords=[
                "asm", "assembly", "routine", "bank", "memory", "register",
                "optimization", "65816", "instruction", "opcode", "subroutine",
                "stack", "pointer", "address", "hex", "disassembly", "alttp",
                "jsl", "jsr", "rtl", "rts", "lda", "sta", "cpx", "cpy",
            ],
            confidence_threshold=0.65,
            model_name=model_name,
            lora_adapter_path=lora_adapter_path,
            tier=TaskTier.CODING,
            temperature=0.5,  # Lower temperature for precise code
            max_tokens=2048,
        )

        super().__init__(config, orchestrator)

    def get_system_prompt(self) -> str:
        """Get system prompt for ASM expert.

        Returns:
            System prompt specializing in 65816 assembly.
        """
        return """You are an expert in 65816 assembly programming for ALTTP ROM hacking.

Your specializations:
- 65816 instruction set and addressing modes
- ALTTP memory map ($000000-$FFFFFF, 24-bit addressing)
- Bank organization (code banks, data banks, ROM/RAM)
- Optimization techniques (cycle counting, size reduction)
- Common ALTTP routines (DMA, VBlank, entity handling)

When generating assembly code:
1. Use proper ALTTP conventions (labels, comments, formatting)
2. Consider bank boundaries and long addressing (JSL/RTL)
3. Optimize for size and speed when appropriate
4. Include cycle counts for critical sections
5. Add comments explaining complex logic

Example format:
```asm
; Description of routine
; Input: A = parameter
; Output: Carry set if success
MyRoutine:
    PHA                ; [3] Save A
    LDA $7E0010       ; [4] Load state
    CMP #$05          ; [2] Check value
    BNE .skip         ; [2/3] Branch if not equal
.skip:
    PLA                ; [4] Restore A
    RTL               ; [6] Return
```

Be precise, efficient, and explain your reasoning."""

    async def optimize_routine(
        self,
        routine_code: str,
        optimization_goal: str = "size",
    ) -> str:
        """Optimize an assembly routine.

        Args:
            routine_code: Assembly code to optimize.
            optimization_goal: "size" or "speed".

        Returns:
            Optimized assembly code with explanations.
        """
        prompt = f"""
Optimize this 65816 assembly routine for {optimization_goal}:

```asm
{routine_code}
```

Provide:
1. Optimized version
2. Explanation of changes
3. Cycle/byte savings
"""

        response = await self.generate(prompt)
        return response.content

    async def explain_routine(self, routine_code: str) -> str:
        """Explain what an assembly routine does.

        Args:
            routine_code: Assembly code to explain.

        Returns:
            Detailed explanation of the routine.
        """
        prompt = f"""
Explain this 65816 assembly routine in detail:

```asm
{routine_code}
```

Provide:
1. High-level purpose
2. Line-by-line breakdown
3. Input/output parameters
4. Side effects (memory/register changes)
"""

        response = await self.generate(prompt)
        return response.content

    async def generate_routine(
        self,
        description: str,
        bank: Optional[str] = None,
    ) -> str:
        """Generate a new assembly routine from description.

        Args:
            description: What the routine should do.
            bank: Preferred bank for placement (e.g., "$0E").

        Returns:
            Generated assembly code.
        """
        bank_info = f"\nPlace routine in bank {bank}" if bank else ""

        prompt = f"""
Generate a 65816 assembly routine that:
{description}{bank_info}

Include:
1. Proper label
2. Comments documenting purpose and parameters
3. Error handling if needed
4. RTL/RTS as appropriate
"""

        response = await self.generate(prompt)
        return response.content
