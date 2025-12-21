"""ALTTP Module and Control Flow Analyzer.

Specialized agent for analyzing game modules, sub-modules, and the main routing loop.
Provides deep insight into how the game state machine operates.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agents.core import BaseAgent
from agents.knowledge.alttp import ALTTPKnowledgeBase
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)


class ALTTPModuleAnalyzer(BaseAgent):
    """Expert in ALTTP's Module_MainRouting and sub-module systems.

    Analyzes:
    - Main module loop (bank 00)
    - Sub-module jump tables
    - Module transitions and interrupts
    - Static vs dynamic module behavior

    Example:
        analyzer = ALTTPModuleAnalyzer()
        await analyzer.setup()

        # Analyze current game state
        state = await analyzer.identify_state(module_id=0x07, submodule_id=0x01)

        # Trace module path
        trace = await analyzer.trace_transition(source=0x09, target=0x07)
    """

    def __init__(self, kb: Optional[ALTTPKnowledgeBase] = None):
        super().__init__(
            "ALTTPModuleAnalyzer",
            "Expert in ALTTP game state machinery and module control flow."
        )

        self._kb = kb
        self._orchestrator = None

    async def setup(self):
        """Initialize the analyzer."""
        await super().setup()

        self._orchestrator = UnifiedOrchestrator()
        await self._orchestrator.initialize()

        if not self._kb:
            self._kb = ALTTPKnowledgeBase()
            await self._kb.setup()

        logger.info("ALTTPModuleAnalyzer ready")

    async def identify_state(self, module_id: int, submodule_id: int) -> dict[str, Any]:
        """Identify exactly what the game is doing for a given state pair.

        Args:
            module_id: Main module index ($10).
            submodule_id: Sub-module index ($11).

        Returns:
            Description and analysis of the state.
        """
        # Get module info from KB
        module = self._kb._modules.get(module_id)

        prompt = f"""Explain what ALTTP is doing in this state:
        
MODULE: {module.name if module else hex(module_id)} ({hex(module_id)})
SUB-MODULE: {hex(submodule_id)}

Context: This state is part of the Module_MainRouting loop in Bank 00.

Provide:
1. What phase of the game this represents
2. Key routines executing during this state
3. What inputs or events cause it to transition
4. Common modifications or hooks for this module"""

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.FAST,
                provider=Provider.GEMINI,
            )
            return {
                "module": module.name if module else hex(module_id),
                "module_id": module_id,
                "submodule_id": submodule_id,
                "analysis": result.content,
            }
        except Exception as e:
            return {"error": f"State identification failed: {e}"}

    async def analyze_main_loop(self) -> str:
        """Analyze the core Module_MainRouting routine."""
        routine = self._kb._routines.get("Module_MainRouting")
        if not routine:
            return "Module_MainRouting not found in knowledge base"

        prompt = f"""Analyze the main game loop routing routine for ALTTP:

CODE:
{routine.code}

Explain:
1. How it utilizes addresses $7E0010 and $7E0011
2. How the jump table works
3. How it handles bank switching for module routines
4. Any special handling for NMI or IRQ interrupts"""

        try:
            result = await self._orchestrator.generate(
                prompt=prompt,
                tier=TaskTier.REASONING,
                provider=Provider.GEMINI,
            )
            return result.content
        except Exception as e:
            return f"Main loop analysis failed: {e}"

    async def run_task(self, task: str) -> dict[str, Any]:
        """Run an analysis task."""
        if task.startswith("state:"):
            # Format: state:MOD_ID,SUBMOD_ID
            try:
                parts = task[6:].split(",")
                mod = int(parts[0], 16)
                sub = int(parts[1], 16) if len(parts) > 1 else 0
                return await self.identify_state(mod, sub)
            except:
                return {"error": "Invalid state format. Use: state:07,01"}
        elif task == "main_loop":
            return {"analysis": await self.analyze_main_loop()}

        return {"error": "Unknown task"}
