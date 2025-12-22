"""ALTTP Module and Control Flow Analyzer.

Specialized agent for analyzing game modules, sub-modules, and the main routing loop.
Provides deep insight into how the game state machine operates.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.core import BaseAgent
from agents.knowledge.alttp import ALTTPKnowledgeBase
from core.orchestrator_v2 import UnifiedOrchestrator, TaskTier, Provider

logger = logging.getLogger(__name__)
REPORTS_ROOT = Path.home() / ".context" / "reports"


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

    async def analyze_module(self, module_id: int) -> dict[str, Any]:
        """Generate a focused report for a specific module."""
        if not self._kb:
            await self.setup()
        if not self._kb:
            return {"error": "Knowledge base not available"}

        module = self._kb._modules.get(module_id)
        if not module:
            return {"error": f"Module {module_id:02X} not found"}

        routine_snippets = []
        for routine_name in module.routines[:12]:
            routine = self._kb._routines.get(routine_name)
            if not routine or not routine.code:
                continue
            routine_snippets.append(
                f"{routine.name}:\n{routine.code[:800]}"
            )

        routines_list = "\n".join(module.routines[:20])
        snippets = "\n\n".join(routine_snippets)

        prompt = f"""Analyze this ALTTP module for ROM hacking insight.

Module {module_id:02X}: {module.name}
Description: {module.description}

Known routines:
{routines_list}

Routine snippets:
{snippets}

Provide:
1. What gameplay state this module represents
2. Key routines and why they matter
3. Likely transitions in/out of this module
4. Hook points and risks for modding
5. Suggested follow-up routines to inspect
"""

        report_text = ""
        if self._orchestrator:
            try:
                result = await self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.RESEARCH,
                )
                report_text = result.content or ""
            except Exception as e:
                report_text = f"Module analysis failed: {e}"
        else:
            report_text = "Module analysis unavailable: orchestrator not initialized."

        reports_dir = REPORTS_ROOT / "alttp" / "modules"
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe_name = module.name.replace(" ", "_").replace("/", "_")[:40]
        filename = f"module_{module_id:02X}_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        path = reports_dir / filename
        report = (
            f"# ALTTP Module Analysis: {module.name} ({module_id:02X})\n\n"
            f"Generated: {datetime.now().isoformat()}\n\n"
            f"{report_text}\n"
        )
        path.write_text(report)

        logger.info("Module report saved: %s", path)
        return {
            "module_id": module_id,
            "module_name": module.name,
            "report_path": str(path),
            "report": report_text,
        }

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
        elif task.startswith("module:"):
            try:
                mod = int(task.split(":", 1)[1], 16)
                return await self.analyze_module(mod)
            except Exception:
                return {"error": "Invalid module format. Use: module:07"}
        elif task == "main_loop":
            return {"analysis": await self.analyze_main_loop()}

        return {"error": "Unknown task"}
