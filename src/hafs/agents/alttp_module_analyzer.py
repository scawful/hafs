"""ALTTP Module Analyzer - Specialized pipeline for game module analysis.

Generates comprehensive reports on A Link to the Past game modules,
analyzing state machines, routines, memory usage, and module transitions.

Usage:
    analyzer = ALTTPModuleAnalyzer()
    await analyzer.setup()
    result = await analyzer.analyze_module(0x07)  # Underworld
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from hafs.agents.base import BaseAgent
from hafs.agents.context_report_pipeline import (
    ContextReportPipeline,
    EmbeddingResearchAgent,
    ResearchContext,
)

logger = logging.getLogger(__name__)

REPORTS_ROOT = Path.home() / ".context" / "reports"


@dataclass
class ModuleAnalysisContext(ResearchContext):
    """Extended context for module analysis."""

    module_id: int = 0
    module_name: str = ""
    module_routines: List[str] = field(default_factory=list)
    state_machine_analysis: str = ""
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    submodule_analysis: Dict[str, Any] = field(default_factory=dict)


class ModuleResearchAgent(EmbeddingResearchAgent):
    """Specialized researcher for game modules."""

    async def analyze_module_routines(self, module_name: str) -> Dict[str, Any]:
        """Analyze routines belonging to a specific module."""

        # Search for module-related routines
        module_routines = await self.search(f"Module {module_name}", limit=30)

        # Get submodule routines
        submodule_results = await self.search(f"Submodule {module_name}", limit=20)

        # Search for state handler patterns
        state_results = await self.search(f"{module_name} State", limit=15)

        # Analyze state machine patterns
        state_patterns = []
        for routine in module_routines + state_results:
            name = routine.get("name", "")
            if any(p in name for p in ["State", "SubModule", "Handler", "Mode"]):
                state_patterns.append(routine)

        return {
            "main_routines": module_routines,
            "submodule_routines": submodule_results,
            "state_patterns": state_patterns,
            "state_results": state_results,
        }


class StateFlowAnalyzer(BaseAgent):
    """Analyzes game state machine flows."""

    def __init__(self):
        super().__init__("StateFlowAnalyzer", "Analyze game state machines and module transitions.")
        self.model_tier = "reasoning"

    async def analyze_state_machine(
        self,
        module_name: str,
        routines: List[Dict[str, Any]]
    ) -> str:
        """Generate state machine analysis."""

        routine_list = json.dumps(routines[:20], indent=2, default=str)

        prompt = f"""Analyze the state machine for ALTTP game module: {module_name}

ROUTINES FOUND:
{routine_list}

Based on ALTTP's architecture (where modules control game flow via Module_MainRouting):
1. Identify the main module handler routine
2. Map submodule states and their purposes
3. Identify state transition patterns
4. Document the game flow through this module
5. Note any important WRAM addresses used for state tracking
6. Describe interactions with other modules

Provide a detailed state machine analysis with:
- Entry points and exit conditions
- Submodule state transitions
- Key memory variables
- Frame-by-frame update patterns"""

        return await self.generate_thought(prompt)


class MemoryAnalyzer(BaseAgent):
    """Analyzes module memory usage patterns."""

    def __init__(self):
        super().__init__("MemoryAnalyzer", "Analyze WRAM usage patterns for game modules.")
        self.model_tier = "fast"

    async def analyze_memory_usage(
        self,
        module_name: str,
        symbols: Dict[str, Any],
        routines: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze memory usage for a module."""

        # Extract memory references from routines
        memory_refs = []
        for name, routine in routines.items():
            if hasattr(routine, 'memory_access'):
                memory_refs.extend(routine.memory_access)

        # Categorize by memory region
        regions = {
            "direct_page": [],  # $00-$FF
            "wram_low": [],     # $0100-$1FFF
            "wram_high": [],    # $7E0000-$7FFFFF
            "registers": [],    # $2100-$21FF, $4200-$43FF
        }

        for ref in memory_refs:
            addr = ref.get("address", 0) if isinstance(ref, dict) else 0
            if addr < 0x100:
                regions["direct_page"].append(ref)
            elif addr < 0x2000:
                regions["wram_low"].append(ref)
            elif 0x7E0000 <= addr <= 0x7FFFFF:
                regions["wram_high"].append(ref)
            elif 0x2100 <= addr <= 0x21FF or 0x4200 <= addr <= 0x43FF:
                regions["registers"].append(ref)

        return {
            "total_references": len(memory_refs),
            "regions": {k: len(v) for k, v in regions.items()},
            "key_variables": list(symbols.keys())[:20],
        }


class ALTTPModuleAnalyzer(ContextReportPipeline):
    """Specialized pipeline for ALTTP game module analysis.

    Provides detailed analysis of ALTTP's game modules including:
    - State machine flow
    - Submodule organization
    - Memory usage patterns
    - Module transitions
    """

    # Known ALTTP modules (from Module_MainRouting)
    KNOWN_MODULES = {
        0x00: "TitleScreen",
        0x01: "FileSelect",
        0x02: "CopyErase",
        0x03: "PlayerName",
        0x04: "LoadingFile",
        0x05: "LoadingDungeon",
        0x06: "PreUnderworld",
        0x07: "Underworld",
        0x08: "PreOverworld",
        0x09: "Overworld",
        0x0A: "PreOverworld_Special",
        0x0B: "OverworldSpecial",
        0x0C: "UnknownModule_0C",
        0x0D: "SubscreenOverlay",
        0x0E: "ViewingMap",
        0x0F: "TextMode",
        0x10: "ClosingSpotlight",
        0x11: "OpeningSpotlight",
        0x12: "PreDungeon",
        0x13: "PostDungeon",
        0x14: "SaveQuit",
        0x15: "MirrorTransition",
        0x16: "IntroSequence",
        0x17: "Attract",
        0x18: "Credits",
        0x19: "GameOver",
        0x1A: "TriforceRoom",
        0x1B: "Unknown1B",
    }

    def __init__(self):
        super().__init__(project="alttp")
        self.name = "ALTTPModuleAnalyzer"
        self._state_analyzer: Optional[StateFlowAnalyzer] = None
        self._memory_analyzer: Optional[MemoryAnalyzer] = None

    async def setup(self):
        await super().setup()

        # Replace researchers with module-specific ones
        if self._kb:
            self._researchers = [
                ModuleResearchAgent(self._kb, f"_mod_{i}")
                for i in range(3)
            ]
            for r in self._researchers:
                await r.setup()

        # Add specialized analyzers
        self._state_analyzer = StateFlowAnalyzer()
        await self._state_analyzer.setup()

        self._memory_analyzer = MemoryAnalyzer()
        await self._memory_analyzer.setup()

        logger.info("ALTTPModuleAnalyzer initialized")

    async def analyze_module(self, module_id: int) -> Dict[str, Any]:
        """Analyze a specific game module by ID.

        Args:
            module_id: Module ID (0x00-0x1B)

        Returns:
            Analysis results including report path and state machine analysis
        """

        module_name = self.KNOWN_MODULES.get(module_id, f"Module{module_id:02X}")

        context = ModuleAnalysisContext(
            topic=f"Game Module: {module_name} (0x{module_id:02X})",
            project="alttp",
            module_id=module_id,
            module_name=module_name,
            research_queries=[
                f"Module{module_id:02X}",
                module_name,
                f"Module_{module_name}",
                f"MainRoute_{module_name}",
                f"Submodule_{module_name}",
            ]
        )

        # Research phase
        await self._step_research(context)

        # Module-specific research
        if self._researchers:
            module_researcher = self._researchers[0]
            if hasattr(module_researcher, 'analyze_module_routines'):
                module_data = await module_researcher.analyze_module_routines(module_name)
                context.gathered_context.update(module_data)
                context.submodule_analysis = module_data

        # State machine analysis
        state_patterns = context.gathered_context.get("state_patterns", [])
        if self._state_analyzer:
            context.state_machine_analysis = await self._state_analyzer.analyze_state_machine(
                module_name,
                state_patterns
            )

        # Memory analysis
        if self._memory_analyzer:
            context.memory_usage = await self._memory_analyzer.analyze_memory_usage(
                module_name,
                context.gathered_context.get("symbols", {}),
                context.gathered_context.get("routines", {})
            )

        # Standard analysis phase
        await self._step_analyze(context)
        context.analysis_results["state_machine"] = context.state_machine_analysis
        context.analysis_results["memory_usage"] = context.memory_usage

        # Review
        await self._step_review(context)

        # Generate specialized module report
        report = await self._generate_module_report(context)

        return {
            "module_id": module_id,
            "module_name": module_name,
            "report_path": str(context.report_path) if context.report_path else None,
            "report": report,
            "state_machine_analysis": context.state_machine_analysis,
            "memory_usage": context.memory_usage,
            "submodule_count": len(context.gathered_context.get("submodule_routines", [])),
        }

    async def _generate_module_report(self, context: ModuleAnalysisContext) -> str:
        """Generate specialized module report."""

        prompt = f"""Generate a comprehensive ALTTP game module report.

MODULE: {context.module_name} (0x{context.module_id:02X})

STATE MACHINE ANALYSIS:
{context.state_machine_analysis}

MEMORY USAGE:
{json.dumps(context.memory_usage, indent=2, default=str)}

GATHERED DATA:
- Main routines: {len(context.gathered_context.get('main_routines', []))}
- Submodule routines: {len(context.gathered_context.get('submodule_routines', []))}
- State patterns: {len(context.gathered_context.get('state_patterns', []))}

ANALYSIS:
{json.dumps(context.analysis_results, indent=2, default=str)[:4000]}

Generate a detailed technical report covering:
1. **Module Overview**: What this module handles in the game
2. **State Machine**: States, transitions, and submodules
3. **Key Routines**: Most important functions and their purposes
4. **Memory Usage**: Important WRAM addresses used
5. **Related Modules**: Transitions to/from other modules
6. **ROM Hacking Notes**: How to modify this module's behavior

Format as clean Markdown with code blocks for addresses and routines."""

        report = await self.generate_thought(prompt)

        # Save to modules subdirectory
        reports_dir = REPORTS_ROOT / "alttp" / "modules"
        reports_dir.mkdir(parents=True, exist_ok=True)

        filename = f"Module{context.module_id:02X}_{context.module_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        path = reports_dir / filename
        path.write_text(report)
        context.report_path = path

        logger.info(f"Module report saved: {path}")
        return report

    async def analyze_all_modules(self, delay_seconds: float = 2.0) -> Dict[str, Any]:
        """Generate reports for all known modules.

        Args:
            delay_seconds: Delay between module analyses (rate limiting)

        Returns:
            Summary of all module analyses
        """
        results = {}

        for module_id, module_name in self.KNOWN_MODULES.items():
            logger.info(f"Analyzing module 0x{module_id:02X}: {module_name}")
            try:
                result = await self.analyze_module(module_id)
                results[f"0x{module_id:02X}"] = {
                    "name": module_name,
                    "report_path": result.get("report_path"),
                    "status": "success"
                }
            except Exception as e:
                logger.error(f"Failed to analyze module 0x{module_id:02X}: {e}")
                results[f"0x{module_id:02X}"] = {
                    "name": module_name,
                    "status": "error",
                    "error": str(e)
                }

            # Rate limiting
            await asyncio.sleep(delay_seconds)

        return {
            "total_modules": len(self.KNOWN_MODULES),
            "analyzed": sum(1 for r in results.values() if r["status"] == "success"),
            "failed": sum(1 for r in results.values() if r["status"] == "error"),
            "results": results
        }

    async def run_task(self, task: str = "help") -> Dict[str, Any]:
        """Run module analyzer task.

        Tasks:
            help - Show usage information
            list - List all known modules
            module:ID - Analyze module by hex ID (e.g., module:07)
            module:NAME - Analyze module by name (e.g., module:Underworld)
            all - Analyze all known modules
        """
        if task == "help":
            return {
                "usage": [
                    "list - List all known ALTTP modules",
                    "module:ID - Analyze module by hex ID (e.g., module:07)",
                    "module:NAME - Analyze module by name (e.g., module:Underworld)",
                    "all - Analyze all known modules (takes time)",
                ]
            }

        if task == "list":
            return {
                "modules": {
                    f"0x{mid:02X}": name
                    for mid, name in self.KNOWN_MODULES.items()
                }
            }

        if task == "all":
            return await self.analyze_all_modules()

        if task.startswith("module:"):
            identifier = task[7:].strip()

            # Try hex ID
            try:
                module_id = int(identifier, 16)
                if module_id in self.KNOWN_MODULES:
                    return await self.analyze_module(module_id)
                return {"error": f"Unknown module ID: 0x{module_id:02X}"}
            except ValueError:
                pass

            # Try name lookup
            for mid, mname in self.KNOWN_MODULES.items():
                if mname.lower() == identifier.lower():
                    return await self.analyze_module(mid)

            return {"error": f"Unknown module: {identifier}"}

        return await super().run_task(task)


# CLI entry point
async def main():
    """CLI entry point for module analysis."""
    import sys

    analyzer = ALTTPModuleAnalyzer()
    await analyzer.setup()

    if len(sys.argv) < 2:
        result = await analyzer.run_task("help")
    else:
        task = sys.argv[1]
        if len(sys.argv) > 2:
            task = f"{task}:{sys.argv[2]}"
        result = await analyzer.run_task(task)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
