"""Mesen2 Swarm Coordinator for Integration and Lua Scripting.

Orchestrates specialized agents to create Mesen2 integration with YAZE
and generate comprehensive Lua debugging script library.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.swarm.swarm.mesen2_specialists import (
    DebuggingToolsBuilderAgent,
    IntegrationArchitectAgent,
    LuaScriptGeneratorAgent,
    LuaScriptLibraryGenerator,
    TestAutomationAgent,
)
from agents.swarm.swarm.yaze_specialists import SwarmSynthesizer

logger = logging.getLogger(__name__)


class Mesen2IntegrationMission:
    """Mission 4: Mesen2 Integration & Lua Scripting.

    Objectives:
    1. Create Lua debugging scripts for Mesen2
    2. Design YAZE ↔ Mesen2 integration
    3. Add memory watch/breakpoint helpers
    4. Create ROM testing automation
    5. Evaluate fork vs plugin architecture
    """

    def __init__(
        self,
        yaze_path: Path,
        mesen2_path: Optional[Path] = None,
    ):
        """Initialize Mesen2 integration mission.

        Args:
            yaze_path: Path to YAZE repository
            mesen2_path: Path to Mesen2 repository (optional)
        """
        self.name = "Mesen2 Integration"
        self.description = "YAZE-Mesen2 integration and Lua script library"
        self.yaze_path = Path(yaze_path)
        self.mesen2_path = Path(mesen2_path) if mesen2_path else None
        self.duration_hours = 2.0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.status = "pending"
        self.results: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert mission to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "yaze_path": str(self.yaze_path),
            "mesen2_path": str(self.mesen2_path) if self.mesen2_path else None,
            "duration_hours": self.duration_hours,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class Mesen2SwarmCoordinator:
    """Coordinates Mesen2 integration and scripting missions."""

    def __init__(
        self,
        yaze_path: Optional[Path] = None,
        mesen2_path: Optional[Path] = None,
    ):
        """Initialize Mesen2 swarm coordinator.

        Args:
            yaze_path: Path to YAZE repository
            mesen2_path: Path to Mesen2 repository (optional)
        """
        self.yaze_path = yaze_path or Path.home() / "Code" / "yaze"
        self.mesen2_path = mesen2_path

        if not self.yaze_path.exists():
            raise ValueError(f"YAZE path not found: {self.yaze_path}")

        # Initialize specialist agents
        self.agents = {
            "lua_script_generator": LuaScriptGeneratorAgent(),
            "integration_architect": IntegrationArchitectAgent(),
            "debugging_tools_builder": DebuggingToolsBuilderAgent(),
            "test_automation": TestAutomationAgent(),
            "lua_library_generator": LuaScriptLibraryGenerator(),
            "synthesizer": SwarmSynthesizer(),
        }

        # Output directory
        self.output_dir = Path.home() / ".context" / "swarms" / "mesen2"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lua scripts output directory
        self.scripts_dir = self.output_dir / "lua_scripts"
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize all swarm agents."""
        for agent in self.agents.values():
            await agent.setup()

    async def launch_integration_mission(
        self,
        parallel: bool = True,
    ) -> Dict[str, Any]:
        """Launch Mesen2 integration mission.

        Args:
            parallel: Run agents in parallel if True

        Returns:
            Mission results with integration design and Lua scripts
        """
        mission = Mesen2IntegrationMission(self.yaze_path, self.mesen2_path)

        logger.info(f"Launching mission: {mission.name}")
        mission.status = "running"
        mission.start_time = datetime.now()

        # Phase 1: Generate Lua script library
        logger.info("Phase 1: Generating Lua script library...")
        script_categories = """
1. Memory Watch - Monitor custom items, flags, game state
2. Performance Profiling - Track CPU cycles and routine calls
3. Event Logging - Log item pickups, room transitions, boss fights
4. Input Recording - Record and playback input for TAS-style testing
5. Automated Testing - Validate ROM hack features
        """

        lua_library = await self.agents["lua_library_generator"].run_task(
            script_categories
        )

        # Save generated Lua scripts to disk
        await self._save_lua_scripts(lua_library)

        # Phase 2: Design integration architecture
        logger.info("Phase 2: Designing YAZE-Mesen2 integration...")
        integration_context = f"""
YAZE ROM editor: {self.yaze_path}
Mesen2 emulator: {self.mesen2_path or "Not specified"}

Requirements:
- Launch Mesen2 from YAZE with current ROM
- Sync memory view and disassembly between tools
- Share breakpoints and watch expressions
- Support automated testing workflows

Constraints:
- Must work on macOS (user's platform)
- Prefer minimal changes to Mesen2 (plugin > fork)
- Should integrate with existing YAZE C++ API
        """

        integration_design = await self.agents["integration_architect"].run_task(
            integration_context
        )

        # Phase 3: Design debugging tools
        logger.info("Phase 3: Designing debugging tools...")
        debugging_requirements = """
Tools needed:
1. Memory Inspector Sync - Click address in YAZE → jump to same in Mesen2
2. Disassembly Viewer Sync - Show same code in both tools
3. Breakpoint Manager - Add breakpoint in one tool → appears in both
4. Watch Expression System - Monitor values across both tools
        """

        debugging_tools = await self.agents["debugging_tools_builder"].run_task(
            debugging_requirements
        )

        # Phase 4: Design test automation
        logger.info("Phase 4: Designing test automation framework...")
        testing_needs = """
Test automation requirements:
- Run regression tests on ROM hacks (validate core mechanics work)
- Performance benchmarks (FPS, load times)
- Input playback tests (record sequences, replay for testing)
- Automated validation (check game state after actions)
        """

        test_framework = await self.agents["test_automation"].run_task(
            testing_needs
        )

        # Collect all findings
        findings = {
            "lua_library": lua_library,
            "integration_design": integration_design,
            "debugging_tools": debugging_tools,
            "test_framework": test_framework,
        }

        # Synthesize results
        logger.info("Synthesizing Mesen2 integration plan...")
        synthesis = await self.agents["synthesizer"].run_task(findings)

        mission.status = "completed"
        mission.end_time = datetime.now()
        mission.results = {
            "findings": findings,
            "synthesis": synthesis,
        }

        # Save mission report
        await self._save_mission_report(mission)

        return mission.results

    async def _save_lua_scripts(self, lua_library: Dict[str, Any]):
        """Save generated Lua scripts to disk.

        Args:
            lua_library: Library of generated scripts
        """
        library = lua_library.get("library", [])

        for category_data in library:
            category = category_data.get("category", "misc")
            category_dir = self.scripts_dir / category.lower().replace(" ", "_")
            category_dir.mkdir(parents=True, exist_ok=True)

            for script_data in category_data.get("scripts", []):
                script_name = script_data.get("name", "script.lua")
                script_code = script_data.get("code", "-- No code generated")
                script_path = category_dir / script_name

                # Create script with header
                header = f"""-- {script_name}
-- Category: {category}
-- Description: {script_data.get('description', 'N/A')}
-- Generated: {datetime.now().isoformat()}
--
-- Usage: {script_data.get('usage', 'Load in Mesen2 Script Window')}
--

"""
                with open(script_path, "w") as f:
                    f.write(header + script_code)

                logger.info(f"Saved Lua script: {script_path}")

        # Save library index
        index_path = self.scripts_dir / "README.md"
        with open(index_path, "w") as f:
            f.write("# Mesen2 Lua Script Library\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write(lua_library.get("quick_start_guide", "No guide available."))
            f.write("\n\n## Script Categories\n\n")

            for category_data in library:
                category = category_data.get("category", "misc")
                f.write(f"### {category}\n\n")
                for script_data in category_data.get("scripts", []):
                    f.write(f"- **{script_data.get('name')}**: {script_data.get('description')}\n")
                f.write("\n")

        logger.info(f"Saved script library index: {index_path}")

    async def _save_mission_report(self, mission: Mesen2IntegrationMission):
        """Save mission report to disk.

        Args:
            mission: Completed mission
        """
        mission_dir = self.output_dir / "integration_mission"
        mission_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_path = mission_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump({
                "mission": mission.to_dict(),
                "results": mission.results,
            }, f, indent=2)

        # Save markdown synthesis
        synthesis_path = mission_dir / "synthesis.md"
        with open(synthesis_path, "w") as f:
            f.write(f"# {mission.name}\n\n")
            f.write(f"**Started**: {mission.start_time.isoformat()}\n")
            f.write(f"**Completed**: {mission.end_time.isoformat()}\n\n")
            f.write("---\n\n")
            f.write(mission.results.get("synthesis", "No synthesis available."))

        # Save integration design separately
        integration_path = mission_dir / "integration_design.json"
        with open(integration_path, "w") as f:
            json.dump(
                mission.results["findings"]["integration_design"],
                f,
                indent=2
            )

        logger.info(f"Mission report saved to {mission_dir}")


async def main():
    """Main entry point for Mesen2 integration mission."""
    coordinator = Mesen2SwarmCoordinator()
    await coordinator.setup()

    logger.info("Launching Mesen2 integration mission...")
    results = await coordinator.launch_integration_mission()

    print("\n=== MESEN2 INTEGRATION MISSION COMPLETE ===\n")
    print(results["synthesis"])
    print(f"\n\nLua scripts saved to: {coordinator.scripts_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
