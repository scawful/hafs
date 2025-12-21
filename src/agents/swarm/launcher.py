"""Unified Swarm Launcher for YAZE and Mesen2 missions.

Command-line interface for launching swarm analysis missions on YAZE
and Mesen2 codebases.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from agents.swarm.mesen2_swarm import Mesen2SwarmCoordinator
from agents.swarm.yaze_swarm import YazeSwarmCoordinator

logger = logging.getLogger(__name__)


class SwarmLauncher:
    """Unified launcher for all swarm missions."""

    def __init__(
        self,
        yaze_path: Optional[Path] = None,
        mesen2_path: Optional[Path] = None,
    ):
        """Initialize swarm launcher.

        Args:
            yaze_path: Path to YAZE repository
            mesen2_path: Path to Mesen2 repository
        """
        self.yaze_path = yaze_path or Path.home() / "Code" / "yaze"
        self.mesen2_path = mesen2_path

        # Initialize coordinators
        self.yaze_coordinator: Optional[YazeSwarmCoordinator] = None
        self.mesen2_coordinator: Optional[Mesen2SwarmCoordinator] = None

    async def setup_yaze(self):
        """Setup YAZE swarm coordinator."""
        if not self.yaze_coordinator:
            self.yaze_coordinator = YazeSwarmCoordinator(self.yaze_path)
            await self.yaze_coordinator.setup()
            logger.info("YAZE swarm coordinator ready")

    async def setup_mesen2(self):
        """Setup Mesen2 swarm coordinator."""
        if not self.mesen2_coordinator:
            self.mesen2_coordinator = Mesen2SwarmCoordinator(
                self.yaze_path,
                self.mesen2_path
            )
            await self.mesen2_coordinator.setup()
            logger.info("Mesen2 swarm coordinator ready")

    async def launch_yaze_performance(self):
        """Launch YAZE performance optimization mission."""
        await self.setup_yaze()
        logger.info("=" * 60)
        logger.info("LAUNCHING: YAZE Performance Optimization Mission")
        logger.info("=" * 60)

        results = await self.yaze_coordinator.launch_performance_mission()

        print("\n" + "=" * 60)
        print("YAZE PERFORMANCE OPTIMIZATION - SYNTHESIS")
        print("=" * 60)
        print(results["synthesis"])
        print("\n")

        return results

    async def launch_yaze_audio(self):
        """Launch YAZE audio debugging mission."""
        await self.setup_yaze()
        logger.info("=" * 60)
        logger.info("LAUNCHING: YAZE Audio System Debug Mission")
        logger.info("=" * 60)

        results = await self.yaze_coordinator.launch_audio_mission()

        print("\n" + "=" * 60)
        print("YAZE AUDIO SYSTEM DEBUG - SYNTHESIS")
        print("=" * 60)
        print(results["synthesis"])
        print("\n")

        return results

    async def launch_yaze_input(self):
        """Launch YAZE input fix mission."""
        await self.setup_yaze()
        logger.info("=" * 60)
        logger.info("LAUNCHING: YAZE Input System Fix Mission")
        logger.info("=" * 60)

        results = await self.yaze_coordinator.launch_input_mission()

        print("\n" + "=" * 60)
        print("YAZE INPUT SYSTEM FIX - SYNTHESIS")
        print("=" * 60)
        print(results["synthesis"])
        print("\n")

        return results

    async def launch_yaze_all(self):
        """Launch all YAZE missions sequentially."""
        await self.setup_yaze()

        logger.info("=" * 60)
        logger.info("LAUNCHING: ALL YAZE IMPROVEMENT MISSIONS")
        logger.info("=" * 60)

        results = await self.yaze_coordinator.launch_all_missions()

        print("\n" + "=" * 60)
        print("ALL YAZE MISSIONS COMPLETE")
        print("=" * 60)
        print(f"Completed {len(results)} missions")
        print(f"Reports saved to: {self.yaze_coordinator.output_dir}")
        print("\n")

        return results

    async def launch_mesen2_integration(self):
        """Launch Mesen2 integration mission."""
        await self.setup_mesen2()
        logger.info("=" * 60)
        logger.info("LAUNCHING: Mesen2 Integration Mission")
        logger.info("=" * 60)

        results = await self.mesen2_coordinator.launch_integration_mission()

        print("\n" + "=" * 60)
        print("MESEN2 INTEGRATION - SYNTHESIS")
        print("=" * 60)
        print(results["synthesis"])
        print("\n" + "=" * 60)
        print("LUA SCRIPTS")
        print("=" * 60)
        print(f"Generated Lua scripts saved to:")
        print(f"  {self.mesen2_coordinator.scripts_dir}")
        print("\n")

        return results

    async def launch_all(self):
        """Launch all missions (YAZE + Mesen2)."""
        logger.info("=" * 60)
        logger.info("LAUNCHING: FULL SWARM CAMPAIGN")
        logger.info("=" * 60)

        # Launch YAZE missions
        yaze_results = await self.launch_yaze_all()

        # Launch Mesen2 mission
        mesen2_results = await self.launch_mesen2_integration()

        print("\n" + "=" * 60)
        print("FULL SWARM CAMPAIGN COMPLETE")
        print("=" * 60)
        print(f"YAZE missions: {len(yaze_results)}")
        print(f"Mesen2 mission: 1")
        print(f"\nReports saved to:")
        print(f"  YAZE: {self.yaze_coordinator.output_dir}")
        print(f"  Mesen2: {self.mesen2_coordinator.output_dir}")
        print(f"  Lua scripts: {self.mesen2_coordinator.scripts_dir}")
        print("\n")

        return {
            "yaze": yaze_results,
            "mesen2": mesen2_results,
        }


async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Launch YAZE and Mesen2 swarm improvement missions"
    )

    parser.add_argument(
        "mission",
        choices=[
            "yaze-performance",
            "yaze-audio",
            "yaze-input",
            "yaze-all",
            "mesen2-integration",
            "all",
        ],
        help="Mission to launch"
    )

    parser.add_argument(
        "--yaze-path",
        type=Path,
        default=Path.home() / "Code" / "yaze",
        help="Path to YAZE repository (default: ~/Code/yaze)"
    )

    parser.add_argument(
        "--mesen2-path",
        type=Path,
        default=None,
        help="Path to Mesen2 repository (optional)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / ".context" / "swarms",
        help="Output directory for swarm reports"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Create launcher
    launcher = SwarmLauncher(
        yaze_path=args.yaze_path,
        mesen2_path=args.mesen2_path,
    )

    # Dispatch to appropriate mission
    mission_map = {
        "yaze-performance": launcher.launch_yaze_performance,
        "yaze-audio": launcher.launch_yaze_audio,
        "yaze-input": launcher.launch_yaze_input,
        "yaze-all": launcher.launch_yaze_all,
        "mesen2-integration": launcher.launch_mesen2_integration,
        "all": launcher.launch_all,
    }

    try:
        await mission_map[args.mission]()
        logger.info("Mission completed successfully")
    except Exception as e:
        logger.error(f"Mission failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
