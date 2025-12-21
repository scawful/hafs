"""Training Campaign Orchestrator.

Coordinates autonomous agents for pilot monitoring, validation, and
full campaign launch.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents.autonomous.campaign_launcher import (
    CampaignLauncher,
    CampaignMonitor,
)
from agents.autonomous.pilot_quality_monitor import (
    CampaignValidator,
    PilotQualityMonitor,
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates autonomous training campaign workflow."""

    def __init__(
        self,
        quality_threshold: float = 0.75,
        min_pilot_samples: int = 150,
        campaign_target: int = 34500,
        auto_launch: bool = True,
    ):
        """Initialize orchestrator.

        Args:
            quality_threshold: Minimum pilot quality score
            min_pilot_samples: Minimum pilot samples required
            campaign_target: Full campaign sample target
            auto_launch: Auto-launch campaign if validation passes
        """
        self.quality_threshold = quality_threshold
        self.min_pilot_samples = min_pilot_samples
        self.campaign_target = campaign_target
        self.auto_launch = auto_launch

        # Initialize agents
        self.pilot_monitor = PilotQualityMonitor(
            checkpoint_interval=10,
            quality_threshold=quality_threshold,
        )

        self.validator = CampaignValidator(
            quality_threshold=quality_threshold,
            min_samples=min_pilot_samples,
        )

        self.launcher = CampaignLauncher(
            campaign_target=campaign_target,
            auto_launch=auto_launch,
        )

        self.campaign_monitor = CampaignMonitor(
            check_interval=60,
        )

        # State file
        self.state_file = Path.home() / ".context/training/orchestrator_state.json"

    async def setup(self):
        """Initialize all agents."""
        logger.info("Initializing training orchestrator...")
        await self.pilot_monitor.setup()
        await self.validator.setup()
        await self.launcher.setup()
        await self.campaign_monitor.setup()
        logger.info("All agents initialized")

    def _save_state(self, state: Dict[str, Any]):
        """Save orchestrator state.

        Args:
            state: Current state dict
        """
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def run_autonomous(self):
        """Run full autonomous workflow.

        Workflow:
        1. Monitor pilot generation
        2. Validate pilot results
        3. Launch full campaign if approved
        4. Monitor campaign progress
        """
        logger.info("=" * 70)
        logger.info("AUTONOMOUS TRAINING CAMPAIGN ORCHESTRATOR")
        logger.info("=" * 70)
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Minimum pilot samples: {self.min_pilot_samples}")
        logger.info(f"Campaign target: {self.campaign_target}")
        logger.info(f"Auto-launch: {self.auto_launch}")
        logger.info("=" * 70)
        logger.info("")

        state = {
            "started": datetime.now().isoformat(),
            "phase": "monitoring_pilot",
        }
        self._save_state(state)

        # Phase 1: Monitor pilot
        logger.info("PHASE 1: Monitoring pilot generation...")
        logger.info("")

        pilot_result = await self.pilot_monitor.run_task("monitor")

        state["phase"] = "validating_pilot"
        state["pilot_result"] = pilot_result
        self._save_state(state)

        # Phase 2: Validate pilot
        logger.info("")
        logger.info("PHASE 2: Validating pilot results...")
        logger.info("")

        validation_result = await self.validator.run_task("validate")

        state["phase"] = "validation_complete"
        state["validation_result"] = validation_result
        self._save_state(state)

        # Phase 3: Launch campaign if approved
        if validation_result.get("approved", False):
            logger.info("")
            logger.info("PHASE 3: Launching full campaign...")
            logger.info("")

            launch_result = await self.launcher.run_task("launch")

            state["phase"] = "campaign_launched"
            state["launch_result"] = launch_result
            self._save_state(state)

            # Phase 4: Monitor campaign
            if launch_result.get("launched", False):
                logger.info("")
                logger.info("PHASE 4: Monitoring campaign progress...")
                logger.info("")

                campaign_result = await self.campaign_monitor.run_task("monitor")

                state["phase"] = "campaign_complete"
                state["campaign_result"] = campaign_result
                state["completed"] = datetime.now().isoformat()
                self._save_state(state)

                logger.info("")
                logger.info("=" * 70)
                logger.info("AUTONOMOUS WORKFLOW COMPLETE")
                logger.info("=" * 70)
                logger.info(f"Total samples generated: {campaign_result.get('progress', 0)}")
                logger.info("")

            else:
                logger.error("Campaign launch failed")
                state["phase"] = "launch_failed"
                state["completed"] = datetime.now().isoformat()
                self._save_state(state)

        else:
            logger.warning("Pilot validation failed - stopping workflow")
            logger.warning(f"Reason: {validation_result.get('reason')}")
            logger.warning(f"Recommendation: {validation_result.get('recommendation')}")

            state["phase"] = "validation_failed"
            state["completed"] = datetime.now().isoformat()
            self._save_state(state)

        return state

    async def monitor_only(self):
        """Only monitor pilot - don't launch campaign."""
        logger.info("Running in MONITOR ONLY mode...")
        result = await self.pilot_monitor.run_task("monitor")
        return result

    async def validate_only(self):
        """Only validate pilot results."""
        logger.info("Running in VALIDATE ONLY mode...")
        result = await self.validator.run_task("validate")
        return result

    async def launch_only(self):
        """Only launch campaign (skip monitoring/validation)."""
        logger.info("Running in LAUNCH ONLY mode...")
        result = await self.launcher.run_task("launch")
        return result


async def main():
    """Main entry point for orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous training campaign orchestrator"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "monitor", "validate", "launch"],
        default="auto",
        help="Operation mode"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.75,
        help="Minimum quality threshold"
    )
    parser.add_argument(
        "--campaign-target",
        type=int,
        default=34500,
        help="Full campaign sample target"
    )
    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Disable automatic campaign launch"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        quality_threshold=args.quality_threshold,
        campaign_target=args.campaign_target,
        auto_launch=not args.no_auto_launch,
    )

    await orchestrator.setup()

    # Run based on mode
    if args.mode == "auto":
        result = await orchestrator.run_autonomous()
    elif args.mode == "monitor":
        result = await orchestrator.monitor_only()
    elif args.mode == "validate":
        result = await orchestrator.validate_only()
    elif args.mode == "launch":
        result = await orchestrator.launch_only()

    print("")
    print("=" * 70)
    print("FINAL RESULT:")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
