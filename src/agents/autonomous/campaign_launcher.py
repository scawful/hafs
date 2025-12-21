"""Campaign Launcher Agent.

Autonomous agent that launches the full 34.5K sample generation campaign
after pilot validation passes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents.core.base import BaseAgent

logger = logging.getLogger(__name__)


class CampaignLauncher(BaseAgent):
    """Agent that launches and monitors full generation campaign."""

    def __init__(
        self,
        validation_file: Optional[Path] = None,
        campaign_target: int = 34500,
        auto_launch: bool = True,
    ):
        """Initialize campaign launcher.

        Args:
            validation_file: Path to campaign validation results
            campaign_target: Target number of samples
            auto_launch: Automatically launch if validation passes
        """
        super().__init__(
            "CampaignLauncher",
            "Launch full training data generation campaign"
        )

        self.validation_file = validation_file or (
            Path.home() / ".context/training/campaign_validation.json"
        )
        self.campaign_target = campaign_target
        self.auto_launch = auto_launch

        # Campaign state
        self.campaign_log = Path.home() / ".context/training/full_campaign.log"
        self.campaign_pid_file = Path.home() / ".context/training/campaign.pid"
        self.campaign_status_file = Path.home() / ".context/training/campaign_status.json"

    def _load_validation(self) -> Dict[str, Any]:
        """Load campaign validation results.

        Returns:
            Validation dict
        """
        if not self.validation_file.exists():
            logger.warning(f"Validation file not found: {self.validation_file}")
            return {}

        try:
            with open(self.validation_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load validation: {e}")
            return {}

    def _is_campaign_running(self) -> bool:
        """Check if campaign is already running.

        Returns:
            True if running
        """
        if not self.campaign_pid_file.exists():
            return False

        try:
            with open(self.campaign_pid_file, "r") as f:
                pid = int(f.read().strip())

            # Check if process exists
            result = subprocess.run(
                ["ps", "-p", str(pid)],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to check campaign status: {e}")
            return False

    async def _launch_campaign(self) -> Dict[str, Any]:
        """Launch the full generation campaign.

        Returns:
            Launch result
        """
        logger.info("=" * 60)
        logger.info("LAUNCHING FULL GENERATION CAMPAIGN")
        logger.info("=" * 60)
        logger.info(f"Target: {self.campaign_target} samples")
        logger.info(f"Log: {self.campaign_log}")
        logger.info("")

        # Build command
        # Assuming there's a generation script
        hafs_root = Path(__file__).parent.parent.parent.parent
        campaign_script = hafs_root / "src/agents/training/scripts/generate_full_campaign.py"

        if not campaign_script.exists():
            # Fallback: use Python module
            cmd = [
                "python", "-m",
                "agents.training.scripts.generate_campaign",
                "--target", str(self.campaign_target),
                "--output-name", f"full_campaign_{self.campaign_target}",
                "--parallel",
                "--use-active-learning",
            ]
        else:
            cmd = [
                "python", str(campaign_script),
                "--target", str(self.campaign_target),
                "--output-name", f"full_campaign_{self.campaign_target}",
                "--parallel",
                "--use-active-learning",
            ]

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("")

        try:
            # Launch as background process
            with open(self.campaign_log, "w") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=hafs_root,
                    env={**subprocess.os.environ, "PYTHONPATH": str(hafs_root / "src")},
                )

            # Save PID
            with open(self.campaign_pid_file, "w") as f:
                f.write(str(process.pid))

            logger.info(f"✓ Campaign launched (PID: {process.pid})")
            logger.info(f"  Log: {self.campaign_log}")
            logger.info(f"  Monitor with: tail -f {self.campaign_log}")
            logger.info("=" * 60)

            return {
                "launched": True,
                "pid": process.pid,
                "log_path": str(self.campaign_log),
                "target": self.campaign_target,
                "launch_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to launch campaign: {e}", exc_info=True)
            return {
                "launched": False,
                "error": str(e),
                "launch_time": datetime.now().isoformat(),
            }

    def _save_status(self, status: Dict[str, Any]):
        """Save campaign status.

        Args:
            status: Status dict
        """
        try:
            with open(self.campaign_status_file, "w") as f:
                json.dump(status, f, indent=2)
            logger.info(f"Saved campaign status to {self.campaign_status_file}")
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    async def run_task(self, task: str) -> Dict[str, Any]:
        """Launch campaign if validation passed.

        Args:
            task: Launch parameters (unused)

        Returns:
            Launch result
        """
        logger.info("=" * 60)
        logger.info("CAMPAIGN LAUNCHER STARTED")
        logger.info("=" * 60)

        # Check if already running
        if self._is_campaign_running():
            logger.warning("Campaign already running - skipping launch")
            return {
                "launched": False,
                "reason": "Campaign already running",
                "pid_file": str(self.campaign_pid_file),
            }

        # Load validation
        validation = self._load_validation()

        if not validation:
            logger.error("No validation results available")
            return {
                "launched": False,
                "reason": "No validation results",
                "recommendation": "Wait for pilot validation",
            }

        # Check approval
        approved = validation.get("approved", False)

        if not approved:
            logger.warning("Campaign not approved by validator")
            logger.warning(f"Reason: {validation.get('reason')}")
            logger.warning(f"Recommendation: {validation.get('recommendation')}")
            return {
                "launched": False,
                "reason": "Validation failed",
                "validation": validation,
            }

        # Validation passed - launch campaign
        if not self.auto_launch:
            logger.info("Auto-launch disabled - manual launch required")
            return {
                "launched": False,
                "reason": "Auto-launch disabled",
                "validation": validation,
                "recommendation": "Enable auto_launch or launch manually",
            }

        # Launch!
        result = await self._launch_campaign()

        # Save status
        self._save_status({
            **result,
            "validation": validation,
        })

        return result


class CampaignMonitor(BaseAgent):
    """Agent that monitors ongoing campaign progress."""

    def __init__(
        self,
        campaign_log: Optional[Path] = None,
        status_file: Optional[Path] = None,
        check_interval: int = 30,
    ):
        """Initialize campaign monitor.

        Args:
            campaign_log: Path to campaign log
            status_file: Path to status file
            check_interval: Check interval in seconds
        """
        super().__init__(
            "CampaignMonitor",
            "Monitor full campaign progress"
        )

        self.campaign_log = campaign_log or (
            Path.home() / ".context/training/full_campaign.log"
        )
        self.status_file = status_file or (
            Path.home() / ".context/training/campaign_monitor_status.json"
        )
        self.check_interval = check_interval

    def _parse_campaign_log(self) -> Dict[str, Any]:
        """Parse campaign log for progress.

        Returns:
            Current status dict
        """
        if not self.campaign_log.exists():
            return {
                "progress": 0,
                "total": 34500,
                "percentage": 0.0,
                "is_running": False,
            }

        progress = 0
        total = 34500

        try:
            with open(self.campaign_log, "r") as f:
                lines = f.readlines()

            # Parse from end
            for line in reversed(lines[-100:]):
                if "Progress:" in line:
                    parts = line.split("Progress:")
                    if len(parts) > 1:
                        progress_str = parts[1].strip().split()[0]
                        if "/" in progress_str:
                            progress = int(progress_str.split("/")[0])
                            total = int(progress_str.split("/")[1])
                            break

        except Exception as e:
            logger.error(f"Failed to parse campaign log: {e}")

        return {
            "progress": progress,
            "total": total,
            "percentage": (progress / total * 100) if total > 0 else 0.0,
            "is_running": progress < total,
            "last_update": datetime.now().isoformat(),
        }

    async def run_task(self, task: str) -> Dict[str, Any]:
        """Monitor campaign until completion.

        Args:
            task: Monitor parameters

        Returns:
            Final status
        """
        logger.info("=" * 60)
        logger.info("CAMPAIGN MONITORING STARTED")
        logger.info("=" * 60)

        last_progress = 0

        while True:
            status = self._parse_campaign_log()

            if status["progress"] != last_progress:
                logger.info(
                    f"Campaign progress: {status['progress']}/{status['total']} "
                    f"({status['percentage']:.1f}%)"
                )
                last_progress = status["progress"]

                # Save status
                with open(self.status_file, "w") as f:
                    json.dump(status, f, indent=2)

            if not status["is_running"]:
                logger.info("")
                logger.info("=" * 60)
                logger.info("CAMPAIGN COMPLETE")
                logger.info("=" * 60)
                logger.info(f"Total samples: {status['progress']}")
                break

            await asyncio.sleep(self.check_interval)

        return status


async def main():
    """Test campaign launcher."""
    launcher = CampaignLauncher(auto_launch=False)
    await launcher.setup()

    result = await launcher.run_task("launch")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    asyncio.run(main())
