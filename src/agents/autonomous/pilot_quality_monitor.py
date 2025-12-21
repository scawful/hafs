"""Pilot Quality Monitoring Agent.

Autonomous agent that monitors pilot generation progress and quality metrics,
then triggers full campaign if quality thresholds are met.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.core.base import BaseAgent

logger = logging.getLogger(__name__)


class PilotQualityMonitor(BaseAgent):
    """Agent that monitors pilot generation and validates quality."""

    def __init__(
        self,
        pilot_log_path: Optional[Path] = None,
        checkpoint_interval: int = 10,
        quality_threshold: float = 0.75,
    ):
        """Initialize quality monitor.

        Args:
            pilot_log_path: Path to pilot generation log
            checkpoint_interval: How often to check (seconds)
            quality_threshold: Minimum quality score to pass
        """
        super().__init__(
            "PilotQualityMonitor",
            "Monitor pilot generation quality and progress"
        )

        self.pilot_log_path = pilot_log_path or (
            Path.home() / ".context/training/pilot_campaign.log"
        )
        self.checkpoint_interval = checkpoint_interval
        self.quality_threshold = quality_threshold

        # Monitoring state
        self.last_progress = 0
        self.samples_checked = 0
        self.quality_scores: List[float] = []
        self.acceptance_rate = 0.0
        self.is_complete = False

        # Status file for other agents
        self.status_file = Path.home() / ".context/training/pilot_monitor_status.json"

    def _parse_log(self) -> Dict[str, Any]:
        """Parse pilot log for current status.

        Returns:
            Dict with progress, quality metrics, and completion status
        """
        if not self.pilot_log_path.exists():
            return {
                "progress": 0,
                "total": 190,
                "quality_scores": [],
                "acceptance_rate": 0.0,
                "is_complete": False,
                "last_update": datetime.now().isoformat(),
            }

        progress = 0
        total = 190
        quality_scores = []
        accepted = 0
        rejected = 0

        try:
            with open(self.pilot_log_path, "r") as f:
                lines = f.readlines()

            # Parse from the end for efficiency
            for line in reversed(lines[-200:]):
                # Extract progress
                if "Progress:" in line:
                    parts = line.split("Progress:")
                    if len(parts) > 1:
                        progress_str = parts[1].strip().split()[0]
                        if "/" in progress_str:
                            progress = int(progress_str.split("/")[0])
                            total = int(progress_str.split("/")[1])
                            break

            # Look for quality indicators
            for line in lines:
                if "quality_score" in line.lower():
                    try:
                        # Try to extract score
                        if ":" in line:
                            score_str = line.split("quality_score")[-1].strip()
                            score = float(score_str.split()[0].rstrip(","))
                            quality_scores.append(score)
                    except (ValueError, IndexError):
                        pass

                if "accepted" in line.lower():
                    accepted += 1
                elif "rejected" in line.lower():
                    rejected += 1

        except Exception as e:
            logger.error(f"Failed to parse pilot log: {e}")

        acceptance_rate = (
            accepted / (accepted + rejected)
            if (accepted + rejected) > 0
            else 0.0
        )

        is_complete = progress >= total

        return {
            "progress": progress,
            "total": total,
            "quality_scores": quality_scores,
            "acceptance_rate": acceptance_rate,
            "accepted": accepted,
            "rejected": rejected,
            "is_complete": is_complete,
            "last_update": datetime.now().isoformat(),
        }

    def _estimate_quality(self, status: Dict[str, Any]) -> float:
        """Estimate overall quality from available metrics.

        Args:
            status: Current status dict

        Returns:
            Estimated quality score (0.0-1.0)
        """
        # If we have explicit quality scores, use average
        if status["quality_scores"]:
            avg_quality = sum(status["quality_scores"]) / len(status["quality_scores"])
            return avg_quality

        # Otherwise use acceptance rate as proxy
        # (Assuming high acceptance rate indicates good quality)
        return status["acceptance_rate"]

    def _save_status(self, status: Dict[str, Any], quality: float):
        """Save monitoring status for other agents.

        Args:
            status: Current status
            quality: Estimated quality score
        """
        report = {
            **status,
            "estimated_quality": quality,
            "quality_threshold": self.quality_threshold,
            "quality_pass": quality >= self.quality_threshold,
            "monitor_timestamp": datetime.now().isoformat(),
        }

        try:
            with open(self.status_file, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved monitoring status to {self.status_file}")
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    async def run_task(self, task: str) -> Dict[str, Any]:
        """Monitor pilot generation until completion.

        Args:
            task: Monitoring parameters (unused, uses init params)

        Returns:
            Final status report
        """
        logger.info("=" * 60)
        logger.info("PILOT QUALITY MONITORING STARTED")
        logger.info("=" * 60)
        logger.info(f"Log path: {self.pilot_log_path}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Check interval: {self.checkpoint_interval}s")
        logger.info("")

        while not self.is_complete:
            # Parse current status
            status = self._parse_log()
            quality = self._estimate_quality(status)

            # Update state
            progress = status["progress"]
            total = status["total"]
            self.is_complete = status["is_complete"]

            # Log progress if changed
            if progress != self.last_progress:
                logger.info(
                    f"Progress: {progress}/{total} ({progress/total*100:.1f}%) | "
                    f"Quality: {quality:.3f} | "
                    f"Acceptance: {status['acceptance_rate']:.2%}"
                )
                self.last_progress = progress

            # Save status for other agents
            self._save_status(status, quality)

            # Wait before next check
            if not self.is_complete:
                await asyncio.sleep(self.checkpoint_interval)

        # Final report
        logger.info("")
        logger.info("=" * 60)
        logger.info("PILOT GENERATION COMPLETE")
        logger.info("=" * 60)

        final_status = self._parse_log()
        final_quality = self._estimate_quality(final_status)

        logger.info(f"Total samples: {final_status['progress']}/{final_status['total']}")
        logger.info(f"Estimated quality: {final_quality:.3f}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
        logger.info(f"Acceptance rate: {final_status['acceptance_rate']:.2%}")

        quality_pass = final_quality >= self.quality_threshold

        if quality_pass:
            logger.info("")
            logger.info("✓ QUALITY PASSED - Ready for full campaign")
        else:
            logger.warning("")
            logger.warning("✗ QUALITY FAILED - Manual review required")

        logger.info("=" * 60)

        # Save final status
        self._save_status(final_status, final_quality)

        return {
            **final_status,
            "estimated_quality": final_quality,
            "quality_pass": quality_pass,
        }


class CampaignValidator(BaseAgent):
    """Agent that validates pilot results and approves campaign launch."""

    def __init__(
        self,
        status_file: Optional[Path] = None,
        quality_threshold: float = 0.75,
        min_samples: int = 150,
    ):
        """Initialize campaign validator.

        Args:
            status_file: Path to pilot monitor status file
            quality_threshold: Minimum quality score
            min_samples: Minimum samples required
        """
        super().__init__(
            "CampaignValidator",
            "Validate pilot results and approve campaign"
        )

        self.status_file = status_file or (
            Path.home() / ".context/training/pilot_monitor_status.json"
        )
        self.quality_threshold = quality_threshold
        self.min_samples = min_samples

    def _load_pilot_status(self) -> Dict[str, Any]:
        """Load pilot monitoring status.

        Returns:
            Pilot status dict
        """
        if not self.status_file.exists():
            logger.error(f"Status file not found: {self.status_file}")
            return {}

        try:
            with open(self.status_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load status: {e}")
            return {}

    async def run_task(self, task: str) -> Dict[str, Any]:
        """Validate pilot and decide on campaign launch.

        Args:
            task: Validation parameters (unused)

        Returns:
            Validation result with launch decision
        """
        logger.info("=" * 60)
        logger.info("CAMPAIGN VALIDATION STARTED")
        logger.info("=" * 60)

        status = self._load_pilot_status()

        if not status:
            logger.error("No pilot status available - cannot validate")
            return {
                "approved": False,
                "reason": "No pilot status available",
                "recommendation": "Wait for pilot to complete",
            }

        # Validation checks
        checks = {
            "completion": status.get("is_complete", False),
            "min_samples": status.get("progress", 0) >= self.min_samples,
            "quality": status.get("estimated_quality", 0.0) >= self.quality_threshold,
        }

        logger.info("")
        logger.info("Validation Checks:")
        logger.info(f"  Pilot complete: {checks['completion']}")
        logger.info(f"  Minimum samples ({self.min_samples}): {checks['min_samples']}")
        logger.info(f"  Quality threshold ({self.quality_threshold}): {checks['quality']}")
        logger.info("")

        # All checks must pass
        approved = all(checks.values())

        if approved:
            logger.info("✓ VALIDATION PASSED - Approving full campaign launch")
            reason = "All validation checks passed"
            recommendation = "Launch full 34.5K sample campaign"
        else:
            failed = [k for k, v in checks.items() if not v]
            logger.warning(f"✗ VALIDATION FAILED - Checks failed: {failed}")
            reason = f"Failed checks: {', '.join(failed)}"
            recommendation = "Review pilot results manually before proceeding"

        logger.info("=" * 60)

        result = {
            "approved": approved,
            "reason": reason,
            "recommendation": recommendation,
            "checks": checks,
            "pilot_status": status,
            "validation_timestamp": datetime.now().isoformat(),
        }

        # Save validation result
        validation_file = Path.home() / ".context/training/campaign_validation.json"
        with open(validation_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved validation result to {validation_file}")

        return result


async def main():
    """Test quality monitoring."""
    monitor = PilotQualityMonitor(
        checkpoint_interval=5,
        quality_threshold=0.75,
    )
    await monitor.setup()

    result = await monitor.run_task("monitor")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    asyncio.run(main())
