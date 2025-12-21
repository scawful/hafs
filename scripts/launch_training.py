#!/usr/bin/env python3
"""Python launcher for autonomous training campaign.

Provides more control and monitoring capabilities than the bash script.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

# Add hafs to path
HAFS_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(HAFS_ROOT / "src"))

from agents.autonomous.training_orchestrator import TrainingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def launch_autonomous(
    quality_threshold: float = 0.75,
    campaign_target: int = 34500,
    auto_launch: bool = True,
):
    """Launch autonomous training workflow.

    Args:
        quality_threshold: Minimum pilot quality
        campaign_target: Full campaign sample target
        auto_launch: Auto-launch campaign if validation passes
    """
    logger.info("=" * 70)
    logger.info("AUTONOMOUS TRAINING CAMPAIGN")
    logger.info("=" * 70)

    orchestrator = TrainingOrchestrator(
        quality_threshold=quality_threshold,
        campaign_target=campaign_target,
        auto_launch=auto_launch,
    )

    await orchestrator.setup()

    # Run autonomous workflow
    result = await orchestrator.run_autonomous()

    return result


async def monitor_pilot_only():
    """Only monitor pilot generation."""
    orchestrator = TrainingOrchestrator()
    await orchestrator.setup()
    result = await orchestrator.monitor_only()
    return result


async def validate_pilot_only():
    """Only validate pilot results."""
    orchestrator = TrainingOrchestrator()
    await orchestrator.setup()
    result = await orchestrator.validate_only()
    return result


async def launch_campaign_only(campaign_target: int = 34500):
    """Only launch campaign (skip monitoring/validation)."""
    orchestrator = TrainingOrchestrator(campaign_target=campaign_target)
    await orchestrator.setup()
    result = await orchestrator.launch_only()
    return result


def show_status():
    """Show current status of training workflow."""
    status_files = {
        "Orchestrator": Path.home() / ".context/training/orchestrator_state.json",
        "Pilot Monitor": Path.home() / ".context/training/pilot_monitor_status.json",
        "Validation": Path.home() / ".context/training/campaign_validation.json",
        "Campaign": Path.home() / ".context/training/campaign_status.json",
    }

    print("=" * 70)
    print("TRAINING WORKFLOW STATUS")
    print("=" * 70)
    print()

    for name, path in status_files.items():
        print(f"{name}:")
        if path.exists():
            try:
                with open(path, "r") as f:
                    status = json.load(f)
                print(json.dumps(status, indent=2))
            except Exception as e:
                print(f"  Error reading status: {e}")
        else:
            print("  No status file found")
        print()


def tail_logs():
    """Tail all relevant log files."""
    log_files = [
        Path.home() / ".context/logs/training_orchestrator.log",
        Path.home() / ".context/training/pilot_campaign.log",
        Path.home() / ".context/training/full_campaign.log",
    ]

    existing_logs = [str(f) for f in log_files if f.exists()]

    if not existing_logs:
        print("No log files found")
        return

    print(f"Tailing {len(existing_logs)} log files...")
    print("Press Ctrl+C to stop")
    print()

    # Use tail -f to follow all logs
    subprocess.run(["tail", "-f"] + existing_logs)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch autonomous training campaign"
    )

    parser.add_argument(
        "command",
        choices=["auto", "monitor", "validate", "launch", "status", "logs"],
        help="Command to execute"
    )

    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.75,
        help="Minimum quality threshold (default: 0.75)"
    )

    parser.add_argument(
        "--campaign-target",
        type=int,
        default=34500,
        help="Campaign sample target (default: 34500)"
    )

    parser.add_argument(
        "--no-auto-launch",
        action="store_true",
        help="Disable automatic campaign launch"
    )

    parser.add_argument(
        "--background",
        action="store_true",
        help="Run in background (detached)"
    )

    args = parser.parse_args()

    # Handle non-async commands
    if args.command == "status":
        show_status()
        return

    if args.command == "logs":
        tail_logs()
        return

    # Handle background mode
    if args.background and args.command == "auto":
        script_path = Path(__file__).parent / "launch_autonomous_training.sh"

        env = {
            **subprocess.os.environ,
            "QUALITY_THRESHOLD": str(args.quality_threshold),
            "CAMPAIGN_TARGET": str(args.campaign_target),
            "AUTO_LAUNCH": "false" if args.no_auto_launch else "true",
        }

        subprocess.run([str(script_path)], env=env)
        return

    # Run async commands
    if args.command == "auto":
        result = await launch_autonomous(
            quality_threshold=args.quality_threshold,
            campaign_target=args.campaign_target,
            auto_launch=not args.no_auto_launch,
        )
    elif args.command == "monitor":
        result = await monitor_pilot_only()
    elif args.command == "validate":
        result = await validate_pilot_only()
    elif args.command == "launch":
        result = await launch_campaign_only(args.campaign_target)

    print()
    print("=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
