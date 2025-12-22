#!/usr/bin/env python3
"""Train Oracle experts using configuration presets.

Usage:
    python train.py <preset_name>
    python train.py oracle-rauru-mac
    python train.py oracle-rauru-cloud
    python train.py --list-presets
    python train.py --list-experts
    python train.py --list-hardware
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hafs.training.config_trainer import ConfigTrainer

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore


def list_presets(config_path: Path):
    """List available training presets."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    print("Available Training Presets:")
    print("=" * 80)
    for name, preset in config["presets"].items():
        expert = config["experts"][preset["expert"]]
        hardware = config["hardware"][preset["hardware"]]
        print(f"\n{name}:")
        print(f"  Expert:   {expert['display_name']} ({expert['role']})")
        print(f"  Hardware: {hardware['name']}")
        print(f"  Dataset:  {preset['dataset']}")
        print(f"  LoRA:     {preset['lora']}")
        print(f"  Batch:    {preset['batch_size']} x {preset['gradient_accumulation']}")


def list_experts(config_path: Path):
    """List available Oracle experts."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    print("Available Oracle Experts:")
    print("=" * 80)
    for name, expert in config["experts"].items():
        print(f"\n{name}:")
        print(f"  Name:      {expert['display_name']}")
        print(f"  Role:      {expert['role']}")
        print(f"  Group:     {expert['group']}")
        print(f"  Base:      {expert['base_model']}")
        print(f"  Special:   {expert['specialization']}")


def list_hardware(config_path: Path):
    """List available hardware profiles."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    print("Available Hardware Profiles:")
    print("=" * 80)
    for name, hw in config["hardware"].items():
        status = hw.get("status", "available")
        status_icon = "✓" if status == "available" else "✗"

        print(f"\n{status_icon} {name}:")
        print(f"  Name:   {hw['name']}")
        print(f"  Device: {hw['device']}")
        print(f"  Memory: {hw['available_memory_gb']} GB")
        print(f"  Batch:  {hw['max_batch_size']}")
        print(f"  SeqLen: {hw['max_sequence_length']}")
        print(f"  LoRA:   r={hw['max_lora_rank']}")
        print(f"  Status: {status}")
        if status != "available":
            print(f"  Reason: {hw.get('reason', 'Unknown')}")
        if hw.get("cost_per_hour"):
            print(f"  Cost:   ${hw['cost_per_hour']:.2f}/hour")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Oracle experts using configuration presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py oracle-rauru-mac       # Train on Mac MPS
  python train.py oracle-rauru-cloud     # Train on cloud GPU
  python train.py --list-presets         # Show all presets
  python train.py --list-experts         # Show all experts
  python train.py --list-hardware        # Show hardware profiles
        """,
    )

    parser.add_argument(
        "preset",
        nargs="?",
        help="Training preset name (e.g., oracle-rauru-mac)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets",
    )
    parser.add_argument(
        "--list-experts",
        action="store_true",
        help="List available Oracle experts",
    )
    parser.add_argument(
        "--list-hardware",
        action="store_true",
        help="List available hardware profiles",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "config" / "training.toml",
        help="Path to training config file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Handle list commands
    if args.list_presets:
        list_presets(args.config)
        return 0

    if args.list_experts:
        list_experts(args.config)
        return 0

    if args.list_hardware:
        list_hardware(args.config)
        return 0

    # Train with preset
    if not args.preset:
        parser.print_help()
        print("\nError: preset name required (or use --list-presets)")
        return 1

    try:
        trainer = ConfigTrainer(args.config)
        output_dir = trainer.train(args.preset)

        print()
        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Model saved to: {output_dir}")
        print()

        return 0

    except KeyError as e:
        print(f"Error: Preset '{args.preset}' not found")
        print(f"\nRun 'python train.py --list-presets' to see available presets")
        return 1

    except Exception as e:
        logging.exception("Training failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
