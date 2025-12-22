#!/usr/bin/env python3
"""Setup fine-tuning models on medical-mechanica.

Downloads base models from Hugging Face and prepares them for LoRA fine-tuning.
Run on Windows GPU server via SSH.
"""

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


MODELS = {
    "qwen2.5-coder-14b": {
        "repo": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "size_gb": 28,
        "vram_gb": 14,
        "quality": "highest",
        "recommended": True,
    },
    "qwen2.5-coder-7b": {
        "repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "size_gb": 14,
        "vram_gb": 8,
        "quality": "high",
        "fast": True,
    },
    "deepseek-coder-33b": {
        "repo": "deepseek-ai/DeepSeek-Coder-33B-instruct",
        "size_gb": 66,
        "vram_gb": 16,
        "quality": "highest",
        "asm_specialist": True,
    },
    "codellama-34b": {
        "repo": "meta-llama/CodeLlama-34b-Instruct-hf",
        "size_gb": 68,
        "vram_gb": 16,
        "quality": "high",
    },
    "phi-3.5": {
        "repo": "microsoft/Phi-3.5-mini-instruct",
        "size_gb": 8,
        "vram_gb": 4,
        "quality": "medium",
        "fast": True,
    },
}


def check_disk_space(models_dir: Path) -> float:
    """Check available disk space in GB."""
    import shutil
    stat = shutil.disk_usage(models_dir)
    return stat.free / (1024**3)


def download_model(model_name: str, models_dir: Path, use_hf_transfer: bool = True):
    """Download model from Hugging Face."""
    model_info = MODELS[model_name]
    repo = model_info["repo"]

    logger.info(f"Downloading {model_name} ({repo})...")
    logger.info(f"  Size: ~{model_info['size_gb']}GB")
    logger.info(f"  VRAM needed: {model_info['vram_gb']}GB")

    # Set up environment for fast downloads
    env = {}
    if use_hf_transfer:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    # Use huggingface-cli to download
    output_dir = models_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "huggingface-cli",
        "download",
        repo,
        "--local-dir", str(output_dir),
        "--local-dir-use-symlinks", "False",
    ]

    try:
        subprocess.run(cmd, check=True, env=env)
        logger.info(f"✓ Downloaded {model_name} to {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to download {model_name}: {e}")
        return False


def setup_unsloth():
    """Install Unsloth for fast LoRA training."""
    logger.info("Setting up Unsloth for fast LoRA training...")

    commands = [
        # Install Unsloth
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        # Install dependencies
        'pip install --no-deps "trl<0.9.0" peft accelerate bitsandbytes',
        # Install training tools
        'pip install datasets transformers torch wandb',
    ]

    for cmd in commands:
        try:
            logger.info(f"Running: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {e}")
            return False

    logger.info("✓ Unsloth setup complete")
    return True


def verify_cuda():
    """Verify CUDA is available."""
    logger.info("Verifying CUDA setup...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✓ CUDA available: {gpu_name} ({vram_gb:.1f}GB VRAM)")
            return True
        else:
            logger.error("✗ CUDA not available")
            return False
    except ImportError:
        logger.error("✗ PyTorch not installed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup fine-tuning models on medical-mechanica"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("D:/hafs_training/models"),
        help="Directory to store models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all", "recommended"],
        default=["recommended"],
        help="Models to download",
    )
    parser.add_argument(
        "--setup-unsloth",
        action="store_true",
        help="Install Unsloth for fast LoRA training",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify CUDA, don't download",
    )

    args = parser.parse_args()

    # Verify CUDA first
    if not verify_cuda():
        logger.error("CUDA verification failed. Exiting.")
        return 1

    if args.verify_only:
        return 0

    # Setup Unsloth if requested
    if args.setup_unsloth:
        if not setup_unsloth():
            logger.error("Unsloth setup failed. Continuing with downloads...")

    # Create models directory
    args.models_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models directory: {args.models_dir}")

    # Check disk space
    free_gb = check_disk_space(args.models_dir)
    logger.info(f"Available disk space: {free_gb:.1f}GB")

    # Determine which models to download
    models_to_download = []
    if "all" in args.models:
        models_to_download = list(MODELS.keys())
    elif "recommended" in args.models:
        models_to_download = [
            name for name, info in MODELS.items()
            if info.get("recommended") or info.get("fast")
        ]
    else:
        models_to_download = args.models

    # Calculate total size
    total_size = sum(MODELS[m]["size_gb"] for m in models_to_download)
    logger.info(f"Total download size: ~{total_size}GB")

    if total_size > free_gb:
        logger.warning(f"Not enough disk space! Need {total_size}GB, have {free_gb:.1f}GB")
        logger.info("Consider downloading models individually or freeing up space.")
        return 1

    # Download models
    success_count = 0
    for model in models_to_download:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {model}...")
        logger.info(f"{'='*60}")

        if download_model(model, args.models_dir):
            success_count += 1

        # Update free space
        free_gb = check_disk_space(args.models_dir)
        logger.info(f"Remaining disk space: {free_gb:.1f}GB\n")

    logger.info(f"\n{'='*60}")
    logger.info(f"Setup complete: {success_count}/{len(models_to_download)} models downloaded")
    logger.info(f"{'='*60}")

    if success_count == len(models_to_download):
        logger.info("✓ All models downloaded successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Prepare your training dataset")
        logger.info("2. Run: python -m agents.training.fine_tune --model qwen2.5-coder-14b")
        return 0
    else:
        logger.warning(f"⚠ Only {success_count} models downloaded")
        return 1


if __name__ == "__main__":
    exit(main())
