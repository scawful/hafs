#!/usr/bin/env python3
"""
Test Model Training and Inference Workflow

This script helps you:
1. Check if any training data exists
2. Test the Unsloth training setup
3. Run inference on a trained model
4. Compare model outputs

Usage:
  python scripts/test_model_training.py --check-data
  python scripts/test_model_training.py --train-demo
  python scripts/test_model_training.py --test-model <path_to_model>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add hafs to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_training_data():
    """Check if any usable training data exists."""
    print("=" * 80)
    print("Checking Training Data")
    print("=" * 80)

    dataset_dir = Path.home() / ".context" / "training" / "datasets"

    if not dataset_dir.exists():
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False

    # Find all dataset directories
    dataset_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]

    if not dataset_dirs:
        print(f"✗ No dataset directories found in {dataset_dir}")
        return False

    print(f"\n✓ Found {len(dataset_dirs)} dataset directories")

    # Check each dataset for non-empty JSONL files
    usable_datasets = []

    for ds_dir in sorted(dataset_dirs, key=lambda x: x.stat().st_mtime, reverse=True):
        train_file = ds_dir / "train.jsonl"
        metadata_file = ds_dir / "metadata.json"

        if not train_file.exists():
            continue

        # Check file size
        file_size = train_file.stat().st_size

        if file_size == 0:
            continue

        # Count lines
        with open(train_file) as f:
            line_count = sum(1 for _ in f)

        # Read metadata
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)

        usable_datasets.append({
            "path": ds_dir,
            "name": ds_dir.name,
            "samples": line_count,
            "size_mb": file_size / (1024 * 1024),
            "metadata": metadata
        })

    if not usable_datasets:
        print("\n✗ No usable training data found (all datasets are empty)")
        print("\nThe training campaigns are generating data but it's being rejected by quality filtering.")
        print("\nOptions:")
        print("  1. Lower quality thresholds in the campaign scripts")
        print("  2. Run a demo training with example data: python scripts/test_model_training.py --train-demo")
        print("  3. Debug quality pipeline to see why samples are rejected")
        return False

    print(f"\n✓ Found {len(usable_datasets)} datasets with training data:")
    print()

    for ds in usable_datasets[:10]:  # Show top 10
        print(f"  {ds['name']}")
        print(f"    Samples: {ds['samples']}")
        print(f"    Size: {ds['size_mb']:.2f} MB")
        if 'domains' in ds['metadata']:
            print(f"    Domains: {', '.join(ds['metadata']['domains'])}")
        print()

    # Recommend the best dataset
    best_dataset = max(usable_datasets, key=lambda x: x['samples'])
    print(f"Recommended dataset for training: {best_dataset['name']}")
    print(f"  Path: {best_dataset['path']}")
    print(f"  Samples: {best_dataset['samples']}")

    return True


def train_demo_model():
    """Run a quick demo training with example data."""
    print("=" * 80)
    print("Demo Model Training")
    print("=" * 80)
    print()
    print("This will run the example Unsloth training script")
    print("on a small dataset (1000 samples) to verify your GPU setup.")
    print()

    example_script = Path(__file__).parent / "example_unsloth_training.py"

    if not example_script.exists():
        print(f"✗ Example script not found: {example_script}")
        return False

    print(f"Running: {example_script}")
    print()

    import subprocess

    result = subprocess.run(
        [sys.executable, str(example_script)],
        capture_output=False,
        text=True,
    )

    return result.returncode == 0


def test_trained_model(model_path: str):
    """Test inference on a trained model."""
    print("=" * 80)
    print("Testing Trained Model")
    print("=" * 80)
    print()

    model_dir = Path(model_path)

    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        return False

    print(f"Model: {model_dir}")
    print()

    try:
        import torch
        from unsloth import FastLanguageModel

        # Load model
        print("[1/3] Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir),
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print("✓ Model loaded")

        # Enable fast inference
        FastLanguageModel.for_inference(model)

        # Test prompts
        test_prompts = [
            "What is the purpose of the REP #$30 instruction in 65816 assembly?",
            "Explain how DMA works in the SNES",
            "What does the JSL instruction do?",
        ]

        print("\n[2/3] Running inference tests...")
        print()

        for i, prompt in enumerate(test_prompts, 1):
            print(f"Test {i}/{len(test_prompts)}")
            print(f"Prompt: {prompt}")

            # Format as instruction
            formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

            inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the response part
            if "### Response:" in response:
                response = response.split("### Response:")[1].strip()

            print(f"Response: {response[:200]}...")
            print()

        print("[3/3] Inference tests complete")
        print()
        print("✓ Model is working!")

        return True

    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall training dependencies:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  pip install unsloth transformers")
        return False
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test model training and inference")
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check if training data exists",
    )
    parser.add_argument(
        "--train-demo",
        action="store_true",
        help="Run demo training with example data",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        metavar="MODEL_PATH",
        help="Test inference on a trained model",
    )

    args = parser.parse_args()

    if args.check_data:
        success = check_training_data()
        return 0 if success else 1

    elif args.train_demo:
        success = train_demo_model()
        return 0 if success else 1

    elif args.test_model:
        success = test_trained_model(args.test_model)
        return 0 if success else 1

    else:
        parser.print_help()
        print("\n" + "=" * 80)
        print("Quick Start:")
        print("=" * 80)
        print("\n1. Check if you have training data:")
        print("     python scripts/test_model_training.py --check-data")
        print("\n2. Run demo training (tests GPU setup):")
        print("     python scripts/test_model_training.py --train-demo")
        print("\n3. Test a trained model:")
        print("     python scripts/test_model_training.py --test-model D:/training/models/my_model")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
