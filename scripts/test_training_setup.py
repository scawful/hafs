#!/usr/bin/env python3
"""
PyTorch/Unsloth Training Environment Validation Script
Target: medical-mechanica (Windows 11 Pro, RTX 5060 Ti 16GB, CUDA 11.2)
Created: 2025-12-21

This script validates the training environment setup by checking:
- PyTorch installation and CUDA availability
- GPU detection and memory
- Unsloth installation
- Required dependencies
- System information
"""

import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_section(message: str):
    """Print a section header"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}[SECTION] {message}{Colors.ENDC}")


def print_success(message: str):
    """Print a success message"""
    print(f"{Colors.OKGREEN}[OK] {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message"""
    print(f"{Colors.FAIL}[ERROR] {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print a warning message"""
    print(f"{Colors.WARNING}[WARN] {message}{Colors.ENDC}")


def print_info(message: str):
    """Print an info message"""
    print(f"{Colors.OKBLUE}[INFO] {message}{Colors.ENDC}")


def check_python_version() -> bool:
    """Check Python version"""
    print_section("Python Environment")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print_info(f"Python version: {version_str}")
    print_info(f"Executable: {sys.executable}")
    print_info(f"Platform: {platform.platform()}")

    if version.major >= 3 and version.minor >= 8:
        print_success("Python version is compatible (>= 3.8)")
        return True
    else:
        print_error(f"Python version {version_str} is too old. Requires >= 3.8")
        return False


def check_pytorch() -> Tuple[bool, Dict]:
    """Check PyTorch installation and CUDA availability"""
    print_section("PyTorch Installation")
    results = {
        'installed': False,
        'version': None,
        'cuda_available': False,
        'cuda_version': None,
        'cudnn_version': None,
    }

    try:
        import torch
        results['installed'] = True
        results['version'] = torch.__version__
        print_success(f"PyTorch installed: {torch.__version__}")

        # Check CUDA
        results['cuda_available'] = torch.cuda.is_available()
        if results['cuda_available']:
            results['cuda_version'] = torch.version.cuda
            results['cudnn_version'] = torch.backends.cudnn.version()
            print_success(f"CUDA available: {torch.version.cuda}")
            print_success(f"cuDNN version: {torch.backends.cudnn.version()}")
            print_success(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        else:
            print_error("CUDA is not available")
            print_warning("PyTorch may be CPU-only version")

    except ImportError as e:
        print_error(f"PyTorch not installed: {e}")
        return False, results
    except Exception as e:
        print_error(f"Error checking PyTorch: {e}")
        return False, results

    return results['cuda_available'], results


def check_gpu() -> Tuple[bool, Dict]:
    """Check GPU detection and memory"""
    print_section("GPU Information")
    results = {
        'detected': False,
        'count': 0,
        'devices': [],
    }

    try:
        import torch

        if not torch.cuda.is_available():
            print_error("CUDA not available - cannot detect GPU")
            return False, results

        gpu_count = torch.cuda.device_count()
        results['count'] = gpu_count
        results['detected'] = gpu_count > 0

        print_success(f"GPUs detected: {gpu_count}")

        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            total_memory_gb = total_memory / (1024 ** 3)

            device_info = {
                'id': i,
                'name': device_name,
                'capability': device_capability,
                'total_memory_gb': total_memory_gb,
            }
            results['devices'].append(device_info)

            print_info(f"\nGPU {i}:")
            print_info(f"  Name: {device_name}")
            print_info(f"  Compute Capability: {device_capability}")
            print_info(f"  Total Memory: {total_memory_gb:.2f} GB")

            # Test memory allocation
            try:
                # Allocate a small tensor to test
                test_tensor = torch.zeros(1000, 1000, device=f'cuda:{i}')
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                free = total_memory_gb - allocated

                print_info(f"  Allocated Memory: {allocated:.2f} GB")
                print_info(f"  Reserved Memory: {reserved:.2f} GB")
                print_info(f"  Free Memory: {free:.2f} GB")
                print_success(f"  Memory allocation test: PASSED")

                # Clean up
                del test_tensor
                torch.cuda.empty_cache()

            except Exception as e:
                print_error(f"  Memory allocation test failed: {e}")

        return True, results

    except ImportError:
        print_error("PyTorch not available - cannot check GPU")
        return False, results
    except Exception as e:
        print_error(f"Error checking GPU: {e}")
        return False, results


def check_unsloth() -> bool:
    """Check Unsloth installation"""
    print_section("Unsloth Installation")

    try:
        import unsloth
        version = getattr(unsloth, '__version__', 'unknown')
        print_success(f"Unsloth installed: {version}")

        # Try to import key components
        try:
            from unsloth import FastLanguageModel
            print_success("FastLanguageModel available")
        except ImportError as e:
            print_warning(f"FastLanguageModel not available: {e}")

        return True

    except ImportError as e:
        print_error(f"Unsloth not installed: {e}")
        print_info("Install with: pip install unsloth")
        return False
    except Exception as e:
        print_error(f"Error checking Unsloth: {e}")
        return False


def check_dependencies() -> Dict[str, bool]:
    """Check required training dependencies"""
    print_section("Training Dependencies")

    dependencies = {
        'transformers': 'Hugging Face Transformers',
        'accelerate': 'Accelerate (distributed training)',
        'bitsandbytes': 'BitsAndBytes (quantization)',
        'datasets': 'Hugging Face Datasets',
        'peft': 'PEFT (Parameter-Efficient Fine-Tuning)',
        'trl': 'TRL (Transformer Reinforcement Learning)',
        'wandb': 'Weights & Biases (optional)',
        'xformers': 'xFormers (memory-efficient attention)',
        'triton': 'Triton (GPU kernels)',
        'einops': 'Einops (tensor operations)',
        'scipy': 'SciPy (scientific computing)',
        'tensorboard': 'TensorBoard (visualization)',
        'sentencepiece': 'SentencePiece (tokenization)',
    }

    results = {}

    for package, description in dependencies.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print_success(f"{package} ({description}): {version}")
            results[package] = True
        except ImportError:
            if package in ['wandb', 'xformers', 'triton']:
                print_warning(f"{package} ({description}): NOT INSTALLED (optional)")
                results[package] = False
            else:
                print_error(f"{package} ({description}): NOT INSTALLED")
                results[package] = False
        except Exception as e:
            print_error(f"{package}: Error checking - {e}")
            results[package] = False

    return results


def check_nvidia_smi() -> bool:
    """Check nvidia-smi availability"""
    print_section("NVIDIA SMI")

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            print_success("nvidia-smi is available")
            print_info("\nGPU Information from nvidia-smi:")
            for line in result.stdout.strip().split('\n'):
                print_info(f"  {line}")
            return True
        else:
            print_warning("nvidia-smi command failed")
            return False

    except FileNotFoundError:
        print_warning("nvidia-smi not found in PATH")
        return False
    except Exception as e:
        print_error(f"Error running nvidia-smi: {e}")
        return False


def check_training_directories(base_path: str = "D:\\training") -> bool:
    """Check training directory structure"""
    print_section("Training Directory Structure")

    base = Path(base_path)
    required_dirs = ['datasets', 'models', 'checkpoints', 'logs', 'configs', 'outputs']

    all_exist = True

    if not base.exists():
        print_error(f"Base directory does not exist: {base}")
        return False

    print_success(f"Base directory exists: {base}")

    for dir_name in required_dirs:
        dir_path = base / dir_name
        if dir_path.exists():
            print_success(f"  {dir_name}/: exists")
        else:
            print_warning(f"  {dir_name}/: MISSING")
            all_exist = False

    return all_exist


def run_quick_test() -> bool:
    """Run a quick PyTorch + CUDA test"""
    print_section("Quick Integration Test")

    try:
        import torch

        if not torch.cuda.is_available():
            print_warning("Skipping CUDA test - CUDA not available")
            return False

        # Create a simple tensor operation on GPU
        print_info("Creating tensor on GPU...")
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        print_info("Performing matrix multiplication...")
        z = torch.mm(x, y)

        print_info("Moving result to CPU...")
        result = z.cpu()

        print_success("GPU computation test: PASSED")
        print_info(f"  Result shape: {result.shape}")
        print_info(f"  Result mean: {result.mean().item():.4f}")

        # Clean up
        del x, y, z, result
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print_error(f"GPU computation test failed: {e}")
        return False


def main():
    """Main validation function"""
    print_header("PyTorch/Unsloth Training Environment Validation")
    print_info("Target: medical-mechanica (Windows 11 Pro, RTX 5060 Ti 16GB)")
    print_info(f"Validation started: {platform.node()}")

    results = {
        'python': False,
        'pytorch': False,
        'gpu': False,
        'unsloth': False,
        'nvidia_smi': False,
        'directories': False,
        'quick_test': False,
    }

    # Run all checks
    results['python'] = check_python_version()
    results['pytorch'], pytorch_info = check_pytorch()
    results['gpu'], gpu_info = check_gpu()
    results['unsloth'] = check_unsloth()
    dependency_results = check_dependencies()
    results['nvidia_smi'] = check_nvidia_smi()
    results['directories'] = check_training_directories()

    if results['pytorch'] and results['gpu']:
        results['quick_test'] = run_quick_test()

    # Print summary
    print_header("Validation Summary")

    print_info(f"Python Environment: {'PASS' if results['python'] else 'FAIL'}")
    print_info(f"PyTorch Installation: {'PASS' if results['pytorch'] else 'FAIL'}")
    print_info(f"GPU Detection: {'PASS' if results['gpu'] else 'FAIL'}")
    print_info(f"Unsloth Installation: {'PASS' if results['unsloth'] else 'FAIL'}")
    print_info(f"NVIDIA SMI: {'PASS' if results['nvidia_smi'] else 'FAIL'}")
    print_info(f"Directory Structure: {'PASS' if results['directories'] else 'FAIL'}")
    print_info(f"Quick Integration Test: {'PASS' if results['quick_test'] else 'FAIL'}")

    # Count dependencies
    deps_installed = sum(1 for v in dependency_results.values() if v)
    deps_total = len(dependency_results)
    print_info(f"Dependencies: {deps_installed}/{deps_total} installed")

    # Overall status
    critical_checks = ['python', 'pytorch', 'gpu']
    all_critical_pass = all(results[check] for check in critical_checks)

    print()
    if all_critical_pass:
        print_success("Environment validation PASSED - Ready for training!")
        return 0
    else:
        print_error("Environment validation FAILED - Please fix the issues above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
