#!/usr/bin/env python3
"""
Quick GPU Test Script
Runs a simple test to verify PyTorch + CUDA + GPU are working correctly
Usage: python quick_gpu_test.py
"""

import sys


def main():
    print("=" * 80)
    print("Quick GPU Test")
    print("=" * 80)

    # Test 1: Import PyTorch
    print("\n[1/5] Importing PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return 1

    # Test 2: Check CUDA availability
    print("\n[2/5] Checking CUDA availability...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available (version {torch.version.cuda})")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("✗ CUDA is NOT available")
        print("  PyTorch may be CPU-only version")
        return 1

    # Test 3: Detect GPU
    print("\n[3/5] Detecting GPU...")
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        print(f"✓ Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
    else:
        print("✗ No GPUs detected")
        return 1

    # Test 4: Allocate GPU memory
    print("\n[4/5] Testing GPU memory allocation...")
    try:
        # Allocate 100MB tensor
        x = torch.randn(10000, 1000, device='cuda')
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        print(f"✓ Successfully allocated {allocated:.1f} MB on GPU")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Failed to allocate GPU memory: {e}")
        return 1

    # Test 5: Run GPU computation
    print("\n[5/5] Running GPU computation test...")
    try:
        import time

        # Create two large matrices
        size = 5000
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')

        # Warm up
        _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # Time the computation
        start = time.time()
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        print(f"✓ Matrix multiplication ({size}x{size}) completed in {elapsed:.3f} seconds")
        print(f"  Performance: {(2 * size**3) / elapsed / 1e9:.2f} GFLOPS")

        # Clean up
        del a, b, c
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        return 1

    # Final summary
    print("\n" + "=" * 80)
    print("GPU Test Results: ALL TESTS PASSED ✓")
    print("=" * 80)

    print("\nGPU is ready for training!")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
