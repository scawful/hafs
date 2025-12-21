# Native Similarity Module

High-performance vector similarity operations for hafs, implemented in C++ with ARM NEON SIMD optimization and Python bindings via pybind11.

## Overview

The native similarity module accelerates embedding-based semantic search by replacing pure Python similarity calculations with optimized C++ code. On Apple Silicon (M1/M2/M3), it uses ARM NEON SIMD instructions to process 4 floats simultaneously.

### Performance

| Operation | Pure Python | NumPy | Native SIMD | Speedup |
|-----------|------------|-------|-------------|---------|
| Single cosine (768d) | 73 μs | 2.4 μs | 1.5 μs | **47x** |
| 1M comparisons | ~73s | ~2.4s | ~1.5s | **47x** |

## Architecture

```
cc/
├── CMakeLists.txt              # Build configuration
├── include/hafs/
│   ├── similarity.h            # Public API declarations
│   └── simd_utils.h            # ARM NEON intrinsics
└── src/
    ├── similarity.cc           # Core algorithms
    ├── batch_ops.cc            # Batch matrix operations
    └── bindings.cc             # pybind11 Python bindings

src/hafs/core/
└── similarity.py               # Python wrapper with NumPy fallback
```

## How It Works

### 1. SIMD Vectorization

The module uses ARM NEON intrinsics to process vectors in parallel:

```cpp
// cc/include/hafs/simd_utils.h
inline float DotProductNeon(const float* a, const float* b, size_t n) {
  float32x4_t sum = vdupq_n_f32(0.0f);  // Initialize 4-wide sum

  for (size_t i = 0; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);   // Load 4 floats from a
    float32x4_t vb = vld1q_f32(b + i);   // Load 4 floats from b
    sum = vmlaq_f32(sum, va, vb);        // Fused multiply-add
  }

  return vaddvq_f32(sum);  // Horizontal sum of 4 lanes
}
```

This processes 4 dimensions per instruction instead of 1, giving ~4x speedup at the hardware level.

### 2. Cosine Similarity

```cpp
// cc/src/similarity.cc
float CosineSimilarity(const float* a, const float* b, size_t n) {
  float dot = DotProductNeon(a, b, n);
  float norm_a = std::sqrt(SquaredNormNeon(a, n));
  float norm_b = std::sqrt(SquaredNormNeon(b, n));

  if (norm_a < 1e-8f || norm_b < 1e-8f) return 0.0f;
  return dot / (norm_a * norm_b);
}
```

### 3. Python Bindings

pybind11 provides zero-copy access to NumPy arrays:

```cpp
// cc/src/bindings.cc
float PyCosineSimilarity(py::array_t<float> a, py::array_t<float> b) {
  auto buf_a = a.request();  // Get buffer info (pointer, shape, strides)
  auto buf_b = b.request();

  return CosineSimilarity(
    static_cast<float*>(buf_a.ptr),
    static_cast<float*>(buf_b.ptr),
    static_cast<size_t>(buf_a.size)
  );
}
```

### 4. Automatic Fallback

The Python wrapper gracefully degrades to NumPy if the native module isn't available:

```python
# src/hafs/core/similarity.py
try:
    from hafs.core._similarity import cosine_similarity as _native
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False

def cosine_similarity(a, b) -> float:
    if _NATIVE_AVAILABLE:
        return _native(np.ascontiguousarray(a, dtype=np.float32), ...)
    # NumPy fallback
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

## Build System

### Requirements

- CMake >= 3.15
- pybind11 >= 2.11
- scikit-build-core >= 0.5
- C++17 compiler (clang++ from Xcode CLT)

### Building

```bash
# Install build dependencies
pip install scikit-build-core pybind11

# Build and install
pip install -e .

# Verify native module loaded
python -c "from hafs.core.similarity import get_backend_info; print(get_backend_info())"
# {'native': 'yes', 'simd': 'ARM NEON', 'blas': 'NEON SIMD'}
```

### pyproject.toml Configuration

```toml
[build-system]
requires = ["scikit-build-core>=0.5", "pybind11>=2.11"]
build-backend = "scikit_build_core.build"

[tool.scikit-build]
cmake.minimum-version = "3.15"
cmake.source-dir = "cc"

[tool.scikit-build.cmake.define]
CMAKE_OSX_ARCHITECTURES = "arm64"
```

## Usage

### Basic Usage

```python
from hafs.core.similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    top_k_similar,
    get_backend_info,
)

# Check backend
print(get_backend_info())
# {'native': 'yes', 'simd': 'ARM NEON', 'blas': 'NEON SIMD'}

# Single pair
score = cosine_similarity(embedding_a, embedding_b)

# Batch (returns n_queries x n_corpus matrix)
scores = cosine_similarity_batch(query_embeddings, corpus_embeddings)

# Top-k search
indices, scores = top_k_similar(query, corpus, k=10)
```

### Integration Points

The module is integrated into:

1. **`EmbeddingService.semantic_cross_reference()`** - Cross-project symbol matching
2. **`HistoryEmbeddingIndex.search()`** - Semantic search over command history

## Future Expansions

### 1. Apple Accelerate BLAS Integration

For large batch operations, Apple's Accelerate framework provides highly optimized BLAS:

```cpp
#include <Accelerate/Accelerate.h>

void BatchCosineMatrixAccelerate(const float* A, const float* B,
                                  float* C, int m, int n, int k) {
  // Normalize matrices first, then:
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
}
```

**Status:** Deferred due to SDK compatibility issues with macOS 26.0 beta headers.

### 2. Approximate Nearest Neighbor (ANN) Index

For large corpora (100K+ embeddings), exact k-NN becomes slow. Options:

| Algorithm | Library | Speedup | Trade-off |
|-----------|---------|---------|-----------|
| HNSW | hnswlib | 100-1000x | ~95% recall |
| IVF-PQ | faiss | 1000x+ | ~90% recall |
| LSH | custom | 50x | ~85% recall |

```cpp
// Future: cc/include/hafs/ann_index.h
class HNSWIndex {
 public:
  void Build(const float* embeddings, size_t n, size_t dim);
  void Search(const float* query, size_t k, int* indices, float* distances);
};
```

### 3. Quantized Embeddings

Reduce memory and improve cache utilization:

| Precision | Size (768d) | Relative Speed |
|-----------|-------------|----------------|
| float32 | 3072 bytes | 1.0x |
| float16 | 1536 bytes | 1.5-2x |
| int8 | 768 bytes | 2-4x |

```cpp
// Future: int8 quantized dot product
int32_t DotProductInt8(const int8_t* a, const int8_t* b, size_t n) {
  // Use ARM NEON sdot instruction for 4x int8 multiply-accumulate
  int32x4_t sum = vdupq_n_s32(0);
  for (size_t i = 0; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    sum = vdotq_s32(sum, va, vb);  // ARM dot product instruction
  }
  return vaddvq_s32(sum);
}
```

### 4. GPU Acceleration (Metal)

For batch operations on Apple Silicon:

```cpp
// Future: cc/src/metal_similarity.mm
#import <Metal/Metal.h>

class MetalSimilarity {
  id<MTLDevice> device_;
  id<MTLComputePipelineState> pipeline_;

 public:
  void BatchCosine(const float* queries, const float* corpus,
                   float* output, size_t n_q, size_t n_c, size_t dim);
};
```

### 5. Streaming/Incremental Updates

Support real-time embedding updates without full reindex:

```cpp
// Future: cc/include/hafs/streaming_index.h
class StreamingIndex {
 public:
  void Add(const std::string& id, const float* embedding, size_t dim);
  void Remove(const std::string& id);
  void Update(const std::string& id, const float* embedding, size_t dim);

  // Search with consistency guarantees
  void Search(const float* query, size_t k, ...);
};
```

### 6. simdjson for Fast Embedding Loading

Replace stdlib JSON parsing with SIMD-accelerated parsing:

```cpp
#include <simdjson.h>

std::vector<float> LoadEmbedding(const std::string& path) {
  simdjson::dom::parser parser;
  auto doc = parser.load(path);

  std::vector<float> embedding;
  for (auto val : doc["embedding"].get_array()) {
    embedding.push_back(val.get_double());
  }
  return embedding;
}
```

**Expected speedup:** 5-10x for JSON parsing.

### 7. Cross-Platform SIMD

Support x86-64 with AVX2/AVX-512:

```cpp
#if defined(__ARM_NEON)
  #include <arm_neon.h>
  #define HAFS_SIMD_WIDTH 4
#elif defined(__AVX2__)
  #include <immintrin.h>
  #define HAFS_SIMD_WIDTH 8
#elif defined(__AVX512F__)
  #include <immintrin.h>
  #define HAFS_SIMD_WIDTH 16
#else
  #define HAFS_SIMD_WIDTH 1  // Scalar fallback
#endif
```

## Benchmarking

```python
# Run benchmark
python -c "
import numpy as np
import time
from hafs.core.similarity import cosine_similarity, get_backend_info

print('Backend:', get_backend_info())

# Benchmark
dim = 768
iterations = 10000
a = np.random.randn(dim).astype(np.float32)
b = np.random.randn(dim).astype(np.float32)

start = time.perf_counter()
for _ in range(iterations):
    cosine_similarity(a, b)
elapsed = (time.perf_counter() - start) / iterations * 1e6

print(f'Single cosine ({dim}d): {elapsed:.2f} μs')
"
```

## Troubleshooting

### Native module not loading

```python
>>> from hafs.core.similarity import get_backend_info
>>> get_backend_info()
{'native': 'no', 'simd': 'NumPy', 'blas': 'NumPy'}
```

**Solutions:**
1. Rebuild: `pip install -e . --force-reinstall`
2. Check CMake output for errors
3. Verify pybind11 is installed: `pip show pybind11`

### Build fails with SDK errors

If you see errors about `__builtin_verbose_trap` or `visionOS`:

```
error: use of undeclared identifier '__builtin_verbose_trap'
```

This is due to beta SDK headers. The Accelerate framework integration is disabled to avoid this.

### Performance not as expected

1. Ensure float32 input: `np.asarray(x, dtype=np.float32)`
2. Use contiguous arrays: `np.ascontiguousarray(x)`
3. For batch ops, NumPy BLAS may be faster than our SIMD implementation
