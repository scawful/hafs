# Native Module

High-performance vector operations for hafs, implemented in C++ with ARM NEON SIMD optimization and Python bindings via pybind11.

## Overview

The native module provides optimized implementations of:
- **Similarity**: Cosine similarity with ARM NEON SIMD (47x speedup)
- **HNSW Index**: Approximate nearest neighbor search (O(log n) vs O(n))
- **Quantization**: Int8/Float16 embeddings (4x/2x memory reduction)
- **simdjson Loading**: SIMD-accelerated JSON parsing (5-10x faster)
- **Streaming Index**: Thread-safe real-time embedding updates
- **EmbeddingGemma**: Local embedding generation via Ollama

### Performance Summary

| Operation | NumPy | Native | Speedup |
|-----------|-------|--------|---------|
| Cosine similarity (768d) | 2.4 μs | 1.5 μs | **1.6x** |
| Cosine vs pure Python | 73 μs | 1.5 μs | **47x** |
| HNSW search (10K vectors) | O(n) ~100ms | O(log n) ~0.7ms | **140x** |
| Int8 quantize (768d) | - | 1.1 μs | - |
| Int8 cosine (768d) | - | 1.1 μs | - |
| JSON load (1MB file) | ~50ms | ~5ms | **10x** |
| Streaming add/update | N/A | O(log n) | - |

## Architecture

```
src/cc/
├── CMakeLists.txt              # Root build configuration
├── common/
│   ├── simd_utils.h            # ARM NEON float32 intrinsics
│   └── types.h                 # Common types (QuantizationParams)
├── similarity/
│   ├── similarity.h            # Cosine, L2, dot product API
│   ├── similarity.cc           # Core implementations
│   └── batch_ops.cc            # Batch matrix operations
├── index/
│   ├── hnsw_index.h            # HNSW ANN index API
│   └── hnsw_index.cc           # hnswlib wrapper
├── quantize/
│   ├── quantize.h              # Quantization API
│   ├── quantize.cc             # Float32 <-> Int8/Float16
│   └── int8_ops.h              # ARM NEON int8 SIMD (sdot)
├── io/
│   ├── json_loader.h           # simdjson loading API
│   └── json_loader.cc          # SIMD-accelerated JSON parsing
├── stream/
│   ├── streaming_index.h       # Thread-safe streaming index API
│   └── streaming_index.cc      # Real-time embedding updates
└── bindings/
    └── main_bindings.cc        # Unified pybind11 module

src/hafs/core/
├── similarity.py               # Cosine similarity wrapper
├── index.py                    # HNSW index wrapper
├── quantize.py                 # Quantization wrapper
├── io.py                       # JSON loading wrapper
├── streaming_index.py          # Streaming index wrapper
└── embed.py                    # EmbeddingGemma integration
```

## Similarity Operations

### Usage

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

### SIMD Implementation

ARM NEON processes 4 floats per instruction:

```cpp
// src/cc/common/simd_utils.h
inline float DotProductNeon(const float* a, const float* b, size_t n) {
  float32x4_t sum = vdupq_n_f32(0.0f);
  for (size_t i = 0; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum = vmlaq_f32(sum, va, vb);  // Fused multiply-add
  }
  return vaddvq_f32(sum);
}
```

## HNSW Index

Hierarchical Navigable Small World graph for approximate nearest neighbor search.

### Usage

```python
from hafs.core.index import HNSWIndex, get_index_backend_info
import numpy as np

print(get_index_backend_info())
# {'native': 'yes', 'hnsw': 'yes'}

# Create index
dim = 768
index = HNSWIndex(dim, max_elements=10000)

# Build from data
data = np.random.randn(1000, dim).astype(np.float32)
index.build(data)

# Search
query = data[0]
labels, scores = index.search(query, k=10)

# Persistence
index.save("embeddings.hnsw")
index.load("embeddings.hnsw")
```

### Performance

| Corpus Size | Brute Force | HNSW | Speedup |
|-------------|-------------|------|---------|
| 1,000 | ~10ms | ~0.1ms | 100x |
| 10,000 | ~100ms | ~0.7ms | 140x |
| 100,000 | ~1s | ~2ms | 500x |

## Quantized Embeddings

Reduce memory usage and improve cache utilization.

### Usage

```python
from hafs.core.quantize import (
    quantize_to_int8,
    dequantize_from_int8,
    quantize_to_f16,
    dequantize_from_f16,
    cosine_similarity_int8,
    get_quantize_backend_info,
)

print(get_quantize_backend_info())
# {'native': 'yes', 'quantize': 'yes'}

# Int8 quantization (4x memory reduction)
embedding = np.random.randn(768).astype(np.float32)
quantized, scale, zero_point = quantize_to_int8(embedding)
restored = dequantize_from_int8(quantized, scale, zero_point)

# Memory: 3072 bytes -> 768 bytes

# Float16 (2x reduction, near-lossless)
f16 = quantize_to_f16(embedding)
f32 = dequantize_from_f16(f16)

# Quantized cosine similarity
a_q, scale_a, _ = quantize_to_int8(a)
b_q, scale_b, _ = quantize_to_int8(b)
similarity = cosine_similarity_int8(a_q, b_q, scale_a, scale_b)
```

### Precision

| Format | Size (768d) | MSE vs float32 |
|--------|-------------|----------------|
| float32 | 3072 bytes | 0 |
| float16 | 1536 bytes | ~1e-8 |
| int8 | 768 bytes | ~5e-5 |

### Int8 SIMD

Uses ARM NEON sdot instruction for 16 int8 multiplies per instruction:

```cpp
// src/cc/quantize/int8_ops.h
inline int32_t DotProductInt8Neon(const int8_t* a, const int8_t* b, size_t n) {
  int32x4_t sum = vdupq_n_s32(0);
  for (size_t i = 0; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    sum = vdotq_s32(sum, va, vb);  // ARMv8.2+ dot product
  }
  return vaddvq_s32(sum);
}
```

## EmbeddingGemma Integration

Local embedding generation using Google's EmbeddingGemma model via Ollama.

### Setup

```bash
# Install embeddinggemma via Ollama
ollama pull embeddinggemma
```

### Usage

```python
from hafs.core.embed import EmbeddingGemma, check_embeddinggemma_available
import asyncio

async def generate_embeddings():
    # Check availability
    available = await check_embeddinggemma_available()
    print(f"EmbeddingGemma available: {available}")

    # Initialize
    gemma = EmbeddingGemma()
    await gemma.initialize()
    print(f"Model: {gemma.model}")
    print(f"Is EmbeddingGemma: {gemma.is_embeddinggemma}")

    # Generate embedding
    embedding = await gemma.embed("Hello, world!")
    print(f"Embedding shape: {embedding.shape}")  # (768,)

    # Batch embeddings
    texts = ["Hello", "World", "Test"]
    embeddings = await gemma.embed_batch(texts)
    print(f"Batch count: {len(embeddings)}")

asyncio.run(generate_embeddings())
```

### Features

- **768-dim embeddings**: Compatible with existing hafs embedding infrastructure
- **100+ languages**: Multilingual support built-in
- **Local execution**: No API calls, runs on your machine
- **Automatic fallback**: Falls back to nomic-embed-text if unavailable

## Configuration

All native features are **optional** and configurable via `hafs.toml`:

```toml
[native]
enabled = true                  # Master switch for all native acceleration

# Individual feature toggles
similarity = true               # SIMD-accelerated cosine similarity
hnsw_index = true               # HNSW approximate nearest neighbor index
quantization = true             # Int8/Float16 embedding quantization
simdjson = true                 # SIMD-accelerated JSON parsing
streaming_index = true          # Thread-safe real-time embedding index

# Embedding model preferences
embedding_model = "embeddinggemma"
embedding_fallback = "nomic-embed-text"
```

### Checking Configuration

```python
from hafs.core.native_config import native_config

# Check overall status
print(native_config.module_available)  # Is C++ module built?
print(native_config.enabled)           # Is native enabled in config?

# Check individual features
if native_config.use_similarity:
    print("Using SIMD similarity")
if native_config.use_hnsw:
    print("Using HNSW index")

# Get full status
print(native_config.get_status())
```

### Fallback Behavior

When native features are unavailable or disabled:
- **Similarity**: Falls back to NumPy vectorized operations
- **HNSW Index**: Falls back to brute-force search
- **Quantization**: Falls back to NumPy casting
- **simdjson**: Falls back to Python stdlib json
- **Streaming Index**: Falls back to dict-based index with NumPy search

## Build System

### Requirements

- CMake >= 3.15
- pybind11 >= 2.11
- scikit-build-core >= 0.5
- C++17 compiler (clang++ from Xcode CLT)
- hnswlib (fetched automatically via CMake)
- simdjson (optional, `brew install simdjson` for fast JSON loading)

### Building

```bash
# Install build dependencies
pip install scikit-build-core pybind11

# Build and install
pip install -e .

# Verify native module
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
cmake.source-dir = "src/cc"

[tool.scikit-build.cmake.define]
CMAKE_OSX_ARCHITECTURES = "arm64"
```

## simdjson Loading

SIMD-accelerated JSON parsing for 5-10x faster embedding file loading.

### Usage

```python
from hafs.core.io import (
    load_embedding_file,
    load_embeddings_from_directory,
    get_io_backend_info,
)

print(get_io_backend_info())
# {'native': 'yes', 'simdjson': 'yes'}

# Load single file
# Supports: {"embeddings": [{"id": "...", "vector": [...]}, ...]}
# Or: [{"id": "...", "embedding": [...]}, ...]
ids, embeddings = load_embedding_file("embeddings.json")

# Load all JSON files from directory
ids, embeddings, stats = load_embeddings_from_directory("./embeddings/")
print(stats)
# {'files_loaded': 10, 'files_failed': 0, 'total_embeddings': 5000, 'dimension': 768}
```

### Requirements

Install simdjson via Homebrew:
```bash
brew install simdjson
```

## Streaming Index

Thread-safe streaming embedding index with real-time updates.

### Usage

```python
from hafs.core.streaming_index import StreamingIndex, get_streaming_backend_info
import numpy as np

print(get_streaming_backend_info())
# {'native': 'yes', 'streaming': 'yes'}

# Create index
index = StreamingIndex(dim=768, max_elements=100000)

# Add embeddings
index.add("doc1", embedding1)
index.add_batch(["doc2", "doc3", "doc4"], embeddings_array)

# Search (returns IDs and similarity scores)
ids, scores = index.search(query, k=10)

# Update existing embedding
index.update("doc1", new_embedding)

# Remove embedding
index.remove("doc2")

# Check existence
exists = index.contains("doc1")

# Get statistics
stats = index.get_stats()
# {'total_added': 4, 'total_removed': 1, 'active_count': 3, 'deleted_count': 1, ...}

# Compact (remove deleted entries, rebuild index)
index.compact()

# Persistence
index.save("my_index")  # Creates my_index.hnsw and my_index.ids
index.load("my_index")
```

### Features

- **Thread-safe**: Concurrent reads, exclusive writes via std::shared_mutex
- **O(log n) operations**: Add, remove, update, search
- **Lazy deletion**: Removed entries are marked, compacted on demand
- **Persistence**: Save/load index state to disk

## Future Expansions

### Metal GPU Acceleration
For large batch operations on Apple Silicon.

### Native EmbeddingGemma
Direct GGUF model inference via llama.cpp for maximum performance.

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

### HNSW not available

```python
>>> from hafs.core.index import get_index_backend_info
>>> get_index_backend_info()
{'native': 'yes', 'hnsw': 'no'}
```

This means hnswlib wasn't fetched. Check CMake output.

### EmbeddingGemma returns 500 error

The model may not be pulled. Run:
```bash
ollama pull embeddinggemma
```
