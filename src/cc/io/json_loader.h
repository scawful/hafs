// SIMD-accelerated JSON loading for embedding files
// Uses simdjson for 5-10x faster parsing than standard JSON libraries

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace hafs {

// Result of loading an embedding file
struct EmbeddingFileResult {
  std::vector<std::string> ids;
  std::vector<std::vector<float>> embeddings;
  size_t dimension;
  bool success;
  std::string error;
};

// Result of loading a batch of embedding files
struct EmbeddingBatchResult {
  std::vector<std::string> ids;
  std::vector<std::vector<float>> embeddings;
  size_t dimension;
  size_t files_loaded;
  size_t files_failed;
  std::vector<std::string> errors;
};

// Load embeddings from a single JSON file
// Expected format: {"embeddings": [{"id": "...", "vector": [...]}, ...]}
// or: [{"id": "...", "embedding": [...]}, ...]
EmbeddingFileResult LoadEmbeddingFile(const std::string& path);

// Load embeddings from multiple JSON files
EmbeddingBatchResult LoadEmbeddingFiles(const std::vector<std::string>& paths);

// Load embeddings from a directory (all .json files)
EmbeddingBatchResult LoadEmbeddingsFromDirectory(const std::string& dir_path);

// Register Python bindings
void RegisterIOBindings(py::module_& m);

}  // namespace hafs
