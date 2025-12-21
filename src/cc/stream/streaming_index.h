// Thread-safe streaming embedding index with real-time updates
// Uses std::shared_mutex for concurrent read/exclusive write access

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef HAFS_HAS_HNSW
#include "hnswlib/hnswlib.h"
#endif

namespace py = pybind11;

namespace hafs {

// Thread-safe streaming index with incremental updates
class StreamingIndex {
 public:
  // Create a new streaming index
  // dim: embedding dimension
  // max_elements: maximum capacity (can be resized)
  // ef_construction: HNSW construction parameter (higher = better quality)
  // M: HNSW connections per node (higher = better recall, more memory)
  StreamingIndex(size_t dim, size_t max_elements = 100000,
                 size_t ef_construction = 200, size_t M = 16);

  ~StreamingIndex();

  // Add a single embedding with ID
  // Returns true if added, false if ID already exists
  bool Add(const std::string& id, const float* embedding);

  // Add multiple embeddings
  // Returns number of embeddings added (skips existing IDs)
  size_t AddBatch(const std::vector<std::string>& ids, const float* embeddings);

  // Remove an embedding by ID
  // Returns true if removed, false if not found
  bool Remove(const std::string& id);

  // Update an embedding (removes old, adds new)
  // Returns true if updated, false if ID not found
  bool Update(const std::string& id, const float* embedding);

  // Search for k nearest neighbors
  // Returns (labels, distances) where labels are internal IDs
  void Search(const float* query, size_t k, uint64_t* labels,
              float* distances) const;

  // Search and return string IDs
  std::vector<std::pair<std::string, float>> SearchWithIds(const float* query,
                                                           size_t k) const;

  // Get embedding by ID (returns nullptr if not found)
  const float* Get(const std::string& id) const;

  // Check if ID exists
  bool Contains(const std::string& id) const;

  // Get current size (number of active embeddings)
  size_t Size() const;

  // Get capacity
  size_t Capacity() const { return max_elements_; }

  // Get dimension
  size_t Dimension() const { return dim_; }

  // Resize capacity (creates new index, rebuilds)
  void Resize(size_t new_max_elements);

  // Compact the index (remove deleted entries, rebuild)
  void Compact();

  // Save to file
  void Save(const std::string& path) const;

  // Load from file
  void Load(const std::string& path);

  // Get statistics
  struct Stats {
    size_t total_added;
    size_t total_removed;
    size_t active_count;
    size_t deleted_count;
    size_t capacity;
    size_t dimension;
  };
  Stats GetStats() const;

 private:
  size_t dim_;
  size_t max_elements_;
  size_t ef_construction_;
  size_t M_;

  // Thread safety
  mutable std::shared_mutex mutex_;

  // ID mapping: string ID -> internal label
  std::unordered_map<std::string, uint64_t> id_to_label_;
  std::unordered_map<uint64_t, std::string> label_to_id_;

  // Deleted labels (for lazy deletion)
  std::unordered_set<uint64_t> deleted_labels_;

  // Next available label
  std::atomic<uint64_t> next_label_{0};

  // Stats
  std::atomic<size_t> total_added_{0};
  std::atomic<size_t> total_removed_{0};

#ifdef HAFS_HAS_HNSW
  std::unique_ptr<hnswlib::InnerProductSpace> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
#endif

  // Internal helpers
  void InitIndex();
  void RebuildIndex(const std::vector<std::pair<std::string, std::vector<float>>>& embeddings);
};

// Register Python bindings
void RegisterStreamingBindings(py::module_& m);

}  // namespace hafs
