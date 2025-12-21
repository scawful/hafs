#ifndef HAFS_HNSW_INDEX_H_
#define HAFS_HNSW_INDEX_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hafs {

// Distance type for HNSW index
enum class DistanceType { L2 = 0, InnerProduct = 1, Cosine = 2 };

// =============================================================================
// HNSW Index C++ API
// =============================================================================

class HNSWIndex {
 public:
  // Create an index for vectors of given dimension
  // max_elements: Maximum number of vectors to store
  // M: Number of bi-directional links per element (default: 16)
  // ef_construction: Size of dynamic list for construction (default: 200)
  HNSWIndex(size_t dim, size_t max_elements,
            DistanceType distance = DistanceType::Cosine,
            size_t M = 16, size_t ef_construction = 200);

  ~HNSWIndex();

  // Disable copy, allow move
  HNSWIndex(const HNSWIndex&) = delete;
  HNSWIndex& operator=(const HNSWIndex&) = delete;
  HNSWIndex(HNSWIndex&&) noexcept;
  HNSWIndex& operator=(HNSWIndex&&) noexcept;

  // Build index from data array (n_elements x dim)
  void Build(const float* data, size_t n_elements);

  // Add single vector with label
  void Add(const float* data, uint64_t label);

  // Add batch of vectors with labels
  void AddBatch(const float* data, const uint64_t* labels, size_t n);

  // Search for k nearest neighbors
  // Returns vectors of (label, distance) pairs sorted by distance
  void Search(const float* query, size_t k,
              uint64_t* labels, float* distances) const;

  // Batch search
  void SearchBatch(const float* queries, size_t n_queries, size_t k,
                   uint64_t* labels, float* distances) const;

  // Persistence
  void Save(const std::string& path) const;
  void Load(const std::string& path);

  // Properties
  size_t Size() const;
  size_t Dimension() const;
  size_t MaxElements() const;

  // Set ef parameter for search (higher = more accurate but slower)
  void SetEf(size_t ef);
  size_t GetEf() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Python Bindings
// =============================================================================

// Python wrapper class
class PyHNSWIndex {
 public:
  PyHNSWIndex(size_t dim, size_t max_elements = 100000,
              int distance = 2,  // Cosine
              size_t M = 16, size_t ef_construction = 200);

  void Build(py::array_t<float> data);
  void Add(py::array_t<float> data, uint64_t label);

  std::tuple<py::array_t<uint64_t>, py::array_t<float>> Search(
      py::array_t<float> query, size_t k);

  std::tuple<py::array_t<uint64_t>, py::array_t<float>> SearchBatch(
      py::array_t<float> queries, size_t k);

  void Save(const std::string& path);
  void Load(const std::string& path);

  size_t Size() const;
  size_t Dimension() const;
  void SetEf(size_t ef);

 private:
  std::unique_ptr<HNSWIndex> index_;
};

// Register HNSW bindings on a pybind11 module
void RegisterHNSWBindings(py::module& m);

}  // namespace hafs

#endif  // HAFS_HNSW_INDEX_H_
