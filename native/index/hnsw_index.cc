#include "hnsw_index.h"

#include <stdexcept>
#include <fstream>
#include <cmath>

#ifdef HAFS_HAS_HNSW
#include <hnswlib/hnswlib.h>
#endif

#include "../common/simd_utils.h"

namespace hafs {

// =============================================================================
// Implementation with hnswlib
// =============================================================================

#ifdef HAFS_HAS_HNSW

class HNSWIndex::Impl {
 public:
  Impl(size_t dim, size_t max_elements, DistanceType distance,
       size_t M, size_t ef_construction)
      : dim_(dim), max_elements_(max_elements), distance_(distance) {

    // Create appropriate space
    switch (distance) {
      case DistanceType::L2:
        space_ = std::make_unique<hnswlib::L2Space>(dim);
        break;
      case DistanceType::InnerProduct:
        space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
        break;
      case DistanceType::Cosine:
        // Cosine = 1 - InnerProduct on normalized vectors
        space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);
        break;
    }

    // Create index
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(), max_elements, M, ef_construction);
  }

  void Build(const float* data, size_t n_elements) {
    std::vector<float> normalized;

    for (size_t i = 0; i < n_elements; ++i) {
      const float* vec = data + i * dim_;

      if (distance_ == DistanceType::Cosine) {
        // Normalize for cosine similarity
        normalized.resize(dim_);
        std::copy(vec, vec + dim_, normalized.begin());
        NormalizeVectorNeon(normalized.data(), dim_);
        index_->addPoint(normalized.data(), i);
      } else {
        index_->addPoint(vec, i);
      }
    }
  }

  void Add(const float* data, uint64_t label) {
    if (distance_ == DistanceType::Cosine) {
      std::vector<float> normalized(data, data + dim_);
      NormalizeVectorNeon(normalized.data(), dim_);
      index_->addPoint(normalized.data(), label);
    } else {
      index_->addPoint(data, label);
    }
  }

  void Search(const float* query, size_t k,
              uint64_t* labels, float* distances) const {
    std::vector<float> normalized;
    const float* query_ptr = query;

    if (distance_ == DistanceType::Cosine) {
      normalized.assign(query, query + dim_);
      NormalizeVectorNeon(normalized.data(), dim_);
      query_ptr = normalized.data();
    }

    auto result = index_->searchKnn(query_ptr, k);

    // Capture result size before emptying
    size_t result_size = result.size();

    // Results come as a max-heap, we need to reverse them
    size_t i = result_size;
    while (!result.empty()) {
      --i;
      auto& [dist, label] = result.top();
      labels[i] = label;
      // Convert IP distance to similarity for Cosine
      if (distance_ == DistanceType::Cosine) {
        distances[i] = 1.0f - dist;  // IP on normalized = cosine
      } else {
        distances[i] = dist;
      }
      result.pop();
    }

    // Fill remaining slots if k > results
    for (size_t j = result_size; j < k; ++j) {
      labels[j] = static_cast<uint64_t>(-1);
      distances[j] = 0.0f;
    }
  }

  void Save(const std::string& path) const {
    index_->saveIndex(path);
  }

  void Load(const std::string& path) {
    index_->loadIndex(path, space_.get());
  }

  size_t Size() const { return index_->cur_element_count; }
  size_t Dimension() const { return dim_; }
  size_t MaxElements() const { return max_elements_; }

  void SetEf(size_t ef) { index_->ef_ = ef; }
  size_t GetEf() const { return index_->ef_; }

 private:
  size_t dim_;
  size_t max_elements_;
  DistanceType distance_;
  std::unique_ptr<hnswlib::SpaceInterface<float>> space_;
  std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
};

#else  // Fallback without hnswlib

// Brute-force fallback implementation
class HNSWIndex::Impl {
 public:
  Impl(size_t dim, size_t max_elements, DistanceType distance,
       size_t /*M*/, size_t /*ef_construction*/)
      : dim_(dim), max_elements_(max_elements), distance_(distance) {
    data_.reserve(max_elements * dim);
    labels_.reserve(max_elements);
  }

  void Build(const float* data, size_t n_elements) {
    data_.assign(data, data + n_elements * dim_);
    labels_.resize(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
      labels_[i] = i;
    }
  }

  void Add(const float* data, uint64_t label) {
    data_.insert(data_.end(), data, data + dim_);
    labels_.push_back(label);
  }

  void Search(const float* query, size_t k,
              uint64_t* labels, float* distances) const {
    if (labels_.empty()) {
      for (size_t i = 0; i < k; ++i) {
        labels[i] = static_cast<uint64_t>(-1);
        distances[i] = 0.0f;
      }
      return;
    }

    // Compute all distances
    std::vector<std::pair<float, uint64_t>> scores;
    scores.reserve(labels_.size());

    for (size_t i = 0; i < labels_.size(); ++i) {
      const float* vec = data_.data() + i * dim_;
      float score = 0.0f;

      switch (distance_) {
        case DistanceType::L2:
          score = L2DistanceSquaredNeon(query, vec, dim_);
          break;
        case DistanceType::InnerProduct:
          score = -DotProductNeon(query, vec, dim_);  // Negate for min-heap
          break;
        case DistanceType::Cosine: {
          float dot = DotProductNeon(query, vec, dim_);
          float norm_q = std::sqrt(SquaredNormNeon(query, dim_));
          float norm_v = std::sqrt(SquaredNormNeon(vec, dim_));
          score = (norm_q > 1e-8f && norm_v > 1e-8f)
                  ? dot / (norm_q * norm_v)
                  : 0.0f;
          score = -score;  // Negate for min-heap (higher similarity = lower score)
          break;
        }
      }
      scores.emplace_back(score, labels_[i]);
    }

    // Partial sort to get top-k
    size_t actual_k = std::min(k, scores.size());
    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end());

    for (size_t i = 0; i < actual_k; ++i) {
      labels[i] = scores[i].second;
      // Convert back from negated scores
      if (distance_ == DistanceType::Cosine ||
          distance_ == DistanceType::InnerProduct) {
        distances[i] = -scores[i].first;
      } else {
        distances[i] = scores[i].first;
      }
    }

    for (size_t i = actual_k; i < k; ++i) {
      labels[i] = static_cast<uint64_t>(-1);
      distances[i] = 0.0f;
    }
  }

  void Save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open file for writing");

    // Write header
    size_t size = labels_.size();
    out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(&distance_), sizeof(distance_));

    // Write data
    out.write(reinterpret_cast<const char*>(data_.data()),
              data_.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(labels_.data()),
              labels_.size() * sizeof(uint64_t));
  }

  void Load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open file for reading");

    size_t size;
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    in.read(reinterpret_cast<char*>(&distance_), sizeof(distance_));

    data_.resize(size * dim_);
    labels_.resize(size);

    in.read(reinterpret_cast<char*>(data_.data()),
            data_.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(labels_.data()),
            labels_.size() * sizeof(uint64_t));
  }

  size_t Size() const { return labels_.size(); }
  size_t Dimension() const { return dim_; }
  size_t MaxElements() const { return max_elements_; }

  void SetEf(size_t /*ef*/) {}  // No-op for fallback
  size_t GetEf() const { return 0; }

 private:
  size_t dim_;
  size_t max_elements_;
  DistanceType distance_;
  std::vector<float> data_;
  std::vector<uint64_t> labels_;
};

#endif  // HAFS_HAS_HNSW

// =============================================================================
// HNSWIndex Implementation (delegates to Impl)
// =============================================================================

HNSWIndex::HNSWIndex(size_t dim, size_t max_elements,
                     DistanceType distance, size_t M, size_t ef_construction)
    : impl_(std::make_unique<Impl>(dim, max_elements, distance, M, ef_construction)) {}

HNSWIndex::~HNSWIndex() = default;

HNSWIndex::HNSWIndex(HNSWIndex&&) noexcept = default;
HNSWIndex& HNSWIndex::operator=(HNSWIndex&&) noexcept = default;

void HNSWIndex::Build(const float* data, size_t n_elements) {
  impl_->Build(data, n_elements);
}

void HNSWIndex::Add(const float* data, uint64_t label) {
  impl_->Add(data, label);
}

void HNSWIndex::AddBatch(const float* data, const uint64_t* labels, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    impl_->Add(data + i * Dimension(), labels[i]);
  }
}

void HNSWIndex::Search(const float* query, size_t k,
                       uint64_t* labels, float* distances) const {
  impl_->Search(query, k, labels, distances);
}

void HNSWIndex::SearchBatch(const float* queries, size_t n_queries, size_t k,
                            uint64_t* labels, float* distances) const {
  size_t dim = Dimension();
  for (size_t i = 0; i < n_queries; ++i) {
    impl_->Search(queries + i * dim, k, labels + i * k, distances + i * k);
  }
}

void HNSWIndex::Save(const std::string& path) const {
  impl_->Save(path);
}

void HNSWIndex::Load(const std::string& path) {
  impl_->Load(path);
}

size_t HNSWIndex::Size() const { return impl_->Size(); }
size_t HNSWIndex::Dimension() const { return impl_->Dimension(); }
size_t HNSWIndex::MaxElements() const { return impl_->MaxElements(); }
void HNSWIndex::SetEf(size_t ef) { impl_->SetEf(ef); }
size_t HNSWIndex::GetEf() const { return impl_->GetEf(); }

// =============================================================================
// Python Wrapper
// =============================================================================

PyHNSWIndex::PyHNSWIndex(size_t dim, size_t max_elements,
                         int distance, size_t M, size_t ef_construction)
    : index_(std::make_unique<HNSWIndex>(
          dim, max_elements, static_cast<DistanceType>(distance), M, ef_construction)) {}

void PyHNSWIndex::Build(py::array_t<float> data) {
  auto buf = data.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Data must be 2-dimensional");
  }
  index_->Build(static_cast<float*>(buf.ptr), buf.shape[0]);
}

void PyHNSWIndex::Add(py::array_t<float> data, uint64_t label) {
  auto buf = data.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Data must be 1-dimensional");
  }
  index_->Add(static_cast<float*>(buf.ptr), label);
}

std::tuple<py::array_t<uint64_t>, py::array_t<float>> PyHNSWIndex::Search(
    py::array_t<float> query, size_t k) {
  auto buf = query.request();
  if (buf.ndim != 1) {
    throw std::runtime_error("Query must be 1-dimensional");
  }

  auto labels = py::array_t<uint64_t>(k);
  auto distances = py::array_t<float>(k);

  index_->Search(static_cast<float*>(buf.ptr), k,
                 static_cast<uint64_t*>(labels.request().ptr),
                 static_cast<float*>(distances.request().ptr));

  return std::make_tuple(labels, distances);
}

std::tuple<py::array_t<uint64_t>, py::array_t<float>> PyHNSWIndex::SearchBatch(
    py::array_t<float> queries, size_t k) {
  auto buf = queries.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Queries must be 2-dimensional");
  }

  size_t n_queries = buf.shape[0];
  auto labels = py::array_t<uint64_t>({n_queries, k});
  auto distances = py::array_t<float>({n_queries, k});

  index_->SearchBatch(static_cast<float*>(buf.ptr), n_queries, k,
                      static_cast<uint64_t*>(labels.request().ptr),
                      static_cast<float*>(distances.request().ptr));

  return std::make_tuple(labels, distances);
}

void PyHNSWIndex::Save(const std::string& path) {
  index_->Save(path);
}

void PyHNSWIndex::Load(const std::string& path) {
  index_->Load(path);
}

size_t PyHNSWIndex::Size() const { return index_->Size(); }
size_t PyHNSWIndex::Dimension() const { return index_->Dimension(); }
void PyHNSWIndex::SetEf(size_t ef) { index_->SetEf(ef); }

// =============================================================================
// Register Bindings
// =============================================================================

void RegisterHNSWBindings(py::module& m) {
  py::class_<PyHNSWIndex>(m, "HNSWIndex")
      .def(py::init<size_t, size_t, int, size_t, size_t>(),
           py::arg("dim"),
           py::arg("max_elements") = 100000,
           py::arg("distance") = 2,  // Cosine
           py::arg("M") = 16,
           py::arg("ef_construction") = 200,
           R"doc(
           Create an HNSW index for approximate nearest neighbor search.

           Args:
               dim: Dimension of vectors
               max_elements: Maximum number of vectors to store
               distance: Distance type (0=L2, 1=InnerProduct, 2=Cosine)
               M: Number of bi-directional links per element
               ef_construction: Size of dynamic list for construction
           )doc")
      .def("build", &PyHNSWIndex::Build, py::arg("data"),
           "Build index from 2D numpy array of shape (n, dim)")
      .def("add", &PyHNSWIndex::Add, py::arg("data"), py::arg("label"),
           "Add single vector with label")
      .def("search", &PyHNSWIndex::Search, py::arg("query"), py::arg("k") = 10,
           "Search for k nearest neighbors, returns (labels, distances)")
      .def("search_batch", &PyHNSWIndex::SearchBatch,
           py::arg("queries"), py::arg("k") = 10,
           "Batch search, returns (labels, distances) with shape (n_queries, k)")
      .def("save", &PyHNSWIndex::Save, py::arg("path"),
           "Save index to file")
      .def("load", &PyHNSWIndex::Load, py::arg("path"),
           "Load index from file")
      .def("size", &PyHNSWIndex::Size, "Number of vectors in index")
      .def("dimension", &PyHNSWIndex::Dimension, "Vector dimension")
      .def("set_ef", &PyHNSWIndex::SetEf, py::arg("ef"),
           "Set ef parameter for search (higher = more accurate)");
}

}  // namespace hafs
