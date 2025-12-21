// Thread-safe streaming embedding index implementation

#include "streaming_index.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace hafs {

StreamingIndex::StreamingIndex(size_t dim, size_t max_elements,
                               size_t ef_construction, size_t M)
    : dim_(dim),
      max_elements_(max_elements),
      ef_construction_(ef_construction),
      M_(M) {
  InitIndex();
}

StreamingIndex::~StreamingIndex() = default;

void StreamingIndex::InitIndex() {
#ifdef HAFS_HAS_HNSW
  // Use inner product space (for normalized vectors, this equals cosine)
  space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
  index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space_.get(), max_elements_, M_, ef_construction_);
  index_->setEf(std::max(static_cast<size_t>(50), ef_construction_));
#endif
}

bool StreamingIndex::Add(const std::string& id, const float* embedding) {
  std::unique_lock lock(mutex_);

  // Check if ID already exists
  if (id_to_label_.count(id) > 0) {
    return false;
  }

  uint64_t label = next_label_.fetch_add(1);

  // Check capacity
  if (label >= max_elements_) {
    throw std::runtime_error("Index at capacity. Call Resize() to expand.");
  }

#ifdef HAFS_HAS_HNSW
  index_->addPoint(embedding, label);
#endif

  id_to_label_[id] = label;
  label_to_id_[label] = id;
  total_added_++;

  return true;
}

size_t StreamingIndex::AddBatch(const std::vector<std::string>& ids,
                                const float* embeddings) {
  size_t added = 0;
  for (size_t i = 0; i < ids.size(); i++) {
    if (Add(ids[i], embeddings + i * dim_)) {
      added++;
    }
  }
  return added;
}

bool StreamingIndex::Remove(const std::string& id) {
  std::unique_lock lock(mutex_);

  auto it = id_to_label_.find(id);
  if (it == id_to_label_.end()) {
    return false;
  }

  uint64_t label = it->second;

#ifdef HAFS_HAS_HNSW
  // hnswlib supports marking as deleted
  index_->markDelete(label);
#endif

  deleted_labels_.insert(label);
  id_to_label_.erase(it);
  label_to_id_.erase(label);
  total_removed_++;

  return true;
}

bool StreamingIndex::Update(const std::string& id, const float* embedding) {
  std::unique_lock lock(mutex_);

  auto it = id_to_label_.find(id);
  if (it == id_to_label_.end()) {
    return false;
  }

  uint64_t old_label = it->second;

  // Mark old as deleted
#ifdef HAFS_HAS_HNSW
  index_->markDelete(old_label);
#endif
  deleted_labels_.insert(old_label);
  label_to_id_.erase(old_label);

  // Add new with new label
  uint64_t new_label = next_label_.fetch_add(1);
  if (new_label >= max_elements_) {
    throw std::runtime_error("Index at capacity during update. Call Compact() or Resize().");
  }

#ifdef HAFS_HAS_HNSW
  index_->addPoint(embedding, new_label);
#endif

  id_to_label_[id] = new_label;
  label_to_id_[new_label] = id;

  return true;
}

void StreamingIndex::Search(const float* query, size_t k, uint64_t* labels,
                            float* distances) const {
  std::shared_lock lock(mutex_);

#ifdef HAFS_HAS_HNSW
  auto result = index_->searchKnn(query, k);
  size_t result_size = result.size();
  size_t i = result_size;

  // Initialize outputs
  for (size_t j = 0; j < k; j++) {
    labels[j] = static_cast<uint64_t>(-1);
    distances[j] = -1.0f;
  }

  while (!result.empty()) {
    --i;
    auto& [dist, label] = result.top();
    labels[i] = label;
    distances[i] = 1.0f - dist;  // Convert from inner product to similarity
    result.pop();
  }
#else
  // No HNSW - return empty results
  for (size_t j = 0; j < k; j++) {
    labels[j] = static_cast<uint64_t>(-1);
    distances[j] = -1.0f;
  }
#endif
}

std::vector<std::pair<std::string, float>> StreamingIndex::SearchWithIds(
    const float* query, size_t k) const {
  std::vector<uint64_t> labels(k);
  std::vector<float> distances(k);
  Search(query, k, labels.data(), distances.data());

  std::vector<std::pair<std::string, float>> results;
  std::shared_lock lock(mutex_);

  for (size_t i = 0; i < k; i++) {
    if (labels[i] == static_cast<uint64_t>(-1)) break;

    auto it = label_to_id_.find(labels[i]);
    if (it != label_to_id_.end()) {
      results.emplace_back(it->second, distances[i]);
    }
  }

  return results;
}

const float* StreamingIndex::Get(const std::string& id) const {
  std::shared_lock lock(mutex_);

  auto it = id_to_label_.find(id);
  if (it == id_to_label_.end()) {
    return nullptr;
  }

#ifdef HAFS_HAS_HNSW
  return static_cast<const float*>(index_->getDataByLabel<float>(it->second));
#else
  return nullptr;
#endif
}

bool StreamingIndex::Contains(const std::string& id) const {
  std::shared_lock lock(mutex_);
  return id_to_label_.count(id) > 0;
}

size_t StreamingIndex::Size() const {
  std::shared_lock lock(mutex_);
  return id_to_label_.size();
}

void StreamingIndex::Resize(size_t new_max_elements) {
  if (new_max_elements <= max_elements_) {
    return;
  }

  std::unique_lock lock(mutex_);

  // Collect all current embeddings
  std::vector<std::pair<std::string, std::vector<float>>> embeddings;
  for (const auto& [id, label] : id_to_label_) {
#ifdef HAFS_HAS_HNSW
    const float* data =
        static_cast<const float*>(index_->getDataByLabel<float>(label));
    embeddings.emplace_back(id, std::vector<float>(data, data + dim_));
#endif
  }

  max_elements_ = new_max_elements;
  RebuildIndex(embeddings);
}

void StreamingIndex::Compact() {
  std::unique_lock lock(mutex_);

  if (deleted_labels_.empty()) {
    return;
  }

  // Collect all current embeddings
  std::vector<std::pair<std::string, std::vector<float>>> embeddings;
  for (const auto& [id, label] : id_to_label_) {
#ifdef HAFS_HAS_HNSW
    const float* data =
        static_cast<const float*>(index_->getDataByLabel<float>(label));
    embeddings.emplace_back(id, std::vector<float>(data, data + dim_));
#endif
  }

  RebuildIndex(embeddings);
}

void StreamingIndex::RebuildIndex(
    const std::vector<std::pair<std::string, std::vector<float>>>& embeddings) {
  // Reset state
  id_to_label_.clear();
  label_to_id_.clear();
  deleted_labels_.clear();
  next_label_ = 0;

  // Rebuild HNSW index
  InitIndex();

  // Re-add all embeddings
  for (const auto& [id, vec] : embeddings) {
    uint64_t label = next_label_.fetch_add(1);
#ifdef HAFS_HAS_HNSW
    index_->addPoint(vec.data(), label);
#endif
    id_to_label_[id] = label;
    label_to_id_[label] = id;
  }
}

void StreamingIndex::Save(const std::string& path) const {
  std::shared_lock lock(mutex_);

#ifdef HAFS_HAS_HNSW
  // Save HNSW index
  index_->saveIndex(path + ".hnsw");

  // Save ID mappings
  std::ofstream ofs(path + ".ids", std::ios::binary);
  size_t count = id_to_label_.size();
  ofs.write(reinterpret_cast<const char*>(&count), sizeof(count));
  ofs.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
  ofs.write(reinterpret_cast<const char*>(&max_elements_), sizeof(max_elements_));

  for (const auto& [id, label] : id_to_label_) {
    size_t len = id.size();
    ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
    ofs.write(id.data(), len);
    ofs.write(reinterpret_cast<const char*>(&label), sizeof(label));
  }
#endif
}

void StreamingIndex::Load(const std::string& path) {
  std::unique_lock lock(mutex_);

#ifdef HAFS_HAS_HNSW
  // Load ID mappings first to get dimensions
  std::ifstream ifs(path + ".ids", std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Cannot open " + path + ".ids");
  }

  size_t count;
  ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
  ifs.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
  ifs.read(reinterpret_cast<char*>(&max_elements_), sizeof(max_elements_));

  id_to_label_.clear();
  label_to_id_.clear();
  deleted_labels_.clear();

  uint64_t max_label = 0;
  for (size_t i = 0; i < count; i++) {
    size_t len;
    ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string id(len, '\0');
    ifs.read(id.data(), len);
    uint64_t label;
    ifs.read(reinterpret_cast<char*>(&label), sizeof(label));

    id_to_label_[id] = label;
    label_to_id_[label] = id;
    max_label = std::max(max_label, label);
  }
  next_label_ = max_label + 1;

  // Load HNSW index
  space_ = std::make_unique<hnswlib::InnerProductSpace>(dim_);
  index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
      space_.get(), path + ".hnsw");
  index_->setEf(std::max(static_cast<size_t>(50), ef_construction_));
#endif
}

StreamingIndex::Stats StreamingIndex::GetStats() const {
  std::shared_lock lock(mutex_);
  return Stats{
      total_added_.load(),
      total_removed_.load(),
      id_to_label_.size(),
      deleted_labels_.size(),
      max_elements_,
      dim_,
  };
}

// Python bindings
void RegisterStreamingBindings(py::module_& m) {
  py::class_<StreamingIndex>(m, "StreamingIndex",
                             "Thread-safe streaming embedding index")
      .def(py::init<size_t, size_t, size_t, size_t>(), py::arg("dim"),
           py::arg("max_elements") = 100000, py::arg("ef_construction") = 200,
           py::arg("M") = 16)
      .def(
          "add",
          [](StreamingIndex& self, const std::string& id,
             py::array_t<float> embedding) {
            auto buf = embedding.unchecked<1>();
            return self.Add(id, buf.data(0));
          },
          py::arg("id"), py::arg("embedding"), "Add a single embedding")
      .def(
          "add_batch",
          [](StreamingIndex& self, const std::vector<std::string>& ids,
             py::array_t<float> embeddings) {
            auto buf = embeddings.unchecked<2>();
            return self.AddBatch(ids, buf.data(0, 0));
          },
          py::arg("ids"), py::arg("embeddings"), "Add multiple embeddings")
      .def("remove", &StreamingIndex::Remove, py::arg("id"),
           "Remove an embedding by ID")
      .def(
          "update",
          [](StreamingIndex& self, const std::string& id,
             py::array_t<float> embedding) {
            auto buf = embedding.unchecked<1>();
            return self.Update(id, buf.data(0));
          },
          py::arg("id"), py::arg("embedding"), "Update an embedding")
      .def(
          "search",
          [](const StreamingIndex& self, py::array_t<float> query, size_t k) {
            auto buf = query.unchecked<1>();
            std::vector<uint64_t> labels(k);
            std::vector<float> distances(k);
            self.Search(buf.data(0), k, labels.data(), distances.data());

            // Filter invalid results
            size_t valid = 0;
            for (size_t i = 0; i < k; i++) {
              if (labels[i] != static_cast<uint64_t>(-1)) valid++;
            }

            py::array_t<uint64_t> py_labels(valid);
            py::array_t<float> py_distances(valid);
            auto l_buf = py_labels.mutable_unchecked<1>();
            auto d_buf = py_distances.mutable_unchecked<1>();
            for (size_t i = 0; i < valid; i++) {
              l_buf(i) = labels[i];
              d_buf(i) = distances[i];
            }

            return py::make_tuple(py_labels, py_distances);
          },
          py::arg("query"), py::arg("k") = 10, "Search for k nearest neighbors")
      .def(
          "search_with_ids",
          [](const StreamingIndex& self, py::array_t<float> query, size_t k) {
            auto buf = query.unchecked<1>();
            auto results = self.SearchWithIds(buf.data(0), k);

            std::vector<std::string> ids;
            std::vector<float> scores;
            for (const auto& [id, score] : results) {
              ids.push_back(id);
              scores.push_back(score);
            }

            py::array_t<float> py_scores(scores.size());
            auto s_buf = py_scores.mutable_unchecked<1>();
            for (size_t i = 0; i < scores.size(); i++) {
              s_buf(i) = scores[i];
            }

            return py::make_tuple(ids, py_scores);
          },
          py::arg("query"), py::arg("k") = 10,
          "Search and return string IDs with scores")
      .def("contains", &StreamingIndex::Contains, py::arg("id"),
           "Check if ID exists")
      .def("size", &StreamingIndex::Size, "Get number of active embeddings")
      .def_property_readonly("capacity", &StreamingIndex::Capacity)
      .def_property_readonly("dimension", &StreamingIndex::Dimension)
      .def("resize", &StreamingIndex::Resize, py::arg("new_max_elements"),
           "Resize capacity (rebuilds index)")
      .def("compact", &StreamingIndex::Compact,
           "Remove deleted entries and rebuild")
      .def("save", &StreamingIndex::Save, py::arg("path"), "Save to files")
      .def("load", &StreamingIndex::Load, py::arg("path"), "Load from files")
      .def(
          "get_stats",
          [](const StreamingIndex& self) {
            auto stats = self.GetStats();
            return py::dict("total_added"_a = stats.total_added,
                            "total_removed"_a = stats.total_removed,
                            "active_count"_a = stats.active_count,
                            "deleted_count"_a = stats.deleted_count,
                            "capacity"_a = stats.capacity,
                            "dimension"_a = stats.dimension);
          },
          "Get index statistics");

#ifdef HAFS_HAS_HNSW
  m.attr("__has_streaming__") = true;
#else
  m.attr("__has_streaming__") = false;
#endif
}

}  // namespace hafs
