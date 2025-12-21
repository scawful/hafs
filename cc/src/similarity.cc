#include "hafs/similarity.h"

#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

#include "hafs/simd_utils.h"

namespace hafs {

// =============================================================================
// Core Similarity Functions
// =============================================================================

float CosineSimilarity(const float* a, const float* b, size_t n) {
  float dot = DotProductNeon(a, b, n);
  float norm_a = std::sqrt(SquaredNormNeon(a, n));
  float norm_b = std::sqrt(SquaredNormNeon(b, n));

  if (norm_a < 1e-8f || norm_b < 1e-8f) {
    return 0.0f;
  }

  return dot / (norm_a * norm_b);
}

float L2Distance(const float* a, const float* b, size_t n) {
  return std::sqrt(L2DistanceSquaredNeon(a, b, n));
}

float DotProduct(const float* a, const float* b, size_t n) {
  return DotProductNeon(a, b, n);
}

void NormalizeVector(float* vec, size_t n) {
  NormalizeVectorNeon(vec, n);
}

// =============================================================================
// Batch Operations
// =============================================================================

void CosineSimilarityBatch(const float* queries, const float* corpus,
                           float* output, size_t n_queries, size_t n_corpus,
                           size_t dim) {
  // For each query, compute similarity with all corpus vectors
  for (size_t i = 0; i < n_queries; ++i) {
    const float* query = queries + i * dim;
    float query_norm = std::sqrt(SquaredNormNeon(query, dim));

    for (size_t j = 0; j < n_corpus; ++j) {
      const float* corpus_vec = corpus + j * dim;
      float corpus_norm = std::sqrt(SquaredNormNeon(corpus_vec, dim));

      if (query_norm < 1e-8f || corpus_norm < 1e-8f) {
        output[i * n_corpus + j] = 0.0f;
      } else {
        float dot = DotProductNeon(query, corpus_vec, dim);
        output[i * n_corpus + j] = dot / (query_norm * corpus_norm);
      }
    }
  }
}

void TopKSimilar(const float* query, const float* corpus,
                 int32_t* indices, float* scores,
                 size_t n_corpus, size_t dim, size_t k) {
  // Use a min-heap to track top-k
  using ScoreIndex = std::pair<float, int32_t>;
  std::priority_queue<ScoreIndex, std::vector<ScoreIndex>,
                      std::greater<ScoreIndex>> min_heap;

  float query_norm = std::sqrt(SquaredNormNeon(query, dim));

  for (size_t i = 0; i < n_corpus; ++i) {
    const float* corpus_vec = corpus + i * dim;
    float corpus_norm = std::sqrt(SquaredNormNeon(corpus_vec, dim));

    float score = 0.0f;
    if (query_norm >= 1e-8f && corpus_norm >= 1e-8f) {
      float dot = DotProductNeon(query, corpus_vec, dim);
      score = dot / (query_norm * corpus_norm);
    }

    if (min_heap.size() < k) {
      min_heap.push({score, static_cast<int32_t>(i)});
    } else if (score > min_heap.top().first) {
      min_heap.pop();
      min_heap.push({score, static_cast<int32_t>(i)});
    }
  }

  // Extract results in descending order
  size_t result_size = min_heap.size();
  for (size_t i = result_size; i > 0; --i) {
    auto [score, idx] = min_heap.top();
    min_heap.pop();
    indices[i - 1] = idx;
    scores[i - 1] = score;
  }

  // Fill remaining slots if k > n_corpus
  for (size_t i = result_size; i < k; ++i) {
    indices[i] = -1;
    scores[i] = 0.0f;
  }
}

// =============================================================================
// Python Bindings (pybind11 wrappers)
// =============================================================================

float PyCosineSimilarity(py::array_t<float> a, py::array_t<float> b) {
  auto buf_a = a.request();
  auto buf_b = b.request();

  if (buf_a.ndim != 1 || buf_b.ndim != 1) {
    throw std::runtime_error("Input arrays must be 1-dimensional");
  }

  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Input arrays must have the same size");
  }

  return CosineSimilarity(static_cast<float*>(buf_a.ptr),
                          static_cast<float*>(buf_b.ptr),
                          static_cast<size_t>(buf_a.size));
}

py::array_t<float> PyCosineSimilarityBatch(py::array_t<float> queries,
                                            py::array_t<float> corpus) {
  auto buf_q = queries.request();
  auto buf_c = corpus.request();

  if (buf_q.ndim != 2 || buf_c.ndim != 2) {
    throw std::runtime_error("Input arrays must be 2-dimensional");
  }

  size_t n_queries = buf_q.shape[0];
  size_t dim_q = buf_q.shape[1];
  size_t n_corpus = buf_c.shape[0];
  size_t dim_c = buf_c.shape[1];

  if (dim_q != dim_c) {
    throw std::runtime_error("Query and corpus must have same dimension");
  }

  // Create output array
  auto result = py::array_t<float>({n_queries, n_corpus});
  auto buf_r = result.request();

  CosineSimilarityBatch(static_cast<float*>(buf_q.ptr),
                        static_cast<float*>(buf_c.ptr),
                        static_cast<float*>(buf_r.ptr),
                        n_queries, n_corpus, dim_q);

  return result;
}

std::tuple<py::array_t<int32_t>, py::array_t<float>> PyTopKSimilar(
    py::array_t<float> query, py::array_t<float> corpus, size_t k) {
  auto buf_q = query.request();
  auto buf_c = corpus.request();

  if (buf_q.ndim != 1) {
    throw std::runtime_error("Query must be 1-dimensional");
  }

  if (buf_c.ndim != 2) {
    throw std::runtime_error("Corpus must be 2-dimensional");
  }

  size_t dim = buf_q.size;
  size_t n_corpus = buf_c.shape[0];
  size_t dim_c = buf_c.shape[1];

  if (dim != dim_c) {
    throw std::runtime_error("Query and corpus must have same dimension");
  }

  // Create output arrays
  auto indices = py::array_t<int32_t>(k);
  auto scores = py::array_t<float>(k);

  TopKSimilar(static_cast<float*>(buf_q.ptr),
              static_cast<float*>(buf_c.ptr),
              static_cast<int32_t*>(indices.request().ptr),
              static_cast<float*>(scores.request().ptr),
              n_corpus, dim, k);

  return std::make_tuple(indices, scores);
}

}  // namespace hafs
