#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "../common/simd_utils.h"
#include "similarity.h"

namespace hafs {

// =============================================================================
// Batch Operations using NEON SIMD
// =============================================================================

// Compute pairwise cosine similarity matrix.
// Uses NEON SIMD for dot products and norms.
void BatchCosineMatrixSimd(const float* queries, const float* corpus,
                           float* output,
                           size_t n_queries, size_t n_corpus,
                           size_t dim) {
  // Pre-compute corpus norms for efficiency
  std::vector<float> corpus_norms(n_corpus);
  for (size_t j = 0; j < n_corpus; ++j) {
    corpus_norms[j] = std::sqrt(SquaredNormNeon(corpus + j * dim, dim));
  }

  // Compute similarity for each query
  for (size_t i = 0; i < n_queries; ++i) {
    const float* query = queries + i * dim;
    float query_norm = std::sqrt(SquaredNormNeon(query, dim));

    for (size_t j = 0; j < n_corpus; ++j) {
      if (query_norm < 1e-8f || corpus_norms[j] < 1e-8f) {
        output[i * n_corpus + j] = 0.0f;
      } else {
        float dot = DotProductNeon(query, corpus + j * dim, dim);
        output[i * n_corpus + j] = dot / (query_norm * corpus_norms[j]);
      }
    }
  }
}

// L2 normalize rows of a matrix in place.
void NormalizeRowsSimd(float* matrix, size_t n_rows, size_t dim) {
  for (size_t i = 0; i < n_rows; ++i) {
    NormalizeVectorNeon(matrix + i * dim, dim);
  }
}

}  // namespace hafs
