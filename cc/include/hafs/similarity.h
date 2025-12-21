#ifndef HAFS_SIMILARITY_H_
#define HAFS_SIMILARITY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hafs {

// =============================================================================
// Core Similarity Functions (C++ API)
// =============================================================================

// Compute cosine similarity between two vectors.
// Vectors must have the same dimension n.
// Returns: similarity score in range [-1, 1]
float CosineSimilarity(const float* a, const float* b, size_t n);

// Compute L2 (Euclidean) distance between two vectors.
float L2Distance(const float* a, const float* b, size_t n);

// Compute dot product of two vectors.
float DotProduct(const float* a, const float* b, size_t n);

// L2-normalize a vector in place.
void NormalizeVector(float* vec, size_t n);

// =============================================================================
// Batch Operations (C++ API)
// =============================================================================

// Compute pairwise cosine similarity matrix between queries and corpus.
// Output is a (n_queries x n_corpus) matrix stored row-major.
void CosineSimilarityBatch(const float* queries, const float* corpus,
                           float* output, size_t n_queries, size_t n_corpus,
                           size_t dim);

// Find top-k most similar vectors in corpus for a single query.
// indices and scores must have space for k elements.
void TopKSimilar(const float* query, const float* corpus,
                 int32_t* indices, float* scores,
                 size_t n_corpus, size_t dim, size_t k);

// =============================================================================
// Python Bindings (pybind11 wrappers)
// =============================================================================

// Single vector cosine similarity from numpy arrays.
float PyCosineSimilarity(py::array_t<float> a, py::array_t<float> b);

// Batch cosine similarity returning (n_queries x n_corpus) matrix.
py::array_t<float> PyCosineSimilarityBatch(py::array_t<float> queries,
                                            py::array_t<float> corpus);

// Top-k similar vectors, returns (indices, scores) tuple.
std::tuple<py::array_t<int32_t>, py::array_t<float>> PyTopKSimilar(
    py::array_t<float> query, py::array_t<float> corpus, size_t k);

}  // namespace hafs

#endif  // HAFS_SIMILARITY_H_
