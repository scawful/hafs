#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hafs/similarity.h"

namespace py = pybind11;

PYBIND11_MODULE(_similarity, m) {
  m.doc() = "HAFS native similarity operations with M1/ARM64 SIMD optimization";

  m.def("cosine_similarity", &hafs::PyCosineSimilarity,
        py::arg("a"), py::arg("b"),
        R"doc(
        Compute cosine similarity between two vectors.

        Args:
            a: First vector (1D numpy array of float32)
            b: Second vector (1D numpy array of float32)

        Returns:
            Cosine similarity score in range [-1, 1]
        )doc");

  m.def("cosine_similarity_batch", &hafs::PyCosineSimilarityBatch,
        py::arg("queries").noconvert(),
        py::arg("corpus").noconvert(),
        R"doc(
        Compute pairwise cosine similarity matrix.

        Args:
            queries: Query vectors (2D numpy array, shape [n_queries, dim])
            corpus: Corpus vectors (2D numpy array, shape [n_corpus, dim])

        Returns:
            Similarity matrix of shape [n_queries, n_corpus]
        )doc");

  m.def("top_k_similar", &hafs::PyTopKSimilar,
        py::arg("query"), py::arg("corpus"), py::arg("k") = 10,
        R"doc(
        Find the k most similar vectors in corpus to query.

        Args:
            query: Query vector (1D numpy array)
            corpus: Corpus vectors (2D numpy array, shape [n, dim])
            k: Number of top results to return (default: 10)

        Returns:
            Tuple of (indices, scores) arrays
        )doc");

  // Version info
  m.attr("__version__") = "0.1.0";

#ifdef HAFS_ARM64
  m.attr("__simd__") = "ARM NEON";
#else
  m.attr("__simd__") = "Scalar";
#endif

#ifdef HAFS_ARM64
  m.attr("__blas__") = "NEON SIMD";
#else
  m.attr("__blas__") = "Scalar";
#endif
}
