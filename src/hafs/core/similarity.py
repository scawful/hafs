"""Similarity operations with native C++ acceleration and NumPy fallback.

This module provides cosine similarity and related operations, using
native SIMD-optimized C++ when available, falling back to NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Try to import native module, fall back to NumPy
_NATIVE_AVAILABLE = False
_SIMD_TYPE = "NumPy"
_BLAS_TYPE = "NumPy"

try:
    from hafs.core._native import (
        cosine_similarity as _native_cosine_similarity,
        cosine_similarity_batch as _native_cosine_similarity_batch,
        top_k_similar as _native_top_k_similar,
    )
    from hafs.core._native import __simd__, __blas__

    _NATIVE_AVAILABLE = True
    _SIMD_TYPE = __simd__
    _BLAS_TYPE = __blas__
except ImportError:
    pass


def cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (list or array)
        b: Second vector (list or array)

    Returns:
        Cosine similarity in range [-1, 1]
    """
    # Handle empty/None inputs
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0

    if _NATIVE_AVAILABLE:
        return _native_cosine_similarity(
            np.ascontiguousarray(a, dtype=np.float32),
            np.ascontiguousarray(b, dtype=np.float32),
        )

    # NumPy fallback
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(
    queries: NDArray[np.float32], corpus: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Compute pairwise cosine similarity matrix.

    Args:
        queries: Query vectors, shape (n_queries, dim)
        corpus: Corpus vectors, shape (n_corpus, dim)

    Returns:
        Similarity matrix, shape (n_queries, n_corpus)
    """
    if _NATIVE_AVAILABLE:
        return _native_cosine_similarity_batch(
            np.ascontiguousarray(queries, dtype=np.float32),
            np.ascontiguousarray(corpus, dtype=np.float32),
        )

    # NumPy fallback
    queries = np.asarray(queries, dtype=np.float32)
    corpus = np.asarray(corpus, dtype=np.float32)

    # Normalize
    q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    c_norms = np.linalg.norm(corpus, axis=1, keepdims=True)

    # Avoid division by zero
    q_norms = np.where(q_norms < 1e-8, 1.0, q_norms)
    c_norms = np.where(c_norms < 1e-8, 1.0, c_norms)

    q_normalized = queries / q_norms
    c_normalized = corpus / c_norms

    return q_normalized @ c_normalized.T


def top_k_similar(
    query: NDArray[np.float32], corpus: NDArray[np.float32], k: int = 10
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """Find the k most similar vectors in corpus to query.

    Args:
        query: Query vector, shape (dim,)
        corpus: Corpus vectors, shape (n, dim)
        k: Number of results to return

    Returns:
        Tuple of (indices, scores) arrays
    """
    if _NATIVE_AVAILABLE:
        return _native_top_k_similar(
            np.ascontiguousarray(query, dtype=np.float32),
            np.ascontiguousarray(corpus, dtype=np.float32),
            k,
        )

    # NumPy fallback
    query = np.asarray(query, dtype=np.float32)
    corpus = np.asarray(corpus, dtype=np.float32)

    # Compute all similarities
    scores = cosine_similarity_batch(query.reshape(1, -1), corpus).flatten()

    # Get top-k indices
    if k >= len(scores):
        indices = np.argsort(scores)[::-1]
        return indices.astype(np.int32), scores[indices].astype(np.float32)

    indices = np.argpartition(scores, -k)[-k:]
    indices = indices[np.argsort(scores[indices])[::-1]]

    return indices.astype(np.int32), scores[indices].astype(np.float32)


def get_backend_info() -> dict[str, str]:
    """Get information about the similarity backend.

    Returns:
        Dict with 'native', 'simd', and 'blas' keys
    """
    return {
        "native": "yes" if _NATIVE_AVAILABLE else "no",
        "simd": _SIMD_TYPE,
        "blas": _BLAS_TYPE,
    }
