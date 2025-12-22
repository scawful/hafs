"""HNSW Approximate Nearest Neighbor Index with NumPy fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Try native implementation
_NATIVE_AVAILABLE = False
_HAS_HNSW = False

try:
    from core._native import HNSWIndex as _NativeHNSWIndex
    from core._native import __has_hnsw__

    _NATIVE_AVAILABLE = True
    _HAS_HNSW = __has_hnsw__
except ImportError:
    pass


class HNSWIndex:
    """HNSW index for approximate nearest neighbor search.

    Uses native hnswlib when available, falls back to brute-force NumPy.

    Args:
        dim: Dimension of vectors
        max_elements: Maximum number of vectors to store
        distance: Distance type ("l2", "ip", "cosine")
        M: Number of bi-directional links per element (affects memory/quality)
        ef_construction: Size of dynamic list for construction (affects build time/quality)
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 100000,
        distance: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
    ):
        self.dim = dim
        self.max_elements = max_elements
        self.distance = distance
        self._M = M
        self._ef_construction = ef_construction

        if _NATIVE_AVAILABLE:
            dist_map = {"l2": 0, "ip": 1, "cosine": 2}
            self._index = _NativeHNSWIndex(
                dim, max_elements, dist_map.get(distance.lower(), 2), M, ef_construction
            )
            self._use_native = True
        else:
            self._vectors: List[NDArray[np.float32]] = []
            self._labels: List[int] = []
            self._use_native = False

    @property
    def is_native(self) -> bool:
        """Whether using native HNSW implementation."""
        return self._use_native and _HAS_HNSW

    def build(
        self, data: NDArray[np.float32], labels: Optional[NDArray] = None
    ) -> None:
        """Build index from data array.

        Args:
            data: Vectors of shape (n, dim)
            labels: Optional labels for each vector (defaults to 0..n-1)
        """
        data = np.ascontiguousarray(data, dtype=np.float32)

        if self._use_native:
            self._index.build(data)
        else:
            self._vectors = [data[i] for i in range(len(data))]
            self._labels = (
                list(labels) if labels is not None else list(range(len(data)))
            )

    def add(self, vector: NDArray[np.float32], label: int) -> None:
        """Add single vector with label.

        Args:
            vector: Vector of shape (dim,)
            label: Integer label for the vector
        """
        vector = np.ascontiguousarray(vector, dtype=np.float32)

        if self._use_native:
            self._index.add(vector, label)
        else:
            self._vectors.append(vector.copy())
            self._labels.append(label)

    def search(
        self, query: NDArray[np.float32], k: int = 10
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Search for k nearest neighbors.

        Args:
            query: Query vector of shape (dim,)
            k: Number of neighbors to return

        Returns:
            Tuple of (labels, distances/similarities) arrays
        """
        query = np.ascontiguousarray(query, dtype=np.float32)

        if self._use_native:
            labels, distances = self._index.search(query, k)
            return labels.astype(np.int64), distances.astype(np.float32)

        # NumPy brute-force fallback
        if not self._vectors:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        from core.similarity import cosine_similarity

        scores = np.array(
            [cosine_similarity(query, v) for v in self._vectors], dtype=np.float32
        )

        actual_k = min(k, len(scores))
        if actual_k == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        top_k_idx = np.argpartition(scores, -actual_k)[-actual_k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        return (
            np.array([self._labels[i] for i in top_k_idx], dtype=np.int64),
            scores[top_k_idx].astype(np.float32),
        )

    def search_batch(
        self, queries: NDArray[np.float32], k: int = 10
    ) -> Tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Batch search for k nearest neighbors.

        Args:
            queries: Query vectors of shape (n_queries, dim)
            k: Number of neighbors to return per query

        Returns:
            Tuple of (labels, distances) with shape (n_queries, k)
        """
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        if self._use_native:
            labels, distances = self._index.search_batch(queries, k)
            return labels.astype(np.int64), distances.astype(np.float32)

        # NumPy fallback
        n_queries = len(queries)
        all_labels = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)

        for i, query in enumerate(queries):
            labels, distances = self.search(query, k)
            actual_k = len(labels)
            all_labels[i, :actual_k] = labels
            all_distances[i, :actual_k] = distances

        return all_labels, all_distances

    def save(self, path: Union[str, Path]) -> None:
        """Save index to file.

        Args:
            path: Path to save the index
        """
        if self._use_native:
            self._index.save(str(path))
        else:
            import json

            data = {
                "dim": self.dim,
                "distance": self.distance,
                "vectors": [v.tolist() for v in self._vectors],
                "labels": self._labels,
            }
            Path(path).write_text(json.dumps(data))

    def load(self, path: Union[str, Path]) -> None:
        """Load index from file.

        Args:
            path: Path to load the index from
        """
        if self._use_native:
            self._index.load(str(path))
        else:
            import json

            data = json.loads(Path(path).read_text())
            self._vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
            self._labels = data["labels"]

    def set_ef(self, ef: int) -> None:
        """Set ef parameter for search (higher = more accurate but slower).

        Args:
            ef: Search parameter
        """
        if self._use_native:
            self._index.set_ef(ef)

    def __len__(self) -> int:
        """Return number of vectors in index."""
        if self._use_native:
            return self._index.size()
        return len(self._vectors)


def get_index_backend_info() -> Dict[str, str]:
    """Get information about the index backend.

    Returns:
        Dict with 'native', 'hnsw' keys
    """
    return {
        "native": "yes" if _NATIVE_AVAILABLE else "no",
        "hnsw": "yes" if _HAS_HNSW else "no",
    }
