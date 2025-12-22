"""Thread-safe streaming embedding index with real-time updates.

Provides O(log n) search with incremental add/remove/update operations,
all thread-safe for concurrent access.

Usage:
    from core.streaming_index import StreamingIndex, get_streaming_backend_info

    # Create index
    index = StreamingIndex(dim=768, max_elements=100000)

    # Add embeddings
    index.add("doc1", embedding1)
    index.add_batch(["doc2", "doc3"], embeddings_array)

    # Search
    ids, scores = index.search(query, k=10)

    # Update/Remove
    index.update("doc1", new_embedding)
    index.remove("doc2")

    # Persistence
    index.save("embeddings")
    index.load("embeddings")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Try native implementation
_NATIVE_AVAILABLE = False
_HAS_STREAMING = False

try:
    from core._native import StreamingIndex as _NativeStreamingIndex
    from core._native import __has_streaming__

    _NATIVE_AVAILABLE = True
    _HAS_STREAMING = __has_streaming__
except ImportError:
    pass


class StreamingIndex:
    """Thread-safe streaming embedding index.

    Supports concurrent reads and exclusive writes with automatic locking.
    Uses HNSW for O(log n) approximate nearest neighbor search.

    Args:
        dim: Embedding dimension
        max_elements: Maximum capacity (can be resized)
        ef_construction: HNSW construction parameter (higher = better quality)
        M: HNSW connections per node (higher = better recall, more memory)
    """

    def __init__(
        self,
        dim: int,
        max_elements: int = 100000,
        ef_construction: int = 200,
        M: int = 16,
    ):
        self._dim = dim
        self._max_elements = max_elements

        if _NATIVE_AVAILABLE and _HAS_STREAMING:
            self._index = _NativeStreamingIndex(dim, max_elements, ef_construction, M)
        else:
            # Python fallback: simple dict-based index
            self._embeddings: Dict[str, NDArray[np.float32]] = {}
            self._index = None

    def add(self, id: str, embedding: NDArray[np.float32]) -> bool:
        """Add a single embedding.

        Args:
            id: Unique identifier
            embedding: Embedding vector

        Returns:
            True if added, False if ID already exists
        """
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        if self._index is not None:
            return self._index.add(id, embedding)

        # Python fallback
        if id in self._embeddings:
            return False
        self._embeddings[id] = embedding
        return True

    def add_batch(
        self, ids: List[str], embeddings: NDArray[np.float32]
    ) -> int:
        """Add multiple embeddings.

        Args:
            ids: List of unique identifiers
            embeddings: (n, dim) array of embeddings

        Returns:
            Number of embeddings added (skips existing IDs)
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        if self._index is not None:
            return self._index.add_batch(ids, embeddings)

        # Python fallback
        added = 0
        for i, id in enumerate(ids):
            if self.add(id, embeddings[i]):
                added += 1
        return added

    def remove(self, id: str) -> bool:
        """Remove an embedding by ID.

        Args:
            id: Identifier to remove

        Returns:
            True if removed, False if not found
        """
        if self._index is not None:
            return self._index.remove(id)

        # Python fallback
        if id in self._embeddings:
            del self._embeddings[id]
            return True
        return False

    def update(self, id: str, embedding: NDArray[np.float32]) -> bool:
        """Update an existing embedding.

        Args:
            id: Identifier to update
            embedding: New embedding vector

        Returns:
            True if updated, False if ID not found
        """
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        if self._index is not None:
            return self._index.update(id, embedding)

        # Python fallback
        if id not in self._embeddings:
            return False
        self._embeddings[id] = embedding
        return True

    def search(
        self, query: NDArray[np.float32], k: int = 10
    ) -> Tuple[List[str], NDArray[np.float32]]:
        """Search for k nearest neighbors.

        Args:
            query: Query embedding
            k: Number of neighbors to return

        Returns:
            Tuple of (ids, scores) where scores are similarities in [0, 1]
        """
        query = np.ascontiguousarray(query, dtype=np.float32)

        if self._index is not None:
            return self._index.search_with_ids(query, k)

        # Python fallback: brute force
        if not self._embeddings:
            return [], np.array([], dtype=np.float32)

        ids = list(self._embeddings.keys())
        embeddings = np.array(list(self._embeddings.values()))

        # Cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        scores = emb_norms @ query_norm

        # Top k
        k = min(k, len(ids))
        top_indices = np.argsort(scores)[::-1][:k]
        top_ids = [ids[i] for i in top_indices]
        top_scores = scores[top_indices]

        return top_ids, top_scores.astype(np.float32)

    def contains(self, id: str) -> bool:
        """Check if ID exists."""
        if self._index is not None:
            return self._index.contains(id)
        return id in self._embeddings

    def size(self) -> int:
        """Get number of active embeddings."""
        if self._index is not None:
            return self._index.size()
        return len(self._embeddings)

    @property
    def capacity(self) -> int:
        """Maximum capacity."""
        if self._index is not None:
            return self._index.capacity
        return self._max_elements

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        if self._index is not None:
            return self._index.dimension
        return self._dim

    def resize(self, new_max_elements: int) -> None:
        """Resize capacity (rebuilds index)."""
        if self._index is not None:
            self._index.resize(new_max_elements)
        self._max_elements = new_max_elements

    def compact(self) -> None:
        """Remove deleted entries and rebuild index."""
        if self._index is not None:
            self._index.compact()

    def save(self, path: Union[str, Path]) -> None:
        """Save index to files.

        Creates two files: {path}.hnsw and {path}.ids
        """
        path = str(path)

        if self._index is not None:
            self._index.save(path)
        else:
            # Python fallback: save as npz
            np.savez(
                path + ".npz",
                ids=np.array(list(self._embeddings.keys())),
                embeddings=np.array(list(self._embeddings.values())),
            )

    def load(self, path: Union[str, Path]) -> None:
        """Load index from files."""
        path = str(path)

        if self._index is not None:
            self._index.load(path)
        else:
            # Python fallback
            data = np.load(path + ".npz")
            self._embeddings = dict(zip(data["ids"], data["embeddings"]))

    def get_stats(self) -> Dict[str, int]:
        """Get index statistics."""
        if self._index is not None:
            return self._index.get_stats()
        return {
            "total_added": len(self._embeddings),
            "total_removed": 0,
            "active_count": len(self._embeddings),
            "deleted_count": 0,
            "capacity": self._max_elements,
            "dimension": self._dim,
        }


def get_streaming_backend_info() -> Dict[str, str]:
    """Get information about the streaming index backend.

    Returns:
        Dict with 'native' and 'streaming' keys
    """
    return {
        "native": "yes" if _NATIVE_AVAILABLE else "no",
        "streaming": "yes" if _HAS_STREAMING else "no",
    }
