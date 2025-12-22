"""Fast JSON loading for embedding files with simdjson acceleration.

Uses SIMD-accelerated JSON parsing when available (5-10x faster than stdlib json).

Usage:
    from core.io import (
        load_embedding_file,
        load_embeddings_from_directory,
        get_io_backend_info,
    )

    # Load single file
    ids, embeddings = load_embedding_file("embeddings.json")

    # Load all JSON files from directory
    ids, embeddings, stats = load_embeddings_from_directory("./embeddings/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Try native implementation
_NATIVE_AVAILABLE = False
_HAS_SIMDJSON = False

try:
    from core._native import (
        load_embedding_file as _native_load_file,
        load_embeddings_from_directory as _native_load_dir,
    )
    from core._native import __has_simdjson__

    _NATIVE_AVAILABLE = True
    _HAS_SIMDJSON = __has_simdjson__
except ImportError:
    pass


def load_embedding_file(
    path: Union[str, Path],
) -> Tuple[List[str], NDArray[np.float32]]:
    """Load embeddings from a JSON file.

    Supports two formats:
    - {"embeddings": [{"id": "...", "vector": [...]}, ...]}
    - [{"id": "...", "embedding": [...]}, ...]

    Args:
        path: Path to JSON file

    Returns:
        Tuple of (ids, embeddings) where embeddings is (n, dim) array

    Raises:
        ValueError: If file format is invalid
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)

    if _NATIVE_AVAILABLE and _HAS_SIMDJSON:
        ids, embeddings, dim, success, error = _native_load_file(str(path))
        if not success:
            raise ValueError(f"Failed to load {path}: {error}")
        return ids, embeddings

    # Python fallback
    return _load_file_python(path)


def _load_file_python(path: Path) -> Tuple[List[str], NDArray[np.float32]]:
    """Pure Python JSON loading fallback."""
    with open(path, "r") as f:
        data = json.load(f)

    ids: List[str] = []
    embeddings: List[List[float]] = []

    # Try format 1: {"embeddings": [...]}
    if isinstance(data, dict) and "embeddings" in data:
        for item in data["embeddings"]:
            ids.append(item["id"])
            vec = item.get("vector") or item.get("embedding")
            embeddings.append(vec)
    # Format 2: [{...}, ...]
    elif isinstance(data, list):
        for item in data:
            ids.append(item["id"])
            vec = item.get("vector") or item.get("embedding")
            embeddings.append(vec)
    else:
        raise ValueError(f"Unknown JSON format in {path}")

    return ids, np.array(embeddings, dtype=np.float32)


def load_embeddings_from_directory(
    dir_path: Union[str, Path],
) -> Tuple[List[str], NDArray[np.float32], Dict[str, int]]:
    """Load all embedding JSON files from a directory.

    Args:
        dir_path: Directory containing JSON files

    Returns:
        Tuple of (ids, embeddings, stats) where:
        - ids: List of embedding IDs
        - embeddings: (n, dim) array of embeddings
        - stats: Dict with 'files_loaded', 'files_failed', 'total_embeddings'

    Raises:
        ValueError: If no valid files found
    """
    dir_path = Path(dir_path)

    if _NATIVE_AVAILABLE and _HAS_SIMDJSON:
        ids, embeddings, dim, loaded, failed, errors = _native_load_dir(str(dir_path))
        stats = {
            "files_loaded": loaded,
            "files_failed": failed,
            "total_embeddings": len(ids),
            "dimension": dim,
        }
        if errors:
            stats["errors"] = errors
        return ids, embeddings, stats

    # Python fallback
    return _load_dir_python(dir_path)


def _load_dir_python(
    dir_path: Path,
) -> Tuple[List[str], NDArray[np.float32], Dict[str, int]]:
    """Pure Python directory loading fallback."""
    all_ids: List[str] = []
    all_embeddings: List[NDArray[np.float32]] = []
    loaded = 0
    failed = 0
    errors: List[str] = []

    json_files = list(dir_path.glob("*.json"))

    for json_file in json_files:
        try:
            ids, embeddings = _load_file_python(json_file)
            all_ids.extend(ids)
            all_embeddings.append(embeddings)
            loaded += 1
        except Exception as e:
            failed += 1
            errors.append(f"{json_file}: {e}")

    if not all_embeddings:
        raise ValueError(f"No valid embedding files in {dir_path}")

    combined = np.vstack(all_embeddings)
    stats = {
        "files_loaded": loaded,
        "files_failed": failed,
        "total_embeddings": len(all_ids),
        "dimension": combined.shape[1] if combined.size > 0 else 0,
    }
    if errors:
        stats["errors"] = errors

    return all_ids, combined, stats


def get_io_backend_info() -> Dict[str, str]:
    """Get information about the IO backend.

    Returns:
        Dict with 'native' and 'simdjson' keys
    """
    return {
        "native": "yes" if _NATIVE_AVAILABLE else "no",
        "simdjson": "yes" if _HAS_SIMDJSON else "no",
    }
