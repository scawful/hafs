"""I/O Manager for optimized file operations with batching and caching.

Provides:
- Write batching: Queue writes and flush periodically
- Read caching: In-memory cache with TTL
- Lazy loading: Load files on-demand
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config.loader import CognitiveProtocolConfig, PerformanceConfig, get_config


class CacheEntry:
    """Cache entry with TTL."""

    def __init__(self, data: dict, ttl_seconds: float):
        """Initialize cache entry.

        Args:
            data: Cached data.
            ttl_seconds: Time to live in seconds.
        """
        self.data = data
        self.timestamp = time.time()
        self.ttl_seconds = ttl_seconds

    def is_expired(self) -> bool:
        """Check if cache entry is expired.

        Returns:
            True if expired, False otherwise.
        """
        return (time.time() - self.timestamp) > self.ttl_seconds


class IOManager:
    """Manages file I/O with batching and caching for performance.

    Features:
    - Write batching: Queues writes and flushes periodically
    - Read caching: In-memory cache with configurable TTL
    - Lazy loading: Load files on-demand rather than eagerly

    Example:
        manager = get_io_manager()
        data = manager.read_json(Path("state.json"))
        manager.write_json(Path("state.json"), data)
        manager.flush()  # Force immediate write
    """

    def __init__(self, config: PerformanceConfig | None = None):
        """Initialize I/O manager.

        Args:
            config: Performance configuration. If None, uses default config.
        """
        cfg = config or get_config().performance
        self._config = cfg

        # Caching
        self._cache: dict[Path, CacheEntry] = {}
        self._cache_lock = threading.Lock()

        # Batching
        self._write_queue: dict[Path, dict] = {}
        self._write_lock = threading.Lock()
        self._last_flush = time.time()

        # Stats
        self._cache_hits = 0
        self._cache_misses = 0
        self._writes_batched = 0
        self._writes_immediate = 0

    def read_json(self, path: Path) -> dict:
        """Read JSON file with caching.

        Args:
            path: Path to JSON file.

        Returns:
            Parsed JSON data as dictionary.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If file contains invalid JSON.
        """
        path = path.resolve()

        # Check cache first
        if self._config.enable_caching:
            with self._cache_lock:
                if path in self._cache:
                    entry = self._cache[path]
                    if not entry.is_expired():
                        self._cache_hits += 1
                        return entry.data.copy()
                    else:
                        # Expired - remove
                        del self._cache[path]

        # Cache miss - load from disk
        self._cache_misses += 1

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(content)

            # Cache it
            if self._config.enable_caching:
                with self._cache_lock:
                    self._cache[path] = CacheEntry(
                        data.copy(), self._config.cache_ttl_seconds
                    )

            return data

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in {path}: {e.msg}", e.doc, e.pos
            )

    def write_json(
        self,
        path: Path,
        data: dict,
        immediate: bool = False,
        indent: int = 2,
    ) -> None:
        """Write JSON file with optional batching.

        Args:
            path: Path to write to.
            data: Data to write.
            immediate: If True, write immediately. If False, batch the write.
            indent: JSON indentation level.
        """
        path = path.resolve()

        # Invalidate cache
        if self._config.enable_caching:
            with self._cache_lock:
                if path in self._cache:
                    del self._cache[path]

        if immediate or not self._config.enable_batching:
            # Write immediately
            self._write_immediate(path, data, indent)
            self._writes_immediate += 1
        else:
            # Queue for batching
            with self._write_lock:
                self._write_queue[path] = {
                    "data": data,
                    "indent": indent,
                }
                self._writes_batched += 1

            # Auto-flush if interval exceeded
            if self._should_flush():
                self.flush()

    def _write_immediate(self, path: Path, data: dict, indent: int) -> None:
        """Write JSON file immediately.

        Args:
            path: Path to write to.
            data: Data to write.
            indent: JSON indentation level.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        json_str = json.dumps(data, indent=indent, default=str)
        path.write_text(json_str + "\n", encoding="utf-8")

    def _should_flush(self) -> bool:
        """Check if we should auto-flush the write queue.

        Returns:
            True if flush interval has elapsed.
        """
        elapsed_ms = (time.time() - self._last_flush) * 1000
        return elapsed_ms >= self._config.batch_flush_interval_ms

    def flush(self) -> int:
        """Flush all pending writes to disk.

        Returns:
            Number of files written.
        """
        with self._write_lock:
            count = len(self._write_queue)
            if count == 0:
                return 0

            # Write all queued files
            for path, write_info in self._write_queue.items():
                try:
                    self._write_immediate(
                        path, write_info["data"], write_info["indent"]
                    )
                except OSError:
                    # Log error but continue
                    pass

            self._write_queue.clear()
            self._last_flush = time.time()

        return count

    def clear_cache(self, path: Optional[Path] = None) -> None:
        """Clear cached data.

        Args:
            path: If provided, clear only this file. Otherwise clear all.
        """
        with self._cache_lock:
            if path:
                path = path.resolve()
                if path in self._cache:
                    del self._cache[path]
            else:
                self._cache.clear()

    def invalidate_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        with self._cache_lock:
            expired = [
                path for path, entry in self._cache.items() if entry.is_expired()
            ]
            for path in expired:
                del self._cache[path]
            return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get I/O manager statistics.

        Returns:
            Dictionary with performance stats.
        """
        with self._cache_lock:
            cache_size = len(self._cache)
        with self._write_lock:
            queue_size = len(self._write_queue)

        total_reads = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_reads if total_reads > 0 else 0.0
        )

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": cache_size,
            "queue_size": queue_size,
            "writes_batched": self._writes_batched,
            "writes_immediate": self._writes_immediate,
            "config": {
                "batching_enabled": self._config.enable_batching,
                "caching_enabled": self._config.enable_caching,
                "batch_flush_interval_ms": self._config.batch_flush_interval_ms,
                "cache_ttl_seconds": self._config.cache_ttl_seconds,
            },
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        self._writes_batched = 0
        self._writes_immediate = 0


# Global singleton
_io_manager: Optional[IOManager] = None
_io_manager_lock = threading.Lock()


def get_io_manager(
    config: CognitiveProtocolConfig | None = None, reload: bool = False
) -> IOManager:
    """Get the global I/O manager singleton.

    Args:
        config: Cognitive protocol configuration. If None, uses default config.
        reload: If True, create a new instance even if one exists.

    Returns:
        IOManager instance.
    """
    global _io_manager

    with _io_manager_lock:
        if _io_manager is None or reload:
            cfg = config or get_config()
            _io_manager = IOManager(cfg.performance)

    return _io_manager
