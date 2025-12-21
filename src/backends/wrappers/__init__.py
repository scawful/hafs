"""Backend wrappers for extending functionality."""

from backends.wrappers.history import HistoryBackend, wrap_with_history

__all__ = [
    "HistoryBackend",
    "wrap_with_history",
]
