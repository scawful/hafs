"""Deprecated shim for backends.base."""

from __future__ import annotations

import warnings

from backends.base import (  # noqa: F401
    BackendCapabilities,
    BackendRegistry,
    BaseChatBackend,
    ChatMessage,
)

warnings.warn(
    "backends.base is deprecated. Import from 'backends.base' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BackendCapabilities",
    "BackendRegistry",
    "BaseChatBackend",
    "ChatMessage",
]
