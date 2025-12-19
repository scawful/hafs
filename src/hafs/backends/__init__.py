"""Chat backend module for AI agent orchestration."""

from __future__ import annotations

from hafs.backends.base import (
    BackendCapabilities,
    BackendRegistry,
    BaseChatBackend,
    ChatMessage,
)
from hafs.backends.claude import ClaudeCliBackend
from hafs.backends.gemini import GeminiCliBackend
from hafs.backends.history import HistoryBackend, wrap_with_history
from hafs.backends.oneshot import ClaudeOneShotBackend, GeminiOneShotBackend
from hafs.backends.pty import PtyWrapper

# Register built-in backends at module load time
BackendRegistry.register(GeminiCliBackend)
BackendRegistry.register(ClaudeCliBackend)
BackendRegistry.register(GeminiOneShotBackend)
BackendRegistry.register(ClaudeOneShotBackend)

__all__ = [
    "BackendCapabilities",
    "BackendRegistry",
    "BaseChatBackend",
    "ChatMessage",
    "ClaudeCliBackend",
    "ClaudeOneShotBackend",
    "GeminiCliBackend",
    "GeminiOneShotBackend",
    "HistoryBackend",
    "PtyWrapper",
    "wrap_with_history",
]
