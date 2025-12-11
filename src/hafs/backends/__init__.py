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
from hafs.backends.pty import PtyWrapper

# Register built-in backends at module load time
BackendRegistry.register(GeminiCliBackend)
BackendRegistry.register(ClaudeCliBackend)

__all__ = [
    "BackendCapabilities",
    "BackendRegistry",
    "BaseChatBackend",
    "ChatMessage",
    "ClaudeCliBackend",
    "GeminiCliBackend",
    "PtyWrapper",
]
