"""Chat backend module for AI agent orchestration."""

from __future__ import annotations

import logging

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

# New multi-provider backends (always available - dependencies are lazy-loaded)
from hafs.backends.ollama import OllamaBackend
from hafs.backends.anthropic import AnthropicBackend
from hafs.backends.openai import OpenAIBackend

logger = logging.getLogger(__name__)

# Register built-in backends at module load time
BackendRegistry.register(GeminiCliBackend)
BackendRegistry.register(ClaudeCliBackend)
BackendRegistry.register(GeminiOneShotBackend)
BackendRegistry.register(ClaudeOneShotBackend)

# Register new multi-provider backends
BackendRegistry.register(OllamaBackend)
BackendRegistry.register(AnthropicBackend)
BackendRegistry.register(OpenAIBackend)

logger.debug(f"Registered backends: {BackendRegistry.list_backends()}")

__all__ = [
    # Base classes
    "BackendCapabilities",
    "BackendRegistry",
    "BaseChatBackend",
    "ChatMessage",
    # CLI backends
    "ClaudeCliBackend",
    "ClaudeOneShotBackend",
    "GeminiCliBackend",
    "GeminiOneShotBackend",
    # Multi-provider backends
    "OllamaBackend",
    "AnthropicBackend",
    "OpenAIBackend",
    # Utilities
    "HistoryBackend",
    "PtyWrapper",
    "wrap_with_history",
]
