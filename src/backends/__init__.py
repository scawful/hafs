"""Chat backend module for AI agent orchestration.

This is the canonical location for backends. For backward compatibility,
backends re-exports from this module.
"""

from __future__ import annotations

import logging

# Base classes
from backends.base import (
    BackendCapabilities,
    BackendRegistry,
    BaseChatBackend,
    ChatMessage,
)

# CLI backends
from backends.cli.claude import ClaudeCliBackend, ClaudeResponseParser
from backends.cli.gemini import GeminiCliBackend, GeminiResponseParser, strip_ansi
from backends.cli.pty import PtyOptions, PtyWrapper

# API backends
from backends.api.anthropic import AnthropicBackend
from backends.api.llamacpp import LlamaCppBackend
from backends.api.ollama import OllamaBackend
from backends.api.openai import OpenAIBackend
from backends.api.halext import HalextBackend

# One-shot backends
from backends.oneshot.claude import ClaudeOneShotBackend
from backends.oneshot.gemini import GeminiOneShotBackend

# Wrappers
from backends.wrappers.history import HistoryBackend, wrap_with_history

logger = logging.getLogger(__name__)

# Register built-in backends at module load time
BackendRegistry.register(GeminiCliBackend)
BackendRegistry.register(ClaudeCliBackend)
BackendRegistry.register(GeminiOneShotBackend)
BackendRegistry.register(ClaudeOneShotBackend)

# Register API backends
BackendRegistry.register(OllamaBackend)
BackendRegistry.register(AnthropicBackend)
BackendRegistry.register(OpenAIBackend)
BackendRegistry.register(LlamaCppBackend)
BackendRegistry.register(HalextBackend)

logger.debug(f"Registered backends: {BackendRegistry.list_backends()}")

__all__ = [
    # Base classes
    "BackendCapabilities",
    "BackendRegistry",
    "BaseChatBackend",
    "ChatMessage",
    # CLI backends
    "ClaudeCliBackend",
    "ClaudeResponseParser",
    "GeminiCliBackend",
    "GeminiResponseParser",
    "PtyOptions",
    "PtyWrapper",
    "strip_ansi",
    # API backends
    "AnthropicBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAIBackend",
    "HalextBackend",
    # One-shot backends
    "ClaudeOneShotBackend",
    "GeminiOneShotBackend",
    # Wrappers
    "HistoryBackend",
    "wrap_with_history",
]
