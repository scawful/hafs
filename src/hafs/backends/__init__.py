"""Chat backend module for AI agent orchestration.

DEPRECATED: This module re-exports from the new 'backends' package.
Please import directly from 'backends' instead.

Example:
    # Old (deprecated):
    from hafs.backends import BackendRegistry

    # New (preferred):
    from backends import BackendRegistry
"""

import importlib
import warnings

_DEPRECATION_MESSAGE = "hafs.backends is deprecated. Import from 'backends' instead."

warnings.warn(
    _DEPRECATION_MESSAGE,
    DeprecationWarning,
    stacklevel=2,
)

_canonical_backends = importlib.import_module("backends")
if not _canonical_backends.BackendRegistry.list_backends():
    importlib.reload(_canonical_backends)

_EXPORTS = [
    # Base classes
    ("BackendCapabilities", "backends.base", "BackendCapabilities"),
    ("BackendRegistry", "backends.base", "BackendRegistry"),
    ("BaseChatBackend", "backends.base", "BaseChatBackend"),
    ("ChatMessage", "backends.base", "ChatMessage"),
    # CLI backends
    ("ClaudeCliBackend", "backends.cli.claude", "ClaudeCliBackend"),
    ("ClaudeResponseParser", "backends.cli.claude", "ClaudeResponseParser"),
    ("GeminiCliBackend", "backends.cli.gemini", "GeminiCliBackend"),
    ("GeminiResponseParser", "backends.cli.gemini", "GeminiResponseParser"),
    ("strip_ansi", "backends.cli.gemini", "strip_ansi"),
    ("PtyOptions", "backends.cli.pty", "PtyOptions"),
    ("PtyWrapper", "backends.cli.pty", "PtyWrapper"),
    # API backends
    ("AnthropicBackend", "backends.api.anthropic", "AnthropicBackend"),
    ("LlamaCppBackend", "backends.api.llamacpp", "LlamaCppBackend"),
    ("OllamaBackend", "backends.api.ollama", "OllamaBackend"),
    ("OpenAIBackend", "backends.api.openai", "OpenAIBackend"),
    # One-shot backends
    ("ClaudeOneShotBackend", "backends.oneshot.claude", "ClaudeOneShotBackend"),
    ("GeminiOneShotBackend", "backends.oneshot.gemini", "GeminiOneShotBackend"),
    # Wrappers
    ("HistoryBackend", "backends.wrappers.history", "HistoryBackend"),
    ("wrap_with_history", "backends.wrappers.history", "wrap_with_history"),
]

_EXPORT_MAP = {name: (module_path, attr) for name, module_path, attr in _EXPORTS}


def __getattr__(name: str):
    if name in _EXPORT_MAP:
        warnings.warn(
            _DEPRECATION_MESSAGE,
            DeprecationWarning,
            stacklevel=2,
        )
        module_path, attr = _EXPORT_MAP[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_EXPORT_MAP.keys()))


__all__ = [name for name, _, _ in _EXPORTS]
