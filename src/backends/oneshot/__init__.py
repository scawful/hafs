"""One-shot (non-PTY) CLI backends for fast headless answers."""

from backends.oneshot.claude import ClaudeOneShotBackend
from backends.oneshot.gemini import GeminiOneShotBackend

__all__ = [
    "ClaudeOneShotBackend",
    "GeminiOneShotBackend",
]
