"""LLM Backend Module for hafs.

Provides pluggable LLM backends for analysis modes per PROTOCOL_SPEC.md Section 5.5.
Supports local (Ollama, llama.cpp) and cloud providers.
"""

from llm.base import BaseLLMBackend, LLMResponse, LLMConfig
from llm.ollama import OllamaBackend

__all__ = [
    "BaseLLMBackend",
    "LLMConfig",
    "LLMResponse",
    "OllamaBackend",
]
