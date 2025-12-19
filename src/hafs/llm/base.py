"""Base LLM backend interface.

Defines the abstract interface for LLM backends used by analysis modes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for an LLM backend."""

    host: str = "http://localhost:11434"
    model: str = "llama3"
    embed_model: str = "nomic-embed-text"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9

    # Timeout
    timeout_seconds: int = 60

    # Extension point
    extensions: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Response from an LLM backend."""

    text: str
    model: str
    done: bool = True

    # Usage statistics
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Timing
    duration_ms: Optional[int] = None

    # Metadata
    stop_reason: Optional[str] = None
    extensions: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    """Response containing embeddings."""

    embedding: list[float]
    model: str
    dimensions: int

    # Timing
    duration_ms: Optional[int] = None


class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends.

    Backends provide text generation and embedding capabilities
    for use in analysis modes.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """Initialize the backend.

        Args:
            config: Optional configuration.
        """
        self.config = config or LLMConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass

    @property
    def is_local(self) -> bool:
        """Whether this backend runs locally (privacy-preserving)."""
        return False

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            **kwargs: Additional generation parameters.

        Returns:
            LLMResponse with generated text.
        """
        pass

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text generation.

        Default implementation falls back to non-streaming generate.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            **kwargs: Additional generation parameters.

        Yields:
            Text chunks as they're generated.
        """
        response = await self.generate(prompt, system, **kwargs)
        yield response.text

    async def embed(
        self,
        text: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings for text.

        Args:
            text: Text to embed.
            **kwargs: Additional parameters.

        Returns:
            EmbeddingResponse with embedding vector.

        Raises:
            NotImplementedError: If backend doesn't support embeddings.
        """
        raise NotImplementedError(f"{self.name} does not support embeddings")

    async def health_check(self) -> bool:
        """Check if backend is available.

        Returns:
            True if backend is healthy and responding.
        """
        try:
            await self.generate("Hello", max_tokens=5)
            return True
        except Exception:
            return False


class LLMBackendRegistry:
    """Registry for LLM backends."""

    _backends: dict[str, type[BaseLLMBackend]] = {}
    _instances: dict[str, BaseLLMBackend] = {}

    @classmethod
    def register(cls, backend_class: type[BaseLLMBackend]) -> None:
        """Register a backend class."""
        instance = backend_class()
        cls._backends[instance.name] = backend_class

    @classmethod
    def get(
        cls,
        name: str,
        config: Optional[LLMConfig] = None,
    ) -> Optional[BaseLLMBackend]:
        """Get or create a backend instance.

        Args:
            name: Backend name.
            config: Optional configuration.

        Returns:
            Backend instance, or None if not found.
        """
        cache_key = f"{name}:{id(config)}" if config else name

        if cache_key not in cls._instances and name in cls._backends:
            cls._instances[cache_key] = cls._backends[name](config)

        return cls._instances.get(cache_key)

    @classmethod
    def list_backends(cls) -> list[str]:
        """List registered backend names."""
        return list(cls._backends.keys())

    @classmethod
    def list_local_backends(cls) -> list[str]:
        """List backends that run locally."""
        return [
            name for name, backend_class in cls._backends.items()
            if backend_class().is_local
        ]
