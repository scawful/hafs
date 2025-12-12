"""Ollama LLM Backend.

Provides local LLM capabilities via Ollama for privacy-preserving analysis.
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator, Optional

from hafs.llm.base import (
    BaseLLMBackend,
    EmbeddingResponse,
    LLMConfig,
    LLMResponse,
)


class OllamaConfig(LLMConfig):
    """Ollama-specific configuration."""

    host: str = "http://localhost:11434"
    model: str = "llama3"
    embed_model: str = "nomic-embed-text"

    # Ollama-specific options
    num_ctx: int = 4096
    num_gpu: int = -1  # -1 = auto
    keep_alive: str = "5m"


class OllamaBackend(BaseLLMBackend):
    """Ollama backend for local LLM inference.

    Uses the Ollama API for text generation and embeddings.
    Runs completely locally for privacy-sensitive analysis.
    """

    def __init__(self, config: Optional[OllamaConfig] = None) -> None:
        """Initialize Ollama backend.

        Args:
            config: Optional Ollama configuration.
        """
        super().__init__(config or OllamaConfig())
        self._config: OllamaConfig = self.config  # type: ignore

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_local(self) -> bool:
        return True

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text using Ollama.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            **kwargs: Additional generation parameters.

        Returns:
            LLMResponse with generated text.
        """
        import urllib.request
        import urllib.error

        start_time = time.time()

        # Build request payload
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._config.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._config.temperature),
                "num_predict": kwargs.get("max_tokens", self._config.max_tokens),
                "top_p": kwargs.get("top_p", self._config.top_p),
                "num_ctx": self._config.num_ctx,
            },
        }

        if system:
            payload["system"] = system

        # Make request
        url = f"{self._config.host}/api/generate"
        data = json.dumps(payload).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(
                req, timeout=self._config.timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))

            duration_ms = int((time.time() - start_time) * 1000)

            return LLMResponse(
                text=result.get("response", ""),
                model=result.get("model", self._config.model),
                done=result.get("done", True),
                prompt_tokens=result.get("prompt_eval_count"),
                completion_tokens=result.get("eval_count"),
                total_tokens=(
                    (result.get("prompt_eval_count") or 0) +
                    (result.get("eval_count") or 0)
                ) or None,
                duration_ms=duration_ms,
                stop_reason=result.get("done_reason"),
            )

        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid response from Ollama: {e}") from e

    async def stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text generation from Ollama.

        Args:
            prompt: The user prompt.
            system: Optional system prompt.
            **kwargs: Additional generation parameters.

        Yields:
            Text chunks as they're generated.
        """
        import urllib.request
        import urllib.error

        # Build request payload
        payload: dict[str, Any] = {
            "model": kwargs.get("model", self._config.model),
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self._config.temperature),
                "num_predict": kwargs.get("max_tokens", self._config.max_tokens),
                "top_p": kwargs.get("top_p", self._config.top_p),
                "num_ctx": self._config.num_ctx,
            },
        }

        if system:
            payload["system"] = system

        # Make streaming request
        url = f"{self._config.host}/api/generate"
        data = json.dumps(payload).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(
                req, timeout=self._config.timeout_seconds
            ) as response:
                for line in response:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        text = chunk.get("response", "")
                        if text:
                            yield text
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue

        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error: {e}") from e

    async def embed(
        self,
        text: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Generate embeddings using Ollama.

        Args:
            text: Text to embed.
            **kwargs: Additional parameters.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        import urllib.request
        import urllib.error

        start_time = time.time()

        payload = {
            "model": kwargs.get("model", self._config.embed_model),
            "prompt": text,
        }

        url = f"{self._config.host}/api/embeddings"
        data = json.dumps(payload).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(
                req, timeout=self._config.timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))

            embedding = result.get("embedding", [])
            duration_ms = int((time.time() - start_time) * 1000)

            return EmbeddingResponse(
                embedding=embedding,
                model=kwargs.get("model", self._config.embed_model),
                dimensions=len(embedding),
                duration_ms=duration_ms,
            )

        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid response from Ollama: {e}") from e

    async def list_models(self) -> list[str]:
        """List available models in Ollama.

        Returns:
            List of model names.
        """
        import urllib.request
        import urllib.error

        url = f"{self._config.host}/api/tags"

        try:
            with urllib.request.urlopen(
                url, timeout=self._config.timeout_seconds
            ) as response:
                result = json.loads(response.read().decode("utf-8"))

            models = result.get("models", [])
            return [m.get("name", "") for m in models]

        except urllib.error.URLError:
            return []
        except json.JSONDecodeError:
            return []

    async def health_check(self) -> bool:
        """Check if Ollama is running.

        Returns:
            True if Ollama is responding.
        """
        import urllib.request
        import urllib.error

        url = f"{self._config.host}/api/tags"

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, Exception):
            return False


# Register with the backend registry
try:
    from hafs.llm.base import LLMBackendRegistry
    LLMBackendRegistry.register(OllamaBackend)
except Exception:
    pass
