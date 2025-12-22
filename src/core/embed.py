"""EmbeddingGemma integration for local embedding generation.

EmbeddingGemma is Google's 300M parameter embedding model that runs locally
via Ollama. It produces 768-dim embeddings and supports 100+ languages.

Usage:
    from core.embed import EmbeddingGemma

    # Using Ollama (recommended)
    gemma = EmbeddingGemma()
    await gemma.initialize()

    embedding = await gemma.embed("Hello world")  # noqa: W605
    embeddings = await gemma.embed_batch(["Hello", "World"], batch_size=32)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import NDArray

# Default EmbeddingGemma model names in Ollama
EMBEDDINGGEMMA_MODELS = [
    "embeddinggemma",
    "embeddinggemma:latest",
    "embeddinggemma:300m",
]


class EmbeddingGemma:
    """EmbeddingGemma embedding model via Ollama.

    Provides efficient local embedding generation using Google's
    EmbeddingGemma model (300M parameters, 768-dim output).

    Args:
        model: Ollama model name (default: "embeddinggemma")
        host: Ollama host URL (default: "http://localhost:11434")
        fallback_model: Fallback model if EmbeddingGemma unavailable
    """

    def __init__(
        self,
        model: str = "embeddinggemma",
        host: str = "http://localhost:11434",
        fallback_model: str = "nomic-embed-text",
    ):
        self._model = model
        self._host = host
        self._fallback_model = fallback_model
        self._active_model: Optional[str] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize and verify model availability.

        Returns:
            True if EmbeddingGemma is available, False if using fallback
        """
        self._active_model = await self._find_available_model()
        self._initialized = True
        return self._active_model in EMBEDDINGGEMMA_MODELS

    async def _find_available_model(self) -> str:
        """Find an available embedding model."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Check if Ollama is running
                async with session.get(
                    f"{self._host}/api/tags", timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status != 200:
                        return self._fallback_model

                    data = await resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]

                    # Check for EmbeddingGemma variants
                    for model in models:
                        if model.startswith("embeddinggemma"):
                            return model

                    # Check for requested model
                    if self._model in models:
                        return self._model

                    # Use fallback
                    if self._fallback_model in models:
                        return self._fallback_model

                    return self._fallback_model

        except Exception:
            return self._fallback_model

    @property
    def model(self) -> str:
        """Currently active model name."""
        return self._active_model or self._model

    @property
    def is_embeddinggemma(self) -> bool:
        """Whether using EmbeddingGemma model."""
        return (
            self._active_model is not None
            and self._active_model.startswith("embeddinggemma")
        )

    async def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            768-dim float32 embedding vector
        """
        if not self._initialized:
            await self.initialize()

        try:
            import aiohttp

            payload = {
                "model": self._active_model,
                "prompt": text,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._host}/api/embeddings",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Ollama error: {resp.status}")

                    data = await resp.json()
                    embedding = data.get("embedding", [])
                    return np.array(embedding, dtype=np.float32)

        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}") from e

    async def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[NDArray[np.float32]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (for progress tracking)

        Returns:
            List of 768-dim float32 embedding vectors
        """
        if not self._initialized:
            await self.initialize()

        results = []
        for text in texts:
            embedding = await self.embed(text)
            results.append(embedding)

        return results

    def get_info(self) -> Dict[str, str]:
        """Get embedding model information.

        Returns:
            Dict with model info
        """
        return {
            "model": self._active_model or self._model,
            "host": self._host,
            "is_embeddinggemma": str(self.is_embeddinggemma),
            "initialized": str(self._initialized),
            "dimension": "768",
        }


async def check_embeddinggemma_available(host: str = "http://localhost:11434") -> bool:
    """Check if EmbeddingGemma is available via Ollama.

    Args:
        host: Ollama host URL

    Returns:
        True if EmbeddingGemma model is available
    """
    gemma = EmbeddingGemma(host=host)
    return await gemma.initialize()


async def pull_embeddinggemma(host: str = "http://localhost:11434") -> bool:
    """Pull EmbeddingGemma model via Ollama.

    Args:
        host: Ollama host URL

    Returns:
        True if pull succeeded
    """
    try:
        import aiohttp

        payload = {"name": "embeddinggemma"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{host}/api/pull",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600),  # 10 min for download
            ) as resp:
                if resp.status != 200:
                    return False

                # Stream response to track progress
                async for line in resp.content:
                    pass  # Could parse progress here

                return True

    except Exception:
        return False
