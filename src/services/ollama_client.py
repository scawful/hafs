"""Ollama client for local AI inference.

Provides async/sync interface to local Ollama API with support for:
- Standard text generation
- Function/tool calling
- Streaming responses
- Model management
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Iterator, Optional

import aiohttp
import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for local Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client.

        Args:
            base_url: Ollama API endpoint (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip("/")
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session.

        Returns:
            aiohttp ClientSession
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def is_available(self) -> bool:
        """Check if Ollama is running.

        Returns:
            True if Ollama API is accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    async def is_available_async(self) -> bool:
        """Check if Ollama is running (async).

        Returns:
            True if Ollama API is accessible
        """
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def list_models(self) -> list[dict[str, Any]]:
        """List available models.

        Returns:
            List of model info dictionaries
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json().get("models", [])
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Generate text using Ollama (synchronous).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions for function calling
            **kwargs: Additional options passed to Ollama

        Returns:
            Response dictionary or None on error
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        # Add tools if provided
        if tools:
            request_data["tools"] = tools

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=300,  # 5 minute timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            return None

    async def generate_async(
        self,
        prompt: str,
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Generate text using Ollama (asynchronous).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions for function calling
            **kwargs: Additional options passed to Ollama

        Returns:
            Response dictionary or None on error
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        # Add tools if provided
        if tools:
            request_data["tools"] = tools

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    return None

        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            return None

    def generate_stream(
        self,
        prompt: str,
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Generate text with streaming (synchronous).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options

        Yields:
            Response chunks
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                stream=True,
                timeout=300,
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to decode chunk: {line}")
            else:
                logger.error(f"Ollama API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to stream from Ollama: {e}")

    async def generate_stream_async(
        self,
        prompt: str,
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Generate text with streaming (asynchronous).

        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options

        Yields:
            Response chunks
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line)
                                yield chunk
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to decode chunk: {line}")
                else:
                    logger.error(f"Ollama API error: {response.status}")

        except Exception as e:
            logger.error(f"Failed to stream from Ollama: {e}")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Chat completion using Ollama (synchronous).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            **kwargs: Additional options

        Returns:
            Response dictionary or None on error
        """
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        if tools:
            request_data["tools"] = tools

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=300,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to call Ollama chat: {e}")
            return None

    async def chat_async(
        self,
        messages: list[dict[str, str]],
        model: str = "qwen2.5:7b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Optional[dict[str, Any]]:
        """Chat completion using Ollama (asynchronous).

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional tool definitions
            **kwargs: Additional options

        Returns:
            Response dictionary or None on error
        """
        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                **kwargs,
            },
        }

        if tools:
            request_data["tools"] = tools

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/chat",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Ollama API error: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Failed to call Ollama chat: {e}")
            return None

    def pull_model(self, model: str) -> bool:
        """Pull/download a model.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model, "stream": False},
                timeout=600,  # 10 minute timeout for downloads
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Synchronous example
    client = OllamaClient()

    if client.is_available():
        print("Ollama is available!")

        # List models
        models = client.list_models()
        print(f"Available models: {[m['name'] for m in models]}")

        # Generate text
        response = client.generate(
            prompt="What is the capital of France?",
            model="qwen2.5:3b",
            max_tokens=100,
        )

        if response:
            print(f"Response: {response['response']}")

    # Async example
    async def async_example():
        client = OllamaClient()

        if await client.is_available_async():
            response = await client.generate_async(
                prompt="Explain Python asyncio in one sentence.",
                model="qwen2.5:3b",
            )

            if response:
                print(f"Async response: {response['response']}")

        await client.close()

    asyncio.run(async_example())
