"""Ollama backend for local and remote model inference."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

from hafs.backends.base import BackendCapabilities, BaseChatBackend

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

# Lazy import aiohttp
_aiohttp = None


def _ensure_aiohttp():
    """Lazy load aiohttp."""
    global _aiohttp
    if _aiohttp is None:
        try:
            import aiohttp as _aio
            _aiohttp = _aio
        except ImportError:
            raise ImportError(
                "aiohttp package not installed. "
                "Install with: pip install aiohttp"
            )
    return _aiohttp


class OllamaBackend(BaseChatBackend):
    """Ollama backend supporting local and remote (Tailscale) nodes.

    Connects to Ollama API over HTTP for model inference.
    Supports streaming responses via NDJSON.

    Example:
        # Local Ollama
        backend = OllamaBackend(model="llama3:8b")

        # Remote GPU node via Tailscale
        backend = OllamaBackend(
            host="100.x.x.x",  # Tailscale IP
            port=11434,
            model="codellama:34b"
        )
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama3:8b",
        timeout: float = 300.0,
    ):
        """Initialize Ollama backend.

        Args:
            host: Ollama server host (IP or hostname).
            port: Ollama server port (default 11434).
            model: Model name to use (e.g., "llama3:8b", "codellama:34b").
            timeout: Request timeout in seconds.
        """
        self._host = host
        self._port = port
        self._model = model
        self._timeout = timeout
        self._base_url = f"http://{host}:{port}"

        self._session: Optional[aiohttp.ClientSession] = None
        self._messages: list[dict[str, str]] = []
        self._pending_message: Optional[str] = None
        self._running = False
        self._busy = False
        self._context_injection: Optional[str] = None

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return f"Ollama ({self._model})"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=False,  # Basic Ollama doesn't support tools
            supports_images=False,  # Depends on model (llava does)
            supports_files=True,
            max_context_tokens=self._get_context_size(),
        )

    def _get_context_size(self) -> int:
        """Estimate context size based on model."""
        model_lower = self._model.lower()
        if "70b" in model_lower or "72b" in model_lower:
            return 8192
        elif "34b" in model_lower:
            return 16384
        elif "llama3" in model_lower:
            return 8192
        elif "mistral" in model_lower:
            return 32768
        elif "codellama" in model_lower:
            return 16384
        return 4096  # Conservative default

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def host(self) -> str:
        """Get the configured host."""
        return self._host

    @property
    def model(self) -> str:
        """Get the configured model."""
        return self._model

    async def start(self) -> bool:
        """Start the backend by creating HTTP session and verifying connection."""
        if self._running:
            return True

        try:
            aiohttp = _ensure_aiohttp()
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Verify Ollama is accessible
            async with self._session.get(f"{self._base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    logger.info(f"Connected to Ollama at {self._base_url}")
                    logger.info(f"Available models: {models}")
                    self._running = True
                    return True
                else:
                    logger.error(f"Ollama returned status {resp.status}")
                    return False

        except Exception as e:
            # Check if it's a connection error
            if "ClientConnectorError" in type(e).__name__:
                logger.error(f"Cannot connect to Ollama at {self._base_url}: {e}")
            else:
                logger.error(f"Failed to start Ollama backend: {e}")
            return False

    async def stop(self) -> None:
        """Stop the backend and close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._running = False
        self._messages.clear()

    async def send_message(self, message: str) -> None:
        """Queue a message to send to Ollama."""
        if not self._running or not self._session:
            raise RuntimeError("Backend not running. Call start() first.")

        # Apply context injection if present
        if self._context_injection:
            message = f"{self._context_injection}\n\n{message}"
            self._context_injection = None

        self._pending_message = message
        self._messages.append({"role": "user", "content": message})

    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response chunks from Ollama."""
        if not self._pending_message or not self._session:
            return

        self._busy = True
        full_response = ""

        try:
            payload = {
                "model": self._model,
                "messages": self._messages,
                "stream": True,
            }

            async with self._session.post(
                f"{self._base_url}/api/chat",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Ollama error {resp.status}: {error_text}")
                    yield f"[Error: Ollama returned {resp.status}]"
                    return

                # Stream NDJSON response
                async for line in resp.content:
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))

                        # Extract content from message
                        message = data.get("message", {})
                        content = message.get("content", "")

                        if content:
                            full_response += content
                            yield content

                        # Check if done
                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        continue

            # Add assistant response to history
            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except asyncio.TimeoutError:
            logger.error("Ollama request timed out")
            yield "[Error: Request timed out]"
        except Exception as e:
            # Handle aiohttp client errors
            if "ClientError" in type(e).__name__ or "aiohttp" in type(e).__module__:
                logger.error(f"Ollama request failed: {e}")
                yield f"[Error: {e}]"
            else:
                raise
        finally:
            self._pending_message = None
            self._busy = False

    async def inject_context(self, context: str) -> None:
        """Inject context into the next message."""
        self._context_injection = context

    async def generate_one_shot(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate a single response without conversation history.

        Useful for simple queries that don't need context.

        Args:
            prompt: The prompt to send.
            system: Optional system prompt.

        Returns:
            The complete response text.
        """
        if not self._session:
            await self.start()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        async with self._session.post(
            f"{self._base_url}/api/chat",
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"Ollama error {resp.status}: {error_text}")

            data = await resp.json()
            return data.get("message", {}).get("content", "")

    async def list_models(self) -> list[dict[str, Any]]:
        """List available models on this Ollama instance.

        Returns:
            List of model info dicts with name, size, etc.
        """
        if not self._session:
            await self.start()

        async with self._session.get(f"{self._base_url}/api/tags") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("models", [])
            return []

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model_name: Name of model to pull (e.g., "llama3:8b").

        Returns:
            True if pull succeeded.
        """
        if not self._session:
            await self.start()

        payload = {"name": model_name}

        try:
            async with self._session.post(
                f"{self._base_url}/api/pull",
                json=payload,
            ) as resp:
                # Stream the pull progress
                async for line in resp.content:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        status = data.get("status", "")
                        if "pulling" in status or "downloading" in status:
                            logger.info(f"Pull progress: {status}")
                    except json.JSONDecodeError:
                        continue

                return resp.status == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    async def check_health(self) -> dict[str, Any]:
        """Check health of this Ollama instance.

        Returns:
            Health status dict with version, models, etc.
        """
        if not self._session:
            try:
                await self.start()
            except Exception:
                return {"status": "offline", "error": "Cannot connect"}

        try:
            # Get version
            async with self._session.get(f"{self._base_url}/api/version") as resp:
                if resp.status == 200:
                    version = await resp.json()
                else:
                    version = {"version": "unknown"}

            # Get models
            models = await self.list_models()

            return {
                "status": "online",
                "host": self._host,
                "port": self._port,
                "version": version.get("version", "unknown"),
                "models": [m.get("name") for m in models],
                "model_count": len(models),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def set_model(self, model: str) -> None:
        """Change the model (clears history).

        Args:
            model: New model name.
        """
        self._model = model
        self._messages.clear()
