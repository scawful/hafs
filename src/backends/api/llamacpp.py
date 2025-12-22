"""Llama.cpp backend using the OpenAI-compatible REST API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

from backends.base import BackendCapabilities, BaseChatBackend

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

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


def _normalize_base_url(base_url: str) -> str:
    """Normalize base URL to include /v1."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _parse_int(value: Optional[str], fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


class LlamaCppBackend(BaseChatBackend):
    """Llama.cpp backend using the OpenAI-compatible API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        timeout: float = 300.0,
        api_key: Optional[str] = None,
        context_size: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.05,
        repeat_penalty: float = 1.1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        mirostat: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        stop: Optional[list[str]] = None,
    ):
        env_base_url = (
            os.environ.get("LLAMACPP_BASE_URL")
            or os.environ.get("LLAMA_CPP_BASE_URL")
        )
        if base_url is None:
            base_url = env_base_url

        if not base_url:
            host = (
                host
                or os.environ.get("LLAMACPP_HOST")
                or os.environ.get("LLAMA_CPP_HOST")
                or "localhost"
            )
            port_value = (
                port
                or os.environ.get("LLAMACPP_PORT")
                or os.environ.get("LLAMA_CPP_PORT")
                or "11435"
            )
            base_url = f"http://{host}:{port_value}"

        self._base_url = _normalize_base_url(base_url)
        self._api_key = (
            api_key
            or os.environ.get("LLAMACPP_API_KEY")
            or os.environ.get("LLAMA_CPP_API_KEY")
        )
        self._model = (
            model
            or os.environ.get("LLAMACPP_MODEL")
            or os.environ.get("LLAMA_CPP_MODEL")
            or "qwen3-14b"
        )
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._timeout = timeout
        self._max_context_tokens = context_size or _parse_int(
            os.environ.get("LLAMACPP_CTX") or os.environ.get("LLAMA_CTX"),
            8192,
        )

        # Advanced sampling parameters
        self._top_p = top_p
        self._top_k = top_k
        self._min_p = min_p
        self._repeat_penalty = repeat_penalty
        self._presence_penalty = presence_penalty
        self._frequency_penalty = frequency_penalty
        self._mirostat = mirostat
        self._mirostat_tau = mirostat_tau
        self._mirostat_eta = mirostat_eta
        self._stop = stop


        self._session: Optional[aiohttp.ClientSession] = None
        self._messages: list[dict[str, str]] = []
        self._pending_message: Optional[str] = None
        self._running = False
        self._busy = False
        self._context_injection: Optional[str] = None
        self._available_models: list[str] = []

    @property
    def name(self) -> str:
        return "llamacpp"

    @property
    def display_name(self) -> str:
        return f"llama.cpp ({self._model})"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=False,
            supports_images=False,
            supports_files=True,
            max_context_tokens=self._max_context_tokens,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def model(self) -> str:
        return self._model

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def start(self) -> bool:
        """Start the backend by creating HTTP session and verifying connection."""
        if self._running:
            return True

        try:
            aiohttp = _ensure_aiohttp()
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._build_headers(),
            )

            async with self._session.get(f"{self._base_url}/models") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._available_models = self._extract_models(data)
                    self._running = True
                    if self._system_prompt:
                        self._messages.append(
                            {"role": "system", "content": self._system_prompt}
                        )
                    logger.info(f"Connected to llama.cpp at {self._base_url}")
                    return True

                error_text = await resp.text()
                logger.error(
                    f"llama.cpp returned status {resp.status}: {error_text}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to start llama.cpp backend: {e}")
            return False

    async def stop(self) -> None:
        """Stop the backend and close HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
        self._running = False
        self._messages.clear()

    def _extract_models(self, data: dict[str, Any]) -> list[str]:
        models: list[str] = []
        if isinstance(data.get("data"), list):
            for item in data.get("data", []):
                model_id = item.get("id") or item.get("model") or item.get("name")
                if model_id:
                    models.append(model_id)
        elif isinstance(data.get("models"), list):
            for item in data.get("models", []):
                model_id = item.get("name") or item.get("model") or item.get("id")
                if model_id:
                    models.append(model_id)
        return models

    async def send_message(self, message: str) -> None:
        """Queue a message to send to llama.cpp."""
        if not self._running or not self._session:
            raise RuntimeError("Backend not running. Call start() first.")

        if self._context_injection:
            message = f"{self._context_injection}\n\n{message}"
            self._context_injection = None

        self._pending_message = message
        self._messages.append({"role": "user", "content": message})

    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response chunks from llama.cpp."""
        if not self._pending_message or not self._session:
            return

        self._busy = True
        full_response = ""

        payload = {
            "model": self._model,
            "messages": self._messages,
            "stream": True,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "min_p": self._min_p,
            "repeat_penalty": self._repeat_penalty,
            "presence_penalty": self._presence_penalty,
            "frequency_penalty": self._frequency_penalty,
            "mirostat": self._mirostat,
            "mirostat_tau": self._mirostat_tau,
            "mirostat_eta": self._mirostat_eta,
            "stop": self._stop,
        }

        try:
            async with self._session.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"llama.cpp error {resp.status}: {error_text}")
                    yield f"[Error: llama.cpp returned {resp.status}]"
                    return

                async for raw_line in resp.content:
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue

                    data_text = line[len("data:"):].strip()
                    if data_text == "[DONE]":
                        break

                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if not content:
                        content = choices[0].get("message", {}).get("content", "")

                    if content:
                        full_response += content
                        yield content

            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except asyncio.TimeoutError:
            logger.error("llama.cpp request timed out")
            yield "[Error: Request timed out]"
        except Exception as e:
            logger.error(f"llama.cpp request failed: {e}")
            yield f"[Error: {e}]"
        finally:
            self._pending_message = None
            self._busy = False

    async def inject_context(self, context: str) -> None:
        """Inject context into the next message."""
        self._context_injection = context

    async def generate_one_shot(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a single response without conversation history."""
        if not self._session:
            await self.start()

        messages = []
        if system or self._system_prompt:
            messages.append({"role": "system", "content": system or self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": (
                temperature if temperature is not None else self._temperature
            ),
            "max_tokens": max_tokens or self._max_tokens,
            "top_p": self._top_p,
            "top_k": self._top_k,
            "min_p": self._min_p,
            "repeat_penalty": self._repeat_penalty,
            "presence_penalty": self._presence_penalty,
            "frequency_penalty": self._frequency_penalty,
            "mirostat": self._mirostat,
            "mirostat_tau": self._mirostat_tau,
            "mirostat_eta": self._mirostat_eta,
            "stop": self._stop,
        }

        async with self._session.post(
            f"{self._base_url}/chat/completions",
            json=payload,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"llama.cpp error {resp.status}: {error_text}")

            data = await resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")

    async def list_models(self) -> list[str]:
        """List available models on this llama.cpp instance."""
        if not self._session:
            await self.start()

        async with self._session.get(f"{self._base_url}/models") as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return self._extract_models(data)

    async def check_health(self) -> dict[str, Any]:
        """Check health of this llama.cpp instance."""
        if not self._session:
            try:
                await self.start()
            except Exception:
                return {"status": "offline", "error": "Cannot connect"}

        try:
            async with self._session.get(f"{self._base_url}/models") as resp:
                if resp.status != 200:
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                data = await resp.json()
                models = self._extract_models(data)
                return {
                    "status": "online",
                    "base_url": self._base_url,
                    "models": models,
                    "model_count": len(models),
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def set_model(self, model: str) -> None:
        """Change the model (clears history)."""
        self._model = model
        self._messages.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt."""
        self._system_prompt = prompt
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = prompt
        elif self._messages:
            self._messages.insert(0, {"role": "system", "content": prompt})
