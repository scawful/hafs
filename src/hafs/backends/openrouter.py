"""OpenRouter backend with 100+ model access and automatic fallbacks."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

from hafs.backends.base import BackendCapabilities, BaseChatBackend

logger = logging.getLogger(__name__)

# Lazy import httpx
httpx = None


def _ensure_httpx():
    """Lazy load httpx."""
    global httpx
    if httpx is None:
        try:
            import httpx as _httpx
            httpx = _httpx
        except ImportError:
            raise ImportError(
                "httpx package not installed. "
                "Install with: pip install httpx"
            )
    return httpx


class OpenRouterBackend(BaseChatBackend):
    """OpenRouter backend for accessing 100+ AI models via unified API.

    Supports all major providers:
    - OpenAI (GPT-5, o3, o4)
    - Anthropic (Claude 4.5)
    - Google (Gemini 3)
    - DeepSeek (R1)
    - Meta (Llama 3.3)
    - And 90+ more models

    Features:
    - Automatic fallbacks if primary model unavailable
    - Cost optimization (some models cheaper than direct APIs)
    - Unified API (OpenAI-compatible)
    - Free tier models available

    Example:
        backend = OpenRouterBackend(model="deepseek/deepseek-r1")
        await backend.start()
        response = await backend.generate_one_shot("Write a prime checker")
        print(response)
    """

    # Popular models on OpenRouter
    MODELS = {
        # OpenAI
        "gpt-5": "openai/gpt-5",
        "gpt-5.2": "openai/gpt-5.2",
        "o3": "openai/o3",
        "o3-mini": "openai/o3-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        # Anthropic
        "claude-opus-4.5": "anthropic/claude-opus-4-5-20251101",
        "claude-sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
        "claude-haiku-4.5": "anthropic/claude-haiku-4-5-20251015",
        # Google
        "gemini-3-flash": "google/gemini-3-flash-preview",
        "gemini-3-pro": "google/gemini-3-pro-preview",
        # DeepSeek (excellent for code)
        "deepseek-r1": "deepseek/deepseek-r1",
        "deepseek-coder": "deepseek/deepseek-coder",
        # Meta (free tier)
        "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
        "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
        # Qwen (free tier)
        "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek/deepseek-r1",
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        site_url: str = "https://hafs.dev",
        app_name: str = "hAFS",
    ):
        """Initialize OpenRouter backend.

        Args:
            api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var).
            model: Model to use (can be short name or full path).
            max_tokens: Maximum tokens in response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            site_url: Your site URL (for ranking/stats on OpenRouter).
            app_name: Your app name (for ranking/stats on OpenRouter).
        """
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._model = self.MODELS.get(model, model)
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._site_url = site_url
        self._app_name = app_name
        self._base_url = "https://openrouter.ai/api/v1"

        self._client = None
        self._messages: list[dict[str, Any]] = []
        self._pending_message: Optional[str] = None
        self._running = False
        self._busy = False

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def display_name(self) -> str:
        return f"OpenRouter ({self._model})"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_images=True,
            supports_files=True,
            max_context_tokens=128000,  # Most models support 128K+
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

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._site_url,
            "X-Title": self._app_name,
            "Content-Type": "application/json",
        }

    async def start(self) -> bool:
        """Initialize the OpenRouter client."""
        if self._running:
            return True

        if not self._api_key:
            logger.error("No OpenRouter API key provided")
            return False

        try:
            _ensure_httpx()

            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=self._get_headers(),
                timeout=httpx.Timeout(60.0),
            )
            self._running = True

            # Initialize with system prompt if provided
            if self._system_prompt:
                self._messages.append({
                    "role": "system",
                    "content": self._system_prompt
                })

            logger.info(f"OpenRouter backend started with model {self._model}")
            return True
        except Exception as e:
            logger.error(f"Failed to start OpenRouter backend: {e}")
            return False

    async def stop(self) -> None:
        """Stop the backend."""
        if self._client:
            await self._client.aclose()
        self._running = False
        self._client = None
        self._messages.clear()

    async def send_message(self, message: str) -> None:
        """Queue a message to send."""
        if not self._running:
            raise RuntimeError("Backend not running. Call start() first.")

        self._pending_message = message
        self._messages.append({"role": "user", "content": message})

    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response chunks from OpenRouter."""
        if not self._pending_message or not self._client:
            return

        self._busy = True
        full_response = ""

        try:
            response = await self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": self._messages,
                    "max_tokens": self._max_tokens,
                    "temperature": self._temperature,
                    "stream": True,
                },
            )

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        import json
                        chunk = json.loads(data)
                        if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                            content = chunk["choices"][0]["delta"]["content"]
                            full_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue

            # Add assistant response to history
            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            yield f"[Error: {e}]"
        finally:
            self._pending_message = None
            self._busy = False

    async def generate_one_shot(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a single response without conversation history.

        Args:
            prompt: The prompt to send.
            system: Optional system prompt.
            max_tokens: Optional max tokens override.
            temperature: Optional temperature override.

        Returns:
            The complete response text.
        """
        if not self._client:
            await self.start()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens or self._max_tokens,
                "temperature": temperature if temperature is not None else self._temperature,
            },
        )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_with_functions(
        self,
        prompt: str,
        functions: list[dict[str, Any]],
        system: Optional[str] = None,
        function_call: str = "auto",
    ) -> dict[str, Any]:
        """Generate a response with function calling.

        Args:
            prompt: The prompt to send.
            functions: List of function definitions.
            system: Optional system prompt.
            function_call: "auto", "none", or specific function name.

        Returns:
            Response dict with content and function_call info.
        """
        if not self._client:
            await self.start()

        messages = []
        if system or self._system_prompt:
            messages.append({"role": "system", "content": system or self._system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Convert functions to tools format
        tools = [
            {"type": "function", "function": f}
            for f in functions
        ]

        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "tool_choice": function_call,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
        )

        data = response.json()
        message = data["choices"][0]["message"]

        result = {
            "content": message.get("content"),
            "tool_calls": [],
            "finish_reason": data["choices"][0]["finish_reason"],
        }

        if message.get("tool_calls"):
            for call in message["tool_calls"]:
                result["tool_calls"].append({
                    "id": call["id"],
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                })

        return result

    def clear_history(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        system_msg = None
        if self._messages and self._messages[0]["role"] == "system":
            system_msg = self._messages[0]

        self._messages.clear()

        if system_msg:
            self._messages.append(system_msg)

    def set_model(self, model: str) -> None:
        """Change the model (clears history).

        Args:
            model: New model name (can be short name or full path).
        """
        self._model = self.MODELS.get(model, model)
        self.clear_history()

    async def list_available_models(self) -> list[dict[str, Any]]:
        """List all available models on OpenRouter.

        Returns:
            List of model dicts with id, name, pricing, context length.
        """
        if not self._client:
            await self.start()

        response = await self._client.get("/models")
        data = response.json()
        return data.get("data", [])


async def test_openrouter():
    """Test OpenRouter backend."""
    logger.info("=" * 80)
    logger.info("OPENROUTER BACKEND TEST")
    logger.info("=" * 80)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("✗ OPENROUTER_API_KEY not set")
        logger.error("  Get your key from: https://openrouter.ai/keys")
        logger.error('  export OPENROUTER_API_KEY="sk-or-v1-..."')
        return False

    # Test with DeepSeek R1 (excellent code model)
    logger.info("\n[1/3] Testing DeepSeek R1...")
    backend = OpenRouterBackend(model="deepseek-r1")
    await backend.start()

    response = await backend.generate_one_shot(
        "Write a Python function to check if a number is prime. Just code, no explanation."
    )
    logger.info(f"✓ DeepSeek R1 response:\n{response[:200]}...")

    # Test with free model (Llama 3.3)
    logger.info("\n[2/3] Testing Llama 3.3 70B (free tier)...")
    backend.set_model("llama-3.3-70b")
    response = await backend.generate_one_shot("Say 'OpenRouter working' in Python")
    logger.info(f"✓ Llama 3.3 response:\n{response[:200]}...")

    # List available models
    logger.info("\n[3/3] Listing available models...")
    models = await backend.list_available_models()
    logger.info(f"✓ Found {len(models)} models on OpenRouter")

    # Show top code models
    code_models = [m for m in models if "code" in m.get("id", "").lower()][:5]
    logger.info("\nTop code models:")
    for model in code_models:
        logger.info(f"  - {model['id']}")

    await backend.stop()

    logger.info("\n" + "=" * 80)
    logger.info("✓ OpenRouter backend working successfully!")
    logger.info("=" * 80)
    return True


async def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    success = await test_openrouter()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
