"""Anthropic Claude backend with streaming support."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

from backends.base import BackendCapabilities, BaseChatBackend

logger = logging.getLogger(__name__)

# Lazy import anthropic SDK
anthropic = None


def _ensure_anthropic():
    """Lazy load anthropic SDK."""
    global anthropic
    if anthropic is None:
        try:
            import anthropic as _anthropic
            anthropic = _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
    return anthropic


class AnthropicBackend(BaseChatBackend):
    """Anthropic Claude backend using the Messages API.

    Supports streaming responses and message history accumulation.

    Example:
        backend = AnthropicBackend(model="claude-opus-4.5")
        await backend.start()
        await backend.send_message("Hello!")
        async for chunk in backend.stream_response():
            print(chunk, end="")
    """

    # Model aliases -> API model IDs
    # Use names from core.models.registry
    MODELS = {
        # Current models (December 2025)
        "claude-opus-4.5": "claude-opus-4-5-20251101",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-haiku-3.5": "claude-3-5-haiku-20241022",
        # Legacy aliases (deprecated)
        "claude-3-5-sonnet": "claude-sonnet-4-20250514",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-opus-4-5-20251101",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5-20251101",
        max_tokens: int = 8192,
        system_prompt: Optional[str] = None,
    ):
        """Initialize Anthropic backend.

        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var).
            model: Model to use (full name or alias like "claude-3-5-sonnet").
            max_tokens: Maximum tokens in response.
            system_prompt: Optional system prompt.
        """
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = self.MODELS.get(model, model)  # Resolve aliases
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

        self._client = None
        self._messages: list[dict[str, str]] = []
        self._pending_message: Optional[str] = None
        self._running = False
        self._busy = False
        self._context_injection: Optional[str] = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        # Extract short model name
        short_name = self._model.split("-")[1:3]  # e.g., ["3", "5"]
        return f"Claude {'.'.join(short_name)}"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=True,
            supports_tool_use=True,
            supports_images=True,
            supports_files=True,
            max_context_tokens=self._get_context_size(),
        )

    def _get_context_size(self) -> int:
        """Get context size for current model."""
        if "opus" in self._model:
            return 200000
        elif "sonnet" in self._model:
            return 200000
        elif "haiku" in self._model:
            return 200000
        return 200000  # All Claude 3+ models have 200K

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def model(self) -> str:
        """Get the configured model."""
        return self._model

    async def start(self) -> bool:
        """Initialize the Anthropic client."""
        if self._running:
            return True

        if not self._api_key:
            logger.error("No Anthropic API key provided")
            return False

        try:
            _ensure_anthropic()
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
            self._running = True
            logger.info(f"Anthropic backend started with model {self._model}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Anthropic backend: {e}")
            return False

    async def stop(self) -> None:
        """Stop the backend."""
        self._running = False
        self._client = None
        self._messages.clear()

    async def send_message(self, message: str) -> None:
        """Queue a message to send."""
        if not self._running:
            raise RuntimeError("Backend not running. Call start() first.")

        # Apply context injection if present
        if self._context_injection:
            message = f"{self._context_injection}\n\n{message}"
            self._context_injection = None

        self._pending_message = message
        self._messages.append({"role": "user", "content": message})

    async def stream_response(self) -> AsyncGenerator[str, None]:
        """Stream response chunks from Claude."""
        if not self._pending_message or not self._client:
            return

        self._busy = True
        full_response = ""

        try:
            # Build request
            kwargs = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": self._messages,
            }

            if self._system_prompt:
                kwargs["system"] = self._system_prompt

            # Stream response
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    yield text

            # Add assistant response to history
            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            yield f"[Error: {e.message}]"
        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
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
    ) -> str:
        """Generate a single response without conversation history.

        Args:
            prompt: The prompt to send.
            system: Optional system prompt override.
            max_tokens: Optional max tokens override.

        Returns:
            The complete response text.
        """
        if not self._client:
            await self.start()

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system or self._system_prompt:
            kwargs["system"] = system or self._system_prompt

        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system: Optional[str] = None,
    ) -> dict[str, Any]:
        """Generate a response with tool use.

        Args:
            prompt: The prompt to send.
            tools: List of tool definitions.
            system: Optional system prompt.

        Returns:
            Response dict with content and tool_use info.
        """
        if not self._client:
            await self.start()

        kwargs = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
        }

        if system or self._system_prompt:
            kwargs["system"] = system or self._system_prompt

        response = await self._client.messages.create(**kwargs)

        result = {
            "content": [],
            "tool_use": [],
            "stop_reason": response.stop_reason,
        }

        for block in response.content:
            if block.type == "text":
                result["content"].append(block.text)
            elif block.type == "tool_use":
                result["tool_use"].append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        return result

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def set_model(self, model: str) -> None:
        """Change the model (clears history).

        Args:
            model: New model name or alias.
        """
        self._model = self.MODELS.get(model, model)
        self._messages.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt.

        Args:
            prompt: New system prompt.
        """
        self._system_prompt = prompt

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Note: This is an approximation. For exact counts,
        use the tokenizer directly.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Rough approximation: ~4 chars per token for English
        return len(text) // 4
