"""OpenAI GPT backend with streaming and function calling support."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any, Optional

from hafs.backends.base import BackendCapabilities, BaseChatBackend

logger = logging.getLogger(__name__)

# Lazy import openai SDK
openai = None


def _ensure_openai():
    """Lazy load openai SDK."""
    global openai
    if openai is None:
        try:
            import openai as _openai
            openai = _openai
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
    return openai


class OpenAIBackend(BaseChatBackend):
    """OpenAI GPT backend using the Chat Completions API.

    Supports streaming responses, function calling, and message history.

    Example:
        backend = OpenAIBackend(model="gpt-4-turbo")
        await backend.start()
        await backend.send_message("Hello!")
        async for chunk in backend.stream_response():
            print(chunk, end="")
    """

    MODELS = {
        # GPT-5 series (Dec 2025)
        "gpt-5": "gpt-5",
        "gpt-5.1": "gpt-5.1",
        "gpt-5.2": "gpt-5.2",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5.1-codex": "gpt-5.1-codex",
        "gpt-5.2-codex": "gpt-5.2-codex",
        # o-series reasoning models (Dec 2025)
        "o3": "o3",
        "o3-mini": "o3-mini",
        "o4-mini": "o4-mini",
        # Legacy o1 series
        "o1": "o1",
        "o1-mini": "o1-mini",
        "o1-preview": "o1-preview",
        # GPT-4 series
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI backend.

        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var).
            model: Model to use.
            max_tokens: Maximum tokens in response.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature (0.0-2.0).
            base_url: Optional custom API base URL (for Azure, proxies, etc.).
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = self.MODELS.get(model, model)
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        self._client = None
        self._messages: list[dict[str, Any]] = []
        self._pending_message: Optional[str] = None
        self._running = False
        self._busy = False
        self._context_injection: Optional[str] = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return f"OpenAI ({self._model})"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_streaming=not self._uses_completion_tokens_param(),  # GPT-5/o-series don't stream
            supports_tool_use=not self._model.startswith("o1"),
            supports_images="gpt-4" in self._model or "gpt-5" in self._model or "o" in self._model,
            supports_files=True,
            max_context_tokens=self._get_context_size(),
        )

    def _uses_completion_tokens_param(self) -> bool:
        """Check if model uses max_completion_tokens instead of max_tokens.

        GPT-5, o3, o4 series use max_completion_tokens.
        o1 series also uses max_completion_tokens.
        GPT-4 and earlier use max_tokens.
        """
        return (
            self._model.startswith(("gpt-5", "o1", "o3", "o4"))
        )

    def _supports_temperature(self) -> bool:
        """Check if model supports temperature parameter.

        o1, o3, o4 series don't support temperature.
        GPT-5 and earlier support temperature.
        """
        return not self._model.startswith(("o1", "o3", "o4"))

    def _get_context_size(self) -> int:
        """Get context size for current model."""
        if "gpt-5" in self._model or self._model.startswith(("o3", "o4")):
            return 200000  # GPT-5 and o3/o4 series have 200K context
        elif "gpt-4-turbo" in self._model or "gpt-4o" in self._model:
            return 128000
        elif "gpt-4" in self._model:
            return 8192
        elif "gpt-3.5-turbo" in self._model:
            return 16385
        elif self._model.startswith("o1"):
            return 128000
        return 8192

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
        """Initialize the OpenAI client."""
        if self._running:
            return True

        if not self._api_key and not self._base_url:
            logger.error("No OpenAI API key provided")
            return False
        if not self._api_key and self._base_url:
            # Local OpenAI-compatible endpoints may not require auth.
            self._api_key = "local"

        try:
            _ensure_openai()

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            self._client = openai.AsyncOpenAI(**kwargs)
            self._running = True

            # Initialize with system prompt if provided
            if self._system_prompt:
                self._messages.append({
                    "role": "system",
                    "content": self._system_prompt
                })

            logger.info(f"OpenAI backend started with model {self._model}")
            return True
        except Exception as e:
            logger.error(f"Failed to start OpenAI backend: {e}")
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
        """Stream response chunks from OpenAI."""
        if not self._pending_message or not self._client:
            return

        self._busy = True
        full_response = ""

        try:
            # Build request
            kwargs = {
                "model": self._model,
                "messages": self._messages,
            }

            # GPT-5/o-series models don't support streaming and use different parameters
            if self._uses_completion_tokens_param():
                # GPT-5, o1, o3, o4 models - no streaming, use max_completion_tokens
                kwargs["max_completion_tokens"] = self._max_tokens

                # o1, o3, o4 series don't support temperature
                if self._supports_temperature():
                    kwargs["temperature"] = self._temperature

                response = await self._client.chat.completions.create(**kwargs)

                if response.choices:
                    content = response.choices[0].message.content
                    full_response = content
                    yield content
            else:
                # GPT-4 and earlier - support streaming
                kwargs["max_tokens"] = self._max_tokens
                kwargs["temperature"] = self._temperature
                kwargs["stream"] = True

                # Stream response
                stream = await self._client.chat.completions.create(**kwargs)

                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content

            # Add assistant response to history
            if full_response:
                self._messages.append({"role": "assistant", "content": full_response})

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            yield f"[Error: {e.message}]"
        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
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

        kwargs = {
            "model": self._model,
            "messages": messages,
        }

        if self._uses_completion_tokens_param():
            kwargs["max_completion_tokens"] = max_tokens or self._max_tokens
            # o1, o3, o4 series don't support temperature
            if self._supports_temperature():
                kwargs["temperature"] = temperature if temperature is not None else self._temperature
        else:
            kwargs["max_tokens"] = max_tokens or self._max_tokens
            kwargs["temperature"] = temperature if temperature is not None else self._temperature

        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

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

        kwargs = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": function_call,
        }

        if self._uses_completion_tokens_param():
            kwargs["max_completion_tokens"] = self._max_tokens
            # o1, o3, o4 series don't support temperature
            if self._supports_temperature():
                kwargs["temperature"] = self._temperature
        else:
            kwargs["max_tokens"] = self._max_tokens
            kwargs["temperature"] = self._temperature

        response = await self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        result = {
            "content": message.content,
            "tool_calls": [],
            "finish_reason": response.choices[0].finish_reason,
        }

        if message.tool_calls:
            for call in message.tool_calls:
                result["tool_calls"].append({
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                })

        return result

    async def generate_embeddings(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed.
            model: Embedding model to use.

        Returns:
            List of embedding vectors.
        """
        if not self._client:
            await self.start()

        response = await self._client.embeddings.create(
            model=model,
            input=texts,
        )

        return [item.embedding for item in response.data]

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
            model: New model name.
        """
        self._model = self.MODELS.get(model, model)
        self.clear_history()

    def set_system_prompt(self, prompt: str) -> None:
        """Set or update the system prompt.

        Args:
            prompt: New system prompt.
        """
        self._system_prompt = prompt

        # Update in messages if present
        if self._messages and self._messages[0]["role"] == "system":
            self._messages[0]["content"] = prompt
        elif self._messages:
            self._messages.insert(0, {"role": "system", "content": prompt})

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Note: This is an approximation. For exact counts,
        use tiktoken directly.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Rough approximation: ~4 chars per token for English
        return len(text) // 4
