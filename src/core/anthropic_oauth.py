"""Anthropic Claude Max OAuth Client.

Implements OAuth authentication for Anthropic Claude Max plans,
similar to Claude Code's OAuth flow.

This allows using Claude Max plan features with OAuth tokens instead of API keys.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OAuthConfig:
    """OAuth configuration for Anthropic."""

    client_id: str = "claude-code"  # Default client ID
    oauth_token: Optional[str] = None
    token_env: str = "CLAUDE_CODE_OAUTH_TOKEN"


class AnthropicOAuthClient:
    """OAuth client for Anthropic Claude Max plans.

    Supports OAuth token-based authentication for Claude Max users,
    providing access to extended features and higher rate limits.

    Usage:
        client = AnthropicOAuthClient(oauth_token="sk-ant-oat01-...")
        response = await client.create_message(
            model="claude-opus-4-5-20251101",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    def __init__(
        self,
        oauth_token: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
    ):
        """Initialize OAuth client.

        Args:
            oauth_token: OAuth token (or load from env)
            base_url: Anthropic API base URL
        """
        self.oauth_token = oauth_token or os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
        self.base_url = base_url
        self.api_version = "2023-06-01"

        if not self.oauth_token:
            raise ValueError(
                "OAuth token required. Set CLAUDE_CODE_OAUTH_TOKEN environment variable."
            )

        # Validate token format (OAuth tokens start with sk-ant-oat)
        if not self.oauth_token.startswith("sk-ant-oat"):
            logger.warning(
                "OAuth token doesn't start with 'sk-ant-oat'. "
                "This might be a regular API key instead of an OAuth token."
            )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(60.0),
            headers=self._get_headers(),
        )

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for OAuth requests.

        OAuth tokens use Bearer authentication (NOT x-api-key).
        Based on OpenCode implementation (provider.ts lines 42-71).

        Critical differences from API key:
        - Authorization: Bearer {token} (not x-api-key)
        - anthropic-beta: oauth-2025-04-20 header required
        - NO x-api-key header (explicitly removed)
        """
        return {
            "authorization": f"Bearer {self.oauth_token}",
            "anthropic-version": self.api_version,
            "anthropic-beta": "oauth-2025-04-20",  # Required for OAuth
            "content-type": "application/json",
        }

    async def create_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a message using Claude API with OAuth token.

        Args:
            model: Model ID (e.g., "claude-opus-4-5-20251101")
            messages: List of message dicts with "role" and "content"
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-1)
            system: System prompt
            stream: Whether to stream response
            **kwargs: Additional API parameters

        Returns:
            API response dict

        Raises:
            httpx.HTTPStatusError: If API returns error status
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        # For OAuth tokens, include system prompt if not provided
        # This helps with authentication validation
        if system:
            payload["system"] = system
        elif "system" not in kwargs:
            # Add minimal system context for OAuth validation
            payload["system"] = "You are a helpful AI assistant."

        if stream:
            payload["stream"] = True

        logger.debug(f"Creating message with model: {model}")

        response = await self._client.post("/v1/messages", json=payload)

        # Handle errors
        if response.status_code != 200:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "Unknown error")

            if response.status_code == 401:
                raise ValueError(
                    f"Authentication failed: {error_msg}\n"
                    f"OAuth token might be invalid or expired.\n"
                    f"Token prefix: {self.oauth_token[:15]}..."
                )
            elif response.status_code == 429:
                raise ValueError(f"Rate limit exceeded: {error_msg}")
            else:
                raise ValueError(f"API error {response.status_code}: {error_msg}")

        return response.json()

    async def stream_message(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system: Optional[str] = None,
        **kwargs: Any,
    ):
        """Stream a message using Claude API.

        Args:
            model: Model ID
            messages: List of message dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            **kwargs: Additional API parameters

        Yields:
            Server-sent events from the API
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }

        if system:
            payload["system"] = system

        async with self._client.stream("POST", "/v1/messages", json=payload) as response:
            if response.status_code != 200:
                error_data = await response.aread()
                raise ValueError(f"Stream error {response.status_code}: {error_data}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break
                    yield data

    async def list_models(self) -> list[str]:
        """List available models.

        Note: Anthropic doesn't provide a models endpoint,
        so this returns known Claude models.

        Returns:
            List of available model IDs
        """
        # Known Claude models (as of Dec 2025)
        return [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251015",
            "claude-opus-4-1-20250814",
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250224",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]

    async def validate_token(self) -> bool:
        """Validate OAuth token by making a test request.

        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Make minimal request to test auth
            # Try common model IDs
            test_models = [
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20241022",
                "claude-sonnet-4-20250514",
            ]

            for model in test_models:
                try:
                    response = await self.create_message(
                        model=model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=10,
                    )
                    logger.info(f"✓ OAuth token validated successfully with {model}")
                    return True
                except ValueError as e:
                    if "404" in str(e):
                        continue  # Try next model
                    raise  # Other errors should be raised

            logger.error("✗ No valid model found")
            return False

        except ValueError as e:
            logger.error(f"✗ OAuth token validation failed: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


async def test_oauth_client():
    """Test OAuth client with Claude Max."""
    logger.info("=" * 80)
    logger.info("ANTHROPIC CLAUDE MAX OAUTH CLIENT TEST")
    logger.info("=" * 80)

    oauth_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
    if not oauth_token:
        logger.error("✗ CLAUDE_CODE_OAUTH_TOKEN not set")
        logger.error("  Set your OAuth token from Claude Max plan:")
        logger.error('  export CLAUDE_CODE_OAUTH_TOKEN="sk-ant-oat01-..."')
        return False

    logger.info(f"OAuth token: {oauth_token[:15]}...")

    async with AnthropicOAuthClient(oauth_token=oauth_token) as client:
        # Validate token
        logger.info("\n[1/3] Validating OAuth token...")
        is_valid = await client.validate_token()

        if not is_valid:
            return False

        # Test message generation
        logger.info("\n[2/3] Testing message generation...")
        try:
            response = await client.create_message(
                model="claude-3-haiku-20240307",  # Working with OAuth
                messages=[
                    {"role": "user", "content": "Say 'OAuth working' in Python code"}
                ],
                max_tokens=100,
            )

            content = response["content"][0]["text"]
            logger.info(f"✓ Haiku response: {content[:200]}...")

        except Exception as e:
            logger.error(f"✗ Message generation test failed: {e}")
            return False

        # List available models
        logger.info("\n[3/3] Listing available models...")
        models = await client.list_models()
        logger.info(f"Available models: {len(models)}")
        for model in models[:5]:
            logger.info(f"  - {model}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ OAuth client working successfully!")
    logger.info("=" * 80)
    return True


async def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)
    success = await test_oauth_client()
    return 0 if success else 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
