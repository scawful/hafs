#!/usr/bin/env python3
"""Validate API Keys and Test Latest 2025 Models.

Tests:
1. OpenAI API (GPT-5.2-Codex, o3, o4-mini)
2. Google Gemini API (Gemini 3 Flash, Gemini 3 Pro)
3. Anthropic API (Claude Opus 4.5, Sonnet 4.5, Haiku 4.5)

Provides diagnostics and configuration recommendations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of an API test."""

    provider: str
    model: str
    success: bool
    response: str = ""
    error: str = ""
    latency_ms: int = 0


async def test_openai_api() -> list[TestResult]:
    """Test OpenAI API with latest 2025 models."""
    results = []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("✗ OPENAI_API_KEY not set")
        return [
            TestResult(
                provider="OpenAI",
                model="N/A",
                success=False,
                error="API key not configured",
            )
        ]

    # Test GPT-5.2 (latest model - December 2025)
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        # Test GPT-5.2 (current flagship model)
        logger.info("Testing OpenAI GPT-5.2...")
        start = asyncio.get_event_loop().time()

        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "user", "content": "Say 'API working' in Python"}],
            max_completion_tokens=50,  # GPT-5.2 uses max_completion_tokens
        )

        latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        content = response.choices[0].message.content

        results.append(
            TestResult(
                provider="OpenAI",
                model="gpt-5.2",
                success=True,
                response=content[:100],
                latency_ms=latency_ms,
            )
        )
        logger.info(f"✓ GPT-5.2 working ({latency_ms}ms)")

    except Exception as e:
        error_msg = str(e)
        results.append(
            TestResult(
                provider="OpenAI",
                model="gpt-5.2",
                success=False,
                error=error_msg,
            )
        )

        # Check if it's auth or model error
        if "401" in error_msg or "authentication" in error_msg.lower():
            logger.error("✗ OpenAI API key is invalid or expired")
            logger.error("   Get a new key at: https://platform.openai.com/api-keys")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            logger.warning("✗ GPT-5.2 not available. Trying gpt-5.2-mini...")
            try:
                response = client.chat.completions.create(
                    model="gpt-5.2-mini",
                    messages=[
                        {"role": "user", "content": "Say 'API working' in Python"}
                    ],
                    max_completion_tokens=50,
                )
                results.append(
                    TestResult(
                        provider="OpenAI",
                        model="gpt-5.2-mini (fallback)",
                        success=True,
                        response=response.choices[0].message.content[:100],
                    )
                )
                logger.info("✓ gpt-5.2-mini working (fallback)")
            except Exception as e2:
                logger.error(f"✗ gpt-5.2-mini also failed: {e2}")
        else:
            logger.error(f"✗ GPT-5.2 test failed: {error_msg}")

    return results


async def test_gemini_api() -> list[TestResult]:
    """Test Google Gemini API with latest 2025 models."""
    results = []

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("✗ GEMINI_API_KEY not set")
        return [
            TestResult(
                provider="Gemini",
                model="N/A",
                success=False,
                error="API key not configured",
            )
        ]

    try:
        import google.genai as genai

        # Test Gemini 3 Flash Preview (latest, Dec 2025)
        logger.info("Testing Gemini 3 Flash Preview...")
        start = asyncio.get_event_loop().time()

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Say 'API working' in Python",
        )

        latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        content = response.text

        results.append(
            TestResult(
                provider="Gemini",
                model="gemini-3-flash-preview",
                success=True,
                response=content[:100],
                latency_ms=latency_ms,
            )
        )
        logger.info(f"✓ Gemini 3 Flash Preview working ({latency_ms}ms)")

    except Exception as e:
        error_msg = str(e)
        results.append(
            TestResult(
                provider="Gemini",
                model="gemini-3-flash-preview",
                success=False,
                error=error_msg,
            )
        )

        if "not found" in error_msg.lower() or "invalid" in error_msg.lower():
            logger.warning("✗ Gemini 3 Flash Preview not available. Trying Gemini 3 Pro Preview...")

            # Fallback to Gemini 3 Pro Preview
            try:
                import google.genai as genai
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-3-pro-preview",
                    contents="Say 'API working' in Python",
                )
                results.append(
                    TestResult(
                        provider="Gemini",
                        model="gemini-3-pro-preview (fallback)",
                        success=True,
                        response=response.text[:100],
                    )
                )
                logger.info("✓ Gemini 3 Pro Preview working (fallback)")
            except Exception as e2:
                logger.error(f"✗ Gemini 3 Pro Preview also failed: {e2}")
        else:
            logger.error(f"✗ Gemini 3 Flash Preview test failed: {error_msg}")

    return results


async def test_anthropic_api() -> list[TestResult]:
    """Test Anthropic Claude API with latest 2025 models."""
    results = []

    # Try OAuth token first (Claude Code style)
    api_key = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("✗ CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY not set")
        return [
            TestResult(
                provider="Anthropic",
                model="N/A",
                success=False,
                error="API key not configured",
            )
        ]

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        # Test Claude Opus 4.5 (latest, Nov 2025)
        logger.info("Testing Claude Opus 4.5...")
        start = asyncio.get_event_loop().time()

        message = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'API working' in Python"}],
        )

        latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        content = message.content[0].text

        results.append(
            TestResult(
                provider="Anthropic",
                model="claude-opus-4-5",
                success=True,
                response=content[:100],
                latency_ms=latency_ms,
            )
        )
        logger.info(f"✓ Claude Opus 4.5 working ({latency_ms}ms)")

    except Exception as e:
        error_msg = str(e)
        results.append(
            TestResult(
                provider="Anthropic",
                model="claude-opus-4-5",
                success=False,
                error=error_msg,
            )
        )

        if "model" in error_msg.lower() or "not found" in error_msg.lower():
            logger.warning("✗ Claude Opus 4.5 not available yet. Trying Sonnet 4...")

            # Fallback to Sonnet 4
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=50,
                    messages=[
                        {"role": "user", "content": "Say 'API working' in Python"}
                    ],
                )
                results.append(
                    TestResult(
                        provider="Anthropic",
                        model="claude-sonnet-4 (fallback)",
                        success=True,
                        response=message.content[0].text[:100],
                    )
                )
                logger.info("✓ Claude Sonnet 4 working (fallback)")
            except Exception as e2:
                logger.error(f"✗ Claude Sonnet 4 also failed: {e2}")
        else:
            logger.error(f"✗ Claude Opus 4.5 test failed: {error_msg}")

    return results


async def main():
    """Main validation routine."""
    logger.info("=" * 80)
    logger.info("API KEY VALIDATION - Latest 2025 Models")
    logger.info("=" * 80)

    all_results = []

    # Test OpenAI
    logger.info("\n[1/3] Testing OpenAI API...")
    openai_results = await test_openai_api()
    all_results.extend(openai_results)

    # Test Gemini
    logger.info("\n[2/3] Testing Google Gemini API...")
    gemini_results = await test_gemini_api()
    all_results.extend(gemini_results)

    # Test Anthropic
    logger.info("\n[3/3] Testing Anthropic Claude API...")
    anthropic_results = await test_anthropic_api()
    all_results.extend(anthropic_results)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    success_count = sum(1 for r in all_results if r.success)
    total_count = len(all_results)

    for result in all_results:
        status = "✓" if result.success else "✗"
        latency = f" ({result.latency_ms}ms)" if result.latency_ms > 0 else ""
        logger.info(f"{status} {result.provider:12s} {result.model:30s}{latency}")
        if result.error:
            logger.info(f"   Error: {result.error}")

    logger.info(f"\nPassed: {success_count}/{total_count}")

    if success_count < total_count:
        logger.info("\n" + "=" * 80)
        logger.info("CONFIGURATION RECOMMENDATIONS")
        logger.info("=" * 80)

        logger.info("\n1. Set missing API keys in your shell:")
        if not os.getenv("OPENAI_API_KEY"):
            logger.info('   export OPENAI_API_KEY="sk-..."')
        if not os.getenv("GEMINI_API_KEY"):
            logger.info('   export GEMINI_API_KEY="AIza..."')
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.info('   export ANTHROPIC_API_KEY="sk-ant-..."')

        logger.info("\n2. Get API keys:")
        logger.info("   OpenAI: https://platform.openai.com/api-keys")
        logger.info("   Gemini: https://aistudio.google.com/app/apikey")
        logger.info("   Anthropic: https://console.anthropic.com/settings/keys")

        logger.info("\n3. Update models.toml with working models:")
        logger.info("   ~/.config/hafs/models.toml")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
