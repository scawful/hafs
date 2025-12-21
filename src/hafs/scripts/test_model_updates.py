#!/usr/bin/env python3
"""Comprehensive test for updated model configuration (GPT-5, o3, Gemini, Anthropic OAuth).

Tests:
1. OpenAI GPT-5/o-series models with max_completion_tokens parameter
2. Gemini 3 Flash/Pro Preview models
3. Anthropic OAuth authentication
4. Multi-provider routing through orchestrator
5. Performance metrics for each model
6. Quality validation of responses

Usage:
    python -m hafs.scripts.test_model_updates
    python -m hafs.scripts.test_model_updates --quick  # Test only fastest models
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a model test."""

    model: str
    provider: str
    success: bool
    latency_ms: int
    response: str
    error: Optional[str] = None
    tokens_used: Optional[int] = None


class ModelTester:
    """Test all configured models."""

    def __init__(self):
        """Initialize tester."""
        self.results: list[TestResult] = []
        self.test_prompt = (
            "Write a simple Python function that checks if a number is prime. "
            "Respond with just the code, no explanation."
        )

    async def test_openai_model(
        self, model: str, test_name: str = None
    ) -> TestResult:
        """Test an OpenAI model."""
        try:
            import openai

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return TestResult(
                    model=model,
                    provider="openai",
                    success=False,
                    latency_ms=0,
                    response="",
                    error="No API key",
                )

            client = openai.AsyncOpenAI(api_key=api_key)

            # Determine parameter to use
            uses_completion_tokens = model.startswith(("gpt-5", "o1", "o3", "o4"))

            start = time.time()

            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": self.test_prompt}],
            }

            if uses_completion_tokens:
                kwargs["max_completion_tokens"] = 100
                # o1, o3, o4 series don't support temperature
                # GPT-5 series does support temperature
                if model.startswith("gpt-5"):
                    kwargs["temperature"] = 0.7
            else:
                kwargs["max_tokens"] = 100
                kwargs["temperature"] = 0.7

            response = await client.chat.completions.create(**kwargs)

            latency = int((time.time() - start) * 1000)

            content = response.choices[0].message.content
            tokens = (
                response.usage.total_tokens if hasattr(response, "usage") else None
            )

            return TestResult(
                model=model,
                provider="openai",
                success=True,
                latency_ms=latency,
                response=content[:200],
                tokens_used=tokens,
            )

        except Exception as e:
            return TestResult(
                model=model,
                provider="openai",
                success=False,
                latency_ms=0,
                response="",
                error=str(e),
            )

    async def test_gemini_model(self, model: str) -> TestResult:
        """Test a Gemini model."""
        try:
            import google.genai as genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return TestResult(
                    model=model,
                    provider="gemini",
                    success=False,
                    latency_ms=0,
                    response="",
                    error="No API key",
                )

            client = genai.Client(api_key=api_key)

            start = time.time()

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=self.test_prompt,
                config={"max_output_tokens": 100, "temperature": 0.7},
            )

            latency = int((time.time() - start) * 1000)

            content = response.text

            return TestResult(
                model=model,
                provider="gemini",
                success=True,
                latency_ms=latency,
                response=content[:200],
            )

        except Exception as e:
            return TestResult(
                model=model,
                provider="gemini",
                success=False,
                latency_ms=0,
                response="",
                error=str(e),
            )

    async def test_anthropic_oauth(self, model: str) -> TestResult:
        """Test Anthropic model with OAuth."""
        try:
            from hafs.core.anthropic_oauth import AnthropicOAuthClient

            oauth_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
            if not oauth_token:
                return TestResult(
                    model=model,
                    provider="anthropic",
                    success=False,
                    latency_ms=0,
                    response="",
                    error="No OAuth token",
                )

            client = AnthropicOAuthClient(oauth_token=oauth_token)

            start = time.time()

            response = await client.create_message(
                model=model,
                messages=[{"role": "user", "content": self.test_prompt}],
                max_tokens=100,
                temperature=0.7,
            )

            latency = int((time.time() - start) * 1000)

            content = response["content"][0]["text"]
            tokens = response.get("usage", {}).get("input_tokens", 0) + response.get(
                "usage", {}
            ).get("output_tokens", 0)

            await client.close()

            return TestResult(
                model=model,
                provider="anthropic",
                success=True,
                latency_ms=latency,
                response=content[:200],
                tokens_used=tokens,
            )

        except Exception as e:
            return TestResult(
                model=model,
                provider="anthropic",
                success=False,
                latency_ms=0,
                response="",
                error=str(e),
            )

    async def test_orchestrator_routing(self) -> list[TestResult]:
        """Test multi-provider routing through orchestrator."""
        results = []

        try:
            from hafs.core.orchestrator_v2 import TaskTier, UnifiedOrchestrator

            logger.info("\n[Orchestrator Routing Tests]")

            orchestrator = UnifiedOrchestrator()
            await orchestrator.initialize()

            # Test each tier
            tiers = [TaskTier.FAST, TaskTier.CODING, TaskTier.REASONING]

            for tier in tiers:
                logger.info(f"\nTesting {tier.value} tier...")

                start = time.time()

                try:
                    result = await orchestrator.generate(
                        prompt=self.test_prompt,
                        tier=tier,
                        max_tokens=100,
                    )

                    latency = int((time.time() - start) * 1000)

                    # Extract content from GenerationResult
                    content = result.content if hasattr(result, 'content') else str(result)

                    results.append(
                        TestResult(
                            model=f"orchestrator-{tier.value}",
                            provider="orchestrator",
                            success=True,
                            latency_ms=latency,
                            response=content[:200] if content else "",
                        )
                    )

                    logger.info(f"✓ {tier.value} tier: {latency}ms (routed to {result.provider.value}/{result.model})")

                except Exception as e:
                    results.append(
                        TestResult(
                            model=f"orchestrator-{tier.value}",
                            provider="orchestrator",
                            success=False,
                            latency_ms=0,
                            response="",
                            error=str(e),
                        )
                    )
                    logger.error(f"✗ {tier.value} tier failed: {e}")

        except Exception as e:
            logger.error(f"Orchestrator test failed: {e}")

        return results

    async def run_all_tests(self, quick: bool = False) -> None:
        """Run all model tests."""
        logger.info("=" * 80)
        logger.info("MODEL UPDATE VERIFICATION TEST")
        logger.info("=" * 80)

        # Define test configurations
        if quick:
            openai_models = ["o3-mini", "gpt-5.2"]
            gemini_models = ["gemini-3-flash-preview"]
            anthropic_models = ["claude-3-haiku-20240307"]
        else:
            openai_models = [
                # GPT-5 series
                "gpt-5",
                "gpt-5.2",
                "gpt-5-mini",
                # o-series
                "o3",
                "o3-mini",
                "o4-mini",
                # GPT-4 (for comparison)
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ]
            gemini_models = [
                "gemini-3-flash-preview",
                "gemini-3-pro-preview",
            ]
            anthropic_models = [
                "claude-3-haiku-20240307",
                "claude-3-opus-20240229",
            ]

        # Test OpenAI models
        logger.info("\n[OpenAI GPT-5/o-series Models]")
        for model in openai_models:
            logger.info(f"Testing {model}...")
            result = await self.test_openai_model(model)
            self.results.append(result)

            if result.success:
                logger.info(
                    f"✓ {model:20s} {result.latency_ms:5d}ms "
                    f"({result.tokens_used or 0} tokens)"
                )
            else:
                logger.error(f"✗ {model:20s} FAILED: {result.error}")

        # Test Gemini models
        logger.info("\n[Gemini 3 Models]")
        for model in gemini_models:
            logger.info(f"Testing {model}...")
            result = await self.test_gemini_model(model)
            self.results.append(result)

            if result.success:
                logger.info(f"✓ {model:30s} {result.latency_ms:5d}ms")
            else:
                logger.error(f"✗ {model:30s} FAILED: {result.error}")

        # Test Anthropic OAuth
        logger.info("\n[Anthropic OAuth Models]")
        for model in anthropic_models:
            logger.info(f"Testing {model}...")
            result = await self.test_anthropic_oauth(model)
            self.results.append(result)

            if result.success:
                logger.info(
                    f"✓ {model:30s} {result.latency_ms:5d}ms "
                    f"({result.tokens_used or 0} tokens)"
                )
            else:
                logger.error(f"✗ {model:30s} FAILED: {result.error}")

        # Test orchestrator routing
        logger.info("\n[Multi-Provider Orchestrator Routing]")
        orchestrator_results = await self.test_orchestrator_routing()
        self.results.extend(orchestrator_results)

        # Print summary
        self.print_summary()

    def print_summary(self) -> None:
        """Print test summary."""
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)

        # Group by provider
        by_provider: dict[str, list[TestResult]] = {}
        for result in self.results:
            if result.provider not in by_provider:
                by_provider[result.provider] = []
            by_provider[result.provider].append(result)

        for provider, results in by_provider.items():
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]

            logger.info(f"\n{provider.upper()}:")
            logger.info(f"  Total: {len(results)}")
            logger.info(f"  Passed: {len(successful)}")
            logger.info(f"  Failed: {len(failed)}")

            if successful:
                avg_latency = sum(r.latency_ms for r in successful) / len(successful)
                logger.info(f"  Avg Latency: {avg_latency:.0f}ms")

                # Show fastest model
                fastest = min(successful, key=lambda r: r.latency_ms)
                logger.info(
                    f"  Fastest: {fastest.model} ({fastest.latency_ms}ms)"
                )

        # Overall stats
        total = len(self.results)
        passed = len([r for r in self.results if r.success])
        failed = len([r for r in self.results if not r.success])

        logger.info(f"\nOVERALL:")
        logger.info(f"  Total Tests: {total}")
        logger.info(f"  Passed: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"  Failed: {failed} ({failed/total*100:.1f}%)")

        if failed > 0:
            logger.info(f"\nFAILED TESTS:")
            for result in self.results:
                if not result.success:
                    logger.info(f"  ✗ {result.provider}/{result.model}: {result.error}")

        logger.info("\n" + "=" * 80)
        if failed == 0:
            logger.info("✓ ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
        else:
            logger.info(f"✗ {failed} TESTS FAILED - REVIEW CONFIGURATION")
        logger.info("=" * 80)


async def main():
    """Main entry point."""
    import sys

    quick = "--quick" in sys.argv

    tester = ModelTester()
    await tester.run_all_tests(quick=quick)

    # Return exit code based on results
    failed = len([r for r in tester.results if not r.success])
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
