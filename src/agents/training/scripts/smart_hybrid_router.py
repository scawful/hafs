#!/usr/bin/env python3
"""Smart Hybrid Router - Multi-Provider Intelligence with Quota Management.

Routes generation requests across ALL available providers:
1. medical-mechanica (Windows GPU - free, local)
2. Gemini API (smart, quota-limited)
3. OpenAI API (fallback, quota-limited)
4. Claude API (fallback, quota-limited)

Uses quota_manager to automatically rotate providers when limits are hit.

Strategy:
- Gigaleak (Japanese→English): Gemini (smart) → Claude → OpenAI
- Oracle (ROM hack analysis): Gemini → Claude → medical-mechanica
- Errors (diagnostics): Gemini Pro → Claude Opus → OpenAI
- ASM (vanilla): medical-mechanica → Gemini → OpenAI
- YAZE (C++ tools): medical-mechanica → Gemini → OpenAI
- Text: medical-mechanica → Gemini (cheapest)

Usage:
    python -m agents.training.scripts.smart_hybrid_router --pilot
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from core.orchestrator_v2 import Provider, TaskTier, UnifiedOrchestrator
from core.quota import quota_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


class SmartHybridRouter:
    """Intelligent multi-provider router with quota management.

    Uses model IDs from centralized registry (core.models.registry).
    """

    # Provider preferences per domain (in priority order)
    # Model IDs should match those in core.models.registry
    DOMAIN_PREFERENCES = {
        "gigaleak": [
            (Provider.GEMINI, "gemini-3-flash-preview"),  # Best for translation
            (Provider.ANTHROPIC, "claude-sonnet-4-20250514"),
            (Provider.OPENAI, "gpt-5.2-mini"),
        ],
        "oracle": [
            (Provider.GEMINI, "gemini-3-flash-preview"),
            (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),  # Best for complex Oracle
            (Provider.OPENAI, "gpt-5.2"),
            (Provider.OLLAMA, "qwen2.5-coder:14b"),  # medical-mechanica fallback
        ],
        "errors": [
            (Provider.GEMINI, "gemini-3-pro-preview"),  # Best for reasoning
            (Provider.ANTHROPIC, "claude-opus-4-5-20251101"),
            (Provider.OPENAI, "gpt-5.2"),
        ],
        "asm": [
            (Provider.OLLAMA, "qwen2.5-coder:14b"),  # medical-mechanica first
            (Provider.GEMINI, "gemini-3-flash-preview"),
            (Provider.OPENAI, "gpt-5.2-mini"),
        ],
        "yaze": [
            (Provider.OLLAMA, "qwen2.5-coder:14b"),  # medical-mechanica first
            (Provider.GEMINI, "gemini-3-flash-preview"),
            (Provider.OPENAI, "gpt-5.2-mini"),
        ],
        "text": [
            (Provider.OLLAMA, "qwen2.5-coder:14b"),  # medical-mechanica first
            (Provider.GEMINI, "gemini-3-flash-preview"),
        ],
    }

    def __init__(self, orchestrator: UnifiedOrchestrator):
        self.orchestrator = orchestrator
        self.quota = quota_manager
        self._provider_stats = {
            "gemini": {"used": 0, "failed": 0},
            "anthropic": {"used": 0, "failed": 0},
            "openai": {"used": 0, "failed": 0},
            "ollama": {"used": 0, "failed": 0},
        }

    async def route_for_domain(
        self, domain: str, prompt: str, timeout: float = 30.0
    ) -> Optional[str]:
        """Route generation request to best available provider for domain.

        Tries providers in priority order, skipping those over quota.
        Falls back through entire chain until success or exhaustion.

        Args:
            domain: Generator domain (asm, gigaleak, oracle, etc.)
            prompt: Teacher prompt
            timeout: Timeout in seconds

        Returns:
            Generated response or None if all providers failed
        """
        preferences = self.DOMAIN_PREFERENCES.get(
            domain, [(Provider.GEMINI, "gemini-3-flash-preview")]
        )

        for provider, model in preferences:
            # Check quota
            if provider == Provider.GEMINI:
                if not quota_manager.can_make_request("gemini"):
                    logger.warning(f"{domain}: Gemini quota exhausted, trying next provider")
                    continue

            elif provider == Provider.ANTHROPIC:
                if not quota_manager.can_make_request("anthropic"):
                    logger.warning(f"{domain}: Claude quota exhausted, trying next provider")
                    continue

            elif provider == Provider.OPENAI:
                if not quota_manager.can_make_request("openai"):
                    logger.warning(f"{domain}: OpenAI quota exhausted, trying next provider")
                    continue

            # Try generation
            try:
                logger.debug(f"{domain}: Trying {provider.value}/{model}")

                # Route through orchestrator
                response = await asyncio.wait_for(
                    self.orchestrator.generate(
                        prompt=prompt,
                        tier=TaskTier.CODING,  # Most domains use coding
                        provider=provider,
                        model=model,
                    ),
                    timeout=timeout,
                )

                # Success!
                self._provider_stats[provider.value]["used"] += 1
                logger.info(
                    f"{domain}: ✓ {provider.value}/{model} "
                    f"({self._provider_stats[provider.value]['used']} total)"
                )
                return response.content

            except asyncio.TimeoutError:
                logger.warning(f"{domain}: {provider.value} timeout, trying next")
                self._provider_stats[provider.value]["failed"] += 1
                continue

            except Exception as e:
                logger.warning(f"{domain}: {provider.value} failed: {e}, trying next")
                self._provider_stats[provider.value]["failed"] += 1
                continue

        # All providers failed
        logger.error(f"{domain}: All providers exhausted!")
        return None

    def get_stats(self) -> dict:
        """Get provider usage statistics."""
        return {
            "providers": self._provider_stats,
            "quota": {
                "gemini": quota_manager.get_status("gemini"),
                "anthropic": quota_manager.get_status("anthropic"),
                "openai": quota_manager.get_status("openai"),
            },
        }


async def configure_generator_with_router(generator, router: SmartHybridRouter):
    """Configure a generator to use smart hybrid routing.

    Monkey-patches the generator's generate_sample method to use
    the smart router instead of direct orchestrator calls.
    """
    original_generate = generator.generate_sample
    domain = generator.domain

    async def smart_generate(item):
        """Wrapped generate_sample with smart routing."""
        # Get teacher prompt
        prompt = generator.get_teacher_prompt(item)

        # Route through smart router
        response = await router.route_for_domain(domain, prompt)

        if not response:
            logger.error(f"Failed to generate for {item.name} - all providers exhausted")
            return None

        # Parse response (same as original)
        import json

        try:
            # Extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

            # Create sample (call original logic)
            from agents.training.base import TrainingSample

            return TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", item.content)),
                domain=domain,
                source=str(item.source),
                teacher_model=f"{router._last_provider}/{router._last_model}",
                teacher_prompt=str(prompt),
                kg_entities=[str(item.name)],
            )

        except Exception as e:
            logger.error(f"Failed to parse response for {item.name}: {e}")
            return None

    # Store last used provider/model for tracking
    router._last_provider = "unknown"
    router._last_model = "unknown"

    # Monkey-patch
    generator.generate_sample = smart_generate
    logger.info(f"✓ {domain} generator configured with smart hybrid routing")


async def main():
    """Test smart hybrid routing."""
    logger.info("=" * 80)
    logger.info("SMART HYBRID ROUTER TEST")
    logger.info("=" * 80)

    # Initialize
    orchestrator = UnifiedOrchestrator()
    router = SmartHybridRouter(orchestrator)

    # Test each domain
    test_domains = ["asm", "gigaleak", "oracle", "yaze", "text"]

    for domain in test_domains:
        logger.info(f"\nTesting {domain} routing...")
        response = await router.route_for_domain(
            domain, "Test prompt: Explain what this code does"
        )
        if response:
            logger.info(f"✓ {domain}: Success ({len(response)} chars)")
        else:
            logger.error(f"✗ {domain}: Failed")

    # Show stats
    stats = router.get_stats()
    logger.info("\n" + "=" * 80)
    logger.info("PROVIDER STATISTICS")
    logger.info("=" * 80)
    for provider, counts in stats["providers"].items():
        logger.info(
            f"{provider:12s}: {counts['used']:4d} used, {counts['failed']:4d} failed"
        )


if __name__ == "__main__":
    asyncio.run(main())
