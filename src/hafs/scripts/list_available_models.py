#!/usr/bin/env python3
"""List all available models from each AI provider.

Queries:
- OpenAI API for GPT models
- Google Gemini API for Gemini models
- Anthropic API for Claude models
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


async def list_openai_models():
    """List all available OpenAI models."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("✗ OPENAI_API_KEY not set - skipping OpenAI")
        return

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        logger.info("\n" + "="*80)
        logger.info("OPENAI MODELS")
        logger.info("="*80)

        models = client.models.list()

        # Group by category
        gpt_models = []
        o_models = []
        other = []

        for model in models.data:
            model_id = model.id
            if model_id.startswith("gpt"):
                gpt_models.append(model_id)
            elif model_id.startswith("o"):
                o_models.append(model_id)
            else:
                other.append(model_id)

        logger.info("\nGPT Models:")
        for m in sorted(gpt_models):
            logger.info(f"  {m}")

        logger.info("\no-Series (Reasoning) Models:")
        for m in sorted(o_models):
            logger.info(f"  {m}")

        if other:
            logger.info("\nOther Models:")
            for m in sorted(other)[:20]:  # Limit to 20
                logger.info(f"  {m}")

        logger.info(f"\nTotal: {len(models.data)} models")

    except Exception as e:
        logger.error(f"✗ Failed to list OpenAI models: {e}")


async def list_gemini_models():
    """List all available Gemini models."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("✗ GEMINI_API_KEY not set - skipping Gemini")
        return

    try:
        import google.genai as genai

        client = genai.Client(api_key=api_key)

        logger.info("\n" + "="*80)
        logger.info("GOOGLE GEMINI MODELS")
        logger.info("="*80)

        models = client.models.list()

        # Group by generation
        gemini_3 = []
        gemini_2_5 = []
        gemini_2 = []
        other = []

        for model in models:
            model_name = model.name.split("/")[-1] if "/" in model.name else model.name

            if "gemini-3" in model_name:
                gemini_3.append(model_name)
            elif "gemini-2.5" in model_name:
                gemini_2_5.append(model_name)
            elif "gemini-2" in model_name:
                gemini_2.append(model_name)
            else:
                other.append(model_name)

        if gemini_3:
            logger.info("\nGemini 3 Series (Latest):")
            for m in sorted(gemini_3):
                logger.info(f"  {m}")

        if gemini_2_5:
            logger.info("\nGemini 2.5 Series:")
            for m in sorted(gemini_2_5):
                logger.info(f"  {m}")

        if gemini_2:
            logger.info("\nGemini 2 Series:")
            for m in sorted(gemini_2):
                logger.info(f"  {m}")

        if other:
            logger.info("\nOther Models:")
            for m in sorted(other)[:10]:
                logger.info(f"  {m}")

        total = len(gemini_3) + len(gemini_2_5) + len(gemini_2) + len(other)
        logger.info(f"\nTotal: {total} models")

    except Exception as e:
        logger.error(f"✗ Failed to list Gemini models: {e}")


async def list_anthropic_models():
    """List all available Claude models."""
    api_key = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("✗ CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY not set - skipping Anthropic")
        return

    try:
        import anthropic

        # Anthropic doesn't have a list models API endpoint
        # So we'll list the known models from documentation

        logger.info("\n" + "="*80)
        logger.info("ANTHROPIC CLAUDE MODELS (From Documentation)")
        logger.info("="*80)

        # Known models as of Dec 2025
        claude_4_5 = [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251015",
        ]

        claude_4 = [
            "claude-opus-4-1-20250814",
            "claude-sonnet-4-20250514",
        ]

        claude_3 = [
            "claude-3-7-sonnet-20250224",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        logger.info("\nClaude 4.5 Series (Latest):")
        for m in claude_4_5:
            logger.info(f"  {m}")

        logger.info("\nClaude 4 Series:")
        for m in claude_4:
            logger.info(f"  {m}")

        logger.info("\nClaude 3 Series:")
        for m in claude_3:
            logger.info(f"  {m}")

        logger.info("\nNote: Test with actual API to verify access")

    except Exception as e:
        logger.error(f"✗ Failed to check Anthropic: {e}")


async def main():
    """List all models from all providers."""
    logger.info("="*80)
    logger.info("AI MODEL DISCOVERY - List All Available Models")
    logger.info("="*80)

    # List models from each provider
    await list_openai_models()
    await list_gemini_models()
    await list_anthropic_models()

    logger.info("\n" + "="*80)
    logger.info("DONE")
    logger.info("="*80)
    logger.info("\nUse these model names in ~/.config/hafs/models.toml")


if __name__ == "__main__":
    asyncio.run(main())
