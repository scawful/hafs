"""Intelligent Model Orchestrator (Public).

Handles model fallback and quota management, with UnifiedOrchestrator v2
as the preferred backend.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
from typing import Optional

from core.genai_compat import (
    create_genai_client,
    embed_content as genai_embed_content,
    extract_embeddings,
    extract_text,
    extract_usage_tokens,
    generate_content as genai_generate_content,
    genai_available,
)
from core.orchestrator_v2 import Provider, TaskTier, UnifiedOrchestrator
from core.quota import quota_manager

_UNIFIED_SHARED: UnifiedOrchestrator | None = None
_UNIFIED_LOCK = asyncio.Lock()

# Configure logging
logger = logging.getLogger("orchestrator")

# Add venv to path if available (Common setup)
venv_path = os.path.expanduser("~/dotfiles/.venv/lib/python3.13/site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.append(venv_path)

GENAI_AVAILABLE = genai_available()
if not GENAI_AVAILABLE:
    logger.warning("GenAI SDK not found (google-genai or legacy).")

class ModelOrchestrator:
    """Manages model selection and fallback strategies."""

    # December 2025 - Model IDs from core.models.registry
    TIERS = {
        "reasoning": ["gemini-3-pro-preview", "gemini-3-flash-preview"],
        "fast": ["gemini-3-flash-preview"],
        "research": ["gemini-3-pro-preview", "gemini-3-flash-preview"],
        "coding": ["gemini-3-flash-preview", "gemini-3-pro-preview"],
        "creative": ["gemini-3-pro-preview", "gemini-3-flash-preview"],
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.client = None
        self._genai_client = None
        
        if self.api_key:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
            logger.info(f"ModelOrchestrator initialized with API Key: {masked_key}")
            if genai_available():
                self._genai_client = create_genai_client(self.api_key)
                if self._genai_client:
                    self.client = self._genai_client.client
                    logger.info("GenAI client initialized (%s).", self._genai_client.mode)
                else:
                    logger.warning("GenAI client initialization failed.")
        else:
            logger.warning("ModelOrchestrator initialized WITHOUT API Key.")
        
        self.gemini_cli_path = shutil.which("gemini")
        self._reload_env_overrides()

    def _reload_env_overrides(self) -> None:
        self._override_provider = self._parse_provider(os.environ.get("HAFS_MODEL_PROVIDER"))
        self._override_model = os.environ.get("HAFS_MODEL_MODEL") or None
        self._rotation = self._parse_rotation(os.environ.get("HAFS_MODEL_ROTATION"))

    @staticmethod
    def _parse_provider(value: Optional[str]) -> Optional[Provider]:
        if not value:
            return None
        try:
            return Provider(value.strip().lower())
        except ValueError:
            return None

    @staticmethod
    def _parse_rotation(value: Optional[str]) -> list[tuple[Provider, str]]:
        if not value:
            return []
        rotation: list[tuple[Provider, str]] = []
        for item in value.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            provider_raw, model = item.split(":", 1)
            provider = ModelOrchestrator._parse_provider(provider_raw.strip())
            if provider and model.strip():
                rotation.append((provider, model.strip()))
        return rotation

    async def _get_unified(self) -> UnifiedOrchestrator:
        global _UNIFIED_SHARED
        if _UNIFIED_SHARED is None:
            async with _UNIFIED_LOCK:
                if _UNIFIED_SHARED is None:
                    _UNIFIED_SHARED = UnifiedOrchestrator()
        await _UNIFIED_SHARED.initialize()
        return _UNIFIED_SHARED

    def _map_tier(self, tier: str) -> TaskTier:
        return {
            "reasoning": TaskTier.REASONING,
            "fast": TaskTier.FAST,
            "research": TaskTier.RESEARCH,
            "coding": TaskTier.CODING,
            "creative": TaskTier.CREATIVE,
        }.get(tier, TaskTier.FAST)

    async def embed_content(
        self,
        text: str,
        model: str = "text-embedding-004",
        provider: Optional[Provider | str] = None,
    ) -> list[float]:
        """Generate embeddings using GenAI SDK."""
        try:
            unified = await self._get_unified()
            embeddings = await unified.embed(text, provider=provider, model=model)
            if embeddings:
                return embeddings
        except Exception as e:
            logger.debug(f"UnifiedOrchestrator embed failed: {e}")

        if self._genai_client:
            try:
                response = await genai_embed_content(self._genai_client, model, text)
                embeddings = extract_embeddings(response)
                if embeddings:
                    return embeddings
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
        return []

    async def generate_content(
        self,
        prompt: str,
        tier: str = "fast",
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        rotation: Optional[list[str]] = None,
    ) -> str:
        """Attempt to generate content using models in the specified tier."""
        try:
            self._reload_env_overrides()
            unified = await self._get_unified()
            rotation_list = self._rotation
            if rotation:
                if all(isinstance(item, str) for item in rotation):
                    rotation_list = self._parse_rotation(",".join(rotation))
                else:
                    rotation_list = rotation  # type: ignore[assignment]
            if rotation_list:
                for rot_provider, rot_model in rotation_list:
                    result = await unified.generate(
                        prompt=prompt,
                        tier=self._map_tier(tier),
                        provider=rot_provider,
                        model=rot_model,
                    )
                    if result.content:
                        return result.content

            override_provider = provider or self._override_provider
            override_model = model or (self._override_model if override_provider else None)
            result = await unified.generate(
                prompt=prompt,
                tier=self._map_tier(tier),
                provider=override_provider,
                model=override_model,
            )
            if result.content:
                return result.content
        except Exception as e:
            logger.debug(f"UnifiedOrchestrator generate failed: {e}")

        models = self.TIERS.get(tier, self.TIERS["fast"])
        errors_list = []
        
        # 1. Try Direct API
        if self._genai_client:
            for model_name in models:
                est_tokens = len(prompt) // 4
                
                # ... quota check ...

                try:
                    logger.info(f"Attempting genai SDK call with model: {model_name}")
                    response = await genai_generate_content(
                        self._genai_client,
                        model_name,
                        prompt,
                    )

                    usage = extract_usage_tokens(response)
                    if usage:
                        logger.info("GenAI Success! Usage: %s tokens.", usage)

                    quota_manager.log_usage(model_name, usage if usage > 0 else est_tokens)
                    text = extract_text(response)
                    if text:
                        return text
                except Exception as e:
                    logger.warning(f"GenAI API({model_name}) failed: {e}")
                    errors_list.append(f"API({model_name}): {e}")
                    continue

        # 2. Fallback to CLI
        if self.gemini_cli_path:
            logger.info("Falling back to CLI execution...")

        raise Exception(f"All generation attempts failed. Errors: {errors_list}")
