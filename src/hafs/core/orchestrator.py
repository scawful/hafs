"""Intelligent Model Orchestrator (Public).

Handles model fallback and quota management, with UnifiedOrchestrator v2
as the preferred backend.
"""

import asyncio
import importlib
import json
import logging
import os
import shutil
import sys
from typing import Optional
from hafs.core.quota import quota_manager
from hafs.core.orchestrator_v2 import UnifiedOrchestrator, TaskTier

_UNIFIED_SHARED: UnifiedOrchestrator | None = None
_UNIFIED_LOCK = asyncio.Lock()

# Configure logging
logger = logging.getLogger("orchestrator")

# Dynamic import
genai = None
GENAI_AVAILABLE = False

# Add venv to path if available (Common setup)
venv_path = os.path.expanduser("~/dotfiles/.venv/lib/python3.13/site-packages")
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.append(venv_path)

try:
    genai = importlib.import_module("google.genai")
    GENAI_AVAILABLE = True
except ImportError:
    logger.warning("google-genai SDK not found.")

class ModelOrchestrator:
    """Manages model selection and fallback strategies."""

    TIERS = {
        "reasoning": ["gemini-3-pro-preview", "gemini-2.5-pro"],
        "fast": ["gemini-3-flash-preview", "gemini-2.5-flash"],
        "research": ["gemini-3-pro-preview", "gemini-2.5-pro"],
        "coding": ["gemini-3-pro-preview", "gemini-2.5-pro"],
        "creative": ["gemini-3-pro-preview", "gemini-2.5-pro"]
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("AISTUDIO_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.client = None
        
        if self.api_key:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
            logger.info(f"ModelOrchestrator initialized with API Key: {masked_key}")
            if GENAI_AVAILABLE and genai is not None:
                try:
                    # v1.0+ Client style
                    ClientCls = getattr(genai, "Client", None)
                    if ClientCls:
                        self.client = ClientCls(api_key=self.api_key)
                        logger.info("google-genai Client initialized.")
                    else:
                        logger.warning("google.genai.Client not found (version mismatch?)")
                except Exception as e:
                    logger.error(f"Failed to init google-genai Client: {e}")
        else:
            logger.warning("ModelOrchestrator initialized WITHOUT API Key.")
        
        self.gemini_cli_path = shutil.which("gemini")

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

    async def embed_content(self, text: str, model: str = "text-embedding-004") -> list[float]:
        """Generate embeddings using GenAI SDK."""
        try:
            unified = await self._get_unified()
            embeddings = await unified.embed(text)
            if embeddings:
                return embeddings
        except Exception as e:
            logger.debug(f"UnifiedOrchestrator embed failed: {e}")

        if self.client and GENAI_AVAILABLE:
            try:
                response = await self.client.aio.models.embed_content(
                    model=model,
                    contents=text
                )
                return response.embeddings[0].values
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
        return []

    async def generate_content(self, prompt: str, tier: str = "fast") -> str:
        """Attempt to generate content using models in the specified tier."""
        try:
            unified = await self._get_unified()
            result = await unified.generate(prompt=prompt, tier=self._map_tier(tier))
            if result.content:
                return result.content
        except Exception as e:
            logger.debug(f"UnifiedOrchestrator generate failed: {e}")

        models = self.TIERS.get(tier, self.TIERS["fast"])
        errors_list = []
        
        # 1. Try Direct API
        if self.client and GENAI_AVAILABLE:
            for model_name in models:
                est_tokens = len(prompt) // 4
                
                # ... quota check ...

                try:
                    logger.info(f"Attempting genai SDK call with model: {model_name}")
                    response = await self.client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    
                    usage = 0
                    if response.usage_metadata:
                        usage = response.usage_metadata.total_token_count
                        logger.info(f"GenAI Success! Usage: {usage} tokens.")
                    
                    quota_manager.log_usage(model_name, usage if usage > 0 else est_tokens)
                    return response.text
                except Exception as e:
                    logger.warning(f"GenAI API({model_name}) failed: {e}")
                    errors_list.append(f"API({model_name}): {e}")
                    continue

        # 2. Fallback to CLI
        if self.gemini_cli_path:
            logger.info("Falling back to CLI execution...")

        raise Exception(f"All generation attempts failed. Errors: {errors_list}")
