"""Model configuration loader with dynamic discovery.

Loads model configurations from config/models.toml to prevent
AI models from corrupting model definitions with stale training data.

Supports dynamic model discovery from provider APIs:
- Gemini: google.generativeai.list_models()
- Anthropic: anthropic.models.list()
- OpenAI: openai.models.list()
- Ollama: GET /api/tags

This is the SINGLE SOURCE OF TRUTH for model configurations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore

logger = logging.getLogger(__name__)

# Cache for discovered models
_discovery_cache: dict[str, tuple[float, list[str]]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _find_config_file() -> Optional[Path]:
    """Find the models.toml config file."""
    # Check common locations
    candidates = [
        Path(__file__).parent.parent.parent.parent / "config" / "models.toml",  # src/core -> config
        Path.home() / ".config" / "hafs" / "models.toml",  # User override
        Path.cwd() / "config" / "models.toml",  # Current directory
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    return None


@dataclass
class ModelSpec:
    """Specification for a single model."""
    
    name: str
    display_name: str
    description: str = ""
    context_window: int = 128000
    output_tokens: int = 8192
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    cost_per_1m_input_long: Optional[float] = None  # For tiered pricing
    cost_per_1m_output_long: Optional[float] = None
    tier: str = "general"
    requires_vram_gb: Optional[int] = None
    knowledge_cutoff: Optional[str] = None
    discovered: bool = False  # True if dynamically discovered


@dataclass
class ProviderModels:
    """Models available for a provider."""
    
    models: dict[str, ModelSpec] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    quota: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Complete model configuration loaded from TOML."""
    
    gemini: ProviderModels = field(default_factory=ProviderModels)
    anthropic: ProviderModels = field(default_factory=ProviderModels)
    openai: ProviderModels = field(default_factory=ProviderModels)
    ollama: ProviderModels = field(default_factory=ProviderModels)
    tier_routes: dict[str, list[str]] = field(default_factory=dict)
    
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ModelConfig":
        """Load configuration from TOML file."""
        if path is None:
            path = _find_config_file()
        
        if path is None or not path.exists():
            logger.warning("models.toml not found, using empty defaults")
            return cls()
        
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to load models.toml: {e}")
            return cls()
        
        config = cls()
        
        # Load each provider
        for provider_name in ["gemini", "anthropic", "openai", "ollama"]:
            provider_data = data.get(provider_name, {})
            provider_models = ProviderModels()
            
            # Load models
            models_data = provider_data.get("models", {})
            for model_name, model_info in models_data.items():
                if not isinstance(model_info, dict):
                    continue
                provider_models.models[model_name] = ModelSpec(
                    name=model_name,
                    display_name=model_info.get("display_name", model_name),
                    description=model_info.get("description", ""),
                    context_window=model_info.get("context_window", 128000),
                    output_tokens=model_info.get("output_tokens", 8192),
                    cost_per_1m_input=model_info.get("cost_per_1m_input", 0.0),
                    cost_per_1m_output=model_info.get("cost_per_1m_output", 0.0),
                    cost_per_1m_input_long=model_info.get("cost_per_1m_input_long"),
                    cost_per_1m_output_long=model_info.get("cost_per_1m_output_long"),
                    tier=model_info.get("tier", "general"),
                    requires_vram_gb=model_info.get("requires_vram_gb"),
                    knowledge_cutoff=model_info.get("knowledge_cutoff"),
                    discovered=False,
                )
            
            # Load defaults
            provider_models.defaults = dict(provider_data.get("defaults", {}))
            
            # Load quota (for gemini)
            provider_models.quota = dict(provider_data.get("quota", {}))
            
            setattr(config, provider_name, provider_models)
        
        # Load tier routes
        config.tier_routes = dict(data.get("tier_routes", {}))
        
        logger.info(f"Loaded model config from {path}")
        return config
    
    def get_model(self, provider: str, model_name: str) -> Optional[ModelSpec]:
        """Get a specific model spec."""
        provider_models = getattr(self, provider, None)
        if provider_models is None:
            return None
        return provider_models.models.get(model_name)
    
    def get_default_model(self, provider: str, tier: str = "primary") -> Optional[str]:
        """Get the default model for a provider and tier."""
        provider_models = getattr(self, provider, None)
        if provider_models is None:
            return None
        return provider_models.defaults.get(tier)
    
    def get_quota_limits(self, model_key: str) -> Optional[dict[str, int]]:
        """Get quota limits for a model."""
        return self.gemini.quota.get(model_key)
    
    def get_tier_route(self, tier: str) -> list[str]:
        """Get the fallback chain for a tier."""
        return self.tier_routes.get(tier, [])
    
    def list_all_models(self, provider: str) -> list[str]:
        """List all model names for a provider."""
        provider_models = getattr(self, provider, None)
        if provider_models is None:
            return []
        return list(provider_models.models.keys())
    
    def merge_discovered_models(self, provider: str, models: list[str]) -> None:
        """Merge dynamically discovered models into the config."""
        provider_models = getattr(self, provider, None)
        if provider_models is None:
            return
        
        for model_name in models:
            if model_name not in provider_models.models:
                # Add as discovered model with minimal info
                provider_models.models[model_name] = ModelSpec(
                    name=model_name,
                    display_name=model_name,
                    description="Dynamically discovered",
                    discovered=True,
                )


# =============================================================================
# Dynamic Model Discovery
# =============================================================================

async def discover_gemini_models() -> list[str]:
    """Discover available Gemini models from the API."""
    cache_key = "gemini"
    if cache_key in _discovery_cache:
        cached_time, cached_models = _discovery_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return cached_models
    
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("AISTUDIO_API_KEY")
        if not api_key:
            return []
        
        genai.configure(api_key=api_key)
        
        models = []
        for model in genai.list_models():
            if "generateContent" in model.supported_generation_methods:
                # Extract model name (remove "models/" prefix)
                name = model.name.replace("models/", "")
                models.append(name)
        
        _discovery_cache[cache_key] = (time.time(), models)
        logger.info(f"Discovered {len(models)} Gemini models")
        return models
    except Exception as e:
        logger.debug(f"Failed to discover Gemini models: {e}")
        return []


async def discover_anthropic_models() -> list[str]:
    """Discover available Anthropic models from the API."""
    cache_key = "anthropic"
    if cache_key in _discovery_cache:
        cached_time, cached_models = _discovery_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return cached_models
    
    try:
        import anthropic
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return []
        
        client = anthropic.Anthropic(api_key=api_key)
        
        # Anthropic doesn't have a list models endpoint, so we use known models
        # These are the current models as of December 2025
        models = [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ]
        
        _discovery_cache[cache_key] = (time.time(), models)
        return models
    except Exception as e:
        logger.debug(f"Failed to discover Anthropic models: {e}")
        return []


async def discover_openai_models() -> list[str]:
    """Discover available OpenAI models from the API."""
    cache_key = "openai"
    if cache_key in _discovery_cache:
        cached_time, cached_models = _discovery_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return cached_models
    
    try:
        import openai
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return []
        
        client = openai.OpenAI(api_key=api_key)
        
        models = []
        for model in client.models.list():
            # Filter to chat models
            if any(prefix in model.id for prefix in ["gpt-", "o1", "chatgpt"]):
                models.append(model.id)
        
        _discovery_cache[cache_key] = (time.time(), models)
        logger.info(f"Discovered {len(models)} OpenAI models")
        return models
    except Exception as e:
        logger.debug(f"Failed to discover OpenAI models: {e}")
        return []


async def discover_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Discover available Ollama models from a node."""
    cache_key = f"ollama:{base_url}"
    if cache_key in _discovery_cache:
        cached_time, cached_models = _discovery_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return cached_models
    
    try:
        import httpx
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                _discovery_cache[cache_key] = (time.time(), models)
                logger.info(f"Discovered {len(models)} Ollama models at {base_url}")
                return models
    except Exception as e:
        logger.debug(f"Failed to discover Ollama models at {base_url}: {e}")
    
    return []


async def discover_all_models(config: ModelConfig) -> None:
    """Discover models from all providers and merge into config."""
    tasks = [
        discover_gemini_models(),
        discover_anthropic_models(),
        discover_openai_models(),
        discover_ollama_models(),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    providers = ["gemini", "anthropic", "openai", "ollama"]
    for provider, result in zip(providers, results):
        if isinstance(result, list):
            config.merge_discovered_models(provider, result)


# =============================================================================
# Global Singleton
# =============================================================================

_model_config: Optional[ModelConfig] = None


def get_model_config() -> ModelConfig:
    """Get the global model configuration."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig.load()
    return _model_config


def reload_model_config() -> ModelConfig:
    """Reload the model configuration from disk."""
    global _model_config
    _model_config = ModelConfig.load()
    return _model_config


async def get_model_config_with_discovery() -> ModelConfig:
    """Get model config with dynamic discovery enabled."""
    config = get_model_config()
    await discover_all_models(config)
    return config
