"""Model Configuration Manager - Latest 2025 AI Models.

Manages model selection across providers (OpenAI, Gemini, Anthropic, Ollama)
with automatic routing, quota management, and API key validation.

Latest Models (2025):
- OpenAI: GPT-5.2-Codex, o3, o4-mini
- Google Gemini: Gemini 3 Flash, Gemini 3 Pro
- Anthropic Claude: Opus 4.5, Sonnet 4.5, Haiku 4.5
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import tomllib
except ImportError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib

logger = logging.getLogger(__name__)


class TaskTier(Enum):
    """Task complexity tiers for model routing."""

    FAST = "fast"  # Quick, cheap tasks
    CODING = "coding"  # Code generation
    REASONING = "reasoning"  # Complex reasoning
    MULTIMODAL = "multimodal"  # Vision + chat


class Provider(Enum):
    """AI provider identifiers."""

    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HALEXT = "halext"  # Local halext server


@dataclass
class ModelInfo:
    """Model metadata and capabilities."""

    provider: Provider
    model_id: str
    tier: TaskTier
    cost_per_1m_input: float
    cost_per_1m_output: float
    context_window: int = 128000
    supports_vision: bool = False
    is_local: bool = False
    is_deprecated: bool = False


@dataclass
class ProviderConfig:
    """Provider configuration and credentials."""

    provider: Provider
    enabled: bool
    api_key: Optional[str]
    models: dict[str, str]  # tier -> model_id
    deprecated: dict[str, str]


class ModelConfigManager:
    """Manages model configuration and routing across providers."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize model config manager.

        Args:
            config_path: Path to models.toml (default: ~/.config/hafs/models.toml)
        """
        if config_path is None:
            config_path = Path.home() / ".config" / "hafs" / "models.toml"

        self.config_path = config_path
        self.config: dict[str, Any] = {}
        self.providers: dict[Provider, ProviderConfig] = {}
        self._api_keys_validated: dict[Provider, bool] = {}

        self.load_config()

    def load_config(self) -> None:
        """Load model configuration from TOML file."""
        if not self.config_path.exists():
            logger.warning(f"Model config not found: {self.config_path}")
            logger.warning("Using default configuration")
            self._use_defaults()
            return

        try:
            with self.config_path.open("rb") as handle:
                self.config = tomllib.load(handle)
            self._parse_providers()
            logger.info(f"Loaded model config from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            self._use_defaults()

    def _parse_providers(self) -> None:
        """Parse provider configurations from TOML."""
        providers_config = self.config.get("providers", {})

        for provider_name, provider_data in providers_config.items():
            try:
                provider = Provider(provider_name)

                # Get API key from environment
                api_key_env = provider_data.get("api_key_env")
                api_key = os.getenv(api_key_env) if api_key_env else None

                # Parse models
                models = provider_data.get("models", {})
                deprecated = provider_data.get("deprecated", {})

                provider_config = ProviderConfig(
                    provider=provider,
                    enabled=provider_data.get("enabled", True),
                    api_key=api_key,
                    models=models,
                    deprecated=deprecated,
                )

                self.providers[provider] = provider_config

            except ValueError:
                logger.warning(f"Unknown provider: {provider_name}")
            except Exception as e:
                logger.error(f"Error parsing provider {provider_name}: {e}")

    def _use_defaults(self) -> None:
        """Use default configuration when config file not found."""
        # Default to Gemini 3 Flash for everything
        self.providers[Provider.GEMINI] = ProviderConfig(
            provider=Provider.GEMINI,
            enabled=True,
            api_key=os.getenv("GEMINI_API_KEY"),
            models={
                "fast": "gemini-3-flash",
                "coding": "gemini-3-flash",
                "reasoning": "gemini-3-pro",
                "multimodal": "gemini-3-flash",
            },
            deprecated={},
        )

        logger.info("Using default Gemini 3 configuration")

    def get_model_for_tier(
        self, tier: TaskTier, provider: Optional[Provider] = None
    ) -> Optional[tuple[Provider, str]]:
        """Get best model for a given task tier.

        Args:
            tier: Task complexity tier
            provider: Specific provider to use (None = auto-select)

        Returns:
            Tuple of (provider, model_id) or None if no model available
        """
        # Get routing preference for tier
        routing_config = self.config.get("routing", {}).get(tier.value, {})
        provider_order = routing_config.get("providers", [])

        # If specific provider requested, try that first
        if provider:
            if provider in self.providers and self.providers[provider].enabled:
                model_id = self.providers[provider].models.get(tier.value)
                if model_id:
                    return (provider, model_id)

        # Try providers in preference order
        for provider_name in provider_order:
            try:
                prov = Provider(provider_name)
                if prov in self.providers and self.providers[prov].enabled:
                    model_id = self.providers[prov].models.get(tier.value)
                    if model_id:
                        logger.debug(
                            f"Selected {prov.value}/{model_id} for tier {tier.value}"
                        )
                        return (prov, model_id)
            except ValueError:
                continue

        # Fallback: use first available provider
        for prov, config in self.providers.items():
            if config.enabled:
                model_id = config.models.get(tier.value)
                if model_id:
                    logger.warning(
                        f"Using fallback {prov.value}/{model_id} for tier {tier.value}"
                    )
                    return (prov, model_id)

        logger.error(f"No model available for tier {tier.value}")
        return None

    def validate_api_keys(self) -> dict[Provider, bool]:
        """Validate API keys for all enabled providers.

        Returns:
            Dictionary mapping Provider to validation status (True/False)
        """
        validation_results = {}

        for provider, config in self.providers.items():
            if not config.enabled:
                validation_results[provider] = False
                continue

            # Check if API key is set
            if provider == Provider.OLLAMA:
                # Ollama doesn't need API key
                validation_results[provider] = True
                continue

            if not config.api_key:
                logger.warning(f"No API key configured for {provider.value}")
                validation_results[provider] = False
                continue

            # Basic validation: check key format
            if provider == Provider.OPENAI:
                is_valid = config.api_key.startswith("sk-")
            elif provider == Provider.GEMINI:
                is_valid = config.api_key.startswith("AIza")
            elif provider == Provider.ANTHROPIC:
                is_valid = config.api_key.startswith("sk-ant-")
            else:
                is_valid = bool(config.api_key)

            validation_results[provider] = is_valid

            if is_valid:
                logger.info(f"✓ {provider.value} API key validated")
            else:
                logger.warning(f"✗ {provider.value} API key format invalid")

        self._api_keys_validated = validation_results
        return validation_results

    def get_fallback_chain(self, tier: TaskTier) -> list[tuple[Provider, str]]:
        """Get fallback chain for a tier (ordered by preference).

        Args:
            tier: Task complexity tier

        Returns:
            List of (provider, model_id) tuples in fallback order
        """
        routing_config = self.config.get("routing", {}).get(tier.value, {})
        fallback_models = routing_config.get("fallback_order", [])

        chain = []
        for model_id in fallback_models:
            # Find which provider has this model
            for prov, config in self.providers.items():
                if config.enabled and model_id in config.models.values():
                    chain.append((prov, model_id))
                    break

        return chain

    def is_model_deprecated(self, provider: Provider, model_id: str) -> bool:
        """Check if a model is deprecated.

        Args:
            provider: AI provider
            model_id: Model identifier

        Returns:
            True if model is deprecated, False otherwise
        """
        if provider not in self.providers:
            return False

        deprecated = self.providers[provider].deprecated
        return model_id in deprecated.values()

    def get_model_cost(self, provider: Provider, model_id: str) -> tuple[float, float]:
        """Get cost per 1M tokens for a model.

        Args:
            provider: AI provider
            model_id: Model identifier

        Returns:
            Tuple of (input_cost, output_cost) per 1M tokens
        """
        pricing = self.config.get("quota", {}).get("pricing", {})

        if model_id in pricing:
            return (pricing[model_id]["input"], pricing[model_id]["output"])

        # Default pricing if not found
        return (1.0, 3.0)

    def get_status_report(self) -> str:
        """Generate status report of model configuration.

        Returns:
            Formatted status report string
        """
        lines = ["Model Configuration Status", "=" * 80]

        # Validate API keys
        validation = self.validate_api_keys()

        for provider, config in self.providers.items():
            status = "✓ ENABLED" if config.enabled else "✗ DISABLED"
            key_status = "✓ Valid" if validation.get(provider, False) else "✗ Invalid"

            lines.append(f"\n{provider.value.upper()}: {status} (API Key: {key_status})")

            if config.enabled:
                lines.append("  Models:")
                for tier, model_id in config.models.items():
                    deprecated = " [DEPRECATED]" if self.is_model_deprecated(
                        provider, model_id
                    ) else ""
                    lines.append(f"    {tier:12s}: {model_id}{deprecated}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


# Global singleton
_model_config: Optional[ModelConfigManager] = None


def get_model_config() -> ModelConfigManager:
    """Get global model configuration manager."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfigManager()
    return _model_config


def main():
    """Test model configuration."""
    logging.basicConfig(level=logging.INFO)

    config = ModelConfigManager()

    # Print status report
    print(config.get_status_report())

    # Test model selection
    print("\nModel Selection Tests:")
    for tier in TaskTier:
        result = config.get_model_for_tier(tier)
        if result:
            provider, model_id = result
            print(f"{tier.value:12s}: {provider.value}/{model_id}")

    # Test fallback chains
    print("\nFallback Chains:")
    for tier in TaskTier:
        chain = config.get_fallback_chain(tier)
        print(f"{tier.value:12s}: {' → '.join([f'{p.value}/{m}' for p, m in chain])}")


if __name__ == "__main__":
    main()
