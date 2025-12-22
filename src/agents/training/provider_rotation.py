"""Multi-model provider rotation for training data generation.

Enables generating samples from a mixture of models (Gemini, Anthropic, OpenAI, local)
to improve dataset diversity and reduce single-model bias.
"""

from __future__ import annotations

import logging
import random
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from agents.training.quality_config import resolve_training_config_path

logger = logging.getLogger(__name__)


class ProviderWeight(Enum):
    """Preset provider weight configurations."""

    GEMINI_ONLY = "gemini_only"           # Current default
    BALANCED = "balanced"                  # Equal weights across cloud providers
    LOCAL_HEAVY = "local_heavy"            # Prefer local models
    DIVERSE = "diverse"                    # Mix of all available
    COST_OPTIMIZED = "cost_optimized"      # Prefer cheaper providers


@dataclass
class ProviderConfig:
    """Configuration for a single provider in rotation."""

    name: str                              # Provider name (gemini, anthropic, openai, etc.)
    weight: float = 1.0                    # Relative weight (higher = more samples)
    model: str = ""                        # Specific model override (optional)
    enabled: bool = True                   # Whether this provider is active
    max_concurrent: int = 5                # Max concurrent requests
    fallback_order: int = 99               # Order for fallback (lower = higher priority)

    # Rate limiting
    requests_per_minute: int = 60

    # Quality adjustments
    quality_bonus: float = 0.0             # Add to quality score for this provider


@dataclass
class ProviderRotation:
    """Manages provider rotation and selection for training generation.

    Supports:
    - Weighted random selection across providers
    - Fallback on provider failure
    - Provider-specific model overrides
    - Quality tracking per provider
    """

    providers: list[ProviderConfig] = field(default_factory=list)
    _total_weight: float = 0.0
    _selection_counts: dict[str, int] = field(default_factory=dict)
    _failure_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._recalculate_weights()

    def _recalculate_weights(self):
        """Recalculate total weight from enabled providers."""
        self._total_weight = sum(
            p.weight for p in self.providers if p.enabled
        )

    @classmethod
    def from_preset(cls, preset: ProviderWeight) -> "ProviderRotation":
        """Create rotation from a preset configuration."""

        # Latest models as of December 2025:
        # - Gemini: gemini-3-flash-preview (Dec 17, 2025), gemini-3-pro-preview
        # - Anthropic: claude-opus-4-5-20251101 (Nov 24, 2025), claude-sonnet-4-20250514
        # - OpenAI: gpt-5.2-codex, gpt-5.2, o4-mini

        if preset == ProviderWeight.GEMINI_ONLY:
            return cls(providers=[
                ProviderConfig(name="gemini", weight=1.0, model="gemini-3-flash-preview"),
            ])

        elif preset == ProviderWeight.BALANCED:
            return cls(providers=[
                ProviderConfig(name="gemini", weight=1.0, model="gemini-3-flash-preview"),
                ProviderConfig(name="anthropic", weight=1.0, model="claude-sonnet-4-20250514"),
                ProviderConfig(name="openai", weight=1.0, model="gpt-5.2-codex"),
            ])

        elif preset == ProviderWeight.LOCAL_HEAVY:
            return cls(providers=[
                ProviderConfig(name="llamacpp", weight=3.0, model="qwen2.5-coder-7b"),
                ProviderConfig(name="gemini", weight=1.0, model="gemini-3-flash-preview"),
            ])

        elif preset == ProviderWeight.DIVERSE:
            return cls(providers=[
                ProviderConfig(name="gemini", weight=2.0, model="gemini-3-flash-preview"),
                ProviderConfig(name="anthropic", weight=2.0, model="claude-opus-4-5-20251101"),
                ProviderConfig(name="openai", weight=1.5, model="gpt-5.2-codex"),
                ProviderConfig(name="llamacpp", weight=1.0, model="qwen2.5-coder-7b"),
            ])

        elif preset == ProviderWeight.COST_OPTIMIZED:
            return cls(providers=[
                ProviderConfig(name="gemini", weight=3.0, model="gemini-3-flash-preview"),
                ProviderConfig(name="llamacpp", weight=2.0, model="qwen2.5-coder-7b"),  # Local = free
                ProviderConfig(name="openai", weight=1.0, model="o4-mini"),  # Fast/cheap reasoning
            ])

        # Default to Gemini only
        return cls(providers=[
            ProviderConfig(name="gemini", weight=1.0),
        ])

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ProviderRotation":
        """Create rotation from configuration dictionary."""
        providers = []
        for p_config in config.get("providers", []):
            providers.append(ProviderConfig(
                name=p_config["name"],
                weight=p_config.get("weight", 1.0),
                model=p_config.get("model", ""),
                enabled=p_config.get("enabled", True),
                max_concurrent=p_config.get("max_concurrent", 5),
                fallback_order=p_config.get("fallback_order", 99),
                requests_per_minute=p_config.get("requests_per_minute", 60),
                quality_bonus=p_config.get("quality_bonus", 0.0),
            ))
        return cls(providers=providers)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "providers": [
                {
                    "name": p.name,
                    "weight": p.weight,
                    "model": p.model,
                    "enabled": p.enabled,
                    "max_concurrent": p.max_concurrent,
                    "fallback_order": p.fallback_order,
                    "requests_per_minute": p.requests_per_minute,
                    "quality_bonus": p.quality_bonus,
                }
                for p in self.providers
            ],
            "stats": {
                "selection_counts": self._selection_counts,
                "failure_counts": self._failure_counts,
            }
        }

    def select_provider(self) -> Optional[ProviderConfig]:
        """Select a provider based on weights.

        Uses weighted random selection. Returns None if no providers available.
        """
        enabled = [p for p in self.providers if p.enabled]
        if not enabled:
            return None

        # Weighted random selection
        r = random.uniform(0, self._total_weight)
        cumulative = 0.0

        for provider in enabled:
            cumulative += provider.weight
            if r <= cumulative:
                # Track selection
                self._selection_counts[provider.name] = (
                    self._selection_counts.get(provider.name, 0) + 1
                )
                return provider

        # Fallback to last enabled provider
        return enabled[-1]

    def get_fallback(self, failed_provider: str) -> Optional[ProviderConfig]:
        """Get fallback provider after failure.

        Args:
            failed_provider: Name of provider that failed

        Returns:
            Next provider in fallback order, or None if none available
        """
        # Track failure
        self._failure_counts[failed_provider] = (
            self._failure_counts.get(failed_provider, 0) + 1
        )

        # Sort by fallback order, excluding failed provider
        candidates = sorted(
            [p for p in self.providers if p.enabled and p.name != failed_provider],
            key=lambda p: p.fallback_order
        )

        return candidates[0] if candidates else None

    def disable_provider(self, name: str) -> None:
        """Temporarily disable a provider (e.g., on repeated failures)."""
        for p in self.providers:
            if p.name == name:
                p.enabled = False
                logger.warning(f"Disabled provider: {name}")
                break
        self._recalculate_weights()

    def enable_provider(self, name: str) -> None:
        """Re-enable a provider."""
        for p in self.providers:
            if p.name == name:
                p.enabled = True
                logger.info(f"Enabled provider: {name}")
                break
        self._recalculate_weights()

    def get_stats(self) -> dict[str, Any]:
        """Get provider usage statistics."""
        total_selections = sum(self._selection_counts.values())
        total_failures = sum(self._failure_counts.values())

        return {
            "total_selections": total_selections,
            "total_failures": total_failures,
            "success_rate": (total_selections - total_failures) / max(total_selections, 1),
            "per_provider": {
                p.name: {
                    "selections": self._selection_counts.get(p.name, 0),
                    "failures": self._failure_counts.get(p.name, 0),
                    "weight": p.weight,
                    "enabled": p.enabled,
                }
                for p in self.providers
            }
        }


def load_provider_rotation(
    config_path: Optional[Path] = None,
) -> Optional[ProviderRotation]:
    """Load provider rotation configuration from training.toml."""
    resolved_path = resolve_training_config_path(config_path)
    if resolved_path is None or not resolved_path.exists():
        return None

    try:
        with open(resolved_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        logger.warning(f"Failed to load provider rotation config: {e}")
        return None

    rotation_config = data.get("generation", {}).get("provider_rotation", {})
    if not rotation_config:
        return None

    providers = rotation_config.get("providers")
    if isinstance(providers, list) and providers:
        return ProviderRotation.from_dict(rotation_config)

    preset_value = str(rotation_config.get("preset", "")).strip().lower()
    if preset_value:
        for preset in ProviderWeight:
            if preset.value == preset_value:
                return ProviderRotation.from_preset(preset)
        logger.warning(f"Unknown provider rotation preset: {preset_value}")

    return None


def get_provider_enum(name: str):
    """Convert provider name string to Provider enum."""
    from core.orchestrator_v2 import Provider

    name_map = {
        "gemini": Provider.GEMINI,
        "anthropic": Provider.ANTHROPIC,
        "openai": Provider.OPENAI,
        "llamacpp": Provider.LLAMACPP,
        "ollama": Provider.OLLAMA,
        "halext": Provider.HALEXT,
    }
    return name_map.get(name.lower(), Provider.GEMINI)


def get_tier_enum(tier: str):
    """Convert tier string to TaskTier enum."""
    from core.orchestrator_v2 import TaskTier

    tier_map = {
        "fast": TaskTier.FAST,
        "coding": TaskTier.CODING,
        "reasoning": TaskTier.REASONING,
        "creative": TaskTier.CREATIVE,
        "research": TaskTier.RESEARCH,
        "local": TaskTier.LOCAL,
        "cheap": TaskTier.CHEAP,
    }
    return tier_map.get(tier.lower(), TaskTier.CODING)
