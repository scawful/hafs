"""Configuration presets for MoE system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TempPreset(Enum):
    """Temperature presets for different use cases."""

    DETERMINISTIC = 0.1  # Very low - for exact, repeatable outputs
    LOW = 0.3           # Low - for precise, focused responses (code, classification)
    MEDIUM = 0.7        # Medium - balanced creativity and precision
    HIGH = 1.0          # High - more creative and varied outputs
    CREATIVE = 1.2      # Very high - maximum creativity (synthesis, brainstorming)


class TokenPreset(Enum):
    """Token limit presets for different task types."""

    TINY = 100          # Quick classifications
    SHORT = 200         # Brief responses
    MEDIUM = 512        # Standard responses
    LONG = 1024         # Detailed explanations
    VERY_LONG = 2048    # Code with explanations
    MAXIMUM = 4096      # Full synthesis or complex solutions


@dataclass
class ExpertPreset:
    """Configuration preset for an expert."""

    name: str
    description: str
    temp_preset: TempPreset
    token_preset: TokenPreset

    @property
    def temperature(self) -> float:
        """Get temperature value."""
        return self.temp_preset.value

    @property
    def max_tokens(self) -> int:
        """Get max tokens value."""
        return self.token_preset.value


# Predefined expert presets
EXPERT_PRESETS = {
    "asm": ExpertPreset(
        name="asm",
        description="Assembly code generation - precise, low temperature",
        temp_preset=TempPreset.LOW,
        token_preset=TokenPreset.VERY_LONG,
    ),
    "yaze": ExpertPreset(
        name="yaze",
        description="YAZE tool usage - balanced precision",
        temp_preset=TempPreset.MEDIUM,
        token_preset=TokenPreset.VERY_LONG,
    ),
    "debug": ExpertPreset(
        name="debug",
        description="Debugging - focused analysis",
        temp_preset=TempPreset.LOW,
        token_preset=TokenPreset.LONG,
    ),
    "classifier": ExpertPreset(
        name="classifier",
        description="Task classification - deterministic",
        temp_preset=TempPreset.DETERMINISTIC,
        token_preset=TokenPreset.SHORT,
    ),
    "synthesizer": ExpertPreset(
        name="synthesizer",
        description="Multi-expert synthesis - creative integration",
        temp_preset=TempPreset.HIGH,
        token_preset=TokenPreset.MAXIMUM,
    ),
}


def get_preset(preset_name: str) -> ExpertPreset:
    """Get a preset by name.

    Args:
        preset_name: Name of preset (asm, yaze, debug, classifier, synthesizer).

    Returns:
        ExpertPreset configuration.

    Raises:
        KeyError: If preset not found.
    """
    if preset_name not in EXPERT_PRESETS:
        raise KeyError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(EXPERT_PRESETS.keys())}"
        )
    return EXPERT_PRESETS[preset_name]


def list_presets() -> dict[str, str]:
    """List all available presets.

    Returns:
        Dict mapping preset names to descriptions.
    """
    return {
        name: preset.description
        for name, preset in EXPERT_PRESETS.items()
    }


# Example usage:
# from hafs.agents.moe.config import TempPreset, TokenPreset
#
# # Use presets
# classifier = TaskClassifier(
#     temperature=TempPreset.LOW.value,
#     max_tokens=TokenPreset.SHORT.value,
# )
#
# # Or use expert preset
# from hafs.agents.moe.config import get_preset
# asm_preset = get_preset("asm")
# asm_expert = AsmExpert(
#     temperature=asm_preset.temperature,
#     max_tokens=asm_preset.max_tokens,
# )
#
# # Or customize:
# custom_expert = AsmExpert(
#     temperature=0.4,  # Between LOW and MEDIUM
#     max_tokens=3000,  # Custom value
# )
