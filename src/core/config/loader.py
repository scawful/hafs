"""Configuration loader for cognitive protocol.

This module provides Pydantic-based configuration loading with:
- TOML file parsing
- Schema validation with constraints
- Personality profile merging
- Hot reloading support
- Backward compatibility (defaults match hardcoded values)
"""

from pathlib import Path
from typing import Any, Optional
import tomllib
from pydantic import BaseModel, Field


# ============================================================================
# Metacognition Configuration Models
# ============================================================================

class HelpSeekingConfig(BaseModel):
    """Help-seeking behavior configuration."""

    uncertainty_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Ask for help when uncertainty exceeds this threshold"
    )
    failure_threshold: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Seek help after this many consecutive failures"
    )


class FlowStateConfig(BaseModel):
    """Flow state criteria configuration."""

    max_cognitive_load: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Maximum cognitive load for flow state"
    )
    min_strategy_effectiveness: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum strategy effectiveness for flow state"
    )
    max_frustration: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum frustration for flow state"
    )


class FrustrationConfig(BaseModel):
    """Frustration dynamics configuration."""

    delta_on_failure: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Increase frustration by this amount on failure"
    )
    delta_on_success: float = Field(
        default=-0.3,
        ge=-1.0,
        le=0.0,
        description="Decrease frustration by this amount on success"
    )


class StrategyConfig(BaseModel):
    """Strategy management configuration."""

    strategy_change_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Change strategy when effectiveness below this threshold"
    )
    effectiveness_reset: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Reset effectiveness to this value when changing strategy"
    )


class MetacognitionConfig(BaseModel):
    """Metacognition configuration."""

    spinning_threshold: int = Field(
        default=4,
        ge=3,
        le=6,
        description="Repeat action count before considering spinning"
    )
    max_action_history: int = Field(
        default=10,
        ge=5,
        le=50,
        description="Number of recent actions tracked for spin detection"
    )
    cognitive_load_warning: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Warn when cognitive load exceeds this threshold"
    )
    max_items_in_focus: int = Field(
        default=7,
        ge=3,
        le=15,
        description="Maximum items in focus (Miller's Law)"
    )

    help_seeking: HelpSeekingConfig = Field(default_factory=HelpSeekingConfig)
    flow_state: FlowStateConfig = Field(default_factory=FlowStateConfig)
    frustration: FrustrationConfig = Field(default_factory=FrustrationConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)


# ============================================================================
# Epistemic Configuration Models
# ============================================================================

class EpistemicConfig(BaseModel):
    """Epistemic knowledge tracking configuration."""

    auto_record_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Auto-record facts with confidence >= this threshold"
    )
    min_confidence_for_auto_record: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Alias for auto_record_confidence"
    )
    decay_rate_per_hour: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Confidence decay rate per hour"
    )
    prune_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Remove facts with confidence below this threshold"
    )
    max_golden_facts: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum immutable verified facts"
    )
    max_working_facts: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum working facts"
    )


# ============================================================================
# Emotions Configuration Models
# ============================================================================

class DecayRatesConfig(BaseModel):
    """Emotional decay rates configuration."""

    fear: float = Field(default=0.02, ge=0.0, le=1.0, description="Fear decay rate")
    curiosity: float = Field(default=0.1, ge=0.0, le=1.0, description="Curiosity decay rate")
    satisfaction: float = Field(default=0.05, ge=0.0, le=1.0, description="Satisfaction decay rate")
    frustration: float = Field(default=0.08, ge=0.0, le=1.0, description="Frustration decay rate")
    excitement: float = Field(default=0.15, ge=0.0, le=1.0, description="Excitement decay rate")
    determination: float = Field(default=0.03, ge=0.0, le=1.0, description="Determination decay rate")
    caution: float = Field(default=0.06, ge=0.0, le=1.0, description="Caution decay rate")
    relief: float = Field(default=0.12, ge=0.0, le=1.0, description="Relief decay rate")


class EmotionsConfig(BaseModel):
    """Emotional state configuration."""

    decay_rates: DecayRatesConfig = Field(default_factory=DecayRatesConfig)


# ============================================================================
# Grounding Configuration Models
# ============================================================================

class GroundingThresholdsConfig(BaseModel):
    """Auto-grounding threshold configuration."""

    anxiety_spiral_level: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Trigger anxiety spiral at this level"
    )
    anxiety_spiral_duration_minutes: int = Field(
        default=3,
        ge=1,
        le=60,
        description="Sustained anxiety duration for trigger"
    )
    frustration_loop_intensity: int = Field(
        default=7,
        ge=0,
        le=10,
        description="Frustration intensity trigger level"
    )
    frustration_loop_consecutive_failures: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Consecutive failures for frustration loop"
    )
    confidence_crash_drop_amount: int = Field(
        default=30,
        ge=10,
        le=100,
        description="Confidence drop amount for crash detection"
    )
    confidence_crash_window_minutes: int = Field(
        default=2,
        ge=1,
        le=60,
        description="Time window for confidence crash"
    )
    spin_detected_same_action_count: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Same action count for spin detection"
    )
    emotional_overload_total_intensity: int = Field(
        default=35,
        ge=10,
        le=100,
        description="Total emotional intensity for overload"
    )
    emotional_overload_emotion_count: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of active emotions for overload"
    )
    auto_grounding_enabled: bool = Field(
        default=True,
        description="Enable automatic grounding triggers"
    )


class GroundingConfig(BaseModel):
    """Grounding crisis detection configuration."""

    thresholds: GroundingThresholdsConfig = Field(default_factory=GroundingThresholdsConfig)


# ============================================================================
# Analysis Triggers Configuration Models
# ============================================================================

class TriggerCooldownsConfig(BaseModel):
    """Trigger cooldown periods (minutes)."""

    spinning_critic: int = Field(default=10, ge=1, le=120)
    edits_without_tests: int = Field(default=15, ge=1, le=120)
    high_anxiety_caution: int = Field(default=15, ge=1, le=120)
    consecutive_failures: int = Field(default=20, ge=1, le=120)
    high_cognitive_load: int = Field(default=20, ge=1, le=120)
    tool_repetition: int = Field(default=5, ge=1, le=120)
    baseline_too_high: int = Field(default=30, ge=1, le=120)
    error_amplification: int = Field(default=15, ge=1, le=120)


class AnalysisTriggersConfig(BaseModel):
    """Analysis trigger configuration."""

    edits_without_tests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Trigger after this many edits without tests"
    )
    high_anxiety_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Trigger at this anxiety level"
    )
    high_cognitive_load_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Trigger at this cognitive load"
    )
    baseline_too_high_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Trigger when baseline exceeds this"
    )
    error_amplification_threshold: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Trigger after this many amplified errors"
    )
    consecutive_failures_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive failures for trigger"
    )
    consecutive_failures_window_ms: int = Field(
        default=300000,
        ge=10000,
        le=3600000,
        description="Time window for consecutive failures (ms)"
    )
    tool_repetition_count: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Tool calls for repetition trigger"
    )
    tool_repetition_window_ms: int = Field(
        default=60000,
        ge=10000,
        le=600000,
        description="Time window for tool repetition (ms)"
    )
    event_cleanup_interval_ms: int = Field(
        default=600000,
        ge=60000,
        le=3600000,
        description="Event cleanup interval (ms)"
    )

    cooldowns: TriggerCooldownsConfig = Field(default_factory=TriggerCooldownsConfig)


# ============================================================================
# Hivemind Configuration Models
# ============================================================================

class HivemindDecayRatesConfig(BaseModel):
    """Hivemind decay rates."""

    fear: float = Field(default=0.1, ge=0.0, le=1.0)
    satisfaction: float = Field(default=0.1, ge=0.0, le=1.0)
    knowledge: float = Field(default=0.0, ge=0.0, le=1.0)
    decision: float = Field(default=0.05, ge=0.0, le=1.0)
    preference: float = Field(default=0.0, ge=0.0, le=1.0)


class HivemindDecayConfig(BaseModel):
    """Hivemind decay configuration."""

    check_interval_ms: int = Field(
        default=3600000,
        ge=60000,
        le=86400000,
        description="Decay check interval (ms)"
    )
    golden_exempt: bool = Field(
        default=True,
        description="Exempt golden facts from decay"
    )
    preferences_exempt: bool = Field(
        default=True,
        description="Exempt preferences from decay"
    )
    default_rates: HivemindDecayRatesConfig = Field(default_factory=HivemindDecayRatesConfig)


class HivemindConfig(BaseModel):
    """Hivemind cross-agent configuration."""

    global_enabled: bool = Field(
        default=False,
        description="Enable global hivemind"
    )
    council_size: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Council size"
    )
    council_quorum: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Quorum for decisions"
    )
    council_threshold: float = Field(
        default=0.67,
        ge=0.5,
        le=1.0,
        description="Agreement threshold"
    )
    council_agents: list[str] = Field(
        default_factory=lambda: ["explore", "critic", "general"],
        description="Council agent names"
    )

    decay: HivemindDecayConfig = Field(default_factory=HivemindDecayConfig)


# ============================================================================
# Goals Configuration Models
# ============================================================================

class ConflictPatternConfig(BaseModel):
    """Goal conflict pattern."""

    name: str
    keywords_a: list[str]
    keywords_b: list[str]


class GoalsConfig(BaseModel):
    """Goals management configuration."""

    max_conflicts_to_track: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum conflicts to track"
    )
    auto_detect_conflicts: bool = Field(
        default=True,
        description="Auto-detect goal conflicts"
    )
    conflict_patterns: list[ConflictPatternConfig] = Field(
        default_factory=lambda: [
            ConflictPatternConfig(
                name="minimize_vs_refactor",
                keywords_a=["minimize", "small change", "minimal", "quick fix"],
                keywords_b=["refactor", "restructure", "rewrite", "overhaul"]
            ),
            ConflictPatternConfig(
                name="speed_vs_quality",
                keywords_a=["fast", "quick", "asap", "immediately"],
                keywords_b=["quality", "thorough", "comprehensive", "robust"]
            ),
            ConflictPatternConfig(
                name="backward_compat_vs_modernize",
                keywords_a=["backward compatible", "don't break", "preserve", "legacy"],
                keywords_b=["modernize", "upgrade", "latest", "drop support"]
            ),
            ConflictPatternConfig(
                name="simple_vs_feature_rich",
                keywords_a=["simple", "minimal", "basic", "lightweight"],
                keywords_b=["feature-rich", "comprehensive", "full-featured", "advanced"]
            ),
            ConflictPatternConfig(
                name="pragmatic_vs_perfect",
                keywords_a=["pragmatic", "good enough", "ship it", "iterate"],
                keywords_b=["perfect", "optimal", "ideal", "flawless"]
            ),
        ]
    )


# ============================================================================
# Performance Configuration Models
# ============================================================================

class PerformanceConfig(BaseModel):
    """I/O performance configuration."""

    enable_batching: bool = Field(
        default=True,
        description="Enable write batching"
    )
    batch_flush_interval_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Batch flush interval (ms)"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable read caching"
    )
    cache_ttl_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Cache TTL (seconds)"
    )
    lazy_load: bool = Field(
        default=True,
        description="Enable lazy loading"
    )


# ============================================================================
# Full Configuration Model
# ============================================================================

class CognitiveProtocolConfig(BaseModel):
    """Complete cognitive protocol configuration."""

    metacognition: MetacognitionConfig = Field(default_factory=MetacognitionConfig)
    epistemic: EpistemicConfig = Field(default_factory=EpistemicConfig)
    emotions: EmotionsConfig = Field(default_factory=EmotionsConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    analysis_triggers: AnalysisTriggersConfig = Field(default_factory=AnalysisTriggersConfig)
    hivemind: HivemindConfig = Field(default_factory=HivemindConfig)
    goals: GoalsConfig = Field(default_factory=GoalsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """Load and merge cognitive protocol configuration."""

    def __init__(
        self,
        base_config_path: Optional[Path] = None,
        personality_config_path: Optional[Path] = None,
        personality: Optional[str] = None
    ):
        """
        Initialize config loader.

        Args:
            base_config_path: Path to cognitive_protocol.toml
            personality_config_path: Path to agent_personalities.toml
            personality: Personality name to load
        """
        self.base_config_path = base_config_path or Path("config/cognitive_protocol.toml")
        self.personality_config_path = personality_config_path or Path("config/agent_personalities.toml")
        self.personality = personality
        self._config: Optional[CognitiveProtocolConfig] = None

    def load(self) -> CognitiveProtocolConfig:
        """Load and merge configuration."""
        # Load base config
        base_data = self._load_toml(self.base_config_path)
        config = CognitiveProtocolConfig(**base_data)

        # Apply personality overrides if specified
        if self.personality:
            personality_data = self._load_toml(self.personality_config_path)
            if "personalities" in personality_data and self.personality in personality_data["personalities"]:
                # Get personality-specific overrides
                personality_overrides = personality_data["personalities"][self.personality]

                # Remove description if present (not part of config schema)
                personality_overrides.pop("description", None)

                # Merge overrides into config
                config = self._merge_config(config, personality_overrides)

        self._config = config
        return config

    def _load_toml(self, path: Path) -> dict[str, Any]:
        """Load TOML file."""
        if not path.exists():
            return {}

        with open(path, "rb") as f:
            return tomllib.load(f)

    def _merge_config(
        self,
        base: CognitiveProtocolConfig,
        overrides: dict[str, Any]
    ) -> CognitiveProtocolConfig:
        """Merge personality overrides into base config."""
        # Convert base to dict
        base_dict = base.model_dump()

        # Deep merge overrides
        merged = self._deep_merge(base_dict, overrides)

        # Reconstruct from merged dict
        return CognitiveProtocolConfig(**merged)

    def _deep_merge(self, base: dict, overrides: dict) -> dict:
        """Recursively merge dictionaries."""
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @property
    def config(self) -> CognitiveProtocolConfig:
        """Get loaded config (loads if not already loaded)."""
        if self._config is None:
            return self.load()
        return self._config


# ============================================================================
# Global Config Instance
# ============================================================================

_config_loader: Optional[ConfigLoader] = None


def get_config(
    reload: bool = False,
    personality: Optional[str] = None,
    base_config_path: Optional[Path] = None,
    personality_config_path: Optional[Path] = None
) -> CognitiveProtocolConfig:
    """
    Get cognitive protocol configuration (singleton).

    Args:
        reload: Force reload from disk
        personality: Personality to load (if different from current)
        base_config_path: Custom base config path
        personality_config_path: Custom personality config path

    Returns:
        Loaded configuration

    Examples:
        >>> # Load default config
        >>> config = get_config()
        >>> print(config.metacognition.spinning_threshold)
        4

        >>> # Load with personality
        >>> config = get_config(personality="cautious")
        >>> print(config.metacognition.spinning_threshold)
        3

        >>> # Force reload
        >>> config = get_config(reload=True)
    """
    global _config_loader

    # Create new loader if needed
    needs_new_loader = (
        _config_loader is None or
        reload or
        (personality and personality != _config_loader.personality) or
        (base_config_path and base_config_path != _config_loader.base_config_path) or
        (personality_config_path and personality_config_path != _config_loader.personality_config_path)
    )

    if needs_new_loader:
        _config_loader = ConfigLoader(
            base_config_path=base_config_path,
            personality_config_path=personality_config_path,
            personality=personality
        )
        _config_loader.load()

    return _config_loader.config
