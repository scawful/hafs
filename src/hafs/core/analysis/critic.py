"""Adaptive Harsh Critic - Analysis mode for code criticism.

Implements adaptive critic tone per PROTOCOL_SPEC.md Section 5.2.4,
based on "Mind Your Tone" findings (2510.04950v1).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class CriticTone(str, Enum):
    """Critic tone levels per PROTOCOL_SPEC.md."""

    NEUTRAL = "neutral"        # No prefix
    DIRECT = "direct"          # "Identify issues in this:"
    CHALLENGING = "challenging"  # "I doubt this is correct. Find the problems:"
    HARSH = "harsh"            # "This looks wrong. Point out every flaw:"


# Tone prefixes from the paper
TONE_PREFIXES = {
    CriticTone.NEUTRAL: "",
    CriticTone.DIRECT: "Identify issues in this:",
    CriticTone.CHALLENGING: "I doubt this is correct. Find the problems:",
    CriticTone.HARSH: "This looks wrong. Point out every flaw:",
}


class CriticReview(BaseModel):
    """A single review finding from the critic."""

    severity: str = "minor"  # critical | major | minor | nitpick
    aspect: str = ""
    location: Optional[dict[str, Any]] = None  # { file, line }
    issue: str = ""
    suggestion: str = ""
    confidence: float = 0.8


class AdaptiveCriticConfig(BaseModel):
    """Configuration for the adaptive critic per PROTOCOL_SPEC.md Section 5.2.4."""

    # Default: harsh (per Mind Your Tone findings: +4% accuracy)
    default_tone: CriticTone = CriticTone.HARSH

    # Automatic downgrade thresholds
    anxiety_threshold: float = 0.7
    anxiety_downgrade_to: CriticTone = CriticTone.DIRECT

    frustration_count_threshold: int = 3
    frustration_downgrade_to: CriticTone = CriticTone.NEUTRAL

    # Manual override
    allow_user_escalation: bool = True
    allow_user_override: bool = True

    # Gradual escalation when stable
    stable_iterations_for_escalation: int = 5
    escalation_path: list[CriticTone] = Field(
        default_factory=lambda: [
            CriticTone.NEUTRAL,
            CriticTone.DIRECT,
            CriticTone.CHALLENGING,
            CriticTone.HARSH,
        ]
    )


class CriticState(BaseModel):
    """Tracks the critic's adaptive state."""

    current_tone: CriticTone = CriticTone.HARSH
    stable_iterations: int = 0
    user_override: Optional[CriticTone] = None
    last_anxiety: float = 0.0
    frustration_count: int = 0


class AdaptiveCritic:
    """Adaptive harsh critic with tone state machine.

    Tone State Machine (from PROTOCOL_SPEC.md):
    ```
    harsh ──(anxiety > 0.7)──► direct ──(frustrations >= 3)──► neutral
      ▲                          ▲                               │
      └──(stable × 5)────────────┴───────(stable × 5)───────────┘
    ```
    """

    def __init__(self, config: Optional[AdaptiveCriticConfig] = None) -> None:
        """Initialize the adaptive critic.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or AdaptiveCriticConfig()
        self.state = CriticState(current_tone=self.config.default_tone)

    def update_state(
        self,
        anxiety_level: float = 0.0,
        frustration_count: int = 0,
        success: bool = True,
    ) -> CriticTone:
        """Update critic state based on cognitive state.

        Args:
            anxiety_level: Current anxiety level (0-1).
            frustration_count: Number of consecutive frustrations.
            success: Whether the last interaction was successful.

        Returns:
            The current tone after state update.
        """
        # Check for user override
        if self.state.user_override is not None:
            return self.state.user_override

        old_tone = self.state.current_tone
        new_tone = old_tone

        # Downgrade on high anxiety
        if anxiety_level > self.config.anxiety_threshold:
            if old_tone == CriticTone.HARSH:
                new_tone = self.config.anxiety_downgrade_to
            self.state.stable_iterations = 0

        # Downgrade on frustration accumulation
        if frustration_count >= self.config.frustration_count_threshold:
            new_tone = self.config.frustration_downgrade_to
            self.state.stable_iterations = 0

        # Track stability for potential escalation
        if success and anxiety_level < 0.5 and frustration_count == 0:
            self.state.stable_iterations += 1
        else:
            self.state.stable_iterations = 0

        # Escalate if stable enough
        if self.state.stable_iterations >= self.config.stable_iterations_for_escalation:
            current_index = self.config.escalation_path.index(new_tone)
            if current_index < len(self.config.escalation_path) - 1:
                new_tone = self.config.escalation_path[current_index + 1]
            self.state.stable_iterations = 0

        # Update state
        self.state.current_tone = new_tone
        self.state.last_anxiety = anxiety_level
        self.state.frustration_count = frustration_count

        return new_tone

    def set_user_override(self, tone: Optional[CriticTone]) -> None:
        """Allow user to manually set tone.

        Args:
            tone: The tone to force, or None to clear override.
        """
        if not self.config.allow_user_override:
            return
        self.state.user_override = tone

    def get_prefix(self) -> str:
        """Get the current tone prefix for prompts.

        Returns:
            Prefix string to prepend to critic prompts.
        """
        tone = self.state.user_override or self.state.current_tone
        return TONE_PREFIXES[tone]

    def create_prompt(self, content: str) -> str:
        """Create a critic prompt with the appropriate tone prefix.

        Args:
            content: The content to critique.

        Returns:
            Prompt string with tone prefix.
        """
        prefix = self.get_prefix()
        if prefix:
            return f"{prefix}\n\n{content}"
        return content

    def parse_reviews(self, llm_response: str) -> list[CriticReview]:
        """Parse LLM response into structured reviews.

        This is a simple implementation. In practice, you'd use
        structured output or more sophisticated parsing.

        Args:
            llm_response: Raw LLM response text.

        Returns:
            List of structured reviews.
        """
        reviews: list[CriticReview] = []

        # Simple heuristic: look for numbered items or bullet points
        lines = llm_response.strip().split("\n")
        current_review: Optional[dict[str, Any]] = None

        for line in lines:
            line = line.strip()
            if not line:
                if current_review:
                    reviews.append(CriticReview(**current_review))
                    current_review = None
                continue

            # Check for severity markers
            line_lower = line.lower()
            if any(s in line_lower for s in ["critical:", "critical -", "🔴"]):
                if current_review:
                    reviews.append(CriticReview(**current_review))
                current_review = {"severity": "critical", "issue": line}
            elif any(s in line_lower for s in ["major:", "major -", "🟠"]):
                if current_review:
                    reviews.append(CriticReview(**current_review))
                current_review = {"severity": "major", "issue": line}
            elif any(s in line_lower for s in ["minor:", "minor -", "🟡"]):
                if current_review:
                    reviews.append(CriticReview(**current_review))
                current_review = {"severity": "minor", "issue": line}
            elif any(s in line_lower for s in ["nit:", "nitpick:", "💬"]):
                if current_review:
                    reviews.append(CriticReview(**current_review))
                current_review = {"severity": "nitpick", "issue": line}
            elif current_review:
                # Append to current review
                if "suggestion" not in current_review and "suggest" in line_lower:
                    current_review["suggestion"] = line
                else:
                    current_review["issue"] += " " + line

        if current_review:
            reviews.append(CriticReview(**current_review))

        return reviews

    @property
    def current_tone(self) -> CriticTone:
        """Get the current effective tone."""
        return self.state.user_override or self.state.current_tone
