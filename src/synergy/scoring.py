"""Synergy score calculation for human-AI interactions."""

from __future__ import annotations

from typing import Optional

from models.synergy import (
    ResponseQuality,
    SynergyScore,
    ToMMarker,
    ToMMarkerType,
    UserProfile,
)


class SynergyCalculator:
    """Calculates synergy scores for human-AI interactions."""

    # Component weights for overall synergy score
    WEIGHTS: dict[str, float] = {
        "tom_markers": 0.30,  # Theory of Mind marker presence
        "response_quality": 0.35,  # Response quality metrics
        "user_alignment": 0.20,  # Alignment with user preferences
        "context_utilization": 0.15,  # Context awareness and utilization
    }

    def __init__(self) -> None:
        """Initialize the synergy calculator."""
        # Validate weights sum to 1.0
        total = sum(self.WEIGHTS.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    def calculate(
        self,
        markers: list[ToMMarker],
        quality: ResponseQuality,
        profile: Optional[UserProfile] = None,
        context_used: bool = True,
    ) -> SynergyScore:
        """
        Calculate comprehensive synergy score for an interaction.

        Args:
            markers: List of detected Theory of Mind markers.
            quality: Response quality metrics.
            profile: Optional user profile for alignment calculation.
            context_used: Whether contextual information was utilized.

        Returns:
            SynergyScore with breakdown of components.
        """
        # Calculate individual component scores (0-100 scale)
        tom_score = self._calculate_tom_score(markers)
        quality_score = quality.overall * 100.0
        alignment_score = self._calculate_alignment(quality, profile)
        context_score = 100.0 if context_used else 50.0

        # Calculate weighted total
        total = (
            tom_score * self.WEIGHTS["tom_markers"]
            + quality_score * self.WEIGHTS["response_quality"]
            + alignment_score * self.WEIGHTS["user_alignment"]
            + context_score * self.WEIGHTS["context_utilization"]
        )

        # Create marker breakdown

        return SynergyScore(
            total=round(total, 2),
            breakdown={
                "tom_markers": round(tom_score, 2),
                "response_quality": round(quality_score, 2),
                "user_alignment": round(alignment_score, 2),
                "context_utilization": round(context_score, 2),
            },
        )

    def _calculate_tom_score(self, markers: list[ToMMarker]) -> float:
        """
        Calculate Theory of Mind score based on detected markers.

        Args:
            markers: List of detected ToM markers.

        Returns:
            ToM score on 0-100 scale.
        """
        if not markers:
            return 0.0

        # Count unique marker types
        unique_types = len(set(marker.type for marker in markers))

        # Base score on both quantity and diversity
        quantity_score = min(100.0, len(markers) * 15.0)  # Cap at ~7 markers
        diversity_score = (unique_types / len(ToMMarkerType)) * 100.0

        # Weight quantity slightly more than diversity
        tom_score = quantity_score * 0.6 + diversity_score * 0.4

        # Apply confidence weighting
        avg_confidence = sum(m.confidence for m in markers) / len(markers)
        tom_score *= avg_confidence

        return min(100.0, tom_score)

    def _calculate_alignment(
        self, quality: ResponseQuality, profile: Optional[UserProfile]
    ) -> float:
        """
        Calculate alignment with user preferences.

        Args:
            quality: Response quality metrics.
            profile: User profile (if available).

        Returns:
            Alignment score on 0-100 scale.
        """
        if profile is None:
            # Default neutral score without profile
            return 70.0

        # Start with response quality as baseline
        base_score = quality.overall * 100.0

        # Adjust based on interaction history
        if profile.is_new_user:
            # Give benefit of doubt to new users
            return min(100.0, base_score * 1.1)

        # For experienced users, factor in historical quality
        if profile.has_interaction_history:
            # Compare current quality to historical average
            historical_quality = profile.preferences.avg_response_quality * 100.0

            # Reward consistency or improvement
            if base_score >= historical_quality * 0.9:
                return min(100.0, base_score * 1.05)
            else:
                # Penalize significant drops in quality
                return base_score * 0.9

        return base_score

    def _create_marker_breakdown(
        self, markers: list[ToMMarker]
    ) -> dict[ToMMarkerType, int]:
        """
        Create a breakdown of marker types detected.

        Args:
            markers: List of detected ToM markers.

        Returns:
            Dictionary mapping marker types to counts.
        """
        breakdown: dict[ToMMarkerType, int] = {}

        for marker in markers:
            if marker.type not in breakdown:
                breakdown[marker.type] = 0
            breakdown[marker.type] += 1

        return breakdown

    def get_recommendations(self, score: SynergyScore) -> list[str]:
        """
        Generate recommendations based on synergy score.

        Args:
            score: The calculated synergy score.

        Returns:
            List of recommendations for improvement.
        """
        recommendations: list[str] = []

        # Overall score recommendations
        if score.is_poor:
            recommendations.append(
                "Focus on improving Theory of Mind awareness in interactions"
            )
        elif score.is_good and not score.is_excellent:
            recommendations.append(
                "Good synergy - consider diversifying ToM marker usage"
            )

        # Component-specific recommendations
        breakdown = score.breakdown

        if breakdown["tom_markers"] < 50.0:
            recommendations.append(
                "Increase Theory of Mind markers to show awareness of user perspective"
            )

        if breakdown["response_quality"] < 60.0:
            recommendations.append(
                "Improve response quality by focusing on relevance, clarity, and helpfulness"
            )

        if breakdown["user_alignment"] < 65.0:
            recommendations.append(
                "Better align responses with user preferences and communication style"
            )

        if breakdown["context_utilization"] < 80.0:
            recommendations.append(
                "Utilize more contextual information to provide relevant responses"
            )

        # If no specific issues found but score isn't excellent
        if not recommendations and not score.is_excellent:
            recommendations.append(
                "Continue maintaining high quality interactions across all components"
            )

        return recommendations
