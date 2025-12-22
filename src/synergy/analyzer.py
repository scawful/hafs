"""Prompt analysis for Theory of Mind marker detection."""

from __future__ import annotations

from typing import Literal

from models.synergy import ToMMarker, ToMMarkerType
from synergy.markers import get_all_patterns


class PromptAnalyzer:
    """Analyzes prompts and responses for Theory of Mind markers."""

    def __init__(self) -> None:
        """Initialize the prompt analyzer with compiled patterns."""
        self._patterns = get_all_patterns()

    def analyze(self, text: str) -> list[ToMMarker]:
        """
        Analyze text for Theory of Mind markers.

        Args:
            text: The text to analyze (prompt or response).

        Returns:
            List of detected ToM markers.
        """
        markers: list[ToMMarker] = []

        for marker_type, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start, end = match.span()
                    context = self._get_context(text, start, end)

                    marker = ToMMarker(
                        type=marker_type,
                        confidence=1.0,  # Default confidence for regex matches
                        text_span=match.group(0),
                        context=context,
                    )
                    markers.append(marker)

        return markers

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """
        Extract surrounding context for a detected marker.

        Args:
            text: The full text.
            start: Start position of the match.
            end: End position of the match.
            window: Number of characters before and after to include.

        Returns:
            Context string with the match highlighted.
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        # Get the context
        before = text[context_start:start]
        match_text = text[start:end]
        after = text[end:context_end]

        # Add ellipsis if we're not at the boundaries
        if context_start > 0:
            before = "..." + before
        if context_end < len(text):
            after = after + "..."

        return f"{before}[{match_text}]{after}"

    def estimate_complexity(self, text: str) -> Literal["low", "medium", "high"]:
        """
        Estimate the complexity of the text based on ToM markers and length.

        Args:
            text: The text to analyze.

        Returns:
            Complexity level: "low", "medium", or "high".
        """
        markers = self.analyze(text)
        marker_count = len(markers)
        word_count = len(text.split())

        # Calculate complexity based on markers and length
        if marker_count >= 5 or word_count > 200:
            return "high"
        elif marker_count >= 2 or word_count > 75:
            return "medium"
        else:
            return "low"

    def get_improvement_suggestions(self, markers: list[ToMMarker]) -> list[str]:
        """
        Generate suggestions for improving Theory of Mind in communication.

        Args:
            markers: List of detected ToM markers.

        Returns:
            List of improvement suggestions.
        """
        suggestions: list[str] = []

        # Count marker types
        marker_types = {marker.type for marker in markers}

        # Check for missing important marker types
        if ToMMarkerType.CONFIRMATION_SEEKING not in marker_types:
            suggestions.append(
                "Consider asking for confirmation to ensure mutual understanding"
            )

        if ToMMarkerType.PERSPECTIVE_TAKING not in marker_types:
            suggestions.append(
                "Try acknowledging the other party's perspective or viewpoint"
            )

        if ToMMarkerType.KNOWLEDGE_GAP_DETECTION not in marker_types:
            suggestions.append(
                "Consider whether there might be knowledge gaps to address"
            )

        if ToMMarkerType.COMMUNICATION_REPAIR not in marker_types and len(markers) > 3:
            suggestions.append(
                "Use clarification techniques when explaining complex concepts"
            )

        # If very few markers overall
        if len(markers) < 2:
            suggestions.append(
                "Increase Theory of Mind awareness by using more perspective-taking language"
            )

        # If too many challenge/disagree without balance
        challenge_count = sum(
            1 for m in markers if m.type == ToMMarkerType.CHALLENGE_DISAGREE
        )
        if challenge_count > len(markers) / 2:
            suggestions.append(
                "Balance challenges with confirmation-seeking and clarification"
            )

        return suggestions
