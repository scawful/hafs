"""Response quality evaluation for agent interactions."""

from __future__ import annotations

from typing import Optional

from hafs.models.synergy import ResponseQuality, UserProfile
from hafs.synergy.analyzer import PromptAnalyzer


class ResponseEvaluator:
    """Evaluates the quality of agent responses."""

    def __init__(self, analyzer: PromptAnalyzer) -> None:
        """
        Initialize the response evaluator.

        Args:
            analyzer: PromptAnalyzer instance for detecting ToM markers.
        """
        self._analyzer = analyzer

    def evaluate(
        self,
        prompt: str,
        response: str,
        user_context: Optional[dict[str, any]] = None,
    ) -> ResponseQuality:
        """
        Evaluate the quality of a response to a prompt.

        Args:
            prompt: The user's prompt.
            response: The agent's response.
            user_context: Optional context about the user/conversation.

        Returns:
            ResponseQuality metrics.
        """
        relevance = self._calculate_relevance(prompt, response)
        clarity = self._calculate_clarity(response)
        helpfulness = self._calculate_helpfulness(response)

        # Analyze ToM awareness in the response
        markers = self._analyzer.analyze(response)
        tom_awareness = min(1.0, len(markers) / 5.0)  # Cap at 5 markers for full score

        return ResponseQuality(
            relevance=relevance,
            clarity=clarity,
            helpfulness=helpfulness,
            tom_awareness=tom_awareness,
        )

    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """
        Calculate how relevant the response is to the prompt.

        Args:
            prompt: The user's prompt.
            response: The agent's response.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        # Simple keyword overlap heuristic
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())

        # Remove common stop words for better signal
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "could",
            "may",
            "might",
            "can",
        }

        prompt_words = prompt_words - stop_words
        response_words = response_words - stop_words

        if not prompt_words:
            return 0.5  # Neutral score if no meaningful words

        # Calculate overlap
        overlap = len(prompt_words & response_words)
        max_score = len(prompt_words)

        # Base score on overlap, but ensure minimum score for any response
        base_score = overlap / max_score if max_score > 0 else 0.0

        # Boost score if response is substantial (not just echoing)
        if len(response_words) > len(prompt_words):
            base_score = min(1.0, base_score + 0.2)

        return max(0.3, min(1.0, base_score))  # Minimum 0.3, maximum 1.0

    def _calculate_clarity(self, response: str) -> float:
        """
        Calculate the clarity of the response.

        Args:
            response: The agent's response.

        Returns:
            Clarity score between 0.0 and 1.0.
        """
        # Heuristics for clarity
        sentences = response.split(".")
        sentence_count = len([s for s in sentences if s.strip()])

        if sentence_count == 0:
            return 0.0

        words = response.split()
        word_count = len(words)

        # Average sentence length (optimal: 15-20 words)
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

        # Penalize very short or very long sentences
        length_score = 1.0
        if avg_sentence_length < 10:
            length_score = 0.7
        elif avg_sentence_length > 30:
            length_score = 0.7
        elif 15 <= avg_sentence_length <= 20:
            length_score = 1.0
        else:
            length_score = 0.85

        # Check for structure indicators (lists, paragraphs, etc.)
        has_structure = any(
            indicator in response for indicator in ["\n", "- ", "1.", "2.", """]
        )
        structure_score = 1.0 if has_structure and word_count > 50 else 0.9

        # Check for clarification markers
        markers = self._analyzer.analyze(response)
        has_clarification = any(
            m.type.value in ["communication_repair", "confirmation_seeking"]
            for m in markers
        )
        clarification_score = 1.0 if has_clarification else 0.9

        # Combine scores
        clarity = (length_score * 0.4 + structure_score * 0.3 + clarification_score * 0.3)

        return max(0.0, min(1.0, clarity))

    def _calculate_helpfulness(self, response: str) -> float:
        """
        Calculate the helpfulness of the response.

        Args:
            response: The agent's response.

        Returns:
            Helpfulness score between 0.0 and 1.0.
        """
        # Indicators of helpfulness
        helpful_indicators = [
            "here's",
            "you can",
            "try",
            "consider",
            "suggest",
            "recommend",
            "example",
            "step",
            "how to",
            "would help",
            "should",
            "could",
            "might want",
        ]

        response_lower = response.lower()
        indicator_count = sum(
            1 for indicator in helpful_indicators if indicator in response_lower
        )

        # Base score on presence of helpful language
        base_score = min(1.0, indicator_count / 5.0)

        # Boost for substantial content
        word_count = len(response.split())
        if word_count > 50:
            base_score = min(1.0, base_score + 0.2)

        # Check for actionable content (questions, suggestions)
        has_questions = "?" in response
        markers = self._analyzer.analyze(response)
        has_coordination = any(
            m.type.value in ["plan_coordination", "goal_inference"] for m in markers
        )

        if has_questions or has_coordination:
            base_score = min(1.0, base_score + 0.15)

        # Minimum score for any response
        return max(0.3, base_score)
