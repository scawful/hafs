"""Analysis modes and base analyzer.

Implements the analysis mode enumeration and base analyzer interface
per PROTOCOL_SPEC.md Section 5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisMode(str, Enum):
    """Analysis modes available per PROTOCOL_SPEC.md Section 5.1."""

    NONE = "none"              # Standard operation, no analysis overlay
    EVAL = "eval"              # Prompt/response quality evaluation
    TOM = "tom"                # Theory of Mind marker detection
    METRICS = "metrics"        # Coordination efficiency and scaling metrics
    CRITIC = "critic"          # Adaptive harsh criticism
    EMOTIONAL = "emotional"    # Emotional valence and cognitive load
    SYNERGY = "synergy"        # Human-AI collaboration quality (future)
    REVIEW = "review"          # Google-style code review
    DOCUMENTATION = "documentation"  # Comment placement analysis


class AnalysisResult(BaseModel):
    """Result from an analysis mode execution."""

    mode: AnalysisMode
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None

    # Mode-specific results
    data: dict[str, Any] = Field(default_factory=dict)

    # Metadata
    duration_ms: Optional[int] = None
    model_used: Optional[str] = None
    confidence: float = 1.0

    # Extension point
    extensions: dict[str, Any] = Field(default_factory=dict)


class EvalMetrics(BaseModel):
    """Evaluation mode metrics per PROTOCOL_SPEC.md Section 5.2.1."""

    prompt_quality: dict[str, float] = Field(default_factory=lambda: {
        "clarity": 0.0,
        "specificity": 0.0,
        "context_sufficiency": 0.0,
        "ambiguity_count": 0,
    })

    response_quality: dict[str, float] = Field(default_factory=lambda: {
        "correctness": 0.0,
        "helpfulness": 0.0,
        "completeness": 0.0,
        "reasoning_quality": 0.0,
    })

    information_gain: float = 0.0


class ToMMarker(BaseModel):
    """Theory of Mind marker per PROTOCOL_SPEC.md Section 5.2.2."""

    type: str
    text: str
    position: int = 0
    confidence: float = 1.0


class ToMAnalysis(BaseModel):
    """ToM analysis result per PROTOCOL_SPEC.md Section 5.2.2."""

    markers_detected: list[ToMMarker] = Field(default_factory=list)
    trait_tom_score: float = 0.0
    dynamic_tom_deviation: float = 0.0
    predicted_response_quality: float = 0.0


class ScalingMetrics(BaseModel):
    """Scaling metrics per PROTOCOL_SPEC.md Section 5.2.3."""

    coordination_overhead: float = 0.0
    message_density: float = 0.0
    redundancy_rate: float = 0.0
    coordination_efficiency: float = 0.0
    error_amplification: float = 0.0

    baseline_accuracy: float = 0.0
    task_tool_count: int = 0
    task_decomposability: float = 0.0

    should_use_multi_agent: bool = False
    recommended_architecture: str = "single"
    max_effective_agents: int = 1

    error_rates: dict[str, float] = Field(default_factory=dict)


class EmotionalAnalysis(BaseModel):
    """Emotional analysis per PROTOCOL_SPEC.md Section 5.2.5."""

    mood: dict[str, Any] = Field(default_factory=lambda: {
        "current": "neutral",
        "intensity": 0.0,
        "valence": 0.0,
    })

    anxiety: dict[str, Any] = Field(default_factory=lambda: {
        "level": 0.0,
        "sources": [],
        "mitigation_suggestions": [],
    })

    fears: list[dict[str, Any]] = Field(default_factory=list)
    satisfactions: list[dict[str, Any]] = Field(default_factory=list)
    frustrations: list[dict[str, Any]] = Field(default_factory=list)

    high_anxiety_threshold: float = 0.7
    frustration_spinning_threshold: int = 3


class BaseAnalyzer(ABC):
    """Abstract base class for analysis mode implementations.

    All analyzers must implement the analyze() method that takes
    input data and returns an AnalysisResult.
    """

    @property
    @abstractmethod
    def mode(self) -> AnalysisMode:
        """The analysis mode this analyzer implements."""
        pass

    @property
    def name(self) -> str:
        """Human-readable name for this analyzer."""
        return self.mode.value

    @abstractmethod
    async def analyze(
        self,
        input_data: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Run analysis on input data.

        Args:
            input_data: Mode-specific input data.
            context: Optional context (cognitive state, history, etc.).

        Returns:
            AnalysisResult with mode-specific data.
        """
        pass

    async def validate_input(
        self,
        input_data: dict[str, Any],
    ) -> bool:
        """Validate input data before analysis.

        Override this method to add input validation.

        Args:
            input_data: Input data to validate.

        Returns:
            True if input is valid.
        """
        return True


class EvalAnalyzer(BaseAnalyzer):
    """Evaluation mode analyzer."""

    @property
    def mode(self) -> AnalysisMode:
        return AnalysisMode.EVAL

    async def analyze(
        self,
        input_data: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Analyze prompt and response quality.

        Args:
            input_data: Should contain 'prompt' and optionally 'response'.
            context: Optional cognitive context.

        Returns:
            AnalysisResult with EvalMetrics.
        """
        import time
        start_time = time.time()

        prompt = input_data.get("prompt", "")
        response = input_data.get("response", "")

        # Simple heuristic evaluation (LLM-based would be better)
        metrics = EvalMetrics()

        # Prompt quality heuristics
        if prompt:
            metrics.prompt_quality["clarity"] = min(1.0, len(prompt) / 500)
            metrics.prompt_quality["specificity"] = 0.7 if "?" in prompt or any(
                w in prompt.lower() for w in ["how", "what", "why", "when"]
            ) else 0.4
            metrics.prompt_quality["context_sufficiency"] = 0.5  # Default

        # Response quality heuristics
        if response:
            metrics.response_quality["completeness"] = min(1.0, len(response) / 200)
            metrics.response_quality["helpfulness"] = 0.6  # Default

        duration_ms = int((time.time() - start_time) * 1000)

        return AnalysisResult(
            mode=self.mode,
            success=True,
            data=metrics.model_dump(),
            duration_ms=duration_ms,
            confidence=0.5,  # Low confidence for heuristic analysis
        )


class ToMAnalyzer(BaseAnalyzer):
    """Theory of Mind marker detection analyzer."""

    # ToM marker patterns (simplified)
    TOM_PATTERNS = {
        "perspective_taking": ["from your perspective", "if i were you", "in your position"],
        "goal_inference": ["your goal is", "you're trying to", "you want to"],
        "knowledge_gap_detection": ["you might not know", "i should mention", "for context"],
        "communication_repair": ["let me clarify", "in other words", "to be clear"],
        "confirmation_seeking": ["is that correct", "does that make sense", "are you following"],
        "plan_coordination": ["let's work together", "can you handle", "i'll handle"],
        "challenge_disagree": ["are you sure", "i disagree", "that's not quite right"],
    }

    @property
    def mode(self) -> AnalysisMode:
        return AnalysisMode.TOM

    async def analyze(
        self,
        input_data: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Detect ToM markers in text.

        Args:
            input_data: Should contain 'text' to analyze.
            context: Optional cognitive context.

        Returns:
            AnalysisResult with ToMAnalysis.
        """
        import time
        start_time = time.time()

        text = input_data.get("text", "").lower()
        markers_detected: list[ToMMarker] = []

        for marker_type, patterns in self.TOM_PATTERNS.items():
            for pattern in patterns:
                pos = text.find(pattern)
                if pos >= 0:
                    markers_detected.append(ToMMarker(
                        type=marker_type,
                        text=pattern,
                        position=pos,
                        confidence=0.8,
                    ))

        # Calculate trait ToM score (0-1 based on marker density)
        trait_tom_score = min(1.0, len(markers_detected) * 0.15)

        analysis = ToMAnalysis(
            markers_detected=markers_detected,
            trait_tom_score=trait_tom_score,
            dynamic_tom_deviation=0.0,  # Would need session history
            predicted_response_quality=0.5 + (trait_tom_score * 0.27),  # β=0.27 from paper
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return AnalysisResult(
            mode=self.mode,
            success=True,
            data=analysis.model_dump(),
            duration_ms=duration_ms,
            confidence=0.7,
        )


class MetricsAnalyzer(BaseAnalyzer):
    """Coordination metrics analyzer."""

    @property
    def mode(self) -> AnalysisMode:
        return AnalysisMode.METRICS

    async def analyze(
        self,
        input_data: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Calculate coordination metrics.

        Args:
            input_data: Should contain agent/task information.
            context: Optional cognitive context.

        Returns:
            AnalysisResult with ScalingMetrics.
        """
        import time
        start_time = time.time()

        agent_count = input_data.get("agent_count", 1)
        tool_count = input_data.get("tool_count", 0)
        baseline_accuracy = input_data.get("baseline_accuracy", 0.5)
        task_decomposability = input_data.get("task_decomposability", 0.5)

        metrics = ScalingMetrics(
            baseline_accuracy=baseline_accuracy,
            task_tool_count=tool_count,
            task_decomposability=task_decomposability,
        )

        # Architecture recommendation per PROTOCOL_SPEC.md Section 5.2.3
        if baseline_accuracy > 0.45:
            metrics.recommended_architecture = "single"
            metrics.should_use_multi_agent = False
        elif task_decomposability > 0.3:
            metrics.recommended_architecture = "centralized"
            metrics.should_use_multi_agent = True
        else:
            metrics.recommended_architecture = "single"
            metrics.should_use_multi_agent = baseline_accuracy < 0.45 and tool_count <= 4

        # Max effective agents (usually 3-4 per paper)
        metrics.max_effective_agents = 4 if metrics.should_use_multi_agent else 1

        # Coordination overhead (simplified estimation)
        if agent_count > 1:
            metrics.coordination_overhead = (agent_count - 1) * 0.15  # 15% per agent

        duration_ms = int((time.time() - start_time) * 1000)

        return AnalysisResult(
            mode=self.mode,
            success=True,
            data=metrics.model_dump(),
            duration_ms=duration_ms,
            confidence=0.8,
        )


class EmotionalAnalyzer(BaseAnalyzer):
    """Emotional valence analyzer."""

    @property
    def mode(self) -> AnalysisMode:
        return AnalysisMode.EMOTIONAL

    async def analyze(
        self,
        input_data: dict[str, Any],
        context: Optional[dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Analyze emotional state.

        Args:
            input_data: Should contain cognitive state data.
            context: Optional context with history.

        Returns:
            AnalysisResult with EmotionalAnalysis.
        """
        import time
        start_time = time.time()

        # Extract from cognitive state if available
        cognitive_state = context.get("cognitive_state", {}) if context else {}
        emotions = cognitive_state.get("emotions", {})
        try:
            from core.protocol.emotions_compat import normalize_emotions

            emotions = normalize_emotions(emotions) if emotions else {}
        except Exception:
            emotions = emotions or {}

        analysis = EmotionalAnalysis(
            mood={
                "current": emotions.get("session", {}).get("mood", "neutral"),
                "intensity": 0.5,
                "valence": 0.0,  # Would need sentiment analysis
            },
            anxiety={
                "level": emotions.get("session", {}).get("anxietyLevel", 0.0)
                if isinstance(emotions.get("session"), dict)
                else emotions.get("anxiety", 0.0),
                "sources": [],
                "mitigation_suggestions": [],
            },
            fears=list(emotions.get("fears", {}).values() if isinstance(emotions.get("fears"), dict) else emotions.get("fears", [])),
            satisfactions=list(emotions.get("satisfactions", {}).values() if isinstance(emotions.get("satisfactions"), dict) else emotions.get("satisfactions", [])),
            frustrations=list(emotions.get("frustrations", {}).values() if isinstance(emotions.get("frustrations"), dict) else emotions.get("frustrations", [])),
        )

        # Check for spinning
        unresolved_frustrations = [
            f for f in analysis.frustrations
            if not f.get("resolved", False)
        ]
        if len(unresolved_frustrations) >= analysis.frustration_spinning_threshold:
            analysis.anxiety["sources"].append("consecutive_frustrations")
            analysis.anxiety["level"] = max(analysis.anxiety["level"], 0.7)

        duration_ms = int((time.time() - start_time) * 1000)

        return AnalysisResult(
            mode=self.mode,
            success=True,
            data=analysis.model_dump(),
            duration_ms=duration_ms,
            confidence=0.7,
        )


# Analyzer registry
ANALYZERS: dict[AnalysisMode, type[BaseAnalyzer]] = {
    AnalysisMode.EVAL: EvalAnalyzer,
    AnalysisMode.TOM: ToMAnalyzer,
    AnalysisMode.METRICS: MetricsAnalyzer,
    AnalysisMode.EMOTIONAL: EmotionalAnalyzer,
}


def get_analyzer(mode: AnalysisMode) -> Optional[BaseAnalyzer]:
    """Get an analyzer instance for a mode.

    Args:
        mode: The analysis mode.

    Returns:
        Analyzer instance, or None if not implemented.
    """
    analyzer_class = ANALYZERS.get(mode)
    if analyzer_class:
        return analyzer_class()
    return None
