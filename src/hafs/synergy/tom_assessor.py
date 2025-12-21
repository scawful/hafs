"""LLM-based Theory of Mind Assessor.

Uses the LMRA (LLM-as-Research-Assistant) approach from
"Quantifying Human-AI Synergy" research to assess ToM dimensions.

Scores 8 ToM dimensions on a 0-5 scale:
1. Perspective Taking
2. Goal Inference
3. Knowledge Gap Detection
4. Communication Repair
5. Confirmation Seeking
6. Mental State Attribution
7. Plan Coordination
8. Challenge/Disagree
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from hafs.models.irt import ToMAssessment
from hafs.models.synergy_config import AssessmentMode, ToMAssessmentConfig

logger = logging.getLogger(__name__)


# LMRA prompt template from research paper (Appendix C style)
LMRA_PROMPT_TEMPLATE = """You are a Theory of Mind assessor. Analyze the HUMAN's prompt to assess their collaborative reasoning abilities.

Score each ToM dimension from 0-5:
- 0: No evidence of this skill
- 1: Minimal/poor demonstration
- 2: Below average
- 3: Average/adequate
- 4: Above average/good
- 5: Excellent/expert level

## ToM Dimensions to Assess

1. **PERSPECTIVE_TAKING**: Does the human show awareness of the AI's capabilities, limitations, or viewpoint?
2. **GOAL_INFERENCE**: Does the human clearly communicate their underlying goals and intentions?
3. **KNOWLEDGE_GAP_DETECTION**: Does the human identify what they don't know and need help with?
4. **COMMUNICATION_REPAIR**: Does the human clarify or rephrase when something might be ambiguous?
5. **CONFIRMATION_SEEKING**: Does the human ask for verification or confirmation of understanding?
6. **MENTAL_STATE_ATTRIBUTION**: Does the human reason about the AI's "thinking" or reasoning process?
7. **PLAN_COORDINATION**: Does the human propose or discuss collaborative plans and steps?
8. **CHALLENGE_DISAGREE**: Does the human constructively challenge or disagree with the AI?

## Human's Prompt
{prompt}

## AI's Response (for context)
{response}

## Instructions
Respond with a JSON object containing:
- Scores for each dimension (0-5)
- Brief reasoning for each score

Format:
```json
{{
  "perspective_taking": <score>,
  "goal_inference": <score>,
  "knowledge_gap_detection": <score>,
  "communication_repair": <score>,
  "confirmation_seeking": <score>,
  "mental_state_attribution": <score>,
  "plan_coordination": <score>,
  "challenge_disagree": <score>,
  "reasoning": "<brief overall reasoning>"
}}
```

Assess objectively based on evidence in the text. Default to 2.5 if insufficient evidence."""


class ToMAssessor:
    """LLM-based Theory of Mind assessor.

    Uses probabilistic sampling and rate limiting to control costs
    while gathering ToM data for IRT estimation.
    """

    def __init__(
        self,
        config: Optional[ToMAssessmentConfig] = None,
        orchestrator: Optional["UnifiedOrchestrator"] = None,
    ):
        """Initialize the ToM assessor.

        Args:
            config: Assessment configuration (uses defaults if None)
            orchestrator: Orchestrator for LLM calls (lazy-initialized if None)
        """
        self.config = config or ToMAssessmentConfig()
        self._orchestrator = orchestrator
        self._initialized = False

        # Rate limiting state
        self._hourly_count = 0
        self._hourly_reset = datetime.now() + timedelta(hours=1)
        self._daily_cost = 0.0
        self._daily_reset = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        # Batch queue
        self._batch_queue: list[tuple[str, str, str]] = []  # (id, prompt, response)
        self._batch_results: dict[str, ToMAssessment] = {}

        # Stats
        self._total_assessments = 0
        self._total_cost = 0.0

    async def initialize(self) -> None:
        """Initialize the orchestrator if needed."""
        if self._initialized:
            return

        if self._orchestrator is None:
            from hafs.core.orchestrator_v2 import UnifiedOrchestrator

            self._orchestrator = UnifiedOrchestrator()
            await self._orchestrator.initialize()

        self._initialized = True

    def should_assess(self, prompt: str) -> bool:
        """Determine if this interaction should be assessed.

        Uses probabilistic sampling in SAMPLE mode and always
        returns True in FULL mode (subject to rate limits).
        """
        if not self.config.enabled:
            return False

        # Check minimum prompt length
        if len(prompt) < self.config.min_prompt_length:
            return False

        # Check rate limits
        self._update_rate_limits()

        if self.config.max_per_hour > 0 and self._hourly_count >= self.config.max_per_hour:
            logger.debug("ToM assessment skipped: hourly limit reached")
            return False

        if self._daily_cost >= self.config.max_daily_cost:
            logger.debug("ToM assessment skipped: daily cost limit reached")
            return False

        # Mode-based decision
        if self.config.mode == AssessmentMode.FULL:
            return True
        elif self.config.mode == AssessmentMode.SAMPLE:
            return random.random() < self.config.sample_rate
        else:  # BATCH
            return True  # Always queue in batch mode

    async def assess(
        self,
        prompt: str,
        response: str,
        assessor_model: str = "",
    ) -> Optional[ToMAssessment]:
        """Assess Theory of Mind dimensions for an interaction.

        Args:
            prompt: The human's prompt text
            response: The AI's response text
            assessor_model: Model used for assessment (auto-detected if empty)

        Returns:
            ToMAssessment with 8 dimension scores, or None if assessment fails
        """
        if not self.config.enabled:
            return None

        await self.initialize()

        # In batch mode, queue instead of immediate assessment
        if self.config.mode == AssessmentMode.BATCH:
            return await self._queue_for_batch(prompt, response)

        return await self._perform_assessment(prompt, response, assessor_model)

    async def _perform_assessment(
        self,
        prompt: str,
        response: str,
        assessor_model: str = "",
    ) -> Optional[ToMAssessment]:
        """Perform the actual LLM-based assessment."""
        from hafs.core.orchestrator_v2 import TaskTier

        start_time = time.time()

        # Truncate for context efficiency
        truncated_prompt = prompt[:2000] if len(prompt) > 2000 else prompt
        truncated_response = response[:2000] if len(response) > 2000 else response

        assessment_prompt = LMRA_PROMPT_TEMPLATE.format(
            prompt=truncated_prompt,
            response=truncated_response,
        )

        try:
            result = await self._orchestrator.generate(
                prompt=assessment_prompt,
                tier=TaskTier.REASONING,
                system_prompt="You are a precise Theory of Mind assessor. Respond only with valid JSON.",
                temperature=0.3,  # Low temp for consistency
                max_tokens=1000,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            scores = self._parse_assessment_response(result.content)
            if scores is None:
                logger.warning("Failed to parse ToM assessment response")
                return None

            # Update rate limits
            self._hourly_count += 1
            estimated_cost = result.tokens_used * 0.00001  # Rough estimate
            self._daily_cost += estimated_cost
            self._total_cost += estimated_cost
            self._total_assessments += 1

            return ToMAssessment(
                id=uuid4(),
                timestamp=datetime.now(),
                perspective_taking=scores.get("perspective_taking", 2.5),
                goal_inference=scores.get("goal_inference", 2.5),
                knowledge_gap_detection=scores.get("knowledge_gap_detection", 2.5),
                communication_repair=scores.get("communication_repair", 2.5),
                confirmation_seeking=scores.get("confirmation_seeking", 2.5),
                mental_state_attribution=scores.get("mental_state_attribution", 2.5),
                plan_coordination=scores.get("plan_coordination", 2.5),
                challenge_disagree=scores.get("challenge_disagree", 2.5),
                prompt_text=truncated_prompt[:500],
                response_text=truncated_response[:500],
                assessor_model=assessor_model or result.model,
                latency_ms=latency_ms,
                reasoning=scores.get("reasoning", ""),
            )

        except Exception as e:
            logger.error(f"ToM assessment failed: {e}")
            return None

    def _parse_assessment_response(self, content: str) -> Optional[dict]:
        """Parse the LLM's JSON response."""
        try:
            # Try to extract JSON from markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            scores = json.loads(content)

            # Validate and clamp scores
            dimensions = [
                "perspective_taking",
                "goal_inference",
                "knowledge_gap_detection",
                "communication_repair",
                "confirmation_seeking",
                "mental_state_attribution",
                "plan_coordination",
                "challenge_disagree",
            ]

            for dim in dimensions:
                if dim in scores:
                    scores[dim] = max(0.0, min(5.0, float(scores[dim])))
                else:
                    scores[dim] = 2.5  # Default

            return scores

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse ToM assessment: {e}")
            return None

    async def _queue_for_batch(
        self,
        prompt: str,
        response: str,
    ) -> Optional[ToMAssessment]:
        """Queue an interaction for batch assessment.

        Returns None immediately; results retrieved via get_batch_results().
        """
        interaction_id = str(uuid4())
        self._batch_queue.append((interaction_id, prompt, response))

        # Check if we should process the batch
        if len(self._batch_queue) >= self.config.batch_size:
            asyncio.create_task(self._process_batch())

        return None

    async def _process_batch(self) -> None:
        """Process queued interactions in batch."""
        if not self._batch_queue:
            return

        # Take up to batch_size items
        batch = self._batch_queue[: self.config.batch_size]
        self._batch_queue = self._batch_queue[self.config.batch_size :]

        logger.info(f"Processing ToM assessment batch of {len(batch)} items")

        for interaction_id, prompt, response in batch:
            assessment = await self._perform_assessment(prompt, response)
            if assessment:
                self._batch_results[interaction_id] = assessment

    def get_batch_result(self, interaction_id: str) -> Optional[ToMAssessment]:
        """Get a batch assessment result by ID."""
        return self._batch_results.pop(interaction_id, None)

    def _update_rate_limits(self) -> None:
        """Reset rate limit counters if periods have elapsed."""
        now = datetime.now()

        if now >= self._hourly_reset:
            self._hourly_count = 0
            self._hourly_reset = now + timedelta(hours=1)

        if now >= self._daily_reset:
            self._daily_cost = 0.0
            self._daily_reset = now.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)

    def get_stats(self) -> dict:
        """Get assessor statistics."""
        self._update_rate_limits()
        return {
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "total_assessments": self._total_assessments,
            "total_cost": self._total_cost,
            "hourly_count": self._hourly_count,
            "hourly_limit": self.config.max_per_hour,
            "hourly_remaining": max(0, self.config.max_per_hour - self._hourly_count),
            "daily_cost": self._daily_cost,
            "daily_limit": self.config.max_daily_cost,
            "daily_remaining": max(0.0, self.config.max_daily_cost - self._daily_cost),
            "batch_queue_size": len(self._batch_queue),
            "batch_results_pending": len(self._batch_results),
        }

    async def close(self) -> None:
        """Clean up resources."""
        # Process any remaining batch items
        if self._batch_queue:
            await self._process_batch()
