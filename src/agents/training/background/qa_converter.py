"""QA Converter - Converts user Q&A into training samples.

Takes answered questions and generates 3-5 training samples with variations:
- Preserve expert knowledge and technical content
- Vary phrasing, perspective, and presentation
- Generate samples in Alpaca instruction format
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from agents.training.background.question_curator import AnsweredQuestion
from agents.training.base import TrainingSample
from agents.training.json_utils import extract_json_from_response
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


class QAConverter:
    """Converts Q&A pairs into training samples.

    Generates 3-5 variations of each answered question to maximize
    the training value from user's expert knowledge.
    """

    def __init__(self, orchestrator=None):
        """Initialize QA converter.

        Args:
            orchestrator: Model orchestrator for generation
        """
        self._orchestrator = orchestrator

    async def setup(self):
        """Initialize orchestrator if not provided."""
        if not self._orchestrator:
            from core.orchestrator_v2 import UnifiedOrchestrator

            self._orchestrator = UnifiedOrchestrator()

    async def convert_qa_to_samples(
        self,
        answered: AnsweredQuestion,
        num_variations: int = 3,
    ) -> list[TrainingSample]:
        """Convert a Q&A pair into training samples.

        Args:
            answered: Answered question record
            num_variations: Number of sample variations to generate (3-5)

        Returns:
            List of TrainingSample objects
        """
        if not self._orchestrator:
            await self.setup()

        samples: list[TrainingSample] = []

        # Generate variations
        for i in range(num_variations):
            variation_type = self._get_variation_type(i)
            sample = await self._generate_sample(answered, variation_type)

            if sample:
                samples.append(sample)

        logger.info(
            f"Generated {len(samples)} samples from question {answered.question.question_id}"
        )
        return samples

    def _get_variation_type(self, index: int) -> str:
        """Get variation type for sample generation."""
        # Rotate through variation types
        types = ["direct", "tutorial", "reference", "conceptual", "advanced"]
        return types[index % len(types)]

    async def _generate_sample(
        self,
        answered: AnsweredQuestion,
        variation_type: str,
    ) -> Optional[TrainingSample]:
        """Generate a single training sample variation.

        Args:
            answered: Answered question
            variation_type: Type of variation (direct, tutorial, reference, etc.)

        Returns:
            TrainingSample or None if generation failed
        """
        prompt = self._build_conversion_prompt(answered, variation_type)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning(f"Failed to extract JSON from {variation_type} variation")
                return None

            # Create training sample
            return TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", "")).strip(),
                domain="qa_expert",  # Special domain for user Q&A
                source=f"qa_{answered.question.question_id}_{variation_type}",
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=str(prompt),
                kg_entities=[answered.question.pattern.pattern_type],
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating {variation_type} variation")
            return None
        except Exception as e:
            logger.error(f"Failed to generate {variation_type} variation: {e}")
            return None

    def _build_conversion_prompt(
        self,
        answered: AnsweredQuestion,
        variation_type: str,
    ) -> str:
        """Build prompt for Q&A to training sample conversion."""
        question = answered.question
        answer = answered.answer

        # Base template
        base_template = """You are converting expert Q&A into instruction-tuning training samples.

EXPERT QUESTION:
{question_text}

CODE CONTEXT:
File: {file_path}
Line: {line_number}
Code: {code_snippet}

EXPERT ANSWER:
{answer}

VARIATION TYPE: {variation_type}

Generate a training sample that preserves the expert knowledge while using the specified variation style.
"""

        # Variation-specific instructions
        variation_instructions = {
            "direct": """
DIRECT VARIATION: Keep the question and answer straightforward and clear.
- Instruction: Rephrase the original question naturally
- Input: Provide minimal technical context (file, line, code snippet if needed)
- Output: Present the expert's answer with slight rephrasing for clarity
""",
            "tutorial": """
TUTORIAL VARIATION: Transform into a step-by-step teaching format.
- Instruction: "Explain how to..." or "Walk through..."
- Input: Describe what the learner wants to accomplish
- Output: Break the expert's answer into numbered steps with explanations
""",
            "reference": """
REFERENCE VARIATION: Create a terse, technical reference format.
- Instruction: "What is..." or "Describe..."
- Input: Technical context only (addresses, registers, etc.)
- Output: Concise technical explanation (150-200 words max)
""",
            "conceptual": """
CONCEPTUAL VARIATION: Focus on the "why" and design decisions.
- Instruction: "Why does..." or "What's the rationale for..."
- Input: Describe the technical approach or pattern
- Output: Explain the reasoning, trade-offs, and design decisions from the answer
""",
            "advanced": """
ADVANCED VARIATION: Expand with advanced insights and alternatives.
- Instruction: "How could you..." or "What are alternative approaches to..."
- Input: Describe the current implementation
- Output: Expand the expert's answer with alternatives, optimizations, and edge cases
""",
        }

        full_template = (
            base_template + variation_instructions.get(variation_type, "") + """
JSON FORMAT:
{{
  "instruction": "The question/task (1-2 sentences)",
  "input": "Technical context or constraints (optional, can be empty string)",
  "output": "The answer/explanation (200-400 words, preserve expert's technical insights)"
}}

CRITICAL: Preserve ALL technical details from the expert's answer. Only vary the phrasing and presentation style.
"""
        )

        return full_template.format(
            question_text=question.question_text,
            file_path=question.pattern.file_path,
            line_number=question.pattern.line_number,
            code_snippet=question.pattern.code_snippet[:200],
            answer=answer,
            variation_type=variation_type,
        )

    async def convert_batch(
        self,
        answered_questions: list[AnsweredQuestion],
        num_variations: int = 3,
    ) -> list[TrainingSample]:
        """Convert multiple Q&A pairs into training samples.

        Args:
            answered_questions: List of answered questions
            num_variations: Variations per question

        Returns:
            List of all generated training samples
        """
        all_samples: list[TrainingSample] = []

        for answered in answered_questions:
            samples = await self.convert_qa_to_samples(answered, num_variations)
            all_samples.extend(samples)

        logger.info(
            f"Converted {len(answered_questions)} Q&A pairs into {len(all_samples)} training samples"
        )
        return all_samples

    async def save_samples(
        self,
        samples: list[TrainingSample],
        output_path: str | Path,
    ) -> None:
        """Save training samples to JSONL file.

        Args:
            samples: Training samples to save
            output_path: Path to output JSONL file
        """
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for sample in samples:
                f.write(sample.to_jsonl_entry() + "\n")

        logger.info(f"Saved {len(samples)} samples to {output_path}")
