"""Synthetic Augmentation Pipeline for Training Sample Diversity.

Augments high-quality samples with perspective/tone/complexity shifts to
increase embedding diversity while preserving technical content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

from agents.training.base import TrainingSample
from agents.training.json_utils import extract_json_from_response
from config.prompts import get_prompt, load_prompts

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for synthetic augmentation."""

    perspectives: list[str] = None
    tones: list[str] = None
    complexity_levels: list[str] = None
    context_shifts: list[str] = None
    variations_per_sample: int = 3
    min_quality_threshold: float = 0.6  # Only augment high-quality samples

    def __post_init__(self):
        if self.perspectives is None:
            self.perspectives = ["beginner", "expert", "reference", "tutorial"]
        if self.tones is None:
            self.tones = ["formal", "conversational", "terse", "verbose"]
        if self.complexity_levels is None:
            self.complexity_levels = ["simple", "intermediate", "advanced"]
        if self.context_shifts is None:
            self.context_shifts = ["dungeon", "overworld", "menu", "hardware"]


class SyntheticAugmenter:
    """Augments training samples with perspective shifts.

    Takes high-quality samples and generates variations with different:
    - Perspectives (beginner, expert, reference, tutorial)
    - Tones (formal, conversational, terse, verbose)
    - Complexity levels (simple, intermediate, advanced)
    - Context shifts (game feature, hardware focus, optimization)

    Preserves technical content while varying the phrasing and presentation.
    """

    def __init__(self, config: Optional[AugmentationConfig] = None, orchestrator=None):
        """Initialize augmenter.

        Args:
            config: Augmentation configuration
            orchestrator: Model orchestrator for generation
        """
        self.config = config or AugmentationConfig()
        self._orchestrator = orchestrator

    async def setup(self):
        """Initialize orchestrator if not provided."""
        if not self._orchestrator:
            from core.orchestrator_v2 import UnifiedOrchestrator

            self._orchestrator = UnifiedOrchestrator()

    async def augment_sample(
        self,
        original: TrainingSample,
        augmentation_type: str,
        augmentation_value: str,
    ) -> Optional[TrainingSample]:
        """Generate augmented version of a sample.

        Args:
            original: Original high-quality sample
            augmentation_type: Type of augmentation (perspective, tone, complexity, context)
            augmentation_value: Specific value (e.g., "beginner", "formal", etc.)

        Returns:
            Augmented TrainingSample or None if generation failed
        """
        if not self._orchestrator:
            await self.setup()

        # Build augmentation prompt based on type
        prompt = self._build_augmentation_prompt(
            original, augmentation_type, augmentation_value
        )

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            # Use coding tier for technical content, general for text
            tier = TaskTier.CODING if original.domain in ("asm", "oracle", "cpp") else TaskTier.GENERAL

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=tier,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning(f"Failed to extract JSON from augmentation response")
                return None

            # Create augmented sample
            augmented = TrainingSample(
                instruction=str(data.get("instruction", "")).strip(),
                input=str(data.get("input", "")).strip(),
                output=str(data.get("output", "")).strip(),
                domain=f"{original.domain}+augmented",  # Tag as augmented
                source=f"{original.source}_aug_{augmentation_type}_{augmentation_value}",
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=str(prompt),
                kg_entities=original.kg_entities.copy(),
            )

            return augmented

        except asyncio.TimeoutError:
            logger.warning(f"Timeout augmenting sample")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for augmentation: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to augment sample: {e}")
            return None

    def _build_augmentation_prompt(
        self,
        original: TrainingSample,
        augmentation_type: str,
        augmentation_value: str,
    ) -> str:
        """Build prompt for augmentation generation.

        Args:
            original: Original sample to augment
            augmentation_type: Type of augmentation
            augmentation_value: Specific augmentation value

        Returns:
            Prompt string for LLM
        """
        # Base context about original sample
        domain_context = {
            "asm": "65816 assembly for SNES/ALTTP",
            "oracle": "ROM hacking techniques and modifications",
            "cpp": "C++ code from YAZE emulator/ROM editor",
            "text": "ROM hacking documentation and tutorials",
        }

        domain_desc = domain_context.get(original.domain, original.domain)

        # Augmentation-specific instructions
        aug_instructions = self._get_augmentation_instructions(
            augmentation_type, augmentation_value
        )

        output_snippet = original.output[:1000]
        template = get_prompt("agents.training.augmentation.prompt", "")
        if template:
            return template.format(
                domain_desc=domain_desc,
                original_instruction=original.instruction,
                original_input=original.input,
                original_output=output_snippet,
                augmentation_instructions=aug_instructions,
                augmentation_value=augmentation_value,
            )

        return f"""You are rewriting a training sample to increase diversity while preserving technical content.

DOMAIN: {domain_desc}

ORIGINAL SAMPLE:
Instruction: {original.instruction}
Input: {original.input}
Output: {output_snippet}...

AUGMENTATION TASK:
{aug_instructions}

CRITICAL REQUIREMENTS:
1. PRESERVE ALL TECHNICAL CONTENT - same code, same addresses, same registers, same logic
2. CHANGE ONLY the phrasing, presentation style, and explanation approach
3. The augmented version should have DIFFERENT EMBEDDINGS but IDENTICAL technical information
4. Maintain coherence between instruction, input, and output

Generate a JSON object with the augmented version:

{{
  "instruction": "Rewritten instruction from {augmentation_value} perspective",
  "input": "Rewritten input context",
  "output": "Rewritten output with same technical content, different presentation"
}}

JSON FORMAT ONLY - no additional text.
"""

    def _get_augmentation_instructions(
        self, augmentation_type: str, augmentation_value: str
    ) -> str:
        """Get specific instructions for augmentation type/value.

        Args:
            augmentation_type: Type of augmentation
            augmentation_value: Specific value

        Returns:
            Instruction string for the augmentation
        """
        prompt_data = load_prompts()
        config_instructions = (
            prompt_data
            .get("agents", {})
            .get("training", {})
            .get("augmentation", {})
            .get("instructions", {})
            .get(augmentation_type, {})
        )
        if isinstance(config_instructions, dict):
            configured = config_instructions.get(augmentation_value)
            if isinstance(configured, str) and configured.strip():
                return configured

        if augmentation_type == "perspective":
            instructions = {
                "beginner": (
                    "Rewrite for a BEGINNER audience:\n"
                    "- Use simpler language and explain jargon\n"
                    "- Add step-by-step explanations\n"
                    "- Explain WHY things work, not just WHAT they do\n"
                    "- Add helpful context about SNES/ROM hacking basics"
                ),
                "expert": (
                    "Rewrite for an EXPERT audience:\n"
                    "- Use precise technical terminology\n"
                    "- Assume advanced knowledge of SNES hardware and ROM hacking\n"
                    "- Focus on advanced techniques and optimizations\n"
                    "- Reference production-quality patterns and best practices"
                ),
                "reference": (
                    "Rewrite as CONCISE API REFERENCE:\n"
                    "- Use terse, documentation-style language\n"
                    "- Focus on facts: what it does, parameters, return values\n"
                    "- Minimal explanations, maximum information density\n"
                    "- Use bullet points and structured format"
                ),
                "tutorial": (
                    "Rewrite as STEP-BY-STEP TUTORIAL:\n"
                    "- Break down into clear numbered steps\n"
                    "- Explain each step's purpose before showing code\n"
                    "- Add 'why' explanations for each decision\n"
                    "- Include learning objectives and takeaways"
                ),
            }
            return instructions.get(augmentation_value, f"Rewrite from {augmentation_value} perspective")

        elif augmentation_type == "tone":
            instructions = {
                "formal": (
                    "Rewrite in FORMAL academic/professional tone:\n"
                    "- Use proper grammar and complete sentences\n"
                    "- Avoid contractions and casual language\n"
                    "- Use passive voice where appropriate\n"
                    "- Professional technical writing style"
                ),
                "conversational": (
                    "Rewrite in CONVERSATIONAL friendly tone:\n"
                    "- Use contractions and casual language\n"
                    "- Direct address to reader ('you', 'we')\n"
                    "- Explain like teaching a friend\n"
                    "- Use analogies and relatable examples"
                ),
                "terse": (
                    "Rewrite in TERSE minimal style:\n"
                    "- Short sentences, minimal words\n"
                    "- Cut all unnecessary explanations\n"
                    "- Focus on essential information only\n"
                    "- Code-comment style annotations"
                ),
                "verbose": (
                    "Rewrite in VERBOSE detailed style:\n"
                    "- Elaborate explanations with full context\n"
                    "- Multiple perspectives on each concept\n"
                    "- Thorough coverage of edge cases\n"
                    "- Rich descriptive language"
                ),
            }
            return instructions.get(augmentation_value, f"Rewrite in {augmentation_value} tone")

        elif augmentation_type == "complexity":
            instructions = {
                "simple": (
                    "Rewrite at SIMPLE complexity level:\n"
                    "- Focus on core functionality only\n"
                    "- Avoid advanced optimizations\n"
                    "- Explain one concept at a time\n"
                    "- Use straightforward examples"
                ),
                "intermediate": (
                    "Rewrite at INTERMEDIATE complexity level:\n"
                    "- Balance detail with accessibility\n"
                    "- Introduce some optimizations and techniques\n"
                    "- Explain relationships between concepts\n"
                    "- Assume some prior knowledge"
                ),
                "advanced": (
                    "Rewrite at ADVANCED complexity level:\n"
                    "- Deep technical analysis\n"
                    "- Discuss performance implications\n"
                    "- Compare alternative approaches\n"
                    "- Assume expert-level background knowledge"
                ),
            }
            return instructions.get(augmentation_value, f"Rewrite at {augmentation_value} complexity")

        elif augmentation_type == "context":
            instructions = {
                "dungeon": (
                    "Rewrite with DUNGEON game context:\n"
                    "- Frame examples in terms of dungeon rooms, enemies, items\n"
                    "- Reference dungeon-specific mechanics\n"
                    "- Use dungeon scenarios for explanations"
                ),
                "overworld": (
                    "Rewrite with OVERWORLD game context:\n"
                    "- Frame examples in terms of map navigation, NPCs, exploration\n"
                    "- Reference overworld mechanics\n"
                    "- Use overworld scenarios for explanations"
                ),
                "menu": (
                    "Rewrite with MENU/UI game context:\n"
                    "- Frame examples in terms of menus, inventory, HUD\n"
                    "- Reference UI/menu mechanics\n"
                    "- Use menu interaction scenarios"
                ),
                "hardware": (
                    "Rewrite with HARDWARE focus:\n"
                    "- Emphasize PPU/CPU/DMA interactions\n"
                    "- Reference hardware timing and constraints\n"
                    "- Use hardware architecture perspective"
                ),
            }
            return instructions.get(augmentation_value, f"Rewrite with {augmentation_value} context")

        return f"Rewrite using {augmentation_type}: {augmentation_value}"

    async def augment_batch(
        self, samples: list[TrainingSample]
    ) -> list[TrainingSample]:
        """Augment a batch of samples.

        For each high-quality sample, generates multiple variations with
        different perspectives/tones/complexity.

        Args:
            samples: List of training samples to augment

        Returns:
            List of augmented samples (originals not included)
        """
        augmented_samples: list[TrainingSample] = []

        # Filter for high-quality samples
        high_quality = [
            s for s in samples if s.quality_score >= self.config.min_quality_threshold
        ]

        logger.info(
            f"Augmenting {len(high_quality)} high-quality samples "
            f"(filtered from {len(samples)} total)"
        )

        for sample in high_quality:
            # Select random augmentation types to apply
            num_variations = min(
                self.config.variations_per_sample,
                len(self.config.perspectives) + len(self.config.tones),
            )

            # Build pool of augmentation options
            augmentation_options = []

            # Add perspective options
            for perspective in self.config.perspectives:
                augmentation_options.append(("perspective", perspective))

            # Add tone options
            for tone in self.config.tones:
                augmentation_options.append(("tone", tone))

            # Randomly select augmentations to apply
            selected_augmentations = random.sample(
                augmentation_options, min(num_variations, len(augmentation_options))
            )

            # Generate augmented versions
            for aug_type, aug_value in selected_augmentations:
                augmented = await self.augment_sample(sample, aug_type, aug_value)
                if augmented:
                    augmented_samples.append(augmented)
                    logger.debug(
                        f"Augmented sample with {aug_type}={aug_value} "
                        f"(original quality: {sample.quality_score:.2f})"
                    )

        logger.info(f"Generated {len(augmented_samples)} augmented samples")
        return augmented_samples

    def get_augmentation_stats(self, samples: list[TrainingSample]) -> dict[str, Any]:
        """Get statistics about augmentation potential.

        Args:
            samples: List of samples to analyze

        Returns:
            Statistics dict
        """
        high_quality = [
            s for s in samples if s.quality_score >= self.config.min_quality_threshold
        ]

        potential_augmentations = len(high_quality) * self.config.variations_per_sample

        return {
            "total_samples": len(samples),
            "high_quality_samples": len(high_quality),
            "quality_threshold": self.config.min_quality_threshold,
            "variations_per_sample": self.config.variations_per_sample,
            "potential_augmentations": potential_augmentations,
            "expected_total": len(samples) + potential_augmentations,
        }
