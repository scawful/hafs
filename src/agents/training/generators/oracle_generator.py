"""Oracle Data Generator for ROM hack modification training data.

Generates instruction-tuning data from Oracle-of-Secrets ROM hack,
focusing on vanilla vs hack differences and ROM hacking techniques.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from agents.training.json_utils import extract_json_from_response
from agents.training.generators.prompt_templates import PromptTemplateRotator
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class OracleSourceItem(SourceItem):
    """Source item for Oracle ROM hack routines."""

    address: str = ""
    file_path: str = ""
    line_number: int = 0
    description: str = ""
    category: str = ""
    code_snippet: str = ""
    calls: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    is_hook: bool = False
    hooks_vanilla: Optional[str] = None

    @property
    def item_id(self) -> str:
        return f"oracle:{self.name}:{self.address}"


class OracleDataGenerator(DataGenerator):
    """Generate instruction-tuning data from Oracle ROM hack.

    Extracts from Oracle-of-Secrets ROM hack (1,269 routines),
    focusing on:
    - ROM hack modifications vs vanilla code
    - How hooks intercept vanilla routines
    - Custom features and new content
    - ROM hacking techniques and patterns

    Teacher LLM generates instructions explaining:
    - What this ROM hack modification does
    - How it differs from vanilla ALTTP
    - The ROM hacking technique used
    """

    ORACLE_KB_PATH = (
        Path.home() / ".context" / "knowledge" / "oracle-of-secrets" / "routines.json"
    )

    def __init__(self, use_enhanced_prompts: bool = False, use_template_variation: bool = True):
        """Initialize Oracle data generator.

        Args:
            use_enhanced_prompts: Whether to use enhanced v2 prompts with
                reference examples and explicit quality requirements
            use_template_variation: Whether to use prompt template rotation for diversity
        """
        super().__init__(
            name="OracleDataGenerator",
            domain="oracle",
            teacher_tier="coding",
        )
        self._routines: list[dict] = []
        self._orchestrator = None
        self.use_enhanced_prompts = use_enhanced_prompts
        self.use_template_variation = use_template_variation

        # Initialize template rotator for diversity
        if self.use_template_variation:
            self.template_rotator = PromptTemplateRotator(domain="oracle")

    async def setup(self):
        """Initialize resources and load Oracle routines."""
        await super().setup()

        # Load Oracle routines.json
        if not self.ORACLE_KB_PATH.exists():
            logger.error(f"Oracle KB not found: {self.ORACLE_KB_PATH}")
            raise FileNotFoundError(f"Missing Oracle KB: {self.ORACLE_KB_PATH}")

        with open(self.ORACLE_KB_PATH) as f:
            self._routines = json.load(f)

        logger.info(f"Loaded {len(self._routines)} Oracle ROM hack routines")

        # Initialize orchestrator
        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[OracleSourceItem]:
        """Extract routines from Oracle ROM hack KB.

        Filters for high-quality samples:
        - Skip routines with empty code snippets
        - Prefer routines with descriptions
        - Prefer hooks (show vanilla vs hack differences)
        - Prefer routines with call graphs (show integration)
        """
        if not self._routines:
            await self.setup()

        items: list[OracleSourceItem] = []

        for routine in self._routines:
            name = routine.get("name", "")
            code_snippet = routine.get("code_snippet", "")

            # Skip if no code
            if not code_snippet or len(code_snippet) < 10:
                continue

            # Build content for embedding/display
            content_parts = [f"Routine: {name}"]

            if routine.get("description"):
                content_parts.append(f"Description: {routine['description']}")

            if routine.get("category"):
                content_parts.append(f"Category: {routine['category']}")

            if routine.get("is_hook"):
                content_parts.append("Type: Hook (modifies vanilla code)")
                if routine.get("hooks_vanilla"):
                    content_parts.append(
                        f"Hooks: {routine['hooks_vanilla']}"
                    )

            if routine.get("address"):
                content_parts.append(f"Address: {routine['address']}")

            content_parts.append(f"Code:\n{code_snippet[:500]}")  # First 500 chars

            content = "\n".join(content_parts)

            items.append(
                OracleSourceItem(
                    name=name,
                    content=content,
                    source="oracle",
                    address=routine.get("address", ""),
                    file_path=routine.get("file_path", ""),
                    line_number=routine.get("line_number", 0),
                    description=routine.get("description", ""),
                    category=routine.get("category", ""),
                    code_snippet=code_snippet,
                    calls=routine.get("calls", []),
                    called_by=routine.get("called_by", []),
                    is_hook=routine.get("is_hook", False),
                    hooks_vanilla=routine.get("hooks_vanilla"),
                )
            )

        logger.info(f"Extracted {len(items)} Oracle ROM hack routines")
        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for Oracle ROM hack routine."""
        if not isinstance(item, OracleSourceItem):
            raise TypeError(f"Expected OracleSourceItem, got {type(item)}")

        # Use enhanced prompts if enabled
        if self.use_enhanced_prompts:
            from agents.training.generators.enhanced_prompts import get_enhanced_oracle_prompt

            return get_enhanced_oracle_prompt(
                routine_name=item.name,
                code_snippet=item.code_snippet,
                address=item.address,
                file_path=item.file_path,
                description=item.description,
                category=item.category,
                is_hook=item.is_hook,
                hooks_vanilla=item.hooks_vanilla,
                calls=item.calls,
                called_by=item.called_by,
            )

        # Build context sections
        context_parts = []

        # File context
        if item.file_path:
            context_parts.append(f"Source file: {item.file_path} (line {item.line_number})")

        # Category
        if item.category:
            context_parts.append(f"Category: {item.category}")

        # Hook context (important for understanding modifications)
        if item.is_hook:
            context_parts.append("Type: Hook (modifies vanilla ALTTP code)")
            if item.hooks_vanilla:
                context_parts.append(f"Hooks vanilla routine: {item.hooks_vanilla}")

        # Description
        if item.description:
            context_parts.append(f"Description: {item.description}")

        # Call graph
        if item.calls:
            context_parts.append(f"Calls: {', '.join(item.calls[:5])}")
        if item.called_by:
            context_parts.append(f"Called by: {', '.join(item.called_by[:5])}")

        # Address
        if item.address:
            context_parts.append(f"ROM address: {item.address}")

        context = "\n".join(context_parts) if context_parts else "No additional context"

        # Code snippet (truncate if too long)
        code = item.code_snippet
        if len(code) > 1000:
            code = code[:1000] + "\n... (truncated)"

        hook_emphasis = ""
        if item.is_hook:
            hook_emphasis = "**IMPORTANT:** This is a HOOK that modifies vanilla ALTTP code."

        # Use template variation for instruction diversity
        instruction_prefix = "Generate high-quality training data for this Oracle-of-Secrets ROM hack routine"
        if self.use_template_variation and hasattr(self, 'template_rotator'):
            # Get next template and create varied instruction request
            template_obj = self.template_rotator.get_next_template()
            # Use template as instruction variation hint
            instruction_prefix = f"Generate training data using this perspective: '{template_obj.template}'"

        template = get_prompt("agents.training.generators.oracle_generator.prompt", "")
        if not template:
            template = (
                "You are an expert ROM hacker specializing in SNES and ALTTP modifications. "
                f"{instruction_prefix}.\n\n"
                "ROUTINE: {name}\n"
                "CATEGORY: {category}\n"
                "HOOK STATUS: {hook_status}\n"
                "{hooks_vanilla_line}\n\n"
                "CONTEXT:\n{context}\n\n"
                "CODE:\n```asm\n{code}\n```\n"
                "{hook_emphasis}\n\n"
                "Generate a JSON object with:\n\n"
                "1. \"instruction\": A clear question about this ROM hack technique. Make it pedagogical and varied:\n"
                "   - Ask about implementation of specific ROM hack features\n"
                "   - Request explanation of hooking/patching techniques\n"
                "   - Ask about vanilla vs hack behavior differences\n"
                "   - Request guidance on adding similar custom content\n\n"
                "2. \"input\": Technical context (2-3 sentences):\n"
                "   - Vanilla behavior being modified (if hook)\n"
                "   - ROM bank and address information\n"
                "   - Related routines in call graph\n"
                "   - Technical constraints or requirements\n\n"
                "3. \"output\": Comprehensive ROM hacking tutorial (200-350 words) covering:\n\n"
                "   **Functionality Overview:**\n"
                "   - What this routine accomplishes in the game\n"
                "   - Player-visible changes or new features\n"
                "   - Integration with existing game systems\n\n"
                "   **ROM Hacking Technique (REQUIRED):**\n"
                "   - If HOOK: Which vanilla routine at what address ($XX:XXXX)\n"
                "   - If HOOK: Original behavior vs modified behavior\n"
                "   - Code injection method (org directive, pushpc/pullpc, JSL redirect)\n"
                "   - Bank allocation strategy (expanded banks $20-$FF)\n"
                "   - Why this approach was chosen\n\n"
                "   **Implementation Details:**\n"
                "   - Line-by-line code analysis with assembly explanations\n"
                "   - Hardware register usage (PPU: $21XX, CPU: $42XX)\n"
                "   - RAM variable allocation ($7E:XXXX, $7F:XXXX)\n"
                "   - Timing considerations and NMI/IRQ handling\n\n"
                "   **Integration & Testing:**\n"
                "   - How it integrates with other hack components\n"
                "   - Common pitfalls when implementing similar features\n"
                "   - Testing approach and potential bugs\n\n"
                "QUALITY REQUIREMENTS:\n"
                "- Use precise 65816 assembly terminology and syntax\n"
                "- Specify exact addresses for ROM ($XX:XXXX), RAM ($7E:XXXX), and registers ($21XX)\n"
                "- Explain vanilla behavior BEFORE explaining modifications\n"
                "- Include concrete examples and code snippets\n"
                "- Teach the ROM hacking technique, not just describe it\n"
                "- Maintain coherent narrative flow between sections\n\n"
                "EXAMPLE OUTPUT (for a hook):\n"
                "```\n"
                "The OracleCustomSpriteLoader routine is a hook that replaces vanilla ALTTP's sprite loading logic at $0D:B4E0.\n\n"
                "**Vanilla Behavior:** The original game loads sprite graphics from banks $09-$0B using a simple index lookup.\n\n"
                "**Modified Behavior:** Oracle redirects this to bank $32 (custom sprite bank) using:\n"
                "```asm\n"
                "org $0DB4E0\n"
                "    JSL OracleCustomSpriteLoader  ; Jump to bank $32\n"
                "    NOP #3                        ; Fill remaining bytes\n"
                "```\n\n"
                "The custom routine (shown above) expands sprite variety from 128 to 256 by using both the sprite ID and room number...\n\n"
                "[Continue with detailed line-by-line explanation]\n"
                "```\n\n"
                "JSON FORMAT:\n"
                "{{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}}\n"
            )

        hook_status = "HOOK (modifies vanilla)" if item.is_hook else "NEW CODE (custom addition)"
        hooks_vanilla_line = f"HOOKS VANILLA: {item.hooks_vanilla}" if item.hooks_vanilla else ""

        return template.format(
            name=item.name,
            category=item.category,
            hook_status=hook_status,
            hooks_vanilla_line=hooks_vanilla_line,
            context=context,
            code=code,
            hook_emphasis=hook_emphasis,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from Oracle routine."""
        if not isinstance(item, OracleSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            response, model_name = await asyncio.wait_for(
                self.generate_with_rotation(prompt, tier="coding"),
                timeout=120.0,  # Increased for GPU/slower models
            )
            if not response:
                return None

            # Extract JSON from response using robust parser
            data = extract_json_from_response(response)
            if not data:
                logger.warning(f"Failed to extract JSON from response for {item.name}")
                return None

            # Ensure all fields are strings (defensive conversion)
            instruction = str(data.get("instruction", ""))
            input_text = str(data.get("input", ""))
            output = str(data.get("output", item.content if isinstance(item.content, str) else ""))

            # Collect KG entities (ensure all strings)
            kg_entities = [str(item.name)]
            if item.hooks_vanilla:
                kg_entities.append(str(item.hooks_vanilla))
            kg_entities.extend([str(c) for c in item.calls[:3]])  # Top 3 calls

            return TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                domain="oracle",
                source=item.source,
                teacher_model=model_name,
                teacher_prompt=prompt,
                kg_entities=kg_entities,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.name}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {item.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate for {item.name}: {e}")
            return None


if __name__ == "__main__":

    async def main():
        gen = OracleDataGenerator()
        await gen.setup()

        # Quick test
        items = await gen.extract_source_items()
        print(f"Found {len(items)} Oracle ROM hack routines")

        if items:
            # Test first 5
            result = await gen.run_generation(
                limit=5,
                output_path=Path("test_oracle_train.jsonl"),
            )
            print(f"Generated {result.processed} samples")

    import asyncio

    asyncio.run(main())
