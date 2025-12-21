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

    def __init__(self):
        super().__init__(
            name="OracleDataGenerator",
            domain="oracle",
            teacher_tier="coding",
        )
        self._routines: list[dict] = []
        self._orchestrator = None

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
        from hafs.core.orchestrator_v2 import UnifiedOrchestrator

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
            hook_emphasis = """
This is a HOOK - it modifies or replaces vanilla ALTTP code. Your instruction should ask about:
- What vanilla behavior this hook changes
- What new functionality it adds
- How the ROM hacking technique works
"""

        return f"""I will give you a routine from the Oracle-of-Secrets ROM hack for ALTTP.
Your task is to reverse-engineer the intent and write a user prompt (Instruction) that would request information about this ROM hack modification.

This is from a ROM hack, so it may include custom features, hooks to vanilla code, or new content not in the original game.{hook_emphasis}

ROUTINE NAME: {item.name}
CONTEXT:
{context}

CODE:
```asm
{code}
```

Respond with a JSON object containing:
1. "instruction": A natural language question asking about this ROM hack routine. For example:
   - "How does the Oracle hack implement {item.name}?"
   - "Explain the ROM hacking technique used in {item.name}"
   - "What does the {item.name} hook change from vanilla ALTTP?"
   - "How do you add {item.category} content to ALTTP?"

2. "input": Any relevant context (vanilla behavior, ROM address, related routines). Leave empty if not needed.

3. "output": A detailed explanation of:
   - What this routine does
   - If it's a hook: what vanilla code it modifies and why
   - The ROM hacking technique used (code injection, hook, expansion, etc.)
   - How it fits into the overall hack

Focus on:
- ROM hacking pedagogy (teaching how to implement similar features)
- Vanilla vs hack differences
- Technical implementation details

JSON FORMAT:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from Oracle routine."""
        if not isinstance(item, OracleSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            from hafs.core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.CODING,
                    provider=Provider.GEMINI,
                ),
                timeout=30.0,
            )

            response = response_obj.content

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

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
                teacher_model="gemini-2.0-flash",
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
