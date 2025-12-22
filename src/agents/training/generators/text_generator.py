"""Text Data Generator for writing style training data.

Generates instruction-tuning data from text documents,
useful for training models on specific writing styles or domains.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class TextSourceItem(SourceItem):
    """Source item for text passages."""

    text: str = ""
    file_path: str = ""
    section_title: str = ""
    word_count: int = 0
    tags: list[str] = field(default_factory=list)

    @property
    def item_id(self) -> str:
        return f"{self.file_path}:{self.section_title}"


class TextDataGenerator(DataGenerator):
    """Generate instruction-tuning data for writing style.

    Processes markdown and text documents, splitting them into
    coherent passages and generating instruction pairs.
    """

    # Default source directory
    DEFAULT_SOURCE_DIR = Path.home() / ".context" / "knowledge" / "verified"

    # Patterns for splitting documents
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    # Minimum passage length (words)
    MIN_PASSAGE_WORDS = 50

    # Maximum passage length (words)
    MAX_PASSAGE_WORDS = 500

    def __init__(
        self,
        source_dir: Optional[Path] = None,
        include_patterns: Optional[list[str]] = None,
    ):
        super().__init__(
            name="TextDataGenerator",
            domain="text",
            teacher_tier="creative",  # Use creative tier for writing
        )
        self.source_dir = source_dir or self.DEFAULT_SOURCE_DIR
        self.include_patterns = include_patterns or ["*.md", "*.txt"]
        self._orchestrator = None

    async def setup(self):
        """Initialize resources."""
        await super().setup()

        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[TextSourceItem]:
        """Extract text passages from documents."""
        items: list[TextSourceItem] = []

        if not self.source_dir.exists():
            logger.warning(f"Source directory not found: {self.source_dir}")
            return items

        for pattern in self.include_patterns:
            for file_path in self.source_dir.rglob(pattern):
                try:
                    file_items = self._split_into_passages(file_path)
                    items.extend(file_items)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        logger.info(f"Extracted {len(items)} text passages")
        return items

    def _split_into_passages(self, path: Path) -> list[TextSourceItem]:
        """Split a document into coherent passages by headers."""
        items: list[TextSourceItem] = []
        content = path.read_text(errors="replace")

        # Find all headers
        headers = list(self.HEADER_PATTERN.finditer(content))

        if not headers:
            # No headers, treat entire document as one passage
            word_count = len(content.split())
            if self.MIN_PASSAGE_WORDS <= word_count <= self.MAX_PASSAGE_WORDS:
                items.append(
                    TextSourceItem(
                        name=path.stem,
                        content=content,
                        source=str(path),
                        text=content,
                        file_path=str(path),
                        section_title=path.stem,
                        word_count=word_count,
                    )
                )
            return items

        # Split by headers
        for i, match in enumerate(headers):
            header_level = len(match.group(1))
            header_title = match.group(2).strip()
            start = match.end()

            # Find end of section (next header of same or higher level, or EOF)
            end = len(content)
            for next_match in headers[i + 1 :]:
                next_level = len(next_match.group(1))
                if next_level <= header_level:
                    end = next_match.start()
                    break

            section_text = content[start:end].strip()
            word_count = len(section_text.split())

            # Skip sections that are too short or too long
            if not (self.MIN_PASSAGE_WORDS <= word_count <= self.MAX_PASSAGE_WORDS):
                continue

            # Extract tags from content (look for keywords)
            tags = self._extract_tags(section_text)

            items.append(
                TextSourceItem(
                    name=f"{path.stem}_{header_title}",
                    content=section_text,
                    source=str(path.relative_to(self.source_dir))
                    if self.source_dir in path.parents
                    else str(path),
                    text=section_text,
                    file_path=str(path),
                    section_title=header_title,
                    word_count=word_count,
                    tags=tags,
                )
            )

        return items

    def _extract_tags(self, text: str) -> list[str]:
        """Extract semantic tags from text content."""
        tags = []

        # Look for common patterns
        tag_patterns = [
            (r"\b(TODO|FIXME|NOTE)\b", "annotation"),
            (r"\b(example|tutorial|guide)\b", "educational", re.IGNORECASE),
            (r"\b(warning|caution|important)\b", "advisory", re.IGNORECASE),
            (r"\b(API|SDK|library)\b", "technical", re.IGNORECASE),
            (r"```", "code_block"),
        ]

        for pattern_info in tag_patterns:
            if len(pattern_info) == 2:
                pattern, tag = pattern_info
                flags = 0
            else:
                pattern, tag, flags = pattern_info

            if re.search(pattern, text, flags):
                tags.append(tag)

        return tags

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for text passage."""
        if not isinstance(item, TextSourceItem):
            raise TypeError(f"Expected TextSourceItem, got {type(item)}")

        tags_str = ", ".join(item.tags) if item.tags else "general"

        template = get_prompt("agents.training.generators.text_generator.prompt", "")
        if not template:
            template = (
                "I will give you a text passage. Your task is to create an instruction-tuning pair where:\n"
                "1. The instruction asks for content similar to this passage\n"
                "2. The output is the passage itself (or a refined version)\n\n"
                "SECTION: {section_title}\n"
                "SOURCE: {source}\n"
                "TAGS: {tags}\n"
                "WORD COUNT: {word_count}\n\n"
                "TEXT:\n"
                "---\n"
                "{text}\n"
                "---\n\n"
                "Generate a JSON object with:\n"
                "1. \"instruction\": A natural language request that would elicit this kind of writing. "
                "Be specific about style, tone, and content.\n"
                "2. \"input\": Any context the writer would need (topic, audience, constraints). Can be empty.\n"
                "3. \"output\": The text passage (you may lightly edit for clarity but preserve the style).\n\n"
                "JSON FORMAT:\n"
                "{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}\n"
            )

        return template.format(
            section_title=item.section_title,
            source=item.source,
            tags=tags_str,
            word_count=item.word_count,
            text=item.text,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from text passage."""
        if not isinstance(item, TextSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            response, model_name = await asyncio.wait_for(
                self.generate_with_rotation(prompt, tier="creative"),
                timeout=45.0,
            )
            if not response:
                return None

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

            return TrainingSample(
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", item.text),
                domain="text",
                source=item.source,
                teacher_model=model_name,
                teacher_prompt=prompt,
                kg_entities=item.tags,
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
        gen = TextDataGenerator()
        await gen.setup()

        items = await gen.extract_source_items()
        print(f"Found {len(items)} text passages")

        if items:
            result = await gen.run_generation(
                limit=5,
                output_path=Path("test_text_train.jsonl"),
            )
            print(f"Generated {result.processed} samples")

    import asyncio
    asyncio.run(main())
