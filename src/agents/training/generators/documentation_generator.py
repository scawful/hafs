"""Documentation Generator for ROM hacking guides and tutorials.

Generates instruction-tuning data from markdown/text documentation about
ROM hacking, ALTTP mechanics, and Zelda development.
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
from agents.training.json_utils import extract_json_from_response
from agents.training.generators.prompt_templates import PromptTemplateRotator
from agents.training.resource_discovery import ZeldaResourceIndexer
from config.prompts import get_prompt

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_PROMPT = """You are an expert ROM hacker and technical writer. {instruction_prefix}.

This is from {source} documentation - educational content about SNES ROM hacking and ALTTP modding.

SECTION: {section}
CONTEXT:
{context}

CONTENT:
{content}{code_section}

Generate a JSON object with:

1. "instruction": A clear question or request based on this documentation. Vary the approach:
   - Ask how to implement a technique described in the doc
   - Request explanation of a ROM hacking concept
   - Ask for step-by-step tutorial on a topic
   - Request code example for a technique
   - Ask about best practices or common pitfalls

2. "input": Context for the question (1-2 sentences):
   - What ROM hacking task or goal this addresses
   - What knowledge is assumed (beginner/intermediate/advanced)
   - Any specific constraints or requirements

3. "output": Tutorial-style explanation (200-400 words):
   - Clear explanation of the concept/technique
   - Step-by-step instructions if applicable
   - Code examples with explanations
   - Best practices and common pitfalls
   - Related techniques or further reading

QUALITY REQUIREMENTS:
- Maintain pedagogical, tutorial tone
- Use concrete examples and code snippets
- Explain WHY techniques work, not just HOW
- Reference vanilla ALTTP behavior when relevant
- Include practical tips from experience

JSON FORMAT:
{{
  "instruction": "...",
  "input": "...",
  "output": "..."
}}
"""


@dataclass
class DocumentationSourceItem(SourceItem):
    """Source item from documentation files."""

    file_path: str = ""
    file_type: str = ""  # markdown, text
    title: str = ""
    section: str = ""  # Heading or section name
    content_text: str = ""
    code_blocks: list[str] = field(default_factory=list)  # Extracted code blocks
    headings: list[str] = field(default_factory=list)  # Section hierarchy
    has_assembly: bool = False
    has_rom_hacking: bool = False

    @property
    def item_id(self) -> str:
        return f"docs:{Path(self.file_path).stem}:{self.section}"


class DocumentationGenerator(DataGenerator):
    """Generate from markdown/txt ROM hacking documentation.

    Extracts pedagogical content from:
    - book-of-mudora: Oracle of Secrets documentation
    - hyrule-historian: ROM hack guides
    - ~/Code/docs/zelda: Personal notes and guides
    - ~/Documents/Zelda: Additional resources

    Focuses on tutorial-style content that explains ROM hacking techniques.
    """

    # Patterns to detect ROM hacking content
    ROM_HACK_KEYWORDS = [
        "hook", "patch", "JSL", "org", "bank", "ROM address",
        "expanded", "custom", "modification", "vanilla",
        "asar", "sprite", "graphics", "tilemap", "disassembly",
        "snes", "65816", "wram", "vram", "dma", "ppu", "apu",
    ]

    ASM_KEYWORDS = [
        "LDA", "STA", "JSR", "RTS", "RTL", "PHP", "PLP",
        "$7E:", "$21", "register", "accumulator", "NMI",
        "SEP", "REP", "BRK", "RTL", "RTI", "$42", "$43",
    ]

    def __init__(self, use_template_variation: bool = True, min_section_length: int = 200):
        """Initialize documentation generator.

        Args:
            use_template_variation: Whether to use prompt template rotation
            min_section_length: Minimum characters per section (filter short sections)
        """
        super().__init__(
            name="DocumentationGenerator",
            domain="text",
            teacher_tier="general",
        )
        self._indexer: Optional[ZeldaResourceIndexer] = None
        self._orchestrator = None
        self.use_template_variation = use_template_variation
        self.min_section_length = min_section_length

        # Initialize template rotator for diversity
        if self.use_template_variation:
            self.template_rotator = PromptTemplateRotator(domain="text")

    async def setup(self):
        """Initialize resources and index documentation files."""
        await super().setup()

        # Initialize resource indexer
        self._indexer = ZeldaResourceIndexer()

        # Load or build index
        index_result = self._indexer.load_index()
        if not index_result:
            logger.info("Building documentation resource index...")
            index_result = await self._indexer.discover_and_index()

        logger.info(f"Documentation index loaded: {index_result.total_files} files")

        # Initialize orchestrator
        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[DocumentationSourceItem]:
        """Extract sections from documentation files."""
        if not self._indexer:
            await self.setup()

        items: list[DocumentationSourceItem] = []

        # Get markdown and text files from documentation sources
        doc_files = [
            f for f in self._indexer._files
            if f.file_type in ("markdown", "text")
            and any(
                src in f.source_dir
                for src in ["book-of-mudora", "hyrule-historian", "docs", "Documents"]
            )
        ]

        logger.info(f"Found {len(doc_files)} documentation files to process")

        for resource_file in doc_files:
            try:
                file_items = await self._extract_sections_from_file(resource_file.path)
                items.extend(file_items)
            except Exception as e:
                logger.error(f"Error processing {resource_file.path}: {e}")

        logger.info(f"Extracted {len(items)} documentation sections")
        return items

    async def _extract_sections_from_file(self, path: Path) -> list[DocumentationSourceItem]:
        """Extract sections from a documentation file."""
        items: list[DocumentationSourceItem] = []

        try:
            content = path.read_text(errors="replace")

            # Determine if markdown or text
            file_type = "markdown" if path.suffix == ".md" else "text"

            if file_type == "markdown":
                items = await self._extract_markdown_sections(path, content)
            else:
                # For text files, treat whole file as one section
                item = await self._extract_text_section(path, content)
                if item:
                    items.append(item)

        except Exception as e:
            logger.error(f"Error extracting sections from {path}: {e}")

        return items

    async def _extract_markdown_sections(self, path: Path, content: str) -> list[DocumentationSourceItem]:
        """Extract sections from markdown file by headings."""
        items: list[DocumentationSourceItem] = []
        lines = content.split("\n")

        current_section: Optional[dict] = None
        heading_stack: list[tuple[int, str]] = []  # (level, heading)

        for line in lines:
            stripped = line.strip()

            # Check for heading
            if stripped.startswith("#"):
                # Save previous section if exists
                if current_section:
                    item = self._finalize_doc_section(current_section, path, "markdown")
                    if item:
                        items.append(item)

                # Parse heading level and text
                level = len(line) - len(line.lstrip("#"))
                heading = stripped.lstrip("#").strip()

                # Update heading stack (maintain hierarchy)
                heading_stack = [
                    (lvl, hdg) for lvl, hdg in heading_stack if lvl < level
                ]
                heading_stack.append((level, heading))

                # Start new section
                current_section = {
                    "section": heading,
                    "headings": [hdg for _, hdg in heading_stack],
                    "content_lines": [],
                    "code_blocks": [],
                    "in_code_block": False,
                    "code_lang": "",
                }

            elif current_section:
                # Track code blocks
                if stripped.startswith("```"):
                    if not current_section["in_code_block"]:
                        # Start code block
                        current_section["in_code_block"] = True
                        current_section["code_lang"] = stripped[3:].strip()
                        current_section["code_blocks"].append([])
                    else:
                        # End code block
                        current_section["in_code_block"] = False
                elif current_section["in_code_block"]:
                    # Add line to current code block
                    current_section["code_blocks"][-1].append(line)
                else:
                    # Regular content line
                    current_section["content_lines"].append(line)

        # Save final section
        if current_section:
            item = self._finalize_doc_section(current_section, path, "markdown")
            if item:
                items.append(item)

        return items

    async def _extract_text_section(self, path: Path, content: str) -> Optional[DocumentationSourceItem]:
        """Extract content from a plain text file."""
        # Treat whole file as one section
        section_data = {
            "section": path.stem,
            "headings": [path.stem],
            "content_lines": content.split("\n"),
            "code_blocks": [],
        }

        return self._finalize_doc_section(section_data, path, "text")

    def _finalize_doc_section(
        self, section_data: dict, path: Path, file_type: str
    ) -> Optional[DocumentationSourceItem]:
        """Convert section data dict to DocumentationSourceItem."""
        # Join content lines
        content_text = "\n".join(section_data["content_lines"])

        # Filter out very short sections
        if len(content_text) < self.min_section_length:
            return None

        # Join code blocks
        code_blocks = ["\n".join(block) for block in section_data.get("code_blocks", [])]

        # Check for ROM hacking content
        has_rom_hacking = any(
            keyword.lower() in content_text.lower()
            for keyword in self.ROM_HACK_KEYWORDS
        )

        # Check for assembly content
        has_assembly = any(
            keyword in content_text
            for keyword in self.ASM_KEYWORDS
        )

        # Skip sections that don't look ROM-hacking related
        if not has_rom_hacking and not has_assembly:
            return None

        # Build content for embedding
        content_parts = [
            f"Section: {section_data['section']}",
            f"File: {path.name}",
        ]

        if section_data.get("headings"):
            hierarchy = " > ".join(section_data["headings"])
            content_parts.append(f"Hierarchy: {hierarchy}")

        # Add preview of content
        preview = content_text[:300] + "..." if len(content_text) > 300 else content_text
        content_parts.append(f"Content: {preview}")

        content = "\n".join(content_parts)

        # Determine title from first heading
        title = section_data["headings"][0] if section_data["headings"] else path.stem

        return DocumentationSourceItem(
            name=section_data["section"],
            content=content,
            source=f"docs_{path.parent.name}",
            file_path=str(path),
            file_type=file_type,
            title=title,
            section=section_data["section"],
            content_text=content_text,
            code_blocks=code_blocks,
            headings=section_data["headings"],
            has_assembly=has_assembly,
            has_rom_hacking=has_rom_hacking,
        )

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for documentation section."""
        if not isinstance(item, DocumentationSourceItem):
            raise TypeError(f"Expected DocumentationSourceItem, got {type(item)}")

        # Use template variation for instruction diversity
        instruction_prefix = "Generate training data for this ROM hacking documentation"
        if self.use_template_variation and hasattr(self, 'template_rotator'):
            template = self.template_rotator.get_next_template()
            instruction_prefix = f"Generate training data using this perspective: '{template.template}'"

        # Build context
        context_parts = [f"File: {Path(item.file_path).name}"]

        if item.headings:
            hierarchy = " > ".join(item.headings)
            context_parts.append(f"Section hierarchy: {hierarchy}")

        if item.code_blocks:
            context_parts.append(f"Contains {len(item.code_blocks)} code block(s)")

        if item.has_assembly:
            context_parts.append("Contains assembly code")

        if item.has_rom_hacking:
            context_parts.append("Contains ROM hacking techniques")

        context = "\n".join(context_parts)

        # Include code blocks if present
        code_section = ""
        if item.code_blocks:
            code_examples = "\n\n".join([f"```\n{block}\n```" for block in item.code_blocks[:3]])
            code_section = f"\n\nCODE EXAMPLES:\n{code_examples}"

        content_excerpt = f"{item.content_text[:1500]}..."
        template = get_prompt(
            "agents.training.generators.documentation_generator.prompt",
            default=DEFAULT_TEACHER_PROMPT,
        )
        return template.format(
            instruction_prefix=instruction_prefix,
            source=item.source,
            section=item.section,
            context=context,
            content=content_excerpt,
            code_section=code_section,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate instruction from documentation."""
        if not isinstance(item, DocumentationSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.FAST,
                    provider=Provider.GEMINI,
                ),
                timeout=120.0,
            )

            response = response_obj.content

            # Extract JSON from response
            data = extract_json_from_response(response)
            if not data:
                logger.warning(f"Failed to extract JSON from response for {item.section}")
                return None

            # Build training sample
            instruction = str(data.get("instruction", "")).strip()
            input_text = str(data.get("input", "")).strip()
            output = str(data.get("output", "")).strip()

            # KG entities: section name, headings
            kg_entities = [item.section] + item.headings

            return TrainingSample(
                instruction=instruction,
                input=input_text,
                output=output,
                domain="text",
                source=item.source,
                teacher_model="gemini-3-flash-preview",
                teacher_prompt=str(prompt),
                kg_entities=kg_entities,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.section}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {item.section}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate for {item.section}: {e}")
            return None
