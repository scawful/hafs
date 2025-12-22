"""Pattern Analyzer Agent for Expert Question Generation.

Scans codebases for interesting patterns and generates pedagogical questions
that elicit expert knowledge from the user. The user's answers become
high-quality training samples that capture unique insights not found in code alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.autonomy.base import MemoryAwareAgent

logger = logging.getLogger(__name__)


@dataclass
class CodePattern:
    """A detected code pattern worth asking about."""

    pattern_type: str  # "nmi_handler", "dma_transfer", "rom_trick", etc.
    file_path: str
    line_number: int
    code_snippet: str
    context: str  # Surrounding code for context
    complexity_score: float  # 0-1, how complex/interesting
    pedagogical_value: float  # 0-1, how much teaching value
    related_entities: list[str] = field(default_factory=list)  # Functions, registers, etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code_snippet": self.code_snippet,
            "context": self.context,
            "complexity_score": self.complexity_score,
            "pedagogical_value": self.pedagogical_value,
            "related_entities": self.related_entities,
        }


@dataclass
class ExpertQuestion:
    """A pedagogical question generated from a code pattern."""

    question_id: str
    pattern: CodePattern
    question_text: str
    question_type: str  # "why", "how", "what", "trade-off", "alternative"
    difficulty: str  # "beginner", "intermediate", "advanced"
    learning_objectives: list[str]
    estimated_answer_length: int  # Words
    priority_score: float  # 0-1, overall priority

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "pattern": self.pattern.to_dict(),
            "question_text": self.question_text,
            "question_type": self.question_type,
            "difficulty": self.difficulty,
            "learning_objectives": self.learning_objectives,
            "estimated_answer_length": self.estimated_answer_length,
            "priority_score": self.priority_score,
        }


class PatternAnalyzerAgent(MemoryAwareAgent):
    """Analyzes code to find interesting patterns and generate expert questions.

    Scans ASM, C++, C# code for:
    - NMI/IRQ handlers (timing-critical SNES code)
    - DMA transfers (PPU/VRAM manipulation)
    - ROM tricks (bank switching, JSL redirects, hooks)
    - Custom sprite logic
    - Optimization patterns
    - Hardware register usage
    - Memory management strategies

    Generates questions that would elicit expert knowledge:
    - "Why did you choose this NMI timing approach?"
    - "How does this DMA sequence avoid screen tearing?"
    - "What trade-offs did you consider for this hook placement?"
    """

    # Pattern detection rules
    PATTERN_RULES = {
        "nmi_handler": {
            "asm_patterns": [r"NMI:", r"VBlank:", r"\bNMI\b.*:"],
            "keywords": ["NMI", "VBlank", "interrupt"],
            "complexity_weight": 0.9,  # Very complex
        },
        "dma_transfer": {
            "asm_patterns": [r"\$2[0-3][0-9A-F]{2}", r"HDMA", r"\bDMA\b"],
            "keywords": ["DMA", "HDMA", "$21", "$22", "$43"],
            "complexity_weight": 0.8,
        },
        "rom_hook": {
            "asm_patterns": [r"\borg\s+\$", r"pushpc", r"pullpc", r"JSL.*\$[2-9A-F]"],
            "keywords": ["org", "pushpc", "pullpc", "hook", "JSL"],
            "complexity_weight": 0.7,
        },
        "sprite_custom": {
            "asm_patterns": [r"Sprite.*Custom", r"CustomSprite", r"OAM"],
            "keywords": ["sprite", "OAM", "custom", "graphics"],
            "complexity_weight": 0.6,
        },
        "bank_allocation": {
            "asm_patterns": [r"bank\s*=", r"\$[2-9A-F][0-9A-F]:[0-9A-F]{4}"],
            "keywords": ["bank", "expanded", "free space"],
            "complexity_weight": 0.7,
        },
    }

    def __init__(self):
        super().__init__(
            "PatternAnalyzer",
            "Detect interesting code patterns and generate expert questions",
        )
        self._orchestrator = None
        self.questions_db = self.context_root / "training" / "expert_questions.json"
        self.questions_db.parent.mkdir(parents=True, exist_ok=True)

    async def setup(self):
        """Initialize orchestrator."""
        await super().setup()

        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def analyze_codebase(self, root: Path, file_patterns: Optional[list[str]] = None) -> list[CodePattern]:
        """Scan codebase for interesting patterns.

        Args:
            root: Root directory to scan
            file_patterns: File patterns to match (e.g., ["**/*.asm"])

        Returns:
            List of detected code patterns
        """
        if file_patterns is None:
            file_patterns = ["**/*.asm", "**/*.c", "**/*.cpp", "**/*.cs"]

        patterns: list[CodePattern] = []

        for pattern in file_patterns:
            for file_path in root.rglob(pattern):
                if not file_path.is_file():
                    continue

                try:
                    file_patterns_found = await self._analyze_file(file_path)
                    patterns.extend(file_patterns_found)
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")

        logger.info(f"Found {len(patterns)} interesting code patterns")
        return patterns

    async def _analyze_file(self, file_path: Path) -> list[CodePattern]:
        """Analyze a single file for patterns."""
        patterns: list[CodePattern] = []

        try:
            content = file_path.read_text(errors="replace")
            lines = content.split("\n")

            # Check each pattern rule
            for pattern_type, rule in self.PATTERN_RULES.items():
                # Search for ASM patterns
                for asm_pattern in rule["asm_patterns"]:
                    for i, line in enumerate(lines):
                        if re.search(asm_pattern, line, re.IGNORECASE):
                            # Extract context (±5 lines)
                            start = max(0, i - 5)
                            end = min(len(lines), i + 6)
                            context_lines = lines[start:end]
                            context = "\n".join(context_lines)

                            # Calculate scores
                            complexity = rule["complexity_weight"]
                            pedagogical = self._estimate_pedagogical_value(
                                pattern_type, line, context
                            )

                            patterns.append(
                                CodePattern(
                                    pattern_type=pattern_type,
                                    file_path=str(file_path),
                                    line_number=i + 1,
                                    code_snippet=line.strip(),
                                    context=context,
                                    complexity_score=complexity,
                                    pedagogical_value=pedagogical,
                                    related_entities=self._extract_entities(line),
                                )
                            )

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

        return patterns

    def _estimate_pedagogical_value(
        self,
        pattern_type: str,
        code_line: str,
        context: str,
    ) -> float:
        """Estimate how much teaching value this pattern has."""
        # Higher value for patterns with:
        # - Comments explaining design decisions
        # - Multiple related entities (complex interactions)
        # - Non-obvious optimizations

        value = 0.5  # Base value

        # Boost for comments (indicates design decisions)
        if ";" in code_line or "//" in code_line or "/*" in context:
            value += 0.2

        # Boost for complexity indicators
        complexity_indicators = ["optimize", "trick", "custom", "hack", "workaround"]
        for indicator in complexity_indicators:
            if indicator in context.lower():
                value += 0.1

        return min(1.0, value)

    def _extract_entities(self, code_line: str) -> list[str]:
        """Extract function names, registers, variables from code."""
        entities = []

        # Extract function calls (JSR, JSL, call, etc.)
        func_matches = re.findall(r"(?:JSR|JSL|call)\s+(\w+)", code_line, re.IGNORECASE)
        entities.extend(func_matches)

        # Extract register references ($XX, $XXXX, $XX:XXXX)
        reg_matches = re.findall(r"\$[0-9A-Fa-f]{2,6}(?::[0-9A-Fa-f]{4})?", code_line)
        entities.extend(reg_matches)

        # Extract identifiers (function/variable names)
        id_matches = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", code_line)
        entities.extend(id_matches[:5])  # Limit to 5

        return list(set(entities))  # Remove duplicates

    async def generate_question(self, pattern: CodePattern) -> Optional[ExpertQuestion]:
        """Generate a pedagogical question from a code pattern.

        Args:
            pattern: Detected code pattern

        Returns:
            ExpertQuestion or None if generation failed
        """
        if not self._orchestrator:
            await self.setup()

        prompt = self._build_question_prompt(pattern)

        try:
            from core.orchestrator_v2 import Provider, TaskTier

            response_obj = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=prompt,
                    tier=TaskTier.GENERAL,
                    provider=Provider.GEMINI,
                ),
                timeout=60.0,
            )

            response = response_obj.content

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

            # Generate unique question ID
            question_id = f"q_{pattern.pattern_type}_{pattern.file_path.split('/')[-1]}_{pattern.line_number}"

            return ExpertQuestion(
                question_id=question_id,
                pattern=pattern,
                question_text=data.get("question", ""),
                question_type=data.get("question_type", "why"),
                difficulty=data.get("difficulty", "intermediate"),
                learning_objectives=data.get("learning_objectives", []),
                estimated_answer_length=data.get("estimated_answer_length", 200),
                priority_score=self._calculate_priority(pattern, data),
            )

        except Exception as e:
            logger.error(f"Failed to generate question: {e}")
            return None

    def _build_question_prompt(self, pattern: CodePattern) -> str:
        """Build prompt for question generation."""
        return f"""You are generating a pedagogical question for an expert ROM hacker/SNES developer.

DETECTED PATTERN: {pattern.pattern_type}
FILE: {pattern.file_path}
LINE: {pattern.line_number}

CODE:
{pattern.code_snippet}

CONTEXT:
{pattern.context[:500]}

Generate an expert-level question that would elicit unique insights about this code.

Focus on:
- Design decisions and trade-offs
- Why this approach was chosen
- Alternative approaches considered
- Optimization strategies
- Hardware constraints
- Best practices

JSON FORMAT:
{{
  "question": "Thought-provoking question (1-2 sentences)",
  "question_type": "why|how|what|trade-off|alternative",
  "difficulty": "beginner|intermediate|advanced",
  "learning_objectives": ["What students will learn from the answer"],
  "estimated_answer_length": 200
}}"""

    def _calculate_priority(self, pattern: CodePattern, question_data: dict) -> float:
        """Calculate overall priority score for this question."""
        # Combine pattern scores with question quality
        pattern_score = (pattern.complexity_score + pattern.pedagogical_value) / 2

        # Boost for advanced questions (they teach more)
        difficulty_boost = {"beginner": 0.0, "intermediate": 0.1, "advanced": 0.2}
        diff = question_data.get("difficulty", "intermediate")
        priority = pattern_score + difficulty_boost.get(diff, 0.0)

        return min(1.0, priority)

    async def save_questions(self, questions: list[ExpertQuestion]) -> None:
        """Save questions to JSON database."""
        # Load existing questions
        existing = []
        if self.questions_db.exists():
            with open(self.questions_db) as f:
                existing = json.load(f)

        # Add new questions
        question_dicts = [q.to_dict() for q in questions]
        all_questions = existing + question_dicts

        # Save
        with open(self.questions_db, "w") as f:
            json.dump(all_questions, f, indent=2)

        logger.info(f"Saved {len(questions)} questions to {self.questions_db}")

    async def load_questions(self) -> list[ExpertQuestion]:
        """Load questions from JSON database."""
        if not self.questions_db.exists():
            return []

        with open(self.questions_db) as f:
            data = json.load(f)

        # Reconstruct ExpertQuestion objects
        questions = []
        for q_data in data:
            pattern_data = q_data["pattern"]
            pattern = CodePattern(**pattern_data)

            question = ExpertQuestion(
                question_id=q_data["question_id"],
                pattern=pattern,
                question_text=q_data["question_text"],
                question_type=q_data["question_type"],
                difficulty=q_data["difficulty"],
                learning_objectives=q_data["learning_objectives"],
                estimated_answer_length=q_data["estimated_answer_length"],
                priority_score=q_data["priority_score"],
            )
            questions.append(question)

        return questions
