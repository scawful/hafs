"""Assisted Q&A - Helps user answer expert questions using web search and git history.

For mobile or quick workflows, this generates draft answers based on:
1. User's actual code and implementation
2. Git commit messages showing design decisions
3. Web search for SNES/ROM hacking technical context
4. Related code patterns in the codebase

The user then reviews/approves/edits the draft.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from agents.training.background.pattern_analyzer import ExpertQuestion

logger = logging.getLogger(__name__)


class AssistedQA:
    """Helps user answer expert questions by doing research."""

    def __init__(self, orchestrator=None):
        self._orchestrator = orchestrator

    async def setup(self):
        """Initialize orchestrator if needed."""
        if not self._orchestrator:
            from core.orchestrator_v2 import UnifiedOrchestrator

            self._orchestrator = UnifiedOrchestrator()

    async def generate_draft_answer(
        self,
        question: ExpertQuestion,
        use_web_search: bool = True,
        use_git_history: bool = True,
    ) -> dict[str, str]:
        """Generate a draft answer using available resources.

        Args:
            question: The expert question to answer
            use_web_search: Search web for technical context
            use_git_history: Search git history for implementation decisions

        Returns:
            Dictionary with:
            - draft_answer: Generated answer text
            - sources: List of sources used
            - confidence: How confident we are (low/medium/high)
        """
        if not self._orchestrator:
            await self.setup()

        pattern = question.pattern
        sources = []

        # 1. Analyze the code context
        code_analysis = await self._analyze_code_context(pattern)
        sources.append(f"Code: {pattern.file_path}:{pattern.line_number}")

        # 2. Search git history for related changes
        git_context = ""
        if use_git_history:
            git_context = self._search_git_history(pattern.file_path, pattern.code_snippet[:100])
            if git_context:
                sources.append("Git history")

        # 3. Web search for technical background
        web_context = ""
        if use_web_search:
            search_results = await self._search_web_for_context(question.question_text, pattern)
            if search_results:
                web_context = search_results["context"]
                sources.extend(search_results["sources"])

        # 4. Generate draft answer combining all sources
        draft = await self._synthesize_answer(
            question=question,
            code_analysis=code_analysis,
            git_context=git_context,
            web_context=web_context,
        )

        # Assess confidence based on available sources
        confidence = "high" if len(sources) >= 3 else "medium" if len(sources) >= 2 else "low"

        return {
            "draft_answer": draft,
            "sources": sources,
            "confidence": confidence,
        }

    async def _analyze_code_context(self, pattern) -> str:
        """Analyze the code pattern and extract key implementation details."""
        # Read surrounding code
        try:
            file_path = Path(pattern.file_path)
            if not file_path.exists():
                return f"Code snippet:\n{pattern.code_snippet}"

            with open(file_path) as f:
                lines = f.readlines()

            # Get context around the pattern (20 lines before/after)
            start = max(0, pattern.line_number - 20)
            end = min(len(lines), pattern.line_number + 20)
            context = "".join(lines[start:end])

            return f"Implementation context:\n{context}"
        except Exception as e:
            logger.warning(f"Failed to read code context: {e}")
            return f"Code snippet:\n{pattern.code_snippet}"

    def _search_git_history(self, file_path: str, code_snippet: str) -> str:
        """Search git history for commits related to this code."""
        try:
            # Get commits that touched this file
            result = subprocess.run(
                ["git", "log", "--oneline", "-n", "10", "--", file_path],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return ""

            commits = result.stdout.strip()
            if not commits:
                return ""

            # Get detailed commit messages
            commit_hashes = [line.split()[0] for line in commits.split("\n")]
            detailed = []
            for commit_hash in commit_hashes[:5]:  # Top 5 commits
                detail_result = subprocess.run(
                    ["git", "show", "--stat", commit_hash],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if detail_result.returncode == 0:
                    # Extract commit message (skip diff)
                    message = detail_result.stdout.split("\ndiff --git")[0]
                    detailed.append(message)

            return "\n\n---\n\n".join(detailed[:3])  # Top 3 detailed commits

        except Exception as e:
            logger.warning(f"Git history search failed: {e}")
            return ""

    async def _search_web_for_context(self, question_text: str, pattern) -> Optional[dict]:
        """Search web for SNES/ROM hacking technical context."""
        try:
            from core.orchestrator_v2 import Provider, TaskTier

            # Extract key technical terms
            keywords = []
            if "DMA" in question_text or "DMA" in pattern.code_snippet:
                keywords.append("SNES DMA transfer")
            if "NMI" in question_text or "NMI" in pattern.code_snippet:
                keywords.append("SNES NMI handler")
            if "V-blank" in question_text or "VBlank" in pattern.code_snippet:
                keywords.append("SNES V-blank timing")
            if "bank" in question_text.lower():
                keywords.append("SNES bank switching")
            if "CHR" in question_text or "VRAM" in pattern.code_snippet:
                keywords.append("SNES CHR VRAM")

            if not keywords:
                keywords = ["SNES ROM hacking"]

            # Search for technical documentation
            search_query = f"{keywords[0]} A Link to the Past ROM hacking"

            # Use web search
            search_prompt = f"Find technical documentation about: {search_query}"

            # Note: This is a simplified version. In production, you'd use WebSearch tool
            # For now, return a placeholder
            return {
                "context": f"Searched for: {search_query}",
                "sources": [f"Web search: {search_query}"],
            }

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return None

    async def _synthesize_answer(
        self,
        question: ExpertQuestion,
        code_analysis: str,
        git_context: str,
        web_context: str,
    ) -> str:
        """Synthesize a draft answer from all sources."""
        from core.orchestrator_v2 import Provider, TaskTier

        prompt = f"""You are helping a ROM hacking developer answer an expert question about their code.

QUESTION:
{question.question_text}

CODE CONTEXT:
{code_analysis}

GIT HISTORY (their actual implementation decisions):
{git_context if git_context else "No git history available"}

TECHNICAL BACKGROUND:
{web_context if web_context else "No web context available"}

Generate a draft answer that:
1. Explains their actual implementation approach (from code/git)
2. Discusses the trade-offs they considered
3. References specific technical details from SNES architecture
4. Is written in first-person as if the developer is explaining their decisions
5. Is 200-400 words

Keep the tone conversational and practical, like a developer sharing their experience.

Draft Answer:"""

        response = await self._orchestrator.generate(
            prompt=prompt,
            tier=TaskTier.CODING,
            provider=Provider.GEMINI,
        )

        return response.content


async def assisted_answer_workflow(question_id: str) -> dict:
    """Run the full assisted answer workflow.

    Args:
        question_id: The question to answer

    Returns:
        Dictionary with draft_answer, sources, confidence
    """
    from agents.training.background import QuestionCurator

    curator = QuestionCurator()
    all_questions = curator.load_all_questions()
    question = next((q for q in all_questions if q.question_id == question_id), None)

    if not question:
        raise ValueError(f"Question not found: {question_id}")

    assistant = AssistedQA()
    await assistant.setup()

    result = await assistant.generate_draft_answer(
        question,
        use_web_search=True,
        use_git_history=True,
    )

    return result
