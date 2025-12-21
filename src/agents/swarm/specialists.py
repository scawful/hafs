"""Deep Dive and Review Agents (Public Port).

SwarmStrategist, CouncilReviewer, DeepDiveDocumenter.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from agents.core.base import BaseAgent


class SwarmStrategist(BaseAgent):
    """The Planner. Decides search terms and paths."""

    def __init__(self):
        super().__init__("SwarmStrategist", "Plan the research session.")

    async def run_task(self, topic: str) -> Dict[str, Any]:
        prompt = (
            f"Topic: {topic}\n"
            "Task: Plan a BROAD yet accurate search strategy.\n"
            "Output a JSON object with: 'bug_query', 'user_bug_query', 'code_terms' (list), 'knowledge_queries' (list).\n"
            "Output ONLY the raw JSON block."
        )
        response = await self.generate_thought(prompt)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {
            "bug_query": topic,
            "user_bug_query": "status:open",
            "code_terms": topic.split(),
            "knowledge_queries": [topic]
        }


class CouncilReviewer(BaseAgent):
    """The Skeptic. Reviews findings."""

    def __init__(self):
        super().__init__("CouncilReviewer", "Critique data for completeness.")

    async def run_task(self, context: str) -> str:
        prompt = (
            "You are the 'Council Critic'.\n"
            "TASK: Review the gathered intelligence.\n"
            "CRITIQUE CRITERIA:\n"
            "1. Fact Check: Flag hallucinations.\n"
            "2. Gap Analysis: What is MISSING?\n"
            "3. SCORING: Assign 'CONFIDENCE_SCORE' (0-100).\n"
            f"DATA:\n{context[:10000]}"
        )
        return await self.generate_thought(prompt)


class DeepDiveDocumenter(BaseAgent):
    """The Historian. Writes the final report."""

    def __init__(self):
        super().__init__("DeepDiveDocumenter", "Synthesize comprehensive documentation.")

    async def run_task(self, context: str) -> str:
        prompt = (
            "You are writing a 'Deep Context' report for the user.\n"
            "TASK: Synthesize the provided data into a professional technical report.\n"
            "STRUCTURE:\n"
            "1. **Summary**: High-level overview.\n"
            "2. **Key Findings**: Detailed analysis of bugs and code.\n"
            "3. **Action Items**: Concrete next steps.\n"
            "4. **Critique & Gaps**: What was missing from the research.\n"
            "5. **CONFIDENCE_SCORE**: A number from 0-100.\n\n"
            f"DATA:\n{context[:15000]}"
        )
        return await self.generate_thought(prompt)
