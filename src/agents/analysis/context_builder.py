"""Context Builder Agent.

Orchestrates the selection and assembly of context for AI tasks.
Uses embeddings, history, and project state to build the most relevant prompt context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agents.core.base import BaseAgent

logger = logging.getLogger(__name__)


class AutonomousContextAgent(BaseAgent):
    """Selects and prioritizes context for other agents."""

    def __init__(self):
        super().__init__(
            "AutonomousContextAgent",
            "Identify and assemble the most relevant context for a given task."
        )
        self.model_tier = "fast"

    async def identify_relevant_files(
        self,
        task: str,
        project_files: List[str],
        limit: int = 5
    ) -> List[str]:
        """Ask the model to identify which files are most relevant to a task."""

        prompt = f"""Given this task:
{task}

And this list of files:
{json.dumps(project_files[:100], indent=2)}

Identify the {limit} most relevant files for completing this task.
Respond only with a JSON list of file paths."""

        response = await self.generate_thought(prompt)

        try:
            import json
            import re
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except:
            pass

        return []

    async def prioritize_context(
        self,
        task: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 4000
    ) -> List[Dict[str, Any]]:
        """Prioritize context chunks to fit within a token limit."""

        # Simple heuristic for now: sort by relevance if available
        sorted_chunks = sorted(
            context_chunks,
            key=lambda x: x.get("score", 0),
            reverse=True
        )

        # Truncate to fit (very rough estimate)
        final_chunks = []
        current_est_tokens = 0
        for chunk in sorted_chunks:
            chunk_len = len(chunk.get("content", "")) // 4
            if current_est_tokens + chunk_len < max_tokens:
                final_chunks.append(chunk)
                current_est_tokens += chunk_len

        return final_chunks

    async def run_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run context building task."""
        objective = task_info.get("objective", "")
        available_files = task_info.get("files", [])

        relevant_files = await self.identify_relevant_files(objective, available_files)

        return {
            "objective": objective,
            "relevant_files": relevant_files,
            "context_assembled": True
        }
