"""Trend Watcher Agent (Public).

Analyzes recent activity to propose research topics.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from adapters.helpers import get_reviews, search_issues
from agents.core.base import BaseAgent
from core.config import hafs_config, VERIFIED_DIR

class TrendWatcher(BaseAgent):
    """Monitors activity streams to identify trending topics."""

    def __init__(self):
        super().__init__("TrendWatcher", "Analyze work patterns to discover undefined research topics.")
        self.issue_tracker = None
        self.code_review = None
        self.code_search = None

    async def setup(self):
        await super().setup()
        # Dynamic injection via registry - adapters can be registered by plugins
        try:
            from core.registry import agent_registry
            self.issue_tracker = agent_registry.get_adapter("issue_tracker")
            await self.issue_tracker.connect()
        except: pass

        try:
            from core.registry import agent_registry
            self.code_review = agent_registry.get_adapter("code_review")
            await self.code_review.connect()
        except: pass

        try:
            from core.registry import agent_registry
            self.code_search = agent_registry.get_adapter("code_search")
            await self.code_search.connect()
        except: pass

    async def run_task(self, ignore_topics: List[str] = None) -> List[str]:
        """Analyze streams and return a list of topics."""
        print(f"[{self.name}] Scanning activity streams...")

        if ignore_topics is None: ignore_topics = []

        # Load Verified Topics
        verified_topics = []
        if VERIFIED_DIR.exists():
            for f in VERIFIED_DIR.glob("*.md"):
                verified_topics.append(f.stem)

        # 1. Gather Data (Robustly)
        active_issues = []
        if self.issue_tracker:
            try:
                active_issues = await search_issues(
                    self.issue_tracker, "assignee:me status:open"
                )
            except: pass

        pending_reviews = []
        if self.code_review:
            try:
                pending_reviews = await get_reviews(self.code_review)
            except: pass

        # 2. Synthesize
        data_context = (
            f"EXISTING TOPICS:\n" + "\n".join(verified_topics) + "\n\n"
            f"ACTIVE ISSUES ({len(active_issues)}):\n" +
            "\n".join([f"- {getattr(b, 'title', 'Issue')}" for b in active_issues]) + "\n\n" +
            f"PENDING REVIEWS ({len(pending_reviews)}):\n" +
            "\n".join([f"- {getattr(c, 'title', 'Review')}" for c in pending_reviews])
        )

        prompt = (
            "Analyze the following work items. Identify 3 distinct 'Themes' or 'Topics' for the next research session.\n"
            "Prioritize CONTINUITY over NOVELTY.\n"
            f"DATA:\n{data_context}\n"
            "Output ONLY a bulleted list of topics."
        )

        response = await self.generate_thought(prompt)

        # Parse response
        topics = []
        for line in response.splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                topics.append(line[2:].strip())

        print(f"[{self.name}] Proposed Topics: {topics}")
        return topics
