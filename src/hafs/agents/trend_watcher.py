"""Trend Watcher Agent (Public).

Analyzes recent activity to propose research topics.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from hafs.agents.base import BaseAgent
from hafs.core.registry import agent_registry
from hafs.core.config import hafs_config, VERIFIED_DIR

class TrendWatcher(BaseAgent):
    """Monitors activity streams to identify trending topics."""

    def __init__(self):
        super().__init__("TrendWatcher", "Analyze work patterns to discover undefined research topics.")
        self.bugs = None
        self.review = None
        self.cs = None

    async def setup(self):
        await super().setup()
        # Dynamic injection via registry
        try:
            self.bugs = agent_registry.get_adapter("buganizer")
            await self.bugs.connect()
        except: pass
        
        try:
            self.review = agent_registry.get_adapter("critique")
            await self.review.connect()
        except: pass
        
        try:
            self.cs = agent_registry.get_adapter("codesearch")
            await self.cs.connect()
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
        active_bugs = []
        if self.bugs:
            try:
                # Assuming adapter has search_bugs or similar
                if hasattr(self.bugs, "search_bugs"):
                    active_bugs = await self.bugs.search_bugs("assignee:me status:open")
            except: pass
            
        pending_cls = []
        if self.review:
            try:
                pending_cls = await self.review.get_reviews()
            except: pass
        
        # 2. Synthesize
        data_context = (
            f"EXISTING TOPICS:\n" + "\n".join(verified_topics) + "\n\n"
            f"ACTIVE BUGS ({len(active_bugs)}):\n" + 
            "\n".join([f"- {getattr(b, 'title', 'Bug')}" for b in active_bugs]) + "\n\n" + 
            f"PENDING REVIEWS ({len(pending_cls)}):\n" + 
            "\n".join([f"- {getattr(c, 'title', 'CL')}" for c in pending_cls])
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