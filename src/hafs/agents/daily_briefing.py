"""Daily Briefing Agent (Public).

Synthesizes daily status reports and work stream analysis.
This is the public stub version - adapters should be provided via plugins.
"""
import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from hafs.agents.base import BaseAgent
from hafs.core.orchestrator import ModelOrchestrator

# Generic adapter interface for public version
class GenericAdapter:
    """Placeholder adapter for external integrations."""
    async def connect(self): pass
    async def get_submitted(self, user: str, limit: int = 5) -> List: return []
    async def get_reviews(self, user: str = None) -> List: return []
    async def search_bugs(self, query: str) -> List: return []

class DailyBriefingAgent(BaseAgent):
    """Chief of Staff. Summarizes recent activity and provides strategic context."""

    def __init__(self):
        super().__init__("DailyBriefing", "Synthesize daily status reports and work stream analysis.")
        # Adapters (stubs - override via plugins)
        self.code_review = GenericAdapter()
        self.issue_tracker = GenericAdapter()
        self.orchestrator = None
        self.briefings_dir = Path.home() / ".context" / "background_agent" / "briefings"

    async def setup(self):
        await super().setup()
        api_key = os.getenv("AISTUDIO_API_KEY")
        self.orchestrator = ModelOrchestrator(api_key)
        await self.code_review.connect()
        await self.issue_tracker.connect()
        self.briefings_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(self, user: Optional[str] = None):
        """Generate a Morning Download report."""
        user = user or os.getenv("USER", "unknown")
        print(f"[{self.name}] Generating Daily Briefing for {user}...")

        # 1. Gather Data from Adapters (will be empty with stubs)
        submitted = await self.code_review.get_submitted(user, limit=5)
        pending = await self.code_review.get_reviews(user)
        bugs = await self.issue_tracker.search_bugs("assignee:me status:open")

        # 2. Synthesize with Orchestrator (REASONING tier)
        briefing = await self._synthesize_briefing(user, submitted, pending, bugs)

        # 3. Save to briefings directory
        date_str = datetime.now().strftime("%Y%m%d")
        briefing_path = self.briefings_dir / f"briefing_{date_str}.md"
        briefing_path.write_text(briefing)

        # Link as latest
        latest_path = self.briefings_dir / "latest.md"
        latest_path.write_text(briefing_path.read_text())

        print(f"[{self.name}] Briefing complete: {briefing_path}")
        return briefing

    async def _synthesize_briefing(self, user: str, submitted, pending, bugs) -> str:
        date_str = datetime.now().strftime("%B %d, %Y")

        data_block = f"USER: {user}\nDATE: {date_str}\n\n"
        data_block += "RECENTLY SUBMITTED (24h):\n"
        for cl in submitted:
            data_block += f"- CL {getattr(cl, 'id', 'unknown')}: {getattr(cl, 'title', 'Untitled')}\n"

        data_block += "\nPENDING REVIEWS:\n"
        for cl in pending:
            data_block += f"- CL {getattr(cl, 'id', 'unknown')}: {getattr(cl, 'title', 'Untitled')} (Status: {getattr(cl, 'status', 'unknown')})\n"

        data_block += "\nACTIVE ISSUES:\n"
        for b in bugs:
            bug_id = b.get('id', 'unknown') if isinstance(b, dict) else getattr(b, 'id', 'unknown')
            priority = b.get('priority', 'unknown') if isinstance(b, dict) else getattr(b, 'priority', 'unknown')
            title = b.get('title', 'Untitled') if isinstance(b, dict) else getattr(b, 'title', 'Untitled')
            data_block += f"- #{bug_id} ({priority}): {title}\n"

        prompt = (
            "You are a Principal Engineer's Chief of Staff. Synthesize the following data into a high-density 'Morning Briefing'.\n"
            "Format your response in professional markdown.\n\n"
            "Sections:\n"
            "1. **Strategic Focus**: What is the singular most important objective for today?\n"
            "2. **Work Stream Updates**: Group the issues and code reviews into 2-3 logical work streams.\n"
            "3. **Blocking Items**: What needs immediate attention or review?\n"
            "4. **Action Plan**: 3-5 concise bullet points for the user's day.\n\n"
            f"DATA:\n{data_block}"
        )

        try:
            return await self.orchestrator.generate_content(prompt, tier="reasoning")
        except Exception as e:
            return f"# Daily Briefing Failed\nError: {e}"
