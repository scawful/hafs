import asyncio
import os
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from hafs.agents.base import BaseAgent
from hafs.adapters.critique import CritiqueAdapter, CritiqueProvider
from hafs.adapters.buganizer import BuganizerAdapter
from hafs.core.orchestrator import ModelOrchestrator
from hafs.agents.validator import ValidatorAgent

class DailyBriefingAgent(BaseAgent):
    """Chief of Staff. Summarizes recent activity and provides strategic context."""

    def __init__(self):
        super().__init__("DailyBriefing", "Synthesize daily status reports and work stream analysis.")
        self.critique = CritiqueAdapter()
        self.buganizer = BuganizerAdapter()
        self.orchestrator = None
        self.validator = ValidatorAgent()
        self.briefings_dir = Path.home() / ".context" / "background_agent" / "briefings"

    async def setup(self):
        await super().setup()
        api_key = os.getenv("AISTUDIO_API_KEY")
        self.orchestrator = ModelOrchestrator(api_key)
        await self.critique.connect()
        await self.validator.setup()
        self.briefings_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(self, user: Optional[str] = None):
        """Generate a Morning Download report."""
        user = user or os.getenv("USER", "scawful")
        print(f"[{self.name}] Generating Daily Briefing for {user}...")

        # 1. Gather Data from Adapters
        # Last 24h Submitted
        submitted = await self.critique.get_submitted(user, limit=5)
        # Pending Reviews
        pending = await self.critique.get_reviews(user)
        # Active Bugs
        bugs = await self.buganizer.search_bugs(f"assignee:me status:open")

        # 2. Synthesize with Orchestrator (REASONING tier)
        briefing = await self._synthesize_briefing(user, submitted, pending, bugs)

        # 3. Save to briefings directory
        date_str = datetime.now().strftime("%Y%m%d")
        briefing_path = self.briefings_dir / f"briefing_{date_str}.md"
        briefing_path.write_text(briefing)

        # 4. Final Validation Turn
        await self.validator.run_task(briefing_path)
        
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
            data_block += f"- CL {cl.id}: {cl.title}\n"
        
        data_block += "\nPENDING REVIEWS:\n"
        for cl in pending:
            data_block += f"- CL {cl.id}: {cl.title} (Status: {cl.status})\n"
            
        data_block += "\nACTIVE BUGS:\n"
        for b in bugs:
            data_block += f"- b/{b['id']} ({b['priority']}): {b['title']}\n"

        prompt = (
            "You are a Principal Engineer's Chief of Staff. Synthesize the following data into a high-density 'Morning Briefing'.\n"
            "Format your response in professional markdown.\n\n"
            "Sections:\n"
            "1. **Strategic Focus**: What is the singular most important objective for today?\n"
            "2. **Work Stream Updates**: Group the Bugs and CLs into 2-3 logical work streams (e.g., 'Chirp Migration', 'UI Cleanup').\n"
            "3. **Blocking Items**: What needs immediate attention or review?\n"
            "4. **Action Plan**: 3-5 concise bullet points for the user's day.\n\n"
            f"DATA:\n{data_block}"
        )

        try:
            return await self.orchestrator.generate_content(prompt, tier="reasoning")
        except Exception as e:
            return f"# Daily Briefing Failed\nError: {e}"

