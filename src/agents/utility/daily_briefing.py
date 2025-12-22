"""Daily Briefing Agent.

Synthesizes daily status reports and work stream analysis.
Adapters are loaded from the plugin registry.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import List, Optional
from pathlib import Path

from adapters.helpers import get_reviews, get_submitted_reviews, search_issues
from agents.core.base import BaseAgent
from core.orchestrator import ModelOrchestrator
from core.config import BRIEFINGS_DIR

logger = logging.getLogger(__name__)

# Generic adapter interface for fallback
class GenericAdapter:
    """Placeholder adapter when no plugin provides a real implementation."""
    async def connect(self): pass
    async def disconnect(self): pass
    async def get_submitted(self, user: str, limit: int = 5) -> List: return []
    async def get_reviews(self, user: str = None) -> List: return []
    async def search_issues(self, query: str, limit: int = 50) -> List: return []
    async def search_bugs(self, query: str) -> List: return []

class DailyBriefingAgent(BaseAgent):
    """Chief of Staff. Summarizes recent activity and provides strategic context.

    Adapters are loaded from the plugin registry at setup time.
    Register adapters with these names:
    - issue_tracker: For bug/issue tracking
    - code_review: For code review systems
    """

    def __init__(self):
        super().__init__("DailyBriefing", "Synthesize daily status reports and work stream analysis.")
        # Adapters (will be loaded from registry in setup)
        self.code_review = None
        self.issue_tracker = None
        self.orchestrator = None
        self.briefings_dir = BRIEFINGS_DIR

    async def setup(self):
        await super().setup()

        # Load adapters from registry with fallback to stubs
        try:
            from core.registry import agent_registry
            self.code_review = agent_registry.get_adapter("code_review")
            await self.code_review.connect()
            logger.info("code_review adapter loaded from registry")
        except Exception as e:
            logger.debug(f"code_review not available: {e}")
            self.code_review = GenericAdapter()

        try:
            from core.registry import agent_registry
            self.issue_tracker = agent_registry.get_adapter("issue_tracker")
            await self.issue_tracker.connect()
            logger.info("issue_tracker adapter loaded from registry")
        except Exception as e:
            logger.debug(f"issue_tracker not available: {e}")
            self.issue_tracker = GenericAdapter()

        # Initialize model orchestrator
        api_key = os.getenv("AISTUDIO_API_KEY")
        self.orchestrator = ModelOrchestrator(api_key)
        self.briefings_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(self, user: Optional[str] = None):
        """Generate a Morning Download report."""
        user = user or os.getenv("USER", "unknown")
        print(f"[{self.name}] Generating Daily Briefing for {user}...")

        # Gather Data from Adapters
        submitted = []
        pending = []
        bugs = []

        try:
            submitted = await get_submitted_reviews(self.code_review, user, limit=5)
        except Exception as e:
            logger.warning(f"Failed to fetch submitted reviews: {e}")

        try:
            pending = await get_reviews(self.code_review, user)
        except Exception as e:
            logger.warning(f"Failed to fetch pending reviews: {e}")

        try:
            bugs = await search_issues(self.issue_tracker, "assignee:me status:open")
        except Exception as e:
            logger.warning(f"Failed to fetch issues: {e}")

        # Synthesize with Orchestrator
        briefing = await self._synthesize_briefing(user, submitted, pending, bugs)

        # Save to briefings directory
        date_str = datetime.now().strftime("%Y%m%d")
        briefing_path = self.briefings_dir / f"briefing_{date_str}.md"

        try:
            briefing_path.write_text(briefing)
            # Link as latest
            latest_path = self.briefings_dir / "latest.md"
            latest_path.write_text(briefing)
            print(f"[{self.name}] Briefing complete: {briefing_path}")
        except Exception as e:
            logger.error(f"Failed to save briefing: {e}")

        return briefing

    async def _synthesize_briefing(self, user: str, submitted, pending, bugs) -> str:
        date_str = datetime.now().strftime("%B %d, %Y")

        data_block = f"USER: {user}\nDATE: {date_str}\n\n"
        data_block += "RECENTLY SUBMITTED (24h):\n"
        for review in submitted:
            data_block += f"- Review {getattr(review, 'id', 'unknown')}: {getattr(review, 'title', 'Untitled')}\n"

        data_block += "\nPENDING REVIEWS:\n"
        for review in pending:
            data_block += (
                f"- Review {getattr(review, 'id', 'unknown')}: "
                f"{getattr(review, 'title', 'Untitled')} "
                f"(Status: {getattr(review, 'status', 'unknown')})\n"
            )

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
            logger.error(f"Briefing synthesis failed: {e}")
            return f"# Daily Briefing Failed\nError: {e}"
