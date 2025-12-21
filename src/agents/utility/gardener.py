"""Context Gardener Agent (Public Port).

Maintains the knowledge base by pruning stale files and organizing content.
"""

import time
from pathlib import Path
from agents.core.base import BaseAgent

class ContextGardener(BaseAgent):
    """The Librarian. Keeps the knowledge base clean."""

    def __init__(self):
        super().__init__("ContextGardener", "Organize and prune verified context.")
        self.verified_dir = self.knowledge_dir / "verified"

    async def run_task(self):
        if not self.verified_dir.exists(): return "No verified context found."
        
        files = list(self.verified_dir.glob("*.md"))
        now = time.time()
        stale = []
        
        for f in files:
            # Check if older than 30 days
            if now - f.stat().st_mtime > 30 * 86400:
                stale.append(f.name)
                
        prompt = (
            f"You are managing a knowledge base. The following files are >30 days old: {stale}\n"
            "Recommend which ones to ARCHIVE or UPDATE based on their names."
        )
        
        return await self.generate_thought(prompt)

