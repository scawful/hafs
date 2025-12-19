"""Activity Monitor (Public Port of TrendWatcher).

Analyzes git history to find trending topics.
"""

import asyncio
from hafs.agents.base import BaseAgent

class ActivityMonitor(BaseAgent):
    """The Observer. Finds trends in work history."""

    def __init__(self):
        super().__init__("ActivityMonitor", "Analyze git activity to identify work streams.")

    async def run_task(self):
        # Scan git logs
        try:
            proc = await asyncio.create_subprocess_shell(
                "git log --since='7 days ago' --oneline",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            out, _ = await proc.communicate()
            logs = out.decode()
        except:
            logs = "No git history found."
            
        prompt = (
            "Analyze these git commit messages from the last 7 days.\n"
            "Identify 3 key 'Themes' or 'Topics' the user is working on.\n\n"
            f"LOGS:\n{logs[:5000]}"
        )
        
        return await self.generate_thought(prompt)

