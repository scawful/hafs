"""Chronos Agent (Git Version)."""

import asyncio
from hafs.agents.base import BaseAgent

class ChronosAgent(BaseAgent):
    def __init__(self):
        super().__init__("Chronos", "Analyze git history.")

    async def run_task(self, query: str = "1 day") -> str:
        # Heuristic: git log --since="1 day ago"
        cmd = 'git log --since="24 hours ago" --stat'
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, _ = await proc.communicate()
        
        prompt = (
            f"Summarize these recent git changes:\n{out.decode()[:5000]}"
        )
        return await self.generate_thought(prompt)

