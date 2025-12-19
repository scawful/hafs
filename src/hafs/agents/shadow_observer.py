"""Shadow Observer Agent.

Watches the user's shell history to infer intent and proactively gather context.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from hafs.agents.base import BaseAgent

class ShadowObserver(BaseAgent):
    """The Apprentice. Watches over your shoulder."""

    def __init__(self):
        super().__init__("ShadowObserver", "Watch shell history and react to user actions.")
        self.history_file = Path(os.path.expanduser("~/.zsh_history"))
        self.last_pos = 0

    async def setup(self):
        await super().setup()
        # Move pointer to end of file to start watching new commands
        if self.history_file.exists():
            self.last_pos = self.history_file.stat().st_size

    async def watch_loop(self):
        """Continuous loop to watch history."""
        print(f"[{self.name}] Watching {self.history_file}...")
        while True:
            await self.check_history()
            await asyncio.sleep(2) # Check every 2s

    async def check_history(self):
        if not self.history_file.exists(): return

        current_size = self.history_file.stat().st_size
        if current_size < self.last_pos:
            self.last_pos = 0 # File truncated

        if current_size > self.last_pos:
            with open(self.history_file, "rb") as f:
                f.seek(self.last_pos)
                new_data = f.read()
                self.last_pos = current_size

                # Decode (zsh history can have binary metadata, ignore errors)
                lines = new_data.decode("utf-8", errors="ignore").splitlines()
                for line in lines:
                    await self.process_command(line)

    async def process_command(self, raw_line: str):
        # Zsh history format often: : 1678900000:0;command
        cmd = raw_line
        if ";" in raw_line:
            cmd = raw_line.split(";", 1)[1].strip()

        if not cmd: return

        print(f"[{self.name}] User ran: {cmd}")

        # Directory Change -> Map Context
        if cmd.startswith("cd "):
            path = cmd.split(" ", 1)[1]
            expanded = os.path.expanduser(path)
            # Generic directory mapping could go here
            # Plugins can extend this to handle specific workspace types

        # Git operations -> Could track repository context
        if cmd.startswith("git "):
            # Generic git tracking could go here
            pass

    async def run_task(self):
        # One-off check (mostly for testing)
        await self.check_history()

if __name__ == "__main__":
    agent = ShadowObserver()
    asyncio.run(agent.setup())
    asyncio.run(agent.watch_loop())
