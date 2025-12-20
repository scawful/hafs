"""Shadow Observer Agent.

Watches the user's shell history to infer intent and proactively gather context.
"""

import asyncio
import json
import os
from pathlib import Path

from hafs.agents.base import BaseAgent
from hafs.core.history import AgentMemoryManager

class ShadowObserver(BaseAgent):
    """The Apprentice. Watches over your shoulder."""

    def __init__(self):
        super().__init__("ShadowObserver", "Watch shell history and react to user actions.")
        self.history_file = Path(os.path.expanduser("~/.zsh_history"))
        self.last_pos = 0
        self._memory_manager: AgentMemoryManager | None = None
        self._state_file = self.context_root / "autonomy_daemon" / "shadow_observer_state.json"

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self.last_pos = data.get("last_pos", 0)
            except Exception:
                self.last_pos = 0

    def _save_state(self) -> None:
        """Persist state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(json.dumps({"last_pos": self.last_pos}))
        except Exception:
            pass

    async def setup(self):
        await super().setup()
        self._memory_manager = AgentMemoryManager(self.context_root)
        # Load persisted position or start from current end of file
        self._load_state()
        if self.last_pos == 0 and self.history_file.exists():
            self.last_pos = self.history_file.stat().st_size

    async def watch_loop(self):
        """Continuous loop to watch history."""
        print(f"[{self.name}] Watching {self.history_file}...")
        while True:
            await self.check_history()
            await asyncio.sleep(2) # Check every 2s

    async def check_history(self) -> int:
        if not self.history_file.exists():
            return 0

        current_size = self.history_file.stat().st_size
        if current_size < self.last_pos:
            self.last_pos = 0  # File truncated

        if current_size > self.last_pos:
            with open(self.history_file, "rb") as f:
                f.seek(self.last_pos)
                new_data = f.read()
                self.last_pos = current_size

                # Decode (zsh history can have binary metadata, ignore errors)
                lines = new_data.decode("utf-8", errors="ignore").splitlines()
                processed = 0
                for line in lines:
                    await self.process_command(line)
                    processed += 1

                # Persist position after processing
                self._save_state()
                return processed
        return 0

    async def process_command(self, raw_line: str):
        # Zsh history format often: : 1678900000:0;command
        cmd = raw_line
        if ";" in raw_line:
            cmd = raw_line.split(";", 1)[1].strip()

        if not cmd: return

        print(f"[{self.name}] User ran: {cmd}")
        await self._remember_command(cmd)

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

    async def _remember_command(self, cmd: str) -> None:
        """Store observed shell commands into agent memory."""
        if not self._memory_manager:
            return
        try:
            memory = self._memory_manager.get_agent_memory(self.name)
            await memory.remember(
                content=f"User command: {cmd}",
                memory_type="interaction",
                context={"command": cmd},
                importance=0.3,
            )
        except Exception:
            return

    async def run_task(self):
        # One-off check (mostly for testing)
        await self.check_history()

if __name__ == "__main__":
    agent = ShadowObserver()
    asyncio.run(agent.setup())
    asyncio.run(agent.watch_loop())
