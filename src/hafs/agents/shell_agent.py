"""Generic Shell Agent for public HAFS."""
import asyncio
from pathlib import Path
from typing import Dict, Tuple, Optional
from hafs.agents.base import BaseAgent

class ShellAgent(BaseAgent):
    """An agent that can execute shell commands in a local workspace."""

    def __init__(self, workspace_path: str):
        super().__init__("ShellAgent", "Executes shell commands in a workspace.")
        self.workspace_path = Path(workspace_path).expanduser()

    async def setup(self):
        await super().setup()
        if not self.workspace_path.exists():
             self.workspace_path.mkdir(parents=True, exist_ok=True)

    async def run_command(self, command: str, timeout: int = 60) -> Tuple[int, str, str]:
        """Run a shell command in the workspace."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path)
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            return process.returncode, stdout.decode(errors='replace'), stderr.decode(errors='replace')
        except Exception as e:
            return -1, "", str(e)

    async def run_task(self, command: str) -> str:
        code, out, err = await self.run_command(command)
        if code == 0:
            return f"Command Success:\n{out}"
        return f"Command Failed (Exit {code}):\n{err}"
