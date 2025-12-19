"""Builder Agents (Public Port).

CodeSurgeon, Toolsmith, and Debugger.
"""

import os
import re
import stat
import asyncio
from pathlib import Path
from hafs.agents.base import BaseAgent

class CodeSurgeon(BaseAgent):
    """The Surgeon. Applies patches to files."""
    
    def __init__(self):
        super().__init__("CodeSurgeon", "Apply code changes, refactors, and fixes.")

    async def run_task(self, instruction: str) -> str:
        # Format: "path | instruction"
        if "|" not in instruction: return "Invalid format. Use: path | instruction"
        
        path_str, req = instruction.split("|", 1)
        path = Path(path_str.strip())
        
        if not path.exists(): return f"File not found: {path}"
        
        content = path.read_text()
        prompt = (
            f"You are an expert Software Engineer.\n"
            f"TASK: Apply the following change to the file: {req}\n\n"
            f"FILE CONTENT:\n```\n{content}\n```\n"
            "REQUIREMENTS:\n"
            "1. Output the FULL updated file content.\n"
            "2. Ensure the code block is valid and complete.\n"
            "3. Output strictly the code block."
        )
        
        res = await self.generate_thought(prompt)
        
        match = re.search(r"```(?:\w+)?\n(.*?)```", res, re.DOTALL)
        if match:
            path.write_text(match.group(1))
            return f"Updated {path}"
        return "Failed to generate code."

class Toolsmith(BaseAgent):
    """The Toolsmith. Forges new tools."""
    
    def __init__(self):
        super().__init__("Toolsmith", "Create new python tools.")
        self.tools_dir = self.context_root / "tools"
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    async def run_task(self, instruction: str) -> str:
        # Format: "name | purpose"
        if "|" not in instruction: return "Invalid format."
        name, purpose = instruction.split("|", 1)
        safe_name = name.strip().replace(" ", "_").lower() + ".py"
        
        prompt = (
            f"Write a Python script '{safe_name}' to: {purpose}\n"
            "Output ONLY the code block."
        )
        res = await self.generate_thought(prompt)
        
        match = re.search(r"```python\n(.*?)```", res, re.DOTALL)
        if match:
            path = self.tools_dir / safe_name
            path.write_text(match.group(1))
            os.chmod(path, 0o755)
            return f"Tool created: {path}"
        return "Failed."

class DebuggerAgent(BaseAgent):
    """The Debugger. Runs commands and analyzes output."""
    
    def __init__(self):
        super().__init__("Debugger", "Run commands and analyze errors.")

    async def run_task(self, cmd: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        out, err = await proc.communicate()
        
        if proc.returncode == 0:
            return f"Success:\n{out.decode()[:500]}"
            
        error_log = f"Exit: {proc.returncode}\nErr: {err.decode()[-1000:]}"
        prompt = f"Analyze error and propose fix:\nCMD: {cmd}\n{error_log}"
        return await self.generate_thought(prompt)

