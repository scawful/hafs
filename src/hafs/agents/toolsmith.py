"""Toolsmith Agent.

Creates new tools (Python scripts) on demand to extend agent capabilities.
"""

import os
import re
import stat
from pathlib import Path
from hafs.agents.base import BaseAgent

class Toolsmith(BaseAgent):
    """The Toolsmith. Forges new tools."""

    def __init__(self):
        super().__init__("Toolsmith", "Create new python tools/scripts for the swarm.")
        self.tools_dir = self.context_root / "tools"
        self.tools_dir.mkdir(parents=True, exist_ok=True)

    async def create_tool(self, tool_name: str, purpose: str) -> str:
        """Generate and save a new tool."""
        print(f"[{self.name}] Forging tool: {tool_name}...")
        
        safe_name = tool_name.replace(" ", "_").lower()
        if not safe_name.endswith(".py"):
            safe_name += ".py"
        
        prompt = (
            f"Write a standalone Python 3 script named '{safe_name}'.\n"
            f"PURPOSE: {purpose}\n\n"
            "REQUIREMENTS:\n"
            "1. Must be executable (shebang #!/usr/bin/env python3).\n"
            "2. Must have a main block.\n"
            "3. Use standard libraries.\n"
            "4. Output strictly the code block."
        )
        
        response = await self.generate_thought(prompt)
        
        # Extract code
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if not match:
            match = re.search(r"```\n(.*?)```", response, re.DOTALL)
            
        if match:
            code = match.group(1)
            file_path = self.tools_dir / safe_name
            file_path.write_text(code)
            
            # Make executable
            st = os.stat(file_path)
            os.chmod(file_path, st.st_mode | stat.S_IEXEC)
            
            print(f"[{self.name}] Tool created at {file_path}")
            return f"Tool created: {file_path}"
        
        return "Failed to generate tool code."

    async def run_task(self, instruction: str) -> str:
        # Format: "name | purpose"
        if "|" in instruction:
            name, purpose = instruction.split("|", 1)
            return await self.create_tool(name.strip(), purpose.strip())
        return "Invalid format. Use: tool_name | purpose"
