"""Builder Council: The Implementation Pipeline."""
import asyncio
from pathlib import Path
import json

from hafs.agents.base import BaseAgent
from hafs.agents.pipeline.code_writer import CodeWriter
from hafs.agents.pipeline.build_test_agents import BuildAgent

class BuilderCouncil(BaseAgent):
    """Orchestrates the second pipeline: from Plan to Building Code."""

    def __init__(self, workspace_name: str):
        super().__init__("BuilderCouncil", "Orchestrates the Implementation Pipeline.")
        self.workspace_name = workspace_name
        self.planning_dir = Path.home() / "AgentWorkspaces" / workspace_name / ".context" / "planning"
        self.workspace_root = Path.home() / "AgentWorkspaces" / workspace_name 

        # Pipeline Agents
        self.code_writer = CodeWriter()
        self.build_agent = BuildAgent(str(self.workspace_root))

    async def run_task(self) -> str:
        # ... (load plan) ...
        
        for file_path_str in all_files:
            if file_path_str not in instructions:
                continue

            # Resolve full path inside the workspace
            full_path = self.workspace_root / file_path_str.lstrip("/")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[{self.name}] Writing code for: {full_path}")
            
            # ... (rest of logic) ...

