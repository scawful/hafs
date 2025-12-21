"""Builder Council: The Implementation Pipeline."""
import asyncio
from pathlib import Path
import json

from agents.core.base import BaseAgent
from agents.pipeline.code_writer import CodeWriter
from agents.pipeline.build_test_agents import BuildAgent

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
        # Load Plan
        plan_path = self.planning_dir / "plan.json"
        if not plan_path.exists():
            return "Error: plan.json not found. Run Architect pipeline first."
        
        plan = json.loads(plan_path.read_text())
        tdd_path = self.planning_dir / "TDD.md"
        tdd = tdd_path.read_text() if tdd_path.exists() else ""
        
        instructions = plan.get("instructions", {})
        all_files = plan.get("files_to_create", []) + plan.get("files_to_modify", [])
        
        for file_path_str in all_files:
            if file_path_str not in instructions:
                continue

            # Resolve full path inside the workspace
            full_path = self.workspace_root / file_path_str.lstrip("/")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            print(f"[{self.name}] Writing code for: {full_path}")
            
            code = await self.code_writer.run_task(
                file_path=file_path_str,
                instructions=instructions[file_path_str],
                tdd=tdd
            )
            
            full_path.write_text(code)

        # Build & Verify
        build_targets = plan.get("build_targets", [])
        build_command = f"build {' '.join(build_targets)}" if build_targets else "build"
        
        print(f"[{self.name}] Running build: {build_command}...")
        build_result = await self.build_agent.run_task(build_command)
        
        return f"Builder Pipeline complete for '{self.workspace_name}'. Result: {build_result}"
