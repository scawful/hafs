"""Architect Council: The Synthesis & Planning Pipeline."""
import asyncio
from pathlib import Path
import json

from agents.core.base import BaseAgent
from agents.pipeline.doc_writer import DocWriter
from agents.utility.shell_agent import ShellAgent
from agents.swarm.specialists import SwarmStrategist

class ArchitectCouncil(BaseAgent):
    """Orchestrates the first pipeline: from Dev Prompt to TDD and Plan."""

    def __init__(self, workspace_name: str):
        super().__init__("ArchitectCouncil", "Orchestrates the Synthesis & Planning Pipeline.")
        self.workspace_name = workspace_name
        self.output_dir = Path.home() / "AgentWorkspaces" / workspace_name / ".context" / "planning"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pipeline Agents
        self.strategist = SwarmStrategist()
        workspace_path = Path.home() / "AgentWorkspaces" / workspace_name
        self.shell = ShellAgent(str(workspace_path))
        self.doc_writer = DocWriter()

    async def run_task(self, dev_prompt_path: str) -> str:
        """Execute the Architect pipeline."""
        
        # 1. Read the dev prompt
        dev_prompt = Path(dev_prompt_path).read_text()
        
        # 2. Create the workspace (Idempotent)
        print(f"[{self.name}] Ensuring workspace '{self.workspace_name}' exists...")
        await self.shell.setup()
        
        # 3. Generate the TDD
        print(f"[{self.name}] Generating Technical Design Document...")
        tdd_content = await self.doc_writer.run_task(dev_prompt)
        tdd_path = self.output_dir / "TDD.md"
        tdd_path.write_text(tdd_content)
        print(f"[{self.name}] TDD saved to {tdd_path}")
        
        # 4. Generate the structured plan
        print(f"[{self.name}] Generating structured plan.json...")
        plan_prompt = (
            "Based on the following TDD, create a structured `plan.json` for the 'Builder' pipeline.\n"
            "The plan should include:\n"
            "- A list of `files_to_create` (relative paths).\n"
            "- A list of `files_to_modify` (relative paths).\n"
            "- A list of `build_targets` (e.g., build commands or targets).\n"
            "- A dictionary `instructions` mapping each file path to a detailed, one-paragraph instruction for the `CodeWriter` agent.\n\n"
            f"TDD:\n{tdd_content}\n\n"
            "Output ONLY the raw JSON block."
        )
        plan_json_str = await self.strategist.generate_thought(plan_prompt, topic="Generate Implementation Plan")
        
        plan_path = self.output_dir / "plan.json"
        plan_path.write_text(plan_json_str)
        print(f"[{self.name}] Plan saved to {plan_path}")

        return f"Architect Pipeline complete for '{self.workspace_name}'. TDD and plan.json are ready."
