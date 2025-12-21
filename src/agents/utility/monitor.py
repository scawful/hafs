"""Activity Monitor (Public Port of TrendWatcher).

Analyzes git history to find trending topics.
"""

from pathlib import Path
from typing import Optional

from agents.core.base import BaseAgent
from hafs.core.execution import ExecutionPolicy
from hafs.core.projects import ProjectRegistry
from hafs.core.tooling import ToolRunner

class ActivityMonitor(BaseAgent):
    """The Observer. Finds trends in work history."""

    def __init__(
        self,
        workspace_path: str | None = None,
        execution_mode: Optional[str] = None,
    ):
        super().__init__("ActivityMonitor", "Analyze git activity to identify work streams.")
        self.workspace_path = (
            Path(workspace_path).expanduser()
            if workspace_path
            else Path.cwd()
        )
        self.execution_mode = execution_mode
        self._tool_runner: Optional[ToolRunner] = None

    async def setup(self) -> None:
        await super().setup()
        self._tool_runner = self._build_tool_runner()

    def _build_tool_runner(self) -> ToolRunner:
        registry = ProjectRegistry.load()
        project = registry.match_path(self.workspace_path)
        policy = ExecutionPolicy(registry=registry, execution_mode=self.execution_mode)
        tool_profile = policy.resolve_tool_profile(project)
        return ToolRunner(self.workspace_path, tool_profile)

    def _ensure_tool_runner(self) -> ToolRunner:
        if not self._tool_runner:
            self._tool_runner = self._build_tool_runner()
        return self._tool_runner

    async def run_task(self):
        runner = self._ensure_tool_runner()
        try:
            result = await runner.run(
                "git_log",
                args=["--since", "7 days ago", "--oneline"],
            )
            logs = result.stdout if result.exit_code == 0 else ""
        except Exception:
            logs = ""
        if not logs:
            logs = "No git history found."

        prompt = (
            "Analyze these git commit messages from the last 7 days.\n"
            "Identify 3 key 'Themes' or 'Topics' the user is working on.\n\n"
            f"LOGS:\n{logs[:5000]}"
        )
        
        return await self.generate_thought(prompt)
