"""Chronos Agent (Git Version)."""

from pathlib import Path
from typing import Optional

from hafs.agents.base import BaseAgent
from hafs.core.execution import ExecutionPolicy
from hafs.core.projects import ProjectRegistry
from hafs.core.tooling import ToolRunner

class ChronosAgent(BaseAgent):
    def __init__(
        self,
        workspace_path: str | None = None,
        execution_mode: Optional[str] = None,
    ):
        super().__init__("Chronos", "Analyze git history.")
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

    @staticmethod
    def _normalize_since(query: str) -> str:
        cleaned = (query or "").strip()
        if not cleaned:
            return "24 hours ago"
        if "ago" in cleaned:
            return cleaned
        return f"{cleaned} ago"

    async def run_task(self, query: str = "1 day") -> str:
        runner = self._ensure_tool_runner()
        since = self._normalize_since(query)
        try:
            result = await runner.run("git_log", args=["--since", since, "--stat"])
            logs = result.stdout if result.exit_code == 0 else ""
        except Exception:
            logs = ""
        if not logs:
            logs = "No git history found."

        prompt = f"Summarize these recent git changes:\n{logs[:5000]}"
        return await self.generate_thought(prompt)
