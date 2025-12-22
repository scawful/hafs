"""Shell Agent for running commands in a workspace."""

from pathlib import Path
from typing import Optional, Tuple

from agents.core.base import BaseAgent
from core.execution import ExecutionPolicy
from core.projects import ProjectRegistry
from core.tooling import ToolRunner


class ShellAgent(BaseAgent):
    """An agent that can execute shell commands in a local workspace."""

    def __init__(
        self,
        workspace_path: str,
        execution_mode: Optional[str] = None,
        persona: Optional[str] = None,
    ):
        super().__init__("ShellAgent", "Executes shell commands in a workspace.")
        self.workspace_path = Path(workspace_path).expanduser()
        self.execution_mode = execution_mode
        self.persona = persona
        self._tool_runner: Optional[ToolRunner] = None

    async def setup(self):
        await super().setup()
        if not self.workspace_path.exists():
            self.workspace_path.mkdir(parents=True, exist_ok=True)
        self._tool_runner = self._build_tool_runner()

    def _build_tool_runner(self) -> ToolRunner:
        registry = ProjectRegistry.load()
        project = registry.match_path(self.workspace_path)
        policy = ExecutionPolicy(
            registry=registry,
            execution_mode=self.execution_mode,
            persona=self.persona,
        )
        tool_profile = policy.resolve_tool_profile(project)
        return ToolRunner(self.workspace_path, tool_profile)

    def _ensure_tool_runner(self) -> ToolRunner:
        if not self._tool_runner:
            self._tool_runner = self._build_tool_runner()
        return self._tool_runner

    async def run_command(self, command: str, timeout: int = 60) -> Tuple[int, str, str]:
        """Run a shell command in the workspace."""
        runner = self._ensure_tool_runner()
        try:
            result = await runner.run_command_line(command, timeout=timeout)
            return result.exit_code, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)

    async def run_tool(self, name: str, args: Optional[list[str]] = None) -> Tuple[int, str, str]:
        """Run an allowed tool by name."""
        runner = self._ensure_tool_runner()
        try:
            result = await runner.run(name, args=args)
            return result.exit_code, result.stdout, result.stderr
        except Exception as e:
            return -1, "", str(e)

    async def run_task(self, command: str) -> str:
        code, out, err = await self.run_command(command)
        if code == 0:
            return f"Command Success:\n{out}"
        return f"Command Failed (Exit {code}):\n{err}"
