"""Generate curiosity-driven exploration prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from core.execution import ExecutionPolicy
from core.projects import ProjectRegistry
from core.tooling import ToolRunner


class CuriosityExplorerAgent(MemoryAwareAgent):
    """Generate curiosity-driven exploration prompts."""

    def __init__(self, workspace_path: Optional[str] = None, execution_mode: Optional[str] = None):
        super().__init__("CuriosityExplorer", "Identify curiosity-driven exploration topics.")
        self.workspace_path = (
            Path(workspace_path).expanduser()
            if workspace_path
            else Path.cwd()
        )
        self.execution_mode = execution_mode
        self._tool_runner: Optional[ToolRunner] = None
        self.model_tier = "fast"

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

    async def _safe_run(self, tool_name: str, args: Optional[list[str]] = None) -> str:
        runner = self._ensure_tool_runner()
        try:
            result = await runner.run(tool_name, args=args or [])
            if result.ok and result.stdout:
                return result.stdout
        except Exception:
            pass
        return ""

    async def run_task(self) -> LoopReport:
        git_log = await self._safe_run("git_log", ["--oneline", "-n", "20"])
        todos = await self._safe_run("rg_todos")

        fallback_lines = []
        if git_log:
            fallback_lines.append("Recent commits:\n" + git_log[:2000])
        if todos:
            fallback_lines.append("TODO markers:\n" + todos[:2000])
        fallback_context = "\n\n".join(fallback_lines).strip() or "No recent signals found."

        prompt = (
            "You are the Curiosity Explorer loop.\n"
            "Based on the signals below, propose 3 exploration topics.\n"
            "Each topic should include: question, suggested file/module, and why it's interesting.\n\n"
            f"SIGNALS:\n{fallback_context}"
        )
        exploration = await self.generate_thought(prompt)
        if not exploration or exploration.startswith("Error in generate_thought"):
            exploration = fallback_context

        await self.remember(
            content=exploration[:500],
            memory_type="insight",
            context={"signals": fallback_context[:2000]},
            importance=0.4,
        )

        metrics = {
            "git_commits_analyzed": len(git_log.splitlines()) if git_log else 0,
            "todos_found": len(todos.splitlines()) if todos else 0,
            "topics_generated": exploration.count("question") if exploration else 0,
        }

        return LoopReport(
            title="Curiosity Exploration Ideas",
            body=exploration,
            tags=["curiosity", "exploration"],
            metrics=metrics,
        )
