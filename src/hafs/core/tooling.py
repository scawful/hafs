"""Tooling support for background agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class ToolCommand:
    """Defines a tool command accessible to background agents."""

    name: str
    command: list[str]
    description: str
    category: str = "read"
    timeout: int = 30


@dataclass
class ToolResult:
    """Result of running a tool command."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class ToolProfile:
    """Allow/deny list for tool access."""

    name: str
    allow: set[str]
    deny: set[str]

    def allows(self, tool_name: str) -> bool:
        if tool_name in self.deny:
            return False
        if not self.allow:
            return False
        return tool_name in self.allow


DEFAULT_TOOL_CATALOG: dict[str, ToolCommand] = {
    "rg": ToolCommand(
        name="rg",
        command=["rg", "--no-heading", "--line-number"],
        description="Search text within the project.",
    ),
    "rg_files": ToolCommand(
        name="rg_files",
        command=["rg", "--files"],
        description="List tracked files in the project.",
    ),
    "rg_todos": ToolCommand(
        name="rg_todos",
        command=["rg", "--no-heading", "--line-number", "-e", "TODO|FIXME|HACK"],
        description="Find TODO/FIXME/HACK markers.",
    ),
    "git_status": ToolCommand(
        name="git_status",
        command=["git", "status", "--porcelain"],
        description="Show git working tree status.",
    ),
    "git_branch": ToolCommand(
        name="git_branch",
        command=["git", "rev-parse", "--abbrev-ref", "HEAD"],
        description="Show current git branch.",
    ),
    "git_log": ToolCommand(
        name="git_log",
        command=["git", "log", "-1", "--oneline"],
        description="Show latest git commit.",
    ),
    "git_diff": ToolCommand(
        name="git_diff",
        command=["git", "diff", "--stat"],
        description="Show git diff summary.",
    ),
    "ls": ToolCommand(
        name="ls",
        command=["ls", "-la"],
        description="List files in the project root.",
    ),
    "pytest": ToolCommand(
        name="pytest",
        command=["pytest", "-q"],
        description="Run Python tests.",
        category="test",
        timeout=300,
    ),
    "npm_test": ToolCommand(
        name="npm_test",
        command=["npm", "test"],
        description="Run npm tests.",
        category="test",
        timeout=300,
    ),
    "pnpm_test": ToolCommand(
        name="pnpm_test",
        command=["pnpm", "test"],
        description="Run pnpm tests.",
        category="test",
        timeout=300,
    ),
    "cargo_test": ToolCommand(
        name="cargo_test",
        command=["cargo", "test", "--quiet"],
        description="Run Rust tests.",
        category="test",
        timeout=300,
    ),
    "go_test": ToolCommand(
        name="go_test",
        command=["go", "test", "./..."],
        description="Run Go tests.",
        category="test",
        timeout=300,
    ),
    "make_test": ToolCommand(
        name="make_test",
        command=["make", "test"],
        description="Run make test.",
        category="test",
        timeout=300,
    ),
    "just_test": ToolCommand(
        name="just_test",
        command=["just", "test"],
        description="Run just test.",
        category="test",
        timeout=300,
    ),
}


class ToolRunner:
    """Executes allowed tool commands within a project root."""

    def __init__(
        self,
        root: Path,
        profile: ToolProfile,
        catalog: Optional[dict[str, ToolCommand]] = None,
    ) -> None:
        self.root = root
        self.profile = profile
        self.catalog = catalog or DEFAULT_TOOL_CATALOG

    def available_tools(self) -> list[str]:
        return [name for name in self.catalog if self.profile.allows(name)]

    def _resolve_tool(self, name: str) -> ToolCommand:
        if name not in self.catalog:
            raise ValueError(f"Unknown tool: {name}")
        if not self.profile.allows(name):
            raise PermissionError(f"Tool not allowed by profile: {name}")
        return self.catalog[name]

    async def run(
        self,
        name: str,
        args: Optional[Iterable[str]] = None,
        timeout: Optional[int] = None,
    ) -> ToolResult:
        tool = self._resolve_tool(name)
        cmd = tool.command + list(args or [])
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
            )
        except FileNotFoundError as exc:
            return ToolResult(exit_code=127, stdout="", stderr=str(exc))

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout or tool.timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            return ToolResult(exit_code=124, stdout="", stderr="Tool timed out")

        return ToolResult(
            exit_code=process.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
