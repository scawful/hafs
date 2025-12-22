"""Tooling support for background agents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Iterable, Optional, Callable, Awaitable
import json
import shlex


@dataclass(frozen=True)
class ToolCommand:
    """Defines a tool command accessible to background agents."""

    name: str
    command: list[str]
    description: str
    category: str = "read"
    timeout: int = 30
    aliases: list[str] = field(default_factory=list)


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
    requires_confirmation: set[str] = field(default_factory=set)

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
        aliases=["search"],
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
        aliases=["status"],
    ),
    "git_branch": ToolCommand(
        name="git_branch",
        command=["git", "rev-parse", "--abbrev-ref", "HEAD"],
        description="Show current git branch.",
    ),
    "git_log": ToolCommand(
        name="git_log",
        command=["git", "log"],
        description="Show git log (args optional).",
    ),
    "git_diff": ToolCommand(
        name="git_diff",
        command=["git", "diff", "--stat"],
        description="Show git diff summary.",
    ),
    "git_add": ToolCommand(
        name="git_add",
        command=["git", "add"],
        description="Stage changes for commit.",
        category="write",
        aliases=["stage"],
    ),
    "git_commit": ToolCommand(
        name="git_commit",
        command=["git", "commit", "-m"],
        description="Create a git commit.",
        category="write",
        aliases=["commit"],
    ),
    "ls": ToolCommand(
        name="ls",
        command=["ls", "-la"],
        description="List files in the project root.",
    ),
    "uname": ToolCommand(
        name="uname",
        command=["uname", "-a"],
        description="Show system information.",
    ),
    "whoami": ToolCommand(
        name="whoami",
        command=["whoami"],
        description="Show current user.",
    ),
    "uptime": ToolCommand(
        name="uptime",
        command=["uptime"],
        description="Show system uptime and load.",
    ),
    "df": ToolCommand(
        name="df",
        command=["df", "-h"],
        description="Show disk usage.",
    ),
    "du": ToolCommand(
        name="du",
        command=["du", "-sh"],
        description="Show directory size summary.",
    ),
    "ps": ToolCommand(
        name="ps",
        command=["ps", "aux"],
        description="List running processes.",
    ),
    "lsof": ToolCommand(
        name="lsof",
        command=["lsof", "-n", "-P"],
        description="List open files and sockets.",
    ),
    "tail": ToolCommand(
        name="tail",
        command=["tail", "-n", "200"],
        description="Tail a file (default 200 lines).",
    ),
    "journalctl": ToolCommand(
        name="journalctl",
        command=["journalctl", "--user", "-n", "200"],
        description="Show systemd user logs (last 200 lines).",
    ),
    "log_show": ToolCommand(
        name="log_show",
        command=["log", "show", "--last", "1h", "--style", "compact"],
        description="Show macOS unified logs (last hour).",
    ),
    "launchctl": ToolCommand(
        name="launchctl",
        command=["launchctl"],
        description="Manage launchd services.",
        category="deploy",
        timeout=120,
    ),
    "systemctl": ToolCommand(
        name="systemctl",
        command=["systemctl", "--user"],
        description="Manage systemd user services.",
        category="deploy",
        timeout=120,
    ),
    "docker": ToolCommand(
        name="docker",
        command=["docker"],
        description="Run docker commands.",
        category="deploy",
        timeout=300,
    ),
    "docker_compose": ToolCommand(
        name="docker_compose",
        command=["docker", "compose"],
        description="Run docker compose commands.",
        category="deploy",
        timeout=300,
        aliases=["compose"],
    ),
    "kubectl": ToolCommand(
        name="kubectl",
        command=["kubectl"],
        description="Run kubectl commands.",
        category="deploy",
        timeout=300,
    ),
    "ssh": ToolCommand(
        name="ssh",
        command=["ssh"],
        description="Open SSH connections.",
        category="deploy",
        timeout=120,
    ),
    "scp": ToolCommand(
        name="scp",
        command=["scp"],
        description="Copy files over SSH.",
        category="deploy",
        timeout=300,
    ),
    "rsync": ToolCommand(
        name="rsync",
        command=["rsync", "-av"],
        description="Sync files with rsync (archive + verbose).",
        category="deploy",
        timeout=300,
    ),
    "curl": ToolCommand(
        name="curl",
        command=["curl", "-fsSL"],
        description="Fetch URLs with curl.",
        category="deploy",
        timeout=120,
    ),
    "ping": ToolCommand(
        name="ping",
        command=["ping", "-c", "4"],
        description="Ping a host (4 packets).",
        category="deploy",
        timeout=120,
    ),
    "pytest": ToolCommand(
        name="pytest",
        command=["pytest", "-q"],
        description="Run Python tests.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "npm_test": ToolCommand(
        name="npm_test",
        command=["npm", "test"],
        description="Run npm tests.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "pnpm_test": ToolCommand(
        name="pnpm_test",
        command=["pnpm", "test"],
        description="Run pnpm tests.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "cargo_test": ToolCommand(
        name="cargo_test",
        command=["cargo", "test", "--quiet"],
        description="Run Rust tests.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "go_test": ToolCommand(
        name="go_test",
        command=["go", "test", "./..."],
        description="Run Go tests.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "make_test": ToolCommand(
        name="make_test",
        command=["make", "test"],
        description="Run make test.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "just_test": ToolCommand(
        name="just_test",
        command=["just", "test"],
        description="Run just test.",
        category="test",
        timeout=300,
        aliases=["test"],
    ),
    "npm_build": ToolCommand(
        name="npm_build",
        command=["npm", "run", "build"],
        description="Run npm build.",
        category="build",
        timeout=300,
        aliases=["build"],
    ),
    "pnpm_build": ToolCommand(
        name="pnpm_build",
        command=["pnpm", "build"],
        description="Run pnpm build.",
        category="build",
        timeout=300,
        aliases=["build"],
    ),
    "cargo_build": ToolCommand(
        name="cargo_build",
        command=["cargo", "build", "--quiet"],
        description="Run cargo build.",
        category="build",
        timeout=300,
        aliases=["build"],
    ),
    "go_build": ToolCommand(
        name="go_build",
        command=["go", "build", "./..."],
        description="Run go build.",
        category="build",
        timeout=300,
        aliases=["build"],
    ),
    "make_build": ToolCommand(
        name="make_build",
        command=["make", "build"],
        description="Run make build.",
        category="build",
        timeout=300,
        aliases=["build"],
    ),
    "just_build": ToolCommand(
        name="just_build",
        command=["just", "build"],
        description="Run just build.",
        category="build",
        timeout=300,
        aliases=["build"],
    ),
}


class ToolRunner:
    """Executes allowed tool commands within a project root."""

    def __init__(
        self,
        root: Path,
        profile: ToolProfile,
        catalog: Optional[dict[str, ToolCommand]] = None,
        confirmation_callback: Optional[Callable[[ToolCommand], Awaitable[bool]]] = None,
    ) -> None:
        self.root = root
        self.profile = profile
        self.catalog = catalog or DEFAULT_TOOL_CATALOG
        self._alias_map = self._build_alias_map()
        self.confirmation_callback = confirmation_callback

    def available_tools(self) -> list[str]:
        return [name for name in self.catalog if self.profile.allows(name)]

    def _build_alias_map(self) -> dict[str, list[str]]:
        alias_map: dict[str, list[str]] = {}
        for name, tool in self.catalog.items():
            for alias in tool.aliases:
                alias_map.setdefault(alias, []).append(name)
        return alias_map

    def _resolve_tool(self, name: str) -> ToolCommand:
        tool = self.catalog.get(name)
        if tool:
            self._enforce_policy(tool)
            if not self.profile.allows(name):
                raise PermissionError(f"Tool not allowed by profile: {name}")
            return tool

        candidates = self._alias_map.get(name, [])
        if candidates:
            selected = self._select_candidate(name, candidates)
            if selected and self.profile.allows(selected):
                tool = self.catalog[selected]
                self._enforce_policy(tool)
                return tool
            for candidate in candidates:
                if self.profile.allows(candidate):
                    tool = self.catalog[candidate]
                    self._enforce_policy(tool)
                    return tool

        if candidates:
            raise PermissionError(f"Tool not allowed by profile: {name}")
        raise ValueError(f"Unknown tool: {name}")

    def _enforce_policy(self, tool: ToolCommand) -> None:
        context_root = self.root / ".context"
        metadata_path = context_root / "metadata.json"
        if not metadata_path.exists():
            return
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            policy = metadata.get("policy", {})
            executable = set(policy.get("executable", []))
            if tool.category in {"write", "build", "test", "deploy", "shell"}:
                directories = metadata.get("directories", {})
                tools_dir = directories.get("tools", "tools")
                if "tools" not in executable and tools_dir not in executable:
                    raise PermissionError("AFS policy disallows tool execution")
        except PermissionError:
            raise
        except Exception:
            return

    async def run_command_line(
        self,
        command_line: str,
        timeout: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ToolResult:
        parts = shlex.split(command_line)
        if not parts:
            return ToolResult(exit_code=1, stdout="", stderr="No command provided")
        name, args = parts[0], parts[1:]
        return await self.run(name, args=args, timeout=timeout, env=env)

    def _select_candidate(self, alias: str, candidates: list[str]) -> Optional[str]:
        if alias not in {"build", "test"}:
            return None

        checks: list[tuple[Path, str]] = [
            (self.root / "pnpm-lock.yaml", "pnpm"),
            (self.root / "package-lock.json", "npm"),
            (self.root / "package.json", "npm"),
            (self.root / "Cargo.toml", "cargo"),
            (self.root / "go.mod", "go"),
            (self.root / "Makefile", "make"),
            (self.root / "justfile", "just"),
            (self.root / "Justfile", "just"),
        ]

        for path, prefix in checks:
            if not path.exists():
                continue
            tool_name = f"{prefix}_{alias}"
            if tool_name in candidates and self.profile.allows(tool_name):
                return tool_name
        return None

    async def run(
        self,
        name: str,
        args: Optional[Iterable[str]] = None,
        timeout: Optional[int] = None,
        env: Optional[dict[str, str]] = None,
    ) -> ToolResult:
        tool = self._resolve_tool(name)

        # Check confirmation
        if tool.category in self.profile.requires_confirmation:
            if self.confirmation_callback:
                approved = await self.confirmation_callback(tool)
                if not approved:
                    return ToolResult(
                        exit_code=1, stdout="", stderr=f"Execution of '{name}' denied by user."
                    )
            else:
                return ToolResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"Execution of '{name}' requires confirmation (no callback provided).",
                )

        cmd = tool.command + list(args or [])
        env_vars = os.environ.copy()
        if env:
            env_vars.update(env)
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.root),
                env=env_vars,
            )
        except FileNotFoundError as exc:
            return ToolResult(exit_code=127, stdout="", stderr=str(exc))

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout or tool.timeout,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            return ToolResult(exit_code=124, stdout="", stderr="Tool timed out")

        return ToolResult(
            exit_code=process.returncode if process.returncode is not None else 1,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )
