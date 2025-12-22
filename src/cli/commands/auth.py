import os
import shlex
import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console

from config.loader import load_config

auth_app = typer.Typer(
    name="auth",
    help="Authenticate external CLI backends (Claude, etc.)",
)
console = Console()


def _resolve_claude_command() -> tuple[list[str], dict[str, str], Optional[str]]:
    config = load_config()
    backend = config.get_backend_config("claude")

    command = ["claude"]
    env = os.environ.copy()
    cwd = None

    if backend:
        if backend.command:
            command = [str(part) for part in backend.command]
        if backend.env:
            env.update({key: str(value) for key, value in backend.env.items()})
        if backend.working_dir:
            cwd = str(backend.working_dir)

    return command, env, cwd


@auth_app.command("claude")
def claude_auth(
    action: str = typer.Argument(
        "setup-token",
        help="Claude CLI command to run (default: setup-token)",
    ),
    command: Optional[str] = typer.Option(
        None,
        "--command",
        "-c",
        help="Override the Claude CLI command (defaults to backends.claude.command)",
    ),
    extra_args: list[str] = typer.Option(
        [],
        "--arg",
        "-a",
        help="Extra arguments to pass to the Claude CLI (repeatable)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the command and exit without running it",
    ),
) -> None:
    """Run Claude CLI authentication helpers (OAuth/token setup)."""
    base_cmd, env, cwd = _resolve_claude_command()
    if command:
        base_cmd = [command] + base_cmd[1:]

    full_cmd = base_cmd + ([action] if action else []) + extra_args

    if dry_run:
        console.print(" ".join(shlex.quote(part) for part in full_cmd))
        return

    if action == "setup-token" and (not sys.stdin.isatty() or not sys.stdout.isatty()):
        console.print(
            "[red]Claude setup-token requires an interactive TTY. Run this from your terminal.[/red]"
        )
        raise typer.Exit(1)

    try:
        result = subprocess.run(full_cmd, env=env, cwd=cwd)
    except FileNotFoundError:
        console.print("[red]Claude CLI not found. Install it or update backends.claude.command.[/red]")
        raise typer.Exit(1)

    if result.returncode != 0:
        console.print(f"[red]Claude auth command failed (exit {result.returncode}).[/red]")
        raise typer.Exit(result.returncode)
