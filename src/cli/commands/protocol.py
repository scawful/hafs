"""Cognitive protocol CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from config.loader import load_config
from core.afs.discovery import find_context_root
from core.protocol.lint import lint_protocol

protocol_app = typer.Typer(
    name="protocol",
    help="Validate and inspect cognitive protocol artifacts",
)
console = Console()


@protocol_app.callback(invoke_without_command=True)
def protocol_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


def _resolve_context_root(path: Optional[Path], *, agent_workspaces_dir: Path) -> Path | None:
    if path:
        candidate = path.expanduser().resolve()
        if candidate.is_file():
            candidate = candidate.parent
        if candidate.name == ".context":
            return candidate
        direct = candidate / ".context"
        if direct.exists() and direct.is_dir():
            return direct
        return find_context_root(candidate, agent_workspaces_dir=agent_workspaces_dir)
    return find_context_root(agent_workspaces_dir=agent_workspaces_dir)


def _format_path(path: Path | None, base: Path) -> str:
    if path is None:
        return "-"
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


@protocol_app.command("lint")
def lint(
    path: Optional[Path] = typer.Option(
        None, "--path", "-p", help="Project root or .context directory"
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Treat warnings as failures"
    ),
) -> None:
    """Lint cognitive protocol files for compliance issues."""
    config = load_config()
    context_root = _resolve_context_root(
        path, agent_workspaces_dir=config.general.agent_workspaces_dir
    )
    if not context_root:
        console.print("[red]No .context found for linting.[/red]")
        raise typer.Exit(1)

    report = lint_protocol(context_root, config)
    if not report.issues:
        console.print("[green]OK[/green] Protocol lint clean")
        return

    table = Table(title="Protocol Lint", show_lines=True)
    table.add_column("Severity", style="bold")
    table.add_column("Code", style="cyan")
    table.add_column("Message")
    table.add_column("Path", style="dim")

    for issue in report.issues:
        message = issue.message
        if issue.hint:
            message = f"{message}\nHint: {issue.hint}"
        table.add_row(
            issue.severity.upper(),
            issue.code,
            message,
            _format_path(issue.path, context_root),
        )

    console.print(table)
    console.print(
        f"[bold]Errors:[/bold] {len(report.errors)}  "
        f"[bold]Warnings:[/bold] {len(report.warnings)}"
    )

    if report.errors or (strict and report.warnings):
        raise typer.Exit(1)
