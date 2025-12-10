"""HAFS command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree

from hafs.config.loader import load_config
from hafs.core.afs.manager import AFSManager
from hafs.core.afs.discovery import discover_projects, find_context_root
from hafs.models.afs import MountType

app = typer.Typer(
    name="hafs",
    help="HAFS - Halext Agentic File System",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Project path to initialize"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing AFS"),
) -> None:
    """Initialize AFS in a directory."""
    config = load_config()
    manager = AFSManager(config)

    try:
        root = manager.init(path, force=force)
        console.print(f"[green]Initialized AFS at {root.path}[/green]")
        console.print(f"  Project: {root.project_name}")
        console.print(f"  Directories: {', '.join(mt.value for mt in MountType)}")
    except FileExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Use --force to overwrite existing AFS")
        raise typer.Exit(1)


@app.command()
def mount(
    source: Path = typer.Argument(..., help="Source file or directory to mount"),
    mount_type: str = typer.Argument(
        ..., help="Mount type: memory|knowledge|tools|scratchpad|history"
    ),
    alias: Optional[str] = typer.Option(
        None, "--alias", "-a", help="Custom name for the mount"
    ),
) -> None:
    """Mount a resource into AFS."""
    config = load_config()
    manager = AFSManager(config)

    try:
        mt = MountType(mount_type)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid mount type '{mount_type}'")
        console.print(f"Valid types: {', '.join(mt.value for mt in MountType)}")
        raise typer.Exit(1)

    try:
        mount_point = manager.mount(source, mt, alias)
        console.print(
            f"[green]Mounted {source} -> "
            f"{mount_point.mount_type.value}/{mount_point.name}[/green]"
        )
    except (FileNotFoundError, FileExistsError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def unmount(
    alias: str = typer.Argument(..., help="Name of the mount to remove"),
    mount_type: str = typer.Argument(
        ..., help="Mount type: memory|knowledge|tools|scratchpad|history"
    ),
) -> None:
    """Remove a mount from AFS."""
    config = load_config()
    manager = AFSManager(config)

    try:
        mt = MountType(mount_type)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid mount type '{mount_type}'")
        raise typer.Exit(1)

    if manager.unmount(alias, mt):
        console.print(f"[green]Unmounted {mt.value}/{alias}[/green]")
    else:
        console.print(f"[yellow]Mount '{alias}' not found in {mt.value}[/yellow]")


@app.command("list")
def list_context() -> None:
    """List current AFS structure."""
    config = load_config()
    manager = AFSManager(config)

    try:
        root = manager.list()

        # Create tree view
        tree = Tree(f"[bold purple]{root.project_name}[/bold purple] (.context)")

        for mt in MountType:
            mounts = root.mounts.get(mt, [])
            dir_config = config.get_directory_config(mt.value)
            policy_str = dir_config.policy.value if dir_config else "read_only"

            # Color based on policy
            policy_color = {
                "read_only": "blue",
                "writable": "green",
                "executable": "red",
            }.get(policy_str, "white")

            branch = tree.add(
                f"[{policy_color}]{mt.value}[/{policy_color}] "
                f"[dim]({policy_str})[/dim]"
            )

            if mounts:
                for mount in mounts:
                    link_indicator = "→" if mount.is_symlink else "·"
                    branch.add(f"{mount.name} {link_indicator} {mount.source}")
            else:
                branch.add("[dim](empty)[/dim]")

        console.print(tree)
        console.print(f"\n[dim]Path: {root.path}[/dim]")
        console.print(f"[dim]Total mounts: {root.total_mounts}[/dim]")

    except FileNotFoundError:
        console.print("[yellow]No AFS initialized. Run 'hafs init' first.[/yellow]")


@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Remove AFS from current directory."""
    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS found in current directory tree.[/yellow]")
        raise typer.Exit(0)

    if not force:
        confirm = typer.confirm(
            f"Remove AFS at {context_root}? This cannot be undone."
        )
        if not confirm:
            raise typer.Abort()

    config = load_config()
    manager = AFSManager(config)
    manager.clean(context_root)
    console.print("[green]AFS removed.[/green]")


@app.command()
def projects(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show mount details"),
) -> None:
    """List all discovered AFS-enabled projects."""
    config = load_config()

    # Use tracked projects from config if available, otherwise discover
    search_paths = config.tracked_projects if config.tracked_projects else None
    found_projects = discover_projects(search_paths)

    if not found_projects:
        console.print("[yellow]No AFS-enabled projects found.[/yellow]")
        return

    table = Table(title="AFS-Enabled Projects")
    table.add_column("Project", style="purple")
    table.add_column("Path", style="dim")
    table.add_column("Mounts", style="cyan")

    for project in found_projects:
        mount_counts = []
        for mt in MountType:
            count = len(project.mounts.get(mt, []))
            if count > 0:
                mount_counts.append(f"{mt.value}:{count}")

        table.add_row(
            project.project_name,
            str(project.path.parent),
            ", ".join(mount_counts) or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(found_projects)} projects[/dim]")


@app.command()
def logs(
    parser: str = typer.Option("gemini", help="Parser: gemini|claude|antigravity"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search query"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max items to show"),
) -> None:
    """Browse AI agent logs."""
    from hafs.core.parsers.registry import ParserRegistry

    parser_class = ParserRegistry.get(parser)
    if not parser_class:
        console.print(f"[red]Unknown parser: {parser}[/red]")
        console.print(f"Available: {', '.join(ParserRegistry.list_parsers())}")
        raise typer.Exit(1)

    parser_instance = parser_class()

    if not parser_instance.exists():
        console.print(f"[yellow]No logs found at {parser_instance.base_path}[/yellow]")
        return

    items = parser_instance.parse(max_items=limit)

    if search:
        items = parser_instance.search(search, items)

    if not items:
        console.print("[yellow]No items found.[/yellow]")
        return

    # Display based on parser type
    if parser == "gemini":
        _display_gemini_sessions(items)
    elif parser == "claude":
        _display_claude_plans(items)
    elif parser == "antigravity":
        _display_antigravity_brains(items)


def _display_gemini_sessions(sessions) -> None:  # type: ignore[no-untyped-def]
    """Display Gemini sessions."""
    table = Table(title="Gemini Sessions")
    table.add_column("ID", style="purple")
    table.add_column("Time", style="dim")
    table.add_column("Messages", style="cyan")
    table.add_column("Tokens", style="green")

    for session in sessions:
        table.add_row(
            session.short_id,
            session.start_time.strftime("%Y-%m-%d %H:%M"),
            str(len(session.messages)),
            str(session.total_tokens),
        )

    console.print(table)


def _display_claude_plans(plans) -> None:  # type: ignore[no-untyped-def]
    """Display Claude plans."""
    table = Table(title="Claude Plans")
    table.add_column("Title", style="purple")
    table.add_column("Progress", style="cyan")
    table.add_column("Tasks", style="dim")

    for plan in plans:
        done, total = plan.progress
        progress_bar = f"[{'█' * done}{'░' * (total - done)}]" if total > 0 else "-"
        table.add_row(
            plan.title[:40],
            f"{done}/{total} {progress_bar}",
            f"✓{plan.done_count} ⏳{plan.in_progress_count} ○{plan.todo_count}",
        )

    console.print(table)


def _display_antigravity_brains(brains) -> None:  # type: ignore[no-untyped-def]
    """Display Antigravity brains."""
    table = Table(title="Antigravity Brains")
    table.add_column("ID", style="purple")
    table.add_column("Title", style="cyan")
    table.add_column("Tasks", style="dim")

    for brain in brains:
        done, total = brain.progress
        table.add_row(
            brain.short_id,
            brain.title[:40],
            f"{done}/{total}",
        )

    console.print(table)


@app.command()
def tui() -> None:
    """Launch the TUI interface."""
    from hafs.ui.app import run

    run()


@app.command()
def version() -> None:
    """Show version information."""
    from hafs import __version__

    console.print(
        Panel(
            f"[bold purple]HAFS[/bold purple] - Halext Agentic File System\n"
            f"Version: {__version__}",
            border_style="purple",
        )
    )


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
