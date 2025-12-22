"""Console UI for AFS (Agentic File System) commands."""

import typer
from pathlib import Path
from rich.console import Console

from models.afs import ContextRoot, MountType, MountPoint


def render_init_result(console: Console, path: Path) -> None:
    """Render the result of AFS initialization."""
    console.print(f"[green]Initialized AFS in {path}[/green]")


def render_mount_result(console: Console, source: Path, alias: str, mount_type: MountType) -> None:
    """Render the result of a mount operation."""
    console.print(f"[green]Mounted {source} as {alias} in {mount_type.value}/[/green]")


def render_unmount_result(
    console: Console, alias: str, mount_type: MountType, success: bool
) -> None:
    """Render the result of an unmount operation."""
    if success:
        console.print(f"[green]Unmounted {alias} from {mount_type.value}/[/green]")
    else:
        console.print(f"[yellow]Mount {alias} not found in {mount_type.value}/[/yellow]")


def render_clean_result(console: Console, path: Path) -> None:
    """Render the result of a clean operation."""
    console.print(f"[green]Cleaned AFS context at {path}[/green]")


def render_structure(console: Console, root: ContextRoot) -> None:
    """Render the AFS structure and mounts."""
    console.print(f"[bold]Context Root:[/bold] {root.path}")
    console.print(f"[dim]Project: {root.project_name}[/dim]\n")

    # Sort mount types for consistent display
    sorted_types = sorted(root.mounts.keys(), key=lambda x: x.value)

    for mt in sorted_types:
        mounts = root.mounts[mt]
        console.print(f"[bold blue]{mt.value.upper()}[/bold blue]")
        if not mounts:
            console.print("  [dim]No mounts[/dim]")
            continue

        for mount in mounts:
            link_icon = "🔗" if mount.is_symlink else "📁"
            console.print(f"  {link_icon} {mount.name} -> {mount.source}")


def render_no_context_error(console: Console) -> None:
    """Render error when no .context is found."""
    console.print("[red]No .context directory found in current or parent directories.[/red]")
    console.print("[yellow]Run 'hafs afs init' first.[/yellow]")


def render_error(console: Console, message: str) -> None:
    """Render a generic error message."""
    console.print(f"[red]{message}[/red]")
