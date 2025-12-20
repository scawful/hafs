"""Console UI for services commands."""
import typer
from rich.console import Console

from hafs.core.services import ServiceState, ServiceStatus

def render_service_list(console: Console, statuses: dict[str, ServiceStatus], platform_name: str) -> None:
    """Render the list of services."""
    console.print(f"\n[bold]Platform:[/bold] {platform_name}\n")

    for name, status in sorted(statuses.items()):
        state_color = {
            ServiceState.RUNNING: "green",
            ServiceState.STOPPED: "dim",
            ServiceState.FAILED: "red",
        }.get(status.state, "white")

        indicator = "[green]\u25cf[/]" if status.state == ServiceState.RUNNING else "[dim]\u25cb[/]"
        console.print(f"  {indicator} [{state_color}]{name}[/]: {status.state.value}")
        if status.pid:
            console.print(f"      PID: {status.pid}")
        if status.enabled:
            console.print("      [dim]installed[/dim]")

def render_start_result(console: Console, name: str, success: bool) -> None:
    """Render the result of a start command."""
    if success:
        console.print(f"[green]Started {name}[/green]")
    else:
        console.print(f"[red]Failed to start {name}[/red]")
        raise typer.Exit(1)

def render_stop_result(console: Console, name: str, success: bool) -> None:
    """Render the result of a stop command."""
    if success:
        console.print(f"[green]Stopped {name}[/green]")
    else:
        console.print(f"[red]Failed to stop {name}[/red]")
        raise typer.Exit(1)

def render_restart_result(console: Console, name: str, success: bool) -> None:
    """Render the result of a restart command."""
    if success:
        console.print(f"[green]Restarted {name}[/green]")
    else:
        console.print(f"[red]Failed to restart {name}[/red]")
        raise typer.Exit(1)

def render_install_result(console: Console, name: str, success: bool) -> None:
    """Render the result of an install command."""
    if success:
        console.print(f"[green]Installed {name}[/green]")
    else:
        console.print(f"[red]Failed to install {name}[/red]")
        raise typer.Exit(1)

def render_uninstall_result(console: Console, name: str, success: bool) -> None:
    """Render the result of an uninstall command."""
    if success:
        console.print(f"[green]Uninstalled {name}[/green]")
    else:
        console.print(f"[red]Failed to uninstall {name}[/red]")
        raise typer.Exit(1)

def render_enable_result(console: Console, name: str, success: bool) -> None:
    """Render the result of an enable command."""
    if success:
        console.print(f"[green]Enabled auto-start for {name}[/green]")
    else:
        console.print(f"[red]Failed to enable {name}[/red]")
        raise typer.Exit(1)

def render_disable_result(console: Console, name: str, success: bool) -> None:
    """Render the result of a disable command."""
    if success:
        console.print(f"[green]Disabled auto-start for {name}[/green]")
    else:
        console.print(f"[red]Failed to disable {name}[/red]")
        raise typer.Exit(1)

def render_unknown_service(console: Console, name: str, available: list[str]) -> None:
    """Render error for unknown service."""
    console.print(f"[red]Unknown service: {name}[/red]")
    console.print(f"[dim]Available services: {', '.join(available)}[/dim]")

async def stream_logs(console: Console, manager, name: str) -> None:
    """Stream logs to the console."""
    console.print(f"[dim]Following logs for {name}... (Ctrl+C to stop)[/dim]\n")
    try:
        async for line in manager.stream_logs(name):
            console.print(line, end="")
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped following logs[/dim]")

def render_service_logs(console: Console, name: str, logs: str) -> None:
    """Render service logs."""
    console.print(logs)
