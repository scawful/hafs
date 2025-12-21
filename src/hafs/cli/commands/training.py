"""Training campaign monitoring commands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from pathlib import Path
import time
import sys

training_app = typer.Typer(name="training", help="Training campaign monitoring and control")
console = Console()


@training_app.command("status")
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuous monitoring"),
    interval: int = typer.Option(30, "--interval", "-i", help="Update interval in seconds"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
):
    """Show training campaign status."""
    if json_output:
        # Import and run health check in JSON mode
        from agents.training.health_check import get_system_health
        import json as json_lib
        from dataclasses import asdict

        health = get_system_health()
        health_dict = asdict(health)

        # Handle datetime serialization
        if health_dict['campaign']:
            health_dict['campaign']['last_update'] = health_dict['campaign']['last_update'].isoformat()
            if health_dict['campaign']['log_file']:
                health_dict['campaign']['log_file'] = str(health_dict['campaign']['log_file'])
        if health_dict['last_checkpoint']:
            health_dict['last_checkpoint'] = health_dict['last_checkpoint'].isoformat()

        console.print_json(data=health_dict)
        return

    if watch:
        try:
            with Live(console=console, refresh_per_second=1) as live:
                while True:
                    live.update(render_status())
                    time.sleep(interval)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
    else:
        console.print(render_status())


@training_app.command("logs")
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """Show training campaign logs."""
    from agents.training.health_check import find_latest_campaign_log

    log_path = find_latest_campaign_log()
    if not log_path:
        console.print("[red]No campaign log found[/red]")
        raise typer.Exit(1)

    console.print(f"[blue]Log:[/blue] {log_path}")
    console.print()

    if follow:
        # Tail -f style
        import subprocess
        try:
            subprocess.run(['tail', '-f', str(log_path)])
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped following logs.[/yellow]")
    else:
        # Show last N lines
        with open(log_path) as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                console.print(line, end='')


@training_app.command("stop")
def stop(
    force: bool = typer.Option(False, "--force", "-f", help="Force stop without confirmation"),
):
    """Stop running training campaign."""
    from agents.training.health_check import find_campaign_process
    import signal

    proc_info = find_campaign_process()
    if not proc_info:
        console.print("[yellow]No running campaign found[/yellow]")
        return

    pid = proc_info['pid']

    if not force:
        confirm = typer.confirm(f"Stop campaign (PID {pid})?")
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            return

    try:
        import os
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Sent SIGTERM to campaign (PID {pid})[/green]")
        console.print("[blue]Checkpoints will be saved before shutdown[/blue]")
    except ProcessLookupError:
        console.print(f"[red]Process {pid} not found[/red]")
    except PermissionError:
        console.print(f"[red]Permission denied to stop process {pid}[/red]")


def render_status():
    """Render training status panel."""
    from agents.training.health_check import get_system_health
    from datetime import datetime
    from rich.columns import Columns
    from rich import box
    from rich.panel import Panel
    from rich.console import Group

    health = get_system_health()

    # Title
    title = Panel.fit(
        "[bold cyan]TRAINING CAMPAIGN STATUS[/bold cyan]",
        subtitle=f"Updated: {datetime.now().strftime('%H:%M:%S')}",
    )

    # Campaign section
    campaign_table = Table(title="Campaign", show_header=False, box=None)
    campaign_table.add_column("Key", style="cyan")
    campaign_table.add_column("Value")

    if health.campaign:
        c = health.campaign
        status_icon = "🟢" if c.running else "🔴"
        campaign_table.add_row("Status", f"{status_icon} {'Running' if c.running else 'Stopped'}")

        if c.pid:
            campaign_table.add_row("PID", str(c.pid))

        campaign_table.add_row("Progress", f"{c.samples_generated:,} / {c.target_samples:,} ({c.progress_percent:.1f}%)")
        campaign_table.add_row("Domain", c.current_domain)
        campaign_table.add_row("Rate", f"{c.samples_per_min:.1f} samples/min")
        campaign_table.add_row("Quality Pass", f"{c.quality_pass_rate:.1%}")

        if c.eta_hours > 0:
            campaign_table.add_row("ETA", f"{c.eta_hours:.1f} hours")

        campaign_table.add_row("Last Update", c.last_update.strftime("%H:%M:%S"))
    else:
        campaign_table.add_row("Status", "🔵 No active campaign")

    # System section
    system_table = Table(title="System Resources", show_header=False, box=None)
    system_table.add_column("Key", style="cyan")
    system_table.add_column("Value")

    cpu_icon = "🟢" if health.cpu_percent < 70 else "🟡" if health.cpu_percent < 90 else "🔴"
    system_table.add_row("CPU", f"{cpu_icon} {health.cpu_percent:.1f}%")

    mem_icon = "🟢" if health.memory_percent < 70 else "🟡" if health.memory_percent < 90 else "🔴"
    system_table.add_row("Memory", f"{mem_icon} {health.memory_percent:.1f}%")

    disk_icon = "🟢" if health.disk_free_gb > 50 else "🟡" if health.disk_free_gb > 10 else "🔴"
    system_table.add_row("Disk Free", f"{disk_icon} {health.disk_free_gb:.1f} GB")

    # Services section
    services_table = Table(title="Services", show_header=False, box=None)
    services_table.add_column("Key", style="cyan")
    services_table.add_column("Value")

    emb_icon = "🟢" if health.embedding_service_running else "🔴"
    services_table.add_row("Embedding Service", f"{emb_icon} {'Running' if health.embedding_service_running else 'Stopped'}")
    services_table.add_row("Knowledge Bases", str(health.knowledge_bases_loaded))

    # Issues section
    issues_text = ""
    if health.issues:
        issues_text = f"[bold red]Issues ({len(health.issues)}):[/bold red]\n"
        for issue in health.issues:
            issues_text += f"  {issue}\n"
    else:
        issues_text = "[bold green]✅ No issues detected[/bold green]"

    # Combine all
    from rich.text import Text
    layout = Group(
        title,
        Text(""),
        Columns([campaign_table, system_table, services_table]),
        Text(""),
        Text.from_markup(issues_text)
    )

    return layout


if __name__ == "__main__":
    training_app()
