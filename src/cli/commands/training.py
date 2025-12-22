"""Training campaign monitoring commands."""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from pathlib import Path
import time
import sys
import json
from datetime import datetime
from typing import Optional

training_app = typer.Typer(name="training", help="Training campaign monitoring and control")
console = Console()


@training_app.command("status")
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuous monitoring"),
    interval: int = typer.Option(30, "--interval", "-i", help="Update interval in seconds"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
):
    """Show training campaign status."""
    from agents.training.health_check import get_system_health_async

    async def _status():
        health = await get_system_health_async()

        if json_output:
            from dataclasses import asdict
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
                        current_health = await get_system_health_async()
                        live.update(render_status(current_health))
                        time.sleep(interval)
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped.[/yellow]")
        else:
            console.print(render_status(health))

    asyncio.run(_status())


@training_app.command("history")
def history(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of runs to show"),
):
    """List historical training campaigns."""
    from agents.training.health_check import list_historical_campaigns

    campaigns = list_historical_campaigns()

    if not campaigns:
        console.print("[yellow]No historical campaigns found.[/yellow]")
        return

    table = Table(title="Training History")
    table.add_column("Run ID", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Domains")
    table.add_column("Samples", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Duration")

    for c in campaigns[:limit]:
        meta = c.get("metadata", {})
        stats = c.get("stats", {})

        created = meta.get("created", "unknown")
        domains = ", ".join(meta.get("domains", []))
        samples = f"{stats.get('final_count', 0)} / {meta.get('train_count', 0)}"
        quality = f"{stats.get('quality_scores', {}).get('average', 0.0):.1%}"

        duration_sec = stats.get("duration_seconds", 0)
        if duration_sec > 3600:
            duration = f"{duration_sec / 3600:.1f}h"
        elif duration_sec > 60:
            duration = f"{duration_sec / 60:.1f}m"
        else:
            duration = f"{duration_sec:.1f}s"

        table.add_row(
            c["id"],
            created,
            domains,
            samples,
            quality,
            duration
        )

    console.print(table)


@training_app.command("show")
def show(
    run_id: str = typer.Argument(..., help="Run ID (directory name)"),
):
    """Show details of a specific training run."""
    from agents.training.health_check import list_historical_campaigns

    campaigns = list_historical_campaigns()
    campaign = next((c for c in campaigns if c["id"] == run_id), None)

    if not campaign:
        console.print(f"[red]Campaign {run_id} not found.[/red]")
        raise typer.Exit(1)

    meta = campaign.get("metadata", {})
    stats = campaign.get("stats", {})

    console.print(Panel(f"[bold cyan]Run:[/bold cyan] {run_id}", expand=False))

    details = Table(show_header=False, box=None)
    details.add_column("Key", style="cyan")
    details.add_column("Value")

    details.add_row("Created", meta.get("created", "unknown"))
    details.add_row("Template", meta.get("template", "unknown"))
    details.add_row("Domains", ", ".join(meta.get("domains", [])))
    details.add_row("Target Samples", str(meta.get("train_count", 0)))
    details.add_row("Final Samples", str(stats.get("final_count", 0)))
    details.add_row("Quality Average", f"{stats.get('quality_scores', {}).get('average', 0.0):.1%}")

    duration_sec = stats.get("duration_seconds", 0)
    details.add_row("Duration", f"{duration_sec:.1f} seconds")

    console.print(details)

    if "domain_counts" in stats:
        domain_table = Table(title="Domain Distribution")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Count", justify="right")

        for domain, count in stats["domain_counts"].items():
            domain_table.add_row(domain, str(count))

        console.print(domain_table)

    # Find matching log
    from agents.training.health_check import find_latest_campaign_log
    log_dir = Path.home() / ".context" / "logs"
    # Try to find log with timestamp in it
    timestamp = meta.get("created", "")
    if timestamp:
        # Match campaign_*.log files
        matches = list(log_dir.glob(f"campaign_*{timestamp}*.log"))
        if matches:
            console.print(f"\n[blue]Associated Log:[/blue] {matches[0]}")


@training_app.command("logs")
def logs(
    run_id: Optional[str] = typer.Argument(None, help="Optional Run ID to find logs for"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
):
    """Show training campaign logs."""
    from agents.training.health_check import find_latest_campaign_log, list_historical_campaigns

    log_path = None

    if run_id:
        campaigns = list_historical_campaigns()
        campaign = next((c for c in campaigns if c["id"] == run_id), None)
        if campaign:
            timestamp = campaign.get("metadata", {}).get("created", "")
            if timestamp:
                log_dir = Path.home() / ".context" / "logs"
                matches = list(log_dir.glob(f"campaign_*{timestamp}*.log"))
                if matches:
                    log_path = matches[0]

    if not log_path:
        log_path = find_latest_campaign_log()

    if not log_path or not log_path.exists():
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


def render_status(health):
    """Render training status panel."""
    from datetime import datetime
    from rich.columns import Columns
    from rich import box
    from rich.panel import Panel
    from rich.console import Group

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

    # Remote Nodes section
    remote_table = Table(title="Remote Inference", show_header=False, box=None)
    remote_table.add_column("Key", style="cyan")
    remote_table.add_column("Value")

    if health.remote_nodes:
        for node in health.remote_nodes:
            status_icon = "🟢" if node["online"] else "🔴"
            remote_table.add_row(node["name"], f"{status_icon} {'Online' if node['online'] else 'Offline'}")
            if node["online"]:
                remote_table.add_row("  GPU", node["gpu"] or "unknown")
                remote_table.add_row("  Memory", f"{node['memory_gb']} GB")
                # Show first 3 models if many
                models = node["models"]
                model_str = ", ".join(models[:3])
                if len(models) > 3:
                    model_str += f" (+{len(models)-3} more)"
                remote_table.add_row("  Models", model_str)
    else:
        remote_table.add_row("Nodes", "None configured")

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
        Columns([campaign_table, system_table]),
        Columns([services_table, remote_table]),
        Text(""),
        Text.from_markup(issues_text)
    )

    return layout


if __name__ == "__main__":
    training_app()
