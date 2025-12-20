"""Console UI for embedding commands."""
from rich.console import Console
from typing import Any

def render_daemon_status(console: Console, status: dict[str, Any]) -> None:
    """Render embedding daemon status."""
    if status.get("running"):
        console.print("[green]Daemon Status: Running[/green]")
        console.print(f"  PID: {status.get('pid', 'unknown')}")
    else:
        console.print("[dim]Daemon Status: Stopped[/dim]")

    if "total_symbols" in status:
        total = status["total_symbols"]
        done = status["total_embeddings"]
        pct = status.get("coverage_percent", 0)
        console.print(f"\n[bold]Coverage:[/bold] {done:,}/{total:,} ({pct}%)")
        console.print(f"[bold]Daily count:[/bold] {status.get('daily_count', 0)}/{status.get('daily_limit', 5000)}")

    if "last_update" in status:
        console.print(f"[dim]Last update: {status['last_update']}[/dim]")

def render_indexing_complete(console: Console) -> None:
    """Render indexing complete message."""
    console.print("[green]Indexing complete[/green]")

def render_daemon_started(console: Console, pid: int) -> None:
    """Render daemon started message."""
    console.print(f"[green]Started embedding daemon (PID: {pid})[/green]")
    console.print(f"[dim]Log: ~/.context/logs/embedding_daemon.log[/dim]")

def render_daemon_stopped(console: Console, pid: int) -> None:
    """Render daemon stopped message."""
    console.print(f"[green]Stopped daemon (PID: {pid})[/green]")

def render_daemon_not_running(console: Console) -> None:
    """Render daemon not running message."""
    console.print("[dim]Daemon not running[/dim]")

def render_quick_stats(console: Console, stats: dict[str, Any], label: str = "Before") -> None:
    """Render quick embedding stats."""
    console.print(f"[bold]{label}:[/bold] {stats['total_embeddings']:,} embeddings", end="")

def render_quick_progress(console: Console, current: int, total: int) -> None:
    """Render quick progress."""
    console.print(f"  Progress: {current}/{total}")

def render_enhance_stats(console: Console, count: int, type_name: str) -> None:
    """Render enhance stats."""
    console.print(f"  [green]Created {count} {type_name} embeddings[/green]")

def render_all_symbols_have_embeddings(console: Console) -> None:
    """Render message when all symbols have embeddings."""
    console.print("[green]All symbols have embeddings![/green]")

def render_generating_embeddings(console: Console, count: int) -> None:
    """Render generating embeddings message."""
    console.print(f"Generating {count} embeddings...")

def render_enhance_start(console: Console, kb: str) -> None:
    """Render enhance start message."""
    console.print(f"[bold]Generating enhanced embeddings for {kb}...[/bold]")

def render_found_stats(console: Console, symbols: int, routines: int, modules: int) -> None:
    """Render found statistics."""
    console.print(
        f"Found {symbols} symbols, {routines} routines, {modules} modules"
    )

def render_enhance_phase(console: Console, phase: str) -> None:
    """Render enhance phase message."""
    console.print(f"[dim]Generating {phase}...[/dim]")

