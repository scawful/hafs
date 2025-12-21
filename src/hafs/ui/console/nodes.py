"""Console UI for nodes commands."""
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table


def _human_bytes(size: float | int | None) -> str:
    if not size:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _format_modified(raw: Any) -> str:
    if not raw:
        return "-"
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw).strftime("%Y-%m-%d %H:%M")
        except (OSError, ValueError):
            return str(raw)
    if isinstance(raw, str):
        return raw.replace("T", " ").replace("Z", "")
    return str(raw)

def render_nodes_list(console: Console, nodes: list[Any]) -> None:
    """Render list of nodes."""
    if not nodes:
        console.print("[yellow]No nodes configured[/yellow]")
        return

    console.print("\n[bold]Nodes:[/bold]")
    for node in nodes:
        caps = ", ".join(node.capabilities) if node.capabilities else "none"
        console.print(
            f"  [bold]{node.name}[/bold] {node.host}:{node.port} "
            f"[dim]{node.node_type}[/dim] [dim]{node.platform}[/dim] "
            f"[dim]capabilities: {caps}[/dim]"
        )

def render_nodes_status(console: Console, summary: str) -> None:
    """Render node status summary."""
    console.print(summary)

def render_node_details(console: Console, data: dict[str, Any]) -> None:
    """Render detailed node configuration."""
    for key in sorted(data.keys()):
        console.print(f"{key}: {data[key]}")

def render_unknown_node(console: Console, name: str) -> None:
    """Render unknown node error."""
    console.print(f"[red]Unknown node: {name}[/red]")

def render_discovered_nodes(console: Console, nodes: list[Any]) -> None:
    """Render discovered nodes."""
    if not nodes:
        console.print("[yellow]No Tailscale nodes discovered[/yellow]")
        return
    console.print("[green]Discovered nodes:[/green]")
    for node in nodes:
        console.print(f"  {node.name} {node.host}:{node.port}")


def render_models(console: Console, models: list[dict[str, Any]], details: bool) -> None:
    """Render models available on a node."""
    if not models:
        console.print("[yellow]No models reported[/yellow]")
        return

    table = Table(title="Ollama Models")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Modified")
    if details:
        table.add_column("Params")
        table.add_column("Quant")
        table.add_column("Format")

    for model in models:
        name = model.get("name") or model.get("model") or "unknown"
        size = _human_bytes(model.get("size"))
        modified = _format_modified(model.get("modified_at"))
        row = [name, size, modified]
        if details:
            show = model.get("show", {}) or {}
            show_details = show.get("details", {}) if isinstance(show, dict) else {}
            params = show_details.get("parameter_size") or show.get("parameter_size") or "-"
            quant = show_details.get("quantization_level") or show.get("quantization") or "-"
            fmt = show_details.get("format") or show.get("format") or "-"
            row.extend([str(params), str(quant), str(fmt)])
        table.add_row(*row)

    console.print(table)


def render_pull_result(console: Console, model: str, success: bool, node: str) -> None:
    """Render model pull result."""
    if success:
        console.print(f"[green]Pulled {model} on {node}[/green]")
    else:
        console.print(f"[red]Failed to pull {model} on {node}[/red]")


def render_probe_suite(console: Console, results: list[dict[str, Any]]) -> None:
    """Render probe suite results."""
    if not results:
        console.print("[yellow]No probe results[/yellow]")
        return

    table = Table(title="Probe Suite Results")
    table.add_column("Model", style="bold")
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("Latency", justify="right")
    table.add_column("Notes")

    for result in results:
        status = "[green]ok[/green]" if result.get("ok") else "[red]fail[/red]"
        latency = result.get("latency_ms")
        latency_text = f"{latency} ms" if latency is not None else "-"
        notes = result.get("note") or result.get("error") or ""
        table.add_row(
            str(result.get("model", "-")),
            str(result.get("case_id", "-")),
            status,
            latency_text,
            notes[:120],
        )

    console.print(table)
