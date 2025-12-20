"""Console UI for nodes commands."""
from rich.console import Console
from typing import Any

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
