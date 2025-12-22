"""Console UI for memory commands."""
from rich.console import Console
from typing import Any

def render_stored_memory(console: Console, entry_id: str, memory_type: str, importance: float) -> None:
    """Render stored memory details."""
    console.print(f"[green]Stored memory {entry_id}[/green]")
    console.print(f"  Type: {memory_type}")
    console.print(f"  Importance: {importance}")

def render_cross_search_results(console: Console, query: str, results: list[dict[str, Any]]) -> None:
    """Render cross-agent search results."""
    if not results:
        console.print("[yellow]No memories found across agents[/yellow]")
        return

    console.print(f"[bold]Cross-agent search for '{query}':[/bold]")
    for result in results:
        entry = result["entry"]
        console.print(
            f"\n  [{entry['agent_id']}] [{entry['memory_type']}] "
            f"score={result['score']:.3f}"
        )
        console.print(f"  [dim]{entry['content'][:200]}...[/dim]")
