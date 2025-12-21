import typer
import asyncio
from typing import Optional
from rich.console import Console

from hafs.config.loader import load_config
from hafs.ui.console import memory as ui_memory

memory_app = typer.Typer(
    name="memory",
    help="Manage agent memory, recall, and cross-search",
)
console = Console()


@memory_app.command("status")
def status(
    agent: Optional[str] = typer.Option(None, help="Specific agent ID"),
) -> None:
    """Show agent memory status."""
    from hafs.core.history import AgentMemoryManager

    config = load_config()
    manager = AgentMemoryManager(config.general.context_root)

    if agent:
        # Show specific agent
        memory = manager.get_agent_memory(agent)
        stats = memory.get_stats()
        console.print(f"[bold]Agent: {agent}[/bold]")
        console.print(f"  Total entries: {stats['total_entries']}")
        console.print(f"  Working memory: {stats['working_memory']}")
        console.print(f"  Recent memory: {stats['recent_memory']}")
        console.print(f"  Archive memory: {stats['archive_memory']}")
        console.print(f"  Session summaries: {stats['total_summaries']}")
        if stats.get("by_type"):
            console.print("  By type:")
            for mtype, count in stats["by_type"].items():
                console.print(f"    {mtype}: {count}")
    else:
        # Show all agents
        agents = manager.list_agents()
        if not agents:
            console.print("[dim]No agents with memory found[/dim]")
            return

        console.print(f"[bold]Agents with memory: {len(agents)}[/bold]")
        for agent_id in agents:
            memory = manager.get_agent_memory(agent_id)
            stats = memory.get_stats()
            console.print(
                f"  {agent_id}: {stats['total_entries']} entries, "
                f"{stats['total_summaries']} summaries"
            )


@memory_app.command("recall")
def recall(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    agent: str = typer.Option(..., help="Agent ID to search"),
    limit: int = typer.Option(10, help="Max results"),
    bucket: str = typer.Option("all", help="Temporal bucket: working, recent, archive, all"),
    recency: float = typer.Option(0.3, help="Recency weight (0-1)"),
) -> None:
    """Search an agent's memory with temporal awareness."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.history import AgentMemoryManager

    async def _recall() -> None:
        config = load_config()
        manager = AgentMemoryManager(config.general.context_root)
        memory = manager.get_agent_memory(agent)

        results = await memory.recall(
            query=query,
            limit=limit,
            temporal_bucket=bucket,
            recency_weight=recency,
        )

        if not results:
            console.print("[yellow]No memories found[/yellow]")
            return

        console.print(f"[bold]Memories matching '{query}':[/bold]")
        for result in results:
            entry = result["entry"]
            console.print(
                f"\n  [{entry['memory_type']}] "
                f"score={result['score']:.3f} "
                f"({result['temporal_bucket']})"
            )
            console.print(f"  [dim]{entry['content'][:200]}...[/dim]")

    asyncio.run(_recall())


@memory_app.command("remember")
def remember(
    ctx: typer.Context,
    content: Optional[str] = typer.Argument(None, help="Content to remember"),
    agent: str = typer.Option(..., help="Agent ID"),
    memory_type: str = typer.Option(
        "insight", help="Type: decision, interaction, learning, error, insight"
    ),
    importance: float = typer.Option(0.5, help="Importance (0-1)"),
) -> None:
    """Store a memory for an agent."""
    if content is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.history import AgentMemoryManager

    async def _remember() -> None:
        config = load_config()
        manager = AgentMemoryManager(config.general.context_root)
        memory = manager.get_agent_memory(agent)

        entry = await memory.remember(
            content=content,
            memory_type=memory_type,
            importance=importance,
        )

        ui_memory.render_stored_memory(console, entry.id, entry.memory_type, entry.importance)

    asyncio.run(_remember())


@memory_app.command("cross-search")
def cross_search(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
) -> None:
    """Search across all agents' memories."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.history import AgentMemoryManager

    async def _search() -> None:
        config = load_config()
        manager = AgentMemoryManager(config.general.context_root)

        results = await manager.cross_agent_search(query, limit=limit)

        ui_memory.render_cross_search_results(console, query, results)

    asyncio.run(_search())
