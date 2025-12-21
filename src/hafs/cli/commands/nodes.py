import typer
import asyncio
from typing import Optional
from rich.console import Console

from hafs.ui.console import nodes as ui_nodes

nodes_app = typer.Typer(
    name="nodes",
    help="Manage distributed node registry and health checks",
)
console = Console()


@nodes_app.callback(invoke_without_command=True)
def nodes_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@nodes_app.command("list")
def list_nodes() -> None:
    """List configured nodes."""
    from hafs.core.nodes import node_manager

    async def _list() -> None:
        await node_manager.load_config()
        ui_nodes.render_nodes_list(console, node_manager.nodes)

    asyncio.run(_list())


@nodes_app.command("status")
def status() -> None:
    """Check node health status."""
    from hafs.core.nodes import node_manager

    async def _status() -> None:
        await node_manager.load_config()
        await node_manager.health_check_all()
        ui_nodes.render_nodes_status(console, node_manager.summary())

    asyncio.run(_status())


@nodes_app.command("show")
def show(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Node name"),
) -> None:
    """Show detailed node configuration."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.nodes import node_manager

    async def _show() -> None:
        await node_manager.load_config()
        node = node_manager.get_node(name)
        if not node:
            ui_nodes.render_unknown_node(console, name)
            raise typer.Exit(1)
        ui_nodes.render_node_details(console, node.to_dict())

    asyncio.run(_show())


@nodes_app.command("discover")
def discover() -> None:
    """Discover Ollama nodes on Tailscale."""
    from hafs.core.nodes import node_manager

    async def _discover() -> None:
        await node_manager.load_config()
        found = await node_manager.discover_tailscale_nodes()
        ui_nodes.render_discovered_nodes(console, found)

    asyncio.run(_discover())
