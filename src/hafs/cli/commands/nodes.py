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
        try:
            await node_manager.load_config()
            ui_nodes.render_nodes_list(console, node_manager.nodes)
        finally:
            await node_manager.close()

    asyncio.run(_list())


@nodes_app.command("status")
def status() -> None:
    """Check node health status."""
    from hafs.core.nodes import node_manager

    async def _status() -> None:
        try:
            await node_manager.load_config()
            await node_manager.health_check_all()
            ui_nodes.render_nodes_status(console, node_manager.summary())
        finally:
            await node_manager.close()

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
        try:
            await node_manager.load_config()
            node = node_manager.get_node(name)
            if not node:
                ui_nodes.render_unknown_node(console, name)
                raise typer.Exit(1)
            ui_nodes.render_node_details(console, node.to_dict())
        finally:
            await node_manager.close()

    asyncio.run(_show())


@nodes_app.command("discover")
def discover() -> None:
    """Discover Ollama nodes on Tailscale."""
    from hafs.core.nodes import node_manager

    async def _discover() -> None:
        try:
            await node_manager.load_config()
            found = await node_manager.discover_tailscale_nodes()
            ui_nodes.render_discovered_nodes(console, found)
        finally:
            await node_manager.close()

    asyncio.run(_discover())


@nodes_app.command("probe")
def probe(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Node name (defaults to best online)"),
    prompt: str = typer.Option(
        "Say hello from HAFS.", "--prompt", "-p", help="Prompt to send"
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name"),
    task_type: Optional[str] = typer.Option(None, "--task", help="Preferred task type"),
    prefer_gpu: bool = typer.Option(False, "--prefer-gpu", help="Prefer GPU nodes"),
    prefer_local: bool = typer.Option(False, "--prefer-local", help="Prefer local nodes"),
) -> None:
    """Run a one-shot prompt against a node for smoke testing."""
    from hafs.core.nodes import node_manager

    async def _probe() -> None:
        backend = None
        try:
            await node_manager.load_config()

            if name:
                node = node_manager.get_node(name)
                if not node:
                    ui_nodes.render_unknown_node(console, name)
                    raise typer.Exit(1)
                await node_manager.health_check(node)
            else:
                node = await node_manager.get_best_node(
                    task_type=task_type,
                    required_model=model,
                    prefer_gpu=prefer_gpu,
                    prefer_local=prefer_local,
                )

            if not node:
                console.print("[red]No suitable node available[/red]")
                raise typer.Exit(1)

            if model and node.models and model not in node.models:
                console.print(
                    f"[yellow]Model '{model}' not reported by {node.name}; "
                    "attempting anyway.[/yellow]"
                )

            backend = node_manager.create_backend(node, model=model)
            if not await backend.start():
                console.print(f"[red]Failed to connect to {node.name}[/red]")
                raise typer.Exit(1)

            console.print(f"[bold]Node:[/bold] {node.name} ({node.host}:{node.port})")
            console.print(f"[bold]Model:[/bold] {backend.model}")
            console.print(f"[bold]Prompt:[/bold] {prompt}")
            response = await backend.generate_one_shot(prompt)
            console.print("\n[bold green]Response:[/bold green]")
            console.print(response)
        finally:
            if backend:
                await backend.stop()
            await node_manager.close()

    if prompt is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    asyncio.run(_probe())
