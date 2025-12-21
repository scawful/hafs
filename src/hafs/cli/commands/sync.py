import typer
import asyncio
from typing import Optional
from rich.console import Console

from hafs.ui.console import sync as ui_sync

sync_app = typer.Typer(
    name="sync",
    help="Sync AFS data across nodes using sync profiles",
)
console = Console()


@sync_app.callback(invoke_without_command=True)
def sync_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@sync_app.command("list")
def list_sync() -> None:
    """List configured sync profiles."""
    from hafs.services.afs_sync import AFSSyncService

    async def _list() -> None:
        service = AFSSyncService()
        profiles = await service.load()
        ui_sync.render_sync_profiles(console, profiles)

    asyncio.run(_list())


@sync_app.command("show")
def show(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Profile name"),
) -> None:
    """Show details for a sync profile."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.services.afs_sync import AFSSyncService

    async def _show() -> None:
        service = AFSSyncService()
        await service.load()
        profile = service.resolve_profile(name)
        if not profile:
            ui_sync.render_unknown_profile(console, name)
            raise typer.Exit(1)
        ui_sync.render_sync_profile_details(console, profile)

    asyncio.run(_show())


@sync_app.command("run")
def run(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Profile name"),
    direction: Optional[str] = typer.Option(
        None,
        "--direction",
        help="push | pull | bidirectional",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview rsync actions"),
) -> None:
    """Run a sync profile."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.services.afs_sync import AFSSyncService

    async def _run() -> None:
        service = AFSSyncService()
        await service.load()
        results = await service.run_profile(name, direction_override=direction, dry_run=dry_run)
        ui_sync.render_sync_results(console, results)

    asyncio.run(_run())
