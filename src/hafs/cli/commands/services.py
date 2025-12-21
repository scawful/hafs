import typer
import asyncio
from typing import Optional
from rich.console import Console

from hafs.config.loader import load_config
from hafs.ui.console import services as ui_services

services_app = typer.Typer(
    name="services",
    help="Manage background services (orchestrator, coordinator, autonomy, dashboard)",
)
console = Console()


@services_app.callback(invoke_without_command=True)
def services_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@services_app.command("list")
def list_services() -> None:
    """List all services and their status."""
    from hafs.core.services import ServiceManager

    async def _list() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        statuses = await manager.status_all()
        ui_services.render_service_list(console, statuses, manager.platform_name)

    asyncio.run(_list())


@services_app.command("start")
def start(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Start a service."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _start() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        definition = manager.get_service_definition(name)

        if not definition:
            ui_services.render_unknown_service(console, name, manager.list_services())
            raise typer.Exit(1)

        # Install if needed
        await manager.install(definition)
        success = await manager.start(name)

        ui_services.render_start_result(console, name, success)

    asyncio.run(_start())


@services_app.command("stop")
def stop(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Stop a service."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _stop() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.stop(name)
        ui_services.render_stop_result(console, name, success)

    asyncio.run(_stop())


@services_app.command("restart")
def restart(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Restart a service."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _restart() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.restart(name)
        ui_services.render_restart_result(console, name, success)

    asyncio.run(_restart())


@services_app.command("logs")
def logs(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow log output"),
    lines: int = typer.Option(100, "-n", "--lines", help="Number of lines to show"),
) -> None:
    """View service logs."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _logs() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        if follow:
            await ui_services.stream_logs(console, manager, name)
        else:
            logs = await manager.logs(name, lines)
            ui_services.render_service_logs(console, name, logs)

    asyncio.run(_logs())


@services_app.command("install")
def install(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Install service configuration files."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _install() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        definition = manager.get_service_definition(name)

        if not definition:
            ui_services.render_unknown_service(console, name, manager.list_services())
            raise typer.Exit(1)

        success = await manager.install(definition)
        ui_services.render_install_result(console, name, success)

    asyncio.run(_install())


@services_app.command("uninstall")
def uninstall(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Uninstall service configuration files."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _uninstall() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.uninstall(name)
        ui_services.render_uninstall_result(console, name, success)

    asyncio.run(_uninstall())


@services_app.command("enable")
def enable(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Enable service to start at login."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _enable() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.enable(name)
        ui_services.render_enable_result(console, name, success)

    asyncio.run(_enable())


@services_app.command("disable")
def disable(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Service name"),
) -> None:
    """Disable service from starting at login."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.services import ServiceManager

    async def _disable() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.disable(name)
        ui_services.render_disable_result(console, name, success)

    asyncio.run(_disable())
