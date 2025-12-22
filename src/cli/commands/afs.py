import typer
import asyncio
from pathlib import Path
from typing import Optional
from rich.console import Console

from config.loader import load_config
from tui.console import afs as ui_afs

afs_app = typer.Typer(
    name="afs",
    help="Manage Agentic File System (AFS) structure",
)
console = Console()


@afs_app.callback(invoke_without_command=True)
def afs_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@afs_app.command("init")
def init(
    path: Path = typer.Argument(Path("."), help="Path to initialize AFS in"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force initialization even if .context exists"
    ),
) -> None:
    """Initialize AFS (.context) in the target directory."""
    from core.afs.manager import AFSManager

    config = load_config()
    manager = AFSManager(config)
    try:
        root = manager.init(path=path, force=force)
        ui_afs.render_init_result(console, root.path)
    except Exception as e:
        ui_afs.render_error(console, f"Failed to initialize AFS: {e}")
        raise typer.Exit(1)


@afs_app.command("mount")
def mount(
    ctx: typer.Context,
    mount_type: Optional[str] = typer.Argument(
        None, help="Mount type (memory, knowledge, tools, scratchpad, history)"
    ),
    source: Optional[Path] = typer.Argument(None, help="Source path to mount"),
    alias: Optional[str] = typer.Option(
        None, "--alias", "-a", help="Optional alias for the mount point"
    ),
) -> None:
    """Mount a resource into the nearest AFS context."""
    if mount_type is None or source is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from core.afs.manager import AFSManager
    from core.afs.discovery import find_context_root
    from models.afs import MountType

    config = load_config()
    manager = AFSManager(config)

    context_path = find_context_root()
    if not context_path:
        ui_afs.render_no_context_error(console)
        raise typer.Exit(1)

    try:
        mt = MountType(mount_type.lower())
    except ValueError:
        ui_afs.render_error(
            console,
            f"Invalid mount type: {mount_type}. Valid types: {[t.value for t in MountType]}",
        )
        raise typer.Exit(1)

    try:
        manager.mount(source=source, mount_type=mt, alias=alias, context_path=context_path)
        ui_afs.render_mount_result(console, source, alias or source.name, mt)
    except Exception as e:
        ui_afs.render_error(console, f"Mount failed: {e}")
        raise typer.Exit(1)


@afs_app.command("unmount")
def unmount(
    ctx: typer.Context,
    mount_type: Optional[str] = typer.Argument(None, help="Mount type"),
    alias: Optional[str] = typer.Argument(None, help="Alias or name of the mount to remove"),
) -> None:
    """Remove a mount point from the nearest AFS context."""
    if mount_type is None or alias is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from core.afs.manager import AFSManager
    from core.afs.discovery import find_context_root
    from models.afs import MountType

    config = load_config()
    manager = AFSManager(config)

    context_path = find_context_root()
    if not context_path:
        ui_afs.render_no_context_error(console)
        raise typer.Exit(1)

    try:
        mt = MountType(mount_type.lower())
    except ValueError:
        ui_afs.render_error(console, f"Invalid mount type: {mount_type}")
        raise typer.Exit(1)

    success = manager.unmount(alias, mt, context_path=context_path)
    ui_afs.render_unmount_result(console, alias, mt, success)


@afs_app.command("list")
def list_afs() -> None:
    """List current AFS structure and mounts."""
    from core.afs.manager import AFSManager
    from core.afs.discovery import find_context_root

    config = load_config()
    manager = AFSManager(config)

    context_path = find_context_root()
    if not context_path:
        ui_afs.render_no_context_error(console)
        raise typer.Exit(1)

    try:
        root = manager.list_afs_structure(context_path=context_path)
        ui_afs.render_structure(console, root)
    except Exception as e:
        ui_afs.render_error(console, f"Error listing AFS: {e}")


@afs_app.command("clean")
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Force cleaning without confirmation"),
) -> None:
    """Remove the AFS context directory (clean)."""
    from core.afs.manager import AFSManager
    from core.afs.discovery import find_context_root

    context_path = find_context_root()
    if not context_path:
        ui_afs.render_no_context_error(console)
        raise typer.Exit(1)

    if not force:
        if not typer.confirm(f"Are you sure you want to remove AFS at {context_path}?"):
            raise typer.Abort()

    config = load_config()
    manager = AFSManager(config)
    try:
        manager.clean(context_path=context_path)
        ui_afs.render_clean_result(console, context_path)
    except Exception as e:
        ui_afs.render_error(console, f"Error cleaning AFS: {e}")
