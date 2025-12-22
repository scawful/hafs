import typer
import importlib.metadata
from pathlib import Path
from typing import Optional
from rich.console import Console

from cli.commands.orchestrator import orchestrator_app
from cli.commands.services import services_app
from cli.commands.history import history_app
from cli.commands.embed import embed_app
from cli.commands.nodes import nodes_app
from cli.commands.sync import sync_app
from cli.commands.afs import afs_app
from cli.commands.context import context_app
from cli.commands.memory import memory_app
from cli.commands.chat import chat_app
from cli.commands.auth import auth_app
from cli.commands.llamacpp import llamacpp_app
from cli.commands.training import training_app
from cli.commands.config import config_app

app = typer.Typer(
    name="hafs",
    help="""
\b
 _   _    _    _____ ____  
| | | |  / \\  |  ___/ ___| 
| |_| | / _ \\ | |_  \\___ \\ 
|  _  |/ ___ \\|  _|  ___) |
|_| |_/_/   \\_\\_|   |____/ 

hAFS - Halext Agentic File System
(AFS ops, embeddings, and swarm/council orchestration)
""",
    invoke_without_command=True,
    rich_markup_mode="rich",
)
console = Console()

# Register subcommands
app.add_typer(orchestrator_app)
app.add_typer(afs_app)
app.add_typer(embed_app)
app.add_typer(history_app)
app.add_typer(context_app)
app.add_typer(memory_app)
app.add_typer(nodes_app)
app.add_typer(services_app)
app.add_typer(sync_app)
app.add_typer(chat_app)
app.add_typer(chat_app, name="shell", help="Alias for 'chat'")
app.add_typer(auth_app)
app.add_typer(llamacpp_app)
app.add_typer(training_app)
app.add_typer(config_app)


@app.command("init", help="Initialize AFS (.context) in the target directory.")
def init_legacy(
    path: Path = typer.Argument(Path("."), help="Path to initialize AFS in"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force initialization even if .context exists"
    ),
) -> None:
    from cli.commands.afs import init as afs_init

    afs_init(path=path, force=force)


@app.command("mount", help="Mount a resource into the nearest AFS context.")
def mount_legacy(
    ctx: typer.Context,
    mount_type: Optional[str] = typer.Argument(
        None, help="Mount type (memory, knowledge, tools, scratchpad, history)"
    ),
    source: Optional[Path] = typer.Argument(None, help="Source path to mount"),
    alias: Optional[str] = typer.Option(
        None, "--alias", "-a", help="Optional alias for the mount point"
    ),
) -> None:
    from cli.commands.afs import mount as afs_mount

    afs_mount(ctx, mount_type, source, alias)


@app.command("unmount", help="Remove a mount point from the nearest AFS context.")
def unmount_legacy(
    ctx: typer.Context,
    mount_type: Optional[str] = typer.Argument(None, help="Mount type"),
    alias: Optional[str] = typer.Argument(None, help="Alias or name of the mount to remove"),
) -> None:
    from cli.commands.afs import unmount as afs_unmount

    afs_unmount(ctx, mount_type, alias)


@app.command("list", help="List current AFS structure and mounts.")
def list_legacy() -> None:
    from cli.commands.afs import list_afs

    list_afs()


@app.command("clean", help="Remove the AFS context directory (clean).")
def clean_legacy(
    force: bool = typer.Option(False, "--force", "-f", help="Force cleaning without confirmation"),
) -> None:
    from cli.commands.afs import clean as afs_clean

    afs_clean(force=force)


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Launch TUI by default when no command is specified."""
    if ctx.invoked_subcommand is None:
        from tui.app import run

        run()


def load_plugins():
    """Discover and load Typer app plugins from entry points."""
    # For commands like 'hafs google ...'
    command_entry_points = importlib.metadata.entry_points(group="hafs.commands")
    for entry in command_entry_points:
        try:
            plugin_app = entry.load()
            app.add_typer(plugin_app, name=entry.name)
        except Exception as e:
            console.print(f"[yellow]Failed to load plugin {entry.name}: {e}[/yellow]")


# Load plugins at startup
load_plugins()


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
