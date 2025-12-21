import typer
import importlib.metadata
from rich.console import Console

from hafs.cli.commands.orchestrator import orchestrator_app
from hafs.cli.commands.services import services_app
from hafs.cli.commands.history import history_app
from hafs.cli.commands.embed import embed_app
from hafs.cli.commands.nodes import nodes_app
from hafs.cli.commands.sync import sync_app
from hafs.cli.commands.afs import afs_app
from hafs.cli.commands.context import context_app
from hafs.cli.commands.memory import memory_app
from hafs.cli.commands.chat import chat_app

app = typer.Typer(
    name="hafs",
    help="""
\b
 _   _    _    _____ ____  
| | | |  / \\  |  ___/ ___| 
| |_| | / _ \\ | |_  \\___ \\ 
|  _  |/ ___ \\|  _|  ___) |
|_| |_/_/   \\_\\_|   |____/ 

HAFS - Halext Agentic File System
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


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Launch TUI by default when no command is specified."""
    if ctx.invoked_subcommand is None:
        from hafs.ui.app import run

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
