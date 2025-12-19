"""HAFS command-line interface."""

from __future__ import annotations
from pathlib import Path
import importlib.metadata

import typer
from rich.console import Console

# --- Main App ---
app = typer.Typer(
    name="hafs",
    help="HAFS - Halext Agentic File System",
    invoke_without_command=True,
)
console = Console()

# --- Plugin System ---
def load_plugins():
    """Discover and load Typer app plugins from entry points."""
    # For commands like 'hafs google ...'
    command_entry_points = importlib.metadata.entry_points(group="hafs.commands")
    for entry in command_entry_points:
        plugin_app = entry.load()
        app.add_typer(plugin_app, name=entry.name)
    
    # For project discovery plugins
    # (This part of your plugin architecture was not fully implemented,
    # but the entry point exists in your pyproject.toml, so we honor it)
    plugin_entry_points = importlib.metadata.entry_points(group="hafs.plugins")
    # You would iterate here and register them to a manager if needed

# Load plugins at startup
load_plugins()


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Launch TUI by default when no command is specified."""
    if ctx.invoked_subcommand is None:
        from hafs.ui.app import run
        run()

# We will add back other commands like 'init', 'list', etc. later.
# The priority is to fix the plugin system.

def main() -> None:
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
