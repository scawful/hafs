"""HAFS command-line interface."""

from __future__ import annotations

import asyncio
import importlib.metadata

import typer
from rich.console import Console

from hafs.config.loader import load_config

# --- Main App ---
app = typer.Typer(
    name="hafs",
    help="HAFS - Halext Agentic File System",
    invoke_without_command=True,
)
console = Console()

# --- Services Subcommand ---
services_app = typer.Typer(name="services", help="Manage HAFS background services")
app.add_typer(services_app)

# --- History Subcommand ---
history_app = typer.Typer(name="history", help="Manage AFS history embeddings")
app.add_typer(history_app)


@services_app.command("list")
def services_list() -> None:
    """List all services and their status."""
    from hafs.core.services import ServiceManager, ServiceState

    async def _list() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        statuses = await manager.status_all()

        console.print(f"\n[bold]Platform:[/bold] {manager.platform_name}\n")

        for name, status in sorted(statuses.items()):
            state_color = {
                ServiceState.RUNNING: "green",
                ServiceState.STOPPED: "dim",
                ServiceState.FAILED: "red",
            }.get(status.state, "white")

            indicator = "[green]\u25cf[/]" if status.state == ServiceState.RUNNING else "[dim]\u25cb[/]"
            console.print(f"  {indicator} [{state_color}]{name}[/]: {status.state.value}")
            if status.pid:
                console.print(f"      PID: {status.pid}")
            if status.enabled:
                console.print("      [dim]installed[/dim]")

    asyncio.run(_list())


@services_app.command("start")
def services_start(name: str = typer.Argument(..., help="Service name")) -> None:
    """Start a service."""
    from hafs.core.services import ServiceManager

    async def _start() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        definition = manager.get_service_definition(name)

        if not definition:
            console.print(f"[red]Unknown service: {name}[/red]")
            console.print(f"[dim]Available services: {', '.join(manager.list_services())}[/dim]")
            raise typer.Exit(1)

        # Install if needed
        await manager.install(definition)
        success = await manager.start(name)

        if success:
            console.print(f"[green]Started {name}[/green]")
        else:
            console.print(f"[red]Failed to start {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_start())


@services_app.command("stop")
def services_stop(name: str = typer.Argument(..., help="Service name")) -> None:
    """Stop a service."""
    from hafs.core.services import ServiceManager

    async def _stop() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.stop(name)

        if success:
            console.print(f"[green]Stopped {name}[/green]")
        else:
            console.print(f"[red]Failed to stop {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_stop())


@services_app.command("restart")
def services_restart(name: str = typer.Argument(..., help="Service name")) -> None:
    """Restart a service."""
    from hafs.core.services import ServiceManager

    async def _restart() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.restart(name)

        if success:
            console.print(f"[green]Restarted {name}[/green]")
        else:
            console.print(f"[red]Failed to restart {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_restart())


@services_app.command("logs")
def services_logs(
    name: str = typer.Argument(..., help="Service name"),
    follow: bool = typer.Option(False, "-f", "--follow", help="Follow log output"),
    lines: int = typer.Option(100, "-n", "--lines", help="Number of lines to show"),
) -> None:
    """View service logs."""
    from hafs.core.services import ServiceManager

    async def _logs() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        if follow:
            console.print(f"[dim]Following logs for {name}... (Ctrl+C to stop)[/dim]\n")
            try:
                async for line in manager.stream_logs(name):
                    console.print(line, end="")
            except KeyboardInterrupt:
                console.print("\n[dim]Stopped following logs[/dim]")
        else:
            logs = await manager.logs(name, lines)
            console.print(logs)

    asyncio.run(_logs())


@history_app.command("status")
def history_status() -> None:
    """Show history embedding index status."""
    from hafs.core.history import HistoryEmbeddingIndex, HistorySessionSummaryIndex

    config = load_config()
    index = HistoryEmbeddingIndex(config.general.context_root)
    status = index.status()
    summaries = HistorySessionSummaryIndex(config.general.context_root).status()
    console.print("[bold]History Index Status[/bold]")
    console.print(f"- History files: {status['history_files']}")
    console.print(f"- Embeddings: {status['embeddings']}")
    console.print(f"- Sessions: {summaries['sessions']}")
    console.print(f"- Summaries: {summaries['summaries']}")


@history_app.command("index")
def history_index(limit: int = typer.Option(200, help="Max new entries to embed")) -> None:
    """Index new history entries into embeddings."""
    from hafs.core.history import HistoryEmbeddingIndex

    async def _index() -> None:
        config = load_config()
        index = HistoryEmbeddingIndex(config.general.context_root)
        created = await index.index_new_entries(limit=limit)
        console.print(f"[green]Indexed {created} new entries[/green]")

    asyncio.run(_index())


@history_app.command("summarize")
def history_summarize(
    session_id: str | None = typer.Option(None, help="Summarize a specific session"),
    limit: int = typer.Option(20, help="Max sessions to summarize"),
) -> None:
    """Generate session summaries and embeddings."""
    from hafs.core.history import HistorySessionSummaryIndex

    async def _summarize() -> None:
        config = load_config()
        index = HistorySessionSummaryIndex(config.general.context_root)
        if session_id:
            summary = await index.summarize_session(session_id)
            if summary:
                console.print(f"[green]Summarized session {session_id}[/green]")
            else:
                console.print(f"[yellow]No entries for session {session_id}[/yellow]")
            return

        created = await index.index_missing_summaries(limit=limit)
        console.print(f"[green]Created {created} summaries[/green]")

    asyncio.run(_summarize())


@history_app.command("search")
def history_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
    refresh: bool = typer.Option(False, help="Index new entries before searching"),
    sessions: bool = typer.Option(False, help="Search session summaries instead of entries"),
    all_results: bool = typer.Option(False, "--all", help="Search entries and sessions"),
    mode: str | None = typer.Option(None, help="entries|sessions|all"),
) -> None:
    """Semantic search over history embeddings."""
    from hafs.core.history import HistoryEmbeddingIndex, HistorySessionSummaryIndex

    async def _search() -> None:
        selected_mode = (mode or "").strip().lower()
        if not selected_mode:
            selected_mode = "sessions" if sessions else "entries"
        if all_results:
            selected_mode = "all"
        if selected_mode not in {"entries", "sessions", "all"}:
            raise typer.BadParameter("mode must be entries, sessions, or all")

        config = load_config()
        if selected_mode == "sessions":
            index = HistorySessionSummaryIndex(config.general.context_root)
            if refresh:
                await index.index_missing_summaries(limit=50)
            results = await index.search(query, limit=limit)
        elif selected_mode == "entries":
            index = HistoryEmbeddingIndex(config.general.context_root)
            if refresh:
                await index.index_new_entries(limit=200)
            results = await index.search(query, limit=limit)
        else:
            entry_index = HistoryEmbeddingIndex(config.general.context_root)
            summary_index = HistorySessionSummaryIndex(config.general.context_root)
            if refresh:
                await entry_index.index_new_entries(limit=200)
                await summary_index.index_missing_summaries(limit=50)
            entry_results = await entry_index.search(query, limit=limit)
            session_results = await summary_index.search(query, limit=limit)
            results = [
                {"kind": "entry", **result} for result in entry_results
            ] + [
                {"kind": "session", **result} for result in session_results
            ]
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = results[:limit]

        if not results:
            console.print("[yellow]No matches found.[/yellow]")
            return

        console.print(f"[bold]Results for[/bold] '{query}':")
        for idx, result in enumerate(results, start=1):
            score = result.get("score", 0.0)
            kind = result.get("kind", selected_mode)
            if kind in {"session", "sessions"}:
                created_at = result.get("created_at")
                session_id = result.get("session_id")
                title = result.get("title") or "Session summary"
                summary = result.get("summary", "")
                console.print(
                    f"{idx}. [S][{score:.2f}] {created_at} {session_id} {title}\n    {summary}"
                )
            else:
                timestamp = result.get("timestamp")
                session_id = result.get("session_id")
                op_type = result.get("operation_type")
                name = result.get("name")
                preview = result.get("preview")
                console.print(
                    f"{idx}. [E][{score:.2f}] {timestamp} {session_id} {op_type}/{name}\n    {preview}"
                )

    asyncio.run(_search())


@services_app.command("install")
def services_install(name: str = typer.Argument(..., help="Service name")) -> None:
    """Install service configuration files."""
    from hafs.core.services import ServiceManager

    async def _install() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        definition = manager.get_service_definition(name)

        if not definition:
            console.print(f"[red]Unknown service: {name}[/red]")
            console.print(f"[dim]Available services: {', '.join(manager.list_services())}[/dim]")
            raise typer.Exit(1)

        success = await manager.install(definition)

        if success:
            console.print(f"[green]Installed {name}[/green]")
        else:
            console.print(f"[red]Failed to install {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_install())


@services_app.command("uninstall")
def services_uninstall(name: str = typer.Argument(..., help="Service name")) -> None:
    """Uninstall service configuration files."""
    from hafs.core.services import ServiceManager

    async def _uninstall() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.uninstall(name)

        if success:
            console.print(f"[green]Uninstalled {name}[/green]")
        else:
            console.print(f"[red]Failed to uninstall {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_uninstall())


@services_app.command("enable")
def services_enable(name: str = typer.Argument(..., help="Service name")) -> None:
    """Enable service to start at login."""
    from hafs.core.services import ServiceManager

    async def _enable() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.enable(name)

        if success:
            console.print(f"[green]Enabled auto-start for {name}[/green]")
        else:
            console.print(f"[red]Failed to enable {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_enable())


@services_app.command("disable")
def services_disable(name: str = typer.Argument(..., help="Service name")) -> None:
    """Disable service from starting at login."""
    from hafs.core.services import ServiceManager

    async def _disable() -> None:
        try:
            manager = ServiceManager(load_config())
        except NotImplementedError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

        success = await manager.disable(name)

        if success:
            console.print(f"[green]Disabled auto-start for {name}[/green]")
        else:
            console.print(f"[red]Failed to disable {name}[/red]")
            raise typer.Exit(1)

    asyncio.run(_disable())


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
