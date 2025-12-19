"""HAFS command-line interface."""

from __future__ import annotations

import asyncio
import importlib.metadata
from typing import Optional

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

# --- Embedding Subcommand ---
embed_app = typer.Typer(name="embed", help="Manage embedding generation (daemon + indexer)")
app.add_typer(embed_app)

# --- Nodes Subcommand ---
nodes_app = typer.Typer(name="nodes", help="Manage distributed node registry")
app.add_typer(nodes_app)

# --- Sync Subcommand ---
sync_app = typer.Typer(name="sync", help="Sync AFS data across nodes")
app.add_typer(sync_app)


def _parse_agent_spec(value: str):
    parts = [p.strip() for p in value.split(":") if p.strip()]
    if len(parts) < 2:
        raise typer.BadParameter("agent spec must be name:role[:persona]")
    name = parts[0]
    role_str = parts[1].lower()
    persona = parts[2] if len(parts) > 2 else None

    from hafs.models.agent import AgentRole
    from hafs.core.orchestration_entrypoint import AgentSpec

    try:
        role = AgentRole(role_str)
    except ValueError as exc:
        valid = ", ".join(r.value for r in AgentRole)
        raise typer.BadParameter(f"invalid role '{role_str}'. Valid: {valid}") from exc

    return AgentSpec(name=name, role=role, persona=persona)


@app.command("orchestrate")
def orchestrate(
    topic: str = typer.Argument(..., help="Orchestration topic/task"),
    mode: str = typer.Option("coordinator", help="coordinator|swarm"),
    agent: list[str] = typer.Option(
        None,
        "--agent",
        help="Agent spec: name:role[:persona] (repeatable)",
    ),
    backend: str = typer.Option("gemini", help="Default backend for coordinator mode"),
) -> None:
    """Run a plan→execute→verify→summarize orchestration pipeline."""
    from hafs.core.orchestration_entrypoint import run_orchestration

    agent_specs = [_parse_agent_spec(spec) for spec in agent] if agent else None

    async def _run() -> None:
        result = await run_orchestration(
            mode=mode,
            topic=topic,
            agents=agent_specs,
            default_backend=backend,
        )
        if result:
            console.print(result)

    asyncio.run(_run())


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


# --- Embedding Daemon Commands ---
@embed_app.command("status")
def embed_status() -> None:
    """Show embedding daemon status."""
    from hafs.services.embedding_daemon import get_status

    status = get_status()

    if status.get("running"):
        console.print("[green]Daemon Status: Running[/green]")
        console.print(f"  PID: {status.get('pid', 'unknown')}")
    else:
        console.print("[dim]Daemon Status: Stopped[/dim]")

    if "total_symbols" in status:
        total = status["total_symbols"]
        done = status["total_embeddings"]
        pct = status.get("coverage_percent", 0)
        console.print(f"\n[bold]Coverage:[/bold] {done:,}/{total:,} ({pct}%)")
        console.print(f"[bold]Daily count:[/bold] {status.get('daily_count', 0)}/{status.get('daily_limit', 5000)}")

    if "last_update" in status:
        console.print(f"[dim]Last update: {status['last_update']}[/dim]")


@embed_app.command("index")
def embed_index(
    project: Optional[str] = typer.Argument(
        None,
        help="Project name to index (defaults to all configured projects)",
    ),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Sync projects from hafs.toml before indexing",
    ),
) -> None:
    """Run embedding indexer for configured projects."""
    async def _index() -> None:
        from hafs.services.embedding_service import EmbeddingService

        service = EmbeddingService()
        if sync:
            service.sync_projects_from_registry()

        if project:
            resolved = service.resolve_project(project)
            if not resolved:
                console.print(f"[red]Unknown project: {project}[/red]")
                raise typer.Exit(1)
            names = [resolved.name]
        else:
            names = [p.name for p in service.get_projects() if p.enabled]
            if not names:
                console.print("[yellow]No projects configured for indexing[/yellow]")
                return

        await service.run_indexing(names)
        console.print("[green]Indexing complete[/green]")

    asyncio.run(_index())


@embed_app.command("start")
def embed_start(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Embeddings per batch"),
    interval: int = typer.Option(60, "--interval", "-i", help="Seconds between batches"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
) -> None:
    """Start the embedding daemon."""
    import subprocess
    import sys

    if foreground:
        # Run in foreground
        from hafs.services.embedding_daemon import EmbeddingDaemon
        daemon = EmbeddingDaemon(batch_size=batch_size, interval_seconds=interval)
        asyncio.run(daemon.start())
    else:
        # Run in background
        cmd = [
            sys.executable, "-m", "hafs.services.embedding_daemon",
            "--batch-size", str(batch_size),
            "--interval", str(interval),
        ]
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        console.print(f"[green]Started embedding daemon (PID: {proc.pid})[/green]")
        console.print(f"[dim]Log: ~/.context/logs/embedding_daemon.log[/dim]")


@embed_app.command("stop")
def embed_stop() -> None:
    """Stop the embedding daemon."""
    import os
    import signal
    from pathlib import Path

    pid_file = Path.home() / ".context" / "embedding_service" / "daemon.pid"

    if not pid_file.exists():
        console.print("[dim]Daemon not running[/dim]")
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Stopped daemon (PID: {pid})[/green]")
    except ProcessLookupError:
        console.print("[dim]Daemon not running (stale PID file)[/dim]")
        pid_file.unlink()
    except Exception as e:
        console.print(f"[red]Error stopping daemon: {e}[/red]")


@embed_app.command("install")
def embed_install() -> None:
    """Install embedding daemon as launchd service (macOS)."""
    from hafs.services.embedding_daemon import install_launchd
    install_launchd()


@embed_app.command("uninstall")
def embed_uninstall() -> None:
    """Uninstall embedding daemon launchd service."""
    from hafs.services.embedding_daemon import uninstall_launchd
    uninstall_launchd()


@embed_app.command("quick")
def embed_quick(
    count: int = typer.Argument(100, help="Number of embeddings to generate"),
) -> None:
    """Generate embeddings inline (not as daemon)."""
    async def _quick() -> None:
        from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase
        from hafs.core.orchestrator_v2 import UnifiedOrchestrator

        kb = ALTTPKnowledgeBase()
        await kb.setup()

        stats = kb.get_statistics()
        console.print(f"[bold]Before:[/bold] {stats['total_embeddings']:,} embeddings")

        # Get missing
        existing = set(kb._embeddings.keys())
        missing = [s for s in kb._symbols.values() if s.id not in existing and s.description][:count]

        if not missing:
            console.print("[green]All symbols have embeddings![/green]")
            return

        console.print(f"Generating {len(missing)} embeddings...")

        orchestrator = UnifiedOrchestrator(log_thoughts=False)
        await orchestrator.initialize()

        generated = 0
        for i, symbol in enumerate(missing):
            try:
                text = f"{symbol.name}: {symbol.description}"
                result = await orchestrator.embed(text)
                if result:
                    kb._embeddings[symbol.id] = result
                    kb._save_embedding(symbol.id, result)
                    generated += 1

                if (i + 1) % 20 == 0:
                    console.print(f"  Progress: {i+1}/{len(missing)}")
            except Exception as e:
                pass

        stats = kb.get_statistics()
        console.print(f"[bold]After:[/bold] {stats['total_embeddings']:,} embeddings (+{generated})")

    asyncio.run(_quick())


@nodes_app.command("list")
def nodes_list() -> None:
    """List configured nodes."""
    from hafs.core.nodes import node_manager

    async def _list() -> None:
        await node_manager.load_config()
        if not node_manager.nodes:
            console.print("[yellow]No nodes configured[/yellow]")
            return

        console.print("\n[bold]Nodes:[/bold]")
        for node in node_manager.nodes:
            caps = ", ".join(node.capabilities) if node.capabilities else "none"
            console.print(
                f"  [bold]{node.name}[/bold] {node.host}:{node.port} "
                f"[dim]{node.node_type}[/dim] [dim]{node.platform}[/dim] "
                f"[dim]capabilities: {caps}[/dim]"
            )

    asyncio.run(_list())


@nodes_app.command("status")
def nodes_status() -> None:
    """Check node health status."""
    from hafs.core.nodes import node_manager

    async def _status() -> None:
        await node_manager.load_config()
        await node_manager.health_check_all()
        console.print(node_manager.summary())

    asyncio.run(_status())


@nodes_app.command("show")
def nodes_show(name: str = typer.Argument(..., help="Node name")) -> None:
    """Show detailed node configuration."""
    from hafs.core.nodes import node_manager

    async def _show() -> None:
        await node_manager.load_config()
        node = node_manager.get_node(name)
        if not node:
            console.print(f"[red]Unknown node: {name}[/red]")
            raise typer.Exit(1)
        data = node.to_dict()
        for key in sorted(data.keys()):
            console.print(f"{key}: {data[key]}")

    asyncio.run(_show())


@nodes_app.command("discover")
def nodes_discover() -> None:
    """Discover Ollama nodes on Tailscale."""
    from hafs.core.nodes import node_manager

    async def _discover() -> None:
        await node_manager.load_config()
        found = await node_manager.discover_tailscale_nodes()
        if not found:
            console.print("[yellow]No Tailscale nodes discovered[/yellow]")
            return
        console.print("[green]Discovered nodes:[/green]")
        for node in found:
            console.print(f"  {node.name} {node.host}:{node.port}")

    asyncio.run(_discover())


@sync_app.command("list")
def sync_list() -> None:
    """List configured sync profiles."""
    from hafs.services.afs_sync import AFSSyncService

    async def _list() -> None:
        service = AFSSyncService()
        profiles = await service.load()
        if not profiles:
            console.print("[yellow]No sync profiles configured[/yellow]")
            return
        console.print("\n[bold]Sync Profiles:[/bold]")
        for profile in profiles:
            targets = ", ".join(t.label() for t in profile.targets) or "none"
            console.print(
                f"  [bold]{profile.name}[/bold] "
                f"[dim]{profile.scope}[/dim] "
                f"[dim]{profile.direction}[/dim] "
                f"[dim]targets: {targets}[/dim]"
            )

    asyncio.run(_list())


@sync_app.command("show")
def sync_show(name: str = typer.Argument(..., help="Profile name")) -> None:
    """Show details for a sync profile."""
    from hafs.services.afs_sync import AFSSyncService

    async def _show() -> None:
        service = AFSSyncService()
        await service.load()
        profile = service.resolve_profile(name)
        if not profile:
            console.print(f"[red]Unknown sync profile: {name}[/red]")
            raise typer.Exit(1)
        console.print(f"name: {profile.name}")
        console.print(f"source: {profile.source}")
        console.print(f"scope: {profile.scope}")
        console.print(f"direction: {profile.direction}")
        console.print(f"transport: {profile.transport}")
        console.print(f"delete: {profile.delete}")
        console.print(f"exclude: {profile.exclude}")
        console.print(f"targets: {[t.label() for t in profile.targets]}")

    asyncio.run(_show())


@sync_app.command("run")
def sync_run(
    name: str = typer.Argument(..., help="Profile name"),
    direction: Optional[str] = typer.Option(
        None,
        "--direction",
        help="push | pull | bidirectional",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview rsync actions"),
) -> None:
    """Run a sync profile."""
    from hafs.services.afs_sync import AFSSyncService

    async def _run() -> None:
        service = AFSSyncService()
        await service.load()
        results = await service.run_profile(name, direction_override=direction, dry_run=dry_run)
        if not results:
            console.print("[yellow]No sync actions executed[/yellow]")
            return
        ok = sum(1 for r in results if r.ok)
        console.print(f"\n[bold]Results:[/bold] {ok}/{len(results)} succeeded")
        for result in results:
            status = "green" if result.ok else "red"
            console.print(
                f"  [{status}]{result.direction}[/] {result.target} "
                f"(exit {result.exit_code})"
            )
            if result.stderr:
                console.print(f"    [dim]{result.stderr.strip()}[/dim]")

    asyncio.run(_run())


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
