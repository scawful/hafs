"""HAFS command-line interface."""

from __future__ import annotations

import asyncio
import importlib.metadata
from typing import Optional

import typer
from rich.console import Console

from hafs.config.loader import load_config
from hafs.ui.console import services as ui_services
from hafs.ui.console import history as ui_history
from hafs.ui.console import embed as ui_embed
from hafs.ui.console import orchestrator as ui_orchestrator
from hafs.ui.console import memory as ui_memory
from hafs.ui.console import nodes as ui_nodes
from hafs.ui.console import sync as ui_sync

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
            ui_orchestrator.render_orchestration_result(console, result)

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
        ui_services.render_service_list(console, statuses, manager.platform_name)

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
            ui_services.render_unknown_service(console, name, manager.list_services())
            raise typer.Exit(1)

        # Install if needed
        await manager.install(definition)
        success = await manager.start(name)

        ui_services.render_start_result(console, name, success)

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
        ui_services.render_stop_result(console, name, success)

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
        ui_services.render_restart_result(console, name, success)

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
            await ui_services.stream_logs(console, manager, name)
        else:
            logs = await manager.logs(name, lines)
            ui_services.render_service_logs(console, name, logs)

    asyncio.run(_logs())


@history_app.command("status")
def history_status() -> None:
    """Show history embedding index status."""
    from hafs.core.history import HistoryEmbeddingIndex, HistorySessionSummaryIndex

    config = load_config()
    index = HistoryEmbeddingIndex(config.general.context_root)
    status = index.status()
    summaries = HistorySessionSummaryIndex(config.general.context_root).status()
    ui_history.render_history_status(console, status, summaries)


@history_app.command("index")
def history_index(limit: int = typer.Option(200, help="Max new entries to embed")) -> None:
    """Index new history entries into embeddings."""
    from hafs.core.history import HistoryEmbeddingIndex

    async def _index() -> None:
        config = load_config()
        index = HistoryEmbeddingIndex(config.general.context_root)
        created = await index.index_new_entries(limit=limit)
        ui_history.render_index_result(console, created)

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
            ui_history.render_session_summary_result(console, session_id, bool(summary))
            return

        created = await index.index_missing_summaries(limit=limit)
        ui_history.render_summaries_created(console, created)

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

        ui_history.render_search_results(console, query, results, selected_mode)

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
            ui_services.render_unknown_service(console, name, manager.list_services())
            raise typer.Exit(1)

        success = await manager.install(definition)
        ui_services.render_install_result(console, name, success)

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
        ui_services.render_uninstall_result(console, name, success)

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
        ui_services.render_enable_result(console, name, success)

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
        ui_services.render_disable_result(console, name, success)

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
    ui_embed.render_daemon_status(console, status)


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
        ui_embed.render_indexing_complete(console)

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
        ui_embed.render_daemon_started(console, proc.pid)


@embed_app.command("stop")
def embed_stop() -> None:
    """Stop the embedding daemon."""
    import os
    import signal
    from pathlib import Path

    pid_file = Path.home() / ".context" / "embedding_service" / "daemon.pid"

    if not pid_file.exists():
        ui_embed.render_daemon_not_running(console)
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        ui_embed.render_daemon_stopped(console, pid)
    except ProcessLookupError:
        console.print("[dim]Daemon not running (stale PID file)[/dim]")  # Keep this as specific error logic or add a render method?
        # Let's add a render method for stale pid if we want to be strict, but for now I'll just use the existing not_running one or a new one. 
        # Actually I missed adding a specific "stale pid" render to embed.py. I'll just leave this print or use render_daemon_not_running with a note?
        # I'll stick to replacing what matches.
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
        ui_embed.render_quick_stats(console, stats, "Before")

        # Get missing
        existing = set(kb._embeddings.keys())
        missing = [s for s in kb._symbols.values() if s.id not in existing and s.description][:count]

        if not missing:
            ui_embed.render_all_symbols_have_embeddings(console)
            return

        ui_embed.render_generating_embeddings(console, len(missing))

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
                    ui_embed.render_quick_progress(console, i + 1, len(missing))
            except Exception as e:
                pass

        stats = kb.get_statistics()
        ui_embed.render_quick_stats(console, stats, "After")

    asyncio.run(_quick())


@embed_app.command("enhance")
def embed_enhance(
    kb: str = typer.Option("alttp", help="Knowledge base to enhance (alttp, oracle)"),
    patterns: bool = typer.Option(True, help="Generate code pattern embeddings"),
    hubs: bool = typer.Option(True, help="Generate relationship hub embeddings"),
    regions: bool = typer.Option(True, help="Generate WRAM region embeddings"),
    tags: bool = typer.Option(True, help="Generate semantic tag embeddings"),
    banks: bool = typer.Option(True, help="Generate bank embeddings"),
    modules: bool = typer.Option(True, help="Generate module embeddings"),
) -> None:
    """Generate enhanced embeddings with rich context for ALTTP KBs."""

    async def _enhance() -> None:
        from hafs.agents.alttp_embeddings import ALTTPEmbeddingBuilder
        from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase

        ui_embed.render_enhance_start(console, kb)

        kb_instance = ALTTPKnowledgeBase()
        await kb_instance.setup()

        # Get symbols and routines
        symbols = [s.to_dict() for s in kb_instance._symbols.values()]
        routines = [r.to_dict() for r in kb_instance._routines.values()]
        modules_data = []
        if kb_instance._modules:
            from dataclasses import asdict

            modules_data = [asdict(m) for m in kb_instance._modules.values()]

        ui_embed.render_found_stats(console, len(symbols), len(routines), len(modules_data))

        builder = ALTTPEmbeddingBuilder(kb_instance.kb_dir)
        await builder.setup()

        total_created = 0

        # Enrich and embed symbols
        ui_embed.render_enhance_phase(console, "enriched symbol embeddings")
        symbol_items = []
        for sym in symbols[:500]:  # Limit for first run
            item = builder.enrich_symbol(
                symbol_id=sym.get("id", f"symbol:{sym.get('name', '')}"),
                name=sym.get("name", ""),
                address=sym.get("address", ""),
                category=sym.get("category", ""),
                description=sym.get("description", ""),
                references=sym.get("references", []),
                referenced_by=sym.get("referenced_by", []),
                bank=sym.get("bank"),
                semantic_tags=sym.get("semantic_tags", []),
                file_path=sym.get("file_path"),
                line_number=sym.get("line_number"),
                code_context=sym.get("code_context", ""),
            )
            symbol_items.append(item)

        if symbol_items:
            result = await builder.generate_embeddings(symbol_items, kb_name="alttp_symbols_enriched")
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "symbol")

        # Enrich and embed routines
        ui_embed.render_enhance_phase(console, "enriched routine embeddings")
        routine_items = []
        symbol_lookup = {sym.get("name", ""): sym for sym in symbols if sym.get("name")}
        for routine in routines[:200]:  # Limit for first run
            item = builder.enrich_routine(
                routine_name=routine.get("name", ""),
                address=routine.get("address", ""),
                bank=routine.get("bank", ""),
                description=routine.get("description", ""),
                purpose=routine.get("purpose", ""),
                complexity=routine.get("complexity", ""),
                calls=routine.get("calls", []),
                called_by=routine.get("called_by", []),
                memory_access=routine.get("memory_access", []),
                code_snippet=routine.get("code", "")[:500],
                file_path=routine.get("file_path", ""),
                line_start=routine.get("line_start"),
                line_end=routine.get("line_end"),
                symbol_lookup=symbol_lookup,
            )
            routine_items.append(item)

        if routine_items:
            result = await builder.generate_embeddings(routine_items, kb_name="alttp_routines_enriched")
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "routine")

        # Generate pattern embeddings
        if patterns and routines:
            ui_embed.render_enhance_phase(console, "code pattern embeddings")
            result = await builder.generate_code_pattern_embeddings(routines)
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "pattern")

        # Generate hub embeddings
        if hubs and routines:
            ui_embed.render_enhance_phase(console, "relationship hub embeddings")
            result = await builder.generate_relationship_embeddings(routines)
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "hub")

        # Generate memory region embeddings
        if regions and symbols:
            ui_embed.render_enhance_phase(console, "WRAM region embeddings")
            result = await builder.generate_memory_region_embeddings(symbols, routines)
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "region")

        # Generate semantic tag embeddings
        if tags and symbols:
            ui_embed.render_enhance_phase(console, "semantic tag embeddings")
            result = await builder.generate_semantic_tag_embeddings(symbols)
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "tag")

        # Generate bank embeddings
        if banks and routines:
            ui_embed.render_enhance_phase(console, "bank embeddings")
            result = await builder.generate_bank_embeddings(routines)
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "bank")

        # Generate module embeddings
        if modules and modules_data:
            ui_embed.render_enhance_phase(console, "module embeddings")
            result = await builder.generate_module_embeddings(modules_data)
            total_created += result.get("created", 0)
            ui_embed.render_enhance_stats(console, result.get("created", 0), "module")

        ui_embed.render_total_enhanced(console, total_created)

    asyncio.run(_enhance())


# --- Agent Memory Commands ---
memory_app = typer.Typer(name="memory", help="Manage agent memory and history")
app.add_typer(memory_app)


@memory_app.command("status")
def memory_status(
    agent: Optional[str] = typer.Option(None, help="Specific agent ID"),
) -> None:
    """Show agent memory status."""
    from pathlib import Path
    from hafs.core.history import AgentMemoryManager

    config = load_config()
    manager = AgentMemoryManager(config.general.context_root)

    if agent:
        # Show specific agent
        memory = manager.get_agent_memory(agent)
        stats = memory.get_stats()
        console.print(f"[bold]Agent: {agent}[/bold]")
        console.print(f"  Total entries: {stats['total_entries']}")
        console.print(f"  Working memory: {stats['working_memory']}")
        console.print(f"  Recent memory: {stats['recent_memory']}")
        console.print(f"  Archive memory: {stats['archive_memory']}")
        console.print(f"  Session summaries: {stats['total_summaries']}")
        if stats.get('by_type'):
            console.print("  By type:")
            for mtype, count in stats['by_type'].items():
                console.print(f"    {mtype}: {count}")
    else:
        # Show all agents
        agents = manager.list_agents()
        if not agents:
            console.print("[dim]No agents with memory found[/dim]")
            return

        console.print(f"[bold]Agents with memory: {len(agents)}[/bold]")
        for agent_id in agents:
            memory = manager.get_agent_memory(agent_id)
            stats = memory.get_stats()
            console.print(
                f"  {agent_id}: {stats['total_entries']} entries, "
                f"{stats['total_summaries']} summaries"
            )


@memory_app.command("recall")
def memory_recall(
    query: str = typer.Argument(..., help="Search query"),
    agent: str = typer.Option(..., help="Agent ID to search"),
    limit: int = typer.Option(10, help="Max results"),
    bucket: str = typer.Option("all", help="Temporal bucket: working, recent, archive, all"),
    recency: float = typer.Option(0.3, help="Recency weight (0-1)"),
) -> None:
    """Search an agent's memory with temporal awareness."""
    from hafs.core.history import AgentMemoryManager

    async def _recall() -> None:
        config = load_config()
        manager = AgentMemoryManager(config.general.context_root)
        memory = manager.get_agent_memory(agent)

        results = await memory.recall(
            query=query,
            limit=limit,
            temporal_bucket=bucket,
            recency_weight=recency,
        )

        if not results:
            console.print("[yellow]No memories found[/yellow]")
            return

        console.print(f"[bold]Memories matching '{query}':[/bold]")
        for result in results:
            entry = result["entry"]
            console.print(
                f"\n  [{entry['memory_type']}] "
                f"score={result['score']:.3f} "
                f"({result['temporal_bucket']})"
            )
            console.print(f"  [dim]{entry['content'][:200]}...[/dim]")

    asyncio.run(_recall())


@memory_app.command("remember")
def memory_remember(
    content: str = typer.Argument(..., help="Content to remember"),
    agent: str = typer.Option(..., help="Agent ID"),
    memory_type: str = typer.Option("insight", help="Type: decision, interaction, learning, error, insight"),
    importance: float = typer.Option(0.5, help="Importance (0-1)"),
) -> None:
    """Store a memory for an agent."""
    from hafs.core.history import AgentMemoryManager

    async def _remember() -> None:
        config = load_config()
        manager = AgentMemoryManager(config.general.context_root)
        memory = manager.get_agent_memory(agent)

        entry = await memory.remember(
            content=content,
            memory_type=memory_type,
            importance=importance,
        )

        ui_memory.render_stored_memory(console, entry.id, entry.memory_type, entry.importance)

    asyncio.run(_remember())


@memory_app.command("cross-search")
def memory_cross_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
) -> None:
    """Search across all agents' memories."""
    from hafs.core.history import AgentMemoryManager

    async def _search() -> None:
        config = load_config()
        manager = AgentMemoryManager(config.general.context_root)

        results = await manager.cross_agent_search(query, limit=limit)

        ui_memory.render_cross_search_results(console, query, results)

    asyncio.run(_search())


@nodes_app.command("list")
def nodes_list() -> None:
    """List configured nodes."""
    from hafs.core.nodes import node_manager

    async def _list() -> None:
        await node_manager.load_config()
        ui_nodes.render_nodes_list(console, node_manager.nodes)

    asyncio.run(_list())


@nodes_app.command("status")
def nodes_status() -> None:
    """Check node health status."""
    from hafs.core.nodes import node_manager

    async def _status() -> None:
        await node_manager.load_config()
        await node_manager.health_check_all()
        ui_nodes.render_nodes_status(console, node_manager.summary())

    asyncio.run(_status())


@nodes_app.command("show")
def nodes_show(name: str = typer.Argument(..., help="Node name")) -> None:
    """Show detailed node configuration."""
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
def nodes_discover() -> None:
    """Discover Ollama nodes on Tailscale."""
    from hafs.core.nodes import node_manager

    async def _discover() -> None:
        await node_manager.load_config()
        found = await node_manager.discover_tailscale_nodes()
        ui_nodes.render_discovered_nodes(console, found)

    asyncio.run(_discover())


@sync_app.command("list")
def sync_list() -> None:
    """List configured sync profiles."""
    from hafs.services.afs_sync import AFSSyncService

    async def _list() -> None:
        service = AFSSyncService()
        profiles = await service.load()
        ui_sync.render_sync_profiles(console, profiles)

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
            ui_sync.render_unknown_profile(console, name)
            raise typer.Exit(1)
        ui_sync.render_sync_profile_details(console, profile)

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
        ui_sync.render_sync_results(console, results)

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
