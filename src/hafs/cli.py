"""HAFS command-line interface."""

from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path
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
from hafs.ui.console import afs as ui_afs
from hafs.ui.console import context as ui_context


# --- Main App ---
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

# --- AFS Subcommand ---
afs_app = typer.Typer(
    name="afs",
    help="Manage Agentic File System (AFS) structure",
)
app.add_typer(afs_app)


@afs_app.callback(invoke_without_command=True)
def afs_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# --- Embedding Subcommand ---
embed_app = typer.Typer(
    name="embed",
    help="Manage embedding generation, stores, and semantic xref (multi-model)",
)
app.add_typer(embed_app)


@embed_app.callback(invoke_without_command=True)
def embed_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# --- History Subcommand ---
history_app = typer.Typer(
    name="history",
    help="Manage AFS history embeddings, summaries, and search",
)
app.add_typer(history_app)


@history_app.callback(invoke_without_command=True)
def history_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# --- Agent Memory Commands ---
memory_app = typer.Typer(
    name="memory",
    help="Manage agent memory, recall, and cross-search",
)
app.add_typer(memory_app)


# --- Nodes Subcommand ---
nodes_app = typer.Typer(
    name="nodes",
    help="Manage distributed node registry and health checks",
)
app.add_typer(nodes_app)


@nodes_app.callback(invoke_without_command=True)
def nodes_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# --- Services Subcommand ---
services_app = typer.Typer(
    name="services",
    help=(
        "Manage background services (orchestrator, coordinator, autonomy-daemon, "
        "embedding-daemon, context-agent-daemon, dashboard)"
    ),
)
app.add_typer(services_app)


@services_app.callback(invoke_without_command=True)
def services_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# --- Sync Subcommand ---
sync_app = typer.Typer(
    name="sync",
    help="Sync AFS data across nodes using sync profiles",
)
app.add_typer(sync_app)


@sync_app.callback(invoke_without_command=True)
def sync_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# --- Context Engineering Subcommand ---
context_app = typer.Typer(
    name="context",
    help="Context Engineering Pipeline - manage context items, construct windows, evaluate quality",
)
app.add_typer(context_app)


@context_app.callback(invoke_without_command=True)
def context_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


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
    ctx: typer.Context,
    topic: Optional[str] = typer.Argument(None, help="Orchestration topic/task"),
    mode: str = typer.Option(
        "coordinator", help="coordinator|swarm (SwarmCouncil multi-agent mode)"
    ),
    agent: list[str] = typer.Option(
        None,
        "--agent",
        help="Agent spec: name:role[:persona] (repeatable; used for council/swarm)",
    ),
    backend: str = typer.Option("gemini", help="Default backend for coordinator mode"),
) -> None:
    """Run a plan→execute→verify→summarize pipeline with coordinator or swarm."""
    if topic is None:
        console.print(ctx.get_help())
        raise typer.Exit()

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
def services_start(
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
def services_stop(
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
def services_restart(
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
def services_logs(
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
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
    refresh: bool = typer.Option(False, help="Index new entries before searching"),
    sessions: bool = typer.Option(False, help="Search session summaries instead of entries"),
    all_results: bool = typer.Option(False, "--all", help="Search entries and sessions"),
    mode: str | None = typer.Option(None, help="entries|sessions|all"),
) -> None:
    """Semantic search over history embeddings."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

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
            results = [{"kind": "entry", **result} for result in entry_results] + [
                {"kind": "session", **result} for result in session_results
            ]
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = results[:limit]

        ui_history.render_search_results(console, query, results, selected_mode)

    asyncio.run(_search())


@services_app.command("install")
def services_install(
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
def services_uninstall(
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
def services_enable(
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
def services_disable(
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
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Embedding provider override (gemini/openai/ollama/halext)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Embedding model override (provider-specific)",
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

        await service.run_indexing(
            names,
            embedding_provider=provider,
            embedding_model=model,
        )
        ui_embed.render_indexing_complete(console)

    asyncio.run(_index())


@embed_app.command("xref")
def embed_xref(
    ctx: typer.Context,
    source: Optional[str] = typer.Argument(None, help="Source project name"),
    target: Optional[str] = typer.Argument(None, help="Target project name"),
    threshold: float = typer.Option(0.75, "--threshold", "-t", help="Minimum cosine similarity"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Top matches per source item"),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider for both projects"),
    model: Optional[str] = typer.Option(None, "--model", help="Model for both projects"),
    source_provider: Optional[str] = typer.Option(
        None, "--source-provider", help="Provider for source project"
    ),
    target_provider: Optional[str] = typer.Option(
        None, "--target-provider", help="Provider for target project"
    ),
    source_model: Optional[str] = typer.Option(
        None, "--source-model", help="Model for source project"
    ),
    target_model: Optional[str] = typer.Option(
        None, "--target-model", help="Model for target project"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output JSON path"),
    batch_size: int = typer.Option(64, "--batch-size", "-b", help="Batch size for similarity"),
    max_sources: Optional[int] = typer.Option(None, "--max-sources", help="Limit source items"),
) -> None:
    """Generate semantic cross-reference report between embedding indexes."""
    if source is None or target is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    try:
        import numpy as np
    except ModuleNotFoundError:
        from hafs.core.runtime import resolve_python_executable

        console.print("[red]Missing dependency: numpy[/red]")
        python_path = resolve_python_executable()
        console.print(f"Install: {python_path} -m pip install numpy")
        raise typer.Exit(1)

    import json
    from datetime import datetime

    from hafs.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    source_config = service.resolve_project(source)
    target_config = service.resolve_project(target)

    if not source_config:
        console.print(f"[red]Unknown source project: {source}[/red]")
        raise typer.Exit(1)
    if not target_config:
        console.print(f"[red]Unknown target project: {target}[/red]")
        raise typer.Exit(1)

    resolved_source_provider = source_provider or provider
    resolved_target_provider = target_provider or provider
    resolved_source_model = source_model or model
    resolved_target_model = target_model or model

    source_dir = service.get_embedding_dir(
        source_config.name,
        embedding_provider=resolved_source_provider,
        embedding_model=resolved_source_model,
    )
    target_dir = service.get_embedding_dir(
        target_config.name,
        embedding_provider=resolved_target_provider,
        embedding_model=resolved_target_model,
    )

    if not source_dir or not source_dir.exists():
        console.print(f"[red]Source embeddings not found: {source_dir}[/red]")
        raise typer.Exit(1)
    if not target_dir or not target_dir.exists():
        console.print(f"[red]Target embeddings not found: {target_dir}[/red]")
        raise typer.Exit(1)

    def load_embeddings(emb_dir: Path, include_preview: bool, limit: Optional[int]):
        ids = []
        previews = []
        vectors = []
        count = 0

        for path in sorted(emb_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            emb = data.get("embedding")
            if not emb:
                continue
            ids.append(data.get("id", path.name))
            if include_preview:
                previews.append(data.get("text_preview", ""))
            vectors.append(emb)
            count += 1
            if limit and count >= limit:
                break

        if not vectors:
            return ids, previews, np.zeros((0, 0), dtype=np.float32)

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return ids, previews, matrix / norms

    source_ids, source_previews, source_matrix = load_embeddings(
        source_dir,
        include_preview=True,
        limit=max_sources,
    )
    target_ids, _, target_matrix = load_embeddings(
        target_dir,
        include_preview=False,
        limit=None,
    )

    if source_matrix.size == 0 or target_matrix.size == 0:
        console.print("[red]Missing embeddings for source or target[/red]")
        raise typer.Exit(1)

    results = []
    total_matches = 0
    best_score = 0.0

    for start in range(0, source_matrix.shape[0], batch_size):
        batch = source_matrix[start : start + batch_size]
        scores = batch @ target_matrix.T

        for row_idx, row in enumerate(scores):
            src_idx = start + row_idx
            if top_k >= row.size:
                idx = np.argsort(-row)
            else:
                idx = np.argpartition(-row, top_k - 1)[:top_k]
                idx = idx[np.argsort(-row[idx])]

            matches = []
            for j in idx:
                score = float(row[j])
                if score < threshold:
                    continue
                matches.append(
                    {
                        "target_id": target_ids[j],
                        "score": score,
                    }
                )
                if score > best_score:
                    best_score = score

            if matches:
                results.append(
                    {
                        "source_id": source_ids[src_idx],
                        "source_preview": source_previews[src_idx],
                        "matches": matches,
                    }
                )
                total_matches += len(matches)

    output_path = output
    if output_path is None:
        analysis_dir = Path.home() / ".context" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            analysis_dir / f"semantic_xref_{source_config.name}_to_{target_config.name}.json"
        )

    payload = {
        "source": {
            "project": source_config.name,
            "provider": resolved_source_provider,
            "model": resolved_source_model,
            "embedding_dir": str(source_dir),
        },
        "target": {
            "project": target_config.name,
            "provider": resolved_target_provider,
            "model": resolved_target_model,
            "embedding_dir": str(target_dir),
        },
        "threshold": threshold,
        "top_k": top_k,
        "created": datetime.now().isoformat(),
        "stats": {
            "source_count": len(source_ids),
            "target_count": len(target_ids),
            "sources_with_matches": len(results),
            "total_matches": total_matches,
            "best_score": round(best_score, 6),
        },
        "results": results,
    }

    output_path = output_path.expanduser()
    output_path.write_text(json.dumps(payload, indent=2))
    console.print(f"[green]Wrote {output_path}[/green]")


@embed_app.command("stores")
def embed_stores(
    project: Optional[str] = typer.Argument(None, help="Project name (defaults to all)"),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Sync projects from config before listing",
    ),
) -> None:
    """List available embedding stores per project."""
    import json

    from hafs.core.embeddings import BatchEmbeddingManager
    from hafs.services.embedding_service import EmbeddingService

    service = EmbeddingService()
    if sync:
        service.sync_projects_from_registry()

    if project:
        config = service.resolve_project(project)
        if not config:
            console.print(f"[red]Unknown project: {project}[/red]")
            raise typer.Exit(1)
        projects = [config]
    else:
        projects = [p for p in service.get_projects() if p.enabled]

    if not projects:
        console.print("[yellow]No projects configured[/yellow]")
        return

    for config in projects:
        root = service.get_embedding_root(config.name)
        console.print(f"[bold]{config.name}[/bold] ({config.project_type.value})")
        console.print(f"  root: {root}")
        if config.embedding_provider or config.embedding_model:
            console.print(
                f"  configured: provider={config.embedding_provider or '-'} model={config.embedding_model or '-'}"
            )

        if not root or not root.exists():
            console.print("  [dim]No embedding root found[/dim]")
            continue

        stores = []
        default_index = root / "embedding_index.json"
        if default_index.exists():
            stores.append((None, default_index))

        for index_file in sorted(root.glob("embedding_index_*.json")):
            storage_id = index_file.stem.replace("embedding_index_", "", 1)
            stores.append((storage_id, index_file))

        if not stores:
            console.print("  [dim]No embedding stores found[/dim]")
            continue

        for storage_id, index_file in stores:
            store_dir = BatchEmbeddingManager.resolve_embeddings_dir(root, storage_id)
            count = 0
            try:
                data = json.loads(index_file.read_text())
                count = len(data) if isinstance(data, dict) else 0
            except Exception:
                pass

            provider = None
            model = None
            if store_dir and store_dir.exists():
                for emb_file in store_dir.glob("*.json"):
                    try:
                        emb = json.loads(emb_file.read_text())
                        provider = emb.get("embedding_provider")
                        model = emb.get("embedding_model")
                        break
                    except Exception:
                        continue

            label = storage_id or "default"
            console.print(
                f"  {label}: {count} embeddings | provider={provider or '-'} model={model or '-'}"
            )


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
            sys.executable,
            "-m",
            "hafs.services.embedding_daemon",
            "--batch-size",
            str(batch_size),
            "--interval",
            str(interval),
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
        console.print(
            "[dim]Daemon not running (stale PID file)[/dim]"
        )  # Keep this as specific error logic or add a render method?
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
        missing = [s for s in kb._symbols.values() if s.id not in existing and s.description][
            :count
        ]

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
            result = await builder.generate_embeddings(
                symbol_items, kb_name="alttp_symbols_enriched"
            )
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
            result = await builder.generate_embeddings(
                routine_items, kb_name="alttp_routines_enriched"
            )
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
memory_app = typer.Typer(
    name="memory",
    help="Manage agent memory, recall, and cross-search",
)
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
        if stats.get("by_type"):
            console.print("  By type:")
            for mtype, count in stats["by_type"].items():
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
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    agent: str = typer.Option(..., help="Agent ID to search"),
    limit: int = typer.Option(10, help="Max results"),
    bucket: str = typer.Option("all", help="Temporal bucket: working, recent, archive, all"),
    recency: float = typer.Option(0.3, help="Recency weight (0-1)"),
) -> None:
    """Search an agent's memory with temporal awareness."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

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
    ctx: typer.Context,
    content: Optional[str] = typer.Argument(None, help="Content to remember"),
    agent: str = typer.Option(..., help="Agent ID"),
    memory_type: str = typer.Option(
        "insight", help="Type: decision, interaction, learning, error, insight"
    ),
    importance: float = typer.Option(0.5, help="Importance (0-1)"),
) -> None:
    """Store a memory for an agent."""
    if content is None:
        console.print(ctx.get_help())
        raise typer.Exit()

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
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
) -> None:
    """Search across all agents' memories."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

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
def nodes_show(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Node name"),
) -> None:
    """Show detailed node configuration."""
    if name is None:
        console.print(ctx.get_help())
        raise typer.Exit()

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
def sync_show(
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
def sync_run(
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


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Launch TUI by default when no command is specified."""
    if ctx.invoked_subcommand is None:
        from hafs.ui.app import run

        run()


# --- AFS Management Commands ---


@afs_app.command("init")
def afs_init(
    path: Path = typer.Argument(Path("."), help="Path to initialize AFS in"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force initialization even if .context exists"
    ),
) -> None:
    """Initialize AFS (.context) in the target directory."""
    from hafs.core.afs.manager import AFSManager

    config = load_config()
    manager = AFSManager(config)
    try:
        root = manager.init(path=path, force=force)
        ui_afs.render_init_result(console, root.path)
    except Exception as e:
        ui_afs.render_error(console, f"Failed to initialize AFS: {e}")
        raise typer.Exit(1)


@afs_app.command("mount")
def afs_mount(
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

    from hafs.core.afs.manager import AFSManager
    from hafs.core.afs.discovery import find_context_root
    from hafs.models.afs import MountType

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
def afs_unmount(
    ctx: typer.Context,
    mount_type: Optional[str] = typer.Argument(None, help="Mount type"),
    alias: Optional[str] = typer.Argument(None, help="Alias or name of the mount to remove"),
) -> None:
    """Remove a mount point from the nearest AFS context."""
    if mount_type is None or alias is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.core.afs.manager import AFSManager
    from hafs.core.afs.discovery import find_context_root
    from hafs.models.afs import MountType

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
def afs_list() -> None:
    """List current AFS structure and mounts."""
    from hafs.core.afs.manager import AFSManager
    from hafs.core.afs.discovery import find_context_root

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
def afs_clean(
    force: bool = typer.Option(False, "--force", "-f", help="Force cleaning without confirmation"),
) -> None:
    """Remove the AFS context directory (clean)."""
    from hafs.core.afs.manager import AFSManager
    from hafs.core.afs.discovery import find_context_root

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


# --- Context Engineering Commands ---


@context_app.command("status")
def context_status() -> None:
    """Show context engineering pipeline status."""
    from hafs.context import ContextStore, TokenBudgetManager

    store = ContextStore()
    store.load()

    manager = TokenBudgetManager()

    items = store.get_all()
    by_type: dict[str, int] = {}
    for item in items:
        by_type[item.memory_type.value] = by_type.get(item.memory_type.value, 0) + 1

    status = {
        "store": {
            "loaded": True,
            "item_count": len(items),
        },
        "budget": {
            "model": manager.config.model_config.name,
            "available": manager.available_tokens,
        },
        "by_type": by_type,
    }

    ui_context.render_context_status(console, status)


@context_app.command("list")
def context_list(
    memory_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by memory type"
    ),
    limit: int = typer.Option(50, "--limit", "-l", help="Max items to show"),
) -> None:
    """List context items."""
    from hafs.context import ContextStore
    from hafs.models.context import MemoryType

    store = ContextStore()
    store.load()

    if memory_type:
        try:
            mt = MemoryType(memory_type.lower())
            items = store.get_by_type(mt)
        except ValueError:
            ui_context.render_error(
                console,
                f"Invalid memory type: {memory_type}. Valid: {[t.value for t in MemoryType]}",
            )
            raise typer.Exit(1)
    else:
        items = store.get_all()

    if not items:
        ui_context.render_no_items(console)
        return

    items_data = [item.to_dict() for item in items[:limit]]
    ui_context.render_context_items(console, items_data, memory_type)


@context_app.command("read")
def context_read(
    ctx: typer.Context,
    item_id: Optional[str] = typer.Argument(None, help="Context item ID"),
) -> None:
    """Read a specific context item."""
    if item_id is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.context import ContextStore

    store = ContextStore()
    store.load()

    item = store.get(item_id)
    if not item:
        ui_context.render_item_not_found(console, item_id)
        raise typer.Exit(1)

    ui_context.render_context_item_detail(console, item.to_dict())


@context_app.command("write")
def context_write(
    ctx: typer.Context,
    content: Optional[str] = typer.Argument(None, help="Content to store"),
    memory_type: str = typer.Option(
        "fact", "--type", "-t", help="Memory type (scratchpad, episodic, fact, etc.)"
    ),
    priority: str = typer.Option(
        "medium", "--priority", "-p", help="Priority (critical, high, medium, low, background)"
    ),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s", help="Source file path"
    ),
) -> None:
    """Write a new context item."""
    if content is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.context import ContextStore
    from hafs.models.context import ContextItem, ContextPriority, MemoryType

    try:
        mt = MemoryType(memory_type.lower())
    except ValueError:
        ui_context.render_error(
            console,
            f"Invalid memory type: {memory_type}. Valid: {[t.value for t in MemoryType]}",
        )
        raise typer.Exit(1)

    try:
        pri = ContextPriority(priority.lower())
    except ValueError:
        ui_context.render_error(
            console,
            f"Invalid priority: {priority}. Valid: {[p.value for p in ContextPriority]}",
        )
        raise typer.Exit(1)

    store = ContextStore()
    store.load()

    item = ContextItem(
        content=content,
        memory_type=mt,
        priority=pri,
        source_path=source,
    )

    store.save(item)
    ui_context.render_context_write_result(console, str(item.id), mt.value)


@context_app.command("search")
def context_search(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max results"),
) -> None:
    """Search context items by content."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.context import ContextStore

    store = ContextStore()
    store.load()

    items = store.get_all()
    query_lower = query.lower()

    # Simple keyword search (for semantic search, use embed commands)
    results = []
    for item in items:
        if query_lower in item.content.lower():
            results.append({"item": item.to_dict(), "score": item.relevance_score})

    results.sort(key=lambda x: x["score"], reverse=True)
    ui_context.render_context_search_results(console, query, results[:limit])


@context_app.command("construct")
def context_construct(
    ctx: typer.Context,
    task: Optional[str] = typer.Argument(None, help="Task description for context selection"),
    max_tokens: int = typer.Option(128000, "--max-tokens", help="Max tokens for context window"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for context"),
) -> None:
    """Construct an optimized context window for a task."""
    if task is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.context import ContextConstructor, ContextStore, SelectionCriteria, ConstructorConfig
    from hafs.models.context import TokenBudget

    store = ContextStore()
    store.load()

    items = store.get_all()
    if not items:
        ui_context.render_no_items(console)
        return

    budget = TokenBudget(total_budget=max_tokens)
    config = ConstructorConfig(token_budget=budget)
    constructor = ContextConstructor(config=config)

    criteria = SelectionCriteria(query=task)
    window = constructor.construct(items, criteria)

    # Collect stats
    by_type: dict[str, dict] = {}
    for item in window.items:
        mt = item.memory_type.value
        if mt not in by_type:
            by_type[mt] = {"count": 0, "tokens": 0}
        by_type[mt]["count"] += 1
        by_type[mt]["tokens"] += item.estimated_tokens

    window_data = {
        "total_tokens": window.total_tokens,
        "item_count": len(window.items),
        "used_percentage": window.used_percentage,
        "remaining_tokens": window.remaining_tokens,
        "by_type": by_type,
    }

    ui_context.render_context_window(console, window_data)

    if output:
        prompt = window.to_prompt()
        output.write_text(prompt)
        console.print(f"[green]Wrote context to {output}[/green]")


@context_app.command("evaluate")
def context_evaluate(
    ctx: typer.Context,
    task: Optional[str] = typer.Argument(None, help="Task to evaluate context for"),
) -> None:
    """Evaluate context quality for a task."""
    if task is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    from hafs.context import (
        ContextConstructor,
        ContextEvaluator,
        ContextStore,
        SelectionCriteria,
    )

    store = ContextStore()
    store.load()

    items = store.get_all()
    if not items:
        ui_context.render_no_items(console)
        return

    # Construct window first
    constructor = ContextConstructor()
    criteria = SelectionCriteria(query=task)
    window = constructor.construct(items, criteria)

    # Evaluate
    evaluator = ContextEvaluator(store=store)
    result = evaluator.evaluate(window, task=task)

    ui_context.render_evaluation_result(console, {
        "quality_score": result.quality_score,
        "coverage_score": result.coverage_score,
        "coherence_score": result.coherence_score,
        "freshness_score": result.freshness_score,
        "efficiency_score": result.efficiency_score,
        "issues": result.issues,
        "suggestions": result.suggestions,
    })


@context_app.command("types")
def context_types() -> None:
    """Show the memory type taxonomy (from AFS research)."""
    ui_context.render_memory_type_tree(console)


@context_app.command("prune")
def context_prune(
    expired_only: bool = typer.Option(True, "--expired-only", help="Only prune expired items"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be pruned"),
) -> None:
    """Prune old or expired context items."""
    from hafs.context import ContextStore

    store = ContextStore()
    store.load()

    if expired_only:
        items = [item for item in store.get_all() if item.is_expired()]
    else:
        # Prune items older than 30 days with low relevance
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=30)
        items = [
            item for item in store.get_all()
            if item.created_at < cutoff and item.relevance_score < 0.3
        ]

    if dry_run:
        console.print(f"[yellow]Would prune {len(items)} items:[/yellow]")
        for item in items[:10]:
            console.print(f"  - {item.id} ({item.memory_type.value})")
        if len(items) > 10:
            console.print(f"  ... and {len(items) - 10} more")
    else:
        for item in items:
            store.delete(str(item.id))
        console.print(f"[green]Pruned {len(items)} context items[/green]")


# --- Entry Point ---


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
