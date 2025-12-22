import typer
import asyncio
import json
from pathlib import Path
from typing import Optional
from rich.console import Console

from config.loader import load_config
from tui.console import embed as ui_embed

embed_app = typer.Typer(
    name="embed",
    help="Manage embedding generation, stores, and semantic xref (multi-model)",
)
console = Console()


@embed_app.callback(invoke_without_command=True)
def embed_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@embed_app.command("status")
def status() -> None:
    """Show embedding daemon status."""
    try:
        from services.daemons.embedding_daemon import get_status
    except ModuleNotFoundError:
        try:
            from services.embedding_daemon import get_status
        except ModuleNotFoundError as exc:
            console.print("[red]Embedding daemon module not available.[/red]")
            raise typer.Exit(1) from exc

    st = get_status()
    ui_embed.render_daemon_status(console, st)


@embed_app.command("index")
def index(
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
    post_completion: bool = typer.Option(
        True,
        "--post-completion/--no-post-completion",
        help="Run post-completion swarm/context after each project",
    ),
    post_completion_force: bool = typer.Option(
        True,
        "--post-completion-force/--post-completion-cooldown",
        help="Force post-completion even if cooldown has not elapsed",
    ),
) -> None:
    """Run embedding indexer for configured projects."""

    async def _index() -> None:
        from services.embedding_service import EmbeddingService

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

        for name in names:
            await service.run_indexing(
                [name],
                embedding_provider=provider,
                embedding_model=model,
            )
            if post_completion:
                try:
                    from services.daemons.embedding_daemon import EmbeddingDaemon
                except ModuleNotFoundError:
                    try:
                        from services.embedding_daemon import EmbeddingDaemon
                    except ModuleNotFoundError as exc:
                        console.print("[red]Embedding daemon module not available.[/red]")
                        raise typer.Exit(1) from exc

                daemon = EmbeddingDaemon()
                await daemon.notify_embeddings_complete(force=post_completion_force)
        ui_embed.render_indexing_complete(console)

    asyncio.run(_index())


@embed_app.command("xref")
def xref(
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
        from core.runtime import resolve_python_executable

        console.print("[red]Missing dependency: numpy[/red]")
        python_path = resolve_python_executable()
        console.print(f"Install: {python_path} -m pip install numpy")
        raise typer.Exit(1)

    from datetime import datetime
    from services.embedding_service import EmbeddingService

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
def stores(
    project: Optional[str] = typer.Argument(None, help="Project name (defaults to all)"),
    sync: bool = typer.Option(
        True,
        "--sync/--no-sync",
        help="Sync projects from config before listing",
    ),
) -> None:
    """List available embedding stores per project."""
    from core.embeddings import BatchEmbeddingManager
    from services.embedding_service import EmbeddingService

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

        stores_list = []
        default_index = root / "embedding_index.json"
        if default_index.exists():
            stores_list.append((None, default_index))

        for index_file in sorted(root.glob("embedding_index_*.json")):
            storage_id = index_file.stem.replace("embedding_index_", "", 1)
            stores_list.append((storage_id, index_file))

        if not stores_list:
            console.print("  [dim]No embedding stores found[/dim]")
            continue

        for storage_id, index_file in stores_list:
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
def start(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Embeddings per batch"),
    interval: int = typer.Option(60, "--interval", "-i", help="Seconds between batches"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
) -> None:
    """Start the embedding daemon."""
    import subprocess
    import sys

    if foreground:
        # Run in foreground
        from services.embedding_daemon import EmbeddingDaemon

        daemon = EmbeddingDaemon(batch_size=batch_size, interval_seconds=interval)
        asyncio.run(daemon.start())
    else:
        # Run in background
        cmd = [
            sys.executable,
            "-m",
            "services.embedding_daemon",
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
def stop() -> None:
    """Stop the embedding daemon."""
    import os
    import signal

    pid_file = Path.home() / ".context" / "embedding_service" / "daemon.pid"

    if not pid_file.exists():
        ui_embed.render_daemon_not_running(console)
        return

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        ui_embed.render_daemon_stopped(console, pid)
    except ProcessLookupError:
        console.print("[dim]Daemon not running (stale PID file)[/dim]")
        pid_file.unlink()
    except Exception as e:
        console.print(f"[red]Error stopping daemon: {e}[/red]")


@embed_app.command("install")
def install() -> None:
    """Install embedding daemon as launchd service (macOS)."""
    from services.embedding_daemon import install_launchd

    install_launchd()


@embed_app.command("uninstall")
def uninstall() -> None:
    """Uninstall embedding daemon launchd service."""
    from services.embedding_daemon import uninstall_launchd

    uninstall_launchd()


@embed_app.command("quick")
def quick(
    count: int = typer.Argument(100, help="Number of embeddings to generate"),
) -> None:
    """Generate embeddings inline (not as daemon)."""

    async def _quick() -> None:
        # NOTE: This imports specific agents that might not be in hafs/core?
        # Assuming agents is available.
        from agents.knowledge.alttp_knowledge import ALTTPKnowledgeBase
        from core.orchestrator_v2 import UnifiedOrchestrator

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
            except Exception:
                pass

        stats = kb.get_statistics()
        ui_embed.render_quick_stats(console, stats, "After")

    asyncio.run(_quick())


@embed_app.command("enhance")
def enhance(
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
        from agents.knowledge.alttp_embeddings import ALTTPEmbeddingBuilder
        from agents.knowledge.alttp_knowledge import ALTTPKnowledgeBase

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
