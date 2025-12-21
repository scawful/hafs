import typer
import asyncio
from typing import Optional
from rich.console import Console

from hafs.config.loader import load_config
from hafs.ui.console import history as ui_history

history_app = typer.Typer(
    name="history",
    help="Manage AFS history embeddings, summaries, and search",
)
console = Console()


@history_app.callback(invoke_without_command=True)
def history_callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@history_app.command("status")
def status(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Embedding provider (gemini/openai/ollama/halext)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Embedding model override"
    ),
) -> None:
    """Show history embedding index status."""
    from hafs.core.history import HistoryEmbeddingIndex, HistorySessionSummaryIndex

    config = load_config()
    index = HistoryEmbeddingIndex(
        config.general.context_root,
        embedding_provider=provider,
        embedding_model=model,
    )
    status_data = index.status()
    summaries = HistorySessionSummaryIndex(
        config.general.context_root,
        embedding_provider=provider,
        embedding_model=model,
    ).status()
    ui_history.render_history_status(console, status_data, summaries)


@history_app.command("index")
def index(
    limit: int = typer.Option(200, help="Max new entries to embed"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Embedding provider (gemini/openai/ollama/halext)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Embedding model override"
    ),
) -> None:
    """Index new history entries into embeddings."""
    from hafs.core.history import HistoryEmbeddingIndex

    async def _index() -> None:
        config = load_config()
        idx = HistoryEmbeddingIndex(
            config.general.context_root,
            embedding_provider=provider,
            embedding_model=model,
        )
        created = await idx.index_new_entries(limit=limit)
        ui_history.render_index_result(console, created)

    asyncio.run(_index())


@history_app.command("summarize")
def summarize(
    session_id: Optional[str] = typer.Option(None, help="Summarize a specific session"),
    limit: int = typer.Option(20, help="Max sessions to summarize"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Embedding provider (gemini/openai/ollama/halext)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Embedding model override"
    ),
) -> None:
    """Generate session summaries and embeddings."""
    from hafs.core.history import HistorySessionSummaryIndex

    async def _summarize() -> None:
        config = load_config()
        idx = HistorySessionSummaryIndex(
            config.general.context_root,
            embedding_provider=provider,
            embedding_model=model,
        )
        if session_id:
            summary = await idx.summarize_session(session_id)
            ui_history.render_session_summary_result(console, session_id, bool(summary))
            return

        created = await idx.index_missing_summaries(limit=limit)
        ui_history.render_summaries_created(console, created)

    asyncio.run(_summarize())


@history_app.command("search")
def search(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    limit: int = typer.Option(10, help="Max results"),
    refresh: bool = typer.Option(False, help="Index new entries before searching"),
    sessions: bool = typer.Option(False, help="Search session summaries instead of entries"),
    all_results: bool = typer.Option(False, "--all", help="Search entries and sessions"),
    mode: Optional[str] = typer.Option(None, help="entries|sessions|all"),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Embedding provider (gemini/openai/ollama/halext)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Embedding model override"
    ),
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
            idx = HistorySessionSummaryIndex(
                config.general.context_root,
                embedding_provider=provider,
                embedding_model=model,
            )
            if refresh:
                await idx.index_missing_summaries(limit=50)
            results = await idx.search(query, limit=limit)
        elif selected_mode == "entries":
            idx = HistoryEmbeddingIndex(
                config.general.context_root,
                embedding_provider=provider,
                embedding_model=model,
            )
            if refresh:
                await idx.index_new_entries(limit=200)
            results = await idx.search(query, limit=limit)
        else:
            entry_index = HistoryEmbeddingIndex(
                config.general.context_root,
                embedding_provider=provider,
                embedding_model=model,
            )
            summary_index = HistorySessionSummaryIndex(
                config.general.context_root,
                embedding_provider=provider,
                embedding_model=model,
            )
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
