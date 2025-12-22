"""Context engineering CLI commands."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import typer
from rich.console import Console

from config.loader import load_config
from models.context import ContextItem, ContextPriority, MemoryType, TokenBudget
from tui.console import context as ui_context

context_app = typer.Typer(
    name="context",
    help=(
        "Context Engineering Pipeline - manage context items, construct windows, "
        "and evaluate quality"
    ),
)
console = Console()

if TYPE_CHECKING:
    from context import ContextStore


def _context_deps():
    from context import (
        BudgetManagerConfig,
        ConstructorConfig,
        ContextConstructor,
        ContextEvaluator,
        ContextStore,
        MODEL_CONFIGS,
        SelectionCriteria,
        TokenBudgetManager,
        create_budget_for_model,
    )

    return {
        "BudgetManagerConfig": BudgetManagerConfig,
        "ConstructorConfig": ConstructorConfig,
        "ContextConstructor": ContextConstructor,
        "ContextEvaluator": ContextEvaluator,
        "ContextStore": ContextStore,
        "MODEL_CONFIGS": MODEL_CONFIGS,
        "SelectionCriteria": SelectionCriteria,
        "TokenBudgetManager": TokenBudgetManager,
        "create_budget_for_model": create_budget_for_model,
    }


@context_app.callback(invoke_without_command=True)
def context_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


def _load_store() -> ContextStore:
    config = load_config()
    deps = _context_deps()
    return deps["ContextStore"](config.general.context_root)


def _parse_memory_type(value: str) -> MemoryType:
    try:
        return MemoryType(value.lower())
    except ValueError as exc:
        valid = ", ".join(mt.value for mt in MemoryType)
        raise typer.BadParameter(f"invalid memory type '{value}'. Valid: {valid}") from exc


def _parse_memory_types(values: Optional[list[str]]) -> set[MemoryType]:
    if not values:
        return set(MemoryType)
    return {_parse_memory_type(value) for value in values}


def _parse_priority(value: str) -> ContextPriority:
    try:
        return ContextPriority(value.lower())
    except ValueError as exc:
        valid = ", ".join(p.value for p in ContextPriority)
        raise typer.BadParameter(f"invalid priority '{value}'. Valid: {valid}") from exc


def _resolve_model_config(model: Optional[str]):
    deps = _context_deps()
    if not model:
        return deps["BudgetManagerConfig"]().model_config
    model_key = model.strip()
    model_configs = deps["MODEL_CONFIGS"]
    if model_key not in model_configs:
        valid = ", ".join(sorted(model_configs.keys()))
        raise typer.BadParameter(f"unknown model '{model_key}'. Valid: {valid}")
    return model_configs[model_key]


def _infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    source_type_map = {
        ".md": "markdown",
        ".json": "json",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".txt": "text",
    }
    return source_type_map.get(suffix, "text")


def _items_to_dicts(items: list[ContextItem]) -> list[dict[str, object]]:
    return [item.to_dict() for item in items]


@context_app.command("status")
def status(model: Optional[str] = typer.Option(None, help="Model ID for budget reporting")) -> None:
    """Show context pipeline status."""
    store = _load_store()
    items = store.get_all()

    type_counts: dict[str, int] = {mt.value: 0 for mt in MemoryType}
    for item in items:
        type_counts[item.memory_type.value] += 1

    deps = _context_deps()
    model_config = _resolve_model_config(model)
    manager = deps["TokenBudgetManager"](deps["BudgetManagerConfig"](model_config=model_config))

    status_data = {
        "store": {"loaded": True, "item_count": len(items)},
        "budget": {
            "model": model_config.name,
            "available": manager.available_tokens,
        },
        "by_type": type_counts,
    }

    ui_context.render_context_status(console, status_data)


@context_app.command("list")
def list_items(
    memory_types: Optional[list[str]] = typer.Option(
        None, "--type", "-t", help="Filter by memory type (repeatable)"
    ),
    limit: int = typer.Option(50, help="Max items to show"),
) -> None:
    """List context items."""
    store = _load_store()
    selected_types = _parse_memory_types(memory_types)

    items = [item for item in store.get_all() if item.memory_type in selected_types]
    if not items:
        ui_context.render_no_items(console)
        return

    items.sort(key=lambda item: item.accessed_at, reverse=True)
    ui_context.render_context_items(
        console,
        _items_to_dicts(items[:limit]),
        memory_type=", ".join(mt.value for mt in selected_types) if memory_types else None,
    )


@context_app.command("read")
def read_item(
    ctx: typer.Context,
    item_id: Optional[str] = typer.Argument(None, help="Context item ID"),
) -> None:
    """Show a single context item."""
    if item_id is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    store = _load_store()
    item = store.get(item_id)

    if not item:
        ui_context.render_item_not_found(console, item_id)
        raise typer.Exit(1)

    ui_context.render_context_item_detail(console, item.to_dict())


@context_app.command("write")
def write_item(
    ctx: typer.Context,
    content: Optional[str] = typer.Argument(None, help="Content to store"),
    memory_type: str = typer.Option("fact", "--type", "-t", help="Memory type"),
    priority: str = typer.Option("medium", "--priority", "-p", help="Priority"),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s", help="Optional source file path"
    ),
    expires_hours: Optional[float] = typer.Option(
        None, "--expires-hours", help="Expire after N hours"
    ),
) -> None:
    """Store a new context item."""
    if content is None and source is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    if content is None and source is not None:
        try:
            content = source.read_text(encoding="utf-8")
        except OSError as exc:
            ui_context.render_error(console, f"Failed to read source: {exc}")
            raise typer.Exit(1)

    store = _load_store()
    item = ContextItem(
        content=content or "",
        memory_type=_parse_memory_type(memory_type),
        priority=_parse_priority(priority),
        source_path=source,
        source_type=_infer_source_type(source) if source else "text",
    )

    if expires_hours is not None:
        item.expires_at = datetime.now() + timedelta(hours=expires_hours)

    store.save(item)
    ui_context.render_context_write_result(console, str(item.id), item.memory_type.value)


@context_app.command("search")
def search_items(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="Search query"),
    memory_types: Optional[list[str]] = typer.Option(
        None, "--type", "-t", help="Filter by memory type (repeatable)"
    ),
    limit: int = typer.Option(20, help="Max results"),
) -> None:
    """Search context items by content."""
    if query is None:
        console.print(ctx.get_help())
        raise typer.Exit()

    store = _load_store()
    selected_types = _parse_memory_types(memory_types)
    query_lower = query.lower()

    results = []
    for item in store.get_all():
        if item.memory_type not in selected_types:
            continue
        if query_lower in item.content.lower():
            results.append(
                {
                    "item": item.to_dict(),
                    "score": item.relevance_score,
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    ui_context.render_context_search_results(console, query, results[:limit])


@context_app.command("construct")
def construct_context(
    task: Optional[str] = typer.Argument(None, help="Task description"),
    memory_types: Optional[list[str]] = typer.Option(
        None, "--type", "-t", help="Include memory type (repeatable)"
    ),
    min_relevance: float = typer.Option(0.0, help="Minimum relevance score"),
    max_age_hours: Optional[float] = typer.Option(None, help="Max item age in hours"),
    recent_hours: Optional[float] = typer.Option(
        None, "--recent-hours", help="Only include items accessed in last N hours"
    ),
    include_expired: bool = typer.Option(False, help="Include expired items"),
    model: Optional[str] = typer.Option(None, help="Model ID for token budget"),
    show_items: bool = typer.Option(False, help="Show item details"),
) -> None:
    """Construct an optimized context window."""
    store = _load_store()
    items = store.get_all()

    include_types = _parse_memory_types(memory_types)
    deps = _context_deps()
    criteria = deps["SelectionCriteria"](
        query=task or "",
        include_types=include_types,
        min_relevance=min_relevance,
        max_age_hours=max_age_hours,
        recently_accessed_hours=recent_hours,
        include_expired=include_expired,
    )

    if model:
        _resolve_model_config(model)
        token_budget = deps["create_budget_for_model"](model)
    else:
        token_budget = TokenBudget()

    constructor = deps["ContextConstructor"](deps["ConstructorConfig"](token_budget=token_budget))
    window = constructor.construct(items, criteria)

    by_type: dict[str, dict[str, int]] = {}
    for item in window.items:
        entry = by_type.setdefault(item.memory_type.value, {"count": 0, "tokens": 0})
        entry["count"] += 1
        entry["tokens"] += item.estimated_tokens

    window_data = {
        "total_tokens": window.total_tokens,
        "item_count": len(window.items),
        "used_percentage": window.used_percentage,
        "remaining_tokens": window.remaining_tokens,
        "by_type": by_type,
    }
    ui_context.render_context_window(console, window_data)

    if show_items:
        ui_context.render_context_items(console, _items_to_dicts(window.items))


@context_app.command("evaluate")
def evaluate_context(
    task: Optional[str] = typer.Argument(None, help="Task description"),
    memory_types: Optional[list[str]] = typer.Option(
        None, "--type", "-t", help="Include memory type (repeatable)"
    ),
    required_types: Optional[list[str]] = typer.Option(
        None, "--require", help="Required memory type (repeatable)"
    ),
    model: Optional[str] = typer.Option(None, help="Model ID for token budget"),
) -> None:
    """Evaluate context quality for a task."""
    store = _load_store()
    items = store.get_all()

    include_types = _parse_memory_types(memory_types)
    deps = _context_deps()
    criteria = deps["SelectionCriteria"](query=task or "", include_types=include_types)

    if model:
        _resolve_model_config(model)
        token_budget = deps["create_budget_for_model"](model)
    else:
        token_budget = TokenBudget()

    constructor = deps["ContextConstructor"](deps["ConstructorConfig"](token_budget=token_budget))
    window = constructor.construct(items, criteria)

    evaluator = deps["ContextEvaluator"](store=store)
    required = _parse_memory_types(required_types) if required_types else None
    result = evaluator.evaluate(window, task=task, required_types=required)

    ui_context.render_evaluation_result(console, asdict(result))


@context_app.command("deep-dive")
def deep_dive(
    topic: Optional[str] = typer.Option(None, "--topic", help="Report topic"),
    root: Path = typer.Option(Path.cwd(), "--root", "-r", help="Repository root"),
    context_root: Optional[Path] = typer.Option(None, "--context-root", help="Context root override"),
    reports_root: Optional[Path] = typer.Option(None, "--reports-root", help="Reports root override"),
    check_nodes: bool = typer.Option(True, "--check-nodes/--no-check-nodes", help="Run node health checks"),
    llm_summary: bool = typer.Option(True, "--llm/--no-llm", help="Include LLM deep analysis summary"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Run a deep context analysis report."""
    from agents.analysis.deep_context_pipeline import DeepContextPipeline

    pipeline = DeepContextPipeline(
        repo_root=root,
        context_root=context_root,
        reports_root=reports_root,
        check_nodes=check_nodes,
        llm_summary=llm_summary,
    )
    result = asyncio.run(pipeline.generate_report(topic or "Deep Context Analysis"))

    if json_output:
        console.print_json(data=result)
    else:
        ui_context.render_deep_analysis_report(console, result)


@context_app.command("ml-plan")
def ml_plan(
    topic: Optional[str] = typer.Option(None, "--topic", help="Report topic"),
    context_root: Optional[Path] = typer.Option(None, "--context-root", help="Context root override"),
    reports_root: Optional[Path] = typer.Option(None, "--reports-root", help="Reports root override"),
    llm_summary: bool = typer.Option(True, "--llm/--no-llm", help="Include LLM plan summary"),
    json_output: bool = typer.Option(False, "--json", help="Print JSON output"),
) -> None:
    """Generate a smart ML pipeline plan."""
    from agents.analysis.deep_context_pipeline import SmartMLPipeline

    pipeline = SmartMLPipeline(
        context_root=context_root,
        reports_root=reports_root,
        llm_summary=llm_summary,
    )
    result = asyncio.run(pipeline.generate_report(topic or "ML Pipeline Plan"))

    if json_output:
        console.print_json(data=result)
    else:
        ui_context.render_ml_plan(console, result)


@context_app.command("types")
def list_types() -> None:
    """Show memory type taxonomy."""
    ui_context.render_memory_type_tree(console)


@context_app.command("prune")
def prune_context(
    dry_run: bool = typer.Option(False, "--dry-run", help="Report without deleting"),
) -> None:
    """Prune expired context items."""
    store = _load_store()
    items = store.get_all()

    expired = [item for item in items if item.is_expired()]
    if dry_run:
        console.print(f"[yellow]Expired items:[/yellow] {len(expired)}")
        return

    removed = store.prune_expired()
    console.print(f"[green]Removed expired items:[/green] {removed}")
