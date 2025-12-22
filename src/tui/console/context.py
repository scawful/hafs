"""Console rendering for context engineering CLI commands."""

from __future__ import annotations

from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree


def _human_bytes(size: float | int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def render_context_status(
    console: Console,
    status: dict[str, Any],
) -> None:
    """Render context pipeline status."""
    table = Table(title="Context Engineering Pipeline Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="dim")

    # Store status
    store_status = status.get("store", {})
    table.add_row(
        "Context Store",
        "Active" if store_status.get("loaded") else "Not loaded",
        f"{store_status.get('item_count', 0)} items",
    )

    # Budget status
    budget = status.get("budget", {})
    table.add_row(
        "Token Budget",
        f"{budget.get('model', 'Unknown')}",
        f"{budget.get('available', 0):,} tokens available",
    )

    # Memory type breakdown
    for mtype, count in status.get("by_type", {}).items():
        table.add_row(f"  {mtype}", "", f"{count} items")

    console.print(table)


def render_context_items(
    console: Console,
    items: list[dict[str, Any]],
    memory_type: Optional[str] = None,
) -> None:
    """Render a list of context items."""
    title = f"Context Items ({memory_type})" if memory_type else "All Context Items"
    table = Table(title=title)
    table.add_column("ID", style="dim", max_width=12)
    table.add_column("Type", style="cyan")
    table.add_column("Priority", style="yellow")
    table.add_column("Tokens", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Preview", max_width=50)

    for item in items[:50]:  # Limit display
        table.add_row(
            str(item.get("id", ""))[:12],
            item.get("memory_type", ""),
            item.get("priority", ""),
            str(item.get("estimated_tokens", 0)),
            f"{item.get('relevance_score', 0):.2f}",
            (item.get("content", "")[:50] + "...") if len(item.get("content", "")) > 50 else item.get("content", ""),
        )

    console.print(table)

    if len(items) > 50:
        console.print(f"[dim]... and {len(items) - 50} more items[/dim]")


def render_context_item_detail(
    console: Console,
    item: dict[str, Any],
) -> None:
    """Render detailed view of a context item."""
    panel_content = f"""
[bold]ID:[/bold] {item.get('id', 'N/A')}
[bold]Type:[/bold] {item.get('memory_type', 'N/A')}
[bold]Priority:[/bold] {item.get('priority', 'N/A')}
[bold]Tokens:[/bold] {item.get('estimated_tokens', 0)}
[bold]Relevance:[/bold] {item.get('relevance_score', 0):.3f}
[bold]Created:[/bold] {item.get('created_at', 'N/A')}
[bold]Accessed:[/bold] {item.get('accessed_at', 'N/A')}
[bold]Access Count:[/bold] {item.get('access_count', 0)}
[bold]Compressed:[/bold] {item.get('is_compressed', False)}
[bold]Source:[/bold] {item.get('source_path', 'N/A')}

[bold]Content:[/bold]
{item.get('content', '')}
"""
    console.print(Panel(panel_content, title="Context Item", border_style="cyan"))


def render_context_write_result(
    console: Console,
    item_id: str,
    memory_type: str,
) -> None:
    """Render result of writing a context item."""
    console.print(f"[green]Created context item:[/green] {item_id}")
    console.print(f"  Type: {memory_type}")


def render_context_search_results(
    console: Console,
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """Render context search results."""
    console.print(f"[bold]Search results for:[/bold] {query}\n")

    if not results:
        console.print("[yellow]No matching items found[/yellow]")
        return

    for i, result in enumerate(results, 1):
        item = result.get("item", result)
        score = result.get("score", item.get("relevance_score", 0))
        content_preview = item.get("content", "")[:200]

        console.print(f"[cyan]{i}.[/cyan] [{item.get('memory_type', 'unknown')}] "
                     f"score={score:.3f}")
        console.print(f"   [dim]{content_preview}...[/dim]\n")


def render_context_window(
    console: Console,
    window: dict[str, Any],
) -> None:
    """Render constructed context window."""
    console.print(Panel(
        f"[bold]Total Tokens:[/bold] {window.get('total_tokens', 0):,}\n"
        f"[bold]Items:[/bold] {window.get('item_count', 0)}\n"
        f"[bold]Usage:[/bold] {window.get('used_percentage', 0):.1f}%\n"
        f"[bold]Remaining:[/bold] {window.get('remaining_tokens', 0):,}",
        title="Context Window",
        border_style="green",
    ))

    # Show items by type
    by_type = window.get("by_type", {})
    if by_type:
        table = Table(title="Items by Memory Type")
        table.add_column("Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Tokens", justify="right")

        for mtype, info in by_type.items():
            table.add_row(
                mtype,
                str(info.get("count", 0)),
                f"{info.get('tokens', 0):,}",
            )

        console.print(table)


def render_evaluation_result(
    console: Console,
    result: dict[str, Any],
) -> None:
    """Render context evaluation result."""
    # Quality scores
    table = Table(title="Context Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status", style="green")

    def status_icon(score: float) -> str:
        if score >= 0.7:
            return "[green]Good[/green]"
        elif score >= 0.4:
            return "[yellow]Fair[/yellow]"
        else:
            return "[red]Poor[/red]"

    table.add_row(
        "Overall Quality",
        f"{result.get('quality_score', 0):.2f}",
        status_icon(result.get('quality_score', 0)),
    )
    table.add_row(
        "Coverage",
        f"{result.get('coverage_score', 0):.2f}",
        status_icon(result.get('coverage_score', 0)),
    )
    table.add_row(
        "Coherence",
        f"{result.get('coherence_score', 0):.2f}",
        status_icon(result.get('coherence_score', 0)),
    )
    table.add_row(
        "Freshness",
        f"{result.get('freshness_score', 0):.2f}",
        status_icon(result.get('freshness_score', 0)),
    )
    table.add_row(
        "Efficiency",
        f"{result.get('efficiency_score', 0):.2f}",
        status_icon(result.get('efficiency_score', 0)),
    )

    console.print(table)

    # Issues
    issues = result.get("issues", [])
    if issues:
        console.print("\n[bold red]Issues:[/bold red]")
        for issue in issues:
            console.print(f"  [red]•[/red] {issue}")

    # Suggestions
    suggestions = result.get("suggestions", [])
    if suggestions:
        console.print("\n[bold yellow]Suggestions:[/bold yellow]")
        for suggestion in suggestions:
            console.print(f"  [yellow]•[/yellow] {suggestion}")


def render_deep_analysis_report(console: Console, report: dict[str, Any]) -> None:
    """Render deep context analysis summary."""
    snapshot = report.get("snapshot", {})
    signals = report.get("signals", {})
    ml_signals = report.get("ml_signals", {})
    kb_coverage = report.get("kb_coverage", {})
    doc_index = report.get("doc_index", {})
    node_health = report.get("node_health", {})

    summary = Table(title="Deep Context Analysis")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Report", report.get("report_path", "-") or "-")
    summary.add_row("Repo Root", snapshot.get("repo_root", "-") or "-")
    summary.add_row("Total Files", str(snapshot.get("total_files", 0)))
    summary.add_row("Repo Size", _human_bytes(snapshot.get("total_bytes")))
    summary.add_row("TODO Hits", str(signals.get("todo_count", 0)))
    summary.add_row("KB Bases", str(kb_coverage.get("summary", {}).get("bases", 0)))
    summary.add_row("KB Items", str(kb_coverage.get("summary", {}).get("total_items", 0)))
    summary.add_row("Docs Indexed", str(doc_index.get("total_docs", 0)))

    embedding = ml_signals.get("embedding_daemon", {}) if ml_signals else {}
    backlog = "-"
    if embedding.get("total_symbols") is not None and embedding.get("total_embeddings") is not None:
        backlog = str(max(0, int(embedding["total_symbols"]) - int(embedding["total_embeddings"])))
    summary.add_row("Embedding Backlog", backlog)
    summary.add_row("Nodes Offline", str(len(node_health.get("offline", []))))
    console.print(summary)

    recommendations = report.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommended Actions:[/bold]")
        for item in recommendations:
            console.print(f"  • {item}")


def render_ml_plan(console: Console, report: dict[str, Any]) -> None:
    """Render smart ML pipeline plan summary."""
    kb_summary = report.get("kb_coverage", {}).get("summary", {})
    summary = Table(title="ML Pipeline Plan")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="green")
    summary.add_row("Report", report.get("report_path", "-") or "-")
    summary.add_row("Topic", report.get("topic", "-") or "-")
    summary.add_row("KB Bases", str(kb_summary.get("bases", 0)))
    summary.add_row("Recommendations", str(len(report.get("recommendations", []))))
    console.print(summary)

    recommendations = report.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Plan Actions:[/bold]")
        for item in recommendations:
            console.print(f"  • {item}")


def render_memory_type_tree(console: Console) -> None:
    """Render the memory type taxonomy as a tree."""
    tree = Tree("[bold]AFS Memory Type Taxonomy[/bold]")

    scratchpad = tree.add("[cyan]scratchpad[/cyan] - Temporary, task-bounded")
    scratchpad.add("[dim]Dialogue turns, temporary reasoning states[/dim]")

    episodic = tree.add("[cyan]episodic[/cyan] - Medium-term, session-bounded")
    episodic.add("[dim]Session summaries, case histories[/dim]")

    fact = tree.add("[cyan]fact[/cyan] - Long-term, fine-grained")
    fact.add("[dim]Atomic factual statements[/dim]")

    experiential = tree.add("[cyan]experiential[/cyan] - Long-term, cross-task")
    experiential.add("[dim]Observation-action trajectories[/dim]")

    procedural = tree.add("[cyan]procedural[/cyan] - Long-term, system-wide")
    procedural.add("[dim]Functions, tools, procedures[/dim]")

    user = tree.add("[cyan]user[/cyan] - Long-term, personalized")
    user.add("[dim]User attributes, preferences, histories[/dim]")

    historical = tree.add("[cyan]historical[/cyan] - Immutable, full-trace")
    historical.add("[dim]Raw logs of all interactions[/dim]")

    console.print(tree)


def render_no_items(console: Console) -> None:
    """Render message when no items found."""
    console.print("[yellow]No context items found[/yellow]")


def render_item_not_found(console: Console, item_id: str) -> None:
    """Render message when item not found."""
    console.print(f"[red]Context item not found: {item_id}[/red]")


def render_error(console: Console, message: str) -> None:
    """Render an error message."""
    console.print(f"[red]Error: {message}[/red]")
