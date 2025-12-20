"""Console UI for history commands."""
from rich.console import Console
from typing import Any

def render_history_status(console: Console, status: dict[str, Any], summaries: dict[str, Any]) -> None:
    """Render history status."""
    console.print("[bold]History Index Status[/bold]")
    console.print(f"- History files: {status['history_files']}")
    console.print(f"- Embeddings: {status['embeddings']}")
    console.print(f"- Sessions: {summaries['sessions']}")
    console.print(f"- Summaries: {summaries['summaries']}")

def render_index_result(console: Console, count: int) -> None:
    """Render indexing result."""
    console.print(f"[green]Indexed {count} new entries[/green]")

def render_session_summary_result(console: Console, session_id: str, success: bool) -> None:
    """Render session summary result."""
    if success:
        console.print(f"[green]Summarized session {session_id}[/green]")
    else:
        console.print(f"[yellow]No entries for session {session_id}[/yellow]")

def render_summaries_created(console: Console, count: int) -> None:
    """Render summaries created count."""
    console.print(f"[green]Created {count} summaries[/green]")

def render_search_results(console: Console, query: str, results: list[dict[str, Any]], mode: str) -> None:
    """Render search results."""
    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    console.print(f"[bold]Results for[/bold] '{query}':")
    for idx, result in enumerate(results, start=1):
        score = result.get("score", 0.0)
        kind = result.get("kind", mode)
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
