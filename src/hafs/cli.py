"""HAFS command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from hafs.config.loader import load_config
from hafs.core.afs.discovery import discover_projects, find_context_root
from hafs.core.afs.manager import AFSManager
from hafs.models.afs import MountType

app = typer.Typer(
    name="hafs",
    help="HAFS - Halext Agentic File System",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    """Launch TUI by default when no command is specified."""
    # Only run TUI if no subcommand was invoked
    if ctx.invoked_subcommand is None:
        from hafs.ui.app import run

        run()


def _resolve_context_path(context_root: Path, subpath: str) -> Path:
    """Resolve a user-supplied subpath and ensure it stays within .context."""
    resolved_root = context_root.resolve()
    target = (resolved_root / subpath).resolve()

    try:
        target.relative_to(resolved_root)
    except ValueError:
        console.print(f"[red]Refusing to access path outside context:[/red] {target}")
        raise typer.Exit(1)

    return target


@app.command()
def new(
    path: Path = typer.Argument(..., help="Path to create new project"),
    mount: Optional[list[str]] = typer.Option(
        None, "--mount", "-m", help="Mount source: <path>:<type>[:<alias>]"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
) -> None:
    """Create a new AFS project wrapper."""
    path = path.resolve()

    if path.exists() and not force:
        # Simple check if directory is not empty
        if any(path.iterdir()):
            console.print(
                f"[yellow]Warning: {path} is not empty. Use --force to continue.[/yellow]"
            )
            if not force:
                raise typer.Exit(1)

    path.mkdir(parents=True, exist_ok=True)

    config = load_config()
    manager = AFSManager(config)

    try:
        root = manager.init(path, force=force)
        console.print(f"[green]Initialized AFS at {path}[/green]")
        console.print(f"  Project: {root.project_name}")

        if mount:
            for m in mount:
                parts = m.split(":")
                if len(parts) < 2:
                    console.print(f"[red]Invalid mount spec: {m}[/red]")
                    continue

                src = Path(parts[0])
                try:
                    mtype = MountType(parts[1])
                except ValueError:
                    console.print(f"[red]Invalid mount type: {parts[1]}[/red]")
                    continue

                alias = parts[2] if len(parts) > 2 else None

                try:
                    manager.mount(src, mtype, alias, context_path=path / ".context")
                    console.print(
                        f"  Mounted {src} -> {mtype.value}/{alias or src.name}"
                    )
                except Exception as e:
                    console.print(f"[red]Failed to mount {src}: {e}[/red]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def init(
    path: Path = typer.Argument(Path("."), help="Project path to initialize"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing AFS"),
) -> None:
    """Initialize AFS in a directory."""
    config = load_config()
    manager = AFSManager(config)

    try:
        root = manager.init(path, force=force)
        console.print(f"[green]Initialized AFS at {root.path}[/green]")
        console.print(f"  Project: {root.project_name}")
        console.print(f"  Directories: {', '.join(mt.value for mt in MountType)}")
    except FileExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
        console.print("Use --force to overwrite existing AFS")
        raise typer.Exit(1)


@app.command()
def mount(
    source: Path = typer.Argument(..., help="Source file or directory to mount"),
    mount_type: str = typer.Argument(
        ..., help="Mount type: memory|knowledge|tools|scratchpad|history"
    ),
    alias: Optional[str] = typer.Option(
        None, "--alias", "-a", help="Custom name for the mount"
    ),
) -> None:
    """Mount a resource into AFS."""
    config = load_config()
    manager = AFSManager(config)

    try:
        mt = MountType(mount_type)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid mount type '{mount_type}'")
        console.print(f"Valid types: {', '.join(mt.value for mt in MountType)}")
        raise typer.Exit(1)

    try:
        mount_point = manager.mount(source, mt, alias)
        console.print(
            f"[green]Mounted {source} -> "
            f"{mount_point.mount_type.value}/{mount_point.name}[/green]"
        )
    except (FileNotFoundError, FileExistsError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def unmount(
    alias: str = typer.Argument(..., help="Name of the mount to remove"),
    mount_type: str = typer.Argument(
        ..., help="Mount type: memory|knowledge|tools|scratchpad|history"
    ),
) -> None:
    """Remove a mount from AFS."""
    config = load_config()
    manager = AFSManager(config)

    try:
        mt = MountType(mount_type)
    except ValueError:
        console.print(f"[red]Error:[/red] Invalid mount type '{mount_type}'")
        raise typer.Exit(1)

    if manager.unmount(alias, mt):
        console.print(f"[green]Unmounted {mt.value}/{alias}[/green]")
    else:
        console.print(f"[yellow]Mount '{alias}' not found in {mt.value}[/yellow]")


@app.command("list")
def list_context() -> None:
    """List current AFS structure."""
    config = load_config()
    manager = AFSManager(config)

    try:
        root = manager.list_afs_structure()

        # Create tree view
        tree = Tree(f"[bold purple]{root.project_name}[/bold purple] (.context)")

        for mt in MountType:
            mounts = root.mounts.get(mt, [])
            dir_config = config.get_directory_config(mt.value)
            policy_str = dir_config.policy.value if dir_config else "read_only"

            # Color based on policy
            policy_color = {
                "read_only": "blue",
                "writable": "green",
                "executable": "red",
            }.get(policy_str, "white")

            branch = tree.add(
                f"[{policy_color}]{mt.value}[/{policy_color}] "
                f"[dim]({policy_str})[/dim]"
            )

            if mounts:
                for mount in mounts:
                    link_indicator = "→" if mount.is_symlink else "·"
                    branch.add(f"{mount.name} {link_indicator} {mount.source}")
            else:
                branch.add("[dim](empty)[/dim]")

        console.print(tree)
        console.print(f"\n[dim]Path: {root.path}[/dim]")
        console.print(f"[dim]Total mounts: {root.total_mounts}[/dim]")

    except FileNotFoundError:
        console.print("[yellow]No AFS initialized. Run 'hafs init' first.[/yellow]")


@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Remove AFS from current directory."""
    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS found in current directory tree.[/yellow]")
        raise typer.Exit(0)

    if not force:
        confirm = typer.confirm(
            f"Remove AFS at {context_root}? This cannot be undone."
        )
        if not confirm:
            raise typer.Abort()

    config = load_config()
    manager = AFSManager(config)
    manager.clean(context_root)
    console.print("[green]AFS removed.[/green]")


@app.command()
def projects(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show mount details"),
) -> None:
    """List all discovered AFS-enabled projects."""
    config = load_config()

    # Use tracked projects from config if available, otherwise discover
    search_paths = config.tracked_projects if config.tracked_projects else None
    found_projects = discover_projects(search_paths)

    if not found_projects:
        console.print("[yellow]No AFS-enabled projects found.[/yellow]")
        return

    table = Table(title="AFS-Enabled Projects")
    table.add_column("Project", style="purple")
    table.add_column("Path", style="dim")
    table.add_column("Mounts", style="cyan")

    for project in found_projects:
        mount_counts = []
        for mt in MountType:
            count = len(project.mounts.get(mt, []))
            if count > 0:
                mount_counts.append(f"{mt.value}:{count}")

        table.add_row(
            project.project_name,
            str(project.path.parent),
            ", ".join(mount_counts) or "-",
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(found_projects)} projects[/dim]")


@app.command()
def logs(
    parser: str = typer.Option("gemini", help="Parser: gemini|claude|antigravity"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search query"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max items to show"),
) -> None:
    """Browse AI agent logs."""
    from hafs.core.parsers.registry import ParserRegistry

    config = load_config()

    parser_class = ParserRegistry.get(parser)
    if not parser_class:
        console.print(f"[red]Unknown parser: {parser}[/red]")
        console.print(f"Available: {', '.join(ParserRegistry.list_parsers())}")
        raise typer.Exit(1)

    parser_cfg = getattr(config.parsers, parser, None)
    if parser_cfg and not parser_cfg.enabled:
        console.print(f"[yellow]Parser '{parser}' is disabled in configuration.[/yellow]")
        raise typer.Exit(1)

    parser_base_path = parser_cfg.base_path if parser_cfg else None
    parser_instance = parser_class(base_path=parser_base_path)

    if not parser_instance.exists():
        console.print(f"[yellow]No logs found at {parser_instance.base_path}[/yellow]")
        return

    effective_limit = limit or (parser_cfg.max_items if parser_cfg else 50)
    items = parser_instance.parse(max_items=effective_limit)

    if search:
        items = parser_instance.search(search, items)

    if not items:
        console.print("[yellow]No items found.[/yellow]")
        return

    # Display based on parser type
    if parser == "gemini":
        _display_gemini_sessions(items)
    elif parser == "claude":
        _display_claude_plans(items)
    elif parser == "antigravity":
        _display_antigravity_brains(items)


# Subcommand group for context file management
ctx_app = typer.Typer(help="Context file management")
app.add_typer(ctx_app, name="ctx")


@ctx_app.command("list")
def ctx_list(
    subpath: str = typer.Argument(".", help="Subpath within context"),
) -> None:
    """List files in the context directory."""
    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    target_path = _resolve_context_path(context_root, subpath)
    if not target_path.exists():
        console.print(f"[red]Path not found: {target_path}[/red]")
        raise typer.Exit(1)

    tree = Tree(f"[bold]{target_path.name}[/bold]")
    for item in target_path.iterdir():
        if item.is_dir():
            tree.add(f"[blue]{item.name}/[/blue]")
        else:
            tree.add(item.name)
    console.print(tree)


@ctx_app.command("view")
def ctx_view(
    path: str = typer.Argument(..., help="File path within context"),
) -> None:
    """View a file in the context."""
    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    target_path = _resolve_context_path(context_root, path)
    if not target_path.exists():
        console.print(f"[red]File not found: {target_path}[/red]")
        raise typer.Exit(1)

    if target_path.is_dir():
        console.print(f"[yellow]{path} is a directory.[/yellow]")
        return

    try:
        content = target_path.read_text(errors="replace")
        console.print(Panel(content, title=str(path), border_style="blue"))
    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")


@ctx_app.command("edit")
def ctx_edit(
    path: str = typer.Argument(..., help="File path within context"),
    editor: str = typer.Option("nvim", "--editor", "-e", help="Editor to use"),
) -> None:
    """Open a context file in an editor."""
    import os
    import shutil
    import subprocess

    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    target_path = _resolve_context_path(context_root, path)

    # Ensure parent exists
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)

    editor_cmd = shutil.which(editor)
    if not editor_cmd:
        editor_cmd = os.environ.get("EDITOR", "vi")

    try:
        subprocess.call([editor_cmd, str(target_path)])
        console.print(f"[green]Edited {path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to launch editor: {e}[/red]")


@ctx_app.command("append")
def ctx_append(
    path: str = typer.Argument(..., help="File path within context"),
    text: str = typer.Argument(..., help="Text to append"),
) -> None:
    """Append text to a context file."""
    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    target_path = _resolve_context_path(context_root, path)

    # Ensure parent exists
    if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(target_path, "a") as f:
            f.write(text + "\n")
        console.print(f"[green]Appended to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Error writing file: {e}[/red]")


def _display_gemini_sessions(sessions) -> None:  # type: ignore[no-untyped-def]
    """Display Gemini sessions."""
    table = Table(title="Gemini Sessions")
    table.add_column("ID", style="purple")
    table.add_column("Time", style="dim")
    table.add_column("Messages", style="cyan")
    table.add_column("Tokens", style="green")

    for session in sessions:
        table.add_row(
            session.short_id,
            session.start_time.strftime("%Y-%m-%d %H:%M"),
            str(len(session.messages)),
            str(session.total_tokens),
        )

    console.print(table)


def _display_claude_plans(plans) -> None:  # type: ignore[no-untyped-def]
    """Display Claude plans."""
    table = Table(title="Claude Plans")
    table.add_column("Title", style="purple")
    table.add_column("Progress", style="cyan")
    table.add_column("Tasks", style="dim")

    for plan in plans:
        done, total = plan.progress
        progress_bar = f"[{'█' * done}{'░' * (total - done)}]" if total > 0 else "-"
        table.add_row(
            plan.title[:40],
            f"{done}/{total} {progress_bar}",
            f"✓{plan.done_count} ⏳{plan.in_progress_count} ○{plan.todo_count}",
        )

    console.print(table)


def _display_antigravity_brains(brains) -> None:  # type: ignore[no-untyped-def]
    """Display Antigravity brains."""
    table = Table(title="Antigravity Brains")
    table.add_column("ID", style="purple")
    table.add_column("Title", style="cyan")
    table.add_column("Tasks", style="dim")

    for brain in brains:
        done, total = brain.progress
        table.add_row(
            brain.short_id,
            brain.title[:40],
            f"{done}/{total}",
        )

    console.print(table)


@app.command()
def tui() -> None:
    """Launch the TUI interface."""
    from hafs.ui.app import run

    run()


@app.command()
def chat(
    backend: str = typer.Option("gemini", "--backend", "-b", help="Default backend"),
    agents: Optional[str] = typer.Option(
        None, "--agents", "-a", help="Comma-separated list of agents to start (name:role)"
    ),
) -> None:
    """Launch the multi-agent chat TUI.

    Chat with multiple AI agents simultaneously, using @mentions to direct
    messages to specific agents. Agents can collaborate on tasks and share
    context.

    Example:
        hafs chat
        hafs chat --backend claude
        hafs chat --agents "Planner:planner,Coder:coder"
    """
    from hafs.ui.app import run_chat

    # Parse agents if provided
    agent_list = None
    if agents:
        agent_list = []
        for agent_spec in agents.split(","):
            parts = agent_spec.strip().split(":")
            if len(parts) == 2:
                agent_list.append({"name": parts[0], "role": parts[1]})
            else:
                agent_list.append({"name": parts[0], "role": "general"})

    run_chat(default_backend=backend, agents=agent_list)


# Keep orchestrate as deprecated alias for backwards compatibility
@app.command(hidden=True)
def orchestrate(
    backend: str = typer.Option("gemini", "--backend", "-b", help="Default backend"),
    agents: Optional[str] = typer.Option(
        None, "--agents", "-a", help="Comma-separated list of agents to start (name:role)"
    ),
) -> None:
    """[Deprecated] Use 'hafs chat' instead."""
    console.print("[yellow]Note: 'orchestrate' is deprecated. Use 'hafs chat' instead.[/yellow]")
    chat(backend=backend, agents=agents)


# Subcommand group for history management
history_app = typer.Typer(help="History pipeline management")
app.add_typer(history_app, name="history")


@history_app.command("list")
def history_list(
    limit: int = typer.Option(50, "--limit", "-n", help="Max entries to show"),
    session: Optional[str] = typer.Option(None, "--session", "-s", help="Filter by session ID"),
) -> None:
    """List recent history entries."""
    from hafs.core.afs.discovery import find_context_root
    from hafs.core.history import HistoryLogger, HistoryQuery, OperationType

    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    history_dir = context_root / "history"
    if not history_dir.exists():
        console.print("[yellow]No history found. History directory does not exist.[/yellow]")
        return

    logger = HistoryLogger(history_dir)

    query = HistoryQuery(limit=limit, session_id=session)
    entries = logger.query(query)

    if not entries:
        console.print("[yellow]No history entries found.[/yellow]")
        return

    table = Table(title="History Entries")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="purple")
    table.add_column("Success", style="green")
    table.add_column("Session", style="dim")

    for entry in entries:
        # Parse timestamp for display
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            time_str = ts.strftime("%H:%M:%S")
        except Exception:
            time_str = entry.timestamp[:19]

        success_icon = "[green]✓[/green]" if entry.operation.success else "[red]✗[/red]"

        table.add_row(
            time_str,
            entry.operation.type.value,
            entry.operation.name[:30],
            success_icon,
            entry.session_id[:8] + "...",
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(entries)} of {limit} max entries[/dim]")


@history_app.command("show")
def history_show(
    entry_id: str = typer.Argument(..., help="Entry ID to show"),
) -> None:
    """Show details of a specific history entry."""
    import json
    from hafs.core.afs.discovery import find_context_root
    from hafs.core.history import HistoryLogger, HistoryQuery

    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    history_dir = context_root / "history"
    logger = HistoryLogger(history_dir)

    # Search all entries for the ID
    entries = logger.query(HistoryQuery(limit=10000))
    entry = next((e for e in entries if e.id == entry_id or e.id.startswith(entry_id)), None)

    if not entry:
        console.print(f"[red]Entry not found: {entry_id}[/red]")
        raise typer.Exit(1)

    # Display entry details
    console.print(Panel(
        f"[bold]ID:[/bold] {entry.id}\n"
        f"[bold]Timestamp:[/bold] {entry.timestamp}\n"
        f"[bold]Session:[/bold] {entry.session_id}\n"
        f"[bold]Type:[/bold] {entry.operation.type.value}\n"
        f"[bold]Name:[/bold] {entry.operation.name}\n"
        f"[bold]Success:[/bold] {entry.operation.success}\n"
        f"[bold]Duration:[/bold] {entry.operation.duration_ms or 'N/A'} ms",
        title="History Entry",
        border_style="blue",
    ))

    if entry.operation.input:
        console.print("\n[bold]Input:[/bold]")
        console.print(json.dumps(entry.operation.input, indent=2))

    if entry.operation.output:
        console.print("\n[bold]Output:[/bold]")
        output_str = str(entry.operation.output)
        if len(output_str) > 500:
            output_str = output_str[:500] + "..."
        console.print(output_str)

    if entry.operation.error:
        console.print(f"\n[bold red]Error:[/bold red] {entry.operation.error}")


@history_app.command("sessions")
def history_sessions(
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
) -> None:
    """List history sessions."""
    from hafs.core.afs.discovery import find_context_root
    from hafs.core.history import SessionManager, SessionStatus

    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    sessions_dir = context_root / "history" / "sessions"
    if not sessions_dir.exists():
        console.print("[yellow]No sessions found.[/yellow]")
        return

    manager = SessionManager(sessions_dir)

    status_filter = None
    if status:
        try:
            status_filter = SessionStatus(status)
        except ValueError:
            console.print(f"[red]Invalid status: {status}[/red]")
            console.print(f"Valid: {', '.join(s.value for s in SessionStatus)}")
            raise typer.Exit(1)

    sessions = manager.list_sessions(status=status_filter, limit=limit)

    if not sessions:
        console.print("[yellow]No sessions found.[/yellow]")
        return

    table = Table(title="History Sessions")
    table.add_column("ID", style="purple")
    table.add_column("Status", style="cyan")
    table.add_column("Created", style="dim")
    table.add_column("Operations", style="green")
    table.add_column("Title", style="dim")

    for session in sessions:
        # Parse timestamp
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(session.created_at.replace("Z", "+00:00"))
            time_str = ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            time_str = session.created_at[:16]

        status_color = {
            "active": "green",
            "suspended": "yellow",
            "completed": "blue",
            "aborted": "red",
        }.get(session.status.value, "white")

        title = session.summary.title[:30] if session.summary and session.summary.title else "-"

        table.add_row(
            session.id[:12] + "...",
            f"[{status_color}]{session.status.value}[/{status_color}]",
            time_str,
            str(session.stats.operation_count),
            title,
        )

    console.print(table)


@history_app.command("search")
def history_search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
) -> None:
    """Search history entries."""
    from hafs.core.afs.discovery import find_context_root
    from hafs.core.history import HistoryLogger, HistoryQuery

    context_root = find_context_root()
    if not context_root:
        console.print("[yellow]No AFS context found.[/yellow]")
        raise typer.Exit(1)

    history_dir = context_root / "history"
    logger = HistoryLogger(history_dir)

    # Get all entries and filter by query
    all_entries = logger.query(HistoryQuery(limit=10000))

    query_lower = query.lower()
    matches = []
    for entry in all_entries:
        # Search in name, input, output, tags
        searchable = [
            entry.operation.name,
            str(entry.operation.input),
            str(entry.operation.output) if entry.operation.output else "",
        ]
        searchable.extend(entry.metadata.tags)

        if any(query_lower in s.lower() for s in searchable):
            matches.append(entry)
            if len(matches) >= limit:
                break

    if not matches:
        console.print(f"[yellow]No entries matching '{query}'[/yellow]")
        return

    table = Table(title=f"Search Results: '{query}'")
    table.add_column("Time", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="purple")
    table.add_column("Match Context", style="dim")

    for entry in matches:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            time_str = ts.strftime("%H:%M:%S")
        except Exception:
            time_str = entry.timestamp[:19]

        # Find match context
        context = entry.operation.name
        if query_lower in str(entry.operation.input).lower():
            context = str(entry.operation.input)[:40]
        elif query_lower in str(entry.operation.output or "").lower():
            context = str(entry.operation.output)[:40]

        table.add_row(
            time_str,
            entry.operation.type.value,
            entry.operation.name[:25],
            context[:40] + "...",
        )

    console.print(table)
    console.print(f"\n[dim]Found {len(matches)} matches[/dim]")


# Subcommand group for agent management
agent_app = typer.Typer(help="Manage AI agents")
app.add_typer(agent_app, name="agent")


@agent_app.command("list")
def agent_list() -> None:
    """List available backends and roles."""
    from hafs.models.agent import AgentRole

    console.print("\n[bold]Available Backends:[/bold]")
    console.print("  • gemini - Gemini CLI")
    console.print("  • claude - Claude CLI")

    console.print("\n[bold]Available Roles:[/bold]")
    for role in AgentRole:
        console.print(f"  • {role.value}")

    console.print("\n[dim]Use 'hafs chat' to start the multi-agent TUI[/dim]")


@app.command()
def version() -> None:
    """Show version information."""
    from hafs import __version__

    console.print(
        Panel(
            f"[bold purple]HAFS[/bold purple] - Halext Agentic File System\n"
            f"Version: {__version__}",
            border_style="purple",
        )
    )


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
