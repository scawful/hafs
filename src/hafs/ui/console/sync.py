"""Console UI for sync commands."""
from rich.console import Console
from typing import Any

def render_sync_profiles(console: Console, profiles: list[Any]) -> None:
    """Render sync profiles list."""
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

def render_sync_profile_details(console: Console, profile: Any) -> None:
    """Render sync profile details."""
    console.print(f"name: {profile.name}")
    console.print(f"source: {profile.source}")
    console.print(f"scope: {profile.scope}")
    console.print(f"direction: {profile.direction}")
    console.print(f"transport: {profile.transport}")
    console.print(f"delete: {profile.delete}")
    console.print(f"exclude: {profile.exclude}")
    console.print(f"targets: {[t.label() for t in profile.targets]}")

def render_unknown_profile(console: Console, name: str) -> None:
    """Render unknown profile error."""
    console.print(f"[red]Unknown sync profile: {name}[/red]")

def render_sync_results(console: Console, results: list[Any]) -> None:
    """Render sync run results."""
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
