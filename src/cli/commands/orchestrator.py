import asyncio
from typing import List, Optional, TYPE_CHECKING

import typer
from rich.console import Console

from tui.console import orchestrator as ui_orchestrator

if TYPE_CHECKING:
    from core.orchestration_entrypoint import AgentSpec

orchestrator_app = typer.Typer(
    name="orchestrate",
    help="Run a plan→execute→verify→summarize pipeline with coordinator or swarm.",
)
console = Console()


def _parse_agent_spec(value: str):
    from core.orchestration_entrypoint import AgentSpec
    from models.agent import AgentRole

    parts = [p.strip() for p in value.split(":") if p.strip()]
    if len(parts) < 2:
        raise typer.BadParameter("agent spec must be name:role[:persona]")
    name = parts[0]
    role_str = parts[1].lower()
    persona = parts[2] if len(parts) > 2 else None

    try:
        role = AgentRole(role_str)
    except ValueError as exc:
        valid = ", ".join(r.value for r in AgentRole)
        raise typer.BadParameter(f"invalid role '{role_str}'. Valid: {valid}") from exc

    return AgentSpec(name=name, role=role, persona=persona)


@orchestrator_app.command("run")
def run(
    ctx: typer.Context,
    topic: Optional[str] = typer.Argument(None, help="Orchestration topic/task"),
    mode: str = typer.Option(
        "coordinator", help="coordinator|swarm (SwarmCouncil multi-agent mode)"
    ),
    agent: Optional[List[str]] = typer.Option(
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

    agent_specs = [_parse_agent_spec(spec) for spec in agent] if agent else None

    async def _run() -> None:
        from core.orchestration_entrypoint import run_orchestration

        result = await run_orchestration(
            mode=mode,
            topic=topic,
            agents=agent_specs,
            default_backend=backend,
        )
        if result:
            ui_orchestrator.render_orchestration_result(console, result)

    asyncio.run(_run())
