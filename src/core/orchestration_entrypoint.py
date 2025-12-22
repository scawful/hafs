"""Unified orchestration entrypoint for pipeline-driven flows."""

from __future__ import annotations

import asyncio
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from config.loader import load_config
from core.orchestration import OrchestrationPipeline, PipelineContext, PipelineStep
from models.agent import AgentRole


@dataclass(frozen=True)
class AgentSpec:
    """Agent specification for coordinator pipelines."""

    name: str
    role: AgentRole
    persona: Optional[str] = None
    backend_name: Optional[str] = None
    system_prompt: Optional[str] = None


async def _attach_history(coordinator: Any, config: Any) -> None:
    try:
        from core.history import HistoryLogger, SessionManager
    except Exception:
        return

    context_root = getattr(config.general, "context_root", None)
    if context_root is None:
        return

    history_dir = context_root / "history"
    sessions_dir = history_dir / "sessions"
    from pathlib import Path

    project_id = Path.cwd().name

    session_manager = SessionManager(sessions_dir, project_id=project_id)
    history_logger = HistoryLogger(
        history_dir=history_dir,
        session_manager=session_manager,
        project_id=project_id,
    )
    session_manager.set_history_logger(history_logger)
    session_manager.create()

    coordinator.set_session_manager(session_manager)
    coordinator.set_history_logger(history_logger)


async def _collect_response(
    coordinator: Any, agent_name: str, message: str
) -> str:
    recipient = await coordinator.route_message(f"@{agent_name} {message}", sender="system")
    chunks: list[str] = []
    async for chunk in coordinator.stream_agent_response(recipient):
        chunks.append(chunk)
    return "".join(chunks).strip()


def _pick_agent(coordinator: Any, role: AgentRole) -> Optional[str]:
    names = coordinator.list_agents_by_role(role)
    if names:
        return names[0]
    agents = coordinator.list_agents()
    return agents[0] if agents else None


def _persona_name_map(config: Any) -> dict[AgentRole, Optional[str]]:
    try:
        from core.personas import PersonaRegistry

        registry = PersonaRegistry.load()
        return {
            AgentRole.PLANNER: registry.default_for_role(AgentRole.PLANNER),
            AgentRole.RESEARCHER: registry.default_for_role(AgentRole.RESEARCHER),
            AgentRole.CODER: registry.default_for_role(AgentRole.CODER),
            AgentRole.CRITIC: registry.default_for_role(AgentRole.CRITIC),
            AgentRole.GENERAL: registry.default_for_role(AgentRole.GENERAL),
        }
    except Exception:
        return {}


def _builtin_agent_specs(persona_name: dict[AgentRole, Optional[str]]) -> list[AgentSpec]:
    return [
        AgentSpec(name="Planner", role=AgentRole.PLANNER, persona=persona_name.get(AgentRole.PLANNER)),
        AgentSpec(name="Researcher", role=AgentRole.RESEARCHER, persona=persona_name.get(AgentRole.RESEARCHER)),
        AgentSpec(name="Coder", role=AgentRole.CODER, persona=persona_name.get(AgentRole.CODER)),
        AgentSpec(name="Critic", role=AgentRole.CRITIC, persona=persona_name.get(AgentRole.CRITIC)),
        AgentSpec(name="Generalist", role=AgentRole.GENERAL, persona=persona_name.get(AgentRole.GENERAL)),
    ]


def _default_agent_specs(config: Any) -> list[AgentSpec]:
    persona_map = _persona_name_map(config)
    persona_name = {
        role: persona.name if persona else None
        for role, persona in persona_map.items()
    }

    default_agents = []
    if config is not None and hasattr(config, "orchestrator"):
        default_agents = list(getattr(config.orchestrator, "default_agents", []) or [])

    if default_agents:
        specs: list[AgentSpec] = []
        for agent_cfg in default_agents:
            try:
                role = AgentRole(agent_cfg.role)
            except ValueError:
                continue
            specs.append(
                AgentSpec(
                    name=agent_cfg.name,
                    role=role,
                    persona=persona_name.get(role),
                    backend_name=agent_cfg.backend,
                    system_prompt=agent_cfg.system_prompt or None,
                )
            )
        if specs:
            return specs

    return _builtin_agent_specs(persona_name)


def _unique_agent_name(coordinator: Any, base_name: str) -> str:
    existing = {name.lower() for name in coordinator.list_agents()}
    if base_name.lower() not in existing:
        return base_name

    suffix = 2
    while True:
        candidate = f"{base_name}-{suffix}"
        if candidate.lower() not in existing:
            return candidate
        suffix += 1


async def _notify_agent_registered(
    callback: Optional[Callable[[Any], Any]],
    lane: Any,
) -> None:
    if not callback:
        return
    try:
        result = callback(lane)
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


async def _run_coordinator_pipeline(
    topic: str,
    agents: Optional[list[AgentSpec]] = None,
    default_backend: str = "gemini",
    config: Any | None = None,
    coordinator: Any | None = None,
    ensure_roles: bool | None = None,
    start_agents: bool | None = None,
    stop_agents: bool | None = None,
    on_agent_registered: Optional[Callable[[Any], Any]] = None,
) -> str:
    from agents.core.coordinator import AgentCoordinator

    config = config or load_config()
    import backends  # noqa: F401

    created_coordinator = coordinator is None
    if coordinator is None:
        coordinator = AgentCoordinator(config)
        await _attach_history(coordinator, config)

    agent_specs = agents or _default_agent_specs(config)
    persona_map = _persona_name_map(config)
    persona_name = {
        role: persona.name if persona else None
        for role, persona in persona_map.items()
    }
    fallback_specs = _builtin_agent_specs(persona_name)
    fallback_by_role = {spec.role: spec for spec in fallback_specs}
    specs_by_role = {spec.role: spec for spec in agent_specs}
    existing_names = {name.lower() for name in coordinator.list_agents()}

    async def _register_spec(spec: AgentSpec) -> None:
        name = _unique_agent_name(coordinator, spec.name)
        persona = spec.persona or fallback_by_role.get(spec.role, spec).persona
        backend_name = spec.backend_name or default_backend
        system_prompt = spec.system_prompt or ""
        lane = await coordinator.register_agent(
            name=name,
            role=spec.role,
            backend_name=backend_name,
            system_prompt=system_prompt,
            persona=persona,
        )
        existing_names.add(lane.agent.name.lower())
        await _notify_agent_registered(on_agent_registered, lane)

    for spec in agent_specs:
        if spec.name.lower() in existing_names:
            continue
        await _register_spec(spec)

    if ensure_roles is None:
        has_default_agents = bool(
            config
            and hasattr(config, "orchestrator")
            and getattr(config.orchestrator, "default_agents", None)
        )
        ensure_roles = agents is None and not has_default_agents

    if ensure_roles:
        required_roles = [
            AgentRole.RESEARCHER,
            AgentRole.PLANNER,
            AgentRole.CODER,
            AgentRole.CRITIC,
            AgentRole.GENERAL,
        ]
        for role in required_roles:
            if coordinator.list_agents_by_role(role):
                continue
            spec = specs_by_role.get(role) or fallback_by_role.get(role)
            if not spec:
                continue
            await _register_spec(spec)

    if start_agents is None:
        start_agents = True
    if stop_agents is None:
        stop_agents = created_coordinator

    if start_agents:
        if created_coordinator:
            await coordinator.start_all_agents()
        else:
            for lane in coordinator.agents.values():
                if not lane.is_running:
                    await lane.start()

    async def _step_research(context: PipelineContext) -> str:
        agent = _pick_agent(coordinator, AgentRole.RESEARCHER)
        if not agent:
            context.research = ""
            return ""
        prompt = (
            f"Gather any relevant context for: {context.topic}. "
            "Summarize key facts or references succinctly."
        )
        research = await _collect_response(coordinator, agent, prompt)
        context.research = research
        return research

    async def _step_plan(context: PipelineContext) -> str:
        agent = _pick_agent(coordinator, AgentRole.PLANNER)
        if not agent:
            context.plan = ""
            return ""
        research = f"Research Notes:\n{context.research}\n\n" if context.research else ""
        prompt = (
            f"Create a concise plan for: {context.topic}. "
            "Return steps as bullet points.\n\n"
            f"{research}"
        )
        plan = await _collect_response(coordinator, agent, prompt)
        context.plan = plan
        return plan

    async def _step_execute(context: PipelineContext) -> str:
        agent = _pick_agent(coordinator, AgentRole.CODER)
        if not agent:
            context.results = ""
            return ""
        research = f"Research Notes:\n{context.research}\n\n" if context.research else ""
        prompt = (
            f"Execute on this request: {context.topic}.\n\n"
            f"{research}"
            f"Plan:\n{context.plan}\n\n"
            "Provide concrete output or findings."
        )
        results = await _collect_response(coordinator, agent, prompt)
        context.results = results
        return results

    async def _step_verify(context: PipelineContext) -> str:
        agent = _pick_agent(coordinator, AgentRole.CRITIC)
        if not agent:
            context.critique = ""
            return ""
        prompt = (
            "Review the execution output for risks, gaps, or errors.\n\n"
            f"Output:\n{context.results}"
        )
        critique = await _collect_response(coordinator, agent, prompt)
        context.critique = critique
        return critique

    async def _step_summarize(context: PipelineContext) -> str:
        agent = _pick_agent(coordinator, AgentRole.GENERAL)
        if not agent:
            agent = _pick_agent(coordinator, AgentRole.PLANNER)
        if not agent:
            context.summary = context.results
            return context.summary
        research = f"Research Notes:\n{context.research}\n\n" if context.research else ""
        prompt = (
            "Summarize the plan, output, and critique into a concise briefing.\n\n"
            f"{research}"
            f"Plan:\n{context.plan}\n\n"
            f"Output:\n{context.results}\n\n"
            f"Critique:\n{context.critique}"
        )
        summary = await _collect_response(coordinator, agent, prompt)
        context.summary = summary
        return summary

    context = PipelineContext(topic=topic)
    pipeline = OrchestrationPipeline(
        [
            PipelineStep(name="research", kind="research", run=_step_research, required=False),
            PipelineStep(name="plan", kind="plan", run=_step_plan),
            PipelineStep(name="execute", kind="execute", run=_step_execute),
            PipelineStep(name="verify", kind="verify", run=_step_verify),
            PipelineStep(name="summarize", kind="summarize", run=_step_summarize),
        ]
    )

    await pipeline.run(context)

    if stop_agents:
        await coordinator.stop_all_agents()

    return context.summary or str(context.results)


async def _run_swarm(topic: str) -> str:
    from agents.swarm.swarm import SwarmCouncil
    from core.config import hafs_config
    from core.plugin_loader import load_all_agents_from_package, load_plugins
    from core.registry import agent_registry
    import agents as agents_pkg

    try:
        config = hafs_config.context_agents
    except Exception:
        config = None

    if config:
        if config.provider:
            os.environ["HAFS_MODEL_PROVIDER"] = config.provider
        if config.model:
            os.environ["HAFS_MODEL_MODEL"] = config.model
        if config.rotation:
            os.environ["HAFS_MODEL_ROTATION"] = ",".join(config.rotation)
        if config.prefer_gpu_nodes:
            os.environ["HAFS_PREFER_GPU_NODES"] = "1"
        if config.prefer_remote_nodes:
            os.environ["HAFS_PREFER_REMOTE_NODES"] = "1"
        if (config.prefer_remote_nodes or config.prefer_gpu_nodes) and not os.environ.get("HAFS_ENABLE_OLLAMA"):
            os.environ["HAFS_ENABLE_OLLAMA"] = "1"

    load_plugins()
    load_all_agents_from_package(agents_pkg)

    instantiated_agents: dict[str, Any] = {}
    for name, cls in agent_registry.list_agents().items():
        try:
            instantiated_agents[name] = cls()
        except TypeError:
            continue

    council = SwarmCouncil(instantiated_agents)
    await council.setup()
    council.attach_history()
    return await council.run_session(topic)


async def run_orchestration(
    *,
    mode: str,
    topic: str,
    agents: Optional[list[AgentSpec]] = None,
    default_backend: str = "gemini",
    config: Any | None = None,
    coordinator: Any | None = None,
    ensure_roles: bool | None = None,
    start_agents: bool | None = None,
    stop_agents: bool | None = None,
    on_agent_registered: Optional[Callable[[Any], Any]] = None,
) -> str:
    """Unified orchestration entrypoint."""
    mode_key = mode.strip().lower()
    if mode_key == "swarm":
        return await _run_swarm(topic)
    if mode_key == "coordinator":
        return await _run_coordinator_pipeline(
            topic=topic,
            agents=agents,
            default_backend=default_backend,
            config=config,
            coordinator=coordinator,
            ensure_roles=ensure_roles,
            start_agents=start_agents,
            stop_agents=stop_agents,
            on_agent_registered=on_agent_registered,
        )
    raise ValueError("mode must be 'swarm' or 'coordinator'")


def run_orchestration_sync(**kwargs: Any) -> str:
    """Synchronous wrapper for CLI entrypoints."""
    return asyncio.run(run_orchestration(**kwargs))
