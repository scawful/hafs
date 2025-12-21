"""Swarm Controller / Council of Agents (Public).

Orchestrates multiple agents to perform deep research and synthesis.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.core.base import BaseAgent
from agents.swarm.specialists import (
    CouncilReviewer,
    DeepDiveDocumenter,
    SwarmStrategist,
)
from hafs.core.orchestration import OrchestrationPipeline, PipelineContext, PipelineStep

logger = logging.getLogger(__name__)

# Generic Swarm Status
try:
    from hafs.ui.swarm_status import SwarmStatus, AgentNode
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class AgentNode:
        id: str
        label: str
        status: str = "pending"

    @dataclass
    class SwarmStatus:
        nodes: List[AgentNode] = field(default_factory=list)
        edges: List[dict] = field(default_factory=list)
        active_topic: str = ""
        start_time: str = ""


STATUS_FILE = Path.home() / ".context/swarm_status.json"


class SwarmCouncil:
    """Orchestrator for swarm sessions."""

    def __init__(self, instantiated_agents: Dict[str, BaseAgent]):
        """
        Initializes the SwarmCouncil with a dictionary of instantiated agents.
        """
        self.agents_map = dict(instantiated_agents)
        self._wire_default_roles()
        self._history_path: Optional[Path] = None

    def _wire_default_roles(self) -> None:
        """Ensure expected role aliases exist in the agent map."""
        alias_map = {
            "strategist": ["SwarmStrategist"],
            "reviewer": ["CouncilReviewer"],
            "documenter": ["DeepDiveDocumenter"],
        }
        for alias, candidates in alias_map.items():
            if alias in self.agents_map:
                continue
            for name in candidates:
                agent = self.agents_map.get(name)
                if agent is not None:
                    self.agents_map[alias] = agent
                    break

        if "primary_kb" not in self.agents_map:
            kb_agent = self._select_primary_kb()
            if kb_agent:
                self.agents_map["primary_kb"] = kb_agent

    def attach_history(self, context_root: Optional[Path] = None):
        """Attach history logging to swarm sessions."""
        if context_root:
            self._history_path = context_root / "swarm_history.json"
        else:
            self._history_path = Path.home() / ".context/swarm_history.json"

    def _write_status(self, status: SwarmStatus):
        """Write current swarm status to disk for UI/CLI consumption."""
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump(status.__dict__, f, default=str)
        except Exception as e:
            logger.error(f"Failed to write swarm status: {e}")

    async def setup(self):
        """Perform any async setup for agents."""
        for agent in self.agents_map.values():
            await agent.setup()

    async def _run_parallel_tasks(self, task_map: Dict[str, Any]) -> Dict[str, Any]:
        """Run tasks in parallel and return a dict of results."""
        keys = list(task_map.keys())
        tasks = [self.agents_map[k].run_task(task_map[k]) for k in keys]
        results = await asyncio.gather(*tasks)
        return {keys[i]: results[i] for i in range(len(keys))}

    def _status_from_context(self, context: PipelineContext) -> SwarmStatus:
        """Create SwarmStatus object from pipeline context."""
        return SwarmStatus(
            active_topic=getattr(context, "topic", "Research"),
            start_time=datetime.now().isoformat()
        )

    async def _step_plan(self, context: PipelineContext):
        """Phase 1: Planning."""
        logger.info(f"Swarm Planning: {context.topic}")
        strategist = self.agents_map.get("strategist")
        if not strategist:
            return

        plan = await strategist.run_task(context.topic)
        context.data["plan"] = plan

        # Update status
        status = self._status_from_context(context)
        status.nodes.append(AgentNode(id="strategist", label="Strategist", status="complete"))
        self._write_status(status)

    async def _step_collect(self, context: PipelineContext):
        """Phase 2: Data Collection."""
        logger.info("Swarm Collecting...")
        plan = context.data.get("plan", {})
        queries: list[str] = []
        if isinstance(plan, dict):
            raw_queries = plan.get("knowledge_queries", [])
            if isinstance(raw_queries, str):
                queries = [raw_queries]
            elif isinstance(raw_queries, list):
                queries = [str(q) for q in raw_queries if q]
        if not queries:
            queries = [context.topic]

        kb_agent = self._select_primary_kb()
        if not kb_agent:
            logger.warning("Swarm Collecting: no knowledge agent available, skipping KB queries.")
            context.data["gathered_intel"] = {}
            return

        logger.info("Swarm Collecting: using KB agent %s", type(kb_agent).__name__)

        # Example: concurrent knowledge gathering
        results = {}
        for i, q in enumerate(queries[:3]):
            results[f"kb_{i}"] = await kb_agent.run_task(f"search:{q}")

        context.data["gathered_intel"] = results

        status = self._status_from_context(context)
        status.nodes.append(AgentNode(id="collector", label="Collectors", status="complete"))
        self._write_status(status)

    def _select_primary_kb(self) -> Optional[BaseAgent]:
        """Pick a knowledge-capable agent for swarm collection."""
        def supports_run_task(agent: Optional[BaseAgent]) -> bool:
            if agent is None:
                return False
            return type(agent).run_task is not BaseAgent.run_task

        preferred = [
            "primary_kb",
            "UnifiedALTTPKnowledge",
            "OracleOfSecretsKB",
            "ALTTPKnowledgeBase",
            "GigaleakKB",
            "OracleKnowledgeBase",
            "RomHackingSpecialist",
            "KnowledgeGraphAgent",
            "ContextVectorAgent",
        ]
        for name in preferred:
            agent = self.agents_map.get(name)
            if supports_run_task(agent):
                return agent

        for name, agent in self.agents_map.items():
            if ("Knowledge" in name or name.endswith("KB")) and supports_run_task(agent):
                return agent
        return None

    async def _step_verify(self, context: PipelineContext):
        """Phase 3: Verification / Critique."""
        logger.info("Swarm Verifying...")
        reviewer = self.agents_map.get("reviewer")
        if not reviewer:
            return

        raw_data = json.dumps(context.data.get("gathered_intel", {}))
        critique = await reviewer.run_task(raw_data)
        context.data["critique"] = critique

        status = self._status_from_context(context)
        status.nodes.append(AgentNode(id="reviewer", label="Reviewer", status="complete"))
        self._write_status(status)

    async def _step_summarize(self, context: PipelineContext):
        """Phase 4: Synthesis."""
        logger.info("Swarm Synthesizing...")
        documenter = self.agents_map.get("documenter")
        if not documenter:
            return

        full_context = json.dumps({
            "topic": context.topic,
            "data": context.data.get("gathered_intel", {}),
            "critique": context.data.get("critique", "")
        })
        report = await documenter.run_task(full_context)
        context.data["final_report"] = report

        status = self._status_from_context(context)
        status.nodes.append(AgentNode(id="documenter", label="Documenter", status="complete"))
        self._write_status(status)

    async def run_session(self, focus_topic: str = "General Research") -> Dict[str, Any]:
        """Runs the full swarm pipeline."""
        context = PipelineContext(topic=focus_topic)

        pipeline = OrchestrationPipeline([
            PipelineStep(name="plan", kind="collect", run=self._step_plan),
            PipelineStep(name="collect", kind="collect", run=self._step_collect),
            PipelineStep(name="verify", kind="verify", run=self._step_verify),
            PipelineStep(name="summarize", kind="summarize", run=self._step_summarize),
        ])

        result = await pipeline.run(context)
        return {
            "topic": focus_topic,
            "report": context.data.get("final_report"),
            "critique": context.data.get("critique"),
            "status": "success" if result.is_success else "failed"
        }


async def main():
    """Main entry point for testing swarm locally."""
    # Mock agents for testing
    from agents.core.base import BaseAgent

    class MockAgent(BaseAgent):
        async def run_task(self, task):
            return f"Processed: {task}"

    agents = {
        "strategist": SwarmStrategist(),
        "reviewer": CouncilReviewer(),
        "documenter": DeepDiveDocumenter(),
        "primary_kb": MockAgent("KB", "Knowledge Base")
    }

    council = SwarmCouncil(agents)
    await council.setup()
    result = await council.run_session("Refactoring Python Import Systems")
    print(f"Swarm Report:\n{result['report']}")


if __name__ == "__main__":
    asyncio.run(main())
