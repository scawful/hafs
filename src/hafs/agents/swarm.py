"""Swarm Controller / Council of Agents (Public).

Orchestrates multiple agents to perform deep research and synthesis.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from hafs.agents.base import BaseAgent
from hafs.core.history import HistoryLogger, SessionManager
from hafs.core.orchestration import OrchestrationPipeline, PipelineContext, PipelineStep

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
    def __init__(self, instantiated_agents: Dict[str, BaseAgent]):
        """
        Initializes the SwarmCouncil with a dictionary of instantiated agents.
        """
        self.agents_map = instantiated_agents
        self.agents_list = list(self.agents_map.values())
        self.scale = "MEDIUM"
        
        # Primary roles
        self.strategist = self.agents_map.get("SwarmStrategist")
        self.reviewer = self.agents_map.get("CouncilReviewer")
        self.documenter = self.agents_map.get("DeepDiveDocumenter")
        self.visualizer = self.agents_map.get("VisualizerAgent")

        self._history_logger: Optional[HistoryLogger] = None
        self._session_manager: Optional[SessionManager] = None

    def attach_history(self, context_root: Optional[Path] = None) -> None:
        """Attach history logging to swarm sessions."""
        context_root = context_root or Path.home() / ".context"
        history_dir = context_root / "history"
        sessions_dir = history_dir / "sessions"
        project_id = Path.cwd().name

        session_manager = SessionManager(sessions_dir, project_id=project_id)
        history_logger = HistoryLogger(
            history_dir=history_dir,
            session_manager=session_manager,
            project_id=project_id,
        )
        session_manager.set_history_logger(history_logger)
        session_manager.create()

        self._history_logger = history_logger
        self._session_manager = session_manager

    def _write_status(self, status: SwarmStatus):
        try:
            STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o), indent=2)
        except: pass

    async def setup(self):
        print("--- Convening the Council ---")
        await asyncio.gather(*[a.setup() for a in self.agents_list])

    async def _run_parallel_tasks(self, task_map: Dict[str, Any]) -> Dict[str, Any]:
        """Run tasks in parallel and return a dict of results."""
        keys = list(task_map.keys())
        tasks = list(task_map.values())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for k, res in zip(keys, results):
            if isinstance(res, Exception):
                print(f"[Swarm] Task {k} failed: {res}")
                output[k] = {"raw": [], "summary": f"Task failed: {res}"}
            else:
                output[k] = res
        return output

    def _status_from_context(self, context: PipelineContext) -> Optional[SwarmStatus]:
        status = context.metadata.get("status")
        if isinstance(status, SwarmStatus):
            return status
        return None

    async def _step_plan(self, context: PipelineContext) -> Any:
        print("Phase 0: Planning...")
        status = self._status_from_context(context)
        if not self.strategist:
            context.plan = {}
            return {}

        plan = await self.strategist.run_task(context.topic)
        context.plan = plan

        if self._history_logger:
            self._history_logger.log_agent_message("SwarmStrategist", str(plan))

        if status:
            status.nodes.append(AgentNode(id="strategist", label="Strategist", status="success"))
            self._write_status(status)

        return plan

    async def _step_collect(self, context: PipelineContext) -> Dict[str, Any]:
        print("Phase 1: Gathering Information...")
        status = self._status_from_context(context)
        task_map = {}
        for name, agent in self.agents_map.items():
            if name in ["SwarmStrategist", "CouncilReviewer", "DeepDiveDocumenter", "VisualizerAgent"]:
                continue
            try:
                task_map[name] = agent.run_task(context.topic)
            except Exception:
                continue

        results = await self._run_parallel_tasks(task_map)
        context.results = results

        if status:
            for agent_name in task_map.keys():
                node_status = "success" if results.get(agent_name) else "error"
                status.nodes.append(AgentNode(id=agent_name, label=agent_name, status=node_status))
                status.edges.append({"source": "strategist", "target": agent_name})
            self._write_status(status)

        return results

    async def _step_verify(self, context: PipelineContext) -> str:
        print("Phase 2: Deliberation...")
        status = self._status_from_context(context)
        critique = "No reviewer available."
        if self.reviewer:
            critique = await self.reviewer.run_task(str(context.results))

        context.critique = critique

        if self._history_logger:
            self._history_logger.log_agent_message("CouncilReviewer", critique)

        if status:
            status.nodes.append(AgentNode(id="reviewer", label="Reviewer", status="success"))
            self._write_status(status)

        return critique

    async def _step_summarize(self, context: PipelineContext) -> str:
        print("Phase 3: Synthesis...")
        status = self._status_from_context(context)
        if self.documenter:
            summary = await self.documenter.run_task(
                f"DATA:\n{str(context.results)}\n\nCRITIQUE:\n{context.critique}"
            )
        else:
            summary = f"Report:\n{str(context.results)}"

        context.summary = summary

        if self._history_logger:
            self._history_logger.log_agent_message("DeepDiveDocumenter", summary)

        if status:
            status.nodes.append(AgentNode(id="documenter", label="Documenter", status="success"))
            self._write_status(status)

        output_path = Path.home() / ".context" / "background_agent" / "reports"
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"{context.topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        filename.write_text(summary)
        context.artifacts["report_path"] = str(filename)

        return summary

    async def run_session(self, focus_topic: str = "General Research"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        print(f"[{ts}] Session Started: {focus_topic}")
        
        status = SwarmStatus()
        status.active_topic = focus_topic
        status.start_time = ts
        self._write_status(status)

        if self._history_logger:
            self._history_logger.log_system_event(
                "swarm_session_started",
                {"topic": focus_topic},
            )

        context = PipelineContext(topic=focus_topic, metadata={"status": status})
        pipeline = OrchestrationPipeline(
            [
                PipelineStep(name="plan", kind="plan", run=self._step_plan),
                PipelineStep(name="collect", kind="execute", run=self._step_collect),
                PipelineStep(name="verify", kind="verify", run=self._step_verify),
                PipelineStep(name="summarize", kind="summarize", run=self._step_summarize),
            ]
        )

        await pipeline.run(context)

        final_doc = context.summary or f"Report:\n{str(context.results)}"
        report_path = context.artifacts.get("report_path")
        if report_path:
            print(f"Session Complete: {report_path}")
        else:
            print("Session Complete.")

        if self._session_manager:
            self._session_manager.complete()

        return final_doc

async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str)
    args = parser.parse_args()

    from hafs.core.orchestration_entrypoint import run_orchestration

    topic = args.topic or "HAFS Status"
    await run_orchestration(mode="swarm", topic=topic)

if __name__ == "__main__":
    asyncio.run(main())
