"""Swarm Controller / Council of Agents (Public).

Orchestrates multiple agents to perform deep research and synthesis.
"""

import asyncio
import re as regex
from datetime import datetime
from pathlib import Path
from typing import List, Any, Dict, Tuple, Optional
import json

from hafs.agents.base import BaseAgent
from hafs.core.plugin_loader import load_plugins, load_all_agents_from_package
from hafs.core.history import HistoryLogger, SessionManager
from hafs.core.registry import agent_registry

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
        
        # Phase 0: Planning
        print("Phase 0: Planning...")
        if not self.strategist:
             print("Error: Strategist agent missing. Skipping planning.")
             plan = {}
        else:
            plan = await self.strategist.run_task(focus_topic) # Simplified call

        if self._history_logger:
            self._history_logger.log_agent_message("SwarmStrategist", str(plan))

        status.nodes.append(AgentNode(id="strategist", label="Strategist", status="success"))
        self._write_status(status)
        
        # Phase 1: Dynamic Collection
        print("Phase 1: Gathering Information...")
        task_map = {}
        for name, agent in self.agents_map.items():
            if name in ["SwarmStrategist", "CouncilReviewer", "DeepDiveDocumenter", "VisualizerAgent"]:
                continue
            try:
                task_map[name] = agent.run_task(focus_topic)
            except: pass

        results = await self._run_parallel_tasks(task_map)
        
        for agent_name in task_map.keys():
            node_status = "success" if results.get(agent_name) else "error"
            status.nodes.append(AgentNode(id=agent_name, label=agent_name, status=node_status))
            status.edges.append({"source": "strategist", "target": agent_name})
        self._write_status(status)

        # Phase 2: Deliberation
        print("Phase 2: Deliberation...")
        critique = "No reviewer available."
        if self.reviewer:
            critique = await self.reviewer.run_task(str(results))

        if self._history_logger:
            self._history_logger.log_agent_message("CouncilReviewer", critique)
            
        status.nodes.append(AgentNode(id="reviewer", label="Reviewer", status="success"))
        self._write_status(status)
        
        # Phase 3: Synthesis
        print("Phase 3: Synthesis...")
        if self.documenter:
            final_doc = await self.documenter.run_task(f"DATA:\n{str(results)}\n\nCRITIQUE:\n{critique}")
        else:
            final_doc = f"Report:\n{str(results)}"

        if self._history_logger:
            self._history_logger.log_agent_message("DeepDiveDocumenter", final_doc)

        status.nodes.append(AgentNode(id="documenter", label="Documenter", status="success"))
        self._write_status(status)
        
        # Save
        output_path = Path.home() / ".context" / "background_agent" / "reports"
        output_path.mkdir(parents=True, exist_ok=True)
        filename = output_path / f"{focus_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        filename.write_text(final_doc)
        
        print(f"Session Complete: {filename}")

        if self._session_manager:
            self._session_manager.complete()

        return final_doc

async def main():
    import argparse
    import hafs.agents as agents_pkg
    
    # Discover
    load_plugins()
    load_all_agents_from_package(agents_pkg)
    
    # Instantiate
    instantiated_agents = {}
    for name, cls in agent_registry.list_agents().items():
        try:
            instantiated_agents[name] = cls()
        except TypeError:
             pass

    council = SwarmCouncil(instantiated_agents)
    await council.setup()
    council.attach_history()

    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str)
    args = parser.parse_args()
    
    if args.topic:
        await council.run_session(args.topic)
    else:
        await council.run_session("HAFS Status")

if __name__ == "__main__":
    asyncio.run(main())
