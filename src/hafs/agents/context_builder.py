"""Autonomous Context Agent.

A background agent that autonomously discovers work items, explores the codebase,
and synthesizes documentation. This is the public stub version - adapters should
be provided via plugins.
"""
from hafs.core.orchestrator import ModelOrchestrator
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from datetime import datetime

# Generic adapter interface for public version
class GenericAdapter:
    """Placeholder adapter for external integrations.

    Plugins can register real adapters via the adapter registry.
    """
    async def connect(self): pass
    async def search_bugs(self, query: str) -> List: return []
    async def get_reviews(self) -> List: return []
    async def search(self, term: str, limit: int = 5) -> List: return []
    async def read_file(self, path: str) -> str: return ""

class AutonomousContextAgent:
    """Agent that builds, explores, and verifies team context.

    This is the public version with stub adapters. Install platform-specific
    adapter plugins to enable full functionality.
    """

    # Meta-Knowledge about the user and system
    SYSTEM_CONTEXT = """
    You are an autonomous background agent working for the user.
    Your job is to explore their work environment and maintain context documentation.
    """

    def __init__(self, interval_minutes: int = 60):
        self.interval = interval_minutes * 60

        # Paths
        self.context_root = Path.home() / ".context"
        self.knowledge_dir = self.context_root / "knowledge"
        self.agent_docs_dir = self.context_root / "background_agent" / "reports"
        self.agent_docs_dir.mkdir(parents=True, exist_ok=True)

        # Adapters (stubs - override via plugins)
        self.bugs = GenericAdapter()
        self.critique = GenericAdapter()
        self.codesearch = GenericAdapter()

        # State
        self.inventory: Dict[str, Any] = {}
        self.code_context: Dict[str, str] = {}
        self.doc_context: Dict[str, str] = {}
        self.orchestrator = None

    async def setup(self):
        """Initialize connections and models."""
        await self.bugs.connect()
        await self.critique.connect()
        await self.codesearch.connect()

        api_key = os.getenv("AISTUDIO_API_KEY")
        self.orchestrator = ModelOrchestrator(api_key)

        if api_key:
            print("Model Orchestrator initialized.")
        else:
            print("Model Orchestrator initialized (CLI Only).")

    async def gather_inventory(self):
        """Gather basic work inventory."""
        print(f"[{datetime.now()}] Gathering inventory...")

        # Fetch from adapters (will be empty with stubs)
        my_bugs = await self.bugs.search_bugs("assignee:me status:open")
        my_reviews = await self.critique.get_reviews()

        self.inventory = {
            "timestamp": datetime.now().isoformat(),
            "bugs": [{"id": b.id, "title": b.title, "priority": getattr(b, 'priority', 'unknown')} for b in my_bugs] if my_bugs else [],
            "reviews": [{"id": r.id, "title": r.title, "status": getattr(r, 'status', 'unknown')} for r in my_reviews] if my_reviews else []
        }
        return len(self.inventory.get('bugs', [])) + len(self.inventory.get('reviews', []))

    async def explore_environment(self, focus_terms: Optional[List[str]] = None):
        """Explore Codebase and AgentWorkspaces for context."""
        print("  - Exploring environment...")

        if not focus_terms:
            self.code_context = {}
            self.doc_context = {}

        # Explore AgentWorkspaces/Docs
        agent_ws_root = Path.home() / "AgentWorkspaces"
        if agent_ws_root.exists():
            for item in agent_ws_root.glob("**/*.md"):
                if item.stat().st_size < 10000:
                    content = item.read_text()
                    if focus_terms:
                        if any(t.lower() in content.lower() for t in focus_terms):
                            self.doc_context[item.name] = content
                    else:
                        self.doc_context[item.name] = content

    async def run_cycle(self):
        """Run one full autonomous update cycle."""
        count = await self.gather_inventory()
        await self.explore_environment()
        print(f"  - Processed {count} items")

    async def run_directed(self, task: str):
        """Run a directed research task based on user input."""
        print(f"[{datetime.now()}] Starting directed task: '{task}'")
        await self.gather_inventory()
        await self.explore_environment(focus_terms=task.split())

    async def run(self):
        """Main loop for background operation."""
        import asyncio
        print("Starting Autonomous Context Agent...")
        await self.setup()
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                print(f"Error: {e}")
            print(f"Sleeping {self.interval}s...")
            await asyncio.sleep(self.interval)

if __name__ == "__main__":
    import asyncio
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Context Agent")
    parser.add_argument("interval", type=int, nargs="?", default=60, help="Interval in minutes")
    parser.add_argument("--task", type=str, help="Run a specific directed task and exit")

    args = parser.parse_args()

    agent = AutonomousContextAgent(interval_minutes=args.interval)

    async def main_async(agent, args):
        await agent.setup()
        if args.task:
            await agent.run_directed(args.task)
        else:
            await agent.run()

    try:
        asyncio.run(main_async(agent, args))
    except KeyboardInterrupt:
        print("Stopping agent.")
