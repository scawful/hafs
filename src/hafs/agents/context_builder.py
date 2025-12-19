"""Autonomous Context Agent.

A background agent that autonomously discovers work items, explores the codebase,
and synthesizes documentation. Adapters are loaded from the plugin registry.
"""
from hafs.core.orchestrator import ModelOrchestrator
from hafs.core.registry import agent_registry
from hafs.core.config import hafs_config, CONTEXT_ROOT
from hafs.core.projects import ProjectRegistry, Project
from hafs.core.tooling import ToolRunner, DEFAULT_TOOL_CATALOG
from typing import List, Dict, Any, Optional
import json
import logging
import os
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Generic adapter interface for fallback
class GenericAdapter:
    """Placeholder adapter when no plugin provides a real implementation."""
    async def connect(self): pass
    async def disconnect(self): pass
    async def search_bugs(self, query: str) -> List: return []
    async def get_reviews(self) -> List: return []
    async def search(self, term: str, limit: int = 5) -> List: return []
    async def read_file(self, path: str) -> str: return ""

class AutonomousContextAgent:
    """Agent that builds, explores, and verifies team context.

    Adapters are loaded from the plugin registry at setup time.
    Register adapters with these names:
    - issue_tracker: For bug/issue tracking
    - code_review: For code review systems
    - code_search: For codebase search/indexing
    """

    # Meta-Knowledge about the user and system
    SYSTEM_CONTEXT = """
    You are an autonomous background agent working for the user.
    Your job is to explore their work environment and maintain context documentation.
    """

    MANIFEST_FILES = [
        "README.md",
        "README.rst",
        "README.txt",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "requirements.txt",
        "Makefile",
        "CMakeLists.txt",
    ]

    def __init__(self, interval_minutes: int = 60):
        self.interval = interval_minutes * 60

        # Paths from config
        self.context_root = CONTEXT_ROOT
        self.knowledge_dir = self.context_root / "knowledge"
        self.agent_docs_dir = self.context_root / "background_agent" / "reports"
        self.agent_docs_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir = self.knowledge_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        # Adapters (will be loaded from registry in setup)
        self.issue_tracker = None
        self.code_review = None
        self.code_search = None

        # State
        self.inventory: Dict[str, Any] = {}
        self.code_context: Dict[str, str] = {}
        self.doc_context: Dict[str, str] = {}
        self.project_inventory: Dict[str, Any] = {}
        self.project_search_results: Dict[str, Any] = {}
        self.orchestrator = None
        self.project_registry = ProjectRegistry.load()

    async def setup(self):
        """Initialize adapters from registry and models."""
        # Load adapters from registry with fallback to stubs
        try:
            self.issue_tracker = agent_registry.get_adapter("issue_tracker")
            await self.issue_tracker.connect()
            logger.info("issue_tracker adapter loaded from registry")
        except Exception as e:
            logger.debug(f"issue_tracker not available: {e}")
            self.issue_tracker = GenericAdapter()

        try:
            self.code_review = agent_registry.get_adapter("code_review")
            await self.code_review.connect()
            logger.info("code_review adapter loaded from registry")
        except Exception as e:
            logger.debug(f"code_review not available: {e}")
            self.code_review = GenericAdapter()

        try:
            self.code_search = agent_registry.get_adapter("code_search")
            await self.code_search.connect()
            logger.info("code_search adapter loaded from registry")
        except Exception as e:
            logger.debug(f"code_search not available: {e}")
            self.code_search = GenericAdapter()

        # Initialize model orchestrator
        api_key = os.getenv("AISTUDIO_API_KEY")
        self.orchestrator = ModelOrchestrator(api_key)

        if api_key:
            print("Model Orchestrator initialized.")
        else:
            print("Model Orchestrator initialized (CLI Only).")

        self.project_registry = ProjectRegistry.load()

    @staticmethod
    def _trim_output(text: str, limit: int = 4000) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...(truncated)"

    def _project_output_dir(self, project: Project) -> Path:
        path = self.projects_dir / project.slug
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _read_manifest(self, project: Project) -> dict[str, str]:
        manifests: dict[str, str] = {}
        for filename in self.MANIFEST_FILES:
            file_path = project.path / filename
            if file_path.exists() and file_path.is_file():
                try:
                    manifests[filename] = self._trim_output(
                        file_path.read_text(errors="replace"),
                        limit=8000,
                    )
                except Exception:
                    continue
        return manifests

    async def _collect_project_inventory(self, project: Project) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": project.name,
            "path": str(project.path),
            "kind": project.kind,
            "tags": project.tags,
            "description": project.description,
            "timestamp": datetime.now().isoformat(),
            "missing": False,
            "tools": [],
            "git": {},
            "manifests": {},
            "notes": [],
        }

        if not project.path.exists():
            data["missing"] = True
            data["notes"].append("Project path not found.")
            return data

        tool_profile = self.project_registry.resolve_tool_profile(project)
        runner = ToolRunner(project.path, tool_profile, DEFAULT_TOOL_CATALOG)
        data["tools"] = runner.available_tools()

        if (project.path / ".git").exists():
            if "git_status" in data["tools"]:
                result = await runner.run("git_status")
                data["git"]["status"] = self._trim_output(result.stdout)
            if "git_branch" in data["tools"]:
                result = await runner.run("git_branch")
                data["git"]["branch"] = self._trim_output(result.stdout.strip())
            if "git_log" in data["tools"]:
                result = await runner.run("git_log")
                data["git"]["last_commit"] = self._trim_output(result.stdout.strip())

        if "rg_todos" in data["tools"]:
            result = await runner.run("rg_todos")
            data["todos"] = self._trim_output(result.stdout)

        data["manifests"] = self._read_manifest(project)
        return data

    async def gather_project_inventory(self) -> int:
        """Collect per-project snapshots for tracked repositories."""
        projects = self.project_registry.list()
        if not projects:
            return 0

        count = 0
        for project in projects:
            snapshot = await self._collect_project_inventory(project)
            self.project_inventory[project.name] = snapshot

            output_dir = self._project_output_dir(project)
            output_path = output_dir / "inventory.json"
            output_path.write_text(json.dumps(snapshot, indent=2))
            count += 1

        return count

    async def search_projects(self, focus_terms: list[str]) -> None:
        """Search tracked projects using rg for focus terms."""
        if not focus_terms:
            return

        for project in self.project_registry.list():
            if not project.path.exists():
                continue
            tool_profile = self.project_registry.resolve_tool_profile(project)
            runner = ToolRunner(project.path, tool_profile, DEFAULT_TOOL_CATALOG)
            if "rg" not in runner.available_tools():
                continue

            project_results: dict[str, Any] = {}
            for term in focus_terms[:8]:
                result = await runner.run(
                    "rg",
                    args=["--max-count", "25", "--context", "1", term],
                )
                project_results[term] = {
                    "stdout": self._trim_output(result.stdout),
                    "stderr": self._trim_output(result.stderr),
                    "exit_code": result.exit_code,
                }

            self.project_search_results[project.name] = project_results
            output_dir = self._project_output_dir(project)
            output_path = output_dir / "search_terms.json"
            output_path.write_text(json.dumps(project_results, indent=2))

    async def gather_inventory(self):
        """Gather basic work inventory from adapters."""
        print(f"[{datetime.now()}] Gathering inventory...")

        # Fetch from adapters
        my_bugs = []
        my_reviews = []

        try:
            my_bugs = await self.issue_tracker.search_bugs("assignee:me status:open")
        except Exception as e:
            logger.warning(f"Failed to fetch issues: {e}")

        try:
            my_reviews = await self.code_review.get_reviews()
        except Exception as e:
            logger.warning(f"Failed to fetch reviews: {e}")

        self.inventory = {
            "timestamp": datetime.now().isoformat(),
            "bugs": [
                {"id": getattr(b, 'id', 'unknown'), "title": getattr(b, 'title', 'Untitled'), "priority": getattr(b, 'priority', 'unknown')}
                for b in my_bugs
            ] if my_bugs else [],
            "reviews": [
                {"id": getattr(r, 'id', 'unknown'), "title": getattr(r, 'title', 'Untitled'), "status": getattr(r, 'status', 'unknown')}
                for r in my_reviews
            ] if my_reviews else []
        }
        return len(self.inventory.get('bugs', [])) + len(self.inventory.get('reviews', []))

    async def explore_environment(self, focus_terms: Optional[List[str]] = None):
        """Explore Codebase and AgentWorkspaces for context."""
        print("  - Exploring environment...")

        if not focus_terms:
            self.code_context = {}
            self.doc_context = {}

        # Search codebase if adapter available
        if focus_terms and self.code_search:
            for term in focus_terms[:5]:
                try:
                    results = await self.code_search.search(term, limit=3)
                    for res in results:
                        file_path = getattr(res, 'file', None)
                        if file_path:
                            try:
                                content = await self.code_search.read_file(file_path)
                                if len(content) > 5000:
                                    content = content[:5000] + "\n...(truncated)"
                                self.code_context[file_path] = content
                            except Exception:
                                pass
                except Exception as e:
                    logger.debug(f"Search failed for '{term}': {e}")

        # Explore AgentWorkspaces/Docs
        agent_ws_root = hafs_config.agent_workspaces_dir
        if agent_ws_root.exists():
            for item in agent_ws_root.glob("**/*.md"):
                try:
                    if item.stat().st_size < 10000:
                        content = item.read_text()
                        if focus_terms:
                            if any(t.lower() in content.lower() for t in focus_terms):
                                self.doc_context[item.name] = content
                        else:
                            self.doc_context[item.name] = content
                except Exception as e:
                    logger.debug(f"Failed to read {item}: {e}")

        if focus_terms:
            await self.search_projects(focus_terms)

    async def run_cycle(self):
        """Run one full autonomous update cycle."""
        count = await self.gather_inventory()
        project_count = await self.gather_project_inventory()
        await self.explore_environment()
        print(f"  - Processed {count} items across adapters")
        print(f"  - Updated {project_count} project snapshots")

    async def run_directed(self, task: str):
        """Run a directed research task based on user input."""
        print(f"[{datetime.now()}] Starting directed task: '{task}'")
        await self.gather_inventory()
        await self.gather_project_inventory()
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
                logger.error(f"Cycle error: {e}")
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
