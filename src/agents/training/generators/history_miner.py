"""History Pattern Miner.

Extracts training samples from HistoryLogger and AgentMemory,
mining successful workflows, session patterns, and operation sequences.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class WorkflowSourceItem(SourceItem):
    """Source item representing a workflow or operation sequence."""

    operations: list[dict[str, Any]] = field(default_factory=list)
    session_id: str = ""
    agent_name: str = ""
    outcome: str = "success"  # success, partial, failed
    duration_seconds: float = 0.0
    files_touched: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    @property
    def item_id(self) -> str:
        return f"{self.session_id}:{self.name}"


@dataclass
class SessionPattern:
    """Pattern extracted from a session."""

    pattern_type: str  # workflow, debugging, research, implementation
    tool_sequence: list[str]
    success_rate: float
    avg_duration: float
    examples: list[str]


class HistoryMiner(DataGenerator):
    """Mine training samples from operation history.

    Extracts:
    - Successful tool call sequences → workflow templates
    - Session summaries → contextual reasoning examples
    - Failed → recovered sequences → troubleshooting guides
    - Agent memory patterns → decision-making samples
    """

    # Minimum operations for a workflow to be interesting
    MIN_WORKFLOW_LENGTH = 3

    # Operation types that indicate workflow boundaries
    BOUNDARY_OPERATIONS = ["session_start", "session_end", "task_complete"]

    def __init__(
        self,
        lookback_days: int = 7,
        min_success_rate: float = 0.7,
    ):
        super().__init__(
            name="HistoryMiner",
            domain="workflows",
            teacher_tier="reasoning",
        )
        self.lookback_days = lookback_days
        self.min_success_rate = min_success_rate
        self._orchestrator = None

        # Paths
        self.history_dir = self.context_root / "history"
        self.memory_dir = self.context_root / "memory"
        self.agents_dir = self.history_dir / "agents"

    async def setup(self):
        await super().setup()

        from core.orchestrator_v2 import UnifiedOrchestrator
        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[WorkflowSourceItem]:
        """Extract workflow patterns from history."""
        items: list[WorkflowSourceItem] = []
        cutoff = datetime.now() - timedelta(days=self.lookback_days)

        # 1. Extract from daily JSONL logs
        items.extend(await self._extract_from_logs(cutoff))

        # 2. Extract from agent-specific history
        items.extend(await self._extract_from_agent_history(cutoff))

        # 3. Extract from session summaries
        items.extend(await self._extract_from_sessions(cutoff))

        # Filter by success rate
        items = [i for i in items if self._calculate_success_rate(i) >= self.min_success_rate]

        logger.info(f"Extracted {len(items)} workflow patterns")
        return items

    async def _extract_from_logs(
        self, cutoff: datetime
    ) -> list[WorkflowSourceItem]:
        """Extract workflows from daily JSONL history logs."""
        items = []

        if not self.history_dir.exists():
            return items

        # Group operations into workflows
        workflows: dict[str, list[dict]] = defaultdict(list)

        for jsonl_file in sorted(self.history_dir.glob("*.jsonl"), reverse=True):
            # Check file date from name (YYYY-MM-DD.jsonl)
            try:
                file_date = datetime.strptime(jsonl_file.stem, "%Y-%m-%d")
                if file_date < cutoff:
                    continue
            except ValueError:
                continue

            try:
                with open(jsonl_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            session_id = entry.get("provenance", {}).get("session", "default")
                            workflows[session_id].append(entry)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.debug(f"Error reading {jsonl_file}: {e}")

        # Convert workflows to source items
        for session_id, ops in workflows.items():
            if len(ops) < self.MIN_WORKFLOW_LENGTH:
                continue

            # Extract metadata
            tools_used = list(set(
                op.get("metadata", {}).get("tool", "")
                for op in ops
                if op.get("metadata", {}).get("tool")
            ))

            files_touched = []
            for op in ops:
                files_touched.extend(
                    op.get("metadata", {}).get("files", [])
                )
            files_touched = list(set(files_touched))[:20]  # Limit

            # Calculate success rate
            success_count = sum(
                1 for op in ops
                if op.get("metadata", {}).get("success", True)
            )
            success_rate = success_count / len(ops) if ops else 0

            # Calculate duration
            if ops:
                try:
                    start = datetime.fromisoformat(ops[0].get("timestamp", ""))
                    end = datetime.fromisoformat(ops[-1].get("timestamp", ""))
                    duration = (end - start).total_seconds()
                except (ValueError, TypeError):
                    duration = 0
            else:
                duration = 0

            # Determine workflow type
            workflow_type = self._classify_workflow(ops, tools_used)

            items.append(
                WorkflowSourceItem(
                    name=f"workflow_{workflow_type}_{session_id[:8]}",
                    content=self._summarize_workflow(ops),
                    source="history_logger",
                    operations=ops[:50],  # Limit operations
                    session_id=session_id,
                    outcome="success" if success_rate > 0.8 else "partial",
                    duration_seconds=duration,
                    files_touched=files_touched,
                    tools_used=tools_used,
                )
            )

        return items

    async def _extract_from_agent_history(
        self, cutoff: datetime
    ) -> list[WorkflowSourceItem]:
        """Extract patterns from per-agent history."""
        items = []

        if not self.agents_dir.exists():
            return items

        for agent_dir in self.agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_name = agent_dir.name

            # Look for session files
            for session_file in agent_dir.glob("*.json"):
                try:
                    data = json.loads(session_file.read_text())

                    # Check timestamp
                    ts = data.get("last_updated", "")
                    if ts:
                        try:
                            if datetime.fromisoformat(ts) < cutoff:
                                continue
                        except ValueError:
                            pass

                    # Extract operations
                    ops = data.get("operations", [])
                    if len(ops) < self.MIN_WORKFLOW_LENGTH:
                        continue

                    items.append(
                        WorkflowSourceItem(
                            name=f"agent_{agent_name}_{session_file.stem}",
                            content=data.get("summary", ""),
                            source=f"agent_history/{agent_name}",
                            operations=ops[:50],
                            session_id=session_file.stem,
                            agent_name=agent_name,
                            outcome=data.get("outcome", "success"),
                        )
                    )

                except Exception as e:
                    logger.debug(f"Error parsing agent history: {e}")

        return items

    async def _extract_from_sessions(
        self, cutoff: datetime
    ) -> list[WorkflowSourceItem]:
        """Extract from session summary files."""
        items = []
        summaries_dir = self.memory_dir / "summaries"

        if not summaries_dir.exists():
            return items

        for summary_file in summaries_dir.glob("*.json"):
            try:
                data = json.loads(summary_file.read_text())

                # Check date
                ts = data.get("timestamp", "")
                if ts:
                    try:
                        if datetime.fromisoformat(ts) < cutoff:
                            continue
                    except ValueError:
                        pass

                # Extract key information
                items.append(
                    WorkflowSourceItem(
                        name=f"session_{summary_file.stem}",
                        content=data.get("summary", ""),
                        source="session_summaries",
                        session_id=summary_file.stem,
                        outcome=data.get("outcome", "success"),
                        tools_used=data.get("tools_used", []),
                        files_touched=data.get("files_modified", []),
                    )
                )

            except Exception as e:
                logger.debug(f"Error parsing session summary: {e}")

        return items

    def _classify_workflow(
        self, ops: list[dict], tools: list[str]
    ) -> str:
        """Classify workflow type based on operations and tools."""
        # Check for patterns
        tool_set = set(tools)

        if "git" in " ".join(tools).lower():
            if any("commit" in str(op) for op in ops):
                return "git_commit"
            elif any("merge" in str(op) for op in ops):
                return "git_merge"
            return "version_control"

        if any(t in tool_set for t in ["pytest", "npm_test", "cargo_test"]):
            return "testing"

        if any(t in tool_set for t in ["rg", "grep", "find"]):
            return "search"

        if any("debug" in str(op).lower() for op in ops):
            return "debugging"

        if any("build" in str(op).lower() for op in ops):
            return "building"

        return "general"

    def _summarize_workflow(self, ops: list[dict]) -> str:
        """Create a text summary of a workflow."""
        lines = []

        for i, op in enumerate(ops[:10]):  # First 10 operations
            op_type = op.get("operation_type", "unknown")
            tool = op.get("metadata", {}).get("tool", "")
            success = "✓" if op.get("metadata", {}).get("success", True) else "✗"

            if tool:
                lines.append(f"{i+1}. [{success}] {op_type}: {tool}")
            else:
                lines.append(f"{i+1}. [{success}] {op_type}")

        if len(ops) > 10:
            lines.append(f"... and {len(ops) - 10} more operations")

        return "\n".join(lines)

    def _calculate_success_rate(self, item: WorkflowSourceItem) -> float:
        """Calculate success rate for a workflow."""
        if not item.operations:
            return 1.0 if item.outcome == "success" else 0.0

        success_count = sum(
            1 for op in item.operations
            if op.get("metadata", {}).get("success", True)
        )
        return success_count / len(item.operations)

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for workflow sample."""
        if not isinstance(item, WorkflowSourceItem):
            raise TypeError(f"Expected WorkflowSourceItem, got {type(item)}")

        tools_str = ", ".join(item.tools_used[:10]) if item.tools_used else "various tools"
        files_str = ", ".join(item.files_touched[:5]) if item.files_touched else "multiple files"

        template = get_prompt("agents.training.generators.history_miner.prompt", "")
        if not template:
            template = (
                "You are an expert at documenting software workflows and best practices.\n"
                "Analyze this workflow and create a training sample that teaches this pattern.\n\n"
                "WORKFLOW TYPE: {workflow_type}\n"
                "OUTCOME: {outcome}\n"
                "DURATION: {duration_seconds}\n"
                "TOOLS USED: {tools_used}\n"
                "FILES TOUCHED: {files_touched}\n\n"
                "WORKFLOW SUMMARY:\n{workflow_summary}\n\n"
                "OPERATIONS:\n{operations}\n\n"
                "Generate a JSON training sample with:\n"
                "1. \"instruction\": A user request that would trigger this workflow. Be specific about the goal.\n"
                "2. \"input\": Any context needed (project type, current state, constraints).\n"
                "3. \"output\": A step-by-step guide explaining how to accomplish this task, including:\n"
                "   - What tools to use and why\n"
                "   - The sequence of operations\n"
                "   - How to verify success at each step\n"
                "   - Common pitfalls to avoid\n\n"
                "The output should be educational and practical.\n\n"
                "JSON FORMAT:\n"
                "{{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}}\n"
            )

        workflow_type = item.name.split('_')[1] if '_' in item.name else 'general'

        return template.format(
            workflow_type=workflow_type,
            outcome=item.outcome,
            duration_seconds=f"{item.duration_seconds:.1f}s",
            tools_used=tools_str,
            files_touched=files_str,
            workflow_summary=item.content,
            operations=self._format_operations(item.operations),
        )

    def _format_operations(self, ops: list[dict]) -> str:
        """Format operations for prompt."""
        lines = []
        for i, op in enumerate(ops[:15]):
            op_type = op.get("operation_type", "unknown")
            tool = op.get("metadata", {}).get("tool", "")
            success = op.get("metadata", {}).get("success", True)

            line = f"{i+1}. {op_type}"
            if tool:
                line += f" ({tool})"
            if not success:
                error = op.get("metadata", {}).get("error", "")[:50]
                line += f" [FAILED: {error}]"

            lines.append(line)

        return "\n".join(lines)

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Generate workflow sample."""
        if not isinstance(item, WorkflowSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            response, model_name = await asyncio.wait_for(
                self.generate_with_rotation(prompt, tier="reasoning"),
                timeout=60.0,
            )
            if not response:
                return None

            # Extract JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{"):response.rfind("}") + 1]

            data = json.loads(response)

            # Higher quality for successful workflows
            quality = 0.8 if item.outcome == "success" else 0.6

            return TrainingSample(
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", ""),
                domain="workflows",
                source=item.source,
                teacher_model=model_name,
                teacher_prompt=prompt,
                kg_entities=item.tools_used[:5],
                quality_score=quality,
            )

        except asyncio.TimeoutError:
            logger.warning(f"Timeout generating for {item.name}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {item.name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to generate for {item.name}: {e}")
            return None


if __name__ == "__main__":
    async def main():
        miner = HistoryMiner(lookback_days=7)
        await miner.setup()

        items = await miner.extract_source_items()
        print(f"Found {len(items)} workflow patterns")

        if items:
            result = await miner.run_generation(
                limit=10,
                output_path=Path("test_workflow_train.jsonl"),
            )
            print(f"Generated {result.processed} samples")

    import asyncio
    asyncio.run(main())
