"""Autonomy agents for background improvement and safety loops."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import re

from hafs.agents.base import BaseAgent
from hafs.core.execution import ExecutionPolicy
from hafs.core.history import AgentMemoryManager, HistoryLogger, OperationType
from hafs.core.projects import ProjectRegistry
from hafs.core.services import ServiceManager, ServiceState
from hafs.core.tooling import ToolRunner


@dataclass
class LoopReport:
    """Structured report output from an autonomy loop."""

    title: str
    body: str
    tags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class MemoryAwareAgent(BaseAgent):
    """Base agent that writes summaries into AgentMemory."""

    def __init__(self, name: str, role_description: str):
        super().__init__(name, role_description)
        self._memory_manager: Optional[AgentMemoryManager] = None

    def _get_memory(self):
        if self._memory_manager is None:
            self._memory_manager = AgentMemoryManager(self.context_root)
        return self._memory_manager.get_agent_memory(self.name)

    async def remember(
        self,
        content: str,
        memory_type: str,
        context: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> None:
        try:
            memory = self._get_memory()
            await memory.remember(
                content=content,
                memory_type=memory_type,
                context=context or {},
                importance=importance,
            )
        except Exception:
            return


class SelfImprovementAgent(MemoryAwareAgent):
    """Identify friction and suggest system improvements."""

    def __init__(self, history_dir: Optional[Path] = None):
        super().__init__("SelfImprovement", "Identify recurring friction and propose improvements.")
        self.history_dir = history_dir or (self.context_root / "history")
        self.model_tier = "fast"

    def _summarize_failures(self, entries) -> tuple[str, dict[str, Any]]:
        failures = [
            entry
            for entry in entries
            if entry.operation.type == OperationType.TOOL_CALL and not entry.operation.success
        ]
        tool_counts = Counter(entry.operation.name for entry in failures)
        error_counts = Counter((entry.operation.error or "unknown").strip() for entry in failures)
        files_touched = Counter(
            file
            for entry in failures
            for file in entry.metadata.files_touched
        )

        summary_lines = []
        if failures:
            summary_lines.append(f"Total tool failures: {len(failures)}")
            if tool_counts:
                summary_lines.append("Top failing tools:")
                for tool, count in tool_counts.most_common(5):
                    summary_lines.append(f"- {tool}: {count}")
            if error_counts:
                summary_lines.append("Top error messages:")
                for err, count in error_counts.most_common(5):
                    summary_lines.append(f"- {err} ({count})")
            if files_touched:
                summary_lines.append("Most affected files:")
                for file, count in files_touched.most_common(5):
                    summary_lines.append(f"- {file} ({count})")
        else:
            summary_lines.append("No recent tool failures detected.")

        metrics = {
            "total_failures": len(failures),
            "top_tools": dict(tool_counts.most_common(5)),
            "top_errors": dict(error_counts.most_common(5)),
            "top_files": dict(files_touched.most_common(5)),
        }
        return "\n".join(summary_lines), metrics

    async def run_task(self) -> LoopReport:
        logger = HistoryLogger(self.history_dir)
        entries = logger.get_recent_entries(limit=200)

        summary, metrics = self._summarize_failures(entries)

        recommendations = ""
        if metrics.get("total_failures", 0) > 0:
            prompt = (
                "You are the Self Improvement loop.\n"
                "Review the failure summary and propose up to 5 concrete improvements.\n"
                "Focus on tooling reliability, error handling, or missing automation.\n\n"
                f"FAILURE SUMMARY:\n{summary}"
            )
            recommendations = await self.generate_thought(prompt)
        else:
            # Even with no failures, provide general system health recommendations
            total_entries = len(entries)
            tool_calls = [e for e in entries if e.operation.type == OperationType.TOOL_CALL]
            success_rate = (
                sum(1 for e in tool_calls if e.operation.success) / len(tool_calls) * 100
                if tool_calls else 100
            )
            prompt = (
                "You are the Self Improvement loop reviewing a healthy system.\n"
                f"Recent activity: {total_entries} operations, {len(tool_calls)} tool calls, "
                f"{success_rate:.1f}% success rate.\n\n"
                "Propose 2-3 proactive improvements to enhance system capabilities:\n"
                "- Consider adding new automation or monitoring\n"
                "- Suggest knowledge base enhancements\n"
                "- Recommend efficiency optimizations\n"
            )
            recommendations = await self.generate_thought(prompt)

        if not recommendations or recommendations.startswith("Error in generate_thought"):
            recommendations = "System is operating normally. No specific recommendations at this time."

        body = f"## Failure Summary\n{summary}\n\n## Recommendations\n{recommendations}"
        await self.remember(
            content=summary,
            memory_type="learning" if metrics.get("total_failures", 0) else "insight",
            context=metrics,
            importance=0.6 if metrics.get("total_failures", 0) else 0.4,
        )

        return LoopReport(
            title="Self Improvement Review",
            body=body,
            tags=["self_improvement", "quality"],
            metrics=metrics,
        )


class CuriosityExplorerAgent(MemoryAwareAgent):
    """Generate curiosity-driven exploration prompts."""

    def __init__(self, workspace_path: Optional[str] = None, execution_mode: Optional[str] = None):
        super().__init__("CuriosityExplorer", "Identify curiosity-driven exploration topics.")
        self.workspace_path = (
            Path(workspace_path).expanduser()
            if workspace_path
            else Path.cwd()
        )
        self.execution_mode = execution_mode
        self._tool_runner: Optional[ToolRunner] = None
        self.model_tier = "fast"

    def _build_tool_runner(self) -> ToolRunner:
        registry = ProjectRegistry.load()
        project = registry.match_path(self.workspace_path)
        policy = ExecutionPolicy(registry=registry, execution_mode=self.execution_mode)
        tool_profile = policy.resolve_tool_profile(project)
        return ToolRunner(self.workspace_path, tool_profile)

    def _ensure_tool_runner(self) -> ToolRunner:
        if not self._tool_runner:
            self._tool_runner = self._build_tool_runner()
        return self._tool_runner

    async def _safe_run(self, tool_name: str, args: Optional[list[str]] = None) -> str:
        runner = self._ensure_tool_runner()
        try:
            result = await runner.run(tool_name, args=args or [])
            if result.ok and result.stdout:
                return result.stdout
        except Exception:
            pass
        return ""

    async def run_task(self) -> LoopReport:
        git_log = await self._safe_run("git_log", ["--oneline", "-n", "20"])
        todos = await self._safe_run("rg_todos")

        fallback_lines = []
        if git_log:
            fallback_lines.append("Recent commits:\n" + git_log[:2000])
        if todos:
            fallback_lines.append("TODO markers:\n" + todos[:2000])
        fallback_context = "\n\n".join(fallback_lines).strip() or "No recent signals found."

        prompt = (
            "You are the Curiosity Explorer loop.\n"
            "Based on the signals below, propose 3 exploration topics.\n"
            "Each topic should include: question, suggested file/module, and why it's interesting.\n\n"
            f"SIGNALS:\n{fallback_context}"
        )
        exploration = await self.generate_thought(prompt)
        if not exploration or exploration.startswith("Error in generate_thought"):
            exploration = fallback_context

        await self.remember(
            content=exploration[:500],
            memory_type="insight",
            context={"signals": fallback_context[:2000]},
            importance=0.4,
        )

        return LoopReport(
            title="Curiosity Exploration Ideas",
            body=exploration,
            tags=["curiosity", "exploration"],
        )


class SelfHealingAgent(MemoryAwareAgent):
    """Detect crashed services and attempt repairs."""

    def __init__(self, auto_restart: bool = True):
        super().__init__("SelfHealing", "Detect service crashes and attempt safe fixes.")
        self.auto_restart = auto_restart
        self.model_tier = "fast"

    def _scan_logs(self, max_files: int = 5) -> list[dict[str, str]]:
        log_dir = self.context_root / "logs"
        findings: list[dict[str, str]] = []
        if not log_dir.exists():
            return findings

        for log_file in sorted(log_dir.glob("*.log"))[:max_files]:
            try:
                lines = log_file.read_text(errors="ignore").splitlines()[-200:]
            except Exception:
                continue
            for line in reversed(lines):
                if "Traceback" in line or "ERROR" in line or "Exception" in line:
                    findings.append({"file": log_file.name, "line": line.strip()})
                    break
        return findings

    async def run_task(self) -> LoopReport:
        issues: list[str] = []
        actions: list[str] = []
        statuses = {}
        running_count = 0
        not_installed_count = 0

        try:
            manager = ServiceManager()
            statuses = await manager.status_all()
        except Exception as exc:
            issues.append(f"Service manager unavailable: {exc}")

        for name, status in statuses.items():
            # Check if service is installed (has a plist/unit file)
            if not status.enabled:
                not_installed_count += 1
                continue  # Skip uninstalled services - they're not "failed"

            if status.state == ServiceState.RUNNING:
                running_count += 1
            elif status.state == ServiceState.FAILED:
                issues.append(f"{name} has failed (exit code: {status.last_exit_code})")
                if self.auto_restart:
                    success = await manager.restart(name)
                    actions.append(f"{'Restarted' if success else 'Failed to restart'} {name}")
                else:
                    actions.append(f"Suggested restart for {name}")
            elif status.state == ServiceState.STOPPED:
                issues.append(f"{name} is stopped but installed")
                if self.auto_restart:
                    success = await manager.start(name)
                    actions.append(f"{'Started' if success else 'Failed to start'} {name}")
                else:
                    actions.append(f"Suggested start for {name}")

        log_findings = self._scan_logs()
        if log_findings:
            issues.append("Recent errors found in logs:")
            for finding in log_findings:
                issues.append(f"- {finding['file']}: {finding['line']}")

        # Build status summary
        body = "## Service Status\n"
        body += f"- Running: {running_count}\n"
        body += f"- Not installed: {not_installed_count}\n"
        if issues:
            body += "\n## Issues\n"
            body += "\n".join(f"- {issue}" for issue in issues)
        else:
            body += "\nAll installed services are healthy."
        if actions:
            body += "\n\n## Actions\n" + "\n".join(f"- {action}" for action in actions)

        await self.remember(
            content="; ".join(issues)[:500] if issues else "All services healthy",
            memory_type="error" if actions else "insight",
            context={"actions": actions, "log_findings": log_findings, "running": running_count},
            importance=0.6 if actions else 0.3,
        )

        return LoopReport(
            title="Self Healing Check",
            body=body,
            tags=["self_healing", "stability"],
            metrics={
                "running": running_count,
                "not_installed": not_installed_count,
                "issues": len(issues),
                "actions": len(actions),
            },
        )


class HallucinationWatcherAgent(MemoryAwareAgent):
    """Flag potential hallucinations by comparing claims to tool history."""

    def __init__(self, history_dir: Optional[Path] = None):
        super().__init__("HallucinationWatcher", "Detect responses that may lack tool evidence.")
        self.history_dir = history_dir or (self.context_root / "history")
        self.model_tier = "fast"

    def _extract_message(self, entry) -> str:
        if entry.operation.output:
            return str(entry.operation.output)
        message = entry.operation.input.get("message")
        return str(message) if message else ""

    async def run_task(self) -> LoopReport:
        logger = HistoryLogger(self.history_dir)
        entries = logger.get_recent_entries(limit=200)

        tool_calls = [e for e in entries if e.operation.type == OperationType.TOOL_CALL]
        tool_files = {
            file
            for entry in tool_calls
            for file in entry.metadata.files_touched
        }
        tool_recent = len(tool_calls) > 0

        suspicious: list[str] = []
        claim_patterns = [
            "I ran",
            "I executed",
            "Output:",
            "stdout",
            "stderr",
            "Command:",
            "Result:",
        ]

        for entry in entries:
            if entry.operation.type != OperationType.AGENT_MESSAGE:
                continue
            message = self._extract_message(entry)
            if not message:
                continue

            has_claim = any(pattern in message for pattern in claim_patterns)
            file_mentions = set(re.findall(r"[A-Za-z0-9_./-]+\\.[A-Za-z0-9]{1,6}", message))

            if has_claim and not tool_recent:
                snippet = message.replace("\n", " ")[:200]
                suspicious.append(f"Claim without recent tool calls: {snippet}...")
            elif file_mentions and tool_files.isdisjoint(file_mentions):
                snippet = ", ".join(sorted(list(file_mentions))[:5])
                suspicious.append(f"Referenced files without tool evidence: {snippet}")

        if not suspicious:
            body = "No high-risk hallucination signals detected."
        else:
            body = "## Potential Issues\n" + "\n".join(f"- {item}" for item in suspicious)

        await self.remember(
            content="; ".join(suspicious)[:500] if suspicious else "No hallucination risks detected.",
            memory_type="error" if suspicious else "insight",
            context={"count": len(suspicious)},
            importance=0.7 if suspicious else 0.3,
        )

        return LoopReport(
            title="Hallucination Watch",
            body=body,
            tags=["hallucination", "safety"],
            metrics={"issues": len(suspicious)},
        )
