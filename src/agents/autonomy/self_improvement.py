"""Identify friction and suggest system improvements."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from core.history import HistoryLogger, OperationType


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
