"""Flag potential hallucinations by comparing claims to tool history."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from hafs.core.history import HistoryLogger, OperationType


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
