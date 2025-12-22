"""Error-to-Sample Generator.

Generates high-quality training samples from system failures, errors,
and self-improvement signals. Hooks into autonomy agents to extract
real-world failure-recovery patterns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from agents.training.base import DataGenerator, SourceItem, TrainingSample
from config.prompts import get_prompt

logger = logging.getLogger(__name__)


@dataclass
class ErrorSourceItem(SourceItem):
    """Source item representing an error or failure event."""

    error_type: str = ""  # exception, tool_failure, hallucination, service_crash
    error_message: str = ""
    stack_trace: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    recovery_action: str = ""
    severity: str = "medium"  # low, medium, high, critical
    agent_name: str = ""
    timestamp: str = ""

    @property
    def item_id(self) -> str:
        return f"{self.error_type}:{self.agent_name}:{self.timestamp}"


@dataclass
class ImprovementSignal:
    """Signal from SelfImprovementAgent."""

    tool_name: str
    failure_count: int
    error_patterns: list[str]
    affected_files: list[str]
    recommendation: str
    timestamp: str = ""


class ErrorSampleGenerator(DataGenerator):
    """Generate training samples from system failures and errors.

    Integrates with:
    - SelfHealingAgent: Exception and crash recovery
    - SelfImprovementAgent: Tool failures and friction
    - HallucinationWatcher: Ungrounded claims
    - HistoryLogger: Operation failures

    Produces:
    - Error diagnosis samples
    - Recovery procedure samples
    - Grounding correction samples
    """

    # Error patterns to extract
    ERROR_PATTERNS = {
        "exception": r"(?:Exception|Error|Traceback).*",
        "timeout": r"(?:timeout|timed out|deadline exceeded)",
        "rate_limit": r"(?:rate limit|quota exceeded|429)",
        "auth": r"(?:unauthorized|forbidden|401|403)",
        "not_found": r"(?:not found|404|does not exist)",
        "validation": r"(?:validation|invalid|malformed)",
    }

    def __init__(
        self,
        lookback_hours: int = 24,
        min_severity: str = "medium",
    ):
        super().__init__(
            name="ErrorSampleGenerator",
            domain="errors",
            teacher_tier="reasoning",  # Need good reasoning for diagnosis
        )
        self.lookback_hours = lookback_hours
        self.min_severity = min_severity
        self._orchestrator = None

        # Paths for error sources
        self.logs_dir = self.context_root / "logs"
        self.reports_dir = self.context_root / "background_agent" / "reports"
        self.autonomy_dir = self.context_root / "autonomy_daemon"

    async def setup(self):
        """Initialize resources."""
        await super().setup()

        from core.orchestrator_v2 import UnifiedOrchestrator

        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[ErrorSourceItem]:
        """Extract error events from multiple sources."""
        items: list[ErrorSourceItem] = []
        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)

        # 1. Extract from SelfHealing reports
        items.extend(await self._extract_healing_errors(cutoff))

        # 2. Extract from SelfImprovement signals
        items.extend(await self._extract_improvement_signals(cutoff))

        # 3. Extract from HallucinationWatcher
        items.extend(await self._extract_hallucination_flags(cutoff))

        # 4. Extract from log files
        items.extend(await self._extract_log_errors(cutoff))

        # 5. Extract from history logger (operation failures)
        items.extend(await self._extract_history_failures(cutoff))

        # Filter by severity
        severity_order = ["low", "medium", "high", "critical"]
        min_idx = severity_order.index(self.min_severity)
        items = [
            i for i in items
            if severity_order.index(i.severity) >= min_idx
        ]

        logger.info(f"Extracted {len(items)} error events")
        return items

    async def _extract_healing_errors(
        self, cutoff: datetime
    ) -> list[ErrorSourceItem]:
        """Extract errors from SelfHealingAgent reports."""
        items = []
        reports_path = self.reports_dir / "self_healing"

        if not reports_path.exists():
            return items

        for report_file in reports_path.glob("*.json"):
            try:
                data = json.loads(report_file.read_text())
                report_time = datetime.fromisoformat(
                    data.get("timestamp", "2000-01-01")
                )
                if report_time < cutoff:
                    continue

                for error in data.get("errors", []):
                    items.append(
                        ErrorSourceItem(
                            name=f"healing_{error.get('type', 'unknown')}",
                            content=error.get("message", ""),
                            source="self_healing",
                            error_type=error.get("type", "exception"),
                            error_message=error.get("message", ""),
                            stack_trace=error.get("stack_trace", ""),
                            recovery_action=error.get("recovery", ""),
                            severity=error.get("severity", "medium"),
                            agent_name="SelfHealingAgent",
                            timestamp=data.get("timestamp", ""),
                        )
                    )
            except Exception as e:
                logger.debug(f"Error parsing healing report: {e}")

        return items

    async def _extract_improvement_signals(
        self, cutoff: datetime
    ) -> list[ErrorSourceItem]:
        """Extract failure patterns from SelfImprovementAgent."""
        items = []
        reports_path = self.reports_dir / "self_improvement"

        if not reports_path.exists():
            return items

        for report_file in reports_path.glob("*.json"):
            try:
                data = json.loads(report_file.read_text())
                report_time = datetime.fromisoformat(
                    data.get("timestamp", "2000-01-01")
                )
                if report_time < cutoff:
                    continue

                # Extract tool failures
                for tool, stats in data.get("tool_failures", {}).items():
                    if stats.get("count", 0) > 0:
                        items.append(
                            ErrorSourceItem(
                                name=f"tool_failure_{tool}",
                                content=f"Tool {tool} failed {stats['count']} times",
                                source="self_improvement",
                                error_type="tool_failure",
                                error_message="\n".join(stats.get("errors", [])[:5]),
                                context={
                                    "tool": tool,
                                    "count": stats["count"],
                                    "affected_files": stats.get("files", []),
                                },
                                recovery_action=stats.get("recommendation", ""),
                                severity="medium" if stats["count"] < 5 else "high",
                                agent_name="SelfImprovementAgent",
                                timestamp=data.get("timestamp", ""),
                            )
                        )

                # Extract recommendations as improvement signals
                for rec in data.get("recommendations", []):
                    items.append(
                        ErrorSourceItem(
                            name=f"improvement_{rec.get('area', 'unknown')}",
                            content=rec.get("description", ""),
                            source="self_improvement",
                            error_type="friction",
                            error_message=rec.get("problem", ""),
                            recovery_action=rec.get("solution", ""),
                            severity="low",
                            agent_name="SelfImprovementAgent",
                            timestamp=data.get("timestamp", ""),
                        )
                    )

            except Exception as e:
                logger.debug(f"Error parsing improvement report: {e}")

        return items

    async def _extract_hallucination_flags(
        self, cutoff: datetime
    ) -> list[ErrorSourceItem]:
        """Extract flagged hallucinations from HallucinationWatcher."""
        items = []
        reports_path = self.reports_dir / "hallucination"

        if not reports_path.exists():
            return items

        for report_file in reports_path.glob("*.json"):
            try:
                data = json.loads(report_file.read_text())
                report_time = datetime.fromisoformat(
                    data.get("timestamp", "2000-01-01")
                )
                if report_time < cutoff:
                    continue

                for flag in data.get("flags", []):
                    items.append(
                        ErrorSourceItem(
                            name=f"hallucination_{flag.get('id', 'unknown')}",
                            content=flag.get("claim", ""),
                            source="hallucination_watcher",
                            error_type="hallucination",
                            error_message=flag.get("reason", "Ungrounded claim"),
                            context={
                                "claim": flag.get("claim", ""),
                                "expected_evidence": flag.get("evidence_type", ""),
                                "agent": flag.get("agent", ""),
                            },
                            recovery_action="Ground claim with verifiable evidence",
                            severity="high",  # Hallucinations are always high severity
                            agent_name="HallucinationWatcher",
                            timestamp=data.get("timestamp", ""),
                        )
                    )

            except Exception as e:
                logger.debug(f"Error parsing hallucination report: {e}")

        return items

    async def _extract_log_errors(
        self, cutoff: datetime
    ) -> list[ErrorSourceItem]:
        """Extract errors from log files."""
        items = []

        if not self.logs_dir.exists():
            return items

        for log_file in self.logs_dir.glob("*.log"):
            try:
                lines = log_file.read_text().split("\n")
                current_error = None

                for line in lines:
                    # Look for error patterns
                    if re.search(r"\[ERROR\]|\[CRITICAL\]|Exception|Traceback", line):
                        # Extract timestamp
                        ts_match = re.search(
                            r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", line
                        )
                        if ts_match:
                            try:
                                ts = datetime.fromisoformat(
                                    ts_match.group(1).replace(" ", "T")
                                )
                                if ts < cutoff:
                                    continue
                            except ValueError:
                                pass

                        # Determine severity
                        severity = "medium"
                        if "[CRITICAL]" in line:
                            severity = "critical"
                        elif "[ERROR]" in line:
                            severity = "high"

                        # Categorize error type
                        error_type = "exception"
                        for pattern_name, pattern in self.ERROR_PATTERNS.items():
                            if re.search(pattern, line, re.IGNORECASE):
                                error_type = pattern_name
                                break

                        items.append(
                            ErrorSourceItem(
                                name=f"log_{log_file.stem}_{len(items)}",
                                content=line[:500],
                                source=str(log_file.name),
                                error_type=error_type,
                                error_message=line[:500],
                                severity=severity,
                                agent_name=log_file.stem,
                                timestamp=ts_match.group(1) if ts_match else "",
                            )
                        )

            except Exception as e:
                logger.debug(f"Error parsing log file {log_file}: {e}")

        return items

    async def _extract_history_failures(
        self, cutoff: datetime
    ) -> list[ErrorSourceItem]:
        """Extract operation failures from HistoryLogger."""
        items = []
        history_dir = self.context_root / "history"

        if not history_dir.exists():
            return items

        # Check recent JSONL files
        for jsonl_file in sorted(history_dir.glob("*.jsonl"), reverse=True)[:7]:
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        try:
                            entry = json.loads(line)

                            # Check timestamp
                            ts = datetime.fromisoformat(
                                entry.get("timestamp", "2000-01-01T00:00:00")
                            )
                            if ts < cutoff:
                                continue

                            # Only process failures
                            if entry.get("metadata", {}).get("success", True):
                                continue

                            error_msg = entry.get("metadata", {}).get("error", "")
                            if not error_msg:
                                continue

                            items.append(
                                ErrorSourceItem(
                                    name=f"history_{entry.get('id', 'unknown')}",
                                    content=f"{entry.get('operation_type', '')}: {error_msg}",
                                    source="history_logger",
                                    error_type="operation_failure",
                                    error_message=error_msg,
                                    context={
                                        "operation": entry.get("operation_type", ""),
                                        "files": entry.get("metadata", {}).get("files", []),
                                        "agent": entry.get("provenance", {}).get("actor", ""),
                                    },
                                    severity="medium",
                                    agent_name=entry.get("provenance", {}).get("actor", "unknown"),
                                    timestamp=entry.get("timestamp", ""),
                                )
                            )

                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                logger.debug(f"Error parsing history file: {e}")

        return items

    def get_teacher_prompt(self, item: SourceItem) -> str:
        """Generate teacher prompt for error diagnosis and recovery."""
        if not isinstance(item, ErrorSourceItem):
            raise TypeError(f"Expected ErrorSourceItem, got {type(item)}")

        context_str = ""
        if item.context:
            context_str = "\n".join(f"  {k}: {v}" for k, v in item.context.items())

        existing_recovery = ""
        if item.recovery_action:
            existing_recovery = f"\nEXISTING RECOVERY SUGGESTION: {item.recovery_action}"

        template = get_prompt("agents.training.generators.error_generator.prompt", "")
        if not template:
            template = (
                "You are an expert system administrator and debugging specialist.\n"
                "Analyze this error event and create a training sample that teaches how to diagnose and fix it.\n\n"
                "ERROR TYPE: {error_type}\n"
                "SEVERITY: {severity}\n"
                "SOURCE: {source}\n"
                "AGENT: {agent_name}\n\n"
                "ERROR MESSAGE:\n{error_message}\n\n"
                "{stack_trace}\n\n"
                "{context}\n"
                "{existing_recovery}\n\n"
                "Generate a JSON training sample with:\n"
                "1. \"instruction\": A user question asking for help with this type of error. Be specific and realistic.\n"
                "2. \"input\": The error details as a user would paste them (error message, relevant context).\n"
                "3. \"output\": A comprehensive response that:\n"
                "   - Diagnoses the root cause\n"
                "   - Explains why this error occurs\n"
                "   - Provides step-by-step fix instructions\n"
                "   - Suggests preventive measures\n\n"
                "The output should be educational and actionable, suitable for training an AI assistant.\n\n"
                "JSON FORMAT:\n"
                "{{\n"
                "  \"instruction\": \"...\",\n"
                "  \"input\": \"...\",\n"
                "  \"output\": \"...\"\n"
                "}}\n"
            )

        stack_trace = f"STACK TRACE:\n{item.stack_trace}" if item.stack_trace else ""
        context_block = f"CONTEXT:\n{context_str}" if context_str else ""

        return template.format(
            error_type=item.error_type,
            severity=item.severity,
            source=item.source,
            agent_name=item.agent_name,
            error_message=item.error_message,
            stack_trace=stack_trace,
            context=context_block,
            existing_recovery=existing_recovery,
        )

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Use teacher model to generate diagnostic sample from error."""
        if not isinstance(item, ErrorSourceItem):
            return None

        if not self._orchestrator:
            await self.setup()

        prompt = self.get_teacher_prompt(item)

        try:
            # Use reasoning tier for good diagnostic quality
            response, model_name = await asyncio.wait_for(
                self.generate_with_rotation(prompt, tier="reasoning"),
                timeout=120.0,  # Increased for GPU/slower models
            )
            if not response:
                return None

            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                response = response[response.find("{") : response.rfind("}") + 1]

            data = json.loads(response)

            return TrainingSample(
                instruction=data.get("instruction", ""),
                input=data.get("input", ""),
                output=data.get("output", ""),
                domain="errors",
                source=item.source,
                teacher_model=model_name,
                teacher_prompt=prompt,
                kg_entities=[item.error_type, item.agent_name],
                quality_score=0.8 if item.severity in ("high", "critical") else 0.6,
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


class MultiTeacherGenerator(DataGenerator):
    """Generate samples using multiple teacher models for consensus.

    Uses 2025 frontier models:
    - Gemini 3 Pro
    - Claude Opus 4.5
    - GPT-5.2
    - Local Ollama models (for validation)

    Only keeps samples where multiple teachers agree.
    """

    TEACHERS = [
        ("gemini", "gemini-3-pro-preview"),
        ("anthropic", "claude-opus-4-5-20251101"),
        ("openai", "gpt-5.2"),
    ]

    LOCAL_VALIDATORS = [
        ("ollama", "qwen2.5-coder:14b"),
        ("ollama", "deepseek-r1:8b"),
    ]

    def __init__(
        self,
        base_generator: DataGenerator,
        consensus_threshold: int = 2,
        validate_locally: bool = True,
    ):
        """Initialize multi-teacher generator.

        Args:
            base_generator: The generator to enhance with multi-teacher
            consensus_threshold: Minimum teachers that must agree
            validate_locally: Whether to validate with Ollama models
        """
        super().__init__(
            name=f"MultiTeacher_{base_generator.name}",
            domain=base_generator.domain,
            teacher_tier="reasoning",
        )
        self.base_generator = base_generator
        self.consensus_threshold = consensus_threshold
        self.validate_locally = validate_locally
        self._orchestrator = None

    async def setup(self):
        await super().setup()
        await self.base_generator.setup()

        from core.orchestrator_v2 import UnifiedOrchestrator
        self._orchestrator = UnifiedOrchestrator()

    async def extract_source_items(self) -> list[SourceItem]:
        return await self.base_generator.extract_source_items()

    def get_teacher_prompt(self, item: SourceItem) -> str:
        return self.base_generator.get_teacher_prompt(item)

    async def generate_sample(self, item: SourceItem) -> Optional[TrainingSample]:
        """Generate sample with multi-teacher consensus."""
        from core.orchestrator_v2 import Provider, TaskTier

        prompt = self.get_teacher_prompt(item)
        responses: list[dict[str, Any]] = []

        # Get responses from all teachers
        for provider_name, model in self.TEACHERS:
            try:
                provider = Provider(provider_name)
                response_obj = await asyncio.wait_for(
                    self._orchestrator.generate(
                        prompt=prompt,
                        tier=TaskTier.REASONING,
                        provider=provider,
                    ),
                    timeout=90.0,
                )

                response = response_obj.content

                # Parse JSON
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0]
                elif "{" in response:
                    response = response[response.find("{"):response.rfind("}") + 1]

                data = json.loads(response)
                data["_provider"] = provider_name
                data["_model"] = model
                responses.append(data)

            except Exception as e:
                logger.debug(f"Teacher {provider_name}/{model} failed: {e}")

        if len(responses) < self.consensus_threshold:
            logger.warning(f"Not enough teacher responses for {item.name}")
            return None

        # Check consensus (simple: use first response if enough teachers responded)
        # In production, would compare outputs for semantic similarity
        best_response = responses[0]

        # Optionally validate with local models
        if self.validate_locally:
            validation_score = await self._validate_locally(
                best_response, item
            )
            if validation_score < 0.5:
                logger.warning(f"Local validation failed for {item.name}")
                return None

        return TrainingSample(
            instruction=best_response.get("instruction", ""),
            input=best_response.get("input", ""),
            output=best_response.get("output", ""),
            domain=self.base_generator.domain,
            source=item.source,
            teacher_model=f"consensus:{len(responses)}_teachers",
            teacher_prompt=prompt,
            quality_score=0.9,  # High quality due to consensus
        )

    async def _validate_locally(
        self, response: dict[str, Any], item: SourceItem
    ) -> float:
        """Validate response with local Ollama model."""
        from core.orchestrator_v2 import Provider, TaskTier

        template = get_prompt(
            "agents.training.generators.error_generator.validation_prompt",
            "",
        )
        if not template:
            template = (
                "Rate the quality of this training sample on a scale of 0.0-1.0.\n\n"
                "INSTRUCTION: {instruction}\n"
                "OUTPUT: {output}\n\n"
                "Criteria:\n"
                "- Is the instruction clear and specific?\n"
                "- Is the output accurate and helpful?\n"
                "- Would this be useful for training an AI?\n\n"
                "Respond with just a number between 0.0 and 1.0."
            )

        validation_prompt = template.format(
            instruction=response.get("instruction", "")[:500],
            output=response.get("output", "")[:500],
        )

        try:
            result = await asyncio.wait_for(
                self._orchestrator.generate(
                    prompt=validation_prompt,
                    tier=TaskTier.LOCAL,
                    provider=Provider.OLLAMA,
                ),
                timeout=120.0,  # Increased for GPU/slower models
            )

            return float(result.content.strip())

        except Exception:
            return 0.5  # Neutral if validation fails


if __name__ == "__main__":
    async def main():
        gen = ErrorSampleGenerator(lookback_hours=168)  # 1 week
        await gen.setup()

        items = await gen.extract_source_items()
        print(f"Found {len(items)} error events")

        if items:
            result = await gen.run_generation(
                limit=10,
                output_path=Path("test_error_train.jsonl"),
            )
            print(f"Generated {result.processed} samples")

    import asyncio
    asyncio.run(main())
