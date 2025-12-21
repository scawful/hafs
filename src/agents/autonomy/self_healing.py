"""Detect crashed services and attempt repairs."""

from __future__ import annotations

from agents.autonomy.base import LoopReport, MemoryAwareAgent
from services import ServiceManager, ServiceState


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
