from __future__ import annotations

from datetime import datetime, timedelta

from agents.utility.observability import Alert, AlertSeverity, HealthCheck, HealthStatus
from hafs.config.schema import ObservabilityRemediationConfig
from hafs.services.observability_daemon import RemediationPlanner
from services.models import ServiceState, ServiceStatus


def _service_status(name: str, state: ServiceState, enabled: bool = True) -> ServiceStatus:
    return ServiceStatus(name=name, state=state, enabled=enabled)


def _sync_check(profile: str, status: HealthStatus) -> HealthCheck:
    return HealthCheck(
        endpoint="node",
        name=f"sync:{profile}",
        status=status,
        latency_ms=0,
        timestamp=datetime.now().isoformat(),
        message="sync check",
        details={"profile": profile, "target": "node"},
    )


def test_remediation_planner_respects_allowlists() -> None:
    config = ObservabilityRemediationConfig(
        enabled=True,
        allowed_actions=["restart_service", "run_afs_sync"],
        allowed_services=["embedding-daemon"],
        allowed_sync_profiles=["global"],
        cooldown_minutes=0,
    )
    planner = RemediationPlanner(config)
    service_statuses = {
        "embedding-daemon": _service_status("embedding-daemon", ServiceState.FAILED),
        "context-agent-daemon": _service_status("context-agent-daemon", ServiceState.STOPPED),
    }
    sync_health = {"sync:global:node": _sync_check("global", HealthStatus.UNHEALTHY)}

    actions = planner.plan(service_statuses=service_statuses, sync_health=sync_health, alerts=[])

    assert any(
        action.action == "restart_service" and action.target == "embedding-daemon"
        for action in actions
    )
    assert not any(action.target == "context-agent-daemon" for action in actions)
    assert any(action.action == "run_afs_sync" and action.target == "global" for action in actions)


def test_remediation_planner_cooldown_blocks_actions() -> None:
    now = datetime.now()
    recent = {
        "restart_service:embedding-daemon": (now - timedelta(minutes=5)).isoformat()
    }
    config = ObservabilityRemediationConfig(
        enabled=True,
        allowed_actions=["restart_service"],
        allowed_services=["embedding-daemon"],
        cooldown_minutes=30,
    )
    planner = RemediationPlanner(config, recent_actions=recent, now=now)
    service_statuses = {
        "embedding-daemon": _service_status("embedding-daemon", ServiceState.FAILED)
    }

    actions = planner.plan(service_statuses=service_statuses, sync_health={}, alerts=[])

    assert actions == []


def test_remediation_planner_triggers_context_burst() -> None:
    config = ObservabilityRemediationConfig(
        enabled=True,
        allowed_actions=["context_burst"],
        cooldown_minutes=0,
        trigger_context_burst_on_alerts=["error"],
    )
    planner = RemediationPlanner(config)
    alerts = [
        Alert(
            severity=AlertSeverity.ERROR,
            source="node:alpha",
            message="offline",
            timestamp=datetime.now().isoformat(),
        )
    ]

    actions = planner.plan(service_statuses={}, sync_health={}, alerts=alerts)

    assert any(action.action == "context_burst" for action in actions)
