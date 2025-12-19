"""Pydantic models for service management."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ServiceState(str, Enum):
    """Service runtime state."""

    STOPPED = "stopped"
    RUNNING = "running"
    FAILED = "failed"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    """Type of service for platform-specific handling."""

    DAEMON = "daemon"  # Long-running background process
    ONESHOT = "oneshot"  # Run once and exit


class ServiceDefinition(BaseModel):
    """Definition of a managed service."""

    name: str  # Unique identifier (e.g., "orchestrator")
    label: str  # Human-readable name
    description: str = ""
    command: list[str]  # Command to execute
    working_directory: Optional[Path] = None
    environment: dict[str, str] = Field(default_factory=dict)
    service_type: ServiceType = ServiceType.DAEMON
    restart_on_failure: bool = True
    restart_delay_seconds: int = 5
    keep_alive: bool = True  # Auto-restart if stopped
    run_at_load: bool = False  # Start at login (launchd) / enable (systemd)
    log_path: Optional[Path] = None  # Custom log file path
    error_log_path: Optional[Path] = None


class ServiceStatus(BaseModel):
    """Runtime status of a service."""

    name: str
    state: ServiceState
    pid: Optional[int] = None
    uptime_seconds: Optional[float] = None
    last_exit_code: Optional[int] = None
    last_started: Optional[datetime] = None
    last_stopped: Optional[datetime] = None
    enabled: bool = False  # Whether service is registered/installed
    error_message: Optional[str] = None
