"""Service manager - main orchestration class."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hafs.core.services.adapters.base import ServiceAdapter
from hafs.core.services.models import ServiceDefinition, ServiceStatus

if TYPE_CHECKING:
    from hafs.config.schema import HafsConfig


class ServiceManager:
    """Cross-platform service manager for HAFS.

    Provides unified API for managing background services across
    macOS (launchd) and Linux (systemd).

    Example:
        manager = ServiceManager()

        # Define a service
        definition = ServiceDefinition(
            name="orchestrator",
            label="HAFS Model Orchestrator",
            command=["python", "-m", "hafs.core.services.daemons.orchestrator"],
        )

        # Install and start
        await manager.install(definition)
        await manager.start("orchestrator")

        # Check status
        status = await manager.status("orchestrator")
        print(f"State: {status.state}")
    """

    def __init__(self, config: Optional["HafsConfig"] = None) -> None:
        self._config = config
        self._adapter = self._create_adapter()
        self._custom_services: dict[str, ServiceDefinition] = {}

    def _create_adapter(self) -> ServiceAdapter:
        """Create platform-appropriate adapter."""
        system = platform.system()

        if system == "Darwin":
            from hafs.core.services.adapters.launchd import LaunchdAdapter

            return LaunchdAdapter()
        elif system == "Linux":
            from hafs.core.services.adapters.systemd import SystemdAdapter

            return SystemdAdapter()
        else:
            raise NotImplementedError(
                f"Service management not supported on {system}. "
                f"Supported platforms: macOS, Linux"
            )

    @property
    def platform_name(self) -> str:
        """Get human-readable platform name."""
        return self._adapter.platform_name

    def _get_python_executable(self) -> str:
        """Get the Python executable path."""
        return sys.executable

    def _get_builtin_services(self) -> dict[str, ServiceDefinition]:
        """Get built-in service definitions."""
        python = self._get_python_executable()
        hafs_root = Path(__file__).parent.parent.parent

        return {
            "orchestrator": ServiceDefinition(
                name="orchestrator",
                label="HAFS Model Orchestrator",
                description="Intelligent model routing with quota management",
                command=[python, "-m", "hafs.core.orchestrator", "--daemon"],
            ),
            "coordinator": ServiceDefinition(
                name="coordinator",
                label="HAFS Agent Coordinator",
                description="Multi-agent swarm orchestration",
                command=[python, "-m", "hafs.agents.coordinator", "--daemon"],
            ),
            "autonomy": ServiceDefinition(
                name="autonomy",
                label="HAFS Autonomy Daemon",
                description="Self-improvement, curiosity, self-healing, and safety loops",
                command=[python, "-m", "hafs.services.autonomy_daemon", "--interval", "30"],
            ),
            "dashboard": ServiceDefinition(
                name="dashboard",
                label="HAFS Web Dashboard",
                description="Streamlit-based monitoring dashboard",
                command=[
                    "streamlit",
                    "run",
                    str(hafs_root / "ui" / "web_dashboard.py"),
                    "--server.headless",
                    "true",
                    "--server.port",
                    "8501",
                ],
            ),
        }

    def get_service_definition(self, name: str) -> Optional[ServiceDefinition]:
        """Get service definition by name."""
        if name in self._custom_services:
            return self._custom_services[name]
        builtin = self._get_builtin_services()
        return builtin.get(name)

    def list_services(self) -> list[str]:
        """List all known service names."""
        builtin = self._get_builtin_services()
        return list(set(builtin.keys()) | set(self._custom_services.keys()))

    def register_service(self, definition: ServiceDefinition) -> None:
        """Register a custom service definition."""
        self._custom_services[definition.name] = definition

    async def install(self, definition: ServiceDefinition) -> bool:
        """Install service configuration (create plist/unit file)."""
        return await self._adapter.install(definition)

    async def install_by_name(self, name: str) -> bool:
        """Install a service by name."""
        definition = self.get_service_definition(name)
        if not definition:
            return False
        return await self.install(definition)

    async def uninstall(self, name: str) -> bool:
        """Uninstall service configuration."""
        return await self._adapter.uninstall(name)

    async def start(self, name: str) -> bool:
        """Start a service."""
        return await self._adapter.start(name)

    async def stop(self, name: str) -> bool:
        """Stop a service."""
        return await self._adapter.stop(name)

    async def restart(self, name: str) -> bool:
        """Restart a service."""
        return await self._adapter.restart(name)

    async def enable(self, name: str) -> bool:
        """Enable service to start at boot/login."""
        return await self._adapter.enable(name)

    async def disable(self, name: str) -> bool:
        """Disable service from starting at boot/login."""
        return await self._adapter.disable(name)

    async def status(self, name: str) -> ServiceStatus:
        """Get service status."""
        return await self._adapter.status(name)

    async def status_all(self) -> dict[str, ServiceStatus]:
        """Get status of all known services."""
        results = {}
        for name in self.list_services():
            results[name] = await self.status(name)
        return results

    async def logs(self, name: str, lines: int = 100) -> str:
        """Get recent log output for a service."""
        return await self._adapter.logs(name, lines)

    def stream_logs(self, name: str):
        """Stream logs for a service (async generator)."""
        return self._adapter.stream_logs(name)
