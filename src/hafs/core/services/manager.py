"""Service manager - main orchestration class."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from hafs.core.runtime import resolve_python_executable
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
        self._aliases = {
            "autonomy": "autonomy-daemon",
            "autonomy-daemon": "autonomy-daemon",
            "embed": "embedding-daemon",
            "embedding": "embedding-daemon",
            "embeddings": "embedding-daemon",
            "embedding-daemon": "embedding-daemon",
            "context": "context-agent-daemon",
            "context-agent": "context-agent-daemon",
            "context-daemon": "context-agent-daemon",
            "context-agent-daemon": "context-agent-daemon",
            "observability": "observability-daemon",
            "observability-daemon": "observability-daemon",
        }

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
        return resolve_python_executable(self._config)

    def _find_repo_root(self) -> Optional[Path]:
        """Best-effort repo root detection for source checkouts."""
        for parent in Path(__file__).resolve().parents:
            if (parent / "pyproject.toml").exists():
                return parent
        return None

    def _get_service_environment(self) -> dict[str, str]:
        """Build environment defaults for managed services."""
        env: dict[str, str] = {}
        repo_root = self._find_repo_root()
        if repo_root:
            src_path = repo_root / "src"
            if src_path.exists():
                env["PYTHONPATH"] = str(src_path)
        user_config = Path.home() / ".config" / "hafs" / "config.toml"
        if user_config.exists():
            env["HAFS_CONFIG_PATH"] = str(user_config)
            env["HAFS_PREFER_USER_CONFIG"] = "1"
        return env

    def _normalize_service_name(self, name: str) -> str:
        """Normalize service names and apply alias mapping."""
        normalized = name.strip().lower().replace("_", "-")
        return self._aliases.get(normalized, normalized)

    def _get_builtin_services(self) -> dict[str, ServiceDefinition]:
        """Get built-in service definitions."""
        python = self._get_python_executable()
        hafs_root = Path(__file__).parent.parent.parent
        repo_root = self._find_repo_root()
        environment = self._get_service_environment()

        return {
            "orchestrator": ServiceDefinition(
                name="orchestrator",
                label="HAFS Model Orchestrator",
                description="Intelligent model routing with quota management",
                command=[python, "-m", "hafs.core.orchestrator", "--daemon"],
                working_directory=repo_root,
                environment=environment,
            ),
            "coordinator": ServiceDefinition(
                name="coordinator",
                label="HAFS Agent Coordinator",
                description="Multi-agent swarm orchestration",
                command=[python, "-m", "hafs.agents.coordinator", "--daemon"],
                working_directory=repo_root,
                environment=environment,
            ),
            "autonomy-daemon": ServiceDefinition(
                name="autonomy-daemon",
                label="HAFS Autonomy Daemon",
                description="Self-improvement, curiosity, self-healing, and safety loops",
                command=[python, "-m", "hafs.services.autonomy_daemon", "--interval", "30"],
                working_directory=repo_root,
                environment=environment,
            ),
            "embedding-daemon": ServiceDefinition(
                name="embedding-daemon",
                label="HAFS Embedding Daemon",
                description="Continuous embedding generation for registered projects",
                command=[
                    python,
                    "-m",
                    "hafs.services.embedding_daemon",
                    "--batch-size",
                    "50",
                    "--interval",
                    "60",
                ],
                working_directory=repo_root,
                environment=environment,
            ),
            "context-agent-daemon": ServiceDefinition(
                name="context-agent-daemon",
                label="HAFS Context Agent Daemon",
                description="Scheduled context reports and AFS sync",
                command=[python, "-m", "hafs.services.context_agent_daemon", "--interval", "300"],
                working_directory=repo_root,
                environment=environment,
            ),
            "observability-daemon": ServiceDefinition(
                name="observability-daemon",
                label="HAFS Observability Daemon",
                description="Distributed health monitoring with safe remediations",
                command=[python, "-m", "hafs.services.observability_daemon"],
                working_directory=repo_root,
                environment=environment,
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
                working_directory=repo_root,
                environment=environment,
            ),
        }

    def get_service_definition(self, name: str) -> Optional[ServiceDefinition]:
        """Get service definition by name."""
        if name in self._custom_services:
            return self._custom_services[name]
        normalized = self._normalize_service_name(name)
        if normalized in self._custom_services:
            return self._custom_services[normalized]
        builtin = self._get_builtin_services()
        return builtin.get(normalized)

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
        normalized = self._normalize_service_name(name)
        return await self._adapter.uninstall(normalized)

    async def start(self, name: str) -> bool:
        """Start a service."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.start(normalized)

    async def stop(self, name: str) -> bool:
        """Stop a service."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.stop(normalized)

    async def restart(self, name: str) -> bool:
        """Restart a service."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.restart(normalized)

    async def enable(self, name: str) -> bool:
        """Enable service to start at boot/login."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.enable(normalized)

    async def disable(self, name: str) -> bool:
        """Disable service from starting at boot/login."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.disable(normalized)

    async def status(self, name: str) -> ServiceStatus:
        """Get service status."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.status(normalized)

    async def status_all(self) -> dict[str, ServiceStatus]:
        """Get status of all known services."""
        results = {}
        for name in self.list_services():
            results[name] = await self.status(name)
        return results

    async def logs(self, name: str, lines: int = 100) -> str:
        """Get recent log output for a service."""
        normalized = self._normalize_service_name(name)
        return await self._adapter.logs(normalized, lines)

    def stream_logs(self, name: str):
        """Stream logs for a service (async generator)."""
        normalized = self._normalize_service_name(name)
        return self._adapter.stream_logs(normalized)
