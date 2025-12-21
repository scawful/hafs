"""Abstract base class for service adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from services.models import ServiceDefinition, ServiceStatus


class ServiceAdapter(ABC):
    """Abstract interface for platform-specific service management.

    Implementations handle the details of interacting with system service
    managers like launchd (macOS) or systemd (Linux).
    """

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Human-readable platform name."""
        pass

    @abstractmethod
    async def install(self, definition: ServiceDefinition) -> bool:
        """Install service configuration files.

        Args:
            definition: Service definition to install.

        Returns:
            True if installation succeeded.
        """
        pass

    @abstractmethod
    async def uninstall(self, name: str) -> bool:
        """Remove service configuration files.

        Args:
            name: Service name to uninstall.

        Returns:
            True if uninstallation succeeded.
        """
        pass

    @abstractmethod
    async def start(self, name: str) -> bool:
        """Start the service.

        Args:
            name: Service name to start.

        Returns:
            True if start succeeded.
        """
        pass

    @abstractmethod
    async def stop(self, name: str) -> bool:
        """Stop the service.

        Args:
            name: Service name to stop.

        Returns:
            True if stop succeeded.
        """
        pass

    @abstractmethod
    async def restart(self, name: str) -> bool:
        """Restart the service.

        Args:
            name: Service name to restart.

        Returns:
            True if restart succeeded.
        """
        pass

    @abstractmethod
    async def enable(self, name: str) -> bool:
        """Enable auto-start at boot/login.

        Args:
            name: Service name to enable.

        Returns:
            True if enable succeeded.
        """
        pass

    @abstractmethod
    async def disable(self, name: str) -> bool:
        """Disable auto-start.

        Args:
            name: Service name to disable.

        Returns:
            True if disable succeeded.
        """
        pass

    @abstractmethod
    async def status(self, name: str) -> ServiceStatus:
        """Get current service status.

        Args:
            name: Service name to query.

        Returns:
            ServiceStatus with current state information.
        """
        pass

    @abstractmethod
    async def logs(self, name: str, lines: int = 100) -> str:
        """Get recent log lines.

        Args:
            name: Service name.
            lines: Number of lines to retrieve.

        Returns:
            Log content as string.
        """
        pass

    @abstractmethod
    def stream_logs(self, name: str) -> AsyncIterator[str]:
        """Stream log output.

        Args:
            name: Service name.

        Yields:
            Log lines as they appear.
        """
        pass
