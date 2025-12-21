"""Linux systemd service adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from services.adapters.base import ServiceAdapter
from services.models import ServiceDefinition, ServiceState, ServiceStatus


class SystemdAdapter(ServiceAdapter):
    """Linux systemd adapter for user-level services.

    Uses ~/.config/systemd/user/ for unit files.
    """

    USER_UNITS_DIR = Path.home() / ".config" / "systemd" / "user"
    UNIT_PREFIX = "hafs"

    @property
    def platform_name(self) -> str:
        return "Linux (systemd)"

    def _unit_path(self, name: str) -> Path:
        """Get unit file path for service."""
        return self.USER_UNITS_DIR / f"{self.UNIT_PREFIX}-{name}.service"

    def _unit_name(self, name: str) -> str:
        """Get systemd unit name for service."""
        return f"{self.UNIT_PREFIX}-{name}"

    def _generate_unit(self, definition: ServiceDefinition) -> str:
        """Generate systemd unit file content."""
        lines = [
            "[Unit]",
            f"Description={definition.description or definition.label}",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={' '.join(definition.command)}",
        ]

        if definition.working_directory:
            lines.append(f"WorkingDirectory={definition.working_directory}")

        for key, value in definition.environment.items():
            lines.append(f"Environment={key}={value}")

        if definition.restart_on_failure:
            lines.append("Restart=on-failure")
            lines.append(f"RestartSec={definition.restart_delay_seconds}")

        lines.extend(
            [
                "",
                "[Install]",
                "WantedBy=default.target",
            ]
        )

        return "\n".join(lines)

    async def _daemon_reload(self) -> None:
        """Reload systemd daemon to pick up unit file changes."""
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "daemon-reload",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    async def install(self, definition: ServiceDefinition) -> bool:
        """Write unit file to systemd user directory."""
        self.USER_UNITS_DIR.mkdir(parents=True, exist_ok=True)

        unit_content = self._generate_unit(definition)
        unit_path = self._unit_path(definition.name)

        unit_path.write_text(unit_content)

        await self._daemon_reload()
        return True

    async def uninstall(self, name: str) -> bool:
        """Remove unit file and stop service."""
        await self.stop(name)
        await self.disable(name)

        unit_path = self._unit_path(name)
        if unit_path.exists():
            unit_path.unlink()
            await self._daemon_reload()
            return True
        return False

    async def start(self, name: str) -> bool:
        """Start service with systemctl."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "start",
            unit_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def stop(self, name: str) -> bool:
        """Stop service with systemctl."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "stop",
            unit_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def restart(self, name: str) -> bool:
        """Restart service with systemctl."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "restart",
            unit_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def enable(self, name: str) -> bool:
        """Enable service to start at boot."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "enable",
            unit_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def disable(self, name: str) -> bool:
        """Disable service from starting at boot."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "disable",
            unit_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def status(self, name: str) -> ServiceStatus:
        """Get service status from systemctl."""
        unit_name = self._unit_name(name)

        # Check if unit file exists
        unit_path = self._unit_path(name)
        enabled = unit_path.exists()

        # Get status using systemctl show
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "--user",
            "show",
            unit_name,
            "--property=ActiveState,MainPID,ExecMainStatus",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        state = ServiceState.UNKNOWN
        pid = None
        last_exit_code = None

        for line in stdout.decode().splitlines():
            if line.startswith("ActiveState="):
                active_state = line.split("=", 1)[1]
                state_map = {
                    "active": ServiceState.RUNNING,
                    "inactive": ServiceState.STOPPED,
                    "failed": ServiceState.FAILED,
                    "activating": ServiceState.STARTING,
                    "deactivating": ServiceState.STOPPING,
                }
                state = state_map.get(active_state, ServiceState.UNKNOWN)
            elif line.startswith("MainPID="):
                pid_str = line.split("=", 1)[1]
                if pid_str != "0":
                    try:
                        pid = int(pid_str)
                    except ValueError:
                        pass
            elif line.startswith("ExecMainStatus="):
                try:
                    last_exit_code = int(line.split("=", 1)[1])
                except ValueError:
                    pass

        return ServiceStatus(
            name=name,
            state=state,
            pid=pid,
            enabled=enabled,
            last_exit_code=last_exit_code,
        )

    async def logs(self, name: str, lines: int = 100) -> str:
        """Get recent logs from journalctl."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "journalctl",
            "--user",
            "-u",
            unit_name,
            "-n",
            str(lines),
            "--no-pager",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()

    async def stream_logs(self, name: str) -> AsyncIterator[str]:
        """Stream logs with journalctl -f."""
        unit_name = self._unit_name(name)
        proc = await asyncio.create_subprocess_exec(
            "journalctl",
            "--user",
            "-u",
            unit_name,
            "-f",
            "--no-pager",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            while proc.stdout:
                line = await proc.stdout.readline()
                if not line:
                    break
                yield line.decode()
        finally:
            proc.terminate()
            await proc.wait()
