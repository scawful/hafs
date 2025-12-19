"""macOS launchd service adapter."""

from __future__ import annotations

import asyncio
import plistlib
from collections.abc import AsyncIterator
from pathlib import Path

from hafs.core.services.adapters.base import ServiceAdapter
from hafs.core.services.models import ServiceDefinition, ServiceStatus, ServiceState


class LaunchdAdapter(ServiceAdapter):
    """macOS launchd adapter for user-level services.

    Uses ~/Library/LaunchAgents for plist files.
    Logs are stored in ~/.local/share/hafs/logs/.
    """

    AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
    LOGS_DIR = Path.home() / ".local" / "share" / "hafs" / "logs"
    LABEL_PREFIX = "com.hafs"

    @property
    def platform_name(self) -> str:
        return "macOS (launchd)"

    def _plist_path(self, name: str) -> Path:
        """Get plist file path for service."""
        return self.AGENTS_DIR / f"{self.LABEL_PREFIX}.{name}.plist"

    def _label(self, name: str) -> str:
        """Get launchd label for service."""
        return f"{self.LABEL_PREFIX}.{name}"

    def _log_path(self, name: str) -> Path:
        """Get log file path for service."""
        return self.LOGS_DIR / f"{name}.log"

    def _error_log_path(self, name: str) -> Path:
        """Get error log path for service."""
        return self.LOGS_DIR / f"{name}.error.log"

    def _generate_plist(self, definition: ServiceDefinition) -> dict:
        """Generate plist dictionary from service definition."""
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        log_path = definition.log_path or self._log_path(definition.name)
        error_log_path = definition.error_log_path or self._error_log_path(
            definition.name
        )

        plist: dict = {
            "Label": self._label(definition.name),
            "ProgramArguments": definition.command,
            "RunAtLoad": definition.run_at_load,
            "StandardOutPath": str(log_path),
            "StandardErrorPath": str(error_log_path),
        }

        if definition.working_directory:
            plist["WorkingDirectory"] = str(definition.working_directory)

        if definition.environment:
            plist["EnvironmentVariables"] = definition.environment

        if definition.keep_alive:
            if definition.restart_on_failure:
                # Restart on non-zero exit only
                plist["KeepAlive"] = {"SuccessfulExit": False}
            else:
                plist["KeepAlive"] = True
            plist["ThrottleInterval"] = definition.restart_delay_seconds
        else:
            plist["KeepAlive"] = False

        return plist

    async def install(self, definition: ServiceDefinition) -> bool:
        """Write plist file to LaunchAgents."""
        self.AGENTS_DIR.mkdir(parents=True, exist_ok=True)

        plist_data = self._generate_plist(definition)
        plist_path = self._plist_path(definition.name)

        with open(plist_path, "wb") as f:
            plistlib.dump(plist_data, f)

        return True

    async def uninstall(self, name: str) -> bool:
        """Remove plist file and stop service."""
        await self.stop(name)

        plist_path = self._plist_path(name)
        if plist_path.exists():
            plist_path.unlink()
            return True
        return False

    async def start(self, name: str) -> bool:
        """Load and start service with launchctl."""
        plist_path = self._plist_path(name)
        if not plist_path.exists():
            return False

        proc = await asyncio.create_subprocess_exec(
            "launchctl",
            "load",
            str(plist_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def stop(self, name: str) -> bool:
        """Unload service with launchctl."""
        plist_path = self._plist_path(name)
        if not plist_path.exists():
            return True  # Already not installed

        proc = await asyncio.create_subprocess_exec(
            "launchctl",
            "unload",
            str(plist_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    async def restart(self, name: str) -> bool:
        """Stop then start service."""
        await self.stop(name)
        await asyncio.sleep(0.5)
        return await self.start(name)

    async def enable(self, name: str) -> bool:
        """Enable RunAtLoad in plist."""
        plist_path = self._plist_path(name)
        if not plist_path.exists():
            return False

        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)

        plist["RunAtLoad"] = True

        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)

        return True

    async def disable(self, name: str) -> bool:
        """Disable RunAtLoad in plist."""
        plist_path = self._plist_path(name)
        if not plist_path.exists():
            return False

        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)

        plist["RunAtLoad"] = False

        with open(plist_path, "wb") as f:
            plistlib.dump(plist, f)

        return True

    async def status(self, name: str) -> ServiceStatus:
        """Get service status from launchctl."""
        label = self._label(name)

        # Check if plist exists
        plist_path = self._plist_path(name)
        enabled = plist_path.exists()

        # Get service info from launchctl list
        proc = await asyncio.create_subprocess_exec(
            "launchctl",
            "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        state = ServiceState.STOPPED
        pid = None
        last_exit_code = None

        for line in stdout.decode().splitlines():
            if label in line:
                parts = line.split()
                if len(parts) >= 3:
                    pid_str = parts[0]
                    exit_str = parts[1]

                    if pid_str != "-":
                        try:
                            pid = int(pid_str)
                            state = ServiceState.RUNNING
                        except ValueError:
                            pass

                    if exit_str != "-":
                        try:
                            last_exit_code = int(exit_str)
                            if pid is None and last_exit_code != 0:
                                state = ServiceState.FAILED
                        except ValueError:
                            pass
                break

        return ServiceStatus(
            name=name,
            state=state,
            pid=pid,
            enabled=enabled,
            last_exit_code=last_exit_code,
        )

    async def logs(self, name: str, lines: int = 100) -> str:
        """Read recent log lines."""
        log_path = self._log_path(name)
        if not log_path.exists():
            return ""

        proc = await asyncio.create_subprocess_exec(
            "tail",
            f"-{lines}",
            str(log_path),
            stdout=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode()

    async def stream_logs(self, name: str) -> AsyncIterator[str]:
        """Stream log output using tail -f."""
        log_path = self._log_path(name)
        if not log_path.exists():
            # Create empty log file so tail -f works
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch()

        proc = await asyncio.create_subprocess_exec(
            "tail",
            "-f",
            str(log_path),
            stdout=asyncio.subprocess.PIPE,
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
