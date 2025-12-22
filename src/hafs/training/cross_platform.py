"""Cross-Platform Path Resolution and Operations.

Handles path conversions and operations across Mac, Windows, SSH, and network mounts.
Standardizes path handling to avoid manual slash/backslash conversions.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Optional, Union

logger = logging.getLogger(__name__)

Platform = Literal["mac", "linux", "windows"]


@dataclass
class RemoteHost:
    """Remote host configuration."""

    hostname: str
    username: str
    platform: Platform
    ssh_key: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.username}@{self.hostname}"


@dataclass
class MountPoint:
    """Network mount point."""

    local_path: Path
    remote_path: str
    remote_host: Optional[RemoteHost] = None
    is_accessible: bool = False


class CrossPlatformPath:
    """Unified path handling across platforms.

    Handles conversions between:
    - Mac paths: /Users/scawful/Code/hafs
    - Windows paths: C:\\Users\\Administrator\\Code\\hafs or C:/Users/Administrator/Code/hafs
    - SSH paths: /c/Users/Administrator/Code/hafs (Git Bash style)
    - Mount paths: /Users/scawful/Mounts/mm-d/hafs_training
    """

    def __init__(self, path: Union[str, Path], platform: Platform):
        """Initialize cross-platform path.

        Args:
            path: Path string or Path object
            platform: Platform where path is valid (mac, linux, windows)
        """
        self.platform = platform
        self._original = str(path)

        # Parse based on platform
        if platform == "windows":
            self._path = PureWindowsPath(path)
        else:
            self._path = PurePosixPath(path)

    def to_windows(self, style: Literal["backslash", "forward"] = "forward") -> str:
        """Convert to Windows path.

        Args:
            style: Use backslashes (C:\\path) or forward slashes (C:/path)

        Returns:
            Windows path string
        """
        if self.platform == "windows":
            path_str = str(self._path)
        else:
            # Convert POSIX to Windows
            path_str = str(self._path).replace("/", "\\")
            # Handle /Users -> C:\\Users conversions if needed
            if path_str.startswith("\\"):
                path_str = "C:" + path_str

        if style == "forward":
            path_str = path_str.replace("\\", "/")

        return path_str

    def to_posix(self) -> str:
        """Convert to POSIX path (Mac/Linux).

        Returns:
            POSIX path string with forward slashes
        """
        if self.platform != "windows":
            return str(self._path)

        # Convert Windows to POSIX
        path_str = str(self._path)
        # C:\path -> /c/path (Git Bash style)
        if len(path_str) > 1 and path_str[1] == ":":
            drive = path_str[0].lower()
            rest = path_str[2:].replace("\\", "/")
            return f"/{drive}{rest}"

        return path_str.replace("\\", "/")

    def to_ssh_path(self) -> str:
        """Convert to SSH-compatible path for remote commands.

        Returns:
            Path suitable for SSH commands (Git Bash style for Windows)
        """
        return self.to_posix()

    def for_shell_escape(self) -> str:
        """Escape path for shell commands.

        Returns:
            Shell-escaped path
        """
        path = self.to_posix() if self.platform != "windows" else self.to_windows("forward")
        # Escape spaces and special characters
        return path.replace(" ", "\\ ").replace("(", "\\(").replace(")", "\\)")

    def __str__(self) -> str:
        """Return platform-native path representation."""
        if self.platform == "windows":
            return self.to_windows("backslash")
        return self.to_posix()

    def __repr__(self) -> str:
        return f"CrossPlatformPath({self._original!r}, platform={self.platform!r})"


class PathResolver:
    """Resolves paths across multiple machines and mount points.

    Handles path conversions for:
    - Local Mac filesystem
    - Windows machine via SSH
    - Windows machine via network mount
    - Remote cloud instances
    """

    def __init__(self):
        """Initialize path resolver."""
        self.local_platform = self._detect_platform()
        self.mounts: dict[str, MountPoint] = {}
        self.remote_hosts: dict[str, RemoteHost] = {}

        # Load configuration
        self._load_config()

    def _detect_platform(self) -> Platform:
        """Detect current platform."""
        import platform

        system = platform.system().lower()
        if system == "darwin":
            return "mac"
        elif system == "windows":
            return "windows"
        else:
            return "linux"

    def _load_config(self) -> None:
        """Load mount points and remote hosts from config."""
        import tomllib

        config_path = Path.home() / ".config" / "hafs" / "sync.toml"
        if not config_path.exists():
            logger.debug(f"No sync config found at {config_path}")
            return

        try:
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            # Load mounts
            for mount_name, mount_config in config.get("mounts", {}).items():
                local = Path(mount_config["local_path"]).expanduser()
                remote = mount_config["remote_path"]

                # Check if accessible
                accessible = local.exists() and local.is_dir()

                self.mounts[mount_name] = MountPoint(
                    local_path=local,
                    remote_path=remote,
                    is_accessible=accessible,
                )

                if accessible:
                    logger.debug(f"Mount {mount_name}: {local} -> {remote}")
                else:
                    logger.warning(f"Mount {mount_name} not accessible: {local}")

            # Load remote hosts
            for host_name, host_config in config.get("hosts", {}).items():
                self.remote_hosts[host_name] = RemoteHost(
                    hostname=host_config["hostname"],
                    username=host_config.get("username", "Administrator"),
                    platform=host_config.get("platform", "windows"),
                    ssh_key=host_config.get("ssh_key"),
                )

                logger.debug(f"Remote host {host_name}: {self.remote_hosts[host_name]}")

        except Exception as e:
            logger.error(f"Failed to load sync config: {e}")

    def resolve_path(
        self, path: Union[str, Path], platform: Platform
    ) -> CrossPlatformPath:
        """Create a cross-platform path object.

        Args:
            path: Path on the specified platform
            platform: Platform where path is valid

        Returns:
            CrossPlatformPath object for conversions
        """
        return CrossPlatformPath(path, platform)

    def get_mount_for_path(self, remote_path: str) -> Optional[MountPoint]:
        """Find mount point for a remote path.

        Args:
            remote_path: Path on remote machine

        Returns:
            MountPoint if found and accessible, None otherwise
        """
        for mount in self.mounts.values():
            if mount.is_accessible and remote_path.startswith(mount.remote_path):
                return mount

        return None

    def remote_to_local(
        self, remote_path: str, remote_host: Optional[str] = None
    ) -> Optional[Path]:
        """Convert remote path to local mount path if available.

        Args:
            remote_path: Path on remote machine (e.g., D:/hafs_training/models)
            remote_host: Optional remote host name

        Returns:
            Local mount path if available, None otherwise
        """
        mount = self.get_mount_for_path(remote_path)
        if not mount:
            return None

        # Calculate relative path
        rel_path = remote_path[len(mount.remote_path) :].lstrip("/\\")

        # Construct local path
        local_path = mount.local_path / rel_path

        return local_path

    def build_ssh_command(
        self, remote_host: str, command: str, working_dir: Optional[str] = None
    ) -> list[str]:
        """Build SSH command with proper escaping.

        Args:
            remote_host: Remote host name (from config)
            command: Command to run
            working_dir: Optional working directory on remote

        Returns:
            SSH command as list (suitable for subprocess)
        """
        host = self.remote_hosts.get(remote_host)
        if not host:
            raise ValueError(f"Remote host {remote_host} not configured")

        # Build SSH command
        ssh_cmd = ["ssh"]

        if host.ssh_key:
            ssh_cmd.extend(["-i", host.ssh_key])

        ssh_cmd.append(str(host))

        # Add working directory if specified
        if working_dir:
            cp_path = CrossPlatformPath(working_dir, host.platform)
            ssh_path = cp_path.to_ssh_path()
            command = f"cd {ssh_path} && {command}"

        ssh_cmd.append(command)

        return ssh_cmd

    def build_scp_command(
        self,
        source: str,
        dest: str,
        remote_host: Optional[str] = None,
        recursive: bool = False,
    ) -> list[str]:
        """Build SCP command for file transfer.

        Args:
            source: Source path (local or remote:path)
            dest: Destination path (local or remote:path)
            remote_host: Remote host name if using remote paths
            recursive: Use recursive copy

        Returns:
            SCP command as list
        """
        scp_cmd = ["scp"]

        if recursive:
            scp_cmd.append("-r")

        # Add SSH key if configured
        if remote_host:
            host = self.remote_hosts.get(remote_host)
            if host and host.ssh_key:
                scp_cmd.extend(["-i", host.ssh_key])

        scp_cmd.extend([source, dest])

        return scp_cmd

    def check_remote_path(self, remote_host: str, remote_path: str) -> bool:
        """Check if a path exists on remote host.

        Args:
            remote_host: Remote host name
            remote_path: Path to check on remote

        Returns:
            True if path exists, False otherwise
        """
        host = self.remote_hosts.get(remote_host)
        if not host:
            logger.error(f"Remote host {remote_host} not configured")
            return False

        # Build appropriate test command based on platform
        cp_path = CrossPlatformPath(remote_path, host.platform)
        ssh_path = cp_path.to_ssh_path()

        if host.platform == "windows":
            # Use PowerShell Test-Path
            command = f'powershell "Test-Path {ssh_path}"'
        else:
            command = f'test -e "{ssh_path}"'

        ssh_cmd = self.build_ssh_command(remote_host, command)

        try:
            result = subprocess.run(
                ssh_cmd, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to check remote path: {e}")
            return False

    def get_windows_path_escaped(self, path: str) -> str:
        """Helper to escape Windows paths for SSH commands.

        Args:
            path: Windows path (C:/foo or C:\\foo)

        Returns:
            Properly escaped path for SSH
        """
        cp_path = CrossPlatformPath(path, "windows")
        return cp_path.to_ssh_path()


# Singleton instance
_path_resolver: Optional[PathResolver] = None


def get_path_resolver() -> PathResolver:
    """Get global path resolver instance."""
    global _path_resolver
    if _path_resolver is None:
        _path_resolver = PathResolver()
    return _path_resolver
