"""SSH utilities for remote server access.

Provides an SSH client for running commands on remote systems. Based on
NetworkInventoryAgent patterns.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SSHClient:
    """SSH client for remote access.

    Provides methods for running commands, reading files, and testing
    connectivity on remote systems via SSH.
    """

    def __init__(
        self,
        host: str = "user@host",
        key_path: Optional[str] = None,
        timeout: int = 30,
    ):
        """Initialize SSH client.

        Args:
            host: SSH host string (user@hostname)
            key_path: Path to SSH private key (defaults to ~/.ssh/id_ed25519)
            timeout: Command timeout in seconds
        """
        self.host = host
        self.key_path = key_path or str(Path.home() / ".ssh" / "id_ed25519")
        self.timeout = timeout

    def test_connection(self) -> bool:
        """Test SSH connectivity to remote host.

        Returns:
            True if connection succeeds
        """
        cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "BatchMode=yes",
            "-i",
            self.key_path,
            self.host,
            "echo",
            "ok",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            success = result.returncode == 0
            if success:
                logger.debug(f"SSH connection to {self.host} successful")
            else:
                logger.warning(f"SSH connection to {self.host} failed: {result.stderr}")
            return success

        except Exception as e:
            logger.debug(f"SSH connection test failed: {e}")
            return False

    def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        """Run command on remote host via SSH.

        Args:
            command: Command to run
            timeout: Command timeout (uses self.timeout if not specified)

        Returns:
            Command output (stdout)

        Raises:
            subprocess.CalledProcessError: If command fails
            subprocess.TimeoutExpired: If command times out
        """
        if timeout is None:
            timeout = self.timeout

        cmd = [
            "ssh",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-i",
            self.key_path,
            self.host,
            command,
        ]

        logger.debug(f"Running SSH command: {command}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )

        return result.stdout

    def run_command_safe(
        self, command: str, timeout: Optional[int] = None
    ) -> tuple[bool, str]:
        """Run command and return success status with output.

        Args:
            command: Command to run
            timeout: Command timeout

        Returns:
            Tuple of (success, output)
        """
        try:
            output = self.run_command(command, timeout)
            return (True, output)
        except Exception as e:
            logger.debug(f"SSH command failed: {e}")
            return (False, str(e))

    def read_file(self, remote_path: str) -> str:
        """Read file contents from remote server.

        Args:
            remote_path: Path to file on remote server

        Returns:
            File contents

        Raises:
            subprocess.CalledProcessError: If file read fails
        """
        command = f"cat {remote_path}"
        return self.run_command(command)

    def list_directory(self, remote_path: str) -> list[str]:
        """List directory contents.

        Args:
            remote_path: Path to directory on remote server

        Returns:
            List of file/directory names
        """
        command = f"ls -1 {remote_path}"
        output = self.run_command(command)
        return [line.strip() for line in output.strip().split("\n") if line.strip()]

    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists on remote server.

        Args:
            remote_path: Path to check

        Returns:
            True if file exists
        """
        command = f"test -f {remote_path} && echo 1 || echo 0"
        output = self.run_command(command).strip()
        return output == "1"

    def directory_exists(self, remote_path: str) -> bool:
        """Check if directory exists on remote server.

        Args:
            remote_path: Path to check

        Returns:
            True if directory exists
        """
        command = f"test -d {remote_path} && echo 1 || echo 0"
        output = self.run_command(command).strip()
        return output == "1"

    def git_log(
        self,
        repo_path: str,
        since: str = "24 hours ago",
        format_str: str = "%H|%an|%s|%ai",
    ) -> list[dict[str, str]]:
        """Get git log from remote repository.

        Args:
            repo_path: Path to git repository
            since: Time period (e.g., "24 hours ago", "1 week ago")
            format_str: Git log format string

        Returns:
            List of commit dictionaries with keys: hash, author, subject, date
        """
        command = (
            f"cd {repo_path} && "
            f"git log --since='{since}' --pretty=format:'{format_str}'"
        )

        output = self.run_command(command).strip()

        if not output:
            return []

        commits = []
        for line in output.split("\n"):
            if not line:
                continue

            parts = line.split("|")
            if len(parts) == 4:
                commits.append(
                    {
                        "hash": parts[0],
                        "author": parts[1],
                        "subject": parts[2],
                        "date": parts[3],
                    }
                )

        return commits

    def tail_file(self, remote_path: str, lines: int = 100) -> str:
        """Read last N lines of a file.

        Args:
            remote_path: Path to file
            lines: Number of lines to read

        Returns:
            File tail content
        """
        command = f"tail -n {lines} {remote_path}"
        return self.run_command(command)

    def disk_usage(self, remote_path: str) -> tuple[int, str]:
        """Get disk usage for path.

        Args:
            remote_path: Path to check

        Returns:
            Tuple of (bytes, human_readable_string)
        """
        # Get bytes
        command_bytes = f"du -sb {remote_path} 2>/dev/null | awk '{{print $1}}'"
        bytes_output = self.run_command(command_bytes).strip()

        # Get human-readable
        command_human = f"du -sh {remote_path} 2>/dev/null | awk '{{print $1}}'"
        human_output = self.run_command(command_human).strip()

        try:
            size_bytes = int(bytes_output)
        except ValueError:
            size_bytes = 0

        return (size_bytes, human_output)
