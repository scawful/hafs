"""Network Inventory Agent - remote filesystem analysis via SSH.

Queries remote systems (Mac, halext-server) for filesystem information
using read-only SSH commands. Generates cross-system consolidation reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.background.base import BackgroundAgent

logger = logging.getLogger(__name__)


class NetworkInventoryAgent(BackgroundAgent):
    """Network inventory agent for remote system analysis.

    Connects to remote systems via SSH and runs read-only commands to gather
    filesystem information for consolidation planning.
    """

    def __init__(self, config_path: str | Path | None = None, verbose: bool = False):
        """Initialize network inventory agent."""
        super().__init__(config_path, verbose)
        self.remote_systems = self.config.tasks.get("remote_systems", [])
        self.remote_commands = self.config.tasks.get("remote_commands", [])
        self.network_paths = self.config.tasks.get("network_paths", [])

    def run(self) -> dict[str, Any]:
        """Execute network inventory.

        Returns:
            Dictionary with network inventory results
        """
        results = {
            "scan_timestamp": datetime.now().isoformat(),
            "remote_systems": [],
            "network_mounts": [],
            "connectivity_status": {},
        }

        # Check network mounts
        for network_path in self.network_paths:
            path = Path(network_path)
            if path.exists():
                logger.info(f"Network mount available: {network_path}")
                results["network_mounts"].append(
                    {"path": network_path, "status": "available"}
                )
            else:
                logger.warning(f"Network mount not available: {network_path}")
                results["network_mounts"].append(
                    {"path": network_path, "status": "unavailable"}
                )

        # Query remote systems
        for system in self.remote_systems:
            system_name = system.get("name", "unknown")
            logger.info(f"Querying remote system: {system_name}")

            system_info = self._query_remote_system(system)
            results["remote_systems"].append(system_info)
            results["connectivity_status"][system_name] = system_info["status"]

        # Save results
        self._save_output(results, "network_inventory")

        # Generate summary
        summary = self._generate_summary(results)
        self._save_output(summary, "network_summary")

        return results

    def _query_remote_system(self, system: dict[str, Any]) -> dict[str, Any]:
        """Query a remote system via SSH.

        Args:
            system: System configuration

        Returns:
            System inventory data
        """
        system_name = system.get("name", "unknown")
        host = system.get("host")
        paths = system.get("paths", [])
        ssh_key = system.get("ssh_key")

        system_info = {
            "name": system_name,
            "host": host,
            "status": "unknown",
            "paths": [],
            "total_size_gb": 0.0,
            "total_files": 0,
            "largest_files": [],
        }

        # Test connectivity
        if not self._test_ssh_connection(host, ssh_key):
            system_info["status"] = "unreachable"
            logger.warning(f"Cannot reach {system_name} at {host}")
            return system_info

        system_info["status"] = "reachable"

        # Query each path
        for path in paths:
            path_info = self._query_remote_path(host, path, ssh_key)
            system_info["paths"].append(path_info)

            # Aggregate totals
            system_info["total_size_gb"] += path_info.get("size_gb", 0.0)
            system_info["total_files"] += path_info.get("file_count", 0)

        # Get largest files
        largest_files = self._get_largest_files(host, paths, ssh_key)
        system_info["largest_files"] = largest_files

        return system_info

    def _test_ssh_connection(
        self, host: str, ssh_key: Optional[str] = None
    ) -> bool:
        """Test SSH connectivity to remote host.

        Args:
            host: SSH host string (user@hostname)
            ssh_key: Path to SSH private key

        Returns:
            True if connection succeeds
        """
        cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]

        if ssh_key:
            cmd.extend(["-i", ssh_key])

        cmd.extend([host, "echo", "ok"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"SSH connection test failed: {e}")
            return False

    def _query_remote_path(
        self, host: str, path: str, ssh_key: Optional[str] = None
    ) -> dict[str, Any]:
        """Query remote path for statistics.

        Args:
            host: SSH host
            path: Remote path to query
            ssh_key: SSH private key path

        Returns:
            Path statistics
        """
        path_info = {
            "path": path,
            "size_gb": 0.0,
            "file_count": 0,
            "status": "unknown",
        }

        # Get disk usage
        du_cmd = f"du -sb {path} 2>/dev/null || echo 0"
        du_output = self._run_ssh_command(host, du_cmd, ssh_key)

        if du_output:
            try:
                size_bytes = int(du_output.split()[0])
                path_info["size_gb"] = size_bytes / (1024**3)
                path_info["status"] = "success"
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse du output for {path}")

        # Get file count
        count_cmd = f"find {path} -type f 2>/dev/null | wc -l"
        count_output = self._run_ssh_command(host, count_cmd, ssh_key)

        if count_output:
            try:
                path_info["file_count"] = int(count_output.strip())
            except ValueError:
                logger.warning(f"Failed to parse file count for {path}")

        return path_info

    def _get_largest_files(
        self, host: str, paths: list[str], ssh_key: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get largest files from remote paths.

        Args:
            host: SSH host
            paths: List of paths to search
            ssh_key: SSH private key path

        Returns:
            List of largest files
        """
        largest_files = []

        for path in paths:
            # Find largest files (top 10)
            find_cmd = (
                f"find {path} -type f -exec ls -lh {{}} \\; 2>/dev/null | "
                f"awk '{{print $5, $9}}' | sort -hr | head -10"
            )

            output = self._run_ssh_command(host, find_cmd, ssh_key)

            if output:
                for line in output.strip().split("\n"):
                    if not line:
                        continue

                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        size_str, file_path = parts
                        largest_files.append({"size": size_str, "path": file_path})

        return largest_files[:20]  # Top 20 overall

    def _run_ssh_command(
        self, host: str, command: str, ssh_key: Optional[str] = None
    ) -> str:
        """Run command on remote host via SSH.

        Args:
            host: SSH host
            command: Command to run
            ssh_key: SSH private key path

        Returns:
            Command output or empty string on error
        """
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes"]

        if ssh_key:
            cmd.extend(["-i", ssh_key])

        cmd.extend([host, command])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                logger.debug(
                    f"SSH command failed: {command} (exit code: {result.returncode})"
                )
                return ""

        except Exception as e:
            logger.debug(f"SSH command error: {e}")
            return ""

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary report.

        Args:
            results: Full network inventory

        Returns:
            Summary dictionary
        """
        total_remote_size = sum(
            s.get("total_size_gb", 0) for s in results["remote_systems"]
        )

        total_remote_files = sum(
            s.get("total_files", 0) for s in results["remote_systems"]
        )

        reachable_systems = [
            s["name"]
            for s in results["remote_systems"]
            if s["status"] == "reachable"
        ]

        return {
            "timestamp": results["scan_timestamp"],
            "remote_systems_total": len(results["remote_systems"]),
            "reachable_systems": len(reachable_systems),
            "reachable_system_names": reachable_systems,
            "network_mounts_available": len(
                [m for m in results["network_mounts"] if m["status"] == "available"]
            ),
            "total_remote_size_gb": round(total_remote_size, 2),
            "total_remote_files": total_remote_files,
            "summary": f"Queried {len(results['remote_systems'])} remote systems. "
            f"{len(reachable_systems)} reachable with {total_remote_size:.1f} GB "
            f"across {total_remote_files:,} files.",
        }


def main():
    """CLI entry point for network inventory agent."""
    parser = argparse.ArgumentParser(description="hafs Network Inventory Agent")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    agent = NetworkInventoryAgent(config_path=args.config, verbose=args.verbose)
    result = agent.execute()

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
