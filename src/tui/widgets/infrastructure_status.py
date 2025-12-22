"""Infrastructure status widget for nodes and AFS sync."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label
from textual.timer import Timer

from core.nodes import node_manager


class InfrastructureStatusWidget(Container):
    """Show node health and AFS sync status."""

    DEFAULT_CSS = """
    InfrastructureStatusWidget {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    InfrastructureStatusWidget .section-header {
        background: $primary;
        padding: 0 1;
        height: 1;
    }

    InfrastructureStatusWidget DataTable {
        height: auto;
        max-height: 14;
    }
    """

    def __init__(self, refresh_seconds: float = 30.0, id: str | None = None) -> None:
        super().__init__(id=id)
        self._refresh_seconds = refresh_seconds
        self._refresh_timer: Timer | None = None
        self._sync_status_file = Path.home() / ".context" / "metrics" / "afs_sync_status.json"

    def compose(self) -> ComposeResult:
        yield Label("[bold]Nodes[/]", classes="section-header")
        yield DataTable(id="infra-nodes", zebra_stripes=True)
        yield Label("[bold]AFS Sync[/]", classes="section-header")
        yield DataTable(id="infra-sync", zebra_stripes=True)

    async def on_mount(self) -> None:
        nodes_table = self.query_one("#infra-nodes", DataTable)
        nodes_table.add_columns("Name", "Status", "Host", "Type", "Latency")

        sync_table = self.query_one("#infra-sync", DataTable)
        sync_table.add_columns("Profile", "Target", "Status", "Last Seen", "Direction")

        await self.refresh_status()
        self._refresh_timer = self.set_interval(self._refresh_seconds, self.refresh_status)

    def on_unmount(self) -> None:
        if self._refresh_timer:
            self._refresh_timer.stop()

    async def refresh_status(self) -> None:
        await self._refresh_nodes()
        self._refresh_sync()

    async def _refresh_nodes(self) -> None:
        table = self.query_one("#infra-nodes", DataTable)
        table.clear()

        try:
            await node_manager.load_config()
            await node_manager.health_check_all()
        except Exception as exc:
            table.add_row("error", "unknown", str(exc), "", "")
            return

        if not node_manager.nodes:
            table.add_row("none", "unknown", "", "", "")
            return

        for node in node_manager.nodes:
            status = node.status.value
            latency = f"{node.latency_ms}ms" if node.latency_ms else "-"
            table.add_row(
                node.name,
                status,
                f"{node.host}:{node.port}",
                node.node_type,
                latency,
            )

    def _refresh_sync(self) -> None:
        table = self.query_one("#infra-sync", DataTable)
        table.clear()

        if not self._sync_status_file.exists():
            table.add_row("none", "-", "unknown", "-", "-")
            return

        try:
            data = json.loads(self._sync_status_file.read_text())
        except Exception as exc:
            table.add_row("error", "-", "unknown", str(exc), "-")
            return

        profiles = data.get("profiles", {})
        rows = 0
        for profile_name, profile_data in profiles.items():
            targets = profile_data.get("targets", {})
            for target_name, record in targets.items():
                status = "ok" if record.get("ok") else "fail"
                table.add_row(
                    profile_name,
                    target_name,
                    status,
                    record.get("timestamp", "-"),
                    record.get("direction", "-"),
                )
                rows += 1

        if rows == 0:
            table.add_row("none", "-", "unknown", "-", "-")
