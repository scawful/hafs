"""Context Summary widget for quick overview of loaded context and KB stats."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Label, Static, ProgressBar
from textual.reactive import reactive
from textual.message import Message


class ContextSummaryWidget(Container):
    """Compact widget showing context and knowledge base overview."""

    DEFAULT_CSS = """
    ContextSummaryWidget {
        height: auto;
        min-height: 8;
        max-height: 20;
        padding: 0 1;
        background: $surface;
    }

    ContextSummaryWidget .section-header {
        background: $primary-darken-2;
        padding: 0 1;
        margin-bottom: 0;
    }

    ContextSummaryWidget .stat-row {
        height: 1;
        padding: 0 1;
    }

    ContextSummaryWidget .stat-label {
        width: 12;
        color: $text-muted;
    }

    ContextSummaryWidget .stat-value {
        width: 1fr;
        text-align: right;
    }

    ContextSummaryWidget .kb-progress {
        height: 1;
        margin: 0 1;
    }

    ContextSummaryWidget .recent-item {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    ContextSummaryWidget .recent-item:hover {
        background: $primary-darken-3;
        color: $text;
    }
    """

    # Reactive stats
    total_symbols: reactive[int] = reactive(0)
    total_embeddings: reactive[int] = reactive(0)
    embedding_coverage: reactive[float] = reactive(0.0)
    active_kbs: reactive[list] = reactive(list)
    recent_files: reactive[list] = reactive(list)

    class KBSelected(Message):
        """Emitted when a KB is selected."""
        def __init__(self, kb_name: str) -> None:
            self.kb_name = kb_name
            super().__init__()

    class FileSelected(Message):
        """Emitted when a recent file is selected."""
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Label("[bold]Context[/]", classes="section-header")

        with Vertical(id="context-stats"):
            yield Static(id="kb-status", classes="stat-row")
            yield ProgressBar(id="embedding-progress", total=100, show_eta=False, classes="kb-progress")
            yield Static(id="symbols-count", classes="stat-row")
            yield Static(id="embeddings-count", classes="stat-row")

        yield Label("[bold]Recent[/]", classes="section-header")
        with Vertical(id="recent-files-list"):
            yield Static("[dim]Loading...[/]", id="recent-placeholder")

    async def on_mount(self) -> None:
        """Load initial data."""
        await self.refresh_stats()

    async def refresh_stats(self) -> None:
        """Refresh all stats from knowledge bases and history."""
        # Load KB stats
        try:
            from hafs.agents.alttp_knowledge import ALTTPKnowledgeBase

            kb = ALTTPKnowledgeBase()
            await kb.setup()
            stats = kb.get_statistics()

            self.total_symbols = stats.get("symbols", 0)
            self.total_embeddings = stats.get("with_embeddings", 0)

            if self.total_symbols > 0:
                self.embedding_coverage = (self.total_embeddings / self.total_symbols) * 100

            self.active_kbs = ["alttp", "oracle-of-secrets"]
        except Exception:
            self.active_kbs = []

        # Load recent history
        try:
            from hafs.core.history.logger import HistoryLogger
            from hafs.core.history.models import HistoryQuery, OperationType

            history_dir = Path.home() / ".context" / "history"
            if history_dir.exists():
                logger = HistoryLogger(history_dir)
                recent = logger.query(HistoryQuery(
                    operation_types=[OperationType.TOOL_CALL],
                    limit=5
                ))

                self.recent_files = [
                    entry.operation.input.get("path", entry.operation.name)
                    for entry in recent
                    if entry.operation.input
                ]
        except Exception:
            self.recent_files = []

        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current stats."""
        try:
            # KB status
            kb_count = len(self.active_kbs)
            kb_names = ", ".join(self.active_kbs[:3]) if self.active_kbs else "None"
            self.query_one("#kb-status", Static).update(
                f"[dim]KBs:[/] {kb_count} active ({kb_names})"
            )

            # Embedding progress
            progress = self.query_one("#embedding-progress", ProgressBar)
            progress.update(progress=self.embedding_coverage)

            # Symbols
            self.query_one("#symbols-count", Static).update(
                f"[dim]Symbols:[/] {self.total_symbols:,}"
            )

            # Embeddings
            self.query_one("#embeddings-count", Static).update(
                f"[dim]Embeddings:[/] {self.total_embeddings:,} ({self.embedding_coverage:.1f}%)"
            )

            # Recent files
            recent_container = self.query_one("#recent-files-list", Vertical)
            placeholder = self.query_one("#recent-placeholder", Static)

            if self.recent_files:
                placeholder.update("")
                # We'd ideally add dynamic children here, but keep it simple
                recent_text = "\n".join(
                    f"  [dim]•[/] {Path(f).name if isinstance(f, str) else f}"
                    for f in self.recent_files[:5]
                )
                placeholder.update(recent_text or "[dim]No recent files[/]")
            else:
                placeholder.update("[dim]No recent files[/]")
        except Exception:
            pass

    def watch_total_symbols(self, value: int) -> None:
        """React to symbol count changes."""
        self._update_display()

    def watch_total_embeddings(self, value: int) -> None:
        """React to embedding count changes."""
        self._update_display()
