"""History embeddings search widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, ListItem, ListView, Static

from config.loader import load_config
from core.history import HistoryEmbeddingIndex, HistorySessionSummaryIndex


class HistoryResultItem(ListItem):
    """List item for history search results."""

    DEFAULT_CSS = """
    HistoryResultItem {
        height: 3;
        padding: 0 1;
    }

    HistoryResultItem:hover {
        background: $surface;
    }
    """

    def __init__(self, label: str, result: dict[str, Any], kind: str) -> None:
        super().__init__()
        self.label = label
        self.result = result
        self.kind = kind

    def compose(self) -> ComposeResult:
        yield Static(self.label)


class HistorySearchView(Widget):
    """Search view for semantic history embeddings."""

    DEFAULT_CSS = """
    HistorySearchView {
        layout: vertical;
        height: 100%;
        width: 100%;
    }

    HistorySearchView .search-bar {
        height: auto;
        background: $surface;
        border-bottom: solid $primary;
        padding: 0 1;
    }

    HistorySearchView #history-search-input {
        width: 1fr;
    }

    HistorySearchView #history-status {
        height: auto;
        padding: 0 1;
        color: $text-disabled;
    }

    HistorySearchView .results-container {
        layout: horizontal;
        height: 1fr;
        width: 100%;
    }

    HistorySearchView .list-panel {
        width: 40%;
        min-width: 30;
        border-right: solid $primary;
    }

    HistorySearchView .detail-panel {
        width: 60%;
        padding: 1 1;
    }

    HistorySearchView #history-results-list {
        height: 100%;
    }
    """

    def __init__(self, id: str | None = None, classes: str | None = None) -> None:
        super().__init__(id=id, classes=classes)
        config = load_config()
        self._index = HistoryEmbeddingIndex(config.general.context_root)
        self._summary_index = HistorySessionSummaryIndex(config.general.context_root)
        self._mode = "entries"

    def compose(self) -> ComposeResult:
        with Container(classes="search-bar"):
            with Horizontal():
                yield Input(
                    placeholder="Search AFS history...",
                    id="history-search-input",
                )
                yield Button("Index New", id="history-index-btn", variant="primary")
                yield Button("Mode: Entries", id="history-mode-btn")
        yield Static("", id="history-status")
        with Horizontal(classes="results-container"):
            with Container(classes="list-panel"):
                yield ListView(id="history-results-list")
            with VerticalScroll(classes="detail-panel"):
                yield Static("", id="history-detail")

    def on_mount(self) -> None:
        self._refresh_status()
        self._show_info("Type a query and press Enter to search.")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "history-search-input":
            return
        query = event.value.strip()
        if not query:
            return
        self._set_status("Searching history embeddings...")
        self.run_worker(self._run_search(query), exclusive=True, group="history-search")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "history-index-btn":
            if self._mode == "entries":
                self._set_status("Indexing new history entries...")
                self.run_worker(self._run_index(), exclusive=True, group="history-index")
            elif self._mode == "sessions":
                self._set_status("Summarizing sessions...")
                self.run_worker(self._run_summaries(), exclusive=True, group="history-index")
            else:
                self._set_status("Indexing entries and session summaries...")
                self.run_worker(self._run_all_indexes(), exclusive=True, group="history-index")
        elif event.button.id == "history-mode-btn":
            self._toggle_mode()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, HistoryResultItem):
            self._set_detail(event.item.result, event.item.kind)

    async def _run_index(self) -> None:
        created = await self._index.index_new_entries(limit=200)
        self._refresh_status()
        self._show_info(f"Indexed {created} new entries.")

    async def _run_summaries(self) -> None:
        created = await self._summary_index.index_missing_summaries(limit=50)
        self._refresh_status()
        self._show_info(f"Created {created} session summaries.")

    async def _run_all_indexes(self) -> None:
        created_entries = await self._index.index_new_entries(limit=200)
        created_summaries = await self._summary_index.index_missing_summaries(limit=50)
        self._refresh_status()
        self._show_info(
            f"Indexed {created_entries} entries and {created_summaries} summaries."
        )

    async def _run_search(self, query: str) -> None:
        if self._mode == "entries":
            results = await self._index.search(query, limit=10)
            results = [{"kind": "entry", **result} for result in results]
        elif self._mode == "sessions":
            results = await self._summary_index.search(query, limit=10)
            results = [{"kind": "session", **result} for result in results]
        else:
            entry_results = await self._index.search(query, limit=10)
            session_results = await self._summary_index.search(query, limit=10)
            results = (
                [{"kind": "entry", **result} for result in entry_results]
                + [{"kind": "session", **result} for result in session_results]
            )
            results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            results = results[:10]

        if not results:
            self._show_info("No matches found.")
            self._set_status("No matches.")
            return

        self._set_results(results)
        self._set_status(f"Found {len(results)} results.")

    def _refresh_status(self) -> None:
        status = self._index.status()
        summaries = self._summary_index.status()
        self._set_status(
            f"Mode: {self._mode} | History files: {status['history_files']} | "
            f"Entry embeddings: {status['embeddings']} | Session summaries: {summaries['summaries']}"
        )

    def _toggle_mode(self) -> None:
        order = ["entries", "sessions", "all"]
        current = order.index(self._mode)
        self._mode = order[(current + 1) % len(order)]
        button = self.query_one("#history-mode-btn", Button)
        label_map = {
            "entries": "Mode: Entries",
            "sessions": "Mode: Sessions",
            "all": "Mode: All",
        }
        button.label = label_map[self._mode]
        self._refresh_status()
        self._show_info(f"Switched to {self._mode} mode.")

    def _set_results(self, results: list[dict[str, Any]]) -> None:
        list_view = self.query_one("#history-results-list", ListView)
        list_view.clear()
        if not results:
            list_view.append(ListItem(Static("[dim]No matches found[/dim]")))
            self._set_detail_text("No matches found.")
            return

        for result in results:
            kind = result.get("kind", "entry")
            label = self._format_label(result, kind)
            list_view.append(HistoryResultItem(label, result, kind))

        list_view.index = 0
        first = results[0]
        self._set_detail(first, first.get("kind", "entry"))

    def _format_label(self, result: dict[str, Any], kind: str) -> str:
        score = result.get("score", 0.0)
        if kind == "session":
            created_at = result.get("created_at")
            session_id = result.get("session_id")
            title = result.get("title") or "Session summary"
            return f"[S] [{score:.2f}] {created_at} {session_id} {title}"

        timestamp = result.get("timestamp")
        session_id = result.get("session_id")
        op_type = result.get("operation_type")
        name = result.get("name")
        return f"[E] [{score:.2f}] {timestamp} {session_id} {op_type}/{name}"

    def _show_info(self, text: str) -> None:
        list_view = self.query_one("#history-results-list", ListView)
        list_view.clear()
        list_view.append(ListItem(Static(text)))
        self._set_detail_text(text)

    def _set_detail(self, result: dict[str, Any], kind: str) -> None:
        detail = self.query_one("#history-detail", Static)
        if kind == "session":
            created_at = result.get("created_at")
            session_id = result.get("session_id")
            title = result.get("title") or "Session summary"
            summary = result.get("summary", "")
            topics = result.get("topics", [])
            decisions = result.get("decisions", [])
            lines = [
                f"Session: {session_id}",
                f"Created: {created_at}",
                f"Title: {title}",
            ]
            if topics:
                lines.append(f"Topics: {', '.join(topics)}")
            if decisions:
                lines.append(f"Decisions: {', '.join(decisions)}")
            lines.append("")
            lines.append(summary)
            detail.update("\n".join(lines).strip())
            return

        entry_id = result.get("entry_id") or result.get("id")
        timestamp = result.get("timestamp")
        session_id = result.get("session_id")
        op_type = result.get("operation_type")
        name = result.get("name")
        preview = result.get("preview", "")
        detail.update(
            "\n".join(
                [
                    f"Entry: {entry_id}",
                    f"Session: {session_id}",
                    f"Timestamp: {timestamp}",
                    f"Type: {op_type}/{name}",
                    "",
                    preview,
                ]
            ).strip()
        )

    def _set_detail_text(self, text: str) -> None:
        detail = self.query_one("#history-detail", Static)
        detail.update(text)

    def _set_status(self, text: str) -> None:
        status = self.query_one("#history-status", Static)
        status.update(text)
