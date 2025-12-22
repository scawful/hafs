"""Development dashboard widget."""

import os
from pathlib import Path

from rapidfuzz import fuzz, process
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Input, DataTable, Label, TabbedContent, TabPane
from core.tools import ToolRegistry
from tui.widgets.context_viewer import ContextViewer
from tui.widgets.policy_summary import PolicySummary
from tui.widgets.stats_panel import StatsPanel
from tui.widgets.protocol_widget import ProtocolWidget
from tui.widgets.reports import ReportsWidget
from tui.widgets.agent_management import AgentManagementWidget
from tui.widgets.swarm_control import SwarmControlWidget
from tui.widgets.infrastructure_status import InfrastructureStatusWidget

class ReviewsWidget(Container):
    """Widget for displaying active reviews."""
    
    DEFAULT_CSS = """
    ReviewsWidget {
        layout: vertical;
        padding: 1;
        height: 100%;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Label("Active Reviews", classes="header", id="reviews-header")
        yield DataTable(id="reviews_table", zebra_stripes=True)

    async def on_mount(self) -> None:
        table = self.query_one("#reviews_table", DataTable)
        table.add_columns("ID", "Title", "Status", "Author")
        await self.load_reviews()

    async def load_reviews(self) -> None:
        provider_cls = ToolRegistry.get_review_provider()
        header = self.query_one("#reviews-header", Label)
        
        if not provider_cls:
            table = self.query_one("#reviews_table", DataTable)
            table.add_row("No Review Provider configured", "", "", "")
            header.update("Active Reviews (No Provider)")
            return
            
        try:
            provider = provider_cls()
            
            # Update header with provider name if available
            if hasattr(provider, "name"):
                header.update(f"Active Reviews ({provider.name.capitalize()})")
            
            reviews = await provider.get_reviews()

            
            table = self.query_one("#reviews_table", DataTable)
            table.clear()
            
            if not reviews:
                table.add_row("No active reviews", "", "", "")
                return

            for review in reviews:
                table.add_row(review.id, review.title, review.status, review.author)
        except Exception as e:
            table = self.query_one("#reviews_table", DataTable)
            table.clear()
            table.add_row(f"Error: {e}", "", "", "")


class SearchWidget(Container):
    """Widget for code search."""

    _INDEX_IGNORE_DIRS = {
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "dist",
        "build",
        ".next",
        ".idea",
        ".vscode",
    }

    _MAX_INDEX_FILES = 200_000
    _DEFAULT_LIMIT = 50
    _DEFAULT_THRESHOLD = 30
    
    DEFAULT_CSS = """
    SearchWidget {
        layout: vertical;
        padding: 1;
        height: 100%;
    }
    SearchWidget Input {
        margin-bottom: 1;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._indexed_root: Path | None = None
        self._indexed_files: list[str] = []
        self._indexed_files_lower: list[str] = []
        self._index_truncated: bool = False
    
    def compose(self) -> ComposeResult:
        yield Label("Code Search", classes="header", id="search-header")
        yield Input(placeholder="Search root folder (default: cwd)", id="search_root")
        yield Input(placeholder="Search query...", id="search_input")
        yield DataTable(id="search_results", zebra_stripes=True)

    def on_mount(self) -> None:
        try:
            self.query_one("#search_root", Input).value = str(Path.cwd())
        except Exception:
            pass
        search_table = self.query_one("#search_results", DataTable)
        search_table.add_columns("File", "Line", "Content")
        
        # Check provider
        provider_cls = ToolRegistry.get_search_provider()
        if provider_cls:
            provider = provider_cls()
            if hasattr(provider, "name"):
                 self.query_one("#search-header", Label).update(f"Code Search ({provider.name.capitalize()})")

    async def on_input_submitted(self, event: Input.Submitted) -> None:

        if event.input.id == "search_root":
            self.query_one("#search_input", Input).focus()
            return

        if event.input.id != "search_input":
            return

        query = event.value.strip()
        if not query:
            return

        provider_cls = ToolRegistry.get_search_provider()
        table = self.query_one("#search_results", DataTable)
        table.clear()
        
        if provider_cls:
            try:
                provider = provider_cls()
                try:
                    results = await provider.search(query, limit=self._DEFAULT_LIMIT)
                except TypeError:
                    results = await provider.search(query)
                
                if not results:
                    table.add_row("No results found", "", "")
                    return

                for res in results:
                    table.add_row(res.file, str(res.line), res.content)
            except Exception as e:
                table.add_row(f"Error: {e}", "", "")
            return
            
        try:
            root = self._resolve_search_root()
        except Exception as e:
            table.add_row(f"Error: {e}", "", "")
            return

        try:
            await self._ensure_file_index(root, table)
        except Exception as e:
            table.clear()
            table.add_row(f"Error: {e}", "", "")
            return
        if not self._indexed_files:
            table.clear()
            table.add_row("No files indexed", "", str(root))
            return

        matches = self._fuzzy_files(query)
        table.clear()
        if not matches:
            table.add_row("No matches found", "", str(root))
            return

        for rel_path, score in matches:
            table.add_row(rel_path, "", f"[dim]file match • score {score:.0f}[/dim]")

        if self._index_truncated:
            table.add_row(
                "[dim]Index truncated[/dim]",
                "",
                f"[dim]First {len(self._indexed_files)} files only[/dim]",
            )

    def _resolve_search_root(self) -> Path:
        raw = self.query_one("#search_root", Input).value.strip()
        root = Path.cwd() if not raw else Path(os.path.expandvars(os.path.expanduser(raw)))
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        else:
            root = root.resolve()

        if root.is_file():
            root = root.parent

        if not root.exists():
            raise ValueError(f"Root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"Root is not a directory: {root}")

        return root

    async def _ensure_file_index(self, root: Path, table: DataTable) -> None:
        if self._indexed_root == root:
            return

        show_hidden = False
        try:
            # Type safe access to config
            if hasattr(self.app, "config") and self.app.config:
                show_hidden = bool(getattr(self.app.config.general, "show_hidden_files", False))
        except Exception:
            show_hidden = False

        table.add_row("[dim]Indexing files…[/dim]", "", str(root))

        worker = self.run_worker(
            lambda: self._build_file_index(root, show_hidden=show_hidden),
            thread=True,
            exclusive=True,
            group="search-index",
        )

        try:
            files, truncated = await worker.wait()
        except Exception as exc:
            raise RuntimeError(f"Indexing failed: {exc}") from exc
        self._indexed_root = root
        self._indexed_files = files
        self._indexed_files_lower = [f.lower() for f in files]
        self._index_truncated = truncated

    def _build_file_index(self, root: Path, *, show_hidden: bool) -> tuple[list[str], bool]:
        files: list[str] = []
        truncated = False

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            # Prune directories for performance.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in self._INDEX_IGNORE_DIRS and (show_hidden or not d.startswith("."))
            ]

            for filename in filenames:
                if not show_hidden and filename.startswith("."):
                    continue

                path = Path(dirpath) / filename
                if not path.is_file():
                    continue

                try:
                    files.append(str(path.relative_to(root)))
                except Exception:
                    files.append(str(path))

                if len(files) >= self._MAX_INDEX_FILES:
                    truncated = True
                    return files, truncated

        return files, truncated

    def _fuzzy_files(self, query: str) -> list[tuple[str, float]]:
        if not query or not self._indexed_files_lower:
            return []

        results = process.extract(
            query.lower(),
            self._indexed_files_lower,
            scorer=fuzz.token_set_ratio,
            limit=self._DEFAULT_LIMIT,
            score_cutoff=self._DEFAULT_THRESHOLD,
        )

        matches: list[tuple[str, float]] = []
        for _match, score, idx in results:
            if 0 <= idx < len(self._indexed_files):
                matches.append((self._indexed_files[idx], float(score)))
        return matches


class DevDashboard(Container):
    """Tabbed development dashboard."""

    DEFAULT_CSS = """
    DevDashboard {
        height: 1fr;
    }
    
    /* Ensure content inside tabs takes full height */
    DevDashboard TabbedContent {
        height: 1fr;
    }

    DevDashboard ContentSwitcher {
        height: 1fr;
    }
    
    DevDashboard TabPane {
        height: 100%;
        padding: 0;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)

    def compose(self) -> ComposeResult:
        with TabbedContent(id="dev-dashboard-tabs", initial="tab-context"):
            # 1. Context Viewer
            with TabPane("Context", id="tab-context"):
                yield ContextViewer(id="context-viewer")

            # 1.5 Protocol helpers
            with TabPane("Protocol", id="tab-protocol"):
                yield ProtocolWidget(id="protocol-widget")

            # 2. Policies + Stats
            with TabPane("Status", id="tab-status"):
                with VerticalScroll():
                    yield PolicySummary(id="policy-summary")
                    yield StatsPanel(id="stats-panel")
                    yield InfrastructureStatusWidget(id="infra-status-widget")

            # 2.5 Reports
            with TabPane("Reports", id="tab-reports"):
                yield ReportsWidget(id="reports-widget")

            # 2.6 Agents
            with TabPane("Agents", id="tab-agent-management"):
                yield AgentManagementWidget(id="agent-management-widget")

            # 2.7 Swarm
            with TabPane("Swarm", id="tab-swarm-control"):
                yield SwarmControlWidget(id="swarm-control-widget")

            # Check for specialized tools
            dev_tools = ToolRegistry.get_dev_tools()
            review_tool = next((t for t in dev_tools if getattr(t, "category", None) == "reviews"), None)
            search_tool = next((t for t in dev_tools if getattr(t, "category", None) == "search"), None)
            
            # Filter out special ones for the generic loop
            generic_tools = [t for t in dev_tools if t not in (review_tool, search_tool)]

            # 3. Reviews (Custom or Default)
            if review_tool:
                with TabPane("Reviews", id="tab-reviews"):
                    yield review_tool().create_widget()
            else:
                with TabPane("Reviews", id="tab-reviews"):
                    yield ReviewsWidget(id="reviews-widget")

            # 4. Search (Custom or Default)
            if search_tool:
                with TabPane("Search", id="tab-search"):
                    yield search_tool().create_widget()
            else:
                with TabPane("Search", id="tab-search"):
                    yield SearchWidget(id="search-widget")

            # 5. Other Registered Plugins
            for tool_cls in generic_tools:
                tool = tool_cls()
                with TabPane(tool.name, id=f"tab-{tool.slug}"):
                    yield tool.create_widget()

    @property
    def active(self) -> str:
        """Proxy active tab id to the internal TabbedContent."""
        return self.query_one("#dev-dashboard-tabs", TabbedContent).active

    @active.setter
    def active(self, tab_id: str) -> None:
        self.query_one("#dev-dashboard-tabs", TabbedContent).active = tab_id
