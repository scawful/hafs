"""Development dashboard widget."""

from textual.app import ComposeResult
from textual.containers import Vertical, Container
from textual.widgets import Static, Input, DataTable, Label, Header
from hafs.core.tools import ToolRegistry

class DevDashboard(Container):
    """Development dashboard widget showing reviews and search."""

    DEFAULT_CSS = """
    DevDashboard {
        layout: vertical;
        padding: 1;
        overflow: auto;
    }
    
    DevDashboard Label {
        margin-top: 1;
        text-style: bold;
    }
    
    .header {
        text-align: center;
        text-style: bold;
        background: $accent;
        color: $text;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Development Dashboard", classes="header")
        
        yield Label("Active Reviews")
        yield DataTable(id="reviews_table", zebra_stripes=True)
        
        yield Label("Code Search")
        yield Input(placeholder="Search query...", id="search_input")
        yield DataTable(id="search_results", zebra_stripes=True)

    async def on_mount(self) -> None:
        """Load initial data."""
        table = self.query_one("#reviews_table", DataTable)
        table.add_columns("ID", "Title", "Status", "Author")
        
        search_table = self.query_one("#search_results", DataTable)
        search_table.add_columns("File", "Line", "Content")
        
        await self.load_reviews()

    async def load_reviews(self) -> None:
        """Load reviews from provider."""
        provider_cls = ToolRegistry.get_review_provider()
        if not provider_cls:
            table = self.query_one("#reviews_table", DataTable)
            table.add_row("No Review Provider configured", "", "", "")
            return
            
        try:
            provider = provider_cls()
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

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input."""
        provider_cls = ToolRegistry.get_search_provider()
        table = self.query_one("#search_results", DataTable)
        table.clear()
        
        if not provider_cls:
            table.add_row("No Search Provider configured", "", "")
            return
            
        try:
            provider = provider_cls()
            results = await provider.search(event.value)
            
            if not results:
                table.add_row("No results found", "", "")
                return

            for res in results:
                table.add_row(res.file, str(res.line), res.content)
        except Exception as e:
            table.add_row(f"Error: {e}", "", "")
