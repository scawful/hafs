"""Reports widget for viewing background agent reports."""

import os
from pathlib import Path
from datetime import datetime

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import DataTable, Label
from textual.binding import Binding

class ReportsWidget(Container):
    """Widget for browsing agent reports."""
    
    DEFAULT_CSS = """
    ReportsWidget {
        layout: vertical;
        padding: 1;
        height: 100%;
    }
    """
    
    REPORTS_DIR = Path.home() / ".context" / "background_agent" / "reports"

    def compose(self) -> ComposeResult:
        yield Label("Agent Reports", classes="header")
        yield DataTable(id="reports_table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one("#reports_table", DataTable)
        table.add_columns("Date", "Topic", "Filename", "Size")
        self.load_reports()

    def load_reports(self) -> None:
        table = self.query_one("#reports_table", DataTable)
        table.clear()
        
        if not self.REPORTS_DIR.exists():
            table.add_row("", "No reports directory", str(self.REPORTS_DIR), "")
            return

        reports = sorted(list(self.REPORTS_DIR.glob("*.md")), key=os.path.getmtime, reverse=True)
        
        if not reports:
            table.add_row("", "No reports found", "", "")
            return

        for p in reports:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            # Parse topic from filename (Topic_Name_YYYYMMDD.md)
            name_parts = p.stem.split('_')
            # Heuristic: Topic is everything before the timestamp part (usually last 2 parts)
            if len(name_parts) > 2 and name_parts[-1].isdigit():
                topic = " ".join(name_parts[:-2]).replace("-", " ")
            else:
                topic = p.stem
            
            size = f"{p.stat().st_size} B"
            table.add_row(mtime, topic, p.name, size, key=str(p))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        if event.row_key.value:
            path = Path(event.row_key.value)
            if path.exists():
                # Signal the app to open this file
                self.app.post_message(self.ReportSelected(path))

    from textual.message import Message

    class ReportSelected(Message):
        """Message sent when a report is selected."""
        def __init__(self, path: Path):
            self.path = path
            super().__init__()
