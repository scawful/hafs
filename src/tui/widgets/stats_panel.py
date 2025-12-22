"""Stats panel widget for displaying aggregate statistics."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

from core.afs.discovery import discover_projects, get_project_stats
from core.parsers.registry import ParserRegistry


class StatsPanel(Widget):
    """Widget for displaying aggregate statistics."""

    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    .sp-row {
        height: 1;
    }

    .sp-label {
        color: $text-disabled;
        width: 20;
    }

    .sp-value {
        color: $secondary;
        text-style: bold;
        width: 10;
    }
    """

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)
        self._stats: dict[str, int | str] = {}

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        with Horizontal(classes="sp-row"):
            yield Static("Projects:", classes="sp-label")
            yield Static(str(self._stats.get("projects", 0)), classes="sp-value")
            yield Static("Mounts:", classes="sp-label")
            yield Static(str(self._stats.get("mounts", 0)), classes="sp-value")

        with Horizontal(classes="sp-row"):
            yield Static("Gemini Sessions:", classes="sp-label")
            yield Static(str(self._stats.get("gemini", 0)), classes="sp-value")
            yield Static("Claude Plans:", classes="sp-label")
            yield Static(str(self._stats.get("claude", 0)), classes="sp-value")

        with Horizontal(classes="sp-row"):
            yield Static("Antigravity:", classes="sp-label")
            yield Static(str(self._stats.get("antigravity", 0)), classes="sp-value")

    def refresh_data(self) -> None:
        """Refresh statistics."""
        # Project stats
        projects = discover_projects()
        project_stats = get_project_stats(projects)

        self._stats["projects"] = project_stats["total_projects"]
        self._stats["mounts"] = project_stats["total_mounts"]

        # Parser stats
        for parser_name in ["gemini", "claude", "antigravity"]:
            parser_class = ParserRegistry.get(parser_name)
            if parser_class:
                parser = parser_class()
                if parser.exists():
                    items = parser.parse(max_items=100)
                    self._stats[parser_name] = len(items)
                else:
                    self._stats[parser_name] = 0
            else:
                self._stats[parser_name] = 0

        self.refresh(recompose=True)

    def on_mount(self) -> None:
        """Load stats when mounted."""
        self.refresh_data()
