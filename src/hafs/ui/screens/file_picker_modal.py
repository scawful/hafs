"""Fuzzy file picker modal for HAFS TUI."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from hafs.core.search import fuzzy_autocomplete


class FilePickerModal(ModalScreen[Path | None]):
    """Modal for picking files/directories with fuzzy search.

    Shows a list of files matching the input as you type.
    """

    DEFAULT_CSS = """
    FilePickerModal {
        align: center middle;
    }

    FilePickerModal #dialog {
        width: 80;
        height: 24;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    FilePickerModal #title {
        width: 100%;
        text-align: center;
        padding-bottom: 1;
    }

    FilePickerModal #search-input {
        width: 100%;
        margin-bottom: 1;
    }

    FilePickerModal #results {
        height: 1fr;
        border: solid $secondary;
    }

    FilePickerModal #hint {
        height: 1;
        color: $text-disabled;
        text-align: center;
    }

    FilePickerModal .file-item {
        height: 1;
        padding: 0 1;
    }

    FilePickerModal .file-item:hover {
        background: $primary;
    }

    FilePickerModal .dir-icon {
        color: $primary;
    }

    FilePickerModal .file-icon {
        color: $secondary;
    }
    """

    def __init__(
        self,
        base_path: Path,
        prompt: str = "Select file or directory",
        allow_new: bool = True,
    ) -> None:
        """Initialize file picker.

        Args:
            base_path: Directory to search in.
            prompt: Title text.
            allow_new: Allow creating new files (typing a non-existent name).
        """
        super().__init__()
        self.base_path = base_path
        self.prompt = prompt
        self.allow_new = allow_new
        self._all_paths: list[Path] = []
        self._current_matches: list[Path] = []

    def compose(self) -> ComposeResult:
        """Compose the modal layout."""
        with Container(id="dialog"):
            yield Label(f"[bold]{self.prompt}[/bold]", id="title")
            yield Input(placeholder="Type to search...", id="search-input")
            yield ListView(id="results")
            hint = "[dim]Enter to select, Escape to cancel"
            if self.allow_new:
                hint += ", type new name to create[/dim]"
            else:
                hint += "[/dim]"
            yield Static(hint, id="hint")

    def on_mount(self) -> None:
        """Load files when mounted."""
        self._load_paths()
        self._update_results("")
        self.query_one("#search-input", Input).focus()

    def _load_paths(self) -> None:
        """Load all paths from base directory."""
        self._all_paths = []

        if not self.base_path.exists():
            return

        try:
            # Get all files and directories (non-hidden)
            for path in sorted(
                self.base_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            ):
                if not path.name.startswith("."):
                    self._all_paths.append(path)
        except PermissionError:
            pass

    def _update_results(self, query: str) -> None:
        """Update results list based on query."""
        results_view = self.query_one("#results", ListView)
        results_view.clear()

        if not query:
            # Show all items when no query
            self._current_matches = self._all_paths[:20]
        else:
            # Use fuzzy matching
            path_names = [p.name for p in self._all_paths]
            matches = fuzzy_autocomplete(query, path_names, limit=20, threshold=30)

            # Map back to paths
            name_to_path = {p.name: p for p in self._all_paths}
            self._current_matches = [
                name_to_path[name] for name, _score in matches if name in name_to_path
            ]

        # Display matches
        for path in self._current_matches:
            if path.is_dir():
                icon = "[dir-icon]\U0001f4c1[/dir-icon]"
                label = f"{icon} {path.name}/"
            else:
                icon = "[file-icon]\U0001f4c4[/file-icon]"
                label = f"{icon} {path.name}"

            results_view.append(ListItem(Static(label), classes="file-item"))

        # Add option to create new if allowed and query doesn't match exactly
        if self.allow_new and query and query not in [p.name for p in self._current_matches]:
            new_path = self.base_path / query
            if not new_path.exists():
                if query.endswith("/"):
                    label = f"[green]\u2795 Create directory: {query}[/green]"
                else:
                    label = f"[green]\u2795 Create file: {query}[/green]"
                results_view.append(ListItem(Static(label), classes="file-item", id="create-new"))

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update results as user types."""
        self._update_results(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter on input."""
        query = event.value.strip()

        # If there's a selected item in the list, use that
        results_view = self.query_one("#results", ListView)
        if results_view.highlighted_child:
            idx = results_view.index
            if idx < len(self._current_matches):
                self.dismiss(self._current_matches[idx])
                return

        # Otherwise, try to create/return the typed path
        if query:
            new_path = self.base_path / query
            if new_path.exists() or self.allow_new:
                self.dismiss(new_path)
                return

        self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection from list."""
        idx = self.query_one("#results", ListView).index

        # Check if it's the "create new" option
        if event.item.id == "create-new":
            query = self.query_one("#search-input", Input).value.strip()
            if query:
                self.dismiss(self.base_path / query)
            return

        if idx < len(self._current_matches):
            self.dismiss(self._current_matches[idx])

    def on_key(self, event) -> None:
        """Handle escape to close."""
        if event.key == "escape":
            self.dismiss(None)
