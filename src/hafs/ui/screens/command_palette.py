"""Command palette modal for quick access to commands and agents.

This module provides an extensible command palette that integrates with
the CommandRegistry for dynamic command discovery and execution.

Features:
- Fuzzy search across all registered commands
- Category filtering with icons
- Agent @mentions
- Recent command history
- Keyboard navigation (j/k or arrows)
- Command preview and parameter input

Usage:
    result = await app.push_screen_wait(CommandPalette())
    if result:
        print(f"Selected: {result.command.name}")
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from hafs.core.search import fuzzy_autocomplete
from hafs.ui.core.command_registry import (
    Command,
    CommandCategory,
    CommandRegistry,
    get_command_registry,
)


class PaletteItemType(Enum):
    """Type of item in the palette."""
    COMMAND = "command"
    AGENT = "agent"
    RECENT = "recent"
    CATEGORY = "category"


@dataclass
class PaletteItem:
    """An item in the command palette."""
    type: PaletteItemType
    id: str
    name: str
    description: str
    icon: Optional[str] = None
    keybinding: Optional[str] = None
    category: Optional[str] = None
    command: Optional[Command] = None


@dataclass
class PaletteResult:
    """Result from command palette selection."""
    item: PaletteItem
    execute: bool = True  # Whether to execute the command


class CommandPalette(ModalScreen[PaletteResult | None]):
    """Quick-access command palette with fuzzy search.

    The command palette integrates with CommandRegistry to provide
    dynamic access to all registered commands, with support for:

    - Fuzzy search with score-based ranking
    - Category-based filtering (:file, :agent, :nav)
    - Agent @mentions
    - Recent command history
    - Keyboard navigation

    Example:
        result = await app.push_screen_wait(CommandPalette())
        if result and result.execute:
            await registry.execute_async(result.item.id)
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("ctrl+c", "dismiss", "Close", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("ctrl+n", "cursor_down", "Down", show=False),
        Binding("ctrl+p", "cursor_up", "Up", show=False),
        Binding("tab", "next_category", "Next Category", show=False),
        Binding("shift+tab", "prev_category", "Prev Category", show=False),
    ]

    DEFAULT_CSS = """
    CommandPalette {
        align: center middle;
        background: rgba(0, 0, 0, 0.6);
    }

    CommandPalette #palette-dialog {
        width: 80;
        height: auto;
        max-height: 30;
        border: thick $primary;
        background: $surface;
        padding: 1;
    }

    CommandPalette #palette-header {
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
    }

    CommandPalette #palette-input {
        width: 100%;
        height: 3;
        border: solid $secondary;
        margin-bottom: 1;
        padding: 0 1;
    }

    CommandPalette #palette-categories {
        height: 1;
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
    }

    CommandPalette .category-chip {
        padding: 0 1;
        margin-right: 1;
    }

    CommandPalette .category-chip.active {
        background: $primary;
        color: $text;
    }

    CommandPalette #palette-results {
        height: auto;
        max-height: 18;
        border: none;
        margin: 0;
        padding: 0;
        background: $surface;
    }

    CommandPalette .palette-item {
        height: 2;
        padding: 0 1;
    }

    CommandPalette .palette-item:hover {
        background: $primary-darken-1;
    }

    CommandPalette .palette-item.--highlight {
        background: $primary;
    }

    CommandPalette .item-icon {
        width: 3;
        color: $secondary;
    }

    CommandPalette .item-name {
        width: 1fr;
    }

    CommandPalette .item-keybinding {
        width: auto;
        color: $text-muted;
        text-style: dim;
    }

    CommandPalette .item-description {
        color: $text-muted;
        text-style: italic;
    }

    CommandPalette #palette-hint {
        height: auto;
        color: $text-muted;
        text-align: center;
        padding: 1 0;
        border-top: solid $primary-darken-2;
        margin-top: 1;
    }

    CommandPalette #palette-preview {
        height: auto;
        max-height: 5;
        padding: 1;
        border-top: solid $primary-darken-2;
        margin-top: 1;
        display: none;
    }

    CommandPalette #palette-preview.visible {
        display: block;
    }
    """

    # Category filter prefixes
    CATEGORY_FILTERS = {
        ":file": CommandCategory.FILE,
        ":f": CommandCategory.FILE,
        ":agent": CommandCategory.AGENT,
        ":a": CommandCategory.AGENT,
        ":nav": CommandCategory.NAVIGATION,
        ":n": CommandCategory.NAVIGATION,
        ":view": CommandCategory.VIEW,
        ":v": CommandCategory.VIEW,
        ":tool": CommandCategory.TOOLS,
        ":t": CommandCategory.TOOLS,
        ":search": CommandCategory.SEARCH,
        ":s": CommandCategory.SEARCH,
        ":help": CommandCategory.HELP,
        ":h": CommandCategory.HELP,
    }

    # Category icons
    CATEGORY_ICONS = {
        CommandCategory.FILE: "",
        CommandCategory.AGENT: "",
        CommandCategory.CONTEXT: "",
        CommandCategory.VIEW: "",
        CommandCategory.NAVIGATION: "",
        CommandCategory.TOOLS: "",
        CommandCategory.SEARCH: "",
        CommandCategory.HELP: "",
        CommandCategory.CHAT: "",
        CommandCategory.ANALYSIS: "",
        CommandCategory.SYSTEM: "",
    }

    def __init__(
        self,
        registry: Optional[CommandRegistry] = None,
        agent_names: Optional[List[str]] = None,
        show_recent: bool = True,
        initial_query: str = "",
    ) -> None:
        """Initialize command palette.

        Args:
            registry: Command registry to use (defaults to global)
            agent_names: List of agent names for @mentions
            show_recent: Whether to show recent commands
            initial_query: Initial search query
        """
        super().__init__()
        self._registry = registry or get_command_registry()
        self._agent_names = agent_names or []
        self._show_recent = show_recent
        self._initial_query = initial_query

        self._all_items: List[PaletteItem] = []
        self._current_matches: List[PaletteItem] = []
        self._active_category: Optional[CommandCategory] = None

    def compose(self) -> ComposeResult:
        """Compose the palette layout."""
        with Vertical(id="palette-dialog"):
            yield Static(
                "[dim]Type to search commands  •  :category to filter  •  @agent to mention[/dim]",
                id="palette-header"
            )
            yield Input(
                placeholder="Search commands...",
                id="palette-input",
                value=self._initial_query,
            )
            yield Horizontal(id="palette-categories")
            yield ListView(id="palette-results")
            yield Static(id="palette-preview", classes="")
            yield Static(
                "[dim]Enter to select  •  Esc to cancel  •  "
                "↑↓ or j/k to navigate  •  Tab for categories[/dim]",
                id="palette-hint",
            )

    def on_mount(self) -> None:
        """Initialize palette when mounted."""
        self._build_item_list()
        self._build_category_chips()
        self._update_results(self._initial_query)

        input_widget = self.query_one("#palette-input", Input)
        input_widget.focus()

        # Move cursor to end if there's initial query
        if self._initial_query:
            input_widget.cursor_position = len(self._initial_query)

    def _build_item_list(self) -> None:
        """Build complete list of searchable items."""
        self._all_items = []

        # Add commands from registry
        for cmd in self._registry.get_all():
            icon = self.CATEGORY_ICONS.get(cmd.category, "") if isinstance(cmd.category, CommandCategory) else ""
            self._all_items.append(PaletteItem(
                type=PaletteItemType.COMMAND,
                id=cmd.id,
                name=cmd.name,
                description=cmd.description,
                icon=icon,
                keybinding=cmd.keybinding,
                category=cmd.category.value if isinstance(cmd.category, CommandCategory) else str(cmd.category),
                command=cmd,
            ))

        # Add agent mentions
        for agent in self._agent_names:
            self._all_items.append(PaletteItem(
                type=PaletteItemType.AGENT,
                id=f"@{agent}",
                name=f"@{agent}",
                description=f"Mention {agent} agent",
                icon="",
            ))

        # Add recent commands
        if self._show_recent:
            for cmd in self._registry.get_recent()[:5]:
                self._all_items.append(PaletteItem(
                    type=PaletteItemType.RECENT,
                    id=f"recent:{cmd.id}",
                    name=cmd.name,
                    description="Recently used",
                    icon="",
                    keybinding=cmd.keybinding,
                    command=cmd,
                ))

    def _build_category_chips(self) -> None:
        """Build category filter chips."""
        container = self.query_one("#palette-categories", Horizontal)
        container.remove_children()

        # Add "All" chip
        all_chip = Static("[dim]All[/dim]", classes="category-chip active")
        all_chip.id = "chip-all"
        container.mount(all_chip)

        # Add category chips
        for category in self._registry.get_categories():
            icon = self.CATEGORY_ICONS.get(category, "")
            chip = Static(f"[dim]{icon} {category.value}[/dim]", classes="category-chip")
            chip.id = f"chip-{category.value}"
            container.mount(chip)

    def _update_results(self, query: str) -> None:
        """Update results list based on query."""
        results_view = self.query_one("#palette-results", ListView)
        results_view.clear()

        # Check for category filter prefix
        category_filter = None
        search_query = query

        for prefix, category in self.CATEGORY_FILTERS.items():
            if query.startswith(prefix):
                category_filter = category
                search_query = query[len(prefix):].strip()
                break

        # Check for @mention prefix
        if query.startswith("@"):
            search_query = query[1:]
            # Filter to only agents
            items = [i for i in self._all_items if i.type == PaletteItemType.AGENT]
        else:
            items = self._all_items

        # Apply category filter
        if category_filter:
            items = [
                i for i in items
                if i.category == category_filter.value or i.type != PaletteItemType.COMMAND
            ]

        if not search_query:
            # Show all items (limited)
            self._current_matches = items[:20]
        else:
            # Fuzzy search
            searchable = [(i, i.name.lower()) for i in items]
            query_lower = search_query.lower()

            scored = []
            for item, name in searchable:
                score = self._calculate_score(query_lower, item)
                if score > 0:
                    scored.append((score, item))

            scored.sort(key=lambda x: x[0], reverse=True)
            self._current_matches = [item for _, item in scored[:20]]

        # Render matches
        for item in self._current_matches:
            results_view.append(self._create_list_item(item))

        # Auto-select first item
        if self._current_matches and results_view.children:
            results_view.index = 0

    def _calculate_score(self, query: str, item: PaletteItem) -> float:
        """Calculate match score for an item."""
        score = 0.0
        name_lower = item.name.lower()
        id_lower = item.id.lower()
        desc_lower = item.description.lower()

        # Exact matches
        if query == name_lower:
            score += 100
        elif query == id_lower:
            score += 90
        # Prefix matches
        elif name_lower.startswith(query):
            score += 80
        elif id_lower.startswith(query):
            score += 70
        # Contains matches
        elif query in name_lower:
            score += 50 + (len(query) / len(name_lower)) * 20
        elif query in id_lower:
            score += 40
        elif query in desc_lower:
            score += 20

        # Boost recent items
        if item.type == PaletteItemType.RECENT:
            score += 15

        # Boost if has keybinding
        if item.keybinding:
            score += 5

        return score

    def _create_list_item(self, item: PaletteItem) -> ListItem:
        """Create a ListItem widget for a palette item."""
        icon = item.icon or ""
        keybinding = f"[dim]{item.keybinding}[/dim]" if item.keybinding else ""

        if item.type == PaletteItemType.RECENT:
            label = f"[dim]{icon}[/dim] {item.name}  [dim italic]{item.description}[/dim italic] {keybinding}"
        elif item.type == PaletteItemType.AGENT:
            label = f"[cyan]{icon}[/cyan] [bold]{item.name}[/bold]  [dim]{item.description}[/dim]"
        else:
            label = f"{icon} [bold]{item.name}[/bold]  [dim]{item.description}[/dim] {keybinding}"

        list_item = ListItem(Static(label), classes="palette-item")
        list_item.data = item  # Store item reference
        return list_item

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update results as user types."""
        self._update_results(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle enter on input."""
        self._select_current()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection from list."""
        self._select_current()

    def _select_current(self) -> None:
        """Select the currently highlighted item."""
        results_view = self.query_one("#palette-results", ListView)

        if results_view.highlighted_child and self._current_matches:
            idx = results_view.index
            if 0 <= idx < len(self._current_matches):
                item = self._current_matches[idx]
                result = PaletteResult(item=item, execute=True)
                self.dismiss(result)
                return

        self.dismiss(None)

    def action_dismiss(self) -> None:
        """Dismiss the palette without selection."""
        self.dismiss(None)

    def action_cursor_down(self) -> None:
        """Move cursor down."""
        results_view = self.query_one("#palette-results", ListView)
        if results_view.children:
            results_view.action_cursor_down()
            self._update_preview()

    def action_cursor_up(self) -> None:
        """Move cursor up."""
        results_view = self.query_one("#palette-results", ListView)
        if results_view.children:
            results_view.action_cursor_up()
            self._update_preview()

    def action_next_category(self) -> None:
        """Move to next category filter."""
        categories = list(self._registry.get_categories())
        if not categories:
            return

        input_widget = self.query_one("#palette-input", Input)

        if self._active_category is None:
            self._active_category = categories[0]
        else:
            try:
                idx = categories.index(self._active_category)
                self._active_category = categories[(idx + 1) % len(categories)]
            except ValueError:
                self._active_category = categories[0]

        # Update input with category prefix
        input_widget.value = f":{self._active_category.value} "
        input_widget.cursor_position = len(input_widget.value)

    def action_prev_category(self) -> None:
        """Move to previous category filter."""
        categories = list(self._registry.get_categories())
        if not categories:
            return

        input_widget = self.query_one("#palette-input", Input)

        if self._active_category is None:
            self._active_category = categories[-1]
        else:
            try:
                idx = categories.index(self._active_category)
                self._active_category = categories[(idx - 1) % len(categories)]
            except ValueError:
                self._active_category = categories[-1]

        input_widget.value = f":{self._active_category.value} "
        input_widget.cursor_position = len(input_widget.value)

    def _update_preview(self) -> None:
        """Update the command preview panel."""
        preview = self.query_one("#palette-preview", Static)
        results_view = self.query_one("#palette-results", ListView)

        if not self._current_matches or results_view.index < 0:
            preview.remove_class("visible")
            return

        idx = results_view.index
        if idx >= len(self._current_matches):
            preview.remove_class("visible")
            return

        item = self._current_matches[idx]

        if item.command and item.command.description:
            preview.update(f"[dim]{item.command.description}[/dim]")
            preview.add_class("visible")
        else:
            preview.remove_class("visible")

    def update_agent_names(self, names: List[str]) -> None:
        """Update the list of available agent names."""
        self._agent_names = names
        self._build_item_list()
        self._update_results(self.query_one("#palette-input", Input).value)
