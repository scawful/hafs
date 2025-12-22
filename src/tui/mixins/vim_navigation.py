"""Vim-style navigation mixin for HAFS TUI components."""

from __future__ import annotations

from enum import Enum
from typing import cast

from textual.binding import Binding
from textual.coordinate import Coordinate
from textual.dom import DOMNode
from textual.message import Message
from textual.widgets import DataTable, ListView, Tree


class VimMode(Enum):
    """Vim navigation modes."""

    NORMAL = "normal"
    COMMAND = "command"
    SEARCH = "search"


class VimNavigationMixin:
    """Mixin providing toggleable Vim-style navigation.

    Add this mixin to screens or widgets to enable vim keybindings
    that work alongside standard arrow key navigation.

    Features:
    - j/k: Move down/up
    - h/l: Collapse/expand (trees) or navigate left/right
    - gg: Jump to first item
    - G: Jump to last item
    - /: Enter search mode
    - :: Enter command mode
    - n/N: Next/previous search match
    - Escape: Return to normal mode

    Example:
        class MyScreen(Screen, VimNavigationMixin):
            BINDINGS = Screen.BINDINGS + VimNavigationMixin.VIM_BINDINGS

            def __init__(self):
                super().__init__()
                self.init_vim_navigation()
    """

    VIM_BINDINGS = [
        Binding("j", "vim_down", "Down", show=False, priority=False),
        Binding("k", "vim_up", "Up", show=False, priority=False),
        Binding("h", "vim_left", "Left/Collapse", show=False, priority=False),
        Binding("l", "vim_right", "Right/Expand", show=False, priority=False),
        Binding("g", "vim_g_prefix", "Go...", show=False, priority=False),
        Binding("G", "vim_goto_end", "Go to End", show=False, priority=False),
        Binding("slash", "vim_search", "Search", show=False, priority=False),
        Binding("colon", "vim_command", "Command", show=False, priority=False),
        Binding("n", "vim_next_match", "Next Match", show=False, priority=False),
        Binding("N", "vim_prev_match", "Prev Match", show=False, priority=False),
        Binding("ctrl+v", "toggle_vim_mode", "Vim Mode", show=True),
    ]

    # State attributes (set by init_vim_navigation)
    _vim_enabled: bool
    _vim_mode: VimMode
    _vim_search_query: str
    _vim_command_buffer: str
    _vim_g_pending: bool
    _vim_search_matches: list[int]
    _vim_search_index: int

    class VimModeToggled(Message):
        """Message sent when vim mode is toggled."""

        def __init__(self, enabled: bool):
            self.enabled = enabled
            super().__init__()

    class VimSearchStarted(Message):
        """Message sent when search mode is entered."""

        pass

    class VimCommandStarted(Message):
        """Message sent when command mode is entered."""

        pass

    def init_vim_navigation(self, enabled: bool | None = None) -> None:
        """Initialize vim navigation state.

        Args:
            enabled: Whether vim mode is enabled. If None, loads from config.
        """
        # Load from config if not explicitly specified
        if enabled is None:
            enabled = self._load_vim_setting()

        self._vim_enabled = enabled
        self._vim_mode = VimMode.NORMAL
        self._vim_search_query = ""
        self._vim_command_buffer = ""
        self._vim_g_pending = False
        self._vim_search_matches = []
        self._vim_search_index = 0

    def _load_vim_setting(self) -> bool:
        """Load vim mode setting from config."""
        try:
            from config.loader import load_config
            config = load_config()
            return config.general.vim_navigation_enabled
        except Exception:
            return False

    def _save_vim_setting(self, enabled: bool) -> None:
        """Save vim mode setting to config."""
        try:
            from config.loader import load_config
            from config.saver import save_config
            config = load_config()
            config.general.vim_navigation_enabled = enabled
            save_config(config)
        except ImportError:
            # tomli_w not installed, can't save
            if hasattr(self, "notify"):
                self.notify("Install tomli-w to persist settings", severity="warning", timeout=3)
        except Exception:
            pass  # Fail silently for other errors

    @property
    def vim_enabled(self) -> bool:
        """Check if vim mode is enabled."""
        return getattr(self, "_vim_enabled", False)

    def _is_input_focused(self) -> bool:
        """Check if an input widget is currently focused.

        Returns:
            True if an Input or TextArea widget has focus.
        """
        if hasattr(self, "focused"):
            from textual.widgets import Input, TextArea
            focused = self.focused
            if isinstance(focused, (Input, TextArea)):
                return True
        return False

    @property
    def vim_mode(self) -> VimMode:
        """Get current vim mode."""
        return getattr(self, "_vim_mode", VimMode.NORMAL)

    def action_toggle_vim_mode(self) -> None:
        """Toggle vim navigation on/off and persist to config."""
        if not hasattr(self, "_vim_enabled"):
            self.init_vim_navigation()

        self._vim_enabled = not self._vim_enabled
        self._vim_mode = VimMode.NORMAL
        self._vim_g_pending = False

        # Persist setting to config
        self._save_vim_setting(self._vim_enabled)

        # Notify via message
        if hasattr(self, "post_message"):
            self.post_message(self.VimModeToggled(self._vim_enabled))

        # Show notification if available
        if hasattr(self, "notify"):
            status = "ON (saved)" if self._vim_enabled else "OFF (saved)"
            self.notify(f"Vim mode: {status}", timeout=2)

    def action_vim_down(self) -> None:
        """Handle vim 'j' key - move down."""
        if not self.vim_enabled or self._is_input_focused():
            return
        self._vim_navigate("down")

    def action_vim_up(self) -> None:
        """Handle vim 'k' key - move up."""
        if not self.vim_enabled or self._is_input_focused():
            return
        self._vim_navigate("up")

    def action_vim_left(self) -> None:
        """Handle vim 'h' key - move left or collapse."""
        if not self.vim_enabled or self._is_input_focused():
            return
        self._vim_navigate("left")

    def action_vim_right(self) -> None:
        """Handle vim 'l' key - move right or expand."""
        if not self.vim_enabled or self._is_input_focused():
            return
        self._vim_navigate("right")

    def action_vim_g_prefix(self) -> None:
        """Handle vim 'g' key - wait for second character."""
        if not self.vim_enabled or self._is_input_focused():
            return

        if self._vim_g_pending:
            # Second 'g' pressed - go to beginning
            self._vim_g_pending = False
            self._vim_goto_start()
        else:
            # First 'g' pressed - wait for second key
            self._vim_g_pending = True
            # Set a timer to reset if no second key pressed
            if hasattr(self, "set_timer"):
                self.set_timer(1.0, self._reset_g_pending)

    def _reset_g_pending(self) -> None:
        """Reset the g pending state."""
        self._vim_g_pending = False

    def action_vim_goto_end(self) -> None:
        """Handle vim 'G' key - go to end."""
        if not self.vim_enabled or self._is_input_focused():
            return
        self._vim_goto_end()

    def action_vim_search(self) -> None:
        """Handle vim '/' key - enter search mode."""
        if not self.vim_enabled or self._is_input_focused():
            return

        self._vim_mode = VimMode.SEARCH
        if hasattr(self, "post_message"):
            self.post_message(self.VimSearchStarted())

        # Try to focus search input if available
        self._focus_search_input()

    def action_vim_command(self) -> None:
        """Handle vim ':' key - enter command mode."""
        if not self.vim_enabled or self._is_input_focused():
            return

        self._vim_mode = VimMode.COMMAND
        if hasattr(self, "post_message"):
            self.post_message(self.VimCommandStarted())

        # Try to focus command input if available
        self._focus_command_input()

    def action_vim_next_match(self) -> None:
        """Handle vim 'n' key - go to next search match."""
        if not self.vim_enabled or self._is_input_focused():
            return

        if self._vim_search_matches:
            self._vim_search_index = (self._vim_search_index + 1) % len(
                self._vim_search_matches
            )
            self._goto_search_match(self._vim_search_index)

    def action_vim_prev_match(self) -> None:
        """Handle vim 'N' key - go to previous search match."""
        if not self.vim_enabled or self._is_input_focused():
            return

        if self._vim_search_matches:
            self._vim_search_index = (self._vim_search_index - 1) % len(
                self._vim_search_matches
            )
            self._goto_search_match(self._vim_search_index)

    def vim_escape(self) -> None:
        """Return to normal mode from search/command mode."""
        self._vim_mode = VimMode.NORMAL
        self._vim_g_pending = False

    # Navigation dispatch methods - override these in subclasses

    def _vim_navigate(self, direction: str) -> None:
        """Navigate in the given direction.

        Override this method to handle navigation for specific widgets.

        Args:
            direction: One of "up", "down", "left", "right"
        """
        # First, try to navigate the currently focused widget
        if hasattr(self, "focused"):
            focused = self.focused
            if isinstance(focused, Tree):
                self._navigate_tree(focused, direction)
                return
            elif isinstance(focused, ListView):
                self._navigate_listview(focused, direction)
                return
            elif isinstance(focused, DataTable):
                self._navigate_datatable(focused, direction)
                return

        # Fallback: Try to find a focusable widget and navigate it
        if hasattr(self, "query_one"):
            # Try Tree widget
            try:
                tree = cast(DOMNode, self).query_one(Tree)
                self._navigate_tree(tree, direction)
                return
            except Exception:
                pass

            # Try ListView widget
            try:
                listview = cast(DOMNode, self).query_one(ListView)
                self._navigate_listview(listview, direction)
                return
            except Exception:
                pass

            # Try DataTable widget
            try:
                table = cast(DOMNode, self).query_one(DataTable)
                self._navigate_datatable(table, direction)
                return
            except Exception:
                pass

    def _navigate_tree(self, tree: "Tree", direction: str) -> None:
        """Navigate a Tree widget.

        Args:
            tree: The Tree widget.
            direction: Navigation direction.
        """
        if direction == "down":
            tree.action_cursor_down()
        elif direction == "up":
            tree.action_cursor_up()
        elif direction == "left":
            # Collapse current node or go to parent
            if tree.cursor_node and tree.cursor_node.is_expanded:
                tree.cursor_node.collapse()
            elif tree.cursor_node and tree.cursor_node.parent:
                tree.cursor_node = tree.cursor_node.parent  # type: ignore[misc]
        elif direction == "right":
            # Expand current node or go to first child
            if tree.cursor_node and not tree.cursor_node.is_expanded:
                tree.cursor_node.expand()
            elif tree.cursor_node and tree.cursor_node.children:
                tree.cursor_node = tree.cursor_node.children[0]  # type: ignore[misc]

    def _navigate_listview(self, listview: "ListView", direction: str) -> None:
        """Navigate a ListView widget.

        Args:
            listview: The ListView widget.
            direction: Navigation direction.
        """
        if direction == "down":
            listview.action_cursor_down()
        elif direction == "up":
            listview.action_cursor_up()
        # left/right could be used for horizontal scrolling if needed

    def _navigate_datatable(self, table: "DataTable", direction: str) -> None:
        """Navigate a DataTable widget.

        Args:
            table: The DataTable widget.
            direction: Navigation direction.
        """
        if direction == "down":
            table.action_cursor_down()
        elif direction == "up":
            table.action_cursor_up()
        elif direction == "left":
            table.action_cursor_left()
        elif direction == "right":
            table.action_cursor_right()

    def _vim_goto_start(self) -> None:
        """Go to the first item. Override for specific widget behavior."""
        # First, try the focused widget
        if hasattr(self, "focused"):
            focused = self.focused
            if isinstance(focused, Tree):
                if focused.root and focused.root.children:
                    focused.cursor_node = focused.root.children[0]  # type: ignore[misc]
                return
            elif isinstance(focused, ListView):
                focused.index = 0
                return
            elif isinstance(focused, DataTable):
                focused.cursor_coordinate = Coordinate(0, 0)
                return

        # Fallback to query_one
        if hasattr(self, "query_one"):
            # Try Tree widget
            try:
                tree = cast(DOMNode, self).query_one(Tree)
                if tree.root and tree.root.children:
                    tree.cursor_node = tree.root.children[0]  # type: ignore[misc]
                return
            except Exception:
                pass

            # Try ListView widget
            try:
                listview = cast(DOMNode, self).query_one(ListView)
                listview.index = 0
                return
            except Exception:
                pass

            # Try DataTable widget
            try:
                table = cast(DOMNode, self).query_one(DataTable)
                table.cursor_coordinate = Coordinate(0, 0)
                return
            except Exception:
                pass

    def _vim_goto_end(self) -> None:
        """Go to the last item. Override for specific widget behavior."""
        # First, try the focused widget
        if hasattr(self, "focused"):
            focused = self.focused
            if isinstance(focused, Tree):
                # Find last visible node
                last_node = focused.root
                while last_node.children and last_node.is_expanded:
                    last_node = last_node.children[-1]
                if last_node != focused.root:
                    focused.cursor_node = last_node  # type: ignore[misc]
                return
            elif isinstance(focused, ListView):
                if focused.children:
                    focused.index = len(focused.children) - 1
                return
            elif isinstance(focused, DataTable):
                row_count = focused.row_count
                if row_count > 0:
                    focused.cursor_coordinate = Coordinate(row_count - 1, 0)
                return

        # Fallback to query_one
        if hasattr(self, "query_one"):
            # Try Tree widget
            try:
                tree = cast(DOMNode, self).query_one(Tree)
                # Find last visible node
                last_node = tree.root
                while last_node.children and last_node.is_expanded:
                    last_node = last_node.children[-1]
                if last_node != tree.root:
                    tree.cursor_node = last_node  # type: ignore[misc]
                return
            except Exception:
                pass

            # Try ListView widget
            try:
                listview = cast(DOMNode, self).query_one(ListView)
                if listview.children:
                    listview.index = len(listview.children) - 1
                return
            except Exception:
                pass

            # Try DataTable widget
            try:
                table = cast(DOMNode, self).query_one(DataTable)
                row_count = table.row_count
                if row_count > 0:
                    table.cursor_coordinate = Coordinate(row_count - 1, 0)
                return
            except Exception:
                pass

    def _focus_search_input(self) -> None:
        """Focus the search input. Override for specific behavior."""
        if hasattr(self, "query_one"):
            try:
                # Try common search input IDs
                for search_id in ["search-input", "search-query", "search"]:
                    try:
                        search = cast(DOMNode, self).query_one(f"#{search_id}")
                        search.focus()
                        return
                    except Exception:
                        pass
            except Exception:
                pass

    def _focus_command_input(self) -> None:
        """Focus the command input. Override for specific behavior."""
        if hasattr(self, "query_one"):
            try:
                # Try common input IDs
                for input_id in ["chat-input", "command-input", "input"]:
                    try:
                        cmd_input = cast(DOMNode, self).query_one(f"#{input_id}")
                        cmd_input.focus()
                        return
                    except Exception:
                        pass
            except Exception:
                pass

    def _goto_search_match(self, index: int) -> None:
        """Go to a specific search match. Override for specific behavior.

        Args:
            index: Index of the match to navigate to.
        """
        pass

    def set_search_matches(self, matches: list[int]) -> None:
        """Set the search match indices.

        Args:
            matches: List of indices where matches were found.
        """
        self._vim_search_matches = matches
        self._vim_search_index = 0

    def clear_search(self) -> None:
        """Clear the current search."""
        self._vim_search_query = ""
        self._vim_search_matches = []
        self._vim_search_index = 0
