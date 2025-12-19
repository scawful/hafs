"""Navigation Controller - Unified input handling for TUI.

This module provides a unified navigation controller that coordinates:
- Vim mode navigation
- Which-key leader sequences
- Mouse interactions
- Command execution

The controller integrates with BindingRegistry and CommandRegistry to provide
a consistent input handling experience across the TUI.

Usage:
    controller = NavigationController(app)

    # Process key events
    handled = controller.handle_key(event)

    # Get current state
    if controller.vim_mode == VimMode.NORMAL:
        ...

    # Execute command
    controller.execute_command("file.save")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from hafs.ui.core.binding_registry import BindingContext, get_binding_registry
from hafs.ui.core.command_registry import get_command_registry
from hafs.ui.core.event_bus import get_event_bus
from hafs.ui.core.state_store import get_state_store

if TYPE_CHECKING:
    from textual.app import App
    from textual.events import Key

logger = logging.getLogger(__name__)


class VimMode(str, Enum):
    """Vim navigation modes."""
    NORMAL = "normal"
    INSERT = "insert"
    VISUAL = "visual"
    COMMAND = "command"
    SEARCH = "search"


class InputMode(str, Enum):
    """Overall input mode."""
    NORMAL = "normal"  # Standard Textual bindings
    VIM = "vim"  # Vim-style navigation
    WHICH_KEY = "which_key"  # Leader key sequence in progress


@dataclass
class WhichKeyState:
    """State for which-key leader sequence."""
    active: bool = False
    prefix: List[str] = field(default_factory=list)
    timeout_timer: Any = None


@dataclass
class VimState:
    """State for vim mode."""
    enabled: bool = False
    mode: VimMode = VimMode.NORMAL
    g_pending: bool = False
    count_buffer: str = ""
    register: str = '"'
    search_query: str = ""
    search_matches: List[int] = field(default_factory=list)
    search_index: int = 0
    macro_recording: Optional[str] = None
    macro_buffer: List[str] = field(default_factory=list)
    macros: Dict[str, List[str]] = field(default_factory=dict)


class NavigationController:
    """Unified navigation controller for TUI input handling.

    Coordinates vim mode, which-key, and standard Textual bindings
    through the centralized registries.
    """

    WHICH_KEY_TIMEOUT = 2.0
    VIM_TIMEOUT = 1.0

    def __init__(self, app: Optional["App"] = None):
        self._app = app
        self._vim = VimState()
        self._which_key = WhichKeyState()
        self._input_mode = InputMode.NORMAL

        # Get registries
        self._bindings = get_binding_registry()
        self._commands = get_command_registry()
        self._event_bus = get_event_bus()
        self._state = get_state_store()

        # Track active contexts
        self._active_contexts: Set[str] = {"global"}

        # Load vim preference from state
        self._vim.enabled = self._state.get("settings.vim_mode", False)
        if self._vim.enabled:
            self._input_mode = InputMode.VIM

    def set_app(self, app: "App") -> None:
        """Set the Textual app reference."""
        self._app = app

    @property
    def vim_enabled(self) -> bool:
        """Check if vim mode is enabled."""
        return self._vim.enabled

    @property
    def vim_mode(self) -> VimMode:
        """Get current vim mode."""
        return self._vim.mode

    @property
    def input_mode(self) -> InputMode:
        """Get current input mode."""
        return self._input_mode

    @property
    def which_key_active(self) -> bool:
        """Check if which-key sequence is in progress."""
        return self._which_key.active

    @property
    def which_key_prefix(self) -> str:
        """Get current which-key prefix."""
        return " ".join(self._which_key.prefix)

    def toggle_vim_mode(self) -> None:
        """Toggle vim mode on/off."""
        self._vim.enabled = not self._vim.enabled
        self._vim.mode = VimMode.NORMAL

        if self._vim.enabled:
            self._input_mode = InputMode.VIM
            self._active_contexts.add(BindingContext.VIM_NORMAL)
        else:
            self._input_mode = InputMode.NORMAL
            self._active_contexts.discard(BindingContext.VIM_NORMAL)
            self._active_contexts.discard(BindingContext.VIM_INSERT)
            self._active_contexts.discard(BindingContext.VIM_VISUAL)
            self._active_contexts.discard(BindingContext.VIM_COMMAND)

        # Save preference
        self._state.set("settings.vim_mode", self._vim.enabled)

        # Notify
        if self._app and hasattr(self._app, "notify"):
            status = "enabled" if self._vim.enabled else "disabled"
            self._app.notify(f"Vim mode {status}", timeout=2)

        logger.info(f"Vim mode: {self._vim.enabled}")

    def set_vim_mode(self, mode: VimMode) -> None:
        """Set the vim mode."""
        old_mode = self._vim.mode
        self._vim.mode = mode

        # Update active contexts
        mode_contexts = {
            VimMode.NORMAL: BindingContext.VIM_NORMAL,
            VimMode.INSERT: BindingContext.VIM_INSERT,
            VimMode.VISUAL: BindingContext.VIM_VISUAL,
            VimMode.COMMAND: BindingContext.VIM_COMMAND,
        }

        # Remove old mode context
        if old_mode in mode_contexts:
            self._active_contexts.discard(mode_contexts[old_mode])

        # Add new mode context
        if mode in mode_contexts:
            self._active_contexts.add(mode_contexts[mode])

        logger.debug(f"Vim mode changed: {old_mode} -> {mode}")

    def add_context(self, context: str) -> None:
        """Add an active context (e.g., screen:main)."""
        self._active_contexts.add(context)

    def remove_context(self, context: str) -> None:
        """Remove an active context."""
        self._active_contexts.discard(context)

    def set_screen_context(self, screen_name: str) -> None:
        """Set the current screen context."""
        # Remove old screen contexts
        to_remove = [c for c in self._active_contexts if c.startswith("screen:")]
        for c in to_remove:
            self._active_contexts.discard(c)

        # Add new screen context
        self._active_contexts.add(f"screen:{screen_name}")

    def handle_key(self, event: "Key") -> bool:
        """Handle a key event.

        Args:
            event: The Textual Key event

        Returns:
            True if the event was handled, False otherwise
        """
        key = event.key

        # Check if input is focused (don't intercept typing)
        if self._is_input_focused():
            # Only handle escape in inputs
            if key == "escape" and self._vim.enabled:
                self.set_vim_mode(VimMode.NORMAL)
                return True
            return False

        # Handle which-key mode
        if self._which_key.active:
            return self._handle_which_key(key)

        # Handle leader key (space) to start which-key
        if key == "space" and not self._vim.mode == VimMode.INSERT:
            self._start_which_key()
            return True

        # Handle vim mode
        if self._vim.enabled:
            return self._handle_vim_key(key)

        # Handle standard bindings
        return self._handle_standard_binding(key)

    def _is_input_focused(self) -> bool:
        """Check if an input widget is focused."""
        if not self._app:
            return False

        try:
            from textual.widgets import Input, TextArea
            focused = getattr(self._app, "focused", None)
            return isinstance(focused, (Input, TextArea))
        except Exception:
            return False

    def _start_which_key(self) -> None:
        """Start a which-key leader sequence."""
        self._which_key.active = True
        self._which_key.prefix = []
        self._input_mode = InputMode.WHICH_KEY

        # Set timeout timer
        if self._app and hasattr(self._app, "set_timer"):
            self._which_key.timeout_timer = self._app.set_timer(
                self.WHICH_KEY_TIMEOUT,
                self._cancel_which_key
            )

        self._update_which_key_bar()
        logger.debug("Which-key started")

    def _cancel_which_key(self) -> None:
        """Cancel the which-key sequence."""
        self._which_key.active = False
        self._which_key.prefix = []

        if self._vim.enabled:
            self._input_mode = InputMode.VIM
        else:
            self._input_mode = InputMode.NORMAL

        self._stop_which_key_timer()
        self._update_which_key_bar()
        logger.debug("Which-key cancelled")

    def _stop_which_key_timer(self) -> None:
        """Stop the which-key timeout timer."""
        if self._which_key.timeout_timer:
            try:
                self._which_key.timeout_timer.stop()
            except Exception:
                pass
            self._which_key.timeout_timer = None

    def _handle_which_key(self, key: str) -> bool:
        """Handle a key in which-key mode."""
        if key in ("escape", "backspace"):
            self._cancel_which_key()
            return True

        # Build current prefix
        prefix = "space " + " ".join(self._which_key.prefix) if self._which_key.prefix else "space"

        # Check for binding at this prefix + key
        full_key = f"{prefix} {key}"

        bindings = self._bindings.get(full_key)
        if bindings:
            # Found a command binding
            binding = bindings[0]  # Take first (highest priority)
            self._cancel_which_key()
            self.execute_command(binding.command_id)
            return True

        # Check if this is a prefix for more bindings
        if self._bindings.is_sequence_prefix(full_key):
            self._which_key.prefix.append(key)
            self._stop_which_key_timer()

            # Reset timer
            if self._app and hasattr(self._app, "set_timer"):
                self._which_key.timeout_timer = self._app.set_timer(
                    self.WHICH_KEY_TIMEOUT,
                    self._cancel_which_key
                )

            self._update_which_key_bar()
            return True

        # No binding found
        if self._app and hasattr(self._app, "notify"):
            prefix_display = "SPC " + " ".join(self._which_key.prefix + [key])
            self._app.notify(f"No binding for {prefix_display}", severity="warning", timeout=1)

        self._cancel_which_key()
        return True

    def _update_which_key_bar(self) -> None:
        """Update the which-key bar widget."""
        if not self._app:
            return

        try:
            from hafs.ui.widgets.which_key_bar import WhichKeyBar
            bar = self._app.query_one(WhichKeyBar)

            if not self._which_key.active:
                bar.hide_hints()
                return

            # Get hints from binding registry
            prefix = "space " + " ".join(self._which_key.prefix) if self._which_key.prefix else "space"
            hints = self._bindings.get_which_key_hints(prefix)

            bar.show_hints(
                prefix=" ".join(self._which_key.prefix),
                hints=[(k, label) for k, label, _ in hints]
            )
        except Exception:
            pass

    def _handle_vim_key(self, key: str) -> bool:
        """Handle a key in vim mode."""
        mode = self._vim.mode

        # Mode-specific handling
        if mode == VimMode.INSERT:
            if key == "escape":
                self.set_vim_mode(VimMode.NORMAL)
                return True
            return False  # Let normal typing through

        if mode == VimMode.COMMAND:
            if key == "escape":
                self.set_vim_mode(VimMode.NORMAL)
                return True
            return False  # Let command input through

        if mode == VimMode.SEARCH:
            if key == "escape":
                self.set_vim_mode(VimMode.NORMAL)
                return True
            return False  # Let search input through

        # Normal mode - check for vim bindings
        if mode == VimMode.NORMAL:
            # Handle 'g' prefix
            if self._vim.g_pending:
                self._vim.g_pending = False
                if key == "g":
                    # gg - go to top
                    self.execute_command("nav.top")
                    return True
                else:
                    # Unknown g combination
                    return False

            if key == "g":
                self._vim.g_pending = True
                if self._app and hasattr(self._app, "set_timer"):
                    self._app.set_timer(self.VIM_TIMEOUT, self._reset_g_pending)
                return True

            # Check binding registry for vim context
            context = BindingContext.VIM_NORMAL
            bindings = self._bindings.get(key)
            for binding in bindings:
                if binding.context == context or binding.context == "global":
                    self.execute_command(binding.command_id)
                    return True

        return False

    def _reset_g_pending(self) -> None:
        """Reset the g pending state."""
        self._vim.g_pending = False

    def _handle_standard_binding(self, key: str) -> bool:
        """Handle a standard (non-vim) key binding."""
        # Get effective bindings for current contexts
        effective = self._bindings.get_for_context(list(self._active_contexts))

        if key in effective:
            binding = effective[key]
            self.execute_command(binding.command_id)
            return True

        return False

    def execute_command(self, command_id: str, **kwargs: Any) -> bool:
        """Execute a command by ID.

        Args:
            command_id: The command to execute
            **kwargs: Arguments for the command

        Returns:
            True if command executed successfully
        """
        result = self._commands.execute(command_id, **kwargs)

        if not result.success:
            if self._app and hasattr(self._app, "notify"):
                error = result.error or "Unknown error"
                self._app.notify(f"Command failed: {error}", severity="error", timeout=2)
            return False

        logger.debug(f"Executed command: {command_id}")
        return True

    def get_effective_bindings(self) -> Dict[str, str]:
        """Get effective bindings for current contexts.

        Returns:
            Dict mapping key to command_id
        """
        effective = self._bindings.get_for_context(list(self._active_contexts))
        return {key: binding.command_id for key, binding in effective.items()}

    def get_which_key_hints(self) -> List[tuple[str, str, Optional[str]]]:
        """Get which-key hints for current prefix.

        Returns:
            List of (key, label, icon) tuples
        """
        prefix = "space " + " ".join(self._which_key.prefix) if self._which_key.prefix else "space"
        return self._bindings.get_which_key_hints(prefix)

    def search_commands(self, query: str, limit: int = 10) -> List[Any]:
        """Search for commands.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching commands
        """
        return self._commands.search(query, limit=limit)


# Global navigation controller instance
_global_controller: Optional[NavigationController] = None


def get_navigation_controller() -> NavigationController:
    """Get the global navigation controller instance."""
    global _global_controller
    if _global_controller is None:
        _global_controller = NavigationController()
    return _global_controller


def reset_navigation_controller() -> None:
    """Reset the global navigation controller (for testing)."""
    global _global_controller
    _global_controller = None
