"""Standard keymaps for HAFS TUI screens.

This module provides shared which-key bindings that can be reused
across multiple screens to ensure consistent navigation and actions.

Keybinding Philosophy:
- SPC (Space) is the universal leader key
- SPC g = Goto (navigation between screens)
- SPC t = Toggle (UI element visibility)
- SPC f = File (file operations)
- SPC a = Agent (agent management)
- SPC h = Help (documentation)
- SPC q = Quit/Close (exit commands)

Usage:
    from hafs.ui.core.standard_keymaps import get_standard_keymap

    class MyScreen(WhichKeyMixin, Screen):
        def get_which_key_map(self):
            keymap = get_standard_keymap(self)
            # Add screen-specific bindings
            keymap["c"] = ("+context", {
                "o": ("open", self.action_open_context),
            })
            return keymap
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from textual.screen import Screen


def get_navigation_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get navigation bindings (g prefix).

    Provides shortcuts to navigate between main screens:
    - g d → Dashboard
    - g c → Chat
    - g l → Logs
    - g s → Services / Settings
    - g a → Analysis
    - g k → Config
    - g t → Training
    - g w → Workspace

    Args:
        screen: The screen instance (used to access app)

    Returns:
        Dict of key → (label, action) tuples
    """

    async def goto_dashboard():
        await screen.app.action_switch_main()

    async def goto_chat():
        await screen.app.action_switch_chat()

    async def goto_logs():
        await screen.app.action_switch_logs()

    async def goto_services():
        await screen.app.action_switch_services()

    async def goto_analysis():
        await screen.app.action_switch_analysis()

    async def goto_config():
        await screen.app.action_switch_config()

    async def goto_training():
        await screen.app.action_switch_training()

    async def goto_workspace():
        await screen.app.action_switch_workspace()

    async def goto_settings():
        await screen.app.action_switch_settings()

    return {
        "d": ("dashboard", goto_dashboard),
        "c": ("chat", goto_chat),
        "l": ("logs", goto_logs),
        "s": ("services", goto_services),
        "a": ("analysis", goto_analysis),
        "k": ("config", goto_config),
        "t": ("training", goto_training),
        "w": ("workspace", goto_workspace),
        "S": ("settings", goto_settings),
    }


def get_toggle_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get toggle bindings (t prefix).

    Provides shortcuts to toggle UI elements:
    - t s → Toggle sidebar
    - t c → Toggle context panel
    - t y → Toggle synergy panel
    - t f → Toggle fullscreen

    Args:
        screen: The screen instance

    Returns:
        Dict of key → (label, action) tuples
    """
    toggles = {}

    # Check if screen has toggle actions
    if hasattr(screen, "action_toggle_sidebar"):
        toggles["s"] = ("sidebar", screen.action_toggle_sidebar)

    if hasattr(screen, "action_toggle_context"):
        toggles["c"] = ("context", screen.action_toggle_context)

    if hasattr(screen, "action_toggle_synergy"):
        toggles["y"] = ("synergy", screen.action_toggle_synergy)

    if hasattr(screen.app, "action_toggle_fullscreen"):
        toggles["f"] = ("fullscreen", screen.app.action_toggle_fullscreen)

    return toggles


def get_file_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get file operation bindings (f prefix).

    Provides shortcuts for file operations:
    - f s → Save file
    - f o → Open file
    - f r → Recent files

    Args:
        screen: The screen instance

    Returns:
        Dict of key → (label, action) tuples
    """
    files = {}

    if hasattr(screen, "action_save_file"):
        files["s"] = ("save", screen.action_save_file)

    if hasattr(screen, "action_open_file"):
        files["o"] = ("open", screen.action_open_file)

    if hasattr(screen, "action_recent_files"):
        files["r"] = ("recent", screen.action_recent_files)

    return files


def get_help_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get help bindings (h prefix).

    Provides shortcuts for help and documentation:
    - h ? → Show help
    - h k → Show keybindings
    - h c → Show cheatsheet

    Args:
        screen: The screen instance

    Returns:
        Dict of key → (label, action) tuples
    """
    help_map = {}

    if hasattr(screen.app, "action_help"):
        help_map["?"] = ("help", screen.app.action_help)
        help_map["h"] = ("help", screen.app.action_help)

    # TODO: Add keybindings viewer
    # help_map["k"] = ("keybindings", show_keybindings)

    return help_map


def get_quit_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get quit/close bindings (q prefix).

    Provides shortcuts for exiting:
    - q q → Quit application
    - q s → Close current screen

    Args:
        screen: The screen instance

    Returns:
        Dict of key → (label, action) tuples
    """

    quit_action = getattr(screen.app, "action_quit", None)
    if callable(quit_action):
        quit_target = quit_action
    else:
        quit_target = screen.app.exit

    def close_screen():
        if len(screen.app.screen_stack) > 1:
            screen.app.pop_screen()
        else:
            screen.notify("Already at root screen", severity="warning", timeout=1)

    return {
        "q": ("quit app", quit_target),
        "s": ("close screen", close_screen),
    }


def get_standard_keymap(screen: "Screen") -> Dict[str, Any]:
    """Get the full standard keymap for a screen.

    Combines all standard keymaps into a single which-key map.
    Screens should call this and add their own bindings.

    Prefix groups:
    - g = goto (navigation)
    - t = toggle (UI elements)
    - f = file (operations)
    - h = help (documentation)
    - q = quit (exit commands)

    Direct actions:
    - r = refresh
    - p = command palette

    Args:
        screen: The screen instance

    Returns:
        Dict representing the full which-key map
    """
    keymap: Dict[str, Any] = {}

    # g = goto (navigation)
    nav_keymap = get_navigation_keymap(screen)
    if nav_keymap:
        keymap["g"] = ("+goto", nav_keymap)

    # t = toggle
    toggle_keymap = get_toggle_keymap(screen)
    if toggle_keymap:
        keymap["t"] = ("+toggle", toggle_keymap)

    # f = file
    file_keymap = get_file_keymap(screen)
    if file_keymap:
        keymap["f"] = ("+file", file_keymap)

    # h = help
    help_keymap = get_help_keymap(screen)
    if help_keymap:
        keymap["h"] = ("+help", help_keymap)

    # q = quit (requires confirmation via submenu)
    quit_keymap = get_quit_keymap(screen)
    keymap["q"] = ("+quit", quit_keymap)

    # Direct actions (no submenu)
    if hasattr(screen, "action_refresh"):
        keymap["r"] = ("refresh", screen.action_refresh)

    if hasattr(screen, "action_command_palette"):
        keymap["p"] = ("palette", screen.action_command_palette)
    elif hasattr(screen.app, "action_command_palette"):
        keymap["p"] = ("palette", screen.app.action_command_palette)

    return keymap
