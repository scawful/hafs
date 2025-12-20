"""Standard keymaps for HAFS TUI screens.

This module provides shared which-key bindings that can be reused
across multiple screens to ensure consistent navigation and actions.

Usage:
    from hafs.ui.core.standard_keymaps import get_standard_keymap

    class MyScreen(WhichKeyMixin, Screen):
        def get_which_key_map(self):
            keymap = get_standard_keymap(self)
            # Add screen-specific bindings
            keymap["r"] = ("refresh", self.action_refresh)
            return keymap
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from textual.screen import Screen


def get_navigation_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get common navigation bindings (g prefix).

    Provides shortcuts to navigate between main screens:
    - g d → Dashboard
    - g c → Chat
    - g l → Logs
    - g s → Services
    - g a → Analysis
    - g k → Config

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

    return {
        "d": ("dashboard", goto_dashboard),
        "c": ("chat", goto_chat),
        "l": ("logs", goto_logs),
        "s": ("services", goto_services),
        "a": ("analysis", goto_analysis),
        "k": ("config", goto_config),
    }


def get_toggle_keymap(screen: "Screen") -> Dict[str, tuple[str, Callable[[], Any]]]:
    """Get common toggle bindings (t prefix).

    Provides shortcuts to toggle UI elements:
    - t s → Toggle sidebar
    - t f → Toggle fullscreen

    Args:
        screen: The screen instance

    Returns:
        Dict of key → (label, action) tuples
    """
    toggles = {}

    # Check if screen has toggle_sidebar action
    if hasattr(screen, "action_toggle_sidebar"):
        toggles["s"] = ("sidebar", screen.action_toggle_sidebar)

    # Add fullscreen toggle if available
    if hasattr(screen.app, "action_toggle_fullscreen"):
        toggles["f"] = ("fullscreen", screen.app.action_toggle_fullscreen)

    return toggles


def get_standard_keymap(screen: "Screen") -> Dict[str, Any]:
    """Get the full standard keymap for a screen.

    Combines all standard keymaps into a single which-key map.
    Screens should call this and add their own bindings.

    Args:
        screen: The screen instance

    Returns:
        Dict representing the full which-key map

    Example:
        def get_which_key_map(self):
            keymap = get_standard_keymap(self)
            keymap["r"] = ("refresh", self.action_refresh)
            keymap["e"] = ("+edit", {
                "f": ("file", self.action_edit_file),
                "c": ("config", self.action_edit_config),
            })
            return keymap
    """
    keymap: Dict[str, Any] = {}

    # Navigation group
    nav_keymap = get_navigation_keymap(screen)
    if nav_keymap:
        keymap["g"] = ("+goto", nav_keymap)

    # Toggle group
    toggle_keymap = get_toggle_keymap(screen)
    if toggle_keymap:
        keymap["t"] = ("+toggle", toggle_keymap)

    # Common actions
    if hasattr(screen, "action_quit"):
        keymap["q"] = ("quit", screen.action_quit)

    if hasattr(screen, "action_command_palette"):
        keymap["p"] = ("palette", screen.action_command_palette)

    if hasattr(screen.app, "action_help"):
        keymap["?"] = ("help", screen.app.action_help)

    return keymap
