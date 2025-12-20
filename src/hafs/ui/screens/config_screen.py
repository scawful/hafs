"""Configuration Screen for HAFS TUI.

This screen provides a UI for configuring TUI settings including:
- Theme selection
- Accessibility options
- Vim mode toggle
- Keybinding preferences
- Display options

Usage:
    screen = ConfigScreen()
    app.push_screen(screen)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Label, Select, Static, Switch

from hafs.ui.core.accessibility import ThemeMode, get_accessibility
from hafs.ui.core.command_registry import Command, CommandCategory, get_command_registry
from hafs.ui.core.navigation_controller import get_navigation_controller
from hafs.ui.core.standard_keymaps import get_standard_keymap
from hafs.ui.core.state_store import get_state_store
from hafs.ui.mixins.which_key import WhichKeyMixin
from hafs.ui.widgets.header_bar import HeaderBar
from hafs.ui.widgets.which_key_bar import WhichKeyBar


class ConfigScreen(WhichKeyMixin, Screen):
    """Configuration screen for TUI settings.

    Allows users to customize theme, accessibility, and behavior options.

    WhichKey bindings:
    - SPC g → goto (navigation)
    - SPC s → save
    - SPC r → reset
    """

    BINDINGS = [
        Binding("escape", "pop_screen", "Back"),
        Binding("s", "save", "Save"),
        Binding("r", "reset", "Reset"),
        Binding("ctrl+p", "command_palette", "Commands", show=False),
        Binding("ctrl+k", "command_palette", "Commands", show=False),
    ]

    DEFAULT_CSS = """
    ConfigScreen {
        layout: vertical;
    }

    ConfigScreen .screen-title {
        height: 1;
        color: $accent;
        text-style: bold;
        background: $surface-darken-1;
        padding: 0 2;
        margin-bottom: 1;
    }

    ConfigScreen #main-content {
        height: 1fr;
        padding: 1 2;
    }

    ConfigScreen .config-section {
        height: auto;
        margin-bottom: 2;
        border: solid $primary-darken-2;
        padding: 1 2;
        background: $surface;
    }

    ConfigScreen .section-title {
        height: 1;
        color: $primary;
        text-style: bold;
        margin-bottom: 1;
    }

    ConfigScreen .config-row {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }

    ConfigScreen .config-label {
        width: 30;
        height: 1;
        color: $text;
    }

    ConfigScreen .config-description {
        width: 100%;
        height: 1;
        color: $text-muted;
        padding-left: 2;
    }

    ConfigScreen .config-control {
        width: 1fr;
        height: auto;
    }

    ConfigScreen Select {
        width: 30;
    }

    ConfigScreen Switch {
        width: auto;
    }

    ConfigScreen .button-row {
        height: auto;
        layout: horizontal;
        margin-top: 2;
    }

    ConfigScreen Button {
        margin-right: 1;
    }

    ConfigScreen #footer-area {
        height: auto;
        background: $surface;
        border-top: solid $primary-darken-2;
    }

    ConfigScreen #which-key-bar {
        width: 100%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._state = get_state_store()
        self._a11y = get_accessibility()
        self._nav = get_navigation_controller()
        self._commands = get_command_registry()
        self._register_commands()

    def get_which_key_map(self):
        """Return which-key bindings for this screen."""
        keymap = get_standard_keymap(self)
        # Add config-specific bindings
        keymap["s"] = ("save", self.action_save)
        keymap["r"] = ("reset", self.action_reset)
        keymap["e"] = ("export", self._export_keybindings)
        return keymap

    def _register_commands(self) -> None:
        """Register config screen commands."""
        try:
            self._commands.register(Command(
                id="config.save",
                name="Save Configuration",
                description="Save current configuration settings",
                handler=self.action_save,
                category=CommandCategory.SYSTEM,
                keybinding="s",
            ))
        except ValueError:
            pass

        try:
            self._commands.register(Command(
                id="config.reset",
                name="Reset to Defaults",
                description="Reset configuration to default values",
                handler=self.action_reset,
                category=CommandCategory.SYSTEM,
                keybinding="r",
            ))
        except ValueError:
            pass

    def compose(self) -> ComposeResult:
        """Compose the configuration screen."""
        yield HeaderBar(id="header-bar")
        yield Static("Configuration", classes="screen-title")

        with VerticalScroll(id="main-content"):
            # Theme section
            with Container(classes="config-section"):
                yield Static("Theme", classes="section-title")

                with Vertical(classes="config-row"):
                    yield Label("Theme Mode", classes="config-label")
                    yield Select(
                        [(mode.value.replace("_", " ").title(), mode.value) for mode in ThemeMode],
                        value=self._a11y.theme_mode.value,
                        id="theme-select",
                    )
                    yield Static("Choose color theme for the interface", classes="config-description")

            # Accessibility section
            with Container(classes="config-section"):
                yield Static("Accessibility", classes="section-title")

                with Horizontal(classes="config-row"):
                    yield Label("High Contrast Focus", classes="config-label")
                    yield Switch(
                        value=self._a11y._settings.high_contrast_focus,
                        id="high-contrast-switch",
                    )
                yield Static("Use high contrast focus indicators", classes="config-description")

                with Horizontal(classes="config-row"):
                    yield Label("Reduced Motion", classes="config-label")
                    yield Switch(
                        value=self._a11y.reduced_motion,
                        id="reduced-motion-switch",
                    )
                yield Static("Disable animations and transitions", classes="config-description")

                with Horizontal(classes="config-row"):
                    yield Label("Screen Reader Hints", classes="config-label")
                    yield Switch(
                        value=self._a11y.screen_reader_hints,
                        id="screen-reader-switch",
                    )
                yield Static("Enable screen reader compatible labels", classes="config-description")

            # Navigation section
            with Container(classes="config-section"):
                yield Static("Navigation", classes="section-title")

                with Horizontal(classes="config-row"):
                    yield Label("Vim Mode", classes="config-label")
                    yield Switch(
                        value=self._nav.vim_enabled,
                        id="vim-mode-switch",
                    )
                yield Static("Enable vim-style navigation (hjkl, modes)", classes="config-description")

                with Horizontal(classes="config-row"):
                    yield Label("Show Which-Key Hints", classes="config-label")
                    yield Switch(
                        value=self._state.get("settings.show_which_key", True),
                        id="which-key-switch",
                    )
                yield Static("Show keybinding hints in leader mode", classes="config-description")

            # Display section
            with Container(classes="config-section"):
                yield Static("Display", classes="section-title")

                with Horizontal(classes="config-row"):
                    yield Label("Sidebar Visible", classes="config-label")
                    yield Switch(
                        value=self._state.get("settings.sidebar_visible", True),
                        id="sidebar-switch",
                    )
                yield Static("Show sidebar panel on dashboard", classes="config-description")

                with Horizontal(classes="config-row"):
                    yield Label("Context Panel Visible", classes="config-label")
                    yield Switch(
                        value=self._state.get("settings.chat_context_visible", True),
                        id="context-switch",
                    )
                yield Static("Show context panel in chat mode", classes="config-description")

                with Horizontal(classes="config-row"):
                    yield Label("Synergy Panel Visible", classes="config-label")
                    yield Switch(
                        value=self._state.get("settings.chat_synergy_visible", True),
                        id="synergy-switch",
                    )
                yield Static("Show synergy metrics in chat mode", classes="config-description")

            # Buttons
            with Horizontal(classes="button-row"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Reset to Defaults", id="reset-btn", variant="warning")
                yield Button("Export Keybindings", id="export-btn", variant="default")

        # Footer
        with Container(id="footer-area"):
            yield WhichKeyBar(id="which-key-bar")

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        self._nav.set_screen_context("config")

        # Initialize which-key hints
        self.init_which_key_hints()

        # Set breadcrumb path
        try:
            header = self.query_one(HeaderBar)
            header.set_path("/config")
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "save-btn":
            self.action_save()
        elif button_id == "reset-btn":
            self.action_reset()
        elif button_id == "export-btn":
            self._export_keybindings()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select changes."""
        if event.select.id == "theme-select":
            value = event.value
            # Skip if blank selection
            if value == Select.BLANK or value is None:
                return
            # Handle string values
            value_str = str(value) if value else "default"
            self._a11y.set_theme(value_str)
            # Apply theme to Textual app
            theme_map = {
                "default": "textual-dark",
                "high_contrast": "textual-light",
                "color_blind": "textual-dark",
                "dark": "textual-dark",
                "light": "textual-light",
            }
            textual_theme = theme_map.get(value_str, "textual-dark")
            self.app.theme = textual_theme
            # Force refresh
            self.app.refresh()
            self.notify(f"Theme changed to {value_str}", timeout=2)

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        switch_id = event.switch.id
        value = event.value

        if switch_id == "high-contrast-switch":
            self._a11y.set_high_contrast_focus(value)
        elif switch_id == "reduced-motion-switch":
            self._a11y.set_reduced_motion(value)
        elif switch_id == "screen-reader-switch":
            self._a11y.set_screen_reader_hints(value)
        elif switch_id == "vim-mode-switch":
            self._nav.toggle_vim_mode()
        elif switch_id == "which-key-switch":
            self._state.set("settings.show_which_key", value)
        elif switch_id == "sidebar-switch":
            self._state.set("settings.sidebar_visible", value)
        elif switch_id == "context-switch":
            self._state.set("settings.chat_context_visible", value)
        elif switch_id == "synergy-switch":
            self._state.set("settings.chat_synergy_visible", value)

    def action_save(self) -> None:
        """Save all configuration settings."""
        self.notify("Configuration saved", timeout=2)

    def action_command_palette(self) -> None:
        """Open command palette."""
        from hafs.ui.screens.command_palette import CommandPalette
        self.app.push_screen(CommandPalette())

    def action_reset(self) -> None:
        """Reset all settings to defaults."""
        # Reset accessibility
        self._a11y.set_theme(ThemeMode.DEFAULT)
        self._a11y.set_reduced_motion(False)
        self._a11y.set_screen_reader_hints(True)
        self._a11y.set_high_contrast_focus(False)

        # Reset navigation
        if self._nav.vim_enabled:
            self._nav.toggle_vim_mode()

        # Reset display
        self._state.set("settings.show_which_key", True)
        self._state.set("settings.sidebar_visible", True)
        self._state.set("settings.chat_context_visible", True)
        self._state.set("settings.chat_synergy_visible", True)

        # Refresh UI
        self.refresh()
        self.notify("Settings reset to defaults", timeout=2)

    def _export_keybindings(self) -> None:
        """Export keybindings to a markdown file."""
        from pathlib import Path
        from hafs.ui.core.cheatsheet import KeybindingCheatsheet

        try:
            cheatsheet = KeybindingCheatsheet()
            export_dir = Path.home() / ".context" / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)

            export_file = export_dir / "keybindings.md"
            cheatsheet.export_to_file(str(export_file), format="markdown")

            self.notify(f"Keybindings exported to {export_file.name}", timeout=3)
        except Exception as e:
            self.notify(f"Export failed: {e}", severity="error")

    def action_pop_screen(self) -> None:
        """Return to previous screen."""
        self.app.pop_screen()

    async def on_header_bar_navigation_requested(self, event: HeaderBar.NavigationRequested) -> None:
        """Handle header bar navigation requests."""
        from hafs.ui.core.screen_router import get_screen_router

        route_map = {
            "dashboard": "/dashboard",
            "chat": "/chat",
            "logs": "/logs",
            "services": "/services",
            "analysis": "/analysis",
            "config": "/config",
        }
        route = route_map.get(event.screen)
        if route:
            router = get_screen_router()
            await router.navigate(route)
