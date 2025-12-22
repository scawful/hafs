"""Accessibility - Theme options and screen reader support for TUI.

This module provides accessibility features for the HAFS TUI:
- High contrast theme option
- Color blind-friendly palettes
- Screen reader hints (aria-like labels)
- Reduced motion option
- Font size preferences (where applicable)

Usage:
    a11y = get_accessibility()
    a11y.set_theme("high_contrast")
    a11y.set_reduced_motion(True)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from tui.core.state_store import get_state_store

logger = logging.getLogger(__name__)


class ThemeMode(str, Enum):
    """Available theme modes."""
    DEFAULT = "default"
    HIGH_CONTRAST = "high_contrast"
    COLOR_BLIND = "color_blind"
    DARK = "dark"
    LIGHT = "light"


@dataclass
class ThemeColors:
    """Color palette for a theme."""
    primary: str = "#7C3AED"
    secondary: str = "#06B6D4"
    accent: str = "#F59E0B"
    success: str = "#10B981"
    warning: str = "#F59E0B"
    error: str = "#EF4444"
    surface: str = "#1E1E2E"
    background: str = "#11111B"
    text: str = "#CDD6F4"
    text_muted: str = "#6C7086"


# Predefined theme palettes
THEME_PALETTES: Dict[ThemeMode, ThemeColors] = {
    ThemeMode.DEFAULT: ThemeColors(),
    ThemeMode.HIGH_CONTRAST: ThemeColors(
        primary="#FFFFFF",
        secondary="#00FFFF",
        accent="#FFFF00",
        success="#00FF00",
        warning="#FFFF00",
        error="#FF0000",
        surface="#000000",
        background="#000000",
        text="#FFFFFF",
        text_muted="#CCCCCC",
    ),
    ThemeMode.COLOR_BLIND: ThemeColors(
        primary="#0072B2",  # Blue
        secondary="#009E73",  # Teal
        accent="#F0E442",  # Yellow
        success="#009E73",  # Teal (instead of green)
        warning="#E69F00",  # Orange
        error="#D55E00",  # Vermillion (instead of red)
        surface="#1E1E2E",
        background="#11111B",
        text="#CDD6F4",
        text_muted="#6C7086",
    ),
    ThemeMode.DARK: ThemeColors(
        primary="#BB86FC",
        secondary="#03DAC6",
        accent="#CF6679",
        success="#03DAC6",
        warning="#FFAB00",
        error="#CF6679",
        surface="#121212",
        background="#000000",
        text="#E1E1E1",
        text_muted="#888888",
    ),
    ThemeMode.LIGHT: ThemeColors(
        primary="#6200EE",
        secondary="#03DAC6",
        accent="#FF5722",
        success="#4CAF50",
        warning="#FF9800",
        error="#F44336",
        surface="#FFFFFF",
        background="#F5F5F5",
        text="#212121",
        text_muted="#757575",
    ),
}


@dataclass
class AccessibilitySettings:
    """Accessibility settings for the TUI."""
    theme_mode: ThemeMode = ThemeMode.DEFAULT
    reduced_motion: bool = False
    screen_reader_hints: bool = True
    large_text: bool = False
    high_contrast_focus: bool = False
    announce_notifications: bool = True


class AccessibilityManager:
    """Manager for accessibility settings and features.

    Provides centralized control over accessibility options including
    themes, motion preferences, and screen reader support.
    """

    def __init__(self):
        self._state = get_state_store()
        self._settings = AccessibilitySettings()
        self._load_settings()

    def _load_settings(self) -> None:
        """Load settings from state store."""
        a11y_state = self._state.get("accessibility", {})

        if theme_str := a11y_state.get("theme_mode"):
            try:
                self._settings.theme_mode = ThemeMode(theme_str)
            except ValueError:
                pass

        self._settings.reduced_motion = a11y_state.get("reduced_motion", False)
        self._settings.screen_reader_hints = a11y_state.get("screen_reader_hints", True)
        self._settings.large_text = a11y_state.get("large_text", False)
        self._settings.high_contrast_focus = a11y_state.get("high_contrast_focus", False)
        self._settings.announce_notifications = a11y_state.get("announce_notifications", True)

    def _save_settings(self) -> None:
        """Save settings to state store."""
        self._state.set("accessibility.theme_mode", self._settings.theme_mode.value)
        self._state.set("accessibility.reduced_motion", self._settings.reduced_motion)
        self._state.set("accessibility.screen_reader_hints", self._settings.screen_reader_hints)
        self._state.set("accessibility.large_text", self._settings.large_text)
        self._state.set("accessibility.high_contrast_focus", self._settings.high_contrast_focus)
        self._state.set("accessibility.announce_notifications", self._settings.announce_notifications)

    @property
    def theme_mode(self) -> ThemeMode:
        """Get current theme mode."""
        return self._settings.theme_mode

    @property
    def colors(self) -> ThemeColors:
        """Get current theme colors."""
        return THEME_PALETTES.get(self._settings.theme_mode, THEME_PALETTES[ThemeMode.DEFAULT])

    @property
    def reduced_motion(self) -> bool:
        """Check if reduced motion is enabled."""
        return self._settings.reduced_motion

    @property
    def screen_reader_hints(self) -> bool:
        """Check if screen reader hints are enabled."""
        return self._settings.screen_reader_hints

    def set_theme(self, theme: ThemeMode | str) -> None:
        """Set the theme mode.

        Args:
            theme: Theme mode to set
        """
        if isinstance(theme, str):
            theme = ThemeMode(theme)

        self._settings.theme_mode = theme
        self._save_settings()
        logger.info(f"Theme set to: {theme.value}")

    def set_reduced_motion(self, enabled: bool) -> None:
        """Enable or disable reduced motion.

        Args:
            enabled: Whether to enable reduced motion
        """
        self._settings.reduced_motion = enabled
        self._save_settings()

    def set_screen_reader_hints(self, enabled: bool) -> None:
        """Enable or disable screen reader hints.

        Args:
            enabled: Whether to enable hints
        """
        self._settings.screen_reader_hints = enabled
        self._save_settings()

    def set_large_text(self, enabled: bool) -> None:
        """Enable or disable large text mode.

        Args:
            enabled: Whether to enable large text
        """
        self._settings.large_text = enabled
        self._save_settings()

    def set_high_contrast_focus(self, enabled: bool) -> None:
        """Enable or disable high contrast focus indicators.

        Args:
            enabled: Whether to enable high contrast focus
        """
        self._settings.high_contrast_focus = enabled
        self._save_settings()

    def get_screen_reader_label(self, widget_type: str, content: str = "") -> str:
        """Generate a screen reader label for a widget.

        Args:
            widget_type: Type of widget (button, input, panel, etc.)
            content: Content description

        Returns:
            Screen reader friendly label
        """
        if not self._settings.screen_reader_hints:
            return ""

        labels = {
            "button": f"Button: {content}",
            "input": f"Text input: {content}",
            "panel": f"Panel: {content}",
            "list": f"List with {content} items",
            "tab": f"Tab: {content}",
            "progress": f"Progress: {content}",
            "status": f"Status: {content}",
            "notification": f"Notification: {content}",
        }

        return labels.get(widget_type, content)

    def get_animation_duration(self, default_ms: int) -> int:
        """Get animation duration respecting reduced motion.

        Args:
            default_ms: Default duration in milliseconds

        Returns:
            0 if reduced motion enabled, otherwise default
        """
        if self._settings.reduced_motion:
            return 0
        return default_ms

    def generate_css_variables(self) -> str:
        """Generate CSS variables for current theme.

        Returns:
            CSS string with color variables
        """
        colors = self.colors
        return f"""
        $primary: {colors.primary};
        $secondary: {colors.secondary};
        $accent: {colors.accent};
        $success: {colors.success};
        $warning: {colors.warning};
        $error: {colors.error};
        $surface: {colors.surface};
        $background: {colors.background};
        $text: {colors.text};
        $text-disabled: {colors.text_muted};
        """

    def get_settings_dict(self) -> Dict:
        """Get all settings as a dictionary.

        Returns:
            Dictionary of all accessibility settings
        """
        return {
            "theme_mode": self._settings.theme_mode.value,
            "reduced_motion": self._settings.reduced_motion,
            "screen_reader_hints": self._settings.screen_reader_hints,
            "large_text": self._settings.large_text,
            "high_contrast_focus": self._settings.high_contrast_focus,
            "announce_notifications": self._settings.announce_notifications,
        }


# Global accessibility manager instance
_global_accessibility: Optional[AccessibilityManager] = None


def get_accessibility() -> AccessibilityManager:
    """Get the global accessibility manager instance."""
    global _global_accessibility
    if _global_accessibility is None:
        _global_accessibility = AccessibilityManager()
    return _global_accessibility


def reset_accessibility() -> None:
    """Reset the global accessibility manager (for testing)."""
    global _global_accessibility
    _global_accessibility = None
