"""Halext purple gradient theme for HAFS TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.theme import Theme

from hafs.config.schema import ThemeConfig


class HalextTheme:
    """Halext purple gradient theme.

    Based on: linear-gradient(to bottom, #4C3B52, #000)
    """

    def __init__(self, config: ThemeConfig | None = None):
        """Initialize theme with optional custom config.

        Args:
            config: Optional theme configuration to override defaults.
        """
        # Defaults
        self.PRIMARY = "#4C3B52"
        self.SECONDARY = "#9B59B6"
        self.ACCENT = "#E74C3C"
        self.BACKGROUND = "#000000"
        self.SURFACE = "#1F1F35"
        self.SURFACE_HIGHLIGHT = "#2A2A4E"
        self.TEXT = "#FFFFFF"
        self.TEXT_MUTED = "#AAAAAA"

        # Policy colors
        self.POLICY_READ_ONLY = "#3498DB"
        self.POLICY_WRITABLE = "#27AE60"
        self.POLICY_EXECUTABLE = "#E74C3C"

        # Status colors
        self.SUCCESS = "#27AE60"
        self.WARNING = "#F39C12"
        self.ERROR = "#E74C3C"
        self.INFO = "#3498DB"

        # Border colors for minimal styling
        self.BORDER_SUBTLE = "#2A2A4E"
        self.BORDER_FOCUS = "#4C3B52"

        if config:
            self.PRIMARY = config.primary
            self.SECONDARY = config.secondary
            self.ACCENT = config.accent

    def create_textual_theme(self) -> "Theme":
        """Create a Textual Theme object with Halext colors.

        Returns:
            Configured Textual Theme.
        """
        from textual.theme import Theme

        return Theme(
            name="hafs-halext",
            primary=self.PRIMARY,
            secondary=self.SECONDARY,
            accent=self.ACCENT,
            background=self.BACKGROUND,
            surface=self.SURFACE,
            panel=self.SURFACE,
            success=self.SUCCESS,
            warning=self.WARNING,
            error=self.ERROR,
        )

    def get_tcss_variables(self) -> str:
        """Get TCSS variable definitions.

        Returns:
            String of TCSS variable definitions.
        """
        return f"""
$primary: {self.PRIMARY};
$primary-darken-1: {self.PRIMARY};
$primary-darken-2: {self.PRIMARY};
$secondary: {self.SECONDARY};
$accent: {self.ACCENT};
$background: {self.BACKGROUND};
$surface: {self.SURFACE};
$surface-highlight: {self.SURFACE_HIGHLIGHT};
$panel: {self.SURFACE};
$text: {self.TEXT};
$foreground: {self.TEXT};
$text-muted: {self.TEXT_MUTED};
$text-disabled: {self.TEXT_MUTED};
$foreground-darken-1: {self.TEXT_MUTED};
$success: {self.SUCCESS};
$warning: {self.WARNING};
$error: {self.ERROR};
$info: {self.INFO};
$boost: {self.PRIMARY};
$policy-readonly: {self.POLICY_READ_ONLY};
$policy-writable: {self.POLICY_WRITABLE};
$policy-executable: {self.POLICY_EXECUTABLE};
$scrollbar-background: {self.SURFACE};
$scrollbar-background-hover: {self.SURFACE};
$scrollbar-background-active: {self.SURFACE};
$scrollbar-color: {self.PRIMARY};
$scrollbar-color-hover: {self.SECONDARY};
$scrollbar-color-active: {self.SECONDARY};
$scrollbar: {self.PRIMARY};
$scrollbar-hover: {self.SECONDARY};
$scrollbar-active: {self.SECONDARY};
$scrollbar-corner-color: {self.SURFACE};
$link-background: transparent;
$link-color: {self.INFO};
$link-style: underline;
$link-background-hover: transparent;
$link-color-hover: {self.SECONDARY};
$link-style-hover: underline;
$input-cursor-background: {self.SECONDARY};
$input-cursor-foreground: {self.BACKGROUND};
$input-cursor-text-style: none;
$input-selection-background: {self.SURFACE_HIGHLIGHT};
$input-suggestion: {self.TEXT_MUTED};
$button-focus-text-style: bold;
$button-foreground: {self.TEXT};
$button-color-foreground: {self.TEXT};
$footer-foreground: {self.TEXT};
$footer-background: {self.SURFACE};
$footer-description-foreground: {self.TEXT_MUTED};
$footer-description-background: {self.SURFACE};
$footer-item-background: {self.SURFACE};
$footer-key-foreground: {self.ACCENT};
$footer-key-background: {self.SURFACE};
$panel-lighten-1: {self.SURFACE};
$block-cursor-background: {self.PRIMARY};
$block-cursor-foreground: {self.TEXT};
$block-cursor-text-style: bold;
$block-cursor-blurred-background: {self.SURFACE};
$block-cursor-blurred-foreground: {self.TEXT_MUTED};
$block-cursor-blurred-text-style: none;
$block-hover-background: {self.SURFACE};
$surface-lighten-1: {self.SURFACE};
$surface-lighten-2: {self.TEXT_MUTED};
$surface-lighten-3: {self.TEXT_MUTED};
$surface-darken-1: {self.SURFACE};
$surface-darken-2: {self.SURFACE};
$surface-darken-3: {self.SURFACE};
$background-lighten-1: {self.BACKGROUND};
$background-lighten-2: {self.BACKGROUND};
$background-lighten-3: {self.BACKGROUND};
$background-darken-1: {self.BACKGROUND};
$background-darken-2: {self.BACKGROUND};
$background-darken-3: {self.BACKGROUND};
$panel-lighten-2: {self.SURFACE};
$panel-lighten-3: {self.SURFACE};
$panel-darken-1: {self.SURFACE};
$panel-darken-2: {self.SURFACE};
$panel-darken-3: {self.SURFACE};
$primary-lighten-1: {self.PRIMARY};
$primary-lighten-2: {self.PRIMARY};
$primary-lighten-3: {self.PRIMARY};
$primary-darken-3: {self.PRIMARY};
$secondary-lighten-1: {self.SECONDARY};
$secondary-lighten-2: {self.SECONDARY};
$secondary-lighten-3: {self.SECONDARY};
$secondary-darken-1: {self.SECONDARY};
$secondary-darken-2: {self.SECONDARY};
$secondary-darken-3: {self.SECONDARY};
$accent-lighten-1: {self.ACCENT};
$accent-lighten-2: {self.ACCENT};
$accent-lighten-3: {self.ACCENT};
$accent-darken-1: {self.ACCENT};
$accent-darken-2: {self.ACCENT};
$accent-darken-3: {self.ACCENT};
$error-lighten-1: {self.ERROR};
$error-lighten-2: {self.ERROR};
$error-lighten-3: {self.ERROR};
$error-darken-1: {self.ERROR};
$error-darken-2: {self.ERROR};
$error-darken-3: {self.ERROR};
$warning-lighten-1: {self.WARNING};
$warning-lighten-2: {self.WARNING};
$warning-lighten-3: {self.WARNING};
$warning-darken-1: {self.WARNING};
$warning-darken-2: {self.WARNING};
$warning-darken-3: {self.WARNING};
$success-lighten-1: {self.SUCCESS};
$success-lighten-2: {self.SUCCESS};
$success-lighten-3: {self.SUCCESS};
$success-darken-1: {self.SUCCESS};
$success-darken-2: {self.SUCCESS};
$success-darken-3: {self.SUCCESS};
$text-primary: {self.TEXT};
$text-secondary: {self.TEXT_MUTED};
$text-warning: {self.WARNING};
$text-error: {self.ERROR};
$text-success: {self.SUCCESS};
$success-muted: {self.SUCCESS};
$text-info: {self.INFO};
$info-muted: {self.INFO};
$error-muted: {self.ERROR};
$warning-muted: {self.WARNING};
$primary-muted: {self.PRIMARY};
$secondary-muted: {self.SECONDARY};
$accent-muted: {self.ACCENT};
$text-accent: {self.ACCENT};
$border: {self.PRIMARY};
$border-blurred: {self.SURFACE};
$markdown-h1-color: {self.SECONDARY};
$markdown-h1-background: transparent;
$markdown-h1-text-style: bold;
$markdown-h2-color: {self.SECONDARY};
$markdown-h2-background: transparent;
$markdown-h2-text-style: bold;
$markdown-h3-color: {self.SECONDARY};
$markdown-h3-background: transparent;
$markdown-h3-text-style: bold;
$markdown-h4-color: {self.SECONDARY};
$markdown-h4-background: transparent;
$markdown-h4-text-style: bold;
$markdown-h5-color: {self.SECONDARY};
$markdown-h5-background: transparent;
$markdown-h5-text-style: bold;
$markdown-h6-color: {self.SECONDARY};
$markdown-h6-background: transparent;
$markdown-h6-text-style: bold;
"""
