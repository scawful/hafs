"""Halext purple gradient theme for HAFS TUI."""

from hafs.config.schema import ThemeConfig


class HalextTheme:
    """Halext purple gradient theme.

    Based on: linear-gradient(to bottom, #4C3B52, #000)
    """

    # Base colors
    PRIMARY = "#4C3B52"  # Deep purple
    SECONDARY = "#9B59B6"  # Lighter purple
    ACCENT = "#E74C3C"  # Accent red
    BACKGROUND = "#000000"  # Black
    SURFACE = "#1A1A2E"  # Dark purple-tinted surface
    TEXT = "#FFFFFF"  # White text
    TEXT_MUTED = "#888888"  # Muted text

    # Policy colors
    POLICY_READ_ONLY = "#3498DB"  # Blue
    POLICY_WRITABLE = "#27AE60"  # Green
    POLICY_EXECUTABLE = "#E74C3C"  # Red

    # Status colors
    SUCCESS = "#27AE60"
    WARNING = "#F39C12"
    ERROR = "#E74C3C"
    INFO = "#3498DB"

    def __init__(self, config: ThemeConfig | None = None):
        """Initialize theme with optional custom config.

        Args:
            config: Optional theme configuration to override defaults.
        """
        if config:
            self.PRIMARY = config.primary
            self.SECONDARY = config.secondary
            self.ACCENT = config.accent

    @classmethod
    def get_tcss_variables(cls) -> str:
        """Get TCSS variable definitions.

        Returns:
            String of TCSS variable definitions.
        """
        return f"""
$primary: {cls.PRIMARY};
$primary-darken-1: {cls.PRIMARY};
$primary-darken-2: {cls.PRIMARY};
$secondary: {cls.SECONDARY};
$accent: {cls.ACCENT};
$background: {cls.BACKGROUND};
$surface: {cls.SURFACE};
$panel: {cls.SURFACE};
$text: {cls.TEXT};
$foreground: {cls.TEXT};
$text-muted: {cls.TEXT_MUTED};
$text-disabled: {cls.TEXT_MUTED};
$foreground-darken-1: {cls.TEXT_MUTED};
$success: {cls.SUCCESS};
$warning: {cls.WARNING};
$error: {cls.ERROR};
$info: {cls.INFO};
$boost: {cls.PRIMARY};
$policy-readonly: {cls.POLICY_READ_ONLY};
$policy-writable: {cls.POLICY_WRITABLE};
$policy-executable: {cls.POLICY_EXECUTABLE};
$scrollbar-background: {cls.SURFACE};
$scrollbar-background-hover: {cls.SURFACE};
$scrollbar-background-active: {cls.SURFACE};
$scrollbar-color: {cls.PRIMARY};
$scrollbar-color-hover: {cls.SECONDARY};
$scrollbar-color-active: {cls.SECONDARY};
$scrollbar: {cls.PRIMARY};
$scrollbar-hover: {cls.SECONDARY};
$scrollbar-active: {cls.SECONDARY};
$scrollbar-corner-color: {cls.SURFACE};
$link-background: transparent;
$link-color: {cls.INFO};
$link-style: underline;
$link-background-hover: transparent;
$link-color-hover: {cls.SECONDARY};
$link-style-hover: underline;
$input-cursor-background: {cls.SECONDARY};
$input-cursor-foreground: {cls.BACKGROUND};
$input-cursor-text-style: none;
$footer-foreground: {cls.TEXT};
$footer-background: {cls.SURFACE};
$footer-description-foreground: {cls.TEXT_MUTED};
$footer-description-background: {cls.SURFACE};
$footer-item-background: {cls.SURFACE};
$footer-key-foreground: {cls.ACCENT};
$footer-key-background: {cls.SURFACE};
$panel-lighten-1: {cls.SURFACE};
$block-cursor-background: {cls.PRIMARY};
$block-cursor-foreground: {cls.TEXT};
$block-cursor-text-style: bold;
$block-cursor-blurred-background: {cls.SURFACE};
$block-cursor-blurred-foreground: {cls.TEXT_MUTED};
$block-cursor-blurred-text-style: none;
$block-hover-background: {cls.SURFACE};
$surface-lighten-1: {cls.SURFACE};
$surface-lighten-2: {cls.TEXT_MUTED};
$surface-lighten-3: {cls.TEXT_MUTED};
$surface-darken-1: {cls.SURFACE};
$surface-darken-2: {cls.SURFACE};
$surface-darken-3: {cls.SURFACE};
$background-lighten-1: {cls.BACKGROUND};
$background-lighten-2: {cls.BACKGROUND};
$background-lighten-3: {cls.BACKGROUND};
$background-darken-1: {cls.BACKGROUND};
$background-darken-2: {cls.BACKGROUND};
$background-darken-3: {cls.BACKGROUND};
$panel-lighten-2: {cls.SURFACE};
$panel-lighten-3: {cls.SURFACE};
$panel-darken-1: {cls.SURFACE};
$panel-darken-2: {cls.SURFACE};
$panel-darken-3: {cls.SURFACE};
$primary-lighten-1: {cls.PRIMARY};
$primary-lighten-2: {cls.PRIMARY};
$primary-lighten-3: {cls.PRIMARY};
$primary-darken-3: {cls.PRIMARY};
$secondary-lighten-1: {cls.SECONDARY};
$secondary-lighten-2: {cls.SECONDARY};
$secondary-lighten-3: {cls.SECONDARY};
$secondary-darken-1: {cls.SECONDARY};
$secondary-darken-2: {cls.SECONDARY};
$secondary-darken-3: {cls.SECONDARY};
$accent-lighten-1: {cls.ACCENT};
$accent-lighten-2: {cls.ACCENT};
$accent-lighten-3: {cls.ACCENT};
$accent-darken-1: {cls.ACCENT};
$accent-darken-2: {cls.ACCENT};
$accent-darken-3: {cls.ACCENT};
$error-lighten-1: {cls.ERROR};
$error-lighten-2: {cls.ERROR};
$error-lighten-3: {cls.ERROR};
$error-darken-1: {cls.ERROR};
$error-darken-2: {cls.ERROR};
$error-darken-3: {cls.ERROR};
$warning-lighten-1: {cls.WARNING};
$warning-lighten-2: {cls.WARNING};
$warning-lighten-3: {cls.WARNING};
$warning-darken-1: {cls.WARNING};
$warning-darken-2: {cls.WARNING};
$warning-darken-3: {cls.WARNING};
$success-lighten-1: {cls.SUCCESS};
$success-lighten-2: {cls.SUCCESS};
$success-lighten-3: {cls.SUCCESS};
$success-darken-1: {cls.SUCCESS};
$success-darken-2: {cls.SUCCESS};
$success-darken-3: {cls.SUCCESS};
$text-primary: {cls.TEXT};
$text-secondary: {cls.TEXT_MUTED};
$text-warning: {cls.WARNING};
$text-error: {cls.ERROR};
$text-success: {cls.SUCCESS};
$success-muted: {cls.SUCCESS};
$text-info: {cls.INFO};
$info-muted: {cls.INFO};
$error-muted: {cls.ERROR};
$warning-muted: {cls.WARNING};
$primary-muted: {cls.PRIMARY};
$secondary-muted: {cls.SECONDARY};
$accent-muted: {cls.ACCENT};
$text-accent: {cls.ACCENT};
$border: {cls.PRIMARY};
$border-blurred: {cls.SURFACE};
$markdown-h1-color: {cls.SECONDARY};
$markdown-h1-background: transparent;
$markdown-h1-text-style: bold;
$markdown-h2-color: {cls.SECONDARY};
$markdown-h2-background: transparent;
$markdown-h2-text-style: bold;
$markdown-h3-color: {cls.SECONDARY};
$markdown-h3-background: transparent;
$markdown-h3-text-style: bold;
$markdown-h4-color: {cls.SECONDARY};
$markdown-h4-background: transparent;
$markdown-h4-text-style: bold;
$markdown-h5-color: {cls.SECONDARY};
$markdown-h5-background: transparent;
$markdown-h5-text-style: bold;
$markdown-h6-color: {cls.SECONDARY};
$markdown-h6-background: transparent;
$markdown-h6-text-style: bold;
"""
