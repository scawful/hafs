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
$secondary: {cls.SECONDARY};
$accent: {cls.ACCENT};
$background: {cls.BACKGROUND};
$surface: {cls.SURFACE};
$text: {cls.TEXT};
$text-muted: {cls.TEXT_MUTED};
$success: {cls.SUCCESS};
$warning: {cls.WARNING};
$error: {cls.ERROR};
$info: {cls.INFO};
$policy-readonly: {cls.POLICY_READ_ONLY};
$policy-writable: {cls.POLICY_WRITABLE};
$policy-executable: {cls.POLICY_EXECUTABLE};
"""
