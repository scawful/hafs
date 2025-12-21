"""Configurable theme system for HAFS TUI.

Supports multiple theme presets with dark/light variants and custom overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from textual.theme import Theme

from hafs.config.schema import ThemeConfig, ThemeVariant
from hafs.ui.theme_presets import get_preset, PRESETS


class HalextTheme:
    """Configurable theme with preset support.

    Supports loading colors from presets (halext, nord, solarized, dracula, gruvbox)
    with dark/light variants and optional custom color overrides.
    """


    def __init__(self, theme: Theme | None = None, config: ThemeConfig | None = None):
        """Initialize theme adapter.

        Args:
            theme: Existing Textual Theme object (optional).
            config: Theme configuration for legacy preset loading (optional).
        """
        from textual.theme import Theme as TextualTheme

        self._theme: TextualTheme | None = None

        if theme:
            self._theme = theme
            self._init_from_theme(theme)
        elif config:
            self._init_from_config(config)
        else:
            # Default fallback
            self._init_from_config(None)

    def _init_from_theme(self, theme: "Theme") -> None:
        """Initialize colors from a Textual Theme object."""
        self._preset_name = theme.name
        self._variant = "dark" if theme.dark else "light"

        # Map base colors
        self.PRIMARY = theme.primary
        self.SECONDARY = theme.secondary
        self.ACCENT = theme.accent
        self.BACKGROUND = theme.background
        self.SURFACE = theme.surface
        self.SURFACE_HIGHLIGHT = theme.surface.lighten(0.05) if theme.dark else theme.surface.darken(0.05)
        self.TEXT = theme.foreground
        self.TEXT_MUTED = theme.foreground.darken(0.3) if theme.dark else theme.foreground.lighten(0.3)
        self.SUCCESS = theme.success
        self.WARNING = theme.warning
        self.ERROR = theme.error
        self.INFO = theme.secondary  # Fallback as textual doesn't have info
        self.BORDER = theme.primary
        self.BORDER_FOCUS = theme.secondary

        self._init_derived_colors()

    def _init_from_config(self, config: ThemeConfig | None) -> None:
        """Initialize colors from legacy configuration."""
        # Determine preset and variant
        preset_name = "halext"
        variant = "dark"

        if config:
            preset_name = config.preset if config.preset in PRESETS else "halext"
            variant = config.variant.value if config.variant else "dark"

        # Load colors from preset
        colors = get_preset(preset_name, variant)

        # Apply custom overrides if provided
        if config and config.custom:
            custom_dict = config.custom.model_dump(exclude_none=True)
            colors = {**colors, **custom_dict}

        # Set all color attributes
        from rich.color import Color
        self.PRIMARY = Color.parse(colors.get("primary", "#4C3B52"))
        self.SECONDARY = Color.parse(colors.get("secondary", "#9B59B6"))
        self.ACCENT = Color.parse(colors.get("accent", "#E74C3C"))
        self.BACKGROUND = Color.parse(colors.get("background", "#000000"))
        self.SURFACE = Color.parse(colors.get("surface", "#1F1F35"))
        self.SURFACE_HIGHLIGHT = Color.parse(colors.get("surface_highlight", "#2A2A4E"))
        self.TEXT = Color.parse(colors.get("text", "#FFFFFF"))
        self.TEXT_MUTED = Color.parse(colors.get("text_muted", "#AAAAAA"))
        self.SUCCESS = Color.parse(colors.get("success", "#27AE60"))
        self.WARNING = Color.parse(colors.get("warning", "#F39C12"))
        self.ERROR = Color.parse(colors.get("error", "#E74C3C"))
        self.INFO = Color.parse(colors.get("info", "#3498DB"))
        self.BORDER = Color.parse(colors.get("border", "#4C3B52"))
        self.BORDER_FOCUS = Color.parse(colors.get("border_focus", "#9B59B6"))

        # Store info
        self._preset_name = preset_name
        self._variant = variant

        self._init_derived_colors()

    def _init_derived_colors(self) -> None:
        """Initialize derived policy and compatibility colors."""
        self.POLICY_READ_ONLY = self.INFO
        self.POLICY_WRITABLE = self.SUCCESS
        self.POLICY_EXECUTABLE = self.ACCENT
        self.BORDER_SUBTLE = self.SURFACE_HIGHLIGHT

    @property
    def preset_name(self) -> str:
        """Get the current preset name."""
        return self._preset_name

    @property
    def variant(self) -> str:
        """Get the current variant (dark/light)."""
        return self._variant

    def create_textual_theme(self) -> "Theme":
        """Create a Textual Theme object (only if initialized from config)."""
        if self._theme:
            return self._theme

        from textual.theme import Theme
        return Theme(
            name=f"hafs-{self.preset_name}",
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

    def get_tcss_variables(self) -> dict[str, str]:
        """Get TCSS variables as a dictionary.

        optimized to return dict directly for creating variables map.
        """
        # Note: We return hex strings for CSS variables
        return {
            "primary": self.PRIMARY.get_truecolor().hex,
            "primary-darken-1": self.PRIMARY.get_truecolor().hex,
            "primary-darken-2": self.PRIMARY.get_truecolor().hex,
            "secondary": self.SECONDARY.get_truecolor().hex,
            "accent": self.ACCENT.get_truecolor().hex,
            "background": self.BACKGROUND.get_truecolor().hex,
            "surface": self.SURFACE.get_truecolor().hex,
            "surface-highlight": self.SURFACE_HIGHLIGHT.get_truecolor().hex,
            "panel": self.SURFACE.get_truecolor().hex,
            "text": self.TEXT.get_truecolor().hex,
            "foreground": self.TEXT.get_truecolor().hex,
            "text-muted": self.TEXT_MUTED.get_truecolor().hex,
            "text-disabled": self.TEXT_MUTED.get_truecolor().hex,
            "foreground-darken-1": self.TEXT_MUTED.get_truecolor().hex,
            "success": self.SUCCESS.get_truecolor().hex,
            "warning": self.WARNING.get_truecolor().hex,
            "error": self.ERROR.get_truecolor().hex,
            "info": self.INFO.get_truecolor().hex,
            "boost": self.PRIMARY.get_truecolor().hex,
            "policy-readonly": self.POLICY_READ_ONLY.get_truecolor().hex,
            "policy-writable": self.POLICY_WRITABLE.get_truecolor().hex,
            "policy-executable": self.POLICY_EXECUTABLE.get_truecolor().hex,
            "scrollbar-background": self.SURFACE.get_truecolor().hex,
            "scrollbar-background-hover": self.SURFACE.get_truecolor().hex,
            "scrollbar-background-active": self.SURFACE.get_truecolor().hex,
            "scrollbar-color": self.PRIMARY.get_truecolor().hex,
            "scrollbar-color-hover": self.SECONDARY.get_truecolor().hex,
            "scrollbar-color-active": self.SECONDARY.get_truecolor().hex,
            "scrollbar": self.PRIMARY.get_truecolor().hex,
            "scrollbar-hover": self.SECONDARY.get_truecolor().hex,
            "scrollbar-active": self.SECONDARY.get_truecolor().hex,
            "scrollbar-corner-color": self.SURFACE.get_truecolor().hex,
            "link-background": "transparent",
            "link-color": self.INFO.get_truecolor().hex,
            "link-style": "underline",
            "link-background-hover": "transparent",
            "link-color-hover": self.SECONDARY.get_truecolor().hex,
            "link-style-hover": "underline",
            "input-cursor-background": self.SECONDARY.get_truecolor().hex,
            "input-cursor-foreground": self.BACKGROUND.get_truecolor().hex,
            "input-cursor-text-style": "none",
            "input-selection-background": self.SURFACE_HIGHLIGHT.get_truecolor().hex,
            "input-suggestion": self.TEXT_MUTED.get_truecolor().hex,
            "button-focus-text-style": "bold",
            "button-foreground": self.TEXT.get_truecolor().hex,
            "button-color-foreground": self.TEXT.get_truecolor().hex,
            "footer-foreground": self.TEXT.get_truecolor().hex,
            "footer-background": self.SURFACE.get_truecolor().hex,
            "footer-description-foreground": self.TEXT_MUTED.get_truecolor().hex,
            "footer-description-background": self.SURFACE.get_truecolor().hex,
            "footer-item-background": self.SURFACE.get_truecolor().hex,
            "footer-key-foreground": self.ACCENT.get_truecolor().hex,
            "footer-key-background": self.SURFACE.get_truecolor().hex,
            "panel-lighten-1": self.SURFACE.get_truecolor().hex,
            "block-cursor-background": self.PRIMARY.get_truecolor().hex,
            "block-cursor-foreground": self.TEXT.get_truecolor().hex,
            "block-cursor-text-style": "bold",
            "block-cursor-blurred-background": self.SURFACE.get_truecolor().hex,
            "block-cursor-blurred-foreground": self.TEXT_MUTED.get_truecolor().hex,
            "block-cursor-blurred-text-style": "none",
            "block-hover-background": self.SURFACE.get_truecolor().hex,
            "surface-lighten-1": self.SURFACE.get_truecolor().hex,
            "surface-lighten-2": self.TEXT_MUTED.get_truecolor().hex,
            "surface-lighten-3": self.TEXT_MUTED.get_truecolor().hex,
            "surface-darken-1": self.SURFACE.get_truecolor().hex,
            "surface-darken-2": self.SURFACE.get_truecolor().hex,
            "surface-darken-3": self.SURFACE.get_truecolor().hex,
            "background-lighten-1": self.BACKGROUND.get_truecolor().hex,
            "background-lighten-2": self.BACKGROUND.get_truecolor().hex,
            "background-lighten-3": self.BACKGROUND.get_truecolor().hex,
            "background-darken-1": self.BACKGROUND.get_truecolor().hex,
            "background-darken-2": self.BACKGROUND.get_truecolor().hex,
            "background-darken-3": self.BACKGROUND.get_truecolor().hex,
            "panel-lighten-2": self.SURFACE.get_truecolor().hex,
            "panel-lighten-3": self.SURFACE.get_truecolor().hex,
            "panel-darken-1": self.SURFACE.get_truecolor().hex,
            "panel-darken-2": self.SURFACE.get_truecolor().hex,
            "panel-darken-3": self.SURFACE.get_truecolor().hex,
            "primary-lighten-1": self.PRIMARY.get_truecolor().hex,
            "primary-lighten-2": self.PRIMARY.get_truecolor().hex,
            "primary-lighten-3": self.PRIMARY.get_truecolor().hex,
            "primary-darken-3": self.PRIMARY.get_truecolor().hex,
            "secondary-lighten-1": self.SECONDARY.get_truecolor().hex,
            "secondary-lighten-2": self.SECONDARY.get_truecolor().hex,
            "secondary-lighten-3": self.SECONDARY.get_truecolor().hex,
            "secondary-darken-1": self.SECONDARY.get_truecolor().hex,
            "secondary-darken-2": self.SECONDARY.get_truecolor().hex,
            "secondary-darken-3": self.SECONDARY.get_truecolor().hex,
            "accent-lighten-1": self.ACCENT.get_truecolor().hex,
            "accent-lighten-2": self.ACCENT.get_truecolor().hex,
            "accent-lighten-3": self.ACCENT.get_truecolor().hex,
            "accent-darken-1": self.ACCENT.get_truecolor().hex,
            "accent-darken-2": self.ACCENT.get_truecolor().hex,
            "accent-darken-3": self.ACCENT.get_truecolor().hex,
            "error-lighten-1": self.ERROR.get_truecolor().hex,
            "error-lighten-2": self.ERROR.get_truecolor().hex,
            "error-lighten-3": self.ERROR.get_truecolor().hex,
            "error-darken-1": self.ERROR.get_truecolor().hex,
            "error-darken-2": self.ERROR.get_truecolor().hex,
            "error-darken-3": self.ERROR.get_truecolor().hex,
            "warning-lighten-1": self.WARNING.get_truecolor().hex,
            "warning-lighten-2": self.WARNING.get_truecolor().hex,
            "warning-lighten-3": self.WARNING.get_truecolor().hex,
            "warning-darken-1": self.WARNING.get_truecolor().hex,
            "warning-darken-2": self.WARNING.get_truecolor().hex,
            "warning-darken-3": self.WARNING.get_truecolor().hex,
            "success-lighten-1": self.SUCCESS.get_truecolor().hex,
            "success-lighten-2": self.SUCCESS.get_truecolor().hex,
            "success-lighten-3": self.SUCCESS.get_truecolor().hex,
            "success-darken-1": self.SUCCESS.get_truecolor().hex,
            "success-darken-2": self.SUCCESS.get_truecolor().hex,
            "success-darken-3": self.SUCCESS.get_truecolor().hex,
            "text-primary": self.TEXT.get_truecolor().hex,
            "text-secondary": self.TEXT_MUTED.get_truecolor().hex,
            "text-warning": self.WARNING.get_truecolor().hex,
            "text-error": self.ERROR.get_truecolor().hex,
            "text-success": self.SUCCESS.get_truecolor().hex,
            "success-muted": self.SUCCESS.get_truecolor().hex,
            "text-info": self.INFO.get_truecolor().hex,
            "info-muted": self.INFO.get_truecolor().hex,
            "error-muted": self.ERROR.get_truecolor().hex,
            "warning-muted": self.WARNING.get_truecolor().hex,
            "primary-muted": self.PRIMARY.get_truecolor().hex,
            "secondary-muted": self.SECONDARY.get_truecolor().hex,
            "accent-muted": self.ACCENT.get_truecolor().hex,
            "text-accent": self.ACCENT.get_truecolor().hex,
            "border": self.PRIMARY.get_truecolor().hex,
            "border-blurred": self.SURFACE.get_truecolor().hex,
            "markdown-h1-color": self.SECONDARY.get_truecolor().hex,
            "markdown-h1-background": "transparent",
            "markdown-h1-text-style": "bold",
            "markdown-h2-color": self.SECONDARY.get_truecolor().hex,
            "markdown-h2-background": "transparent",
            "markdown-h2-text-style": "bold",
            "markdown-h3-color": self.SECONDARY.get_truecolor().hex,
            "markdown-h3-background": "transparent",
            "markdown-h3-text-style": "bold",
            "markdown-h4-color": self.SECONDARY.get_truecolor().hex,
            "markdown-h4-background": "transparent",
            "markdown-h4-text-style": "bold",
            "markdown-h5-color": self.SECONDARY.get_truecolor().hex,
            "markdown-h5-background": "transparent",
            "markdown-h5-text-style": "bold",
            "markdown-h6-color": self.SECONDARY.get_truecolor().hex,
            "markdown-h6-background": "transparent",
            "markdown-h6-text-style": "bold",
        }
