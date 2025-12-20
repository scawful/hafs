"""Built-in theme presets for HAFS TUI.

Provides color palettes for multiple themes with dark and light variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# Type alias for color palettes
ColorPalette = Dict[str, str]


@dataclass
class ThemePreset:
    """A theme preset with dark and light variants."""

    name: str
    dark: ColorPalette
    light: ColorPalette


# =============================================================================
# Halext Theme (Default) - Purple gradient
# =============================================================================

HALEXT_DARK: ColorPalette = {
    "primary": "#4C3B52",
    "secondary": "#9B59B6",
    "accent": "#E74C3C",
    "background": "#000000",
    "surface": "#1F1F35",
    "surface_highlight": "#2A2A4E",
    "text": "#FFFFFF",
    "text_muted": "#AAAAAA",
    "success": "#27AE60",
    "warning": "#F39C12",
    "error": "#E74C3C",
    "info": "#3498DB",
    "border": "#4C3B52",
    "border_focus": "#9B59B6",
}

HALEXT_LIGHT: ColorPalette = {
    "primary": "#6B4E74",
    "secondary": "#8E44AD",
    "accent": "#C0392B",
    "background": "#FAFAFA",
    "surface": "#FFFFFF",
    "surface_highlight": "#F0F0F0",
    "text": "#2C3E50",
    "text_muted": "#7F8C8D",
    "success": "#27AE60",
    "warning": "#E67E22",
    "error": "#C0392B",
    "info": "#2980B9",
    "border": "#BDC3C7",
    "border_focus": "#8E44AD",
}


# =============================================================================
# Nord Theme - Cool blue-gray palette
# https://www.nordtheme.com/
# =============================================================================

NORD_DARK: ColorPalette = {
    "primary": "#5E81AC",
    "secondary": "#81A1C1",
    "accent": "#88C0D0",
    "background": "#2E3440",
    "surface": "#3B4252",
    "surface_highlight": "#434C5E",
    "text": "#ECEFF4",
    "text_muted": "#D8DEE9",
    "success": "#A3BE8C",
    "warning": "#EBCB8B",
    "error": "#BF616A",
    "info": "#81A1C1",
    "border": "#4C566A",
    "border_focus": "#88C0D0",
}

NORD_LIGHT: ColorPalette = {
    "primary": "#5E81AC",
    "secondary": "#81A1C1",
    "accent": "#88C0D0",
    "background": "#ECEFF4",
    "surface": "#E5E9F0",
    "surface_highlight": "#D8DEE9",
    "text": "#2E3440",
    "text_muted": "#4C566A",
    "success": "#A3BE8C",
    "warning": "#EBCB8B",
    "error": "#BF616A",
    "info": "#5E81AC",
    "border": "#D8DEE9",
    "border_focus": "#5E81AC",
}


# =============================================================================
# Solarized Theme - Precision colors for machines and people
# https://ethanschoonover.com/solarized/
# =============================================================================

SOLARIZED_DARK: ColorPalette = {
    "primary": "#268BD2",
    "secondary": "#2AA198",
    "accent": "#CB4B16",
    "background": "#002B36",
    "surface": "#073642",
    "surface_highlight": "#094B58",
    "text": "#839496",
    "text_muted": "#657B83",
    "success": "#859900",
    "warning": "#B58900",
    "error": "#DC322F",
    "info": "#268BD2",
    "border": "#586E75",
    "border_focus": "#2AA198",
}

SOLARIZED_LIGHT: ColorPalette = {
    "primary": "#268BD2",
    "secondary": "#2AA198",
    "accent": "#CB4B16",
    "background": "#FDF6E3",
    "surface": "#EEE8D5",
    "surface_highlight": "#DDD6C3",
    "text": "#657B83",
    "text_muted": "#93A1A1",
    "success": "#859900",
    "warning": "#B58900",
    "error": "#DC322F",
    "info": "#268BD2",
    "border": "#93A1A1",
    "border_focus": "#268BD2",
}


# =============================================================================
# Dracula Theme - Dark theme for vampires
# https://draculatheme.com/
# =============================================================================

DRACULA_DARK: ColorPalette = {
    "primary": "#BD93F9",
    "secondary": "#FF79C6",
    "accent": "#50FA7B",
    "background": "#282A36",
    "surface": "#44475A",
    "surface_highlight": "#6272A4",
    "text": "#F8F8F2",
    "text_muted": "#6272A4",
    "success": "#50FA7B",
    "warning": "#FFB86C",
    "error": "#FF5555",
    "info": "#8BE9FD",
    "border": "#6272A4",
    "border_focus": "#BD93F9",
}

# Dracula is primarily a dark theme, but we provide a lightened version
DRACULA_LIGHT: ColorPalette = {
    "primary": "#9D65D0",
    "secondary": "#E05AA0",
    "accent": "#30DA5B",
    "background": "#F8F8F2",
    "surface": "#EBEBF0",
    "surface_highlight": "#D8D8E0",
    "text": "#282A36",
    "text_muted": "#6272A4",
    "success": "#30DA5B",
    "warning": "#E5984C",
    "error": "#E53535",
    "info": "#6BC9DD",
    "border": "#D8D8E0",
    "border_focus": "#9D65D0",
}


# =============================================================================
# Gruvbox Theme - Retro groove color scheme
# https://github.com/morhetz/gruvbox
# =============================================================================

GRUVBOX_DARK: ColorPalette = {
    "primary": "#458588",
    "secondary": "#98971A",
    "accent": "#D79921",
    "background": "#282828",
    "surface": "#3C3836",
    "surface_highlight": "#504945",
    "text": "#EBDBB2",
    "text_muted": "#A89984",
    "success": "#98971A",
    "warning": "#D79921",
    "error": "#CC241D",
    "info": "#458588",
    "border": "#504945",
    "border_focus": "#83A598",
}

GRUVBOX_LIGHT: ColorPalette = {
    "primary": "#076678",
    "secondary": "#79740E",
    "accent": "#B57614",
    "background": "#FBF1C7",
    "surface": "#EBDBB2",
    "surface_highlight": "#D5C4A1",
    "text": "#3C3836",
    "text_muted": "#7C6F64",
    "success": "#79740E",
    "warning": "#B57614",
    "error": "#9D0006",
    "info": "#076678",
    "border": "#BDAE93",
    "border_focus": "#076678",
}


# =============================================================================
# Preset Registry
# =============================================================================

PRESETS: Dict[str, ThemePreset] = {
    "halext": ThemePreset("Halext", HALEXT_DARK, HALEXT_LIGHT),
    "nord": ThemePreset("Nord", NORD_DARK, NORD_LIGHT),
    "solarized": ThemePreset("Solarized", SOLARIZED_DARK, SOLARIZED_LIGHT),
    "dracula": ThemePreset("Dracula", DRACULA_DARK, DRACULA_LIGHT),
    "gruvbox": ThemePreset("Gruvbox", GRUVBOX_DARK, GRUVBOX_LIGHT),
}


def get_preset(name: str, variant: str = "dark") -> ColorPalette:
    """Get a theme preset by name and variant.

    Args:
        name: Preset name (halext, nord, solarized, dracula, gruvbox)
        variant: "dark" or "light"

    Returns:
        Color palette dictionary
    """
    preset = PRESETS.get(name.lower(), PRESETS["halext"])
    return preset.dark if variant == "dark" else preset.light


def get_preset_names() -> list[str]:
    """Get list of available preset names."""
    return list(PRESETS.keys())


def get_preset_display_name(name: str) -> str:
    """Get the display name for a preset."""
    preset = PRESETS.get(name.lower())
    return preset.name if preset else name.title()
