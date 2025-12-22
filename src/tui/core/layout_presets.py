"""Layout presets for different UI configurations.

Provides predefined layouts that can be quickly applied to change
panel visibility and sizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from tui.core.state_store import StateStore


# Layout preset definitions
LAYOUT_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "sidebar_visible": True,
        "sidebar_width": 32,
        "context_panel_visible": True,
        "context_panel_width": 30,
        "synergy_panel_visible": True,
        "synergy_panel_width": 18,
    },
    "compact": {
        "sidebar_visible": False,
        "sidebar_width": 0,
        "context_panel_visible": False,
        "context_panel_width": 0,
        "synergy_panel_visible": False,
        "synergy_panel_width": 0,
    },
    "wide": {
        "sidebar_visible": True,
        "sidebar_width": 48,
        "context_panel_visible": True,
        "context_panel_width": 40,
        "synergy_panel_visible": True,
        "synergy_panel_width": 28,
    },
    "fullscreen_chat": {
        "sidebar_visible": False,
        "sidebar_width": 0,
        "context_panel_visible": False,
        "context_panel_width": 0,
        "synergy_panel_visible": False,
        "synergy_panel_width": 0,
    },
    "focus": {
        "sidebar_visible": True,
        "sidebar_width": 24,
        "context_panel_visible": False,
        "context_panel_width": 0,
        "synergy_panel_visible": False,
        "synergy_panel_width": 0,
    },
    "analysis": {
        "sidebar_visible": False,
        "sidebar_width": 0,
        "context_panel_visible": True,
        "context_panel_width": 40,
        "synergy_panel_visible": True,
        "synergy_panel_width": 24,
    },
}


def get_layout_preset(name: str) -> Dict[str, Any]:
    """Get a layout preset by name.

    Args:
        name: Preset name (default, compact, wide, fullscreen_chat, focus, analysis)

    Returns:
        Dictionary of layout settings
    """
    return LAYOUT_PRESETS.get(name.lower(), LAYOUT_PRESETS["default"]).copy()


def get_preset_names() -> list[str]:
    """Get list of available layout preset names."""
    return list(LAYOUT_PRESETS.keys())


def apply_layout_preset(state_store: "StateStore", preset_name: str) -> None:
    """Apply a layout preset to the state store.

    Args:
        state_store: The StateStore instance to update
        preset_name: Name of the preset to apply
    """
    preset = get_layout_preset(preset_name)

    # Update state store with preset values
    for key, value in preset.items():
        state_store.set(f"settings.{key}", value)

    # Also store the active preset name
    state_store.set("settings.layout_preset", preset_name)


def save_current_as_preset(
    state_store: "StateStore",
    preset_name: str,
) -> Dict[str, Any]:
    """Save current layout settings as a custom preset.

    Args:
        state_store: The StateStore instance to read from
        preset_name: Name for the new preset

    Returns:
        The saved preset configuration
    """
    preset = {
        "sidebar_visible": state_store.get("settings.sidebar_visible", True),
        "sidebar_width": state_store.get("settings.sidebar_width", 32),
        "context_panel_visible": state_store.get("settings.context_panel_visible", True),
        "context_panel_width": state_store.get("settings.context_panel_width", 30),
        "synergy_panel_visible": state_store.get("settings.synergy_panel_visible", True),
        "synergy_panel_width": state_store.get("settings.synergy_panel_width", 18),
    }

    # Store custom presets in state
    custom_presets = state_store.get("settings.custom_layout_presets", {})
    custom_presets[preset_name] = preset
    state_store.set("settings.custom_layout_presets", custom_presets)

    return preset
