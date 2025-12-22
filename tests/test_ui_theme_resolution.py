from __future__ import annotations

import sys
import types

from config.schema import ThemeConfig, ThemeVariant


sys.modules.setdefault("plotext", types.ModuleType("plotext"))

from tui.app import HafsApp


def _resolve_theme(preset: ThemeConfig | str | None) -> str:
    app = HafsApp.__new__(HafsApp)
    return HafsApp._resolve_theme_name(app, preset)


def test_theme_config_light_variant_resolves_halext_light() -> None:
    config = ThemeConfig(preset="halext", variant=ThemeVariant.LIGHT)
    assert _resolve_theme(config) == "halext-light"


def test_unknown_theme_with_light_variant_falls_back_to_halext_light() -> None:
    config = ThemeConfig(preset="not-a-theme", variant=ThemeVariant.LIGHT)
    assert _resolve_theme(config) == "halext-light"


def test_solarized_dark_maps_to_textual_dark() -> None:
    config = ThemeConfig(preset="solarized", variant=ThemeVariant.DARK)
    assert _resolve_theme(config) == "textual-dark"


def test_none_preset_defaults_to_halext() -> None:
    assert _resolve_theme(None) == "halext"
