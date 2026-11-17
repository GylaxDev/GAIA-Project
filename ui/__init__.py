"""HUD, menus, maps, settings pages and shared interface widgets.

The interface uses an Apple-style "liquid glass" design language: translucent rounded
panels, capsule controls, grouped lists and a single blue accent.  Tokens live in ``theme``,
reusable widgets in ``ui_kit``.
"""
from . import theme
from .hud import Compass, SpeedCard, StatusCard, TimeWeatherCard, Toast

__all__ = ["theme", "SpeedCard", "StatusCard", "Compass", "TimeWeatherCard", "Toast"]
