"""Interface icons: the Lucide set (https://lucide.dev, ISC licence).

Every icon in the interface goes through :func:`draw`.  Icons are Lucide outlines
(see ``lucide_data``) rasterised by ``Overlay2D.icon`` at the exact device-pixel
size, so they stay crisp at any UI scale and take their colour from the theme.

Names are Lucide's own kebab-case names (``"cloud-sun"``, ``"map-pin"``).  The
short names the older menus used (``"pin"``, ``"close"``, ``"sparkle"``...) are
kept as aliases, so existing call sites keep working.  To bundle another Lucide
icon add it to ``tools/build_lucide.py`` and re-run the script.
"""
from __future__ import annotations

from . import icon_raster

# Legacy name -> Lucide name.
ALIASES = {
    "play": "play", "car": "car", "globe": "globe", "pin": "map-pin", "sliders": "sliders-horizontal",
    "keyboard": "keyboard", "info": "info", "power": "power", "monitor": "monitor", "sparkle": "sparkles",
    "camera": "camera", "clock": "clock", "crowd": "users", "speaker": "volume-2",
    "layout": "layout-dashboard", "terminal": "square-terminal", "chevron_l": "chevron-left",
    "chevron_r": "chevron-right", "chevron_d": "chevron-down", "close": "x", "check": "check",
    "lock": "lock", "reset": "rotate-ccw", "drop": "droplet", "trash": "trash-2", "bolt": "zap",
    "cloud": "cloud", "route": "route", "map": "map", "locate": "locate-fixed", "search": "search",
    "guides": "layout-grid", "directions": "corner-up-right", "sidebar": "panel-left",
    "fullscreen": "maximize", "fullscreen_exit": "minimize", "plus": "plus", "minus": "minus",
    "arrow_up_right": "arrow-up-right", "warning": "triangle-alert", "settings": "settings",
}

ICONS = ALIASES          # kept for code that listed the available icon names


def resolve(name):
    """Lucide name for ``name`` (an alias, a Lucide name or ``snake_case``), or ``None``."""
    if not name:
        return None
    for cand in (ALIASES.get(name), name, str(name).replace("_", "-")):
        if cand and icon_raster.has(cand):
            return cand
    return None


def exists(name):
    return resolve(name) is not None


def draw(ui, name, cx, cy, s, color, width=None, angle=0.0):
    """Draw icon ``name`` centred on (cx, cy), ``s`` logical pixels tall.

    ``width`` is accepted for old call sites and ignored: stroke weight now follows
    the icon size so every icon in the interface has the same visual weight.  Call
    ``ui.icon(..., stroke=)`` directly for a deliberate exception.
    """
    lucide = resolve(name)
    if lucide is None:
        return False
    return ui.icon(lucide, cx, cy, s, color, angle=angle)
