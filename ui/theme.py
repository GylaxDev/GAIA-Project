"""Design tokens for the osm3d interface.

One graphite look everywhere: near-black surfaces with a soft top-to-bottom
gradient, a crisp 1 px light hairline, small corner radii and Inter type.
Change a value here and the HUD, the map chrome and every dialog follow.

All sizes are in *logical* pixels; ``Overlay2D`` multiplies them by the display
scale, so the UI stays sharp (and the same physical size) on HiDPI screens.
"""
from __future__ import annotations

from ..core.config import ACCENT

# ------------------------------------------------------------------ surfaces
# Gradient pairs are (top, bottom).  Alpha < 1 lets the scene hint through.
BUTTON_FILL = ((0.227, 0.227, 0.227, 0.97), (0.118, 0.118, 0.118, 0.97))
BUTTON_HOVER = ((0.290, 0.290, 0.290, 0.98), (0.165, 0.165, 0.165, 0.98))
BUTTON_PRESS = ((0.140, 0.140, 0.140, 0.98), (0.090, 0.090, 0.090, 0.98))
PANEL_FILL = ((0.176, 0.176, 0.180, 0.94), (0.082, 0.082, 0.086, 0.95))
POPUP_FILL = ((0.200, 0.200, 0.204, 0.97), (0.100, 0.100, 0.104, 0.98))
FIELD_FILL = ((0.050, 0.050, 0.054, 0.92), (0.082, 0.082, 0.086, 0.92))
INSET_FILL = ((1.0, 1.0, 1.0, 0.055), (1.0, 1.0, 1.0, 0.035))
ACCENT_FILL = ((0.20, 0.62, 1.0, 1.0), (0.02, 0.42, 0.90, 1.0))

BORDER_BUTTON = (1.0, 1.0, 1.0, 0.82)     # the reference button's bright hairline
BORDER_BUTTON_HOVER = (1.0, 1.0, 1.0, 1.0)
BORDER_PANEL = (1.0, 1.0, 1.0, 0.30)
BORDER_SOFT = (1.0, 1.0, 1.0, 0.14)
BORDER_FOCUS = (ACCENT[0], ACCENT[1], ACCENT[2], 1.0)
HAIRLINE = (1.0, 1.0, 1.0, 0.10)

SHADOW = (0.0, 0.0, 0.0, 0.55)

# ---------------------------------------------------------------------- text
TEXT = (0.97, 0.97, 0.98, 1.0)
TEXT_2 = (0.80, 0.81, 0.84, 1.0)
TEXT_3 = (0.58, 0.59, 0.63, 1.0)
TEXT_4 = (0.42, 0.43, 0.47, 1.0)

OK = (0.36, 0.86, 0.52, 1.0)
WARN = (1.0, 0.76, 0.30, 1.0)
ERROR = (1.0, 0.45, 0.42, 1.0)
INFO = TEXT_2
LEVEL = {"info": INFO, "ok": OK, "warn": WARN, "error": ERROR}

# ------------------------------------------------------------------ geometry
R_BUTTON = 8.0
R_PANEL = 12.0
R_FIELD = 8.0
R_PILL = 8.0
R_KEY = 5.0
MARGIN = 16            # distance of floating containers from the window edge
GAP = 8

# ---------------------------------------------------------------- type scale
T_CAPTION = 11
T_SMALL = 12
T_BODY = 13
T_BASE = 14
T_LABEL = 15
T_TITLE = 17
T_HEAD = 20
T_DISPLAY = 40

# weights understood by Overlay2D.text(): regular / medium / semibold / bold
W_REGULAR, W_MEDIUM, W_SEMI, W_BOLD = "regular", "medium", "semibold", "bold"

# ------------------------------------------------------------ menu / garage layer
# Used by ui_kit, menu_ui, the settings pages, the garage and the loading screen.
SIGNAL = (0.36, 0.86, 1.0, 1.0)             # focus colour: brackets, selection edge, icons
SIGNAL_SOFT = (0.36, 0.86, 1.0, 0.16)
ACCENT_SOFT = (ACCENT[0], ACCENT[1], ACCENT[2], 0.20)
ACCENT_FAINT = (ACCENT[0], ACCENT[1], ACCENT[2], 0.08)

SCRIM = (0.016, 0.02, 0.03)                 # veil drawn over the live scene behind a menu
SCRIM_LEFT, SCRIM_RIGHT = 0.90, 0.42
CARD_FILL = ((0.13, 0.135, 0.15, 0.92), (0.07, 0.074, 0.086, 0.94))
CARD_FILL_FOCUS = ((0.15, 0.19, 0.24, 0.96), (0.07, 0.095, 0.13, 0.96))
ROW_FOCUS = ((1.0, 1.0, 1.0, 0.085), (1.0, 1.0, 1.0, 0.045))
ROW_HOVER = ((1.0, 1.0, 1.0, 0.05), (1.0, 1.0, 1.0, 0.025))
TRACK = (1.0, 1.0, 1.0, 0.16)               # slider / progress groove
LOCKED = ERROR                              # unavailable item accent
GOLD = (1.0, 0.8, 0.36, 1.0)

R_CARD = 10.0
R_CHIP = 6.0
TITLE_FONT = 34
