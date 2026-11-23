"""Design tokens for the Project GAIA interface - a DedSec / "hacker terminal" look.

The language, taken from the Watch Dogs 2 front end:

* **Black and flat.**  A black field, near-black panels, no gradients, no rounded corners,
  no shadows.  Depth comes from thin white outlines and dotted halftone patterns.
* **One loud colour.**  Selection is a solid, saturated blue bar with white text.
* **Terminal values.**  Settings read like a console: green for "on" and live values, red for
  "off" and warnings, white for everything else.  Headings are tracked-out capitals.
* **Dots.**  Backdrops are dot-matrix: a world globe built from map data, concentric dotted
  rings, sparse grids.  Controls are drawn with dashes and brackets: ``[-----X-----]``.
* **Hint strip.**  A white bar along the bottom holds the version and the key hints.

All sizes are logical pixels.  Every token the previous themes defined still exists, so older
call sites keep working; their values now map onto this palette.
"""
from __future__ import annotations

from ..core.config import ACCENT

# ────────────────────────────────────────────────────────────── spacing (4 pt grid)
SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5 = 4, 8, 12, 16, 20
SPACE_6, SPACE_8, SPACE_10, SPACE_12, SPACE_16 = 24, 32, 40, 48, 64

# ────────────────────────────────────────────────────────────── corner radii
# Everything is square.  The tokens stay (code passes them around) but are zero.
R_1 = R_2 = R_3 = R_4 = R_5 = R_6 = 0.0
R_KEY = 2.0                      # the only soft corner: the little white key labels
R_CHIP = R_BUTTON = R_FIELD = R_PILL = 0.0
R_TILE = 0.0
R_CARD = R_PANEL = R_POPUP = R_SHEET = 0.0

# ────────────────────────────────────────────────────────────── palette
BLACK = (0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)
WD_BLUE = (0.0, 0.0, 1.0, 1.0)                 # selection bar
WD_BLUE_DEEP = (0.0, 0.0, 0.62, 1.0)
WD_GREEN = (0.10, 1.00, 0.28, 1.0)             # values, tab labels, "on"
WD_GREEN_DIM = (0.06, 0.60, 0.18, 1.0)
WD_RED = (1.00, 0.13, 0.13, 1.0)               # "off", errors, destructive
WD_AMBER = (1.00, 0.76, 0.10, 1.0)             # warnings

SYS_BLUE = WD_BLUE
SYS_GREEN = WD_GREEN
SYS_RED = WD_RED
SYS_ORANGE = WD_AMBER
SYS_YELLOW = WD_AMBER
SYS_TEAL = WD_GREEN
SYS_PURPLE = WD_BLUE
SYS_PINK = WD_RED
SYS_INDIGO = WD_BLUE
SYS_GRAY = (0.55, 0.55, 0.55, 1.0)

# Icon tiles: flat squares.  Only the blue / red / green ones carry colour.
_GRAY_TILE = ((0.13, 0.13, 0.13, 1.0), (0.13, 0.13, 0.13, 1.0))
TILE_BLUE = ((0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0))
TILE_GREEN = ((0.04, 0.50, 0.16, 1.0), (0.04, 0.50, 0.16, 1.0))
TILE_RED = ((0.78, 0.08, 0.08, 1.0), (0.78, 0.08, 0.08, 1.0))
TILE_ORANGE = TILE_PURPLE = TILE_TEAL = TILE_PINK = TILE_INDIGO = TILE_GRAY = _GRAY_TILE
TILE_CYCLE = (_GRAY_TILE,)

# ────────────────────────────────────────────────────────────── surfaces
BAR = (0.075, 0.075, 0.075, 1.0)               # a row / menu item
BAR_HOVER = (0.150, 0.150, 0.150, 1.0)
BAR_SELECTED = WD_BLUE
PANEL_BG = (0.028, 0.028, 0.028, 0.95)         # info panels
SHEET_BG = (0.012, 0.012, 0.012, 0.97)         # full pages

_flat = lambda c: (c, c)
GLASS = _flat((0.030, 0.030, 0.030, 0.92))
GLASS_STRONG = _flat((0.020, 0.020, 0.020, 0.96))
SHEET = _flat(SHEET_BG)
SHEET_RAISED = _flat((0.050, 0.050, 0.050, 0.98))
GROUP = _flat((0.050, 0.050, 0.050, 0.90))
GROUP_HOVER = _flat((0.110, 0.110, 0.110, 0.95))
GROUP_SELECTED = _flat(WD_BLUE)
FIELD = _flat((0.090, 0.090, 0.090, 1.0))

FILL_1 = (0.20, 0.20, 0.20, 1.0)
FILL_2 = (0.15, 0.15, 0.15, 1.0)
FILL_3 = (0.11, 0.11, 0.11, 1.0)
FILL_4 = (0.08, 0.08, 0.08, 1.0)

SURFACE_1, SURFACE_2, SURFACE_3, SURFACE_4 = GLASS, GROUP, FIELD, SHEET
MATERIAL_THIN = _flat((1.0, 1.0, 1.0, 0.04))
MATERIAL_REGULAR = _flat((1.0, 1.0, 1.0, 0.07))
MATERIAL_THICK = _flat((1.0, 1.0, 1.0, 0.12))
MATERIAL_TINTED = _flat((0.0, 0.0, 1.0, 0.55))

# ── Lines: thin and light
HAIRLINE = (1.0, 1.0, 1.0, 0.12)
SEPARATOR = (1.0, 1.0, 1.0, 0.10)
SPECULAR = (1.0, 1.0, 1.0, 0.0)
LINE = (1.0, 1.0, 1.0, 0.85)                   # panel frames
LINE_DIM = (1.0, 1.0, 1.0, 0.22)
BORDER_SUBTLE = (1.0, 1.0, 1.0, 0.12)
BORDER_DEFAULT = (1.0, 1.0, 1.0, 0.28)
BORDER_EMPHASIZED = (1.0, 1.0, 1.0, 0.60)
BORDER_STRONG = (1.0, 1.0, 1.0, 0.90)
BUTTON_BORDER = (1.0, 1.0, 1.0, 0.0)
BUTTON_BORDER_HOVER = (1.0, 1.0, 1.0, 0.0)
BUTTON_BORDER_PRESS = (1.0, 1.0, 1.0, 0.0)

# No shadows: tuples keep the (offset_y, blur, spread, colour) shape, fully transparent.
_NO_SHADOW = (0, 0, 0, (0.0, 0.0, 0.0, 0.0))
SHADOW_1 = SHADOW_2 = SHADOW_3 = SHADOW_4 = INNER_SHADOW = _NO_SHADOW

# ── Accent
ACCENT_BLUE = WD_BLUE
ACCENT_BLUE_SOFT = (0.0, 0.0, 1.0, 0.35)
ACCENT_BLUE_FAINT = (0.0, 0.0, 1.0, 0.16)
ACCENT_BLUE_MEDIUM = (0.0, 0.0, 1.0, 0.60)
ACCENT_BLUE_FILL = (WD_BLUE, WD_BLUE)

# ── Semantic
OK_GREEN = WD_GREEN
OK_GREEN_SOFT = (0.10, 1.0, 0.28, 0.16)
WARN_AMBER = WD_AMBER
WARN_AMBER_SOFT = (1.0, 0.76, 0.10, 0.16)
ERROR_RED = WD_RED
ERROR_RED_SOFT = (1.0, 0.13, 0.13, 0.16)
DESTRUCTIVE_FILL = (WD_RED, WD_RED)

# ── Text
TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)
TEXT_SECONDARY = (0.80, 0.80, 0.80, 1.0)
TEXT_TERTIARY = (0.55, 0.55, 0.55, 1.0)
TEXT_QUATERNARY = (0.34, 0.34, 0.34, 1.0)
TEXT_INVERTED = (0.0, 0.0, 0.0, 1.0)           # text on the white hint strip
TEXT_VALUE = WD_GREEN

FOCUS_RING = (1.0, 1.0, 1.0, 0.9)
FOCUS_RING_STRONG = (1.0, 1.0, 1.0, 1.0)

# ────────────────────────────────────────────────────────────── scrim / backdrop
SCRIM_BASE = (0.0, 0.0, 0.0)
SCRIM_TOP_OPACITY = 0.78
SCRIM_BOTTOM_OPACITY = 0.62
SCRIM_EDGE_FADE = 0

# ────────────────────────────────────────────────────────────── type scale
T_CAPTION_1 = 11
T_CAPTION_2 = 12
T_FOOTNOTE = 13
T_BODY = 14
T_CALLOUT = 15
T_SUBHEAD = 16
T_TITLE_1 = 17
T_TITLE_2 = 20
T_TITLE_3 = 24
T_DISPLAY = 34
T_LARGE_TITLE = 40
T_CAPTION, T_SMALL, T_BASE, T_LABEL, T_TITLE, T_HEAD, T_TITLE_FONT = (
    T_CAPTION_2, T_FOOTNOTE, T_BODY, T_CALLOUT, T_TITLE_1, T_TITLE_2, T_DISPLAY)

# Tracking (extra letter spacing, em) for the capitals used in headings and tabs
LETTER_TRACK = 0.14

# ────────────────────────────────────────────────────────────── motion
SPRING_GENTLE = (0.90, 260)
SPRING_QUICK = (0.90, 420)
SPRING_BOUNCY = (0.85, 420)
SPRING_STIFF = (0.95, 600)
EASE_OUT_EXPO = 3.0

# ────────────────────────────────────────────────────────────── layout
MARGIN = SPACE_4
GAP = SPACE_2
GAP_SM = SPACE_1
GAP_LG = SPACE_3
CONTENT_MAX_WIDTH = 1120
PANEL_MAX_WIDTH = 1180
HINT_BAR_H = 36                                # the white strip along the bottom of menu pages

# ────────────────────────────────────────────────────────────── legacy aliases
SIGNAL = ACCENT_BLUE
SIGNAL_SOFT = ACCENT_BLUE_SOFT
ACCENT_SOFT = ACCENT_BLUE_SOFT
ACCENT_FAINT = ACCENT_BLUE_FAINT
ACCENT_FILL = ACCENT_BLUE_FILL
SCRIM = SCRIM_BASE
SCRIM_LEFT, SCRIM_RIGHT = SCRIM_TOP_OPACITY, SCRIM_BOTTOM_OPACITY

BUTTON_FILL = _flat(BAR)
BUTTON_HOVER = _flat(BAR_HOVER)
BUTTON_PRESS = _flat(WD_BLUE_DEEP)
PANEL_FILL = GLASS
POPUP_FILL = GLASS_STRONG
FIELD_FILL = FIELD
INSET_FILL = MATERIAL_REGULAR

BORDER_BUTTON = BUTTON_BORDER
BORDER_BUTTON_HOVER = BUTTON_BORDER_HOVER
BORDER_PANEL = BORDER_DEFAULT
BORDER_SOFT = BORDER_SUBTLE
BORDER_FOCUS = FOCUS_RING_STRONG

SHADOW = SHADOW_2[3]

TEXT = TEXT_PRIMARY
TEXT_2 = TEXT_SECONDARY
TEXT_3 = TEXT_TERTIARY
TEXT_4 = TEXT_QUATERNARY

OK = OK_GREEN
WARN = WARN_AMBER
ERROR = ERROR_RED
INFO = TEXT_SECONDARY
LEVEL = {"info": INFO, "ok": OK, "warn": WARN, "error": ERROR}

W_REGULAR, W_MEDIUM, W_SEMI, W_BOLD = "regular", "medium", "semibold", "bold"

CARD_FILL = GROUP
CARD_FILL_FOCUS = GROUP_SELECTED
ROW_FOCUS = (0.0, 0.0, 1.0, 1.0)
ROW_HOVER = (0.15, 0.15, 0.15, 1.0)
TRACK = (1.0, 1.0, 1.0, 0.22)
LOCKED = ERROR_RED
GOLD = WD_AMBER

TITLE_FONT = T_DISPLAY
