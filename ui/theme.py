"""Design tokens for the Project GAIA interface — Apple "liquid glass".

The language:

* **Rounded, translucent, layered.**  Panels are frosted glass: a soft dark tint,
  a hairline light border and a wide, low-opacity shadow, so the scene shows
  through and the interface reads as a separate sheet floating above it.
* **One accent.**  Apple system blue for selection, links, primary actions and
  the live value of a control.  Everything else is neutral white at a few
  well-known opacities.
* **Optical type.**  Regular / medium / semibold weights, generous line height,
  sentence case for titles, no tracked-out capitals, no terminal brackets.
* **Grouped lists.**  Rows live inside rounded cards and are separated by
  hairlines that stop short of the edges, exactly like Settings.
* **Motion.**  UIKit-style timing curves and spring presets; nothing snaps.

All sizes are logical pixels.  Every token the previous themes defined still
exists, so older call sites keep working; values now map onto this palette.
"""
from __future__ import annotations

from ..core.config import ACCENT

# ────────────────────────────────────────────────────────────── spacing (4 pt grid)
SPACE_1, SPACE_2, SPACE_3, SPACE_4, SPACE_5 = 4, 8, 12, 16, 20
SPACE_6, SPACE_8, SPACE_10, SPACE_12, SPACE_16 = 24, 32, 40, 48, 64

# ────────────────────────────────────────────────────────────── corner radii
R_1, R_2, R_3 = 4.0, 6.0, 8.0
R_4, R_5, R_6 = 10.0, 12.0, 14.0
R_KEY = 6.0
R_CHIP = 12.0
R_BUTTON = 14.0
R_FIELD = 12.0
R_PILL = 999.0                   # clamped to half the height: a true capsule
R_TILE = 12.0
R_CARD = 18.0
R_PANEL = 22.0
R_POPUP = 26.0
R_SHEET = 28.0

# ────────────────────────────────────────────────────────────── palette
BLACK = (0.0, 0.0, 0.0, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)

SYS_BLUE = (0.039, 0.518, 1.0, 1.0)          # #0A84FF — dark-appearance system blue
SYS_BLUE_DEEP = (0.0, 0.36, 0.86, 1.0)
SYS_GREEN = (0.188, 0.820, 0.345, 1.0)       # #30D158
SYS_RED = (1.0, 0.271, 0.227, 1.0)           # #FF453A
SYS_ORANGE = (1.0, 0.584, 0.0, 1.0)          # #FF9500
SYS_YELLOW = (1.0, 0.839, 0.039, 1.0)        # #FFD60A
SYS_TEAL = (0.275, 0.812, 0.816, 1.0)        # #46D1D0
SYS_MINT = (0.400, 0.867, 0.714, 1.0)
SYS_PURPLE = (0.749, 0.353, 0.949, 1.0)      # #BF5AF2
SYS_PINK = (1.0, 0.392, 0.510, 1.0)          # #FF6482
SYS_INDIGO = (0.369, 0.361, 0.902, 1.0)      # #5E5CE6
SYS_GRAY = (0.557, 0.557, 0.576, 1.0)

# Compatibility aliases used across older call sites.
WD_BLUE = SYS_BLUE
WD_BLUE_DEEP = SYS_BLUE_DEEP
WD_GREEN = SYS_GREEN
WD_GREEN_DIM = (0.11, 0.55, 0.24, 1.0)
WD_RED = SYS_RED
WD_AMBER = SYS_ORANGE

# Icon tiles: rounded squares with a gentle two-stop gradient.
TILE_BLUE = ((0.243, 0.647, 1.0, 1.0), (0.039, 0.435, 0.941, 1.0))
TILE_GREEN = ((0.365, 0.878, 0.478, 1.0), (0.110, 0.678, 0.259, 1.0))
TILE_RED = ((1.0, 0.404, 0.337, 1.0), (0.898, 0.184, 0.157, 1.0))
TILE_ORANGE = ((1.0, 0.690, 0.243, 1.0), (0.949, 0.467, 0.0, 1.0))
TILE_PURPLE = ((0.812, 0.478, 1.0, 1.0), (0.604, 0.208, 0.863, 1.0))
TILE_TEAL = ((0.388, 0.878, 0.878, 1.0), (0.098, 0.647, 0.678, 1.0))
TILE_PINK = ((1.0, 0.494, 0.612, 1.0), (0.929, 0.220, 0.400, 1.0))
TILE_INDIGO = ((0.478, 0.451, 0.980, 1.0), (0.243, 0.220, 0.780, 1.0))
TILE_GRAY = ((0.365, 0.373, 0.400, 1.0), (0.239, 0.247, 0.271, 1.0))
TILE_CYCLE = (TILE_BLUE, TILE_GREEN, TILE_ORANGE, TILE_PURPLE, TILE_PINK, TILE_TEAL, TILE_INDIGO)

# ────────────────────────────────────────────────────────────── surfaces
# Rows and cards drawn over the scene: translucent white film on dark glass.
BAR = (1.0, 1.0, 1.0, 0.06)
BAR_HOVER = (1.0, 1.0, 1.0, 0.13)
BAR_SELECTED = SYS_BLUE
PANEL_BG = (0.075, 0.082, 0.098, 0.78)
SHEET_BG = (0.055, 0.060, 0.072, 0.88)

_flat = lambda c: (c, c)
GLASS = ((0.098, 0.100, 0.108, 0.50), (0.058, 0.060, 0.066, 0.56))      # neutral, lets the scene read through
GLASS_STRONG = ((0.086, 0.094, 0.114, 0.86), (0.047, 0.052, 0.063, 0.90))
GLASS_LIGHT = ((1.0, 1.0, 1.0, 0.16), (1.0, 1.0, 1.0, 0.09))
SHEET = ((0.086, 0.094, 0.114, 0.90), (0.043, 0.047, 0.055, 0.94))
SHEET_RAISED = ((0.125, 0.137, 0.161, 0.88), (0.075, 0.082, 0.098, 0.92))
GROUP = ((1.0, 1.0, 1.0, 0.075), (1.0, 1.0, 1.0, 0.055))
GROUP_HOVER = ((1.0, 1.0, 1.0, 0.15), (1.0, 1.0, 1.0, 0.11))
GROUP_SELECTED = (SYS_BLUE[0], SYS_BLUE[1], SYS_BLUE[2], 0.90)
FIELD = ((1.0, 1.0, 1.0, 0.10), (1.0, 1.0, 1.0, 0.07))

FILL_1 = (1.0, 1.0, 1.0, 0.16)
FILL_2 = (1.0, 1.0, 1.0, 0.11)
FILL_3 = (1.0, 1.0, 1.0, 0.07)
FILL_4 = (1.0, 1.0, 1.0, 0.04)

SURFACE_1, SURFACE_2, SURFACE_3, SURFACE_4 = GLASS, GROUP, FIELD, SHEET
MATERIAL_THIN = _flat((1.0, 1.0, 1.0, 0.055))
MATERIAL_REGULAR = _flat((1.0, 1.0, 1.0, 0.085))
MATERIAL_THICK = _flat((1.0, 1.0, 1.0, 0.14))
MATERIAL_TINTED = _flat((0.039, 0.518, 1.0, 0.30))

# ── Lines: hairlines that catch the light rather than rule the page
HAIRLINE = (1.0, 1.0, 1.0, 0.10)
SEPARATOR = (1.0, 1.0, 1.0, 0.09)
SPECULAR = (1.0, 1.0, 1.0, 0.18)
LINE = (1.0, 1.0, 1.0, 0.16)
LINE_DIM = (1.0, 1.0, 1.0, 0.10)
BORDER_SUBTLE = (1.0, 1.0, 1.0, 0.08)
BORDER_DEFAULT = (1.0, 1.0, 1.0, 0.10)
BORDER_EMPHASIZED = (1.0, 1.0, 1.0, 0.26)
BORDER_STRONG = (1.0, 1.0, 1.0, 0.44)
BUTTON_BORDER = (1.0, 1.0, 1.0, 0.10)
BUTTON_BORDER_HOVER = (1.0, 1.0, 1.0, 0.20)
BUTTON_BORDER_PRESS = (1.0, 1.0, 1.0, 0.28)

# Shadows: (offset_y, blur, spread, colour).  Wide and soft, never hard.
SHADOW_1 = (2.0, 8.0, 0.0, (0.0, 0.0, 0.0, 0.22))
SHADOW_2 = (6.0, 20.0, 2.0, (0.0, 0.0, 0.0, 0.30))
SHADOW_3 = (12.0, 34.0, 4.0, (0.0, 0.0, 0.0, 0.38))
SHADOW_4 = (20.0, 56.0, 8.0, (0.0, 0.0, 0.0, 0.46))
INNER_SHADOW = (0.0, 0.0, 0.0, (0.0, 0.0, 0.0, 0.0))

# ── Accent
ACCENT_BLUE = SYS_BLUE
ACCENT_BLUE_SOFT = (0.039, 0.518, 1.0, 0.28)
ACCENT_BLUE_FAINT = (0.039, 0.518, 1.0, 0.14)
ACCENT_BLUE_MEDIUM = (0.039, 0.518, 1.0, 0.55)
ACCENT_BLUE_FILL = (SYS_BLUE, SYS_BLUE_DEEP)

# ── Semantic
OK_GREEN = SYS_GREEN
OK_GREEN_SOFT = (0.188, 0.820, 0.345, 0.18)
WARN_AMBER = SYS_ORANGE
WARN_AMBER_SOFT = (1.0, 0.584, 0.0, 0.18)
ERROR_RED = SYS_RED
ERROR_RED_SOFT = (1.0, 0.271, 0.227, 0.18)
DESTRUCTIVE_FILL = ((1.0, 0.361, 0.318, 1.0), (0.855, 0.157, 0.129, 1.0))

# ── Text
TEXT_PRIMARY = (1.0, 1.0, 1.0, 0.96)
TEXT_SECONDARY = (1.0, 1.0, 1.0, 0.70)
TEXT_TERTIARY = (1.0, 1.0, 1.0, 0.50)
TEXT_QUATERNARY = (1.0, 1.0, 1.0, 0.30)
TEXT_INVERTED = (0.0, 0.0, 0.0, 0.92)
TEXT_VALUE = TEXT_SECONDARY

FOCUS_RING = (0.039, 0.518, 1.0, 0.95)
FOCUS_RING_STRONG = SYS_BLUE

# ────────────────────────────────────────────────────────────── scrim / backdrop
SCRIM_BASE = (0.02, 0.025, 0.035)
SCRIM_TOP_OPACITY = 0.42
SCRIM_BOTTOM_OPACITY = 0.30
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

# Optical tracking.  Titles sit slightly tighter, small caps labels neutral.
LETTER_TRACK = 0.0
TRACK_TITLE = -0.02

# ────────────────────────────────────────────────────────────── motion
# UIKit curves.  ``EASE_*`` are cubic-bezier control points; the plain floats are
# kept for code that passes a power to ``ease_out_expo``.
EASE_STANDARD = (0.25, 0.10, 0.25, 1.0)       # default
EASE_EMPHASIZED = (0.20, 0.00, 0.00, 1.00)    # sheet / page transitions
EASE_DECELERATE = (0.00, 0.00, 0.20, 1.00)    # entrance
EASE_ACCELERATE = (0.40, 0.00, 1.00, 1.00)    # exit
SPRING_GENTLE = (0.90, 260)
SPRING_QUICK = (0.90, 420)
SPRING_BOUNCY = (0.82, 380)
SPRING_STIFF = (0.95, 600)
EASE_OUT_EXPO = 3.0

DURATION_FAST = 0.22
DURATION_STANDARD = 0.32
DURATION_SLOW = 0.46

# ────────────────────────────────────────────────────────────── layout
MARGIN = SPACE_4
GAP = SPACE_2
GAP_SM = SPACE_1
GAP_LG = SPACE_3
CONTENT_MAX_WIDTH = 1120
PANEL_MAX_WIDTH = 1180
HINT_BAR_H = 36                                # reserve along the bottom of menu pages

# ────────────────────────────────────────────────────────────── legacy aliases
SIGNAL = ACCENT_BLUE
SIGNAL_SOFT = ACCENT_BLUE_SOFT
ACCENT_SOFT = ACCENT_BLUE_SOFT
ACCENT_FAINT = ACCENT_BLUE_FAINT
ACCENT_FILL = ACCENT_BLUE_FILL
SCRIM = SCRIM_BASE
SCRIM_LEFT, SCRIM_RIGHT = SCRIM_TOP_OPACITY, SCRIM_BOTTOM_OPACITY

BUTTON_FILL = ((1.0, 1.0, 1.0, 0.11), (1.0, 1.0, 1.0, 0.07))
BUTTON_HOVER = ((1.0, 1.0, 1.0, 0.19), (1.0, 1.0, 1.0, 0.13))
BUTTON_PRESS = ((1.0, 1.0, 1.0, 0.26), (1.0, 1.0, 1.0, 0.20))
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
ROW_FOCUS = SYS_BLUE
ROW_HOVER = (1.0, 1.0, 1.0, 0.12)
TRACK = (1.0, 1.0, 1.0, 0.18)
LOCKED = ERROR_RED
GOLD = SYS_YELLOW

TITLE_FONT = T_DISPLAY
