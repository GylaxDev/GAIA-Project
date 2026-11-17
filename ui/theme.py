"""Design tokens for the osm3d interface - an Apple-style "liquid glass" look.

The whole UI is built from a handful of ideas, taken from Apple's Human Interface
Guidelines and the iOS 26 design language:

* **Glass, not borders.**  Surfaces are soft, translucent dark panels with a hair-thin light
  rim and a faint specular line along the top edge.  Shadows are wide and quiet.
* **Big, concentric radii.**  Cards 20, sheets 30, and everything you press is a capsule.
* **Grouped lists.**  Settings and details live in inset grouped cards: one rounded surface,
  rows separated by hairlines, section titles above - never a bordered box per row.
* **One accent.**  System blue for selection and the primary action; green for "on";
  red only for destructive actions.  Colour tiles give categories a quick visual identity.
* **Clear hierarchy.**  Four label levels (primary / secondary / tertiary / quaternary) are
  expressed with opacity, so text stays legible on any scene behind it.

All sizes are in *logical* pixels; ``Overlay2D`` multiplies them by the display scale.
Change a value here and the HUD, the map chrome, the menus and every dialog follow.
Every name that existed in the previous theme is still defined, so older call sites keep
working.
"""
from __future__ import annotations

from ..core.config import ACCENT

# ────────────────────────────────────────────────────────────── spacing (4 pt grid)
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_5 = 20
SPACE_6 = 24
SPACE_8 = 32
SPACE_10 = 40
SPACE_12 = 48
SPACE_16 = 64

# ────────────────────────────────────────────────────────────── corner radii
R_1 = 6.0
R_2 = 10.0
R_3 = 14.0
R_4 = 20.0
R_5 = 26.0
R_6 = 34.0

# Semantic aliases.  Capsules (buttons, chips, fields, switches) are drawn with
# radius = height / 2 by the widgets themselves; these values are for code that needs a number.
R_KEY = 6.0
R_CHIP = 12.0
R_BUTTON = 14.0
R_FIELD = 14.0
R_PILL = 14.0
R_TILE = 0.225          # icon tiles: fraction of the tile size (the app-icon "squircle")
R_CARD = R_4
R_PANEL = R_4
R_POPUP = R_5
R_SHEET = 30.0

# ────────────────────────────────────────────────────────────── system colours (dark)
SYS_BLUE = (0.039, 0.518, 1.000, 1.0)       # #0A84FF
SYS_GREEN = (0.188, 0.820, 0.345, 1.0)      # #30D158
SYS_RED = (1.000, 0.271, 0.227, 1.0)        # #FF453A
SYS_ORANGE = (1.000, 0.624, 0.039, 1.0)     # #FF9F0A
SYS_YELLOW = (1.000, 0.839, 0.039, 1.0)     # #FFD60A
SYS_TEAL = (0.392, 0.824, 1.000, 1.0)       # #64D2FF
SYS_PURPLE = (0.749, 0.353, 0.949, 1.0)     # #BF5AF2
SYS_PINK = (1.000, 0.216, 0.373, 1.0)       # #FF375F
SYS_INDIGO = (0.369, 0.361, 0.902, 1.0)     # #5E5CE6
SYS_GRAY = (0.557, 0.557, 0.576, 1.0)       # #8E8E93

# Colour tiles (flat fill, glyph is always white) - the category identity in lists.
TILE_BLUE = ((0.20, 0.58, 1.00, 1.0), (0.02, 0.44, 0.96, 1.0))
TILE_GREEN = ((0.32, 0.86, 0.46, 1.0), (0.13, 0.72, 0.30, 1.0))
TILE_ORANGE = ((1.00, 0.69, 0.20, 1.0), (1.00, 0.55, 0.00, 1.0))
TILE_RED = ((1.00, 0.40, 0.36, 1.0), (0.96, 0.22, 0.19, 1.0))
TILE_PURPLE = ((0.80, 0.46, 0.97, 1.0), (0.66, 0.28, 0.90, 1.0))
TILE_TEAL = ((0.44, 0.84, 1.00, 1.0), (0.20, 0.70, 0.95, 1.0))
TILE_PINK = ((1.00, 0.33, 0.50, 1.0), (0.96, 0.17, 0.34, 1.0))
TILE_INDIGO = ((0.46, 0.45, 0.95, 1.0), (0.30, 0.29, 0.85, 1.0))
TILE_GRAY = ((0.62, 0.62, 0.65, 1.0), (0.45, 0.45, 0.48, 1.0))

# ────────────────────────────────────────────────────────────── surfaces
# (top, bottom) vertical gradients.  Alpha < 1 lets the 3D scene breathe through.
GLASS = ((0.19, 0.19, 0.21, 0.72), (0.10, 0.10, 0.12, 0.80))           # HUD cards over the scene
GLASS_STRONG = ((0.17, 0.17, 0.19, 0.88), (0.09, 0.09, 0.11, 0.92))    # menus / popovers
SHEET = ((0.15, 0.15, 0.17, 0.95), (0.10, 0.10, 0.12, 0.97))           # full pages and dialogs
SHEET_RAISED = ((0.19, 0.19, 0.21, 0.98), (0.14, 0.14, 0.16, 0.98))    # something on top of a sheet

GROUP = ((1.0, 1.0, 1.0, 0.070), (1.0, 1.0, 1.0, 0.055))              # inset grouped card on a sheet
GROUP_HOVER = ((1.0, 1.0, 1.0, 0.110), (1.0, 1.0, 1.0, 0.095))
GROUP_SELECTED = ((0.039, 0.518, 1.0, 0.22), (0.039, 0.518, 1.0, 0.16))
FIELD = ((0.47, 0.47, 0.50, 0.26), (0.47, 0.47, 0.50, 0.22))          # text fields, segmented tracks

FILL_1 = (0.47, 0.47, 0.50, 0.36)
FILL_2 = (0.47, 0.47, 0.50, 0.30)
FILL_3 = (0.47, 0.47, 0.50, 0.24)
FILL_4 = (0.47, 0.47, 0.50, 0.18)

# Legacy surface names, now mapped onto the glass system.
SURFACE_1 = GLASS
SURFACE_2 = GROUP
SURFACE_3 = FIELD
SURFACE_4 = SHEET

MATERIAL_THIN = ((1.0, 1.0, 1.0, 0.050), (1.0, 1.0, 1.0, 0.035))
MATERIAL_REGULAR = ((1.0, 1.0, 1.0, 0.085), (1.0, 1.0, 1.0, 0.065))
MATERIAL_THICK = ((1.0, 1.0, 1.0, 0.130), (1.0, 1.0, 1.0, 0.105))
MATERIAL_TINTED = ((0.039, 0.518, 1.0, 0.24), (0.039, 0.518, 1.0, 0.18))

# ── Hairlines and rims - always thin, always light
HAIRLINE = (1.0, 1.0, 1.0, 0.085)
SEPARATOR = (1.0, 1.0, 1.0, 0.085)
SPECULAR = (1.0, 1.0, 1.0, 0.20)           # the lit top edge of a glass panel
BORDER_SUBTLE = (1.0, 1.0, 1.0, 0.08)
BORDER_DEFAULT = (1.0, 1.0, 1.0, 0.13)
BORDER_EMPHASIZED = (1.0, 1.0, 1.0, 0.22)
BORDER_STRONG = (1.0, 1.0, 1.0, 0.32)

BUTTON_BORDER = (1.0, 1.0, 1.0, 0.12)
BUTTON_BORDER_HOVER = (1.0, 1.0, 1.0, 0.22)
BUTTON_BORDER_PRESS = (1.0, 1.0, 1.0, 0.28)

# ── Shadows: (offset_y, blur, spread, colour).  Wide and quiet.
SHADOW_1 = (1, 4, 0, (0.0, 0.0, 0.0, 0.16))
SHADOW_2 = (4, 18, 0, (0.0, 0.0, 0.0, 0.26))
SHADOW_3 = (10, 34, 0, (0.0, 0.0, 0.0, 0.36))
SHADOW_4 = (22, 60, 0, (0.0, 0.0, 0.0, 0.46))
INNER_SHADOW = (0, 2, 0, (0.0, 0.0, 0.0, 0.20))

# ── Accent
ACCENT_BLUE = SYS_BLUE
ACCENT_BLUE_SOFT = (0.039, 0.518, 1.0, 0.18)
ACCENT_BLUE_FAINT = (0.039, 0.518, 1.0, 0.09)
ACCENT_BLUE_MEDIUM = (0.039, 0.518, 1.0, 0.32)
ACCENT_BLUE_FILL = ((0.21, 0.58, 1.00, 1.0), (0.03, 0.46, 1.00, 1.0))   # primary button

# ── Semantic colours
OK_GREEN = SYS_GREEN
OK_GREEN_SOFT = (0.188, 0.820, 0.345, 0.16)
WARN_AMBER = SYS_ORANGE
WARN_AMBER_SOFT = (1.0, 0.624, 0.039, 0.16)
ERROR_RED = SYS_RED
ERROR_RED_SOFT = (1.0, 0.271, 0.227, 0.16)
DESTRUCTIVE_FILL = ((1.00, 0.36, 0.32, 1.0), (0.94, 0.20, 0.17, 1.0))

# ── Text: one white, four opacities.  Readable on glass, on the scene, on a sheet.
TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)
TEXT_SECONDARY = (0.92, 0.92, 0.96, 0.74)
TEXT_TERTIARY = (0.92, 0.92, 0.96, 0.52)
TEXT_QUATERNARY = (0.92, 0.92, 0.96, 0.30)
TEXT_INVERTED = (1.0, 1.0, 1.0, 1.0)        # label on an accent-filled control

# Focus ring - a clear accent outline, never a glow
FOCUS_RING = (0.039, 0.518, 1.0, 0.90)
FOCUS_RING_STRONG = (0.039, 0.518, 1.0, 1.0)

# ────────────────────────────────────────────────────────────── scrim / backdrop
SCRIM_BASE = (0.0, 0.0, 0.01)
SCRIM_TOP_OPACITY = 0.66
SCRIM_BOTTOM_OPACITY = 0.30
SCRIM_EDGE_FADE = 120

# ────────────────────────────────────────────────────────────── type scale
# Inter stands in for SF Pro.  Sizes follow Apple's text styles (caption 2 ... large title).
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

T_CAPTION = T_CAPTION_2
T_SMALL = T_FOOTNOTE
T_BASE = T_BODY
T_LABEL = T_CALLOUT
T_TITLE = T_TITLE_1
T_HEAD = T_TITLE_2
T_TITLE_FONT = T_DISPLAY

# ────────────────────────────────────────────────────────────── motion
# (damping, stiffness) pairs for ``Overlay2D.anim_spring``
SPRING_GENTLE = (0.85, 180)
SPRING_QUICK = (0.80, 280)
SPRING_BOUNCY = (0.70, 350)
SPRING_STIFF = (0.90, 500)
EASE_OUT_EXPO = 3.0

# ────────────────────────────────────────────────────────────── layout
MARGIN = SPACE_4
GAP = SPACE_2
GAP_SM = SPACE_1
GAP_LG = SPACE_3
CONTENT_MAX_WIDTH = 1040
PANEL_MAX_WIDTH = 1120

# ────────────────────────────────────────────────────────────── legacy aliases
SIGNAL = ACCENT_BLUE
SIGNAL_SOFT = ACCENT_BLUE_SOFT
ACCENT_SOFT = ACCENT_BLUE_SOFT
ACCENT_FAINT = ACCENT_BLUE_FAINT
ACCENT_FILL = ACCENT_BLUE_FILL
SCRIM = SCRIM_BASE
SCRIM_LEFT, SCRIM_RIGHT = SCRIM_TOP_OPACITY, SCRIM_BOTTOM_OPACITY

BUTTON_FILL = GLASS
BUTTON_HOVER = ((0.26, 0.26, 0.28, 0.80), (0.16, 0.16, 0.18, 0.86))
BUTTON_PRESS = ((0.12, 0.12, 0.14, 0.86), (0.08, 0.08, 0.10, 0.90))
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
ROW_FOCUS = (1.0, 1.0, 1.0, 0.085)
ROW_HOVER = (1.0, 1.0, 1.0, 0.060)
TRACK = (1.0, 1.0, 1.0, 0.18)
LOCKED = ERROR_RED
GOLD = SYS_YELLOW

TITLE_FONT = T_DISPLAY
