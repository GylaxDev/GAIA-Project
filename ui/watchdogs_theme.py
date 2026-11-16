"""Watch Dogs themed design tokens and utilities.

A cyberpunk/hacker aesthetic with:
- Signature cyan (#00FFFF) primary accent
- Deep near-black backgrounds with subtle gradients
- Geometric corner brackets as focus indicators
- Scanline and glitch effects
- Monospace technical typography
- Holographic/AR overlay feel
"""
from __future__ import annotations

from . import theme as T
from ..core.config import ACCENT


# ========================================================================
# WATCH DOGS COLOR PALETTE
# ========================================================================

# Primary accent - signature Watch Dogs cyan
WD_CYAN = (0.0, 1.0, 1.0, 1.0)          # #00FFFF - main accent
WD_CYAN_DIM = (0.0, 0.85, 0.85, 1.0)    # dimmed for backgrounds
WD_CYAN_BRIGHT = (0.3, 1.0, 1.0, 1.0)   # bright highlights
WD_CYAN_SOFT = (0.0, 1.0, 1.0, 0.15)    # subtle glow
WD_CYAN_FAINT = (0.0, 1.0, 1.0, 0.06)   # very subtle

# Secondary accents
WD_ORANGE = (1.0, 0.55, 0.15, 1.0)      # warning/hack actions
WD_ORANGE_SOFT = (1.0, 0.55, 0.15, 0.15)
WD_GREEN = (0.36, 1.0, 0.55, 1.0)       # success/safe
WD_GREEN_SOFT = (0.36, 1.0, 0.55, 0.15)
WD_PURPLE = (0.8, 0.4, 1.0, 1.0)        # special/premium
WD_PURPLE_SOFT = (0.8, 0.4, 1.0, 0.15)

# Neutral tones - deep tech blacks
WD_BG_DEEP = (0.02, 0.025, 0.03, 1.0)       # deepest background
WD_BG_DARK = (0.05, 0.06, 0.07, 1.0)        # card/panel background
WD_BG_MEDIUM = (0.08, 0.09, 0.11, 1.0)      # elevated surfaces
WD_BG_LIGHT = (0.12, 0.13, 0.15, 1.0)       # highest elevation

# Gradient pairs for surfaces (top, bottom)
WD_PANEL_GRADIENT = (
    (0.08, 0.09, 0.11, 0.96),  # top
    (0.03, 0.035, 0.045, 0.98) # bottom
)
WD_CARD_GRADIENT = (
    (0.06, 0.07, 0.09, 0.94),
    (0.02, 0.025, 0.03, 0.96)
)
WD_BUTTON_GRADIENT = (
    (0.10, 0.11, 0.13, 0.97),
    (0.04, 0.05, 0.06, 0.97)
)
WD_BUTTON_HOVER_GRADIENT = (
    (0.14, 0.16, 0.18, 0.98),
    (0.06, 0.08, 0.10, 0.98)
)
WD_BUTTON_PRESS_GRADIENT = (
    (0.05, 0.06, 0.07, 0.98),
    (0.02, 0.025, 0.03, 0.98)
)
WD_ACCENT_GRADIENT = (
    (0.0, 0.65, 0.65, 1.0),
    (0.0, 0.4, 0.55, 1.0)
)
WD_ACCENT_HOVER_GRADIENT = (
    (0.1, 0.85, 0.85, 1.0),
    (0.0, 0.6, 0.7, 1.0)
)

# Borders - signature cyan hairlines
WD_BORDER_PRIMARY = (0.0, 1.0, 1.0, 0.9)       # main interactive borders
WD_BORDER_SECONDARY = (0.0, 1.0, 1.0, 0.4)     # subtle dividers
WD_BORDER_SUBTLE = (1.0, 1.0, 1.0, 0.08)       # barely visible
WD_BORDER_FOCUS = WD_CYAN                       # focus state

# Text colors
WD_TEXT_PRIMARY = (0.95, 1.0, 1.0, 1.0)    # main text - slight cyan tint
WD_TEXT_SECONDARY = (0.75, 0.9, 0.92, 1.0) # secondary info
WD_TEXT_MUTED = (0.5, 0.65, 0.68, 1.0)     # disabled/muted
WD_TEXT_DISABLED = (0.35, 0.45, 0.48, 1.0) # disabled state

# Semantic status colors (with cyan tint)
WD_OK = (0.2, 1.0, 0.6, 1.0)
WD_WARN = (1.0, 0.7, 0.2, 1.0)
WD_ERROR = (1.0, 0.3, 0.4, 1.0)
WD_INFO = WD_CYAN

# ========================================================================
# GEOMETRY
# ========================================================================

WD_R_SMALL = 4.0
WD_R_MEDIUM = 8.0
WD_R_LARGE = 12.0
WD_R_PILL = 10.0
WD_R_CARD = 8.0

WD_MARGIN = 16
WD_GAP = 12
WD_GAP_SM = 8

# ========================================================================
# TYPE SCALE
# ========================================================================

WD_T_MICRO = 9
WD_T_CAPTION = 10
WD_T_SMALL = 11
WD_T_BODY = 12
WD_T_BASE = 13
WD_T_LABEL = 14
WD_T_HEAD = 17
WD_T_TITLE = 20
WD_DISPLAY = 36

# ========================================================================
# ANIMATION TIMING
# ========================================================================

WD_ANIM_FAST = 24.0      # hover, focus
WD_ANIM_NORMAL = 16.0    # transitions
WD_ANIM_SLOW = 10.0      # page transitions
WD_STAGGER_DELAY = 0.03  # per-element stagger

# ========================================================================
# EFFECTS
# ========================================================================

WD_SHADOW_SOFT = (0.0, 0.0, 0.0, 0.4)
WD_SHADOW_MEDIUM = (0.0, 0.0, 0.0, 0.5)
WD_SHADOW_STRONG = (0.0, 0.0, 0.0, 0.65)

WD_GLOW_SUBTLE = 0.08
WD_GLOW_NORMAL = 0.18
WD_GLOW_INTENSE = 0.35

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(c0: tuple, c1: tuple, t: float) -> tuple:
    """Linear interpolation between two RGBA colors."""
    t = max(0.0, min(1.0, t))
    return tuple(a + (b - a) * t for a, b in zip(c0, c1))


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - 2.0 * (1.0 - t) ** 2


def stagger(index: int, delay: float = WD_STAGGER_DELAY) -> float:
    """Calculate animation delay for staggered entrance."""
    return index * delay


def glow_color(base: tuple, intensity: float) -> tuple:
    """Add glow effect to a color."""
    return (base[0], base[1], base[2], min(1.0, base[3] + intensity))


# ========================================================================
# COMPATIBILITY - Map to existing theme where needed
# ========================================================================

# These map to the original theme tokens for backwards compatibility
COMPAT = {
    # Surfaces
    "BUTTON_FILL": WD_BUTTON_GRADIENT,
    "BUTTON_HOVER": WD_BUTTON_HOVER_GRADIENT,
    "BUTTON_PRESS": WD_BUTTON_PRESS_GRADIENT,
    "PANEL_FILL": WD_PANEL_GRADIENT,
    "POPUP_FILL": WD_PANEL_GRADIENT,
    "FIELD_FILL": WD_CARD_GRADIENT,
    "INSET_FILL": ((1.0, 1.0, 1.0, 0.04), (1.0, 1.0, 1.0, 0.02)),
    "ACCENT_FILL": WD_ACCENT_GRADIENT,

    # Borders
    "BORDER_BUTTON": WD_BORDER_PRIMARY,
    "BORDER_BUTTON_HOUS": WD_CYAN_BRIGHT,
    "BORDER_PANEL": WD_BORDER_SECONDARY,
    "BORDER_SOFT": WD_BORDER_SUBTLE,
    "BORDER_FOCUS": WD_BORDER_FOCUS,

    # Text
    "TEXT": WD_TEXT_PRIMARY,
    "TEXT_2": WD_TEXT_SECONDARY,
    "TEXT_3": WD_TEXT_MUTED,
    "TEXT_4": WD_TEXT_DISABLED,

    # Status
    "OK": WD_OK,
    "WARN": WD_WARN,
    "ERROR": WD_ERROR,
    "INFO": WD_INFO,

    # Signals
    "SIGNAL": WD_CYAN,
    "SIGNAL_SOFT": WD_CYAN_SOFT,

    # Scrim
    "SCRIM": (0.0, 0.005, 0.01),
    "SCRIM_LEFT": 0.85,
    "SCRIM_RIGHT": 0.35,

    # Geometry
    "R_BUTTON": WD_R_MEDIUM,
    "R_PANEL": WD_R_LARGE,
    "R_FIELD": WD_R_MEDIUM,
    "R_PILL": WD_R_PILL,
    "R_CARD": WD_R_CARD,
    "MARGIN": WD_MARGIN,
    "GAP": WD_GAP,

    # Type scale mapping
    "T_CAPTION": WD_T_CAPTION,
    "T_SMALL": WD_T_SMALL,
    "T_BODY": WD_T_BODY,
    "T_BASE": WD_T_BASE,
    "T_LABEL": WD_T_LABEL,
    "T_TITLE": 17,
    "T_HEAD": 20,
    "T_DISPLAY": WD_DISPLAY,

    # Other
    "SHADOW": WD_SHADOW_MEDIUM,
    "TRACK": (0.0, 0.5, 0.5, 0.2),
    "ACCENT_SOFT": WD_CYAN_SOFT,
    "ACCENT_FAINT": WD_CYAN_FAINT,
}