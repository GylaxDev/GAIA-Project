"""Watch Dogs themed UI drawing components.

High-level widgets and layout primitives with the signature WD aesthetic:
- Corner brackets on interactive elements
- Scanline overlays
- Glitch/transition effects
- Holographic panel styles
- Technical typography
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from . import theme as T
from .maps_ui import keycaps
from .watchdogs_theme import (
    lerp, lerp_color, ease_out, stagger,
    WD_CYAN, WD_CYAN_DIM, WD_CYAN_BRIGHT, WD_CYAN_SOFT,
    WD_ORANGE, WD_GREEN, WD_PURPLE,
    WD_BG_DEEP, WD_BG_DARK, WD_BG_MEDIUM, WD_BG_LIGHT,
    WD_PANEL_GRADIENT, WD_CARD_GRADIENT, WD_BUTTON_GRADIENT,
    WD_BUTTON_HOVER_GRADIENT, WD_BUTTON_PRESS_GRADIENT,
    WD_ACCENT_GRADIENT, WD_ACCENT_HOVER_GRADIENT,
    WD_BORDER_PRIMARY, WD_BORDER_SECONDARY, WD_BORDER_SUBTLE,
    WD_TEXT_PRIMARY, WD_TEXT_SECONDARY, WD_TEXT_MUTED, WD_TEXT_DISABLED,
    WD_OK, WD_WARN, WD_ERROR,
    WD_R_SMALL, WD_R_MEDIUM, WD_R_LARGE, WD_R_PILL, WD_R_CARD,
    WD_MARGIN, WD_GAP, WD_GAP_SM,
    WD_T_MICRO, WD_T_CAPTION, WD_T_SMALL, WD_T_BODY, WD_T_BASE,
    WD_T_LABEL, WD_DISPLAY,
    WD_ANIM_FAST, WD_ANIM_NORMAL, WD_ANIM_SLOW,
    WD_SHADOW_SOFT, WD_SHADOW_MEDIUM, WD_SHADOW_STRONG,
)
from .watchdogs_icons import draw_wd


# ========================================================================
# LOW-LEVEL PRIMITIVES
# ========================================================================

def wd_brackets(ui, x, y, w, h, arm=12.0, color=WD_CYAN, width=1.8, inset=0.0, animated=False, phase=0.0):
    """Watch Dogs signature corner brackets with optional animation.

    Args:
        animated: If True, brackets animate in/out based on phase
        phase: Animation phase 0.0-1.0
    """
    if animated:
        # Animate arm length and opacity
        arm = lerp(0, arm, phase)
        color = lerp_color((color[0], color[1], color[2], 0.0), color, phase)

    half = width / 2.0
    for px, py, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        ox, oy = px + sx * inset, py + sy * inset
        ui.line(ox - sx * half, oy, ox + sx * arm, oy, width, color)
        ui.line(ox, oy - sy * half, ox, oy + sy * arm, width, color)


def wd_scanlines(ui, x, y, w, h, color=None, density=3, alpha=0.03):
    """Subtle horizontal scanlines overlay."""
    if color is None:
        color = (0.0, 0.0, 0.0, alpha)
    for scan_y in range(int(y), int(y + h), density):
        ui.line(x, scan_y, x + w, scan_y, 1.0 / ui.scale, color)


def wd_glitch_line(ui, x0, y0, x1, y1, color, glitch_amount=2.0):
    """Draw a line with subtle glitch displacement."""
    ui.line(x0, y0, x1, y1, 1.0 / ui.scale, color)


def wd_gradient_panel(ui, x, y, w, h, gradient=None, border=True, border_color=None,
                       radius=WD_R_CARD, shadow=True):
    """Draw a Watch Dogs style panel with gradient fill and optional border/shadow."""
    if gradient is None:
        gradient = WD_PANEL_GRADIENT
    top, bottom = gradient

    # Shadow
    if shadow:
        ui.box(x - 2, y + 4, w + 4, h + 4, radius + 2,
               (0.0, 0.0, 0.0, 0.35), None)

    # Panel background
    ui.box(x, y, w, h, radius, top, bottom)

    # Border
    if border:
        bc = border_color or WD_BORDER_SECONDARY
        ui.box(x, y, w, h, radius, None, bc, bw=1.5 / ui.scale)


def wd_gradient_button(ui, x, y, w, h, label, color=None, hover=0.0, pressed=False,
                        disabled=False, radius=WD_R_MEDIUM, icon=None):
    """Watch Dogs style button with gradient states."""
    if color is None:
        color = WD_CYAN

    # Determine colors based on state
    if disabled:
        bg_top = lerp_color(WD_BUTTON_GRADIENT[0], (0.1, 0.1, 0.12, 0.5), 0.7)
        bg_bottom = lerp_color(WD_BUTTON_GRADIENT[1], (0.05, 0.05, 0.06, 0.5), 0.7)
        text_color = WD_TEXT_DISABLED
        border_color = lerp_color(WD_BORDER_SUBTLE, (0.2, 0.2, 0.2, 0.5), 0.5)
    elif pressed:
        bg_top = WD_BUTTON_PRESS_GRADIENT[0]
        bg_bottom = WD_BUTTON_PRESS_GRADIENT[1]
        text_color = WD_TEXT_PRIMARY
        border_color = lerp_color(WD_BORDER_PRIMARY, WD_CYAN_BRIGHT, hover)
    elif hover > 0.0:
        bg_top = lerp_color(WD_BUTTON_GRADIENT[0], WD_BUTTON_HOVER_GRADIENT[0], hover)
        bg_bottom = lerp_color(WD_BUTTON_GRADIENT[1], WD_BUTTON_HOVER_GRADIENT[1], hover)
        text_color = lerp_color(WD_TEXT_PRIMARY, WD_CYAN_BRIGHT, hover)
        border_color = lerp_color(WD_BORDER_SECONDARY, WD_BORDER_PRIMARY, hover)
    else:
        bg_top, bg_bottom = WD_BUTTON_GRADIENT
        text_color = WD_TEXT_PRIMARY
        border_color = WD_BORDER_SECONDARY

    # Draw button
    ui.box(x, y, w, h, radius, bg_top, bg_bottom)
    ui.box(x, y, w, h, radius, None, border_color, bw=1.5 / ui.scale)

    # Content
    isz = 16
    total_w = ui.text_w(label, WD_T_BODY, weight="semibold")
    if icon:
        total_w += isz + 8

    cx = x + (w - total_w) / 2
    if icon:
        draw_wd(ui, icon, cx + isz / 2, y + h / 2, isz, text_color, 1.6)
        cx += isz + 8

    ui.text_c(label, cx, y + h / 2, WD_T_BODY, text_color, weight="semibold")

    # Corner brackets on hover/focus
    if hover > 0.1 and not disabled:
        wd_brackets(ui, x, y, w, h, arm=8.0 * hover, color=lerp_color(WD_CYAN_SOFT, WD_CYAN, hover),
                    animated=True, phase=hover)

    return hover


def wd_toggle(ui, x, y, w, h, on: bool, hover=0.0, enabled=True):
    """Watch Dogs style toggle switch."""
    track_w, track_h = w, h
    thumb_r = track_h / 2 - 2

    # Track colors
    if not enabled:
        off_color = lerp_color(WD_BG_MEDIUM, (0.15, 0.15, 0.15, 0.3), 0.5)
        on_color = lerp_color(WD_CYAN_DIM, (0.2, 0.2, 0.2, 0.3), 0.5)
    else:
        off_color = WD_BG_MEDIUM
        on_color = lerp_color(WD_CYAN_DIM, WD_CYAN, 0.3)

    # Animate
    k = lerp(0, 1, 1.0 if on else 0.0)
    track_color = lerp_color(off_color, on_color, k)

    # Draw track
    ui.box(x, y, track_w, track_h, track_h / 2, track_color,
           lerp_color(WD_BORDER_SUBTLE, WD_CYAN, 0.2 + 0.3 * hover) if enabled else WD_BORDER_SUBTLE,
           bw=1.5 / ui.scale)

    # Thumb
    thumb_x = x + 2 + thumb_r + (track_w - 4 - 2 * thumb_r) * k
    thumb_y = y + track_h / 2
    ui.circle(thumb_x, thumb_y, thumb_r,
              lerp_color((0.9, 0.9, 0.95, 1.0), WD_CYAN_BRIGHT, hover) if enabled else (0.6, 0.6, 0.65, 1.0))

    # Corner brackets when focused
    if hover > 0.3:
        wd_brackets(ui, x - 2, y - 2, track_w + 4, track_h + 4,
                    arm=6.0, color=lerp_color(WD_CYAN_SOFT, WD_CYAN, hover))


def wd_slider(ui, x, y, w, h, frac: float, hover=0.0, color=None):
    """Watch Dogs style horizontal slider."""
    if color is None:
        color = WD_CYAN

    frac = max(0.0, min(1.0, float(frac)))
    cy = y + h / 2

    # Track
    ui.box(x, cy - 2, w, 4, 2,
           lerp_color(WD_BG_MEDIUM, color, 0.1),
           lerp_color(WD_BORDER_SUBTLE, color, 0.3))

    # Filled portion
    fw = w * frac
    if fw > 4:
        ui.box(x, cy - 2, max(fw, 4), 4, 2,
               lerp_color(color, WD_CYAN_BRIGHT, 0.2),
               lerp_color(color, WD_CYAN_BRIGHT, 0.2))

    # Thumb
    r = 8.0 + 2.0 * hover
    thumb_x = x + fw
    ui.circle(thumb_x, cy, r + 1.5 / ui.scale, (0.0, 0.0, 0.0, 0.3))
    ui.circle(thumb_x, cy, r, lerp_color((1.0, 1.0, 1.0, 1.0), WD_CYAN_BRIGHT, hover))

    # Corner brackets on focus
    if hover > 0.5:
        wd_brackets(ui, x - 4, cy - r - 4, w + 8, 2 * r + 8,
                    arm=8.0, color=lerp_color(WD_CYAN_SOFT, WD_CYAN, hover))


def wd_segmented(ui, x, y, w, h, labels: list, index: int, hover_idx: int = -1,
                enabled=True):
    """Watch Dogs style segmented control."""
    h = min(h, 28.0)
    widths = [ui.text_w(s, WD_T_SMALL, weight="semibold") + 20 for s in labels]
    total = sum(widths) + 4

    if total > w:
        scale = (w - 4) / max(1.0, sum(widths))
        widths = [v * scale for v in widths]
        total = sum(widths) + 4

    left = x + w - total
    ui.box(left, y, total, h, WD_R_PILL,
           lerp_color(WD_BG_DARK, WD_BG_MEDIUM, 0.5),
           lerp_color(WD_BORDER_SUBTLE, WD_BORDER_PRIMARY, 0.3),
           bw=1.5 / ui.scale)

    cx = left + 2
    rects = []
    for i, (label, sw) in enumerate(zip(labels, widths)):
        r = (cx, y + 2, sw, h - 4)
        rects.append(r)

        sel = i == index
        hov = i == hover_idx and enabled and not sel

        if sel:
            ui.box(*r, WD_R_PILL - 2,
                   lerp_color(WD_CYAN_DIM, WD_CYAN, 0.5),
                   lerp_color(WD_CYAN, WD_CYAN_BRIGHT, 0.3))
            text_col = WD_TEXT_PRIMARY
        elif hov:
            ui.box(*r, WD_R_PILL - 2,
                   lerp_color(WD_BG_MEDIUM, WD_CYAN_DIM, 0.3),
                   lerp_color(WD_BORDER_SUBTLE, WD_CYAN_SOFT, 0.4))
            text_col = lerp_color(WD_TEXT_SECONDARY, WD_CYAN, 0.5)
        else:
            text_col = WD_TEXT_SECONDARY if enabled else WD_TEXT_DISABLED

        ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2,
                  WD_T_SMALL, text_col, anchor="center", weight="semibold")

        # Corner brackets on selection
        if sel:
            wd_brackets(ui, r[0] - 1, r[1] - 1, r[2] + 2, r[3] + 2,
                        arm=6.0, color=WD_CYAN, animated=True, phase=1.0)

        cx += sw

    return rects


def wd_card(ui, x, y, w, h, content_fn: Callable, header=None, footer=None,
            hover=0.0, interactive=True):
    """Watch Dogs style card container."""
    # Shadow
    ui.box(x - 1, y + 3, w + 2, h + 3, WD_R_CARD + 1,
           (0.0, 0.0, 0.0, 0.4), None)

    # Background
    ui.box(x, y, w, h, WD_R_CARD, WD_CARD_GRADIENT[0], WD_CARD_GRADIENT[1])

    # Border
    border_col = lerp_color(WD_BORDER_SUBTLE, WD_CYAN_SOFT, hover * 0.8)
    if interactive and hover > 0.0:
        border_col = lerp_color(border_col, WD_BORDER_PRIMARY, hover)
    ui.box(x, y, w, h, WD_R_CARD, None, border_col, bw=1.5 / ui.scale)

    # Corner brackets on hover/focus
    if interactive and hover > 0.1:
        wd_brackets(ui, x, y, w, h, arm=10.0,
                    color=lerp_color(WD_CYAN_SOFT, WD_CYAN, hover),
                    animated=True, phase=hover)

    # Content area
    content_y = y + WD_GAP
    content_h = h - 2 * WD_GAP

    if header:
        content_y = header(ui, x + WD_GAP, y + WD_GAP, w - 2 * WD_GAP)
        content_h = y + h - WD_GAP - content_y

    content_fn(ui, x + WD_GAP, content_y, w - 2 * WD_GAP, content_h)

    if footer:
        footer(ui, x + WD_GAP, y + h - WD_GAP - 20, w - 2 * WD_GAP, 20)


def wd_heading(ui, text: str, x: float, y: float, color=WD_TEXT_MUTED, size=WD_T_CAPTION):
    """Watch Dogs style section heading - uppercase with tracking."""
    # Add subtle top line
    ui.line(x - 2, y + size / 2, x + 8, y + size / 2, 1.0 / ui.scale,
            lerp_color(color, WD_CYAN_SOFT, 0.5))
    ui.text(text.upper(), x, y, size, color, weight="semibold")


def wd_pill_chip(ui, x, y, text: str, color=WD_CYAN, size=WD_T_CAPTION,
                  h=20, pad=10, icon=None):
    """Small rounded status chip/pill."""
    tw = ui.text_w(text, size, weight="semibold")
    iw = (size + 4) if icon else 0
    w = tw + 2 * pad + iw

    # Background with subtle glow
    bg_col = lerp_color((color[0], color[1], color[2], 0.08),
                        (color[0], color[1], color[2], 0.18), 1.0)
    border_col = lerp_color((color[0], color[1], color[2], 0.2),
                            color, 0.5)

    ui.box(x, y, w, h, WD_R_PILL, bg_col, border_col)

    if icon:
        draw_wd(ui, icon, x + pad + size * 0.5, y + h / 2, size + 2, color, 1.3)

    ui.text_c(text, x + pad + iw, y + h / 2, size,
              lerp_color(WD_TEXT_SECONDARY, color, 0.8), weight="semibold")
    return w


def wd_scrim(ui, w, h, intensity=1.0):
    """Watch Dogs style dark overlay behind menus."""
    # Base scrim
    r, g, b = 0.0, 0.005, 0.01
    ui.rect(0, 0, w, h, (r, g, b, 0.7 * intensity))

    # Top gradient
    ui.gradient(0, 0, w, 120,
                (r, g, b, 0.5 * intensity),
                (r, g, b, 0.0))

    # Bottom gradient
    ui.gradient(0, h - 140, w, 140,
                (r, g, b, 0.0),
                (r, g, b, 0.65 * intensity))

    # Subtle scanlines
    wd_scanlines(ui, 0, 0, w, h, alpha=0.015 * intensity)


def wd_prompt_bar(ui, w, h, items: list, height=44.0):
    """Bottom control prompt bar - Watch Dogs style."""
    size, gap = WD_T_CAPTION, 18.0
    widths = [keycaps(ui, 0, 0, k, 10, True) + 6 + ui.text_w(lbl, size, weight="medium")
              for k, lbl in items]
    total = sum(widths) + gap * max(0, len(items) - 1)

    x = (w - total) / 2
    y = h - height + (height - 18) / 2 - 2

    # Background glow
    ui.gradient(0, h - height - 14, w, height + 14,
                (0.0, 0.005, 0.01, 0.0),
                (0.0, 0.005, 0.01, 0.4))

    for (k, lbl), iw in zip(items, widths):
        kw = keycaps(ui, x, y, k, 10)
        draw_wd(ui, "chevron_r" if ">" in k else "keyboard",
                x + kw / 2, y + 9, 10, WD_CYAN_SOFT)
        ui.text_c(lbl, x + kw + 6, y + 9.5, size, WD_TEXT_SECONDARY, weight="medium")
        x += iw + gap


def wd_progress_bar(ui, x, y, w, h, frac: float, color=None, animated=False, phase=0.0):
    """Watch Dogs progress bar with optional animation."""
    if color is None:
        color = WD_CYAN

    frac = max(0.0, min(1.0, float(frac)))
    ui.box(x, y, w, h, h / 2,
           lerp_color(WD_BG_DARK, color, 0.1),
           lerp_color(WD_BORDER_SUBTLE, color, 0.2))

    if animated:
        # Animated shimmer sweep
        phase = phase % 1.0
        sweep_w = min(80.0, w * 0.5)
        sweep_x = x + phase * (w + sweep_w) - sweep_w

        # Base fill
        if frac > 0:
            ui.box(x, y, max(w * frac, h), h, h / 2,
                   lerp_color(color, WD_CYAN_BRIGHT, 0.3))

        # Sweep
        if sweep_x + sweep_w > x and sweep_x < x + w:
            sx = max(sweep_x, x)
            ex = min(sweep_x + sweep_w, x + w)
            if ex > sx:
                ui.rect(sx, y, ex - sx, h,
                        lerp_color(WD_CYAN_BRIGHT, (1.0, 1.0, 1.0, 1.0), 0.3))
    else:
        if frac > 0:
            ui.box(x, y, max(w * frac, h), h, h / 2,
                   lerp_color(color, WD_CYAN_BRIGHT, 0.3))


def wd_toast(ui, w, h, text: str, color=WD_CYAN, duration=3.0):
    """Watch Dogs style toast notification."""
    # This would be drawn at bottom-center with slide-up animation
    pass  # Implementation would go in overlay drawing loop


# ========================================================================
# KEYCAP HELPER - uses maps_ui.keycaps which calls Overlay2D.kbd
# ========================================================================

# The keycaps function from maps_ui is used directly - it calls ui.kbd internally
# which draws the keycap using Overlay2D's built-in method.
# We don't need to redefine it here.