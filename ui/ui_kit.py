"""Widget drawing helpers shared by the menu pages, the garage and the loading screen.

Everything here is stateless drawing on top of ``Overlay2D``: callers own the
geometry (so hit-testing and drawing can never disagree) and pass a stable
``key`` for each widget, which ``Overlay2D.anim`` uses to fade hover / toggle
state.  All coordinates are logical pixels.
"""
from __future__ import annotations

import math

import numpy as np

from . import icons
from . import theme as T
from .maps_ui import _in, keycaps

inside = _in


def lerp(a, b, t):
    return a + (b - a) * t


def mix(c0, c1, t):
    return tuple(lerp(a, b, t) for a, b in zip(c0, c1))


def alpha(c, a):
    return (c[0], c[1], c[2], a)


def ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


# ----------------------------------------------------------------- backdrops
def hgradient(ui, x, y, w, h, left, right):
    """Horizontal two-colour gradient (one batched call; vertex colours interpolate)."""
    v = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], np.float32)
    c = np.array([left, right, right, left, right, left], np.float32)
    ui.triangles(v, c)


def scrim(ui, w, h, k=1.0, flat=False):
    """Darken the live scene behind a menu.  ``flat`` is an even veil for full-screen pages."""
    r, g, b = T.SCRIM
    if flat:
        ui.rect(0, 0, w, h, (r, g, b, 0.74 * k))
    else:
        hgradient(ui, 0, 0, w, h, (r, g, b, T.SCRIM_LEFT * k), (r, g, b, T.SCRIM_RIGHT * k))
    ui.gradient(0, 0, w, 130, (r, g, b, 0.55 * k), (r, g, b, 0.0))
    ui.gradient(0, h - 150, w, 150, (r, g, b, 0.0), (r, g, b, 0.72 * k))


# ------------------------------------------------------------------ ornaments
def brackets(ui, x, y, w, h, arm=10.0, color=T.SIGNAL, width=1.6, inset=0.0):
    """Four L-shaped corner marks, the focus indicator used across the menus."""
    half = width / 2.0
    for px, py, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        ox, oy = px + sx * inset, py + sy * inset
        ui.line(ox - sx * half, oy, ox + sx * arm, oy, width, color)
        ui.line(ox, oy - sy * half, ox, oy + sy * arm, width, color)


def chip(ui, x, y, text, color=T.TEXT_2, size=T.T_CAPTION, h=22, pad=9, anchor="left",
         fill=T.INSET_FILL, border=T.BORDER_SOFT, icon=None):
    """A small rounded label.  Returns its width."""
    tw = ui.text_w(text, size, weight="semibold")
    iw = (size + 4) if icon else 0
    w = tw + 2 * pad + iw
    if anchor == "right":
        x -= w
    elif anchor == "center":
        x -= w / 2
    ui.box(x, y, w, h, T.R_CHIP, fill, border)
    if icon:
        icons.draw(ui, icon, x + pad + size * 0.5, y + h / 2, size + 2, color, 1.3)
    ui.text_c(text, x + pad + iw, y + h / 2, size, color, weight="semibold")
    return w


def heading(ui, text, x, y, color=T.TEXT_3, size=T.T_CAPTION):
    """Small-caps style section label."""
    ui.text(text.upper(), x, y, size, color, weight="semibold")


def wrap(ui, text, size, max_w, weight="regular", max_lines=None):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and ui.text_w(trial, size, weight=weight) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = ui.fit(lines[-1] + "…", size, max_w, weight=weight)
    return lines


def paragraph(ui, text, x, y, max_w, size=T.T_BODY, color=T.TEXT_2, weight="regular",
              lead=None, max_lines=None):
    """Draw wrapped text.  Returns the height used."""
    lead = lead or size + 6
    lines = wrap(ui, text, size, max_w, weight, max_lines)
    for i, line in enumerate(lines):
        ui.text(line, x, y + i * lead, size, color, weight=weight)
    return len(lines) * lead


# -------------------------------------------------------------------- widgets
def button(ui, rect, label, mouse, *, key, kind="normal", icon=None, focus=False,
           disabled=False, size=T.T_BODY, hover=None):
    """Push button.  ``kind``: normal | accent | ghost | danger.  Returns the hover level."""
    x, y, w, h = rect
    if hover is None:
        hover = ui.anim(("btn", key), float(inside(rect, *mouse) and not disabled), 20.0)
    if kind == "accent":
        ui.container(x, y, w, h, "accent", shadow=False)
        if hover > 0.01:
            ui.box(x, y, w, h, T.R_BUTTON, (1, 1, 1, 0.12 * hover))
        col = T.TEXT
    elif kind == "ghost":
        ui.container(x, y, w, h, "inset", hover=hover)
        col = T.TEXT_2 if hover < 0.5 else T.TEXT
    elif kind == "danger":
        ui.box(x, y, w, h, T.R_BUTTON, ((0.30, 0.10, 0.10, 0.92 + 0.06 * hover), (0.17, 0.06, 0.06, 0.95)),
               alpha(T.LOCKED, 0.55 + 0.4 * hover))
        col = T.LOCKED
    else:
        ui.container(x, y, w, h, "button", hover=hover, shadow=False)
        col = T.TEXT
    if disabled:
        col = T.TEXT_4
        ui.box(x, y, w, h, T.R_BUTTON, (0.0, 0.0, 0.0, 0.35))
    if focus and not disabled:
        ui.box(x - 2, y - 2, w + 4, h + 4, T.R_BUTTON + 2, None, T.SIGNAL, bw=1.5)
    tw = ui.text_w(label, size, weight="semibold")
    isz = size + 3
    total = tw + ((isz + 7) if icon else 0)
    cx = x + (w - total) / 2
    if icon:
        icons.draw(ui, icon, cx + isz / 2, y + h / 2, isz, col, 1.5)
        cx += isz + 7
    ui.text_c(label, cx, y + h / 2, size, col, weight="semibold")
    return hover


TOGGLE_W, TOGGLE_H = 44.0, 24.0


def toggle_rect(rect):
    x, y, w, h = rect
    return (x + w - TOGGLE_W, y + (h - TOGGLE_H) / 2, TOGGLE_W, TOGGLE_H)


def toggle(ui, rect, on, *, key, focus=False, enabled=True):
    """Pill switch right-aligned inside ``rect``."""
    x, y, w, h = toggle_rect(rect)
    k = ui.anim(("tg", key), 1.0 if on else 0.0, 22.0)
    off, onc = (1, 1, 1, 0.16), (T.ACCENT[0], T.ACCENT[1], T.ACCENT[2], 1.0)
    track = mix(off, onc, k)
    if not enabled:
        track = alpha(track, track[3] * 0.45)
    ui.box(x, y, w, h, h / 2, track, (1, 1, 1, 0.22 + 0.2 * k) if enabled else (1, 1, 1, 0.10))
    r = h / 2 - 3
    kx = x + 3 + r + (w - 6 - 2 * r) * k
    ui.circle(kx, y + h / 2, r, (1, 1, 1, 1.0 if enabled else 0.55))
    if focus:
        ui.box(x - 3, y - 3, w + 6, h + 6, h / 2 + 3, None, T.SIGNAL, bw=1.5)


def slider(ui, rect, frac, *, key, hover=0.0, focus=False, enabled=True, color=None):
    """Horizontal slider; ``rect`` is the full-width hit area."""
    x, y, w, h = rect
    frac = max(0.0, min(1.0, float(frac)))
    cy = y + h / 2
    color = color or T.ACCENT
    live = enabled
    ui.box(x, cy - 2, w, 4, 2, T.TRACK)
    fw = w * frac
    if fw > 0.5:
        ui.box(x, cy - 2, max(fw, 4.0), 4, 2, alpha(color, 1.0) if live else (1, 1, 1, 0.28))
    r = 7.0 + 1.5 * hover
    ui.circle(x + fw, cy + 1, r + 2, (0, 0, 0, 0.30))
    ui.circle(x + fw, cy, r, (1, 1, 1, 1.0) if live else (0.62, 0.62, 0.66, 1.0))
    if focus and live:
        ui.ring(x + fw, cy, r + 4, 1.5, T.SIGNAL)


def segmented(ui, rect, labels, index, mouse, *, key, focus=False, enabled=True, size=T.T_SMALL):
    """Right-aligned segmented control.  Returns one rect per option."""
    x, y, w, h = rect
    h = min(h, 30.0)
    y = rect[1] + (rect[3] - h) / 2
    widths = [ui.text_w(s, size, weight="semibold") + 22 for s in labels]
    total = sum(widths) + 6
    if total > w:                                    # shrink to fit
        k = (w - 6) / max(1.0, sum(widths))
        widths = [v * k for v in widths]
        total = sum(widths) + 6
    left = x + w - total
    ui.box(left, y, total, h, T.R_BUTTON, T.INSET_FILL, T.BORDER_SOFT)
    rects, cx = [], left + 3
    for i, (label, sw) in enumerate(zip(labels, widths)):
        r = (cx, y + 3, sw, h - 6)
        rects.append(r)
        sel = i == index
        hov = ui.anim(("seg", key, i), float(inside(r, *mouse) and enabled and not sel), 20.0)
        if sel:
            ui.box(*r, T.R_BUTTON - 3, T.ACCENT_FILL if enabled else ((0.4, 0.4, 0.42, 1), (0.3, 0.3, 0.32, 1)))
        elif hov > 0.01:
            ui.box(*r, T.R_BUTTON - 3, (1, 1, 1, 0.08 * hov))
        col = T.TEXT if sel else (T.TEXT_2 if enabled else T.TEXT_4)
        ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2, size, col, anchor="center", weight="semibold")
        cx += sw
    if focus and enabled:
        ui.box(left - 2, y - 2, total + 4, h + 4, T.R_BUTTON + 2, None, T.SIGNAL, bw=1.5)
    return rects


def stepper(ui, rect, label, mouse, *, key, focus=False, enabled=True, width=210.0, size=T.T_SMALL):
    """``<  value  >`` control for long option lists.  Returns (left, right) arrow rects."""
    x, y, w, h = rect
    width = min(width, w)
    h = min(h, 30.0)
    y = rect[1] + (rect[3] - h) / 2
    left = x + w - width
    ui.box(left, y, width, h, T.R_BUTTON, T.INSET_FILL, T.BORDER_SOFT)
    arrows = []
    for i, name in enumerate(("chevron_l", "chevron_r")):
        r = (left + (0 if i == 0 else width - h), y, h, h)
        arrows.append(r)
        hov = ui.anim(("stp", key, i), float(inside(r, *mouse) and enabled), 20.0)
        if hov > 0.01:
            ui.box(r[0] + 2, r[1] + 2, r[2] - 4, r[3] - 4, T.R_BUTTON - 2, (1, 1, 1, 0.09 * hov))
        icons.draw(ui, name, r[0] + r[2] / 2, r[1] + r[3] / 2, 16, T.TEXT_2 if enabled else T.TEXT_4, 1.6)
    ui.text_c(ui.fit(label, size, width - 2 * h), left + width / 2, y + h / 2, size,
              T.TEXT if enabled else T.TEXT_4, anchor="center", weight="semibold")
    if focus and enabled:
        ui.box(left - 2, y - 2, width + 4, h + 4, T.R_BUTTON + 2, None, T.SIGNAL, bw=1.5)
    return arrows


def scrollbar(ui, x, y, h, view, content, offset):
    if content <= view + 1:
        return
    ui.box(x, y, 4, h, 2, (1, 1, 1, 0.06))
    th = max(28.0, h * view / content)
    ty = y + (h - th) * (offset / max(1.0, content - view))
    ui.box(x, ty, 4, th, 2, (1, 1, 1, 0.34))


def prompt_bar(ui, w, h, items, *, height=46.0):
    """Bottom prompt strip: ``[(key, label), ...]`` centred along the bottom edge."""
    size, gap = T.T_SMALL, 22.0
    widths = [keycaps(ui, 0, 0, k, 11, dry=True) + 8 + ui.text_w(lbl, size, weight="medium")
              for k, lbl in items]
    total = sum(widths) + gap * max(0, len(items) - 1)
    x = (w - total) / 2
    y = h - height + (height - 21) / 2 - 4
    ui.gradient(0, h - height - 18, w, height + 18, (0, 0, 0, 0), (0.0, 0.0, 0.0, 0.50))
    for (k, lbl), iw in zip(items, widths):
        kw = keycaps(ui, x, y, k, 11)
        ui.text_c(lbl, x + kw + 8, y + 10.5, size, T.TEXT_2, weight="medium")
        x += iw + gap


def panel_header(ui, x, y, w, title, subtitle=None, icon=None):
    """Page title block.  Returns the y below it."""
    tx = x
    if icon:
        ui.box(x, y, 44, 44, 11, T.ACCENT_SOFT, alpha(T.ACCENT, 0.55))
        icons.draw(ui, icon, x + 22, y + 22, 24, T.TEXT)
        tx = x + 58
    ui.text(title, tx, y + (0 if subtitle else 9), T.T_HEAD + 4, T.TEXT, weight="bold")
    if subtitle:
        ui.text(ui.fit(subtitle, T.T_BODY, w - (tx - x)), tx, y + 29, T.T_BODY, T.TEXT_3, weight="medium")
    return y + 52


def close_button(ui, x, y, mouse, key="close", size=32):
    rect = (x, y, size, size)
    hover = ui.anim(("closebtn", key), float(inside(rect, *mouse)), 20.0)
    ui.container(*rect, "inset", hover=hover)
    icons.draw(ui, "close", x + size / 2, y + size / 2, 16, T.TEXT if hover > 0.5 else T.TEXT_2, 1.6)
    return rect


def logo_mark(ui, x, y, s=40.0, t=0.0):
    """The app mark: a tilted city block stack on an accent tile."""
    ui.box(x, y, s, s, s * 0.26, T.ACCENT_FILL, (1, 1, 1, 0.9))
    k = s / 40.0
    cx, cy = x + s / 2, y + s / 2
    for dx, dy, bw, bh in ((-9, -2, 8, 12), (-0.5, -9, 9, 19), (9, 1, 7, 9)):
        ui.box(cx + dx * k - bw * k / 2, cy + dy * k - bh * k / 2 + 3 * k, bw * k, bh * k, 1.2 * k, (1, 1, 1, 0.95))
    ui.line(x + 8 * k, y + s - 8 * k, x + s - 8 * k, y + s - 8 * k, 1.6 * k, (1, 1, 1, 0.95))
