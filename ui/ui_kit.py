"""Widget drawing helpers shared by every screen - the Apple-style building blocks.

Everything here is stateless drawing on top of ``Overlay2D``: callers own the geometry (so
hit-testing and drawing can never disagree) and pass a stable ``key`` for each widget, which
``Overlay2D.anim`` / ``anim_spring`` use to ease hover and switch state.  All coordinates are
logical pixels.

Vocabulary (see ``theme`` for the values):

* ``surface`` / ``glass``      translucent rounded panels with a hairline rim and soft shadow
* ``group``                    inset grouped card; rows inside are separated by hairlines
* ``button``                   capsule: primary (blue), secondary (grey glass), tertiary, destructive
* ``toggle`` / ``slider``      the familiar iOS switch (green) and slider (blue, white knob)
* ``segmented`` / ``stepper``  capsule controls with a sliding selection
* ``icon_tile``                colour "app icon" tile used to give list rows an identity
* ``keycaps`` / ``prompt_bar`` keyboard hints in a floating capsule
"""
from __future__ import annotations

import math

import numpy as np

from . import icons
from . import theme as T

# Pointer state, set once per frame by the menu system, so buttons can show a pressed state
# without every call site having to pass it around.
POINTER = {"down": False}

TOGGLE_W, TOGGLE_H = 50.0, 30.0
WHITE = (1.0, 1.0, 1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Maths & animation helpers
# ═══════════════════════════════════════════════════════════════════════

def lerp(a, b, t):
    return a + (b - a) * t


def mix(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return tuple(lerp(a, b, t) for a, b in zip(c0, c1))


def mix_pair(p0, p1, t):
    """Blend two (top, bottom) gradients."""
    return (mix(p0[0], p1[0], t), mix(p0[1], p1[1], t))


def alpha(c, a):
    return (c[0], c[1], c[2], a)


def scale_alpha(c, k):
    return (c[0], c[1], c[2], max(0.0, min(1.0, c[3] * k)))


def inside(rect, x, y):
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]


_in = inside


def ease_out_expo(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - 2.0 ** (-10.0 * t) if t < 1.0 else 1.0


def ease_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def ease_out(t):
    return ease_out_cubic(t)


def ease_in_out_cubic(t):
    t = max(0.0, min(1.0, t))
    return 4 * t ** 3 if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def spring(ui, key, target, preset=T.SPRING_GENTLE):
    """Spring-eased value; falls back to the plain eased ``anim`` on older overlays."""
    fn = getattr(ui, "anim_spring", None)
    if fn is not None:
        return fn(key, target, *preset)
    return ui.anim(key, target, 18.0)


# ═══════════════════════════════════════════════════════════════════════
# Surfaces
# ═══════════════════════════════════════════════════════════════════════

def hgradient(ui, x, y, w, h, left, right):
    """Horizontal two-colour gradient (one batched call; vertex colours interpolate)."""
    v = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], np.float32)
    c = np.array([left, right, right, left, right, left], np.float32)
    ui.triangles(v, c)


def vgradient(ui, x, y, w, h, top, bottom):
    """Vertical two-colour gradient (square corners; use ``surface`` for rounded fills)."""
    v = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], np.float32)
    c = np.array([top, top, bottom, top, bottom, bottom], np.float32)
    ui.triangles(v, c)


def drop_shadow(ui, x, y, w, h, r, shadow):
    """Soft drop shadow: ``shadow`` is ``(offset_y, blur, spread, colour)``."""
    dy, blur, _spread, col = shadow
    ui.box(x, y, w, h, r, None, None, shadow=blur, shadow_dy=dy, shadow_alpha=col[3])


def surface(ui, x, y, w, h, r, fill=T.GLASS, border=None, shadow=None, specular=False):
    """A rounded, translucent panel.  ``fill`` is a (top, bottom) gradient or one RGBA colour."""
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    if shadow:
        drop_shadow(ui, x, y, w, h, r, shadow)
    ui.box(x, y, w, h, r, fill, border, bw=1.0)
    if specular and w > 3 * r:
        ui.line(x + r, y + 0.5, x + w - r, y + 0.5, 1.0, T.SPECULAR)


GLASS_LEVELS = {
    "hud": (T.GLASS, T.BORDER_DEFAULT, T.SHADOW_2),
    "menu": (T.GLASS_STRONG, T.BORDER_DEFAULT, T.SHADOW_3),
    "sheet": (T.SHEET, T.BORDER_SUBTLE, T.SHADOW_4),
    "raised": (T.SHEET_RAISED, T.BORDER_DEFAULT, T.SHADOW_3),
}


def glass(ui, x, y, w, h, r=T.R_CARD, level="hud", shadow=True):
    """The standard glass panel.  ``level``: hud (over the scene), menu, sheet, raised."""
    fill, border, shd = GLASS_LEVELS.get(level, GLASS_LEVELS["hud"])
    surface(ui, x, y, w, h, r, fill, border, shd if shadow else None, specular=True)


def sheet(ui, x, y, w, h):
    """A full page / dialog sheet."""
    glass(ui, x, y, w, h, T.R_SHEET, "sheet")


def group(ui, x, y, w, h, r=T.R_3 + 2, hover=0.0, selected=False, fill=None):
    """Inset grouped card on a sheet.  Rows are laid out by the caller; use ``separator``."""
    if fill is None:
        fill = T.GROUP_SELECTED if selected else mix_pair(T.GROUP, T.GROUP_HOVER, hover)
    surface(ui, x, y, w, h, r, fill)


def separator(ui, x, y, w, color=None):
    """Hairline between rows of a group."""
    ui.box(x, y, w, 1, 0.5, color or T.SEPARATOR)


def divider(ui, x, y, w, color=None):
    ui.box(x, y, w, 1, 0.5, color or T.HAIRLINE)


def vdivider(ui, x, y, h, color=None):
    ui.box(x, y, 1, h, 0.5, color or T.HAIRLINE)


def scrim(ui, w, h, k=1.0, flat=False):
    """Dim the live scene behind a menu.  ``flat`` is an even veil for full-screen pages."""
    r, g, b = T.SCRIM_BASE
    if flat:
        ui.rect(0, 0, w, h, (r, g, b, 0.52 * k))
    else:
        hgradient(ui, 0, 0, w, h, (r, g, b, T.SCRIM_TOP_OPACITY * k), (r, g, b, T.SCRIM_BOTTOM_OPACITY * k))
    ui.gradient(0, 0, w, T.SCRIM_EDGE_FADE, (r, g, b, 0.30 * k), (r, g, b, 0.0))
    ui.gradient(0, h - 150, w, 150, (r, g, b, 0.0), (r, g, b, 0.45 * k))


def material_box(ui, x, y, w, h, r, variant="regular", border=None):
    """A light translucent fill (chips, hover states, inset wells)."""
    fills = {"thin": T.MATERIAL_THIN, "regular": T.MATERIAL_REGULAR, "thick": T.MATERIAL_THICK,
             "tinted": T.MATERIAL_TINTED, "field": T.FIELD}
    r = max(0.0, min(r, w / 2.0, h / 2.0))
    ui.box(x, y, w, h, r, fills.get(variant, T.MATERIAL_REGULAR), border, bw=1.0)


def focus_ring(ui, x, y, w, h, r, color=T.FOCUS_RING, width=2.0, inset=0.0):
    """Keyboard focus: a crisp accent outline just outside the control."""
    ui.box(x - inset, y - inset, w + 2 * inset, h + 2 * inset, r + inset, None, color, bw=width)


# ═══════════════════════════════════════════════════════════════════════
# Text helpers
# ═══════════════════════════════════════════════════════════════════════

def heading(ui, text, x, y, color=T.TEXT_SECONDARY, size=T.T_FOOTNOTE):
    """Section title above a group: sentence case, semibold, quiet."""
    ui.text(text, x, y, size, color, weight="semibold")


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


def paragraph(ui, text, x, y, max_w, size=T.T_BODY, color=T.TEXT_SECONDARY, weight="regular",
              lead=None, max_lines=None, anchor="left"):
    """Draw wrapped text.  Returns the height used."""
    lead = lead or size + 6
    lines = wrap(ui, text, size, max_w, weight, max_lines)
    for i, line in enumerate(lines):
        ui.text(line, x, y + i * lead, size, color, anchor=anchor, weight=weight)
    return len(lines) * lead


# ═══════════════════════════════════════════════════════════════════════
# Keyboard hints
# ═══════════════════════════════════════════════════════════════════════

def _key_tokens(key):
    """'W A S D' -> W A S D;  'WASD / arrows' -> WASD, arrows;  'Shift' -> Shift."""
    out = []
    for part in key.split(" / "):
        part = part.strip()
        bits = part.split()
        if len(bits) > 1 and all(len(b) == 1 for b in bits):
            out.extend(bits)
        else:
            out.append(part)
    return out


KEY_FACE = ((0.36, 0.36, 0.38, 1.0), (0.29, 0.29, 0.31, 1.0))


def keycaps(ui, x, y, key, size=11, dry=False):
    """Draw ``key`` as one or more keycaps starting at x, y.  Returns the width."""
    cx = x
    kh = size + 10
    for i, tok in enumerate(_key_tokens(key)):
        if i:
            cx += 3
        tw = ui.text_w(tok, size, weight="semibold")
        kw = max(size + 10, tw + 12)
        if not dry:
            ui.box(cx, y + 1.5, kw, kh, 6, (0.0, 0.0, 0.0, 0.38))           # the key's lower edge
            ui.box(cx, y, kw, kh, 6, KEY_FACE)
            ui.text_c(tok, cx + kw / 2, y + kh / 2, size, T.TEXT_PRIMARY, anchor="center", weight="semibold")
        cx += kw
    return cx - x


def prompt_bar(ui, w, h, items, *, height=40.0):
    """Floating capsule of ``[(key, label), ...]`` hints, centred above the bottom edge."""
    size, gap, pad = T.T_CAPTION_2, 22.0, 18.0
    items = list(items)

    def measure(its):
        ws = [keycaps(ui, 0, 0, k, 11, dry=True) + 7 + ui.text_w(lbl, size, weight="medium") for k, lbl in its]
        return ws, sum(ws) + gap * max(0, len(its) - 1) + 2 * pad

    widths, total = measure(items)
    while len(items) > 1 and total > w - 24:           # drop from the end, keep the last (Esc / Back)
        items.pop(len(items) - 2 if len(items) > 2 else 0)
        widths, total = measure(items)
    x, y = (w - total) / 2, h - height - 22
    glass(ui, x, y, total, height, height / 2, "menu")
    cx = x + pad
    for (k, lbl), iw in zip(items, widths):
        kw = keycaps(ui, cx, y + (height - 21) / 2, k, 11)
        ui.text_c(lbl, cx + kw + 7, y + height / 2, size, T.TEXT_SECONDARY, weight="medium")
        cx += iw + gap


# ═══════════════════════════════════════════════════════════════════════
# Buttons, chips, tiles
# ═══════════════════════════════════════════════════════════════════════

BTN_SECONDARY = ((0.47, 0.47, 0.50, 0.36), (0.47, 0.47, 0.50, 0.30))
BTN_SECONDARY_HOVER = ((0.55, 0.55, 0.58, 0.46), (0.55, 0.55, 0.58, 0.40))


def button(ui, rect, label, mouse, *, key, kind="secondary", icon=None, focus=False,
           disabled=False, size=T.T_BODY, hover=None):
    """Capsule push button.  ``kind``: primary, secondary, tertiary, destructive.  Returns hover (0-1)."""
    x, y, w, h = rect
    over = inside(rect, *mouse) and not disabled
    if hover is None:
        hover = spring(ui, ("btn", key), float(over), T.SPRING_GENTLE)
    press = spring(ui, ("btn_press", key), 1.0 if (over and POINTER["down"]) else 0.0, T.SPRING_QUICK)
    hover, press = max(0.0, min(1.0, hover)), max(0.0, min(1.0, press))
    r = h / 2.0

    if kind in ("primary", "destructive"):
        base = T.ACCENT_BLUE_FILL if kind == "primary" else T.DESTRUCTIVE_FILL
        fill = (mix(base[0], WHITE, 0.14 * hover), mix(base[1], WHITE, 0.14 * hover))
        fill = (mix(fill[0], (0, 0, 0, 1), 0.22 * press), mix(fill[1], (0, 0, 0, 1), 0.22 * press))
        col = T.TEXT_INVERTED
        ring = (1.0, 1.0, 1.0, 0.85)
    elif kind == "tertiary":
        fill = ((1.0, 1.0, 1.0, 0.085 * hover + 0.07 * press),) * 2
        col = mix(T.TEXT_SECONDARY, T.TEXT_PRIMARY, hover)
        ring = T.FOCUS_RING
    else:
        fill = mix_pair(BTN_SECONDARY, BTN_SECONDARY_HOVER, hover)
        fill = (mix(fill[0], (0, 0, 0, fill[0][3]), 0.25 * press), mix(fill[1], (0, 0, 0, fill[1][3]), 0.25 * press))
        col = T.TEXT_PRIMARY
        ring = T.FOCUS_RING

    if disabled:
        fill = (scale_alpha(fill[0], 0.45), scale_alpha(fill[1], 0.45))
        col = T.TEXT_QUATERNARY

    if kind == "primary" and not disabled:
        ui.box(x, y, w, h, r, None, None, shadow=10.0, shadow_dy=4.0, shadow_alpha=0.22)
    if kind != "tertiary" or hover > 0.01 or press > 0.01:
        ui.box(x, y, w, h, r, fill, None)
        if kind in ("primary", "destructive") and not disabled and w > 2 * r:
            ui.line(x + r, y + 0.5, x + w - r, y + 0.5, 1.0, (1.0, 1.0, 1.0, 0.30 * (1.0 - press)))

    if focus and not disabled:
        focus_ring(ui, x, y, w, h, r, ring, width=2.0, inset=2.5)

    tw = ui.text_w(label, size, weight="semibold")
    isz = size + 3
    total = tw + ((isz + 7) if icon else 0)
    cx = x + (w - total) / 2
    if icon:
        icons.draw(ui, icon, cx + isz / 2, y + h / 2, isz, col, 1.6)
        cx += isz + 7
    ui.text_c(label, cx, y + h / 2, size, col, weight="semibold")
    return hover


def chip(ui, x, y, text, color=T.TEXT_SECONDARY, size=T.T_CAPTION_2, h=22, pad=9, anchor="left",
         fill=None, border=None, icon=None):
    """A small capsule label.  Returns its width.  Pass a colour for a tinted status chip."""
    tw = ui.text_w(text, size, weight="semibold")
    iw = (size + 4) if icon else 0
    w = tw + 2 * pad + iw
    if anchor == "right":
        x -= w
    elif anchor == "center":
        x -= w / 2
    if fill is None:
        tinted = color not in (T.TEXT_SECONDARY, T.TEXT_TERTIARY, T.TEXT_QUATERNARY, T.TEXT_PRIMARY)
        fill = (color[0], color[1], color[2], 0.16) if tinted else (1.0, 1.0, 1.0, 0.10)
    ui.box(x, y, w, h, h / 2, fill, border)
    if icon:
        icons.draw(ui, icon, x + pad + size * 0.5, y + h / 2, size + 2, color, 1.3)
    ui.text_c(text, x + pad + iw, y + h / 2, size, color, weight="semibold")
    return w


def icon_tile(ui, x, y, s, icon, tile=T.TILE_BLUE, enabled=True):
    """A coloured rounded-square tile with a white glyph - the identity of a list row."""
    k = 1.0 if enabled else 0.4
    fill = (scale_alpha(tile[0], k), scale_alpha(tile[1], k))
    ui.box(x, y, s, s, s * T.R_TILE, fill, None)
    ui.line(x + s * 0.22, y + 0.5, x + s - s * 0.22, y + 0.5, 1.0, (1.0, 1.0, 1.0, 0.32 * k))
    icons.draw(ui, icon, x + s / 2, y + s / 2, s * 0.58, (1.0, 1.0, 1.0, k), max(1.4, s / 18.0))


def close_button(ui, x, y, mouse, key="close", size=32):
    """Round grey close control (the iOS sheet dismiss button)."""
    rect = (x, y, size, size)
    hv = spring(ui, ("closebtn", key), float(inside(rect, *mouse)), T.SPRING_GENTLE)
    ui.circle(x + size / 2, y + size / 2, size / 2, mix(T.FILL_3, T.FILL_1, hv))
    icons.draw(ui, "close", x + size / 2, y + size / 2, size * 0.42, mix(T.TEXT_SECONDARY, T.TEXT_PRIMARY, hv), 1.9)
    return rect


def logo_mark(ui, x, y, s=40.0, t=0.0):
    """The app mark: a city skyline on a blue app-icon tile."""
    drop_shadow(ui, x, y, s, s, s * 0.225, (s * 0.08, s * 0.28, 0, (0, 0, 0, 0.30)))
    ui.box(x, y, s, s, s * 0.225, T.TILE_BLUE, (1.0, 1.0, 1.0, 0.18), bw=1.0)
    ui.line(x + s * 0.2, y + 0.5, x + s * 0.8, y + 0.5, 1.0, (1.0, 1.0, 1.0, 0.45))
    k = s / 40.0
    cx, cy = x + s / 2, y + s / 2
    for dx, dy, bw, bh in ((-9, -2, 8, 12), (-0.5, -9, 9, 19), (9, 1, 7, 9)):
        ui.box(cx + dx * k - bw * k / 2, cy + dy * k - bh * k / 2 + 3 * k, bw * k, bh * k, 1.6 * k, (1, 1, 1, 0.96))
    ui.line(x + 8 * k, y + s - 8 * k, x + s - 8 * k, y + s - 8 * k, 1.8 * k, (1, 1, 1, 0.96))


def panel_header(ui, x, y, w, title, subtitle=None, icon=None, tile=T.TILE_BLUE):
    """Page title block (large title + quiet subtitle).  Returns the y below it."""
    tx = x
    if icon:
        icon_tile(ui, x, y + 2, 44, icon, tile)
        tx = x + 58
    ui.text(title, tx, y + (4 if subtitle else 9), T.T_TITLE_3, T.TEXT_PRIMARY, weight="bold")
    if subtitle:
        ui.text(ui.fit(subtitle, T.T_BODY, w - (tx - x)), tx, y + 33, T.T_BODY, T.TEXT_TERTIARY, weight="regular")
    return y + 56


# ═══════════════════════════════════════════════════════════════════════
# Controls
# ═══════════════════════════════════════════════════════════════════════

def toggle_rect(rect):
    x, y, w, h = rect
    return (x + w - TOGGLE_W, y + (h - TOGGLE_H) / 2, TOGGLE_W, TOGGLE_H)


def toggle(ui, rect, on, *, key, focus=False, enabled=True):
    """iOS switch, right-aligned inside ``rect``: green when on, spring-animated knob."""
    x, y, w, h = toggle_rect(rect)
    k = max(0.0, min(1.0, spring(ui, ("tg", key), 1.0 if on else 0.0, T.SPRING_QUICK)))
    track = mix(T.FILL_1, T.SYS_GREEN, k)
    if not enabled:
        track = scale_alpha(track, 0.5)
    ui.box(x, y, w, h, h / 2, track)
    r = h / 2 - 2.0
    kx = x + 2 + r + (w - 4 - 2 * r) * k
    ui.box(kx - r, y + 2, 2 * r, 2 * r, r, None, None, shadow=5.0, shadow_dy=2.0, shadow_alpha=0.28)
    ui.circle(kx, y + h / 2, r, (1.0, 1.0, 1.0, 1.0 if enabled else 0.62))
    if focus and enabled:
        focus_ring(ui, x, y, w, h, h / 2, T.FOCUS_RING, width=2.0, inset=2.5)


def slider(ui, rect, frac, *, key, hover=0.0, focus=False, enabled=True, color=None):
    """Horizontal slider: slim track, blue fill, white knob with a soft shadow."""
    x, y, w, h = rect
    frac = max(0.0, min(1.0, float(frac)))
    cy = y + h / 2
    th = 5.0
    live = enabled
    ui.box(x, cy - th / 2, w, th, th / 2, (1.0, 1.0, 1.0, 0.16))
    fw = w * frac
    if fw > 0.5:
        ui.box(x, cy - th / 2, max(fw, th), th, th / 2, (color or T.ACCENT_BLUE) if live else (1.0, 1.0, 1.0, 0.30))
    r = 11.0 * (1.0 + 0.10 * max(0.0, min(1.0, hover)))
    tx = x + fw
    ui.box(tx - r, cy - r + 1, 2 * r, 2 * r, r, None, None, shadow=7.0, shadow_dy=2.0, shadow_alpha=0.32)
    ui.circle(tx, cy, r, (1.0, 1.0, 1.0, 1.0) if live else (0.72, 0.72, 0.75, 1.0))
    if focus and live:
        ui.ring(tx, cy, r + 4.0, 2.0, T.FOCUS_RING)


def segmented(ui, rect, labels, index, mouse, *, key, focus=False, enabled=True, size=T.T_SMALL):
    """Capsule segmented control; the selection knob slides.  Returns the segment rects."""
    x, y, w, h = rect
    h = min(h, 32.0)
    y = rect[1] + (rect[3] - h) / 2
    pad = 2.0
    n = len(labels)
    widths = [ui.text_w(s, size, weight="semibold") + 26 for s in labels]
    total = sum(widths) + 2 * pad
    if total > w:
        k = (w - 2 * pad) / max(1.0, sum(widths))
        widths = [v * k for v in widths]
        total = sum(widths) + 2 * pad
    left = x + w - total

    ui.box(left, y, total, h, h / 2, T.FIELD)
    rects, cx = [], left + pad
    for sw in widths:
        rects.append((cx, y + pad, sw, h - 2 * pad))
        cx += sw

    fi = max(0.0, min(n - 1.0, spring(ui, ("seg_pos", key), float(index), T.SPRING_QUICK)))
    i0 = int(math.floor(fi))
    i1 = min(n - 1, i0 + 1)
    f = fi - i0
    kx = lerp(rects[i0][0], rects[i1][0], f)
    kw = lerp(rects[i0][2], rects[i1][2], f)
    kh = h - 2 * pad
    ui.box(kx, y + pad, kw, kh, kh / 2, None, None, shadow=4.0, shadow_dy=1.0, shadow_alpha=0.22)
    knob = ((0.46, 0.46, 0.49, 1.0), (0.40, 0.40, 0.43, 1.0))
    ui.box(kx, y + pad, kw, kh, kh / 2, knob if enabled else (scale_alpha(knob[0], 0.5), scale_alpha(knob[1], 0.5)))

    for i, (label, r) in enumerate(zip(labels, rects)):
        sel = i == index
        hov = spring(ui, ("seg", key, i), float(inside(r, *mouse) and enabled and not sel), T.SPRING_GENTLE)
        col = T.TEXT_PRIMARY if sel else mix(T.TEXT_SECONDARY, T.TEXT_PRIMARY, 0.6 * hov)
        if not enabled:
            col = T.TEXT_QUATERNARY
        ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2, size, col, anchor="center", weight="semibold")

    if focus and enabled:
        focus_ring(ui, left, y, total, h, h / 2, T.FOCUS_RING, width=2.0, inset=2.5)
    return rects


def stepper(ui, rect, label, mouse, *, key, focus=False, enabled=True, width=210.0, size=T.T_SMALL):
    """``<  value  >`` capsule for long option lists.  Returns the two arrow rects."""
    x, y, w, h = rect
    width = min(width, w)
    h = min(h, 32.0)
    y = rect[1] + (rect[3] - h) / 2
    left = x + w - width
    ui.box(left, y, width, h, h / 2, T.FIELD)

    arrows = []
    for i, name in enumerate(("chevron_l", "chevron_r")):
        r = (left + (0 if i == 0 else width - h), y, h, h)
        arrows.append(r)
        hov = spring(ui, ("stp", key, i), float(inside(r, *mouse) and enabled), T.SPRING_GENTLE)
        if hov > 0.01:
            ui.circle(r[0] + r[2] / 2, r[1] + r[3] / 2, h / 2 - 2, (1.0, 1.0, 1.0, 0.10 * hov))
        icons.draw(ui, name, r[0] + r[2] / 2, r[1] + r[3] / 2, 16,
                   T.TEXT_SECONDARY if enabled else T.TEXT_QUATERNARY, 1.8)
    ui.text_c(ui.fit(label, size, width - 2 * h), left + width / 2, y + h / 2, size,
              T.TEXT_PRIMARY if enabled else T.TEXT_QUATERNARY, anchor="center", weight="semibold")
    if focus and enabled:
        focus_ring(ui, left, y, width, h, h / 2, T.FOCUS_RING, width=2.0, inset=2.5)
    return arrows


def search_field(ui, x, y, w, h, text, placeholder, active=True, now=0.0):
    """Capsule search field with a magnifier and a blinking caret.  Returns the caret x."""
    ui.box(x, y, w, h, h / 2, T.FIELD)
    cy = y + h / 2
    ui.ring(x + 19.0, cy - 1.0, 5.2, 1.7, T.TEXT_TERTIARY)
    ui.line(x + 23.0, cy + 3.0, x + 26.5, cy + 6.5, 1.8, T.TEXT_TERTIARY)
    cx = x + 36
    if text:
        shown = ui.fit(text, T.T_BODY + 1, w - 58)
        cx += ui.text_c(shown, cx, cy, T.T_BODY + 1, T.TEXT_PRIMARY, weight="regular") + 1
    else:
        ui.text_c(placeholder, cx + 1, cy, T.T_BODY + 1, T.TEXT_TERTIARY)
    if active and int(now * 2) % 2 == 0:
        ui.rect(round(cx), round(cy - 8), 1.5, 16, T.ACCENT_BLUE)
    return cx


def scrollbar(ui, x, y, h, view, content, offset):
    """Thin, unobtrusive scroll indicator."""
    if content <= view + 1:
        return
    th = max(28.0, h * view / content)
    ty = y + (h - th) * (offset / max(1.0, content - view))
    ui.box(x, ty, 3.0, th, 1.5, (1.0, 1.0, 1.0, 0.30))


def progress(ui, x, y, w, h, frac, color=T.ACCENT_BLUE, shimmer=None):
    """Capsule progress bar; ``shimmer`` (0-1 phase) adds a soft moving highlight."""
    frac = max(0.0, min(1.0, float(frac)))
    ui.box(x, y, w, h, h / 2, (1.0, 1.0, 1.0, 0.14))
    if frac <= 0.0:
        return
    fw = max(h, w * frac)
    ui.box(x, y, fw, h, h / 2, color)
    if shimmer is not None and fw > 24:
        sw = min(60.0, fw * 0.6)
        sx = x + (fw - sw) * (shimmer % 1.0)
        ui.gradient(sx, y + 1, sw, max(1.0, h - 2), (1, 1, 1, 0.0), (1, 1, 1, 0.28))


def activity(ui, cx, cy, r, t, color=T.TEXT_SECONDARY):
    """The familiar eight-tick spinner."""
    for i in range(8):
        a = i * math.tau / 8.0
        lead = (t * 1.25 - i / 8.0) % 1.0
        al = 0.18 + 0.82 * (1.0 - lead)
        s, c = math.sin(a), -math.cos(a)
        ui.line(cx + s * r * 0.52, cy + c * r * 0.52, cx + s * r, cy + c * r, max(1.4, r * 0.2),
                (color[0], color[1], color[2], color[3] * al))


def check_circle(ui, cx, cy, r, color=T.OK_GREEN):
    """Filled circle with a white check - a finished step."""
    ui.circle(cx, cy, r, color)
    k = r / 10.0
    ui.line(cx - 4.2 * k, cy + 0.2 * k, cx - 1.2 * k, cy + 3.2 * k, max(1.6, 1.8 * k), WHITE)
    ui.line(cx - 1.2 * k, cy + 3.2 * k, cx + 4.6 * k, cy - 3.6 * k, max(1.6, 1.8 * k), WHITE)


def toast(ui, cx, y, text, ok=True, size=T.T_BODY):
    """A floating capsule notification centred on ``cx``."""
    tw = ui.text_w(text, size, weight="semibold")
    w, h = tw + 28 + 26, 38.0
    x = cx - w / 2
    glass(ui, x, y, w, h, h / 2, "raised")
    if ok:
        check_circle(ui, x + 21, y + h / 2, 8)
    else:
        ui.circle(x + 21, y + h / 2, 8, T.WARN_AMBER)
        ui.line(x + 21, y + h / 2 - 4, x + 21, y + h / 2 + 1, 1.8, WHITE)
        ui.circle(x + 21, y + h / 2 + 4.2, 1.1, WHITE)
    ui.text_c(text, x + 38, y + h / 2, size, T.TEXT_PRIMARY, weight="semibold")


# ═══════════════════════════════════════════════════════════════════════
# Cards & rows
# ═══════════════════════════════════════════════════════════════════════

def card(ui, x, y, w, h, *, selected=False, hovered=False, focused=False, enabled=True,
         elevation=0, border=None, r=None):
    """A tappable card: grouped fill, brighter on hover, accent outline when selected."""
    r = T.R_CARD if r is None else r
    hv = float(hovered)
    if selected and enabled:
        fill, bd = T.GROUP_SELECTED, border or T.ACCENT_BLUE
    else:
        fill, bd = mix_pair(T.GROUP, T.GROUP_HOVER, hv), border
    if not enabled:
        fill = (scale_alpha(fill[0], 0.6), scale_alpha(fill[1], 0.6))
    shadow = {0: None, 1: T.SHADOW_1, 2: T.SHADOW_2, 3: T.SHADOW_3, 4: T.SHADOW_4}.get(elevation) if enabled else None
    surface(ui, x, y, w, h, r, fill, None, shadow)
    if bd is not None:
        ui.box(x, y, w, h, min(r, w / 2, h / 2), None, bd, bw=2.0 if (selected and enabled) else 1.0)
    if focused and enabled and not selected:
        focus_ring(ui, x, y, w, h, r, T.FOCUS_RING, width=2.0, inset=2.0)


def kv_row(ui, x, y, w, h, label, value, icon=None):
    """A key / value row: quiet label on the left, the value on the right."""
    lx = x
    if icon:
        icons.draw(ui, icon, x + 11, y + h / 2, 18, T.TEXT_TERTIARY, 1.5)
        lx = x + 30
    ui.text_c(label, lx, y + h / 2, T.T_BODY, T.TEXT_SECONDARY)
    ui.text_c(ui.fit(str(value), T.T_BODY, w - (lx - x) - ui.text_w(label, T.T_BODY) - 16), x + w, y + h / 2,
              T.T_BODY, T.TEXT_PRIMARY, anchor="right", weight="semibold")


# Backward compatibility: the design tokens used to be importable from here.
from .theme import *  # noqa: E402,F401,F403
