"""Widget drawing helpers shared by every screen - the DedSec-terminal building blocks.

Everything here is stateless drawing on top of ``Overlay2D``: callers own the geometry (so
hit-testing and drawing can never disagree) and pass a stable ``key`` for each widget, which
``Overlay2D.anim`` / ``anim_spring`` use to ease hover state.  Coordinates are logical pixels.

Vocabulary (see ``theme`` for the values):

* ``frame`` / ``glass`` / ``sheet``   black panels with thin white outlines and corner brackets
* ``button`` / ``card`` / bars        flat dark bars; hover and focus turn them solid blue
* ``toggle``                          the text values ``On`` (green) and ``Off`` (red)
* ``slider``                          a dashed track with a marker:  ``[- - - - X - - - -]``
* ``segmented`` / ``stepper``         option labels; the chosen one is a blue block
* ``caps``                            tracked-out capitals for headings, tabs and hints
* ``dots``                            one batched call that draws thousands of halftone dots
* ``prompt_bar``                      the white strip along the bottom: version + key hints
"""
from __future__ import annotations

import math

import numpy as np

from . import icons
from . import theme as T

# Pointer state, set once per frame by the menu system, so buttons can show a pressed state.
POINTER = {"down": False}

# Legacy accent override (the previous design tinted controls per page); unused here.
TINT = {"color": None}

# Text shown at the left of the hint strip (the host sets it, e.g. "v1.4.0").
CHROME = {"version": "", "user": "guest"}

TOGGLE_W, TOGGLE_H = 64.0, 30.0
WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# Maths, animation, setup
# ═══════════════════════════════════════════════════════════════════════

def lerp(a, b, t):
    return a + (b - a) * t


def mix(c0, c1, t):
    t = max(0.0, min(1.0, t))
    return tuple(lerp(a, b, t) for a, b in zip(c0, c1))


def mix_pair(p0, p1, t):
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


def hsv(h, s=0.6, v=1.0, a=1.0):
    h = (h % 1.0) * 6.0
    i, f = int(h), h - int(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i % 6]
    return (r, g, b, a)


def prepare(ui):
    """Square every corner drawn through ``ui.box``: the whole design is flat rectangles.

    Installed once per overlay instance; call sites keep passing radii, they just no longer apply.
    """
    if getattr(ui, "_gaia_square", False):
        return
    raw = ui.box

    def square_box(x, y, w, h, r=0.0, *args, **kw):
        return raw(x, y, w, h, 0.0, *args, **kw)

    try:
        ui.box = square_box
        ui._gaia_square = True
    except AttributeError:                      # an overlay that forbids new attributes: keep rounded boxes
        pass


def spring(ui, key, target, preset=T.SPRING_GENTLE):
    """Spring-eased value; falls back to the plain eased ``anim`` on older overlays."""
    prepare(ui)
    fn = getattr(ui, "anim_spring", None)
    if fn is not None:
        return fn(key, target, *preset)
    return ui.anim(key, target, 18.0)


# ═══════════════════════════════════════════════════════════════════════
# Tracked capitals and dot batches
# ═══════════════════════════════════════════════════════════════════════

def caps_w(ui, text, size, weight="semibold", track=T.LETTER_TRACK):
    text = text.upper()
    return sum(ui.text_w(ch, size, weight=weight) for ch in text) + track * size * max(0, len(text) - 1)


def caps(ui, text, x, y, size, color, weight="semibold", anchor="left", track=T.LETTER_TRACK, vcenter=False):
    """Capitals with extra letter-spacing.  ``y`` is the top, or the middle when ``vcenter``."""
    text = text.upper()
    w = caps_w(ui, text, size, weight, track)
    if anchor == "center":
        x -= w / 2
    elif anchor == "right":
        x -= w
    put = ui.text_c if vcenter else ui.text
    cx = x
    for ch in text:
        if ch != " ":
            put(ch, cx, y, size, color, weight=weight)
        cx += ui.text_w(ch, size, weight=weight) + track * size
    return w


def dots(ui, xs, ys, sx, sy=None, rgba=(1.0, 1.0, 1.0, 1.0)):
    """Draw many small rectangles in ONE call.  ``sx``/``sy`` are half-sizes (scalar or array);
    ``rgba`` is one colour or an (N, 4) array."""
    xs = np.asarray(xs, np.float32)
    ys = np.asarray(ys, np.float32)
    n = len(xs)
    if n == 0:
        return
    hx = np.broadcast_to(np.asarray(sx, np.float32), (n,))
    hy = hx if sy is None else np.broadcast_to(np.asarray(sy, np.float32), (n,))
    x0, x1, y0, y1 = xs - hx, xs + hx, ys - hy, ys + hy
    v = np.stack([np.stack([x0, y0], 1), np.stack([x1, y0], 1), np.stack([x1, y1], 1),
                  np.stack([x0, y0], 1), np.stack([x1, y1], 1), np.stack([x0, y1], 1)], 1)
    col = np.asarray(rgba, np.float32)
    c = np.broadcast_to(col[:, None, :] if col.ndim == 2 else col[None, None, :], (n, 6, 4))
    ui.triangles(np.ascontiguousarray(v.reshape(-1, 2)), np.ascontiguousarray(c.reshape(-1, 4)))


def dot_grid(ui, x, y, w, h, spacing=26.0, color=(1.0, 1.0, 1.0, 0.22), size=0.9):
    """A sparse grid of single-pixel dots - the DedSec background texture."""
    gx = np.arange(x + spacing / 2, x + w, spacing, dtype=np.float32)
    gy = np.arange(y + spacing / 2, y + h, spacing, dtype=np.float32)
    if len(gx) == 0 or len(gy) == 0:
        return
    xx, yy = np.meshgrid(gx, gy)
    dots(ui, xx.ravel(), yy.ravel(), size, None, color)


# ═══════════════════════════════════════════════════════════════════════
# Surfaces
# ═══════════════════════════════════════════════════════════════════════

def hgradient(ui, x, y, w, h, left, right):
    v = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], np.float32)
    c = np.array([left, right, right, left, right, left], np.float32)
    ui.triangles(v, c)


def vgradient(ui, x, y, w, h, top, bottom):
    v = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y], [x + w, y + h], [x, y + h]], np.float32)
    c = np.array([top, top, bottom, top, bottom, bottom], np.float32)
    ui.triangles(v, c)


def drop_shadow(ui, x, y, w, h, r, shadow):
    """The design has no shadows; kept so older call sites still work."""


def surface(ui, x, y, w, h, r=0.0, fill=T.GLASS, border=None, shadow=None, specular=False):
    """A flat rectangle.  ``fill`` is a (top, bottom) pair or one RGBA colour."""
    prepare(ui)
    ui.box(x, y, w, h, 0.0, fill, border, bw=1.0)


def frame(ui, x, y, w, h, color=T.LINE, title=None, notch=None):
    """The info-panel frame: hair outline, heavy corner brackets and (optionally) a title notch."""
    prepare(ui)
    ui.box(x, y, w, h, 0.0, None, scale_alpha(color, 0.45), bw=1.0)
    leg, th = 14.0, 2.0
    for cx, cy, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        ui.rect(cx if sx > 0 else cx - leg, cy if sy > 0 else cy - th, leg, th, color)
        ui.rect(cx if sx > 0 else cx - th, cy if sy > 0 else cy - leg, th, leg, color)
    if title:
        tw = caps_w(ui, title, T.T_CAPTION_2) + 22
        nx = x + (w - tw) / 2 if notch is None else notch
        ui.rect(nx, y - 1, tw, 3, T.BLACK)
        caps(ui, title, nx + tw / 2, y - 7, T.T_CAPTION_2, color, anchor="center")


GLASS_LEVELS = {
    "hud": (T.GLASS, scale_alpha(T.LINE, 0.30)),
    "menu": (T.GLASS_STRONG, scale_alpha(T.LINE, 0.45)),
    "sheet": (T.SHEET, None),
    "raised": (T.SHEET_RAISED, scale_alpha(T.LINE, 0.65)),
}


def glass(ui, x, y, w, h, r=0.0, level="hud", shadow=True):
    """The standard dark panel.  ``level``: hud, menu, sheet (bracketed), raised."""
    fill, border = GLASS_LEVELS.get(level, GLASS_LEVELS["hud"])
    surface(ui, x, y, w, h, 0.0, fill, border)
    if level == "sheet":
        frame(ui, x, y, w, h)


def sheet(ui, x, y, w, h):
    """A full page / dialog panel with corner brackets."""
    glass(ui, x, y, w, h, 0.0, "sheet")


def group(ui, x, y, w, h, r=0.0, hover=0.0, selected=False, fill=None):
    if fill is None:
        fill = T.GROUP_SELECTED if selected else mix_pair(T.GROUP, T.GROUP_HOVER, hover)
    surface(ui, x, y, w, h, 0.0, fill)


def separator(ui, x, y, w, color=None):
    ui.rect(x, y, w, 1, color or T.SEPARATOR)


def divider(ui, x, y, w, color=None):
    ui.rect(x, y, w, 1, color or T.HAIRLINE)


def vdivider(ui, x, y, h, color=None):
    ui.rect(x, y, 1, h, color or T.HAIRLINE)


def scrim(ui, w, h, k=1.0, flat=False):
    """Darken the live scene behind a menu."""
    r, g, b = T.SCRIM_BASE
    if flat:
        ui.rect(0, 0, w, h, (r, g, b, 0.93 * k))
    else:
        hgradient(ui, 0, 0, w, h, (r, g, b, 0.90 * k), (r, g, b, 0.62 * k))


def material_box(ui, x, y, w, h, r=0.0, variant="regular", border=None):
    fills = {"thin": T.MATERIAL_THIN, "regular": T.MATERIAL_REGULAR, "thick": T.MATERIAL_THICK,
             "tinted": T.MATERIAL_TINTED, "field": T.FIELD}
    prepare(ui)
    ui.box(x, y, w, h, 0.0, fills.get(variant, T.MATERIAL_REGULAR), border, bw=1.0)


def focus_ring(ui, x, y, w, h, r=0.0, color=T.FOCUS_RING, width=2.0, inset=0.0):
    prepare(ui)
    ui.box(x - inset, y - inset, w + 2 * inset, h + 2 * inset, 0.0, None, color, bw=width)


def bar(ui, x, y, w, h, hover=0.0, selected=False, enabled=True):
    """A menu / settings row: dark bar, solid blue when selected."""
    if selected and enabled:
        fill = T.BAR_SELECTED
    else:
        fill = mix(T.BAR, T.BAR_HOVER, hover) if enabled else (0.05, 0.05, 0.05, 1.0)
    ui.rect(x, y, w, h, fill)


# ═══════════════════════════════════════════════════════════════════════
# Text helpers
# ═══════════════════════════════════════════════════════════════════════

def heading(ui, text, x, y, color=T.TEXT_TERTIARY, size=T.T_CAPTION_2):
    """Section title: ``// TRACKED CAPITALS`` in grey."""
    caps(ui, "// " + text, x, y, size, color)


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
    lead = lead or size + 6
    lines = wrap(ui, text, size, max_w, weight, max_lines)
    for i, line in enumerate(lines):
        ui.text(line, x, y + i * lead, size, color, anchor=anchor, weight=weight)
    return len(lines) * lead


# ═══════════════════════════════════════════════════════════════════════
# Key labels and the hint strip
# ═══════════════════════════════════════════════════════════════════════

def _key_tokens(key):
    out = []
    for part in key.split(" / "):
        part = part.strip()
        bits = part.split()
        if len(bits) > 1 and all(len(b) == 1 for b in bits):
            out.extend(bits)
        else:
            out.append(part)
    return out


def keycaps(ui, x, y, key, size=11, dry=False, inverse=False):
    """Key label(s): a white chip with black bold text (black with white text when ``inverse``)."""
    face, ink = (BLACK, WHITE) if inverse else (WHITE, BLACK)
    cx, kh = x, size + 10
    for i, tok in enumerate(_key_tokens(key)):
        if i:
            cx += 3
        tw = ui.text_w(tok, size, weight="bold")
        kw = max(size + 10, tw + 12)
        if not dry:
            ui.rect(cx, y, kw, kh, face)
            ui.text_c(tok, cx + kw / 2, y + kh / 2, size, ink, anchor="center", weight="bold")
        cx += kw
    return cx - x


def prompt_bar(ui, w, h, items, *, height=T.HINT_BAR_H):
    """The white strip along the bottom: version on the left, key hints on the right."""
    prepare(ui)
    y = h - height
    ui.rect(0, y, w, height, WHITE)
    size = 13
    ver = CHROME.get("version") or ""
    vw = 0.0
    if ver:
        vw = ui.text_w(ver, size, weight="medium")
        ui.text_c(ver, 24, y + height / 2, size, T.TEXT_INVERTED, weight="medium")

    gap = 26.0
    items = list(items)

    def widths(its):
        return [keycaps(ui, 0, 0, k, 11, dry=True) + 8 + caps_w(ui, lbl, size, "bold", 0.04) for k, lbl in its]

    ws = widths(items)
    avail = w - 48 - vw - 40
    while len(items) > 1 and sum(ws) + gap * (len(items) - 1) > avail:
        items.pop(len(items) - 2 if len(items) > 2 else 0)
        ws = widths(items)
    cx = w - 24 - sum(ws) - gap * (len(items) - 1)
    for (k, lbl), iw in zip(items, ws):
        kw = keycaps(ui, cx, y + (height - 21) / 2, k, 11, inverse=True)
        caps(ui, lbl, cx + kw + 8, y + height / 2, size, T.TEXT_INVERTED, "bold", track=0.04, vcenter=True)
        cx += iw + gap


# ═══════════════════════════════════════════════════════════════════════
# Buttons, chips, tiles
# ═══════════════════════════════════════════════════════════════════════

def button(ui, rect, label, mouse, *, key, kind="secondary", icon=None, focus=False,
           disabled=False, size=T.T_BODY, hover=None):
    """A flat bar.  Hover or focus makes it solid blue.  ``kind``: primary, secondary, tertiary,
    destructive.  Returns the hover amount (0-1)."""
    prepare(ui)
    x, y, w, h = rect
    over = inside(rect, *mouse) and not disabled
    if hover is None:
        hover = spring(ui, ("btn", key), float(over), T.SPRING_GENTLE)
    hover = max(0.0, min(1.0, hover))
    lit = 1.0 if (focus and not disabled) else hover
    press = POINTER["down"] and over

    if disabled:
        fill, col = (0.05, 0.05, 0.05, 1.0), T.TEXT_QUATERNARY
    elif kind == "destructive":
        fill, col = mix((0.55, 0.05, 0.05, 1.0), T.WD_RED, lit), WHITE
    elif kind == "primary":
        fill, col = mix(T.WD_BLUE_DEEP, T.WD_BLUE, max(lit, 0.6)), WHITE
    elif kind == "tertiary":
        fill, col = mix((0.0, 0.0, 0.0, 0.0), T.WD_BLUE, lit), mix(T.TEXT_SECONDARY, WHITE, lit)
    else:
        fill, col = mix(T.BAR, T.WD_BLUE, lit), WHITE if lit > 0.3 else T.TEXT_PRIMARY
    if press:
        fill = mix(fill, BLACK, 0.25)
    if fill[3] > 0.01:
        ui.rect(x, y, w, h, fill)

    tw = ui.text_w(label, size, weight="medium")
    isz = size + 3
    total = tw + ((isz + 8) if icon else 0)
    cx = x + (w - total) / 2
    if icon:
        icons.draw(ui, icon, cx + isz / 2, y + h / 2, isz, col, 1.5)
        cx += isz + 8
    ui.text_c(label, cx, y + h / 2, size, col, weight="medium")
    return hover


def chip(ui, x, y, text, color=T.TEXT_SECONDARY, size=T.T_CAPTION_1, h=20, pad=8, anchor="left",
         fill=None, border=None, icon=None):
    """A small tracked-caps label in a square outline.  Returns its width."""
    prepare(ui)
    tw = caps_w(ui, text, size)
    iw = (size + 4) if icon else 0
    w = tw + 2 * pad + iw
    if anchor == "right":
        x -= w
    elif anchor == "center":
        x -= w / 2
    ui.box(x, y, w, h, 0.0, fill if fill is not None else (0.0, 0.0, 0.0, 0.0), border or scale_alpha(color, 0.7), bw=1.0)
    if icon:
        icons.draw(ui, icon, x + pad + size * 0.5, y + h / 2, size + 2, color, 1.3)
    caps(ui, text, x + pad + iw, y + h / 2, size, color, vcenter=True)
    return w


def icon_tile(ui, x, y, s, icon, tile=T.TILE_GRAY, enabled=True):
    """A flat square with a white glyph.  Blue / green / red tiles carry colour; the rest are grey."""
    k = 1.0 if enabled else 0.4
    ui.rect(x, y, s, s, scale_alpha(tile[0], k))
    icons.draw(ui, icon, x + s / 2, y + s / 2, s * 0.58, (1.0, 1.0, 1.0, k), max(1.4, s / 18.0))


def close_button(ui, x, y, mouse, key="close", size=32):
    """A small square with an X; blue on hover."""
    rect = (x, y, size, size)
    hv = spring(ui, ("closebtn", key), float(inside(rect, *mouse)), T.SPRING_GENTLE)
    ui.rect(x, y, size, size, mix(T.BAR, T.WD_BLUE, hv))
    icons.draw(ui, "close", x + size / 2, y + size / 2, size * 0.42, WHITE, 1.8)
    return rect


def logo_mark(ui, x, y, s=40.0, t=0.0):
    """The Project GAIA mark: a white wire-frame globe in a black square."""
    prepare(ui)
    ui.box(x, y, s, s, 0.0, BLACK, (1.0, 1.0, 1.0, 0.9), bw=max(1.0, s * 0.035))
    cx, cy, r = x + s / 2, y + s / 2, s * 0.29
    w = max(1.2, s * 0.04)
    ui.ring(cx, cy, r, w, WHITE)
    ui.line(cx - r, cy, cx + r, cy, w * 0.8, WHITE)
    ui.line(cx, cy - r, cx, cy + r, w * 0.8, WHITE)
    for sgn in (-1, 1):
        dy = sgn * r * 0.52
        hw = math.sqrt(max(0.0, r * r - dy * dy))
        ui.line(cx - hw, cy + dy, cx + hw, cy + dy, w * 0.7, (1.0, 1.0, 1.0, 0.8))
    pts = [(cx + r * 0.46 * math.cos(a), cy + r * math.sin(a)) for a in [i * math.tau / 20 for i in range(20)]]
    ui.outline_poly(pts, w * 0.8, (1.0, 1.0, 1.0, 0.85), True)


def panel_header(ui, x, y, w, title, subtitle=None, icon=None, tile=T.TILE_GRAY):
    """Page title: ``// TITLE`` in tracked capitals, a quiet line under it.  Returns the y below."""
    tx = x
    if icon:
        icon_tile(ui, x, y + 2, 44, icon, tile)
        tx = x + 58
    caps(ui, "// " + title, tx, y + (3 if subtitle else 8), T.T_TITLE_3 - 2, WHITE, "bold")
    if subtitle:
        ui.text(ui.fit(subtitle, T.T_BODY, w - (tx - x)), tx, y + 34, T.T_BODY, T.TEXT_TERTIARY)
    return y + 56


# ═══════════════════════════════════════════════════════════════════════
# Controls
# ═══════════════════════════════════════════════════════════════════════

def toggle_rect(rect):
    x, y, w, h = rect
    return (x + w - TOGGLE_W, y + (h - TOGGLE_H) / 2, TOGGLE_W, TOGGLE_H)


def toggle(ui, rect, on, *, key, focus=False, enabled=True):
    """A switch is just its value in the terminal: green ``On`` / red ``Off``."""
    x, y, w, h = toggle_rect(rect)
    col = T.WD_GREEN if on else T.WD_RED
    if not enabled:
        col = T.TEXT_QUATERNARY
    ui.text_c("On" if on else "Off", x + w, y + h / 2, T.T_BODY, col, anchor="right", weight="medium")


def slider(ui, rect, frac, *, key, hover=0.0, focus=False, enabled=True, color=None):
    """Dashed track in brackets with a marker:  ``[ - - - - X - - - - ]``."""
    x, y, w, h = rect
    frac = max(0.0, min(1.0, float(frac)))
    cy = y + h / 2
    live = enabled
    on = (color or T.WD_GREEN) if live else T.TEXT_QUATERNARY
    off = scale_alpha(on, 0.40)
    tx = x + 8 + (w - 16) * frac
    ui.rect(x, cy - 7, 2, 14, off)
    ui.rect(x, cy - 7, 5, 2, off)
    ui.rect(x, cy + 5, 5, 2, off)
    ui.rect(x + w - 2, cy - 7, 2, 14, off)
    ui.rect(x + w - 5, cy - 7, 5, 2, off)
    ui.rect(x + w - 5, cy + 5, 5, 2, off)
    pos, dash, gap = x + 10, 8.0, 5.0
    while pos + dash <= x + w - 10:
        ui.rect(pos, cy - 1, dash, 2, on if pos + dash / 2 <= tx else off)
        pos += dash + gap
    mk = WHITE if live else T.TEXT_QUATERNARY
    ui.rect(tx - 1, cy - 8, 2, 16, mk)
    ui.poly([(tx - 5, cy - 9), (tx + 5, cy - 9), (tx, cy - 15)], mk)
    ui.poly([(tx - 5, cy + 9), (tx + 5, cy + 9), (tx, cy + 15)], mk)


def segmented(ui, rect, labels, index, mouse, *, key, focus=False, enabled=True, size=T.T_SMALL):
    """Option labels in a row; the chosen one is a solid blue block.  Returns the option rects."""
    x, y, w, h = rect
    h = min(h, 30.0)
    y = rect[1] + (rect[3] - h) / 2
    widths = [ui.text_w(s, size, weight="medium") + 24 for s in labels]
    total = sum(widths)
    if total > w:
        k = w / max(1.0, total)
        widths = [v * k for v in widths]
        total = sum(widths)
    left = x + w - total
    rects, cx = [], left
    for sw in widths:
        rects.append((cx, y, sw, h))
        cx += sw
    for i, (label, r) in enumerate(zip(labels, rects)):
        sel = i == index
        hov = spring(ui, ("seg", key, i), float(inside(r, *mouse) and enabled and not sel), T.SPRING_GENTLE)
        if sel:
            ui.rect(*r, T.WD_BLUE if enabled else (0.12, 0.12, 0.12, 1.0))
            col = WHITE if enabled else T.TEXT_QUATERNARY
        else:
            if hov > 0.01:
                ui.rect(*r, (1.0, 1.0, 1.0, 0.10 * hov))
            col = T.WD_GREEN if enabled else T.TEXT_QUATERNARY
            col = mix(scale_alpha(col, 0.55), col, hov)
        ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2, size, col, anchor="center", weight="medium")
    return rects


def stepper(ui, rect, label, mouse, *, key, focus=False, enabled=True, width=210.0, size=T.T_SMALL):
    """``<  VALUE  >`` with small triangles.  Returns the two arrow rects."""
    x, y, w, h = rect
    width = min(width, w)
    h = min(h, 30.0)
    y = rect[1] + (rect[3] - h) / 2
    left = x + w - width
    arrows = []
    for i in range(2):
        r = (left + (0 if i == 0 else width - h), y, h, h)
        arrows.append(r)
        hov = spring(ui, ("stp", key, i), float(inside(r, *mouse) and enabled), T.SPRING_GENTLE)
        if hov > 0.01:
            ui.rect(*r, (0.0, 0.0, 1.0, hov))
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        d = 1 if i else -1
        ui.poly([(cx + d * 4, cy), (cx - d * 3, cy - 6), (cx - d * 3, cy + 6)],
                WHITE if hov > 0.5 else (T.WD_GREEN if enabled else T.TEXT_QUATERNARY))
    ui.text_c(ui.fit(label, size, width - 2 * h), left + width / 2, y + h / 2, size,
              T.WD_GREEN if enabled else T.TEXT_QUATERNARY, anchor="center", weight="medium")
    return arrows


def search_field(ui, x, y, w, h, text, placeholder, active=True, now=0.0):
    """A dark input line with a green prompt and an underline.  Returns the caret x."""
    ui.rect(x, y, w, h, T.FIELD[0])
    ui.rect(x, y + h - 2, w, 2, WHITE if active else T.LINE_DIM)
    cy = y + h / 2
    ui.text_c(">", x + 12, cy, T.T_BODY + 1, T.WD_GREEN, weight="bold")
    cx = x + 30
    if text:
        shown = ui.fit(text, T.T_BODY + 1, w - 52)
        cx += ui.text_c(shown, cx, cy, T.T_BODY + 1, T.TEXT_PRIMARY) + 1
    else:
        ui.text_c(placeholder, cx + 1, cy, T.T_BODY + 1, T.TEXT_TERTIARY)
    if active and int(now * 2) % 2 == 0:
        ui.rect(round(cx), round(cy - 8), 7, 16, T.WD_GREEN)
    return cx


def scrollbar(ui, x, y, h, view, content, offset):
    if content <= view + 1:
        return
    th = max(28.0, h * view / content)
    ty = y + (h - th) * (offset / max(1.0, content - view))
    ui.rect(x, y, 2, h, (1.0, 1.0, 1.0, 0.10))
    ui.rect(x, ty, 2, th, (1.0, 1.0, 1.0, 0.7))


def progress(ui, x, y, w, h, frac, color=T.ACCENT_BLUE, shimmer=None):
    """A thin flat bar.  Blue (the old default) is drawn green: blue on black is too dim to read."""
    frac = max(0.0, min(1.0, float(frac)))
    if color == T.ACCENT_BLUE:
        color = T.WD_GREEN
    ui.rect(x, y, w, h, (1.0, 1.0, 1.0, 0.16))
    if frac <= 0.0:
        return
    fw = max(2.0, w * frac)
    ui.rect(x, y, fw, h, color)
    if shimmer is not None and fw > 24:
        sw = min(46.0, fw * 0.5)
        sx = x + (fw - sw) * (shimmer % 1.0)
        ui.rect(sx, y, sw, h, (1.0, 1.0, 1.0, 0.35))


def activity(ui, cx, cy, r, t, color=T.TEXT_SECONDARY):
    """Eight square dots on a ring, chasing each other."""
    for i in range(8):
        a = i * math.tau / 8.0
        lead = (t * 1.25 - i / 8.0) % 1.0
        al = 0.15 + 0.85 * (1.0 - lead)
        ui.rect(cx + math.sin(a) * r - 1.5, cy - math.cos(a) * r - 1.5, 3, 3, (color[0], color[1], color[2], color[3] * al))


def check_circle(ui, cx, cy, r, color=T.OK_GREEN):
    """A finished step: a green outlined square with a check."""
    s = r * 1.7
    ui.box(cx - s / 2, cy - s / 2, s, s, 0.0, None, color, bw=1.5)
    k = r / 10.0
    ui.line(cx - 3.6 * k, cy + 0.2 * k, cx - 1.0 * k, cy + 2.8 * k, max(1.5, 1.7 * k), color)
    ui.line(cx - 1.0 * k, cy + 2.8 * k, cx + 4.0 * k, cy - 3.2 * k, max(1.5, 1.7 * k), color)


def toast(ui, cx, y, text, ok=True, size=T.T_BODY):
    """A short notification bar with a coloured edge."""
    tw = ui.text_w(text, size, weight="medium")
    w, h = tw + 44, 36.0
    x = cx - w / 2
    ui.rect(x, y, w, h, (0.03, 0.03, 0.03, 0.97))
    ui.rect(x, y, 4, h, T.WD_GREEN if ok else T.WD_AMBER)
    ui.text_c(text, x + 20, y + h / 2, size, WHITE, weight="medium")


# ═══════════════════════════════════════════════════════════════════════
# Cards & rows
# ═══════════════════════════════════════════════════════════════════════

def card(ui, x, y, w, h, *, selected=False, hovered=False, focused=False, enabled=True,
         elevation=0, border=None, r=None, tint=None):
    """A tappable card: a dark bar that turns solid blue when selected."""
    prepare(ui)
    hv = float(hovered)
    if not enabled:
        ui.rect(x, y, w, h, (0.045, 0.045, 0.045, 1.0))
    else:
        ui.rect(x, y, w, h, T.BAR_SELECTED if selected else mix(T.BAR, T.BAR_HOVER, hv))
    if border is not None:
        ui.box(x, y, w, h, 0.0, None, border, bw=1.0)
    if focused and enabled and not selected:
        focus_ring(ui, x, y, w, h, 0.0, T.FOCUS_RING, width=2.0, inset=0.0)


def kv_row(ui, x, y, w, h, label, value, icon=None):
    """A key / value row: white label on the left, the value in green on the right."""
    lx = x
    if icon:
        icons.draw(ui, icon, x + 11, y + h / 2, 18, T.TEXT_TERTIARY, 1.5)
        lx = x + 30
    ui.text_c(label, lx, y + h / 2, T.T_BODY, T.TEXT_SECONDARY)
    ui.text_c(ui.fit(str(value), T.T_BODY, w - (lx - x) - ui.text_w(label, T.T_BODY) - 16), x + w, y + h / 2,
              T.T_BODY, T.WD_GREEN, anchor="right", weight="medium")


# Backward compatibility: the design tokens used to be importable from here.
from .theme import *  # noqa: E402,F401,F403
