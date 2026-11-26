"""Widget drawing helpers shared by every screen — Apple liquid-glass building blocks.

Everything here is stateless drawing on top of ``Overlay2D``: callers own the geometry (so
hit-testing and drawing can never disagree) and pass a stable ``key`` for each widget, which
``Overlay2D.anim`` / ``anim_spring`` use to ease hover state.  Coordinates are logical pixels.

Vocabulary (see ``theme`` for the values):

* ``glass`` / ``sheet`` / ``group``   frosted, rounded panels with a hairline and a soft shadow
* ``button``                          capsules and rounded rectangles, blue for the primary action
* ``toggle``                          an iOS switch with a sliding knob
* ``slider``                          a 4 pt track with a white circular knob
* ``segmented``                       a capsule track with a sliding white selection
* ``stepper``                         ``−   VALUE   +``
* ``caps``                            a title / label (sentence case; a stray ``//`` is removed)
* ``dots``                            one batched call that draws thousands of dots
* ``prompt_bar``                      a floating, translucent key-hint capsule
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

# Text shown in the chrome (the host sets it, e.g. "v1.4.0").
CHROME = {"version": "", "user": "guest"}

TOGGLE_W, TOGGLE_H = 48.0, 28.0
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


def bezier(p0, p1, p2, p3, t):
    """Cubic-bezier value at ``t`` (x is time, y is progress) — UIKit-style easing."""
    t = max(0.0, min(1.0, t))
    u = 1.0 - t
    return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3


def ease_bezier(t, curve=T.EASE_EMPHASIZED):
    """Approximate a CSS cubic-bezier timing function by sampling it."""
    t = max(0.0, min(1.0, t))
    x1, y1, x2, y2 = curve
    lo, hi = 0.0, 1.0
    for _ in range(12):
        mid = (lo + hi) * 0.5
        if bezier(0.0, x1, x2, 1.0, mid) < t:
            lo = mid
        else:
            hi = mid
    return bezier(0.0, y1, y2, 1.0, (lo + hi) * 0.5)


def hsv(h, s=0.6, v=1.0, a=1.0):
    h = (h % 1.0) * 6.0
    i, f = int(h), h - int(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i % 6]
    return (r, g, b, a)


def prepare(ui):
    """Compatibility hook.  The design is rounded now, so nothing is overridden."""
    return ui


def spring(ui, key, target, preset=T.SPRING_GENTLE):
    """Spring-eased value; falls back to the plain eased ``anim`` on older overlays."""
    fn = getattr(ui, "anim_spring", None)
    if fn is not None:
        return fn(key, target, *preset)
    return ui.anim(key, target, 18.0)


# ═══════════════════════════════════════════════════════════════════════
# Titles, labels and dot batches
# ═══════════════════════════════════════════════════════════════════════

def _label(text):
    """Normalise legacy header strings: drop the terminal ``//`` and keep sentence case."""
    text = str(text or "")
    while text.startswith("//"):
        text = text[2:].lstrip()
    return text


def caps_w(ui, text, size, weight="semibold", track=0.0):
    text = _label(text)
    return ui.text_w(text, size, weight=weight)


def caps(ui, text, x, y, size, color, weight="semibold", anchor="left", track=0.0, vcenter=False):
    """A title or label in sentence case.  ``y`` is the top, or the middle when ``vcenter``."""
    text = _label(text)
    put = ui.text_c if vcenter else ui.text
    return put(text, x, y, size, color, anchor=anchor, weight=weight)


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


def dot_grid(ui, x, y, w, h, spacing=26.0, color=(1.0, 1.0, 1.0, 0.10), size=0.7):
    """A very sparse grid of single-pixel dots — a quiet depth cue behind menus."""
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


def _shadow_params(shadow):
    """Accept a ``(dy, blur, spread, colour)`` token, a blur radius or None."""
    if shadow is None or shadow is False:
        return None
    if isinstance(shadow, (int, float)):
        return (float(shadow) * 0.35, float(shadow), T.SHADOW_2[3])
    if len(shadow) >= 4:
        return (shadow[0], shadow[1], shadow[3])
    return (T.SHADOW_2[0], T.SHADOW_2[1], T.SHADOW_2[3])


def drop_shadow(ui, x, y, w, h, r, shadow):
    """Draw only the soft shadow behind a rounded rect."""
    params = _shadow_params(shadow)
    if params is None:
        return
    dy, blur, colour = params
    ui.box(x, y, w, h, r, None, None, shadow=blur, shadow_dy=dy, shadow_alpha=colour[3])


# Panels above this area get a real backdrop blur; small chips and buttons only get a tint.
BLUR_MIN_AREA = 6000.0
# ... but only up to here: a full-screen sheet is nearly opaque and would cost a
# whole-framebuffer copy every frame for a blur nobody can see.
BLUR_MAX_AREA = 900_000.0
BLUR_MAX_ALPHA = 0.90


def surface(ui, x, y, w, h, r=0.0, fill=T.GLASS, border=None, shadow=None, specular=False):
    """A rounded surface.  ``fill`` is a (top, bottom) pair or one RGBA colour.

    Large elevated panels first frost whatever is behind them (a real gaussian
    blend of the scene), so the glass genuinely defocuses the view underneath.
    A one-point specular highlight along the top edge gives the glass its lip.
    """
    params = _shadow_params(shadow)
    if params is not None:
        frost = getattr(ui, "frost", None)
        # Only translucent panes gain from a backdrop blur: a nearly opaque
        # sheet hides it completely, and frosting a full-screen sheet would
        # copy the whole framebuffer every frame for nothing.
        pair = isinstance(fill, tuple) and len(fill) == 2 and hasattr(fill[0], "__len__")
        alpha = max(fill[0][3], fill[1][3]) if pair else fill[3]
        if frost is not None and BLUR_MIN_AREA <= w * h <= BLUR_MAX_AREA and alpha <= BLUR_MAX_ALPHA:
            frost(x, y, w, h, r, 0.94)
        dy, blur, colour = params
        ui.box(x, y, w, h, r, None, None, shadow=blur, shadow_dy=dy, shadow_alpha=colour[3])
    ui.box(x, y, w, h, r, fill, border, bw=1.0)
    if specular and h > 8:
        ui.box(x + r * 0.6, y + 0.5, max(0.0, w - r * 1.2), 1.0, 0.5,
               (1.0, 1.0, 1.0, 0.16))


def frame(ui, x, y, w, h, color=T.LINE, title=None, notch=None):
    """A rounded panel with a hairline and, optionally, a title inside its top-left."""
    surface(ui, x, y, w, h, T.R_PANEL, T.GLASS_STRONG, color, T.SHADOW_3)
    if title:
        caps(ui, title, x + 20, y + 18, T.T_CAPTION_2, T.TEXT_SECONDARY, "semibold")


GLASS_LEVELS = {
    "hud": (T.GLASS, T.BORDER_DEFAULT),
    "menu": (T.GLASS_STRONG, T.BORDER_SUBTLE),
    "sheet": (T.SHEET, T.BORDER_SUBTLE),
    "raised": (T.SHEET_RAISED, T.BORDER_DEFAULT),
}


def glass(ui, x, y, w, h, r=T.R_PANEL, level="hud", shadow=True):
    """The standard frosted panel.  ``level``: hud, menu, sheet, raised."""
    fill, border = GLASS_LEVELS.get(level, GLASS_LEVELS["hud"])
    r = T.R_PANEL if r is None else r
    surface(ui, x, y, w, h, r, fill, border, T.SHADOW_1 if shadow else None, specular=False)


def sheet(ui, x, y, w, h):
    """A full page / dialog panel."""
    glass(ui, x, y, w, h, T.R_SHEET, "sheet")


def group(ui, x, y, w, h, r=T.R_CARD, hover=0.0, selected=False, fill=None):
    if fill is None:
        fill = T.GROUP_SELECTED if selected else mix_pair(T.GROUP, T.GROUP_HOVER, hover)
    surface(ui, x, y, w, h, r, fill)


def separator(ui, x, y, w, color=None):
    ui.rect(x, y, w, 1.0 / max(1.0, getattr(ui, "scale", 1.0)), color or T.SEPARATOR)


def divider(ui, x, y, w, color=None):
    separator(ui, x, y, w, color)


def vdivider(ui, x, y, h, color=None):
    ui.rect(x, y, 1.0 / max(1.0, getattr(ui, "scale", 1.0)), h, color or T.SEPARATOR)


def scrim(ui, w, h, k=1.0, flat=False):
    """Darken the live scene behind a menu, leaving its colour visible."""
    r, g, b = T.SCRIM_BASE
    if flat:
        ui.rect(0, 0, w, h, (r, g, b, 0.62 * k))
    else:
        hgradient(ui, 0, 0, w, h, (r, g, b, 0.58 * k), (r, g, b, 0.34 * k))


def material_box(ui, x, y, w, h, r=T.R_BUTTON, variant="regular", border=None):
    fills = {"thin": T.MATERIAL_THIN, "regular": T.MATERIAL_REGULAR, "thick": T.MATERIAL_THICK,
             "tinted": T.MATERIAL_TINTED, "field": T.FIELD}
    surface(ui, x, y, w, h, r, fills.get(variant, T.MATERIAL_REGULAR), border)


def focus_ring(ui, x, y, w, h, r=T.R_BUTTON, color=T.FOCUS_RING, width=2.5, inset=0.0):
    ui.box(x - inset, y - inset, w + 2 * inset, h + 2 * inset, r, None, color, bw=width)


def bar(ui, x, y, w, h, hover=0.0, selected=False, enabled=True):
    """A list row: translucent white film, blue when selected."""
    if selected and enabled:
        fill = T.BAR_SELECTED
    else:
        fill = mix(T.BAR, T.BAR_HOVER, hover) if enabled else (1.0, 1.0, 1.0, 0.03)
    ui.box(x, y, w, h, T.R_CARD, fill)


# ═══════════════════════════════════════════════════════════════════════
# Text helpers
# ═══════════════════════════════════════════════════════════════════════

def heading(ui, text, x, y, color=T.TEXT_TERTIARY, size=T.T_CAPTION_2):
    """Section title: quiet, sentence case, above a grouped list."""
    caps(ui, text, x, y, size, color, "semibold")


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
    lead = lead or size + 7
    lines = wrap(ui, text, size, max_w, weight, max_lines)
    for i, line in enumerate(lines):
        ui.text(line, x, y + i * lead, size, color, anchor=anchor, weight=weight)
    return len(lines) * lead


# ═══════════════════════════════════════════════════════════════════════
# Key labels and the hint capsule
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
    """Key label(s): a small translucent rounded chip."""
    if inverse:
        face, ink = (1.0, 1.0, 1.0, 0.92), BLACK
    else:
        face, ink = (1.0, 1.0, 1.0, 0.14), (1.0, 1.0, 1.0, 0.95)
    cx, kh = x, size + 11
    for i, tok in enumerate(_key_tokens(key)):
        if i:
            cx += 4
        tw = ui.text_w(tok, size, weight="semibold")
        kw = max(size + 10, tw + 13)
        if not dry:
            ui.box(cx, y, kw, kh, 6.0, face, (1.0, 1.0, 1.0, 0.10) if not inverse else None)
            ui.text_c(tok, cx + kw / 2, y + kh / 2, size, ink, anchor="center", weight="semibold")
        cx += kw
    return cx - x


def prompt_bar(ui, w, h, items, *, height=T.HINT_BAR_H):
    """A floating, translucent key-hint capsule along the bottom centre."""
    size = 12
    items = list(items)
    if not items:
        return

    def widths(its):
        return [keycaps(ui, 0, 0, k, 11, dry=True) + 7 + ui.text_w(lbl, size, weight="medium")
                for k, lbl in its]

    ws = widths(items)
    avail = w - 160
    while len(items) > 1 and sum(ws) + 18 * (len(items) - 1) + 36 > avail:
        items.pop(0 if len(items) > 2 else 1)
        ws = widths(items)
    total = sum(ws) + 18 * (len(items) - 1) + 36
    ch = 36.0
    x = (w - total) / 2
    y = h - ch - 14.0
    glass(ui, x, y, total, ch, ch / 2, "hud")
    cx = x + 18
    for (k, lbl), iw in zip(items, ws):
        kw = keycaps(ui, cx, y + (ch - 22) / 2, k, 11)
        ui.text_c(lbl, cx + kw + 7, y + ch / 2, size, T.TEXT_SECONDARY, weight="medium")
        cx += iw + 18


# ═══════════════════════════════════════════════════════════════════════
# Buttons, chips, tiles
# ═══════════════════════════════════════════════════════════════════════

def button(ui, rect, label, mouse, *, key, kind="secondary", icon=None, focus=False,
           disabled=False, size=T.T_BODY, hover=None):
    """A capsule button.  ``kind``: primary (blue), secondary (glass), tertiary, destructive.

    Returns the hover amount (0-1).
    """
    x, y, w, h = rect
    radius = min(h / 2, T.R_BUTTON + 4)
    over = inside(rect, *mouse) and not disabled
    if hover is None:
        hover = spring(ui, ("btn", key), float(over), T.SPRING_GENTLE)
    hover = max(0.0, min(1.0, hover))
    lit = 1.0 if (focus and not disabled) else hover
    press = POINTER["down"] and over
    depth = 0.94 if press else 1.0

    if disabled:
        fill, col, border = (1.0, 1.0, 1.0, 0.05), T.TEXT_QUATERNARY, T.BORDER_SUBTLE
    elif kind == "destructive":
        fill, col, border = T.DESTRUCTIVE_FILL, WHITE, (1.0, 1.0, 1.0, 0.18)
        if lit > 0.0:
            fill = (mix(fill[0], WHITE, 0.12 * lit), mix(fill[1], WHITE, 0.10 * lit))
    elif kind == "primary":
        fill, col, border = T.ACCENT_BLUE_FILL, WHITE, (1.0, 1.0, 1.0, 0.16)
        if lit > 0.0:
            fill = (mix(fill[0], WHITE, 0.14 * lit), mix(fill[1], WHITE, 0.10 * lit))
    elif kind == "tertiary":
        fill, col, border = (0.0, 0.0, 0.0, 0.0), mix(T.TEXT_SECONDARY, WHITE, lit), (0.0, 0.0, 0.0, 0.0)
    else:
        fill, col, border = mix_pair(T.BUTTON_FILL, T.BUTTON_HOVER, lit), WHITE, T.BUTTON_BORDER
        if focus:
            border = T.BORDER_EMPHASIZED

    cw, ch = w * depth, h * depth
    px, py = x + (w - cw) / 2, y + (h - ch) / 2
    probe = fill[0] if isinstance(fill, tuple) and len(fill) == 2 and hasattr(fill[0], "__len__") else fill
    if probe[3] > 0.005:
        surface(ui, px, py, cw, ch, radius, fill, border, T.SHADOW_1, specular=True)
    elif focus:
        focus_ring(ui, x, y, w, h, radius, T.FOCUS_RING, 2.0)

    tw = ui.text_w(label, size, weight="semibold")
    isz = size + 3
    total = tw + ((isz + 8) if icon else 0)
    cx = x + (w - total) / 2
    if icon:
        icons.draw(ui, icon, cx + isz / 2, y + h / 2, isz, col, 1.6)
        cx += isz + 8
    ui.text_c(label, cx, y + h / 2, size, col, weight="semibold")
    return hover


def chip(ui, x, y, text, color=T.TEXT_SECONDARY, size=T.T_CAPTION_1, h=24, pad=11, anchor="left",
         fill=None, border=None, icon=None):
    """A small capsule label.  Returns its width."""
    text = _label(text)
    tw = ui.text_w(text, size, weight="semibold")
    iw = (size + 4) if icon else 0
    w = tw + 2 * pad + iw
    if anchor == "right":
        x -= w
    elif anchor == "center":
        x -= w / 2
    fill = fill if fill is not None else (1.0, 1.0, 1.0, 0.10)
    ui.box(x, y, w, h, h / 2, fill, border or (1.0, 1.0, 1.0, 0.10))
    if icon:
        icons.draw(ui, icon, x + pad + size * 0.5, y + h / 2, size + 2, color, 1.3)
    ui.text_c(text, x + pad + iw, y + h / 2, size, color, weight="semibold")
    return w


def icon_tile(ui, x, y, s, icon, tile=T.TILE_GRAY, enabled=True):
    """A rounded app-icon square with a white glyph."""
    k = 1.0 if enabled else 0.4
    top, bot = tile if isinstance(tile, tuple) and len(tile) == 2 and hasattr(tile[0], "__len__") else (tile, tile)
    ui.box(x, y, s, s, s * 0.26, (scale_alpha(top, 1.0), scale_alpha(bot, 1.0)), (1.0, 1.0, 1.0, 0.14))
    icons.draw(ui, icon, x + s / 2, y + s / 2, s * 0.56, (1.0, 1.0, 1.0, k), max(1.4, s / 19.0))


def close_button(ui, x, y, mouse, key="close", size=32):
    """A circular glass close button; fills with white on hover."""
    rect = (x, y, size, size)
    hv = spring(ui, ("closebtn", key), float(inside(rect, *mouse)), T.SPRING_GENTLE)
    ui.box(x, y, size, size, size / 2, mix((1.0, 1.0, 1.0, 0.10), (1.0, 1.0, 1.0, 0.26), hv),
           (1.0, 1.0, 1.0, 0.12))
    icons.draw(ui, "close", x + size / 2, y + size / 2, size * 0.36, (1.0, 1.0, 1.0, 0.92), 1.9)
    return rect


def logo_mark(ui, x, y, s=40.0, t=0.0):
    """The Project GAIA mark: a glass rounded square with a wire-frame globe."""
    ui.box(x, y, s, s, s * 0.27, ((0.16, 0.18, 0.22, 0.95), (0.07, 0.08, 0.10, 0.95)),
           (1.0, 1.0, 1.0, 0.16))
    cx, cy, r = x + s / 2, y + s / 2, s * 0.27
    w = max(1.2, s * 0.035)
    ui.ring(cx, cy, r, w, (1.0, 1.0, 1.0, 0.92))
    ui.line(cx - r, cy, cx + r, cy, w * 0.8, (1.0, 1.0, 1.0, 0.75))
    ui.line(cx, cy - r, cx, cy + r, w * 0.8, (1.0, 1.0, 1.0, 0.75))
    for sgn in (-1, 1):
        dy = sgn * r * 0.52
        hw = math.sqrt(max(0.0, r * r - dy * dy))
        ui.line(cx - hw, cy + dy, cx + hw, cy + dy, w * 0.7, (1.0, 1.0, 1.0, 0.55))
    pts = [(cx + r * 0.46 * math.cos(a), cy + r * math.sin(a)) for a in [i * math.tau / 20 for i in range(20)]]
    ui.outline_poly(pts, w * 0.8, (1.0, 1.0, 1.0, 0.6), True)


def panel_header(ui, x, y, w, title, subtitle=None, icon=None, tile=T.TILE_BLUE):
    """Page title: a large title in sentence case with a quiet subtitle under it."""
    tx = x
    if icon:
        icon_tile(ui, x, y + 1, 42, icon, tile)
        tx = x + 56
    ui.text(title, tx, y, T.T_TITLE_3 - 2, T.TEXT_PRIMARY, weight="semibold")
    if subtitle:
        ui.text(ui.fit(subtitle, T.T_BODY, w - (tx - x)), tx, y + 32, T.T_BODY, T.TEXT_TERTIARY)
    return y + 54


# ═══════════════════════════════════════════════════════════════════════
# Controls
# ═══════════════════════════════════════════════════════════════════════

def toggle_rect(rect):
    x, y, w, h = rect
    return (x + w - TOGGLE_W, y + (h - TOGGLE_H) / 2, TOGGLE_W, TOGGLE_H)


def toggle(ui, rect, on, *, key, focus=False, enabled=True):
    """An iOS switch: a green capsule track with a white knob that slides."""
    x, y, w, h = toggle_rect(rect)
    k = spring(ui, ("switch", key), 1.0 if on else 0.0, T.SPRING_QUICK)
    track = mix((1.0, 1.0, 1.0, 0.16), T.SYS_GREEN, k)
    if not enabled:
        track = (1.0, 1.0, 1.0, 0.07)
    ui.box(x, y, w, h, h / 2, track, (1.0, 1.0, 1.0, 0.10 if enabled else 0.04))
    knob = h - 4
    kx = x + 2 + (w - knob - 4) * k
    ui.box(kx, y + 2, knob, knob, knob / 2, None, None, shadow=6.0, shadow_dy=1.5, shadow_alpha=0.35)
    ui.circle(kx + knob / 2, y + h / 2, knob / 2, (1.0, 1.0, 1.0, 1.0 if enabled else 0.5))
    if focus:
        focus_ring(ui, x - 3, y - 3, w + 6, h + 6, (h + 6) / 2, T.FOCUS_RING, 2.0)


def slider(ui, rect, frac, *, key, hover=0.0, focus=False, enabled=True, color=None):
    """A 4 pt track with a filled left side and a white circular knob."""
    x, y, w, h = rect
    frac = max(0.0, min(1.0, float(frac)))
    cy = y + h / 2
    accent = (color or T.SYS_BLUE) if enabled else T.TEXT_QUATERNARY
    x0, x1 = x + 10, x + w - 10
    tw = max(1.0, x1 - x0)
    k = ui.anim(("slider", key), frac, 30.0)
    ui.box(x0, cy - 2, tw, 4, 2, (1.0, 1.0, 1.0, 0.16))
    ui.box(x0, cy - 2, max(4.0, tw * k), 4, 2, accent)
    kx = x0 + tw * k
    ui.box(kx - 9, cy - 9, 18, 18, 9, None, None, shadow=8.0, shadow_dy=2.0, shadow_alpha=0.35)
    ui.circle(kx, cy, 9, WHITE)
    if focus:
        ui.ring(kx, cy, 13, 2.0, T.FOCUS_RING)


def segmented(ui, rect, labels, index, mouse, *, key, focus=False, enabled=True, size=T.T_SMALL):
    """A capsule segmented control with a sliding white selection.  Returns option rects."""
    x, y, w, h = rect
    h = min(h, 34.0)
    y = rect[1] + (rect[3] - h) / 2
    widths = [ui.text_w(s, size, weight="medium") + 26 for s in labels]
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
    ui.box(left, y, total, h, h / 2, (1.0, 1.0, 1.0, 0.08), (1.0, 1.0, 1.0, 0.07))
    if 0 <= index < len(rects):
        sel = rects[index]
        # The selection pill slides between options instead of jumping.
        sx = ui.anim(("segx", key), sel[0] + 2, 34.0)
        sw = ui.anim(("segw", key), sel[2] - 4, 34.0)
        ui.box(sx, y + 2, sw, h - 4, (h - 4) / 2,
               (1.0, 1.0, 1.0, 0.94), None, shadow=6.0, shadow_dy=2.0, shadow_alpha=0.25)
    for i, (label, r) in enumerate(zip(labels, rects)):
        sel = i == index
        hov = spring(ui, ("seg", key, i), float(inside(r, *mouse) and enabled and not sel), T.SPRING_GENTLE)
        if sel:
            col = (0.06, 0.06, 0.07, 0.95) if enabled else T.TEXT_QUATERNARY
        else:
            col = mix(T.TEXT_SECONDARY, WHITE, hov) if enabled else T.TEXT_QUATERNARY
        ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2, size, col, anchor="center", weight="semibold")
    if focus:
        focus_ring(ui, left, y, total, h, h / 2, T.FOCUS_RING, 2.0, inset=2)
    return rects


def stepper(ui, rect, label, mouse, *, key, focus=False, enabled=True, width=210.0, size=T.T_SMALL):
    """``−  VALUE  +`` in a capsule.  Returns the two arrow rects."""
    x, y, w, h = rect
    width = min(width, w)
    h = min(h, 32.0)
    y = rect[1] + (rect[3] - h) / 2
    left = x + w - width
    arrows = []
    for i in range(2):
        r = (left + (0 if i == 0 else width - h), y, h, h)
        arrows.append(r)
        hov = spring(ui, ("stp", key, i), float(inside(r, *mouse) and enabled), T.SPRING_GENTLE)
        ui.box(r[0] + 3, r[1] + 3, r[2] - 6, r[3] - 6, (r[3] - 6) / 2,
               (1.0, 1.0, 1.0, 0.06 + 0.14 * hov))
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        d = 1 if i else -1
        col = WHITE if enabled else T.TEXT_QUATERNARY
        ui.line(cx - d * 5, cy, cx + d * 5, cy, 2.0, col)
        if i:
            ui.line(cx, cy - 5, cx, cy + 5, 2.0, col)
    ui.text_c(ui.fit(label, size, width - 2 * h), left + width / 2, y + h / 2, size,
              T.TEXT_PRIMARY if enabled else T.TEXT_QUATERNARY, anchor="center", weight="semibold")
    if focus:
        focus_ring(ui, left, y, width, h, h / 2, T.FOCUS_RING, 2.0, inset=2)
    return arrows


def search_field(ui, x, y, w, h, text, placeholder, active=True, now=0.0):
    """A capsule search field with a magnifier, a clear affordance and a blue caret."""
    ui.box(x, y, w, h, min(h / 2, 14), T.FIELD, (1.0, 1.0, 1.0, 0.10))
    cy = y + h / 2
    icon = (1.0, 1.0, 1.0, 0.45 if active else 0.3)
    ui.ring(x + 18, cy - 1, 5.0, 1.7, icon)
    ui.line(x + 22, cy + 3, x + 26, cy + 7, 1.8, icon)
    cx = x + 36
    if text:
        shown = ui.fit(text, T.T_BODY, w - 70)
        cx += ui.text_c(shown, cx, cy, T.T_BODY, T.TEXT_PRIMARY) + 1
    else:
        ui.text_c(placeholder, cx + (10 if active else 0), cy, T.T_BODY, T.TEXT_TERTIARY)
    if active and int(now * 2) % 2 == 0:
        ui.box(cx, cy - 8, 2, 16, 1.0, T.SYS_BLUE)
    if text:
        ui.circle(x + w - 20, cy, 8.0, (1.0, 1.0, 1.0, 0.18))
        c = (1.0, 1.0, 1.0, 0.72)
        ui.line(x + w - 23, cy - 3, x + w - 17, cy + 3, 1.6, c)
        ui.line(x + w - 17, cy - 3, x + w - 23, cy + 3, 1.6, c)
    return cx


def scrollbar(ui, x, y, h, view, content, offset):
    if content <= view + 1:
        return
    th = max(28.0, h * view / content)
    ty = y + (h - th) * (offset / max(1.0, content - view))
    ui.box(x, y, 4, h, 2, (1.0, 1.0, 1.0, 0.07))
    ui.box(x, ty, 4, th, 2, (1.0, 1.0, 1.0, 0.42))


def progress(ui, x, y, w, h, frac, color=T.ACCENT_BLUE, shimmer=None):
    """A thin capsule progress bar."""
    frac = max(0.0, min(1.0, float(frac)))
    ui.box(x, y, w, h, h / 2, (1.0, 1.0, 1.0, 0.15))
    if frac <= 0.0:
        return
    fw = max(h, w * frac)
    ui.box(x, y, fw, h, h / 2, color)
    if shimmer is not None and fw > 24:
        sw = min(46.0, fw * 0.5)
        sx = x + (fw - sw) * (shimmer % 1.0)
        ui.box(sx, y, sw, h, h / 2, (1.0, 1.0, 1.0, 0.30))


def activity(ui, cx, cy, r, t, color=T.TEXT_SECONDARY):
    """Eight dots on a ring, chasing each other."""
    for i in range(8):
        a = i * math.tau / 8.0
        lead = (t * 1.25 - i / 8.0) % 1.0
        al = 0.14 + 0.86 * (1.0 - lead)
        rr = 1.5 if i % 2 else 1.8
        ui.circle(cx + math.sin(a) * r, cy - math.cos(a) * r, rr,
                  (color[0], color[1], color[2], color[3] * al))


def check_circle(ui, cx, cy, r, color=T.SYS_BLUE):
    """A finished step: a filled circle with a white tick."""
    ui.circle(cx, cy, r, color)
    k = r / 10.0
    ui.line(cx - 3.6 * k, cy + 0.2 * k, cx - 1.0 * k, cy + 2.8 * k, max(1.5, 1.9 * k), WHITE)
    ui.line(cx - 1.0 * k, cy + 2.8 * k, cx + 4.0 * k, cy - 3.2 * k, max(1.5, 1.9 * k), WHITE)


def toast(ui, cx, y, text, ok=True, size=T.T_BODY):
    """A floating capsule notification with a coloured dot."""
    tw = ui.text_w(text, size, weight="semibold")
    w, h = tw + 62, 42.0
    x = cx - w / 2
    glass(ui, x, y, w, h, h / 2, "hud")
    ui.circle(x + 20, y + h / 2, 4.5, T.SYS_GREEN if ok else T.SYS_ORANGE)
    ui.text_c(text, x + 34, y + h / 2, size, WHITE, weight="semibold")


# ═══════════════════════════════════════════════════════════════════════
# Cards & rows
# ═══════════════════════════════════════════════════════════════════════

def card(ui, x, y, w, h, *, selected=False, hovered=False, focused=False, enabled=True,
         elevation=0, border=None, r=None, tint=None):
    """A tappable card: translucent film, white lift on hover, blue when selected."""
    r = T.R_CARD if r is None else r
    hv = float(hovered)
    if not enabled:
        fill = (1.0, 1.0, 1.0, 0.035)
    elif selected:
        fill = ((T.SYS_BLUE[0] * 0.55 + 0.12, T.SYS_BLUE[1] * 0.55 + 0.14, T.SYS_BLUE[2] * 0.55 + 0.20, 0.92),
                (0.06, 0.28, 0.62, 0.94))
    else:
        fill = mix_pair(T.GROUP, T.GROUP_HOVER, hv)
    surface(ui, x, y, w, h, r, fill, border if border is not None else (1.0, 1.0, 1.0, 0.10),
            T.SHADOW_1 if elevation else None)
    if focused and enabled:
        focus_ring(ui, x, y, w, h, r, T.FOCUS_RING, 2.5, inset=1.0)


def kv_row(ui, x, y, w, h, label, value, icon=None):
    """A key / value row: label on the left, the value in the secondary tone on the right."""
    lx = x
    if icon:
        icons.draw(ui, icon, x + 12, y + h / 2, 18, T.TEXT_TERTIARY, 1.5)
        lx = x + 32
    ui.text_c(label, lx, y + h / 2, T.T_BODY, T.TEXT_SECONDARY)
    ui.text_c(ui.fit(str(value), T.T_BODY, w - (lx - x) - ui.text_w(label, T.T_BODY) - 20), x + w, y + h / 2,
              T.T_BODY, T.TEXT_PRIMARY, anchor="right", weight="medium")


# Backward compatibility: the design tokens used to be importable from here.
from .theme import *  # noqa: E402,F401,F403
