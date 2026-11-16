"""Small vector glyphs for the menus, drawn with the overlay's line / circle primitives.

Every icon is laid out on a 24 x 24 grid and scaled to ``s`` logical pixels, so they
stay crisp on any display scale and take their colour from the theme.
"""
from __future__ import annotations

import math


def _p(cx, cy, k, x, y):
    return cx + (x - 12.0) * k, cy + (y - 12.0) * k


def _poly(ui, cx, cy, k, pts, col, w, closed=False):
    P = [_p(cx, cy, k, *q) for q in pts]
    for a, b in zip(P, P[1:] + (P[:1] if closed else [])):
        ui.line(a[0], a[1], b[0], b[1], w, col)


def _arc(ui, cx, cy, k, ox, oy, r, a0, a1, col, w, n=14):
    pts = [(ox + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
            oy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]
    _poly(ui, cx, cy, k, pts, col, w)


def _ring(ui, cx, cy, k, ox, oy, r, col, w):
    x, y = _p(cx, cy, k, ox, oy)
    ui.ring(x, y, r * k, w, col)


def _dot(ui, cx, cy, k, ox, oy, r, col):
    x, y = _p(cx, cy, k, ox, oy)
    ui.circle(x, y, r * k, col)


def _box(ui, cx, cy, k, x0, y0, x1, y1, r, col, w):
    a, b = _p(cx, cy, k, x0, y0), _p(cx, cy, k, x1, y1)
    ui.box(a[0], a[1], b[0] - a[0], b[1] - a[1], r * k, None, col, bw=w)


def _fillbox(ui, cx, cy, k, x0, y0, x1, y1, r, col):
    a, b = _p(cx, cy, k, x0, y0), _p(cx, cy, k, x1, y1)
    ui.box(a[0], a[1], b[0] - a[0], b[1] - a[1], r * k, col)


def play(ui, cx, cy, k, c, w):
    ui.poly([_p(cx, cy, k, 8, 5), _p(cx, cy, k, 19, 12), _p(cx, cy, k, 8, 19)], c)


def car(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(3, 15), (3, 12.5), (6, 11.5), (9, 7.5), (16, 7.5), (19, 11.5), (21, 12.5), (21, 15)],
          c, w, closed=True)
    _poly(ui, cx, cy, k, [(9.5, 11.5), (10.5, 8.8), (15.2, 8.8), (17, 11.5)], c, w * .8, closed=True)
    _dot(ui, cx, cy, k, 7.5, 16, 2.1, c)
    _dot(ui, cx, cy, k, 16.5, 16, 2.1, c)


def globe(ui, cx, cy, k, c, w):
    _ring(ui, cx, cy, k, 12, 12, 8.5, c, w)
    _poly(ui, cx, cy, k, [(3.5, 12), (20.5, 12)], c, w * .8)
    _poly(ui, cx, cy, k, [(12, 3.5), (12, 20.5)], c, w * .8)
    ex = [(12 + 4.6 * math.cos(math.radians(a)), 12 + 8.5 * math.sin(math.radians(a))) for a in range(-90, 271, 20)]
    _poly(ui, cx, cy, k, ex, c, w * .8, closed=True)


def pin(ui, cx, cy, k, c, w):
    pts = [(12 + 6.5 * math.cos(math.radians(a)), 9.5 + 6.5 * math.sin(math.radians(a))) for a in range(150, 391, 15)]
    _poly(ui, cx, cy, k, pts + [(12, 21)] + [pts[0]], c, w)
    _ring(ui, cx, cy, k, 12, 9.5, 2.2, c, w)


def sliders(ui, cx, cy, k, c, w):
    for y, x in ((6.5, 8), (12, 16), (17.5, 10)):
        _poly(ui, cx, cy, k, [(3.5, y), (20.5, y)], c, w * .8)
        _dot(ui, cx, cy, k, x, y, 2.4, c)


def keyboard(ui, cx, cy, k, c, w):
    _box(ui, cx, cy, k, 2.5, 6, 21.5, 18, 2.5, c, w)
    for x in (6, 9.5, 13, 16.5):
        _dot(ui, cx, cy, k, x, 10, 0.9, c)
    for x in (7.5, 11, 14.5):
        _dot(ui, cx, cy, k, x, 13.5, 0.9, c)
    _poly(ui, cx, cy, k, [(8, 16), (16, 16)], c, w * .8)


def info(ui, cx, cy, k, c, w):
    _ring(ui, cx, cy, k, 12, 12, 9, c, w)
    _poly(ui, cx, cy, k, [(12, 11), (12, 17)], c, w)
    _dot(ui, cx, cy, k, 12, 7.8, 1.2, c)


def power(ui, cx, cy, k, c, w):
    _arc(ui, cx, cy, k, 12, 12.5, 7.5, -60, 240, c, w, n=18)
    _poly(ui, cx, cy, k, [(12, 3.5), (12, 11.5)], c, w)


def monitor(ui, cx, cy, k, c, w):
    _box(ui, cx, cy, k, 3, 4.5, 21, 16, 2, c, w)
    _poly(ui, cx, cy, k, [(9, 20), (15, 20)], c, w)
    _poly(ui, cx, cy, k, [(12, 16), (12, 20)], c, w)


def sparkle(ui, cx, cy, k, c, w):
    ui.poly([_p(cx, cy, k, 12, 3), _p(cx, cy, k, 14, 10), _p(cx, cy, k, 21, 12),
             _p(cx, cy, k, 14, 14), _p(cx, cy, k, 12, 21), _p(cx, cy, k, 10, 14),
             _p(cx, cy, k, 3, 12), _p(cx, cy, k, 10, 10)], c)


def camera(ui, cx, cy, k, c, w):
    _box(ui, cx, cy, k, 2.5, 7, 21.5, 19, 2.5, c, w)
    _poly(ui, cx, cy, k, [(8, 7), (9.5, 4.5), (14.5, 4.5), (16, 7)], c, w)
    _ring(ui, cx, cy, k, 12, 13, 3.6, c, w)


def clock(ui, cx, cy, k, c, w):
    _ring(ui, cx, cy, k, 12, 12, 9, c, w)
    _poly(ui, cx, cy, k, [(12, 6.5), (12, 12), (16, 14.5)], c, w)


def crowd(ui, cx, cy, k, c, w):
    _ring(ui, cx, cy, k, 9, 8, 3, c, w)
    _arc(ui, cx, cy, k, 9, 20, 6.5, 195, 345, c, w)
    _ring(ui, cx, cy, k, 17, 9, 2.4, c, w * .9)
    _arc(ui, cx, cy, k, 17, 20, 5, 205, 335, c, w * .9)


def speaker(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(4, 9.5), (8, 9.5), (13, 5), (13, 19), (8, 14.5), (4, 14.5)], c, w, closed=True)
    _arc(ui, cx, cy, k, 13, 12, 4.5, -45, 45, c, w)
    _arc(ui, cx, cy, k, 13, 12, 8, -45, 45, c, w)


def layout(ui, cx, cy, k, c, w):
    _box(ui, cx, cy, k, 3, 4, 21, 20, 2.5, c, w)
    _poly(ui, cx, cy, k, [(3, 9), (21, 9)], c, w * .8)
    _poly(ui, cx, cy, k, [(9.5, 9), (9.5, 20)], c, w * .8)


def terminal(ui, cx, cy, k, c, w):
    _box(ui, cx, cy, k, 3, 4.5, 21, 19.5, 2.5, c, w)
    _poly(ui, cx, cy, k, [(7, 9.5), (10.5, 12), (7, 14.5)], c, w)
    _poly(ui, cx, cy, k, [(12.5, 15), (17, 15)], c, w)


def chevron_l(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(14.5, 6), (8.5, 12), (14.5, 18)], c, w)


def chevron_r(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(9.5, 6), (15.5, 12), (9.5, 18)], c, w)


def chevron_d(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(6, 9.5), (12, 15.5), (18, 9.5)], c, w)


def close(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(6.5, 6.5), (17.5, 17.5)], c, w)
    _poly(ui, cx, cy, k, [(6.5, 17.5), (17.5, 6.5)], c, w)


def check(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(5.5, 12.5), (10, 17), (18.5, 7.5)], c, w * 1.1)


def lock(ui, cx, cy, k, c, w):
    _box(ui, cx, cy, k, 5.5, 10.5, 18.5, 20, 2.2, c, w)
    _arc(ui, cx, cy, k, 12, 10.5, 4.5, 180, 360, c, w)
    _dot(ui, cx, cy, k, 12, 15, 1.3, c)


def reset(ui, cx, cy, k, c, w):
    _arc(ui, cx, cy, k, 12, 12, 7.5, -40, 250, c, w, n=18)
    ui.poly([_p(cx, cy, k, 6.6, 2.8), _p(cx, cy, k, 13.2, 3.4), _p(cx, cy, k, 8.8, 8.3)], c)


def drop(ui, cx, cy, k, c, w):
    pts = [(12 + 6 * math.cos(math.radians(a)), 14.5 + 6 * math.sin(math.radians(a))) for a in range(-30, 211, 15)]
    _poly(ui, cx, cy, k, [(12, 3.5)] + pts + [(12, 3.5)], c, w)


def trash(ui, cx, cy, k, c, w):
    _poly(ui, cx, cy, k, [(5, 7), (19, 7)], c, w)
    _poly(ui, cx, cy, k, [(9.5, 7), (9.5, 4.5), (14.5, 4.5), (14.5, 7)], c, w)
    _poly(ui, cx, cy, k, [(6.5, 7), (7.5, 20), (16.5, 20), (17.5, 7)], c, w)


def bolt(ui, cx, cy, k, c, w):
    ui.poly([_p(cx, cy, k, 13.5, 3), _p(cx, cy, k, 6, 13.5), _p(cx, cy, k, 11.5, 13.5),
             _p(cx, cy, k, 10, 21), _p(cx, cy, k, 18, 10), _p(cx, cy, k, 12.5, 10)], c)


def cloud(ui, cx, cy, k, c, w):
    _arc(ui, cx, cy, k, 8.5, 14.5, 4, 90, 270, c, w)
    _arc(ui, cx, cy, k, 12.5, 11, 5, 180, 360, c, w)
    _arc(ui, cx, cy, k, 16.5, 14.5, 4, -90, 90, c, w)
    _poly(ui, cx, cy, k, [(8.5, 18.5), (16.5, 18.5)], c, w)


def route(ui, cx, cy, k, c, w):
    _dot(ui, cx, cy, k, 6, 18, 2.2, c)
    _ring(ui, cx, cy, k, 18, 6, 2.4, c, w)
    _poly(ui, cx, cy, k, [(6, 15.6), (6, 10), (11, 10), (13, 14), (18, 14), (18, 9)], c, w * .8)


ICONS = {n: f for n, f in globals().items()
         if callable(f) and not n.startswith("_") and f.__module__ == __name__}


def draw(ui, name, cx, cy, s, color, width=None):
    """Draw icon ``name`` centred on (cx, cy), ``s`` logical pixels tall."""
    fn = ICONS.get(name)
    if fn is None:
        return
    k = s / 24.0
    fn(ui, cx, cy, k, color, width if width is not None else max(1.2, 1.7 * k))
