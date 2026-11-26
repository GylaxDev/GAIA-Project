"""Rasterise Lucide strokes into an alpha mask at an exact device-pixel size.

Pure numpy (no GL), so it can be unit-tested and previewed on its own.  Every
stroke is a polyline with round caps and round joins; coverage is the signed
distance to the nearest segment, one device pixel soft, which is exactly how the
overlay's own line and box primitives are anti-aliased.
"""
from __future__ import annotations

import math

import numpy as np

from .lucide_data import STROKES

GRID = 24.0


def has(name: str) -> bool:
    return name in STROKES


def stroke_width(size: float) -> float:
    """Stroke width in *logical* pixels for an icon ``size`` logical pixels tall.

    Lucide is designed at 2 / 24 of the icon size; below ~20 px that gets too
    hairline, so small icons keep a floor (Lucide's "absolute stroke width"
    trick) and large ones scale up gently so a 48 px icon is not a wire.
    """
    return max(1.5, min(3.0, 2.0 * size / GRID * (0.94 if size >= 20 else 1.0)))


def _rotate(pts, angle):
    if not angle:
        return pts
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    return [(12.0 + (x - 12.0) * c - (y - 12.0) * s, 12.0 + (x - 12.0) * s + (y - 12.0) * c) for x, y in pts]


def rasterize(name: str, px: int, stroke_px: float, angle: float = 0.0) -> np.ndarray:
    """Return a ``(px, px)`` uint8 coverage mask for icon ``name``."""
    px = max(4, int(px))
    k = px / GRID
    hw = max(0.5, stroke_px) / 2.0
    ys, xs = np.mgrid[0:px, 0:px]
    cx, cy = xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5
    dist = np.full((px, px), 1e9, np.float32)
    for closed, pts in STROKES[name]:
        P = [(x * k, y * k) for x, y in _rotate(list(pts), angle)]
        if closed and P[0] != P[-1]:
            P = P + [P[0]]
        if len(P) == 1:
            P = P * 2
        for (x0, y0), (x1, y1) in zip(P, P[1:]):
            dx, dy = x1 - x0, y1 - y0
            ln2 = dx * dx + dy * dy
            # only touch pixels near the segment's bounding box
            lo_x = max(0, int(math.floor(min(x0, x1) - hw - 1.5)))
            hi_x = min(px, int(math.ceil(max(x0, x1) + hw + 1.5)))
            lo_y = max(0, int(math.floor(min(y0, y1) - hw - 1.5)))
            hi_y = min(px, int(math.ceil(max(y0, y1) + hw + 1.5)))
            if lo_x >= hi_x or lo_y >= hi_y:
                continue
            gx, gy = cx[lo_y:hi_y, lo_x:hi_x], cy[lo_y:hi_y, lo_x:hi_x]
            if ln2 < 1e-9:
                d = np.hypot(gx - x0, gy - y0)
            else:
                t = np.clip(((gx - x0) * dx + (gy - y0) * dy) / ln2, 0.0, 1.0)
                d = np.hypot(gx - (x0 + t * dx), gy - (y0 + t * dy))
            region = dist[lo_y:hi_y, lo_x:hi_x]
            np.minimum(region, d, out=region)
    cov = np.clip(hw + 0.5 - dist, 0.0, 1.0)
    return (cov * 255.0 + 0.5).astype(np.uint8)
