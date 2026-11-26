"""Clipped map strokes and small, font-independent place/navigation symbols."""
from __future__ import annotations

import math

import numpy as np

from ..core.config import EARTH_RADIUS
from ..world.places import CATEGORY_COLORS

BLUE = (0.08, 0.52, 1., 1.)
RED = (1., 0.31, 0.34, 1.)
WHITE = (1., 1., 1., 1.)


def stroke(ui, points, width, color, clip, valid=None, radius=0.):
    """Round-capped, batched polyline; all fragments obey the map's rounded clip."""
    points = np.asarray(points, np.float32).reshape(-1, 2)
    if len(points) < 2: return
    a, b = points[:-1], points[1:]
    d = b - a
    length = np.linalg.norm(d, axis=1)
    x, y, w, h = clip
    keep = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (length > .05)
            & (np.maximum(a[:, 0], b[:, 0]) >= x - width)
            & (np.minimum(a[:, 0], b[:, 0]) <= x + w + width)
            & (np.maximum(a[:, 1], b[:, 1]) >= y - width)
            & (np.minimum(a[:, 1], b[:, 1]) <= y + h + width))
    if valid is not None: keep &= np.asarray(valid, bool)
    if not keep.any(): return
    a, b, d, length = a[keep], b[keep], d[keep], length[keep]
    half = width * .5
    n = np.stack((-d[:, 1], d[:, 0]), axis=1) / length[:, None] * half
    quads = np.stack((a + n, a - n, b - n, a + n, b - n, b + n), axis=1).reshape(-1, 2)
    centres = np.concatenate((a, b))
    centres = centres[((centres >= (x - width, y - width))
                       & (centres <= (x + w + width, y + h + width))).all(axis=1)]
    angle = np.arange(9) * (2. * math.pi / 8.)
    ring = np.stack((np.cos(angle), np.sin(angle)), axis=1) * half
    fans = np.stack((np.broadcast_to(centres[:, None], (len(centres), 8, 2)),
                     centres[:, None] + ring[None, :-1], centres[:, None] + ring[None, 1:]), axis=2)
    verts = np.concatenate((quads, fans.reshape(-1, 2))).astype(np.float32, copy=False)
    ui.triangles(verts, np.broadcast_to(color, (len(verts), 4)), clip=clip, clip_radius=radius)


def route_stroke(ui, points, clip, valid=None, radius=0.):
    # Wide, faint halo: the route glows softly against the frosted map.
    stroke(ui, points, 15., (0.10, 0.48, 1., .14), clip, valid, radius)
    stroke(ui, points, 8., (0.015, 0.08, 0.19, .98), clip, valid, radius)
    stroke(ui, points, 5.5, (0.37, 0.72, 1., 1.), clip, valid, radius)
    stroke(ui, points, 3., BLUE, clip, valid, radius)


def world_latlon(points, proj):
    """Vectorised equivalent of LocalProjection.to_latlon for map batches."""
    points = np.asarray(points)
    lat = proj.lat0 + np.degrees(points[:, 1] / EARTH_RADIUS)
    lon = proj.lon0 + np.degrees(points[:, 0] / (EARTH_RADIUS * proj.cos_lat0))
    return lat, lon


def project_world(flat, points, proj):
    lat, lon = world_latlon(points, proj)
    s = np.sin(np.radians(np.clip(lat, -85.0511, 85.0511)))
    wx, wy = (lon + 180.) / 360., .5 - np.log((1. + s) / (1. - s)) / (4. * math.pi)
    return np.stack(((wx - flat.cx) * flat.ppw() + flat.W * .5,
                     (wy - flat.cy) * flat.ppw() + flat.H * .5), axis=1).astype(np.float32)


def place_symbol(ui, cx, cy, category, size=12., mono=None):
    """A category glyph in a disc.  ``mono`` overrides the category colour."""
    color = mono if mono is not None else CATEGORY_COLORS.get(category, CATEGORY_COLORS["landmark"])
    ui.circle(cx, cy, size, (0.01, 0.025, 0.05, .8))
    ui.circle(cx, cy, size - 1.3, color)
    s = size * .44
    if category == "coffee":
        ui.box(cx - s, cy - s * .6, s * 1.6, s * 1.4, 1.3, None, WHITE, bw=1.3)
        ui.ring(cx + s * .7, cy, s * .45, 1.2, WHITE)
    elif category == "food":
        ui.line(cx - s * .6, cy - s, cx - s * .6, cy + s, 1.3, WHITE)
        for dx in (-.3, 0., .3):
            ui.line(cx - s * .6 + dx * s, cy - s, cx - s * .6 + dx * s, cy - s * .2, 1., WHITE)
        ui.line(cx + s * .6, cy - s, cx + s * .6, cy + s, 1.7, WHITE)
    elif category == "health":
        ui.line(cx - s, cy, cx + s, cy, 2., WHITE)
        ui.line(cx, cy - s, cx, cy + s, 2., WHITE)
    elif category == "shopping":
        ui.box(cx - s, cy - s * .4, s * 2, s * 1.4, 1., None, WHITE, bw=1.3)
        ui.ring(cx, cy - s * .35, s * .55, 1.2, WHITE)
    elif category == "park":
        ui.tri((cx, cy - s * 1.1), (cx - s, cy + s * .55), (cx + s, cy + s * .55), WHITE)
        ui.line(cx, cy + s * .3, cx, cy + s * 1.1, 1.5, WHITE)
    elif category == "parking":
        ui.text_c("P", cx, cy, size * 1.25, WHITE, anchor="center", weight="bold")
    elif category == "fuel":
        ui.box(cx - s, cy - s, s * 1.3, s * 2, 1., None, WHITE, bw=1.2)
        ui.line(cx + s * .3, cy - s * .5, cx + s, cy, 1.2, WHITE)
        ui.line(cx + s, cy, cx + s, cy + s, 1.2, WHITE)
    elif category == "transit":
        ui.box(cx - s, cy - s, s * 2, s * 1.8, 2., None, WHITE, bw=1.3)
        ui.line(cx - s * .6, cy, cx + s * .6, cy, 1.1, WHITE)
        for dx in (-.55, .55): ui.circle(cx + dx * s, cy + s * .75, 1., WHITE)
    elif category == "hotel":
        ui.line(cx - s, cy - s * .7, cx - s, cy + s, 1.5, WHITE)
        ui.line(cx - s, cy + s * .4, cx + s, cy + s * .4, 1.8, WHITE)
        ui.box(cx - s * .5, cy - s * .2, s * 1.5, s * .6, 1., WHITE)
    else:
        ui.box(cx - s, cy - s * .3, s * 2, s * 1.25, 1., None, WHITE, bw=1.2)
        ui.tri((cx, cy - s), (cx - s * 1.1, cy - s * .2), (cx + s * 1.1, cy - s * .2), WHITE)


def destination_pin(ui, cx, cy, size=10.):
    ui.tri((cx, cy + size * 1.5), (cx - size * .65, cy), (cx + size * .65, cy), RED)
    ui.circle(cx, cy, size + 1.2, (0.03, 0.015, 0.03, .9))
    ui.circle(cx, cy, size, RED)
    ui.circle(cx, cy, size * .35, WHITE)


def maneuver_symbol(ui, cx, cy, kind, size=17.):
    if kind == "arrive":
        ui.line(cx - size * .4, cy - size, cx - size * .4, cy + size, 2.5, WHITE)
        ui.tri((cx - size * .4, cy - size), (cx + size * .7, cy - size * .5),
               (cx - size * .4, cy), WHITE)
        return
    direction = -1 if kind == 1 else 1
    if kind in (1, -1):
        ui.line(cx - direction * size * .35, cy + size * .7,
                cx - direction * size * .35, cy - size * .2, 3., WHITE)
        ui.line(cx - direction * size * .35, cy - size * .2,
                cx + direction * size * .65, cy - size * .2, 3., WHITE)
        ui.tri((cx + direction * size, cy - size * .2),
               (cx + direction * size * .3, cy - size * .8),
               (cx + direction * size * .3, cy + size * .4), WHITE)
    elif kind == 2:
        ui.line(cx + size * .45, cy + size * .8, cx + size * .45, cy - size * .4, 3., WHITE)
        ui.line(cx + size * .45, cy - size * .4, cx - size * .45, cy - size * .4, 3., WHITE)
        ui.line(cx - size * .45, cy - size * .4, cx - size * .45, cy + size * .2, 3., WHITE)
        ui.tri((cx - size * .45, cy + size * .8), (cx - size, cy), (cx + size * .1, cy), WHITE)
    else:
        ui.line(cx, cy + size * .8, cx, cy - size * .3, 3., WHITE)
        ui.tri((cx, cy - size), (cx - size * .6, cy - size * .15),
               (cx + size * .6, cy - size * .15), WHITE)
