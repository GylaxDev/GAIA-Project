"""Times Square dressing: dense LED billboards, blade signs, marquees, red steps.

Generated per tile in local coordinates, in the same 17-float layout as urban.py.
Only tiles near Times Square pay for it. Screens pick a layer from the drop-in
creatives in ``assets/billboards``; the neon shader falls back to procedural LED
content when that folder is empty or cannot be decoded.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

from .urban import (_Sink, _frontages, _street_points, _ground_z, _hash, _quad_on_edge, _ring,
                    NEON, DARK, METAL)

CENTER = (40.7580, -73.9855)          # the bowtie
STEPS = (40.75868, -73.98508)         # red steps at the north end
CORE_R, OUTER_R = 160.0, 520.0
GRID_BEARING = math.radians(29.0)     # Manhattan grid, east of north
RED = np.array([.80, .05, .07], np.float32)
BILL = 4.0                            # neon style.y: LED billboard
BLADE = 1.0                           # neon style.y: vertical blade sign
BAND = 3.0                            # neon style.y: ticker band

# saturated, advertising-like tints (screens also invert/mix them in the shader)
TINTS = np.array([(1.0, .25, .30), (.20, .55, 1.0), (1.0, .85, .15), (.95, .95, 1.0),
                  (.30, 1.0, .60), (1.0, .45, .90), (1.0, .55, .10), (.45, .30, 1.0)], np.float32)


_WALK = frozenset(("footway", "pedestrian", "steps", "path", "cycleway", "bridleway", "track"))


def _exposed_edges(b, i, tree, geoms):
    """Outer wall segments (>= 8.5 m) not buried against another building."""
    ring = _ring(b.footprint)
    if ring is None: return []
    if not b.footprint.exterior.is_ccw: ring = ring[::-1]
    out = []
    for a, c in zip(ring, np.roll(ring, -1, axis=0)):
        length = float(np.linalg.norm(c - a))
        if length < 8.5: continue
        t = (c - a) / length
        n = np.array([t[1], -t[0]])
        covered = 0
        for f in (.25, .5, .75):
            pt = Point(*(a + (c - a) * f + n * .6))
            if any(j != i and geoms[j].contains(pt) for j in tree.query(pt)): covered += 1
        if covered <= 1:
            out.append((a, c, t, n, length))
    out.sort(key=lambda e: -e[4])
    return out


def _weight(x, y, cx, cy):
    d = math.hypot(x - cx, y - cy)
    return 1.0 if d <= CORE_R else max(0.0, 1.0 - (d - CORE_R) / (OUTER_R - CORE_R))


def _screen(glow, solid, a, t, n, s0, s1, z0, z1, seed, tint, lift=0.):
    """Framed LED panel on a facade edge.

    The glass floats 12 cm in front of its frame (and ``lift`` further), so the depth
    buffer never has to separate near-coplanar surfaces at street-viewing distances.
    """
    w, h = s1 - s0, z1 - z0
    solid.edge_box(a, t, n, s0 - .35, s1 + .35, .04, .30, z0 - .35, z1 + .35, DARK)
    _quad_on_edge(glow, a, t, n, s0, s1, z0, z1, tint, 8.0, (seed, BILL, w / max(h, 1.)),
                  out=.42 + lift, emissive=1.0)


def _claimed(placed, a, t, length):
    """True if this wall run overlaps one already carrying screens.

    OSM often maps a building plus its parts, or flush neighbours, with the same wall;
    each would otherwise add its own panels in the identical plane and they z-fight.
    """
    run = LineString([a, a + t * length])
    zone = run.buffer(1.6, cap_style=2)
    for other in placed:
        if zone.intersects(other) and zone.intersection(other).area > 3.0 * 1.6:
            return True
    placed.append(zone)
    return False


def _blade(glow, solid, a, t, n, s, z0, z1, seed, tint):
    """Vertical sign standing out from the facade, lit on both faces."""
    out0, out1 = .5, 2.9
    solid.edge_box(a, t, n, s - .18, s + .18, out0 - .1, out1 + .1, z0 - .3, z1 + .3, DARK)
    for side, flip in ((s + .20, False), (s - .20, True)):
        p = [np.r_[a + t * side + n * o, z] for z in (z0, z1) for o in (out0, out1)]
        q = [p[0], p[1], p[3], p[2]]
        if flip: q = q[::-1]
        glow.quad(*q, tint, 8.0, (seed, BLADE, 0.), uv=((0, 0), (1, 0), (1, 1), (0, 1)),
                  emissive=1.0)


def _band(glow, a, t, n, s0, s1, z0, z1, seed, tint):
    _quad_on_edge(glow, a, t, n, s0, s1, z0, z1, tint, 8.0, (seed, BAND, 0.), out=.45,
                  emissive=1.0)


def build_times_square(buildings, proj, bbox, hf=None, paths=None):
    """Return (emissive mesh, solid mesh) as (vertices, indices) pairs or None."""
    cx, cy = (float(v) for v in proj.to_local(*CENTER))
    nx = min(max(cx, bbox[0]), bbox[2]); ny = min(max(cy, bbox[1]), bbox[3])
    if math.hypot(nx - cx, ny - cy) > OUTER_R + 60.:
        return None, None
    glow, solid = _Sink(), _Sink()
    tree = _street_points(paths)
    geoms = np.empty(len(buildings), dtype=object)
    geoms[:] = [b.footprint for b in buildings]
    btree = shapely.STRtree(geoms)
    placed, count = [], 0
    for bi, b in enumerate(buildings):
        fp = b.footprint
        if fp is None or fp.is_empty or b.min_height > .5 or b.height < 14. or b.layer < 0:
            continue
        c = fp.centroid
        w = _weight(c.x, c.y, cx, cy)
        if w <= 0. or _hash(b.id, 977) > .10 + .34 * w:   # sparse: a few big screens, not a wall of them
            continue
        foundation = _ground_z(b, hf)
        faces = _frontages(fp, tree) if w < .6 else _exposed_edges(b, bi, btree, geoms)[:2]
        for edge_id, (a, _b, t, n, length) in enumerate(faces):
            if length < 8.5: continue
            if _hash(b.id, 55 + edge_id) > .65 or _claimed(placed, a, t, length): continue
            top = foundation + b.height - 2.0
            lo = foundation + 7.5
            if top - lo < 6.: continue
            seed = _hash(b.id, edge_id + 31)
            reach = min(top, lo + 14. + (28. + 60. * w) * _hash(b.id, 5 + edge_id))
            # stacked tiers of screens, widest first
            z = lo
            tier = 0
            while z + 5. < reach and tier < 3:
                th = min(reach - z, 7. + 9. * _hash(b.id, 40 + tier + edge_id))
                s = .8
                k = 0
                while s + 5.5 < length - .8:
                    pw = min(length - .8 - s, 6. + 10. * _hash(b.id, 90 + tier * 7 + k + edge_id))
                    if pw >= 5.5:
                        tint = TINTS[int(_hash(b.id, 200 + tier * 11 + k) * len(TINTS)) % len(TINTS)]
                        _screen(glow, solid, a, t, n, s, s + pw, z, z + th,
                                _hash(b.id, 300 + tier * 13 + k + edge_id), tint,
                                lift=.02 * (count % 5))
                        count += 1
                    s += pw + 1.6
                    k += 1
                z += th + 1.6
                tier += 1
            # ticker band above the shopfronts
            if w > .35:
                _band(glow, a, t, n, .6, length - .6, foundation + 4.4, foundation + 5.7,
                      seed, TINTS[int(seed * 977) % len(TINTS)])
            # corner blade sign
            if w > .55 and _hash(b.id, 611 + edge_id) < .55 and top - foundation > 22.:
                _blade(glow, solid, a, t, n, .9 if edge_id == 0 else length - .9,
                       foundation + 8., min(top, foundation + 8. + 18. + 24. * w),
                       _hash(b.id, 700 + edge_id), TINTS[int(_hash(b.id, 710) * len(TINTS)) % len(TINTS)])
    _red_steps(glow, solid, proj, bbox, hf, paths, btree, geoms)
    return glow.mesh(), solid.mesh()


def _clear_spot(x, y, up, side, w, d, paths, btree, geoms):
    """Nearest rectangle of plaza free of buildings and carriageways."""
    roads = [LineString(p.points).buffer(p.width * .5 + 1.5) for p in paths or ()
             if p.highway not in _WALK and not p.tunnel and not p.covered]
    rtree = shapely.STRtree(np.array(roads, dtype=object)) if roads else None
    best = None
    for r in range(0, 66, 6):
        for k in range(1 if r == 0 else 12):
            ang = 2 * math.pi * k / 12
            cx, cy = x + r * math.cos(ang), y + r * math.sin(ang)
            c = np.array([cx, cy])
            rect = Polygon([c + side * w * .5, c - side * w * .5,
                            c - side * w * .5 + up * d, c + side * w * .5 + up * d]).buffer(1.0)
            if len(btree.query(rect, predicate="intersects")): continue
            if rtree is not None and len(rtree.query(rect, predicate="intersects")): continue
            return c
    return best


def _red_steps(glow, solid, proj, bbox, hf, paths=None, btree=None, geoms=None):
    """Stepped red grandstand at the north end of the square (only on clear plaza)."""
    x, y = (float(v) for v in proj.to_local(*STEPS))
    if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
        return
    up = np.array([math.sin(GRID_BEARING), math.cos(GRID_BEARING)])
    side = np.array([up[1], -up[0]])
    steps, rise, tread, width = 13, .30, .60, 16.
    spot = _clear_spot(x, y, up, side, width, steps * tread, paths, btree, geoms)
    if spot is None: return
    x, y = float(spot[0]), float(spot[1])
    ground = 0.
    if hf is not None:
        try: ground = float(hf.at(np.array([x]), np.array([y]))[0])
        except Exception: ground = 0.
    ground += .12
    for i in range(steps):
        z1 = ground + (i + 1) * rise
        c = np.array([x, y]) + up * (i + .5) * tread
        p = [c + side * width * .5 - up * tread * .5, c - side * width * .5 - up * tread * .5,
             c - side * width * .5 + up * tread * .5, c + side * width * .5 + up * tread * .5]
        for k in range(4):
            a, b = p[k], p[(k + 1) % 4]
            solid.quad((a[0], a[1], ground), (b[0], b[1], ground), (b[0], b[1], z1), (a[0], a[1], z1),
                       RED, 7.0, emissive=0.)
        solid.quad((p[0][0], p[0][1], z1), (p[1][0], p[1][1], z1), (p[2][0], p[2][1], z1),
                   (p[3][0], p[3][1], z1), RED * 1.15, 7.0, emissive=0.)
        f0 = np.array([x, y]) + up * (i * tread) + side * (-width * .5 + .2)
        f1 = np.array([x, y]) + up * (i * tread) + side * (width * .5 - .2)
        o = -up * .02
        z0 = ground + i * rise
        glow.quad((f0[0] + o[0], f0[1] + o[1], z0 + .03), (f1[0] + o[0], f1[1] + o[1], z0 + .03),
                  (f1[0] + o[0], f1[1] + o[1], z0 + .15), (f0[0] + o[0], f0[1] + o[1], z0 + .15),
                  np.array([1., .25, .22], np.float32), 8.0, (0., BAND, 0.), emissive=1.0)
