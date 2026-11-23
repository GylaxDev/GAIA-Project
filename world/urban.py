"""Procedural street-level urban detail.

Everything here is generated in local tile coordinates and packed into the
same 17-float vertex layout used by buildings and ground ribbons:

    0..2  position        3..5  normal        6..8  colour
    9     AO              10..12 style (seed, kind, height)
    13..14 UV             15    emissive     16    part kind

Part kinds written by this module:

    8.0   neon sign / billboard / marquee   (drawn by the neon shader)
    9.0   crown LED strip / spire beacon    (neon shader, additive)
    10.0  opaque parallax storefront        (building shader)
    7.0   wood/metal/concrete street props   (opaque building shader)
    11.0  road marking decal               (terrain shader, road-material inputs)

The geometry is intentionally cheap: flat storefront portals and fascias,
with cached low-poly furniture and boxes for mounted canopies and hardware. Interior
depth and frames come from the material shader, not extra meshes.
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from ..core.config import ROAD_SURFACE, ROAD_WIDTHS
from .road_layout import road_width
from .surface_geometry import TerrainDraper, refine_triangles, terrain_path, vertex_normals
from .data_fetcher import Building, RoadPath, road_paths_from_roads, sample_path_position
from .street_furniture import bench_template, bin_template, hydrant_template, bollard_template

STRIDE = 17

# cyan, magenta, yellow, warm red, white, green, electric blue, orange
NEON = np.array([
    (0.10, 0.92, 1.00), (1.00, 0.16, 0.86), (1.00, 0.86, 0.12),
    (1.00, 0.26, 0.12), (1.00, 0.97, 0.90), (0.22, 1.00, 0.55),
    (0.28, 0.48, 1.00), (1.00, 0.55, 0.10),
], np.float32)

WARM = np.array([1.00, 0.72, 0.34], np.float32)
COOL = np.array([0.78, 0.90, 1.00], np.float32)
METAL = np.array([0.16, 0.17, 0.19], np.float32)
DARK = np.array([0.055, 0.058, 0.062], np.float32)
CONCRETE = np.array([0.62, 0.62, 0.60], np.float32)
WHITE = np.array([0.88, 0.89, 0.86], np.float32)
AWNING = np.array([
    (0.12, 0.25, 0.20), (0.16, 0.20, 0.25), (0.29, 0.16, 0.14),
    (0.28, 0.27, 0.23), (0.17, 0.18, 0.17),
], np.float32)


def _hash(seed, k):
    # Avalanche both inputs: the old linear hash made neighbouring bays almost identical.
    x = (int(seed) ^ (int(k) * 0x9E3779B9)) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return float(x ^ (x >> 16)) / 0xFFFFFFFF


def _pick(arr, seed, k):
    return arr[int(_hash(seed, k) * len(arr)) % len(arr)]


class _Sink:
    """Accumulates quads/triangles into one vertex/index pair."""

    def __init__(self):
        self.v, self.i = [], []
        self.n = 0

    def tri(self, p0, p1, p2, col, part, uv=((0, 0), (1, 0), (1, 1)),
            style=(0.0, 0.0, 0.0), emissive=1.0, ao=1.0):
        p0 = np.asarray(p0, np.float64); p1 = np.asarray(p1, np.float64)
        p2 = np.asarray(p2, np.float64)
        nrm = np.cross(p1 - p0, p2 - p0)
        ln = np.linalg.norm(nrm)
        nrm = nrm / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])
        rows = np.zeros((3, STRIDE), np.float32)
        rows[0, 0:3] = p0; rows[1, 0:3] = p1; rows[2, 0:3] = p2
        rows[:, 3:6] = nrm
        rows[:, 6:9] = col
        rows[:, 9] = ao
        rows[:, 10:13] = style
        rows[0, 13:15] = uv[0]; rows[1, 13:15] = uv[1]; rows[2, 13:15] = uv[2]
        rows[:, 15] = emissive
        rows[:, 16] = part
        self.v.append(rows)
        self.i.append(np.array([self.n, self.n + 1, self.n + 2], np.uint32))
        self.n += 3

    def quad(self, a, b, c, d, col, part, style=(0.0, 0.0, 0.0),
              uv=None, emissive=1.0, ao=1.0):
        points = np.asarray((a, b, c, d), np.float64)
        if uv is None:
            uv = ((0, 0), (1, 0), (1, 1), (0, 1))
        n0 = np.cross(points[1]-points[0], points[2]-points[0])
        n1 = np.cross(points[2]-points[0], points[3]-points[0])
        n0 /= max(float(np.linalg.norm(n0)), 1e-9)
        n1 /= max(float(np.linalg.norm(n1)), 1e-9)
        if np.linalg.norm(n0-n1) > .001:
            # Bent sidewalk strips retain each triangle's real terrain normal.
            self.tri(a, b, c, col, part, (uv[0], uv[1], uv[2]), style, emissive, ao)
            self.tri(a, c, d, col, part, (uv[0], uv[2], uv[3]), style, emissive, ao)
            return
        rows = np.zeros((4, STRIDE), np.float32)
        rows[:, :3] = points; rows[:, 3:6] = n0; rows[:, 6:9] = col
        rows[:, 9] = ao; rows[:, 10:13] = style; rows[:, 13:15] = uv
        rows[:, 15] = emissive; rows[:, 16] = part
        self.v.append(rows)
        self.i.append(np.array([0, 1, 2, 0, 2, 3], np.uint32)+self.n)
        self.n += 4

    def box(self, cx, cy, z0, z1, sx, sy, col, part, style=(0.0, 0.0, 0.0)):
        hx, hy = sx * 0.5, sy * 0.5
        p = [(cx - hx, cy - hy), (cx + hx, cy - hy),
             (cx + hx, cy + hy), (cx - hx, cy + hy)]
        for k in range(4):
            a, b = p[k], p[(k + 1) % 4]
            self.quad((a[0], a[1], z0), (b[0], b[1], z0),
                      (b[0], b[1], z1), (a[0], a[1], z1),
                      col, part, style, ((0, 0), (1, 0), (1, 1), (0, 1)))
        self.quad((p[0][0], p[0][1], z1), (p[1][0], p[1][1], z1),
                  (p[2][0], p[2][1], z1), (p[3][0], p[3][1], z1),
                  col, part, style)
        self.quad((p[3][0], p[3][1], z0), (p[2][0], p[2][1], z0),
                  (p[1][0], p[1][1], z0), (p[0][0], p[0][1], z0),
                  col, part, style)

    def beam(self, a, b, width, z, col, part, style=(0.0, 0.0, 0.0)):
        a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
        d = b - a
        ln = float(np.hypot(d[0], d[1]))
        if ln < 1e-3: return
        nx, ny = -d[1] / ln * width * 0.5, d[0] / ln * width * 0.5
        self.quad((a[0] + nx, a[1] + ny, z), (b[0] + nx, b[1] + ny, z),
                  (b[0] - nx, b[1] - ny, z), (a[0] - nx, a[1] - ny, z),
                   col, part, style)

    def edge_box(self, base, tangent, normal, s0, s1, out0, out1, z0, z1,
                 col, style=(0.0, 0.0, 0.0)):
        """Closed facade-aligned prism: frames and mounts have actual thickness."""
        p = [np.r_[np.asarray(base) + tangent*s + normal*out, z]
             for z in (z0, z1) for s, out in
             ((s0, out0), (s1, out0), (s1, out1), (s0, out1))]
        center = np.mean(p, axis=0)
        for a, b, c, d in ((0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
                           (3, 0, 4, 7), (4, 5, 6, 7), (3, 2, 1, 0)):
            face_normal = np.cross(p[b]-p[a], p[c]-p[a])
            if np.dot(face_normal, np.mean([p[a], p[b], p[c], p[d]], axis=0)-center) < 0.:
                b, d = d, b
            self.quad(p[a], p[b], p[c], p[d], col, 7.0, style, emissive=0.0)

    def mesh(self):
        if not self.v:
            return None
        return (np.ascontiguousarray(np.vstack(self.v), np.float32),
                np.ascontiguousarray(np.concatenate(self.i), np.uint32))

    def instance(self, pair, base, tangent, normal, z, seed=0., slope=(0., 0.)):
        """Append a cached metric prop, reflecting winding on the opposite curb."""
        v, idx = pair
        rotation = np.array([[tangent[0], normal[0], 0.],
                             [tangent[1], normal[1], 0.], [slope[0], slope[1], 1.]])
        block = v.copy()
        block[:, :3] = v[:, :3] @ rotation.T + np.r_[base, z]
        block[:, 3:6] = v[:, 3:6] @ np.linalg.inv(rotation)
        block[:, 3:6] /= np.maximum(np.linalg.norm(block[:, 3:6], axis=1, keepdims=True), 1e-8)
        block[:, 10] = np.mod(v[:, 10]+seed, 1.)
        indices = idx.reshape(-1, 3)
        if np.linalg.det(rotation) < 0.:
            indices = indices[:, [0, 2, 1]]
        self.v.append(block)
        self.i.append(indices.ravel()+np.uint32(self.n))
        self.n += len(block)


def merge(parts):
    """Merge (verts, indices) tuples; returns one tuple or None."""
    parts = [p for p in parts if p is not None and len(p[1]) > 0]
    if not parts:
        return None
    verts = np.vstack([p[0] for p in parts]).astype(np.float32)
    idx, off = [], 0
    for v, i in parts:
        idx.append(i.astype(np.uint32) + np.uint32(off))
        off += len(v)
    return verts, np.ascontiguousarray(np.concatenate(idx), np.uint32)


def _ring(poly: Polygon):
    c = np.asarray(poly.exterior.coords, np.float64)[:-1]
    return c if len(c) >= 3 else None


def _front_edge(poly: Polygon, toward=None):
    """Street-facing outer edge + outward normal and unit direction.

    ``toward`` is a point on the nearest road; when supplied we prefer the
    facade whose midpoint is close to it and whose outward normal faces it,
    so shopfronts and billboards advertise to the street rather than a court.
    """
    c = _ring(poly)
    if c is None:
        return None
    if not poly.exterior.is_ccw:
        c = c[::-1]
    a_all, b_all = c, np.roll(c, -1, axis=0)
    d = b_all - a_all
    ln = np.hypot(d[:, 0], d[:, 1])

    if toward is not None:
        q = np.array([toward[0], toward[1]], np.float64)
        best = None
        for k in range(len(c)):
            if ln[k] < 3.0:
                continue
            dirv = d[k] / ln[k]
            nrm = np.array([dirv[1], -dirv[0]], np.float64)
            mid = (a_all[k] + b_all[k]) * 0.5
            dist = float(np.linalg.norm(mid - q))
            facing = 0.0 if np.dot(nrm, q - mid) > 0.0 else 25.0
            score = dist + facing
            if best is None or score < best[0]:
                best = (score, a_all[k], b_all[k], dirv, nrm, float(ln[k]))
        if best is not None:
            return best[1:]

    k = int(np.argmax(ln))
    if ln[k] < 3.0:
        return None
    a, b = a_all[k], b_all[k]
    dirv = (b - a) / ln[k]
    nrm = np.array([dirv[1], -dirv[0]], np.float64)
    return a, b, dirv, nrm, float(ln[k])


def _street_points(paths):
    """Spatial index of surface road centrelines for facade snapping.

    Project onto the line, not sparse samples: sample gaps could falsely select
    a party wall perpendicular to the street as another storefront frontage.
    """
    if not paths:
        return None
    try:
        from shapely.geometry import Point
        from shapely.strtree import STRtree
    except Exception:
        return None
    lines = []
    for p in paths:
        if (p.width < 5.0 or p.total < 20.0 or p.tunnel or p.covered
                or p.bridge or p.layer > 0 or (p.level is not None and p.level < 0.)):
            continue
        lines.append(LineString(p.points))
    if not lines:
        return None
    return STRtree(lines)


def _ground_z(b: Building, hf):
    if b.ground_elevation is not None:
        return float(b.ground_elevation)
    if hf is None:
        return 0.0
    try:
        ring = np.asarray(b.footprint.exterior.coords, np.float64)
        # Match the extruder's foundation datum, including below-sea-level tiles.
        return float(np.min(hf.at(ring[:, 0], ring[:, 1]))) - 0.5
    except Exception:
        return 0.0


def _quad_on_edge(sink: _Sink, base, dirv, nrm, s0, s1, z0, z1,
                  col, part, style, out=0.10, emissive=1.0):
    a = np.array([base[0] + dirv[0] * s0 + nrm[0] * out,
                  base[1] + dirv[1] * s0 + nrm[1] * out, z0])
    b = np.array([base[0] + dirv[0] * s1 + nrm[0] * out,
                  base[1] + dirv[1] * s1 + nrm[1] * out, z0])
    c = b.copy(); c[2] = z1
    d = a.copy(); d[2] = z1
    v = (s1 - s0) / max(1e-3, abs(s1 - s0))
    sink.quad(a, b, c, d, col, part, style,
              uv=((0.0, 0.0), (v, 0.0), (v, 1.0), (0.0, 1.0)), emissive=emissive)


def storefront_kind(b, bay=0):
    """0 cafe, 1 display retail, 2 market, 3 shopping arcade, 4 lobby.

    Mapped usage wins; untagged downtown frontage gets a deterministic mix.
    These are representative interiors, not invented OSM businesses.
    """
    shop, amenity = b.shop.lower(), b.amenity.lower()
    if shop in ("mall", "department_store") or amenity == "food_court": return 3
    if shop in ("supermarket", "convenience", "greengrocer") or b.kind == "supermarket": return 2
    if amenity in ("cafe", "restaurant", "bar", "pub", "fast_food") or shop == "bakery": return 0
    if shop or b.kind in ("retail", "kiosk"): return 1
    if b.kind in ("hotel", "office") and bay == 0: return 4
    h = _hash(b.id, bay + 503)
    return 0 if h < .28 else (1 if h < .65 else (2 if h < .84 else 4))


def _frontages(poly, tree):
    """Up to two exposed street-facing edges, including a corner storefront."""
    if tree is None:
        fe = _front_edge(poly)
        return [fe] if fe is not None else []
    from shapely.geometry import Point
    ring = _ring(poly)
    if ring is None: return []
    # Canonical winding keeps geometry normals outward and text unmirrored,
    # not merely the normal used by the road-facing selection heuristic.
    if not poly.exterior.is_ccw:
        ring = ring[::-1]
    edges = []
    for a, b in zip(ring, np.roll(ring, -1, axis=0)):
        length = float(np.linalg.norm(b-a))
        if length < 4.0: continue
        tangent = (b-a)/length
        normal = np.array([tangent[1], -tangent[0]])
        mid = (a+b)*.5
        hit = tree.nearest(Point(*mid))
        line = tree.geometries[int(hit)]
        road = line.interpolate(line.project(Point(*mid)))
        delta = np.array([road.x, road.y])-mid
        distance = float(np.linalg.norm(delta))
        if distance < 42.0 and np.dot(delta, normal) > 1.0:
            edges.append((distance, (a, b, tangent, normal, length)))
    return [fe for _, fe in sorted(edges, key=lambda item: item[0])[:2]]


def _awning(sink, a, tangent, normal, s0, s1, z, color, seed, rigid=False):
    """Wall-mounted awning: closed cloth shell/valance, or a supported entry slab."""
    if rigid:
        sink.edge_box(a, tangent, normal, s0, s1, -.03, .95, z-.12, z, METAL)
    else:
        def p(s, out, h): return np.r_[a+tangent*s+normal*out, h]
        pa, pb = p(s0, -.02, z), p(s1, -.02, z)
        pc, pd = p(s1, 1.05, z-.35), p(s0, 1.05, z-.35)
        style = (seed, 1.0, 0.0)  # cloth material
        uv = ((0, 0), (s1-s0, 0), (s1-s0, 1.1), (0, 1.1))
        sink.quad(pa, pd, pc, pb, color, 7., style,
                  (uv[0], uv[3], uv[2], uv[1]), emissive=0.)
        down = np.array([0., 0., -.035])
        sink.quad(pa+down, pb+down, pc+down, pd+down, color*.72, 7., style, uv, emissive=0.)
        sink.quad(pc, pd, pd+np.array([0., 0., -.16]), pc+np.array([0., 0., -.16]),
                  color, 7., style, uv, emissive=0.)
        sink.tri(pa, np.r_[pa[:2], pd[2]], pd, color*.85, 7., style=style, emissive=0.)
        sink.tri(pb, pc, np.r_[pb[:2], pc[2]], color*.85, 7., style=style, emissive=0.)
    # Brackets and rear rail are embedded in the wall, not hovering in front of it.
    sink.edge_box(a, tangent, normal, s0, s1, -.04, .09, z-.10, z+.04, METAL)
    for s in (s0+.12, s1-.12):
        sink.edge_box(a, tangent, normal, s-.025, s+.025, -.03, .85,
                      z-(.17 if rigid else .43), z-(.12 if rigid else .38), METAL)
        sink.edge_box(a, tangent, normal, s-.035, s+.035, -.04, .07, z-.6, z, METAL)


def build_neon(buildings, hf=None, paths=None):
    """Return (sign emission, opaque storefronts/frames/awnings).

    Storefront portals use metric UVs, style=(seed, usage, room width) and
    emissive=room height. They are depth-writing glass, never additive sign strips.
    """
    solid = _Sink()
    tree = _street_points(paths)
    if paths is not None and tree is None:
        return None, None
    for b in buildings:
        if (b.footprint is None or b.footprint.is_empty or b.min_height > .5
                or not b.street_details or b.layer < 0 or b.height < 3.4):
            continue
        tagged = bool(b.shop or b.amenity in ("cafe", "restaurant", "bank", "bar", "food_court"))
        inferred = b.kind in ("yes", "office", "commercial", "mixed", "hotel", "retail", "supermarket", "kiosk")
        if not tagged and not (inferred and (b.commercial > .05 or b.height > 11.)):
            continue
        foundation = _ground_z(b, hf)
        for edge_id, (a, _, tangent, normal, length) in enumerate(_frontages(b.footprint, tree)):
            count = max(1, min(16, int((length-1.)/6.2)))
            pitch = (length-1.)/count
            for k in range(count):
                width = min(9.5, pitch-.18)
                if width < 2.7: continue
                s0 = .5+k*pitch+(pitch-width)*.5
                s1 = s0+width
                usage = storefront_kind(b, k)
                seed = _hash(b.id, k+edge_id*37+101)
                mid = a+tangent*(s0+s1)*.5
                base = foundation+.23
                if hf is not None:
                    base = max(base, float(hf.at(np.array([mid[0]]), np.array([mid[1]]))[0])+.23)
                height = min(3.55 if usage == 3 else 3.05, foundation+b.height-base-.85)
                if height < 2.45: continue  # no hovering shops on steep/low buildings
                top = base+height
                # A single facade face hides the unrelated old podium-window pattern.
                # Flat jambs/mullions live in the portal material, not twelve-faced boxes.
                _quad_on_edge(solid, a, tangent, normal, s0-.08, s1+.08, foundation-.05, top+.65,
                              np.array([.34, .33, .30]), 7., (seed, 0., 0.), out=.055, emissive=0.)
                corners = [np.r_[a+tangent*s+normal*.085, z]
                           for s, z in ((s0, base), (s1, base), (s1, top), (s0, top))]
                solid.quad(*corners, COOL, 10., (seed, float(usage), width),
                           ((0., 0.), (width, 0.), (width, height), (0., height)), emissive=height)
                # The handle projects from the glass; the frame is opaque shader detail.
                door = s0+width*.5
                dw = min(1.4, width*.32)
                solid.edge_box(a, tangent, normal, door+dw*.34-.018, door+dw*.34+.018,
                               .12, .21, base+.95, base+1.30, np.array([.61, .61, .58]))
                _quad_on_edge(solid, a, tangent, normal, s0, s1, top+.13, top+.59,
                              _pick(AWNING, b.id, k+41), 7., (seed, 2., float(usage)),
                               out=.17, emissive=width)  # legacy fascia metadata; the material is unlettered
                if usage == 0 and (tagged or k == 0) and _hash(b.id, k+73) < .72:
                    _awning(solid, a, tangent, normal, s0+.08, s1-.08, top+.08,
                            _pick(AWNING, b.id, k+41), seed)
                elif usage in (3, 4) and k == 0:
                    _awning(solid, a, tangent, normal, door-1.1, door+1.1, top+.08,
                            METAL, seed, rigid=True)
    return None, solid.mesh()


# crown-light colours: cool white, teal, magenta, warm white (Miami-skyline palette)
CROWN_LED = np.array([
    (0.80, 0.92, 1.00), (0.10, 0.90, 1.00), (1.00, 0.22, 0.62), (1.00, 0.80, 0.52),
], np.float32)


def build_crowns(buildings, hf=None):
    """Thin LED line just under the roof edge of some towers over 100 m.

    The old version built stepped metal hats with glowing collars; those read as
    a floating canopy on top of the building.  This hugs the facade (12 cm proud
    of the wall), is lit only at dusk/night (see the neon shader) and adds no
    geometry above the roof - parapets, plant rooms and the mast come from the
    mesh extruder.  Returns (solid, glow) like before; solid is always empty.
    """
    solid, glow = _Sink(), _Sink()
    for b in buildings:
        if b.height <= 100.0 or b.footprint is None or b.footprint.is_empty:
            continue
        if getattr(b, "roof_shape", "flat") != "flat":
            continue
        seed = float(b.id & 0xFFFF)
        if _hash(seed, 53) > 0.55:          # not every tower has a lit crown
            continue
        try:
            ring = _ring(Polygon(b.footprint.exterior).buffer(0.12, join_style=2))
        except Exception:
            ring = None
        if ring is None or len(ring) < 3:
            continue
        z0 = _ground_z(b, hf)
        from .data_fetcher import _jitter
        top = z0 + b.height + 0.15 * _jitter(b.id + 7919)
        col = _pick(CROWN_LED, seed, 47)
        n = len(ring)
        for j in range(n):
            j2 = (j + 1) % n
            glow.quad((ring[j][0], ring[j][1], top - 1.9), (ring[j2][0], ring[j2][1], top - 1.9),
                      (ring[j2][0], ring[j2][1], top - 1.35), (ring[j][0], ring[j][1], top - 1.35),
                      col, 9.0, (seed, 5.0, top))
    return solid.mesh(), glow.mesh()


def _paths(roads, hf, sea_z, bbox):
    try:
        return road_paths_from_roads(roads, hf=hf, sea_z=sea_z, bbox=bbox)
    except Exception:
        return []


_WALK_FURNITURE_HIGHWAYS = frozenset(('footway', 'pedestrian'))
_WALK_FURNITURE_SPACING = 140.0
_WALK_FURNITURE_CELL = 160.0
_WALK_FURNITURE_PER_CELL = 3
_MAX_WALK_FURNITURE = 240
_FOOTPATH_LIFT = .10


def build_props(roads, paths, hf=None, bbox=None, sea_z=0.0, buildings=None, pavement=None,
                trees=None):
    """Street furniture; real signal furniture comes from the traffic graph.

    Dense centres often map their pavements as separate ``footway`` ways and tag
    the carriageway ``sidewalk:both=separate``. Those roads deliberately have no
    implicit band, so furniture must be seated on the mapped footway instead of
    being skipped. The pass below is spatially thinned to keep the per-tile
    geometry budget bounded.
    """
    props, neon = _Sink(), _Sink()
    if not paths:
        return props.mesh(), neon.mesh()

    # Keep furniture out of buildings, tree trunks and intersecting carriageways.
    # In particular, curb offsets on a short junction arm are not necessarily
    # usable sidewalk.
    obstacles = [b.footprint for b in (buildings or []) if b.min_height < 2.5
                 and b.footprint is not None and not b.footprint.is_empty]
    tree_points = np.asarray(trees, np.float64) if trees is not None and len(trees) else None
    if tree_points is not None:
        tree_points = tree_points[np.isfinite(tree_points[:, :2]).all(axis=1), :2]
        obstacles += [shapely.Point(x, y).buffer(.30) for x, y in tree_points]
    road_buffers = [LineString(p.points).buffer(p.width*.5, cap_style=2)
                    for p in paths if not p.tunnel and not (p.level is not None and p.level < 0.)]
    obstacles += road_buffers
    obstacle_tree = shapely.STRtree(obstacles) if obstacles else None

    occupied = []

    def clear(x, y, radius, footprint=None, surface=None):
        if bbox is not None and not (bbox[0]+radius < x < bbox[2]-radius
                                     and bbox[1]+radius < y < bbox[3]-radius): return False
        if footprint is not None:
            cover = pavement if surface is None else surface
            if cover is not None and not cover.covers(footprint): return False
            if any(footprint.intersects(q) for q in occupied): return False
            return obstacle_tree is None or not len(obstacle_tree.query(footprint, predicate='intersects'))
        return obstacle_tree is None or not len(obstacle_tree.query(
            shapely.Point(x, y), predicate='dwithin', distance=radius))

    for pi, p in enumerate(paths):
        if p.tunnel or p.covered or (p.level is not None and p.level < 0.):
            continue
        if p.total < 30.0:
            continue
        w = float(p.width)
        # ---- overhead light gantry over wide avenues ----------------------
        if w >= 13.0:
            for s in np.arange(210.0, p.total - 30.0, 420.0):
                x, y, hd = sample_path_position(p, s)
                z = float(sample_path_z(p, s))
                nx, ny = math.cos(hd + math.pi * 0.5), math.sin(hd + math.pi * 0.5)
                for side in (1.0, -1.0):
                    px = x + nx * (w * 0.5 + 1.0) * side
                    py = y + ny * (w * 0.5 + 1.0) * side
                    props.box(px, py, z, z + 6.6, 0.24, 0.24, METAL, 7.0)
                props.beam((x + nx * (w * 0.5 + 1.0), y + ny * (w * 0.5 + 1.0)),
                           (x - nx * (w * 0.5 + 1.0), y - ny * (w * 0.5 + 1.0)),
                           0.30, z + 6.4, METAL, 7.0)
                # green direction boards hanging from the gantry
                for k, off in enumerate((-0.22, 0.22)):
                    bx = x + nx * w * off
                    by = y + ny * w * off
                    props.quad((bx - math.cos(hd) * 1.4, by - math.sin(hd) * 1.4, z + 5.2),
                               (bx + math.cos(hd) * 1.4, by + math.sin(hd) * 1.4, z + 5.2),
                               (bx + math.cos(hd) * 1.4, by + math.sin(hd) * 1.4, z + 6.1),
                               (bx - math.cos(hd) * 1.4, by - math.sin(hd) * 1.4, z + 6.1),
                               np.array([0.06, 0.30, 0.14], np.float32), 7.0,
                               (0.0, 0.0, 0.0), emissive=0.0)

        # No synthetic banners: their independent sampling never matched the actual
        # instanced lamp poles. Useful, grounded furniture replaces the floating flags.
        for si, s in enumerate(np.arange(18.0, p.total - 14.0, 48.0)):
            if p.bridge or p.layer > 0 or w < 5.0: continue
            x, y, hd = sample_path_position(p, s)
            z = float(sample_path_z(p, s))
            nx, ny = math.cos(hd + math.pi * 0.5), math.sin(hd + math.pi * 0.5)
            side = 1.0 if si % 2 == 0 else -1.0
            sides = _sidewalk_sides(p)
            if not sides[0 if side > 0. else 1]:
                side = -side
                if not sides[0 if side > 0. else 1]: continue
            px, py = x+nx*(w*.5+1.12)*side, y+ny*(w*.5+1.12)*side
            top = z+.17 if hf is None else max(z, float(hf.at(np.array([x]), np.array([y]))[0])+.06)+.17
            a = np.array([px, py]); tangent = np.array([math.cos(hd), math.sin(hd)])
            normal = np.array([nx, ny])*side
            kind = (si+pi) % 3
            half_x, half_y = (1.02, .42) if kind == 0 else (.38, .38)
            footprint = Polygon([a+tangent*dx+normal*dy for dx, dy in
                                 ((-half_x, -half_y), (half_x, -half_y),
                                  (half_x, half_y), (-half_x, half_y))])
            if not clear(px, py, .65, footprint): continue
            # Sample where the feet stand, not at the road centre. Production
            # furniture shares the raised slab's datum, including side slopes.
            if pavement is not None:
                base = float(hf.at(np.array([px]), np.array([py]))[0]) if hf is not None else sea_z
                top = max(base, sea_z)+ROAD_SURFACE+.064+SIDEWALK_HEIGHT+.023
            slope = (0., 0.)
            if pavement is not None and hf is not None:
                q = np.array([a-tangent*.5, a+tangent*.5, a-normal*.5, a+normal*.5])
                h = np.maximum(hf.at(q[:, 0], q[:, 1]), sea_z)
                slope = (float(h[1]-h[0]), float(h[3]-h[2]))
            seed = _hash(p.uid or pi+1, si+83)
            occupied.append(footprint.buffer(.16))
            if kind == 0:
                props.instance(bench_template(int(seed > .72)), a, tangent, normal, top-.025, seed, slope)
            elif kind == 1:
                props.instance(bin_template(int(seed > .73)), a, tangent, normal, top-.025, seed, slope)
            else:
                props.instance(hydrant_template(), a, tangent, normal, top-.025, seed, slope)
            if pavement is not None and si == 0 and w >= 8. and seed < .32:
                for shift in (-3.7, -2.4):
                    q = a+tangent*shift
                    fp = shapely.Point(*q).buffer(.22)
                    if not clear(*q, .25, fp): continue
                    bz = (float(hf.at(np.array([q[0]]), np.array([q[1]]))[0])
                          if hf is not None else sea_z)
                    bz = max(bz, sea_z)+ROAD_SURFACE+.064+SIDEWALK_HEIGHT-.002
                    props.instance(bollard_template(int(seed > .16)), q, tangent, normal, bz, seed)
                    occupied.append(fp.buffer(.15))

        # ---- broad asphalt repairs; subpixel manholes/grates are omitted ----
        for s in np.arange(55.0, p.total - 12.0, 88.0):
            x, y, hd = sample_path_position(p, s)
            z = float(sample_path_z(p, s))
            # repair patch
            if int(s) % 3 == 0:
                rr = 1.6 + 1.2 * _hash(pi + int(s), 71)
                pts = []
                for k in range(6):
                    ang = k / 6.0 * 2 * math.pi
                    rad = rr * (0.7 + 0.5 * _hash(pi * 13 + int(s), k))
                    pts.append((x + math.cos(ang) * rad, y + math.sin(ang) * rad, z + 0.012))
                for k in range(1, 5):
                    props.tri(pts[0], pts[k], pts[k + 1],
                              np.array([0.045, 0.046, 0.048], np.float32), 11.0,
                              emissive=0.0)

    # ---- furniture on separately mapped pavements -------------------------
    # ``sidewalk:both=separate`` means the visible pavement is its own footway
    # way, not a band invented from the carriageway. Seat furniture on that
    # surface so dense centres do not lose every bench, bin and hydrant.
    walk_paths = road_paths_from_roads(
        [r for r in (roads or ()) if r.highway in _WALK_FURNITURE_HIGHWAYS
         and not r.tunnel and not r.covered and not r.bridge and r.layer <= 0
         and (r.level is None or r.level >= 0.)],
        min_length=12.0, hf=hf, sea_z=sea_z, bbox=bbox,
        highways=_WALK_FURNITURE_HIGHWAYS, require_drivable=False)
    ground_paths = [p for p in paths if not p.tunnel and not p.covered
                    and not p.bridge and p.layer <= 0
                    and (p.level is None or p.level >= 0.)]
    if walk_paths and ground_paths:
        ground_lines = [LineString(p.points) for p in ground_paths]
        road_tree = shapely.STRtree(ground_lines)
        walk_polys = [LineString(p.points).buffer(p.width*.5, cap_style=2, join_style=2)
                      for p in walk_paths if p.width >= 1.4]
        if walk_polys:
            walk_surface = unary_union(walk_polys)
            ground_buffers = [LineString(p.points).buffer(p.width*.5, cap_style=2)
                              for p in ground_paths]
            if ground_buffers:
                walk_surface = walk_surface.difference(unary_union(ground_buffers))
            if not walk_surface.is_empty:
                walk_cover = walk_surface.buffer(.08, join_style=2)
                candidates = []
                for wi, wp in enumerate(walk_paths):
                    if wp.width < 1.4 or wp.total < 12.0: continue
                    seed = wp.uid or wi + 1
                    arcs = ([wp.total*.5] if wp.total < 30.0 else
                            np.arange(8.0 + _hash(seed, 79)*20.0,
                                      wp.total - 6.0, _WALK_FURNITURE_SPACING))
                    for arc in arcs:
                        x, y, heading = sample_path_position(wp, float(arc))
                        a = np.array([x, y])
                        tangent = np.array([math.cos(heading), math.sin(heading)])
                        point = shapely.Point(x, y)
                        d = int(road_tree.nearest(point))
                        line, road = ground_lines[d], ground_paths[d]
                        along = float(line.project(point))
                        rx, ry, road_heading = sample_path_position(road, along)
                        delta = a - np.array([rx, ry])
                        distance = float(np.linalg.norm(delta))
                        if distance > 12.0: continue
                        road_tangent = np.array([math.cos(road_heading), math.sin(road_heading)])
                        # Sidewalk alignments run with their street; crossing
                        # stubs do not, so this also filters zebra footways.
                        if abs(float(road_tangent @ tangent)) < .55: continue
                        left = (road_tangent[0]*delta[1] - road_tangent[1]*delta[0]) > 0.
                        if _sidewalk_sides(road)[0 if left else 1]: continue
                        away = (-delta / distance if distance > .25 else
                                np.array([-tangent[1], tangent[0]]))
                        away = away - tangent * float(away @ tangent)
                        length = float(np.linalg.norm(away))
                        if length < 1e-6: continue
                        away /= length
                        kind = int(_hash(seed, 19) * 3.)
                        half_x, half_y = (1.02, .42) if kind == 0 else (.38, .38)
                        offset = max(0., wp.width*.5 - (.52 if kind == 0 else .42))
                        center = a + away * offset
                        footprint = Polygon([center + tangent*dx + away*dy for dx, dy in
                                             ((-half_x, -half_y), (half_x, -half_y),
                                              (half_x, half_y), (-half_x, half_y))])
                        if not walk_cover.covers(footprint): continue
                        base = (float(hf.at(np.array([center[0]]), np.array([center[1]]))[0])
                                if hf is not None else sea_z)
                        top = max(base, sea_z) + ROAD_SURFACE + _FOOTPATH_LIFT
                        slope = (0., 0.)
                        if hf is not None:
                            q = np.array([center-tangent*.5, center+tangent*.5,
                                          center-away*.5, center+away*.5])
                            h = np.maximum(hf.at(q[:, 0], q[:, 1]), sea_z)
                            slope = (float(h[1]-h[0]), float(h[3]-h[2]))
                        candidates.append((_hash(seed, int(arc*10)), seed, kind,
                                           center, tangent, away, footprint, top, slope))
                # Cap each 160 m cell before the global budget: one long avenue
                # must not consume every piece in the tile.
                cells = {}
                for item in candidates:
                    key = (int(item[3][0] // _WALK_FURNITURE_CELL),
                           int(item[3][1] // _WALK_FURNITURE_CELL))
                    cells.setdefault(key, []).append(item)
                chosen = []
                for group in cells.values():
                    group.sort(key=lambda item: item[0])
                    chosen.extend(group[:_WALK_FURNITURE_PER_CELL])
                chosen.sort(key=lambda item: item[0])
                for _, seed, kind, center, tangent, away, footprint, top, slope in \
                        chosen[:_MAX_WALK_FURNITURE]:
                    if not clear(center[0], center[1], .65, footprint, surface=walk_cover):
                        continue
                    occupied.append(footprint.buffer(.16))
                    if kind == 0:
                        props.instance(bench_template(int(_hash(seed, 23) > .72)),
                                       center, tangent, away, top-.025, seed, slope)
                    elif kind == 1:
                        props.instance(bin_template(int(_hash(seed, 29) > .73)),
                                       center, tangent, away, top-.025, seed, slope)
                    else:
                        props.instance(hydrant_template(), center, tangent, away,
                                       top-.025, seed, slope)

    return props.mesh(), neon.mesh()


def sample_path_z(p: RoadPath, s: float) -> float:
    zs = getattr(p, "z", None)
    if zs is None or len(zs) == 0:
        return 0.0
    i = int(np.clip(np.searchsorted(p.arc, s) - 1, 0, len(zs) - 2))
    seg = max(p.arc[i + 1] - p.arc[i], 1e-6)
    t = (s - p.arc[i]) / seg
    return float(zs[i] + (zs[i + 1] - zs[i]) * min(1.0, max(0.0, t)))


SIDEWALK_WIDTH = 2.15
SIDEWALK_CLEARANCE = .035
SIDEWALK_HEIGHT = .17


def _sidewalk_sides(p):
    return getattr(p, 'sidewalks', ()) or (p.highway not in ('motorway', 'motorway_link'),) * 2


def _sidewalk_band(p):
    """The same source-relative band for pavement layout and final tessellation."""
    sides = _sidewalk_sides(p)
    if not any(sides): return None
    line = LineString(p.points)
    options = dict(cap_style=2, join_style=2, mitre_limit=2.)
    outer = (line.buffer(p.width*.5+SIDEWALK_WIDTH, **options) if all(sides) else
             line.buffer((1. if sides[0] else -1.)*(p.width*.5+SIDEWALK_WIDTH),
                         single_sided=True, **options))
    return outer.difference(line.buffer(p.width*.5, cap_style=2, join_style=2))

# Junction zebra crossings.  Bars run with the traffic and step across the
# carriageway, so every band reaches the kerb (and its sidewalk) on both sides.
CROSSWALK_PITCH = 1.05
CROSSWALK_DEPTH = 2.9
# Above the highest carriageway lift in _build_ground (primary: 4 x .012 + .004)
# so bars read on avenues; previously .03 put them *under* the asphalt there.
CROSSWALK_LIFT = .065
STOP_LINE_WIDTH = .55
_NO_STOP_LINE = frozenset(("service", "living_street", "track", "pedestrian"))
_NO_CROSSWALK = frozenset(("motorway", "trunk", "motorway_link", "trunk_link"))


def _arc_at(line, arc, point):
    """Arc length along a polyline of the point closest to ``point``."""
    coords = np.asarray(line.coords, np.float64)
    best = (math.inf, 0.0)
    for k in range(len(coords) - 1):
        a, b = coords[k], coords[k + 1]
        d = b - a
        L2 = float(d @ d)
        if L2 < 1e-12: continue
        t = float(np.clip(np.dot(point - a, d) / L2, 0.0, 1.0))
        dist = float(np.hypot(*(point - (a + d * t))))
        if dist < best[0]:
            best = (dist, float(arc[k]) + t * (float(arc[k + 1]) - float(arc[k])))
    return best[1]


def _junction_approaches(paths):
    """(path, junction arc length, widest street met) for every approach.

    Junctions come from where two alignments actually meet, not from their node
    lists: clipped ways, collinear simplification and imported path splits all
    move or drop shared nodes, while the centrelines still intersect exactly.
    """
    lines = [LineString(p.points) for p in paths]
    if len(lines) < 2: return []
    tree = shapely.STRtree(lines)
    approaches = []
    for i, j in zip(*tree.query(lines, predicate="intersects")):
        if i >= j: continue
        p, q = paths[i], paths[j]
        if p.uid == q.uid: continue
        try:
            overlap = lines[i].intersection(lines[j])
        except Exception:
            continue
        if overlap.is_empty: continue
        if overlap.geom_type == "Point":
            point = np.array([overlap.x, overlap.y], np.float64)
        else:
            # A duplicated/offset alignment running alongside is not a junction.
            if overlap.length > 8.0: continue
            rep = overlap.representative_point()
            point = np.array([rep.x, rep.y], np.float64)
        approaches.append((p, _arc_at(lines[i], p.arc, point), float(q.width)))
        approaches.append((q, _arc_at(lines[j], q.arc, point), float(p.width)))
    return approaches


def crosswalk_bands(paths):
    """(path, arc length) of every zebra band this tile should paint.

    Each band is laid clear of the junction box of the street it meets, and is
    skipped where a second junction or the arm's own end leaves no room.
    """
    usable = [p for p in paths or ()
              if not p.tunnel and not p.covered and not p.bridge and p.layer <= 0
              and (p.level is None or p.level >= 0.)
              and p.highway not in _NO_CROSSWALK and p.width >= 4.5 and p.total >= 10.0]
    if not usable: return []
    grouped = {}
    for p, s_j, other in _junction_approaches(usable):
        grouped.setdefault(id(p), (p, []))[1].append((s_j, other))
    bands = []
    for p, meets in grouped.values():
        joints = {round(float(s), 3) for s, _ in meets}
        drawn = set()
        for s_j, other in meets:
            inset = max(3.6, min(13.0, other * .5 + 2.3))
            sides = ((-1., 1.) if 1.0 < s_j < p.total - 1.0
                     else ((1.,) if s_j <= 1.0 else (-1.,)))
            for side in sides:
                s = s_j + side * inset
                if s - CROSSWALK_DEPTH * .5 < 1.0 or s + CROSSWALK_DEPTH * .5 > p.total - 1.0:
                    continue
                if any(abs(s - j) < CROSSWALK_DEPTH + inset * .35
                       for j in joints if abs(j - s_j) > .5):
                    continue          # would land inside another junction on this arm
                key = round(s, 2)
                if key in drawn: continue
                drawn.add(key)
                bands.append((p, s, side))
    # Split/duplicated OSM approaches can put two rotated zebras on top of one
    # another. Keep the wider approach where their painted bands substantially
    # overlap, without dropping distinct crossings on opposite junction arms.
    if len(bands) > 1:
        footprints = [_crosswalk_footprint(p, s) for p, s, _ in bands]
        tree = shapely.STRtree(footprints)
        chosen = set()
        order = sorted(range(len(bands)), key=lambda i: (-bands[i][0].width, bands[i][0].uid, bands[i][1]))
        for i in order:
            poly = footprints[i]
            if any(j in chosen and poly.intersection(footprints[j]).area > .20 * min(poly.area, footprints[j].area)
                   for j in tree.query(poly, predicate='intersects')):
                continue
            chosen.add(i)
        bands = [band for i, band in enumerate(bands) if i in chosen]
    return bands


def build_crosswalks(paths, bands=None, hf=None):
    """Zebra crossings on every street-junction approach in this tile.

    These decals belong in the terrain mesh, not the opaque prop mesh. Road
    width/UVs preserve the asphalt underneath; negative paint classes select
    filtered zebra stripes (-1) or a stop line (-2) in the road shader.
    """
    from .mesh_extruder import MeshExtruder

    bands = crosswalk_bands(paths) if bands is None else bands
    if not bands: return None
    sink, draper, extruder = _Sink(), TerrainDraper(hf), MeshExtruder()
    for band in bands:
        p, s = band[:2]
        side = band[2] if len(band) > 2 else 0.
        _crosswalk_band(sink, p, s, draper, extruder)
        if side:
            _stop_line(sink, p, s, side, draper, extruder)
    return sink.mesh()


def _crosswalk_footprint(p, s):
    x, y, heading = sample_path_position(p, s)
    along = np.array([math.cos(heading), math.sin(heading)])
    across = np.array([-along[1], along[0]])
    half, half_w = CROSSWALK_DEPTH * .5, float(p.width) * .5
    c = np.array([x, y])
    return Polygon([c - across * half_w - along * half,
                    c + across * half_w - along * half,
                    c + across * half_w + along * half,
                    c - across * half_w + along * half])


def crosswalk_cutouts(bands):
    """Footprint of every zebra band, for cutting raised sidewalk out of it.

    A pedestrian route that reaches a crossing must give way to the painted
    carriageway: otherwise the raised slab spans the road and buries (clips)
    the bars where the two meet.
    """
    from shapely.ops import unary_union
    polys = [_crosswalk_footprint(*band[:2]) for band in bands or ()]
    return unary_union(polys) if polys else None


def _road_paint_decal(sink, p, s, lo, hi, half, paint_class, pitch, stripe, draper, extruder):
    """A road-aligned, terrain-conforming decal with the ribbon's material UVs."""
    from .data_fetcher import polygons_of

    x, y, heading = sample_path_position(p, s)
    along = np.array([math.cos(heading), math.sin(heading)])
    across = np.array([-along[1], along[0]])
    center = np.array([x, y])
    a, b = center + across * lo, center + across * hi
    footprint = Polygon([a - along * half, a + along * half,
                         b + along * half, b - along * half])
    # A tangent rectangle must not overhang the asphalt on a bent approach.
    carriageway = LineString(p.points).buffer(float(p.width) * .5, cap_style=2, join_style=2)
    for poly in polygons_of(footprint.intersection(carriageway)):
        if poly.area < .01: continue
        pts, tris = extruder._triangulate(poly)
        pts, tris = draper.conform(pts, tris)
        if not len(tris): continue
        arc, offset = extruder._polyline_frame(p.points, p.arc, pts)
        if draper.hf is not None:
            height = np.asarray(draper.hf.at(pts[:, 0], pts[:, 1]), float) + ROAD_SURFACE
        else:
            height = np.interp(arc, p.arc, p.z) if p.z is not None else np.zeros(len(pts))
        xyz = np.c_[pts, height + CROSSWALK_LIFT]
        block = np.zeros((len(pts), STRIDE), np.float32)
        block[:, :3], block[:, 3:6] = xyz, vertex_normals(xyz, tris)
        block[:, 6:9] = (0., float(p.width), paint_class)
        block[:, 9], block[:, 16] = 1., 11.
        block[:, 10] = np.minimum(arc, p.total - arc)
        block[:, 11:13] = (pitch, stripe)
        block[:, 13], block[:, 14] = .5 + offset / float(p.width), arc
        sink.v.append(block)
        sink.i.append(tris.ravel().astype(np.uint32) + np.uint32(sink.n))
        sink.n += len(pts)


def _crosswalk_band(sink, p, s, draper, extruder):
    """One filtered band across the carriageway at arc length ``s``."""
    # Kerb to kerb: a crossing that stops short of the kerb reads as a clipped
    # slab rather than a zebra, and leaves an unpainted asphalt strip at each end.
    width = max(2.0, float(p.width) - .12)
    count = max(2, int(round(width / CROSSWALK_PITCH)))
    pitch = width / count
    # Just under half paint: closely packed pale bars merged into what looked like
    # a sidewalk slab rather than a zebra from any distance.
    stripe = min(.55, pitch * .46)
    half = CROSSWALK_DEPTH * .5
    _road_paint_decal(sink, p, s, -width * .5, width * .5, half,
                      -1., pitch, stripe, draper, extruder)


def _stop_line(sink, p, s, side, draper, extruder):
    """Solid white stop bar across the lanes that approach this crossing."""
    if p.highway in _NO_STOP_LINE or p.width < 5.5: return
    from .road_layout import RoadLayout
    layout = RoadLayout.from_road(p)
    if layout.shared: return
    approaching_forward = side < 0          # traffic moving +s reaches a band that lies ahead
    w = float(p.width)
    if layout.backward == 0:                # one-way: only the arriving end gets a line
        if not approaching_forward: return
        lo, hi = -w * .5 + .12, w * .5 - .12
    else:
        c = -w * .5 + w * (layout.divider_uv or .5)
        lo, hi = ((-w * .5 + .12, c - .35) if approaching_forward else (c + .35, w * .5 - .12))
    if hi - lo < 1.5: return
    s_stop = s + side * (CROSSWALK_DEPTH * .5 + .9 + STOP_LINE_WIDTH * .5)
    if s_stop < 1.0 or s_stop > p.total - 1.0: return
    _road_paint_decal(sink, p, s_stop, lo, hi, STOP_LINE_WIDTH * .5,
                      -2., 0., 0., draper, extruder)


def build_sidewalks(roads, paths, hf=None, sea_z=0.0, cutouts=None, *, aprons=(),
                    exclusions=None, pits=None, clip=None, pavement=None):
    """Conforming, upward-facing bands, clipped against visible same-grade roads.

    Buffer the actual open alignment instead of connecting sampled normals at
    endpoints. Unioning the bands removes duplicates; polygon difference trims
    intersections exactly instead of dropping an entire ten-metre slab.
    ``pavement`` can supply the planner's already-clipped ground-grade union;
    elevated sidewalks still use their own road geometry and deck profiles.
    """
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union
    # Local import avoids a module cycle: mesh_extruder calls this after loading.
    from .mesh_extruder import MeshExtruder
    from .data_fetcher import _densify, polygons_of

    sink, extruder, groups = _Sink(), MeshExtruder(), {}

    def grade(p):
        # Ground grade is one group whether the way is untagged or tagged level=0:
        # splitting them left sidewalks uncut where the two kinds met, so a slab
        # could span a street it should have stopped at the kerb of.
        level = getattr(p, 'level', None)
        ground = level is None or abs(float(level)) < 1e-6
        return (int(p.layer), bool(p.bridge or p.layer > 0), None if ground else level)

    def visible(p):
        return not p.tunnel and not (p.level is not None and p.level < 0.)

    for r in roads:
        if not visible(r): continue
        key = grade(r)
        group = groups.setdefault(key, dict(roads=[], bands=[], paths=[]))
        if pavement is None or key != (0, False, None):
            group['roads'].append(r.geometry.buffer(road_width(r) * .5, cap_style=2, join_style=2))
    for p in paths:
        if not visible(p): continue
        key = grade(p)
        group = groups.setdefault(key, dict(roads=[], bands=[], paths=[]))
        # Include short traffic paths and caller-supplied paths in the obstacle set.
        if pavement is None or key != (0, False, None):
            group['roads'].append(LineString(p.points).buffer(p.width * .5, cap_style=2, join_style=2))
        if p.covered or p.width < 5. or p.total < 30.: continue
        band = _sidewalk_band(p)
        if band is None: continue
        group['bands'].append(band)
        group['paths'].append(p)

    for apron, p in aprons:
        group = groups.get(grade(p))
        if group is not None and not apron.is_empty:
            group['bands'].append(apron)
            group['paths'].append(p)

    draper = TerrainDraper(hf)
    for key, group in groups.items():
        if not group['bands']: continue
        elevated = key[1]
        if pavement is not None and key == (0, False, None):
            surface = pavement
        else:
            carriageways = unary_union(group['roads']).buffer(SIDEWALK_CLEARANCE, join_style=2)
            if cutouts is not None and not cutouts.is_empty:
                # The zebra replaces the sidewalk across the carriageway; keep the
                # raised slab clear of the painted bars instead of burying them.
                carriageways = unary_union([carriageways,
                                            cutouts.buffer(SIDEWALK_CLEARANCE, join_style=2)])
            if not elevated:
                carriageways = unary_union([carriageways, *[g for g in (exclusions, pits)
                                                           if g is not None and not g.is_empty]])
            surface = unary_union(group['bands']).difference(carriageways)
        if clip is not None: surface = surface.intersection(clip)
        # A bridge keeps its deck profile, never drapes down onto the ground.
        if elevated:
            slabs = [(band.difference(carriageways), p) for band, p in zip(group['bands'], group['paths'])]
            if clip is not None: slabs = [(slab.intersection(clip), p) for slab, p in slabs]
        else:
            # Exclusive ownership preserves the union but gives each street its
            # own curb-aligned slab UVs. Curbs come from the UNION, not these
            # partitions: no internal walls or stacked tops at corner joins.
            slabs = []
            order = sorted(zip(group['bands'], group['paths']),
                           key=lambda item: (-item[1].width, item[1].uid,
                                             tuple(item[1].points[0]), -item[1].total))
            tree = shapely.STRtree([band for band, _ in order])
            for rank, (band, p) in enumerate(order):
                earlier = [order[j][0] for j in tree.query(band, predicate='intersects') if j < rank]
                slab = band.intersection(surface)
                if earlier: slab = slab.difference(unary_union(earlier))
                if not slab.is_empty: slabs.append((slab, p))

        def heights(q, path=None):
            if elevated and path is not None:
                along, _ = extruder._polyline_frame(path.points, path.arc, q)
                zs = path.z if path.z is not None else np.full(len(path.points), sea_z)
                return np.interp(along, path.arc, zs)+SIDEWALK_HEIGHT+.064
            base = np.asarray(hf.at(q[:, 0], q[:, 1]), float) if hf is not None else np.full(len(q), sea_z)
            return np.maximum(base, sea_z)+ROAD_SURFACE+.064+SIDEWALK_HEIGHT

        for slab, path in slabs:
            for poly in polygons_of(slab):
                if poly.is_empty or poly.area < .01: continue
                poly = orient(poly, sign=1.)
                pts, tris = extruder._triangulate(poly)
                if not len(tris): continue
                pts, tris = (refine_triangles(pts, tris, 12.) if elevated
                             else draper.conform(pts, tris, max_edge=12.))
                if not len(tris): continue
                xyz = np.c_[pts, heights(pts, path)]
                block = np.zeros((len(pts), STRIDE), np.float32)
                block[:, :3], block[:, 3:6] = xyz, vertex_normals(xyz, tris)
                block[:, 6], block[:, 9], block[:, 16] = 7., 1., 5.
                along, offset = extruder._polyline_frame(path.points, path.arc, pts)
                tangent = path.points[1]-path.points[0]
                tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
                phase = float(path.points[0] @ tangent)
                block[:, 10:13] = (_hash(path.uid, 107), 1., SIDEWALK_WIDTH)
                block[:, 13] = np.abs(offset)-path.width*.5
                block[:, 14] = along+phase
                sink.v.append(block)
                sink.i.append(tris.ravel().astype(np.uint32) + np.uint32(sink.n))
                sink.n += len(pts)
        curb_slabs = slabs if elevated else [(surface, None)]
        for slab, path in curb_slabs:
            for poly in polygons_of(slab):
                if poly.is_empty or poly.area < .01: continue
                poly = orient(poly, sign=1.)
                # CCW outer rings / CW holes put each curb's outward normal on
                # the right, including roads and recessed tree pits.
                for ring in [poly.exterior, *poly.interiors]:
                    q = _densify(np.asarray(ring.coords, float), 8.)
                    if not elevated: q = terrain_path(q, hf)
                    z = heights(q, path)
                    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
                    for j in range(len(q) - 1):
                        a, b = q[j], q[j + 1]
                        if np.linalg.norm(b - a) < 1e-5: continue
                        sink.quad((*a, z[j] - SIDEWALK_HEIGHT), (*b, z[j + 1] - SIDEWALK_HEIGHT),
                                  (*b, z[j + 1]), (*a, z[j]),
                                  np.array([7.5, 0., 0.], np.float32), 5., emissive=0.,
                                  uv=((arc[j], 0.), (arc[j+1], 0.),
                                      (arc[j+1], SIDEWALK_HEIGHT), (arc[j], SIDEWALK_HEIGHT)))
    return sink.mesh()
