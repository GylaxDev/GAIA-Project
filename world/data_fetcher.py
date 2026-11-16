"""Overpass fetching, parsing, and tree scatter."""
from __future__ import annotations

import gzip
import http.client
import itertools
import json
import math
import os
import queue
import re
import ssl
import threading
import time
import urllib.parse
import urllib.request
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

from ..core import telemetry as tm
from ..core.config import (ASSUME_EMPTY_TILE_IS_SEA, EARTH_RADIUS, KIND_DEFAULT_HEIGHT,
                     ROAD_SURFACE, ROAD_WIDTHS)
from ..core.projection import LocalProjection, lonlat_to_tile, tile_bounds
from .foliage import FOLIAGE_PALETTE
from .road_layout import road_width

try:
    import urllib3
except ImportError:
    urllib3 = None


@dataclass
class Building:
    id: int
    footprint: Polygon
    height: float
    min_height: float
    kind: str
    color: np.ndarray
    roof_color: np.ndarray
    material: str
    roof_shape: str = "flat"
    roof_height: float = 0.0
    roof_angle: float = 0.0
    roof_direction: float = 0.0
    layer: int = 0
    material_kind: int = -1
    shop: str = ""
    amenity: str = ""
    name: str = ""
    commercial: float = 0.0
    style: int = -1                       # pre-war facade archetype 5..8; -1 keeps the modern theme
    ground_elevation: float | None = None  # shared datum after a footprint is partitioned
    street_details: bool = True           # false for carved spans: no awnings in passages


@dataclass
class Road:
    highway: str
    geometry: LineString
    layer: int = 0
    bridge: bool = False
    tunnel: bool = False
    oneway: bool = False
    lanes: int = 0                  # zero means missing; explicit lanes=1 stays one lane
    name: str = ""
    structure: str = ""
    uid: int = 0
    speed_limit: float = 0.0        # tagged maxspeed in m/s; zero means hierarchy default
    lanes_forward: int = 0
    lanes_backward: int = 0
    turn_lanes: tuple = ()         # OSM left-to-right order, in the travel direction
    turn_lanes_backward: tuple = ()
    node_ids: tuple = ()           # aligned with geometry; zero for inserted/clip vertices
    signal_nodes: tuple = ()
    junction: str = ""
    drivable: bool = True
    covered: bool = False          # surface passage/arcade, not an underground tunnel
    clearance: float = 0.0         # signed maxheight in metres; zero uses a passage default
    level: float | None = None    # explicit indoor/garage floor, relative to ground
    width: float = 0.0            # tagged carriageway width; zero uses the shared lane layout
    sidewalks: tuple = ()         # optional (left, right) implicit bands; excludes separately mapped footways


@dataclass
class GreenSpace:
    kind: str
    geometry: Polygon
    material: float = 1.0   # NEW: 1=grass, 2=asphalt, 3=concrete, 4=dirt


@dataclass
class WaterArea:
    """Keep the water datum/type when a source outline is clipped to a tile.

    Coastal/tidal water shares the world sea datum, not a noisy bank DEM sample.
    ``elevation`` is an optional absolute OSM ele, converted by the mesh worker.
    """
    geometry: Polygon
    coastal: bool = False
    elevation: float | None = None
    wave_scale: float = 1.0


def water_area(geometry, tags=None, *, coastal=False):
    tags = tags or {}
    coastal = coastal or tags.get('tidal') in ('yes', 'true', '1') or tags.get('saltwater') == 'yes'
    coastal = coastal or tags.get('water') in ('sea', 'ocean', 'lagoon')
    elevation = None
    try:
        match = re.fullmatch(r"\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))\s*(m|ft|')?\s*",
                             str(tags.get('ele', '')), re.IGNORECASE)
        if match:
            elevation = float(match[1].replace(',', '.')) * (.3048 if (match[2] or '').lower() in ('ft', "'") else 1.)
            if not math.isfinite(elevation): elevation = None
    except (ValueError, TypeError):
        pass
    if coastal:
        scale = 1.
    elif tags.get('amenity') == 'fountain' or tags.get('water') in ('reflecting_pool', 'basin', 'pool'):
        scale = .06
    elif tags.get('waterway') in ('stream', 'canal'):
        scale = .18
    else:
        scale = float(np.clip(np.sqrt(geometry.area / 18000.), .08, .65))
    return WaterArea(geometry, bool(coastal), elevation, scale)


@dataclass
class RoadPath:
    points: np.ndarray
    arc:    np.ndarray
    total:  float
    highway: str
    width:  float
    oneway: bool = False
    lanes:  int = 0
    z:      np.ndarray | None = None   # per-point road surface height
    uid: int = 0
    layer: int = 0
    bridge: bool = False
    tunnel: bool = False
    speed_limit: float = 0.0
    lanes_forward: int = 0
    lanes_backward: int = 0
    turn_lanes: tuple = ()
    turn_lanes_backward: tuple = ()
    node_ids: tuple = ()
    signal_nodes: tuple = ()
    junction: str = ""
    covered: bool = False
    clearance: float = 0.0
    level: float | None = None
    sidewalks: tuple = ()
    name: str = ""                    # retained for navigation and map labels


DRIVABLE_HIGHWAYS = frozenset((
    "motorway", "trunk", "primary", "secondary", "tertiary", 
    "residential", "unclassified", "service", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link", "living_street"
))
PEDESTRIAN_HIGHWAYS = frozenset((
    "footway", "pedestrian", "path", "steps", "track", "cycleway", "bridleway"
))
# Kerbs belong to streets. A motorway shoulder or a slip road never parks a car,
# however wide its carriageway happens to be.
PARKABLE_HIGHWAYS = frozenset(h for h in DRIVABLE_HIGHWAYS
                              if h not in ("motorway", "trunk", "motorway_link", "trunk_link"))


# ---------------------------------------------------------------- helpers
def _jitter(bid: int) -> float:
    return ((bid * 2654435761) % (2 ** 32)) / 2 ** 32


_COMMERCIAL_AMENITIES = frozenset((
    "restaurant", "cafe", "bar", "pub", "fast_food", "food_court", "nightclub",
    "theatre", "cinema", "bank", "marketplace", "pharmacy", "hotel", "events_venue",
))


def commercial_score(tags, kind: str) -> float:
    """How much street-level commerce a building should advertise (0..1)."""
    score = 0.0
    if tags.get("shop"): score += 0.55
    amenity = (tags.get("amenity") or "").lower()
    if amenity in _COMMERCIAL_AMENITIES: score += 0.45
    if tags.get("tourism") in ("hotel", "attraction", "museum"): score += 0.25
    if tags.get("office") or tags.get("commercial"): score += 0.20
    if kind in ("commercial", "retail", "hotel", "kiosk", "supermarket"): score += 0.45
    if kind in ("office", "mixed"): score += 0.15
    if tags.get("building:levels"):
        try: score += min(0.15, float(tags["building:levels"]) * 0.01)
        except (TypeError, ValueError): pass
    return float(min(1.0, score))


def theme_code(kind: str) -> float:
    if kind == "office": return 0.0
    if kind in ("apartments", "residential", "house", "detached", "semidetached_house",
                "terrace", "dormitory", "bungalow", "cabin", "hut"): return 1.0
    if kind in ("commercial", "retail", "hotel", "shop", "kiosk", "supermarket"): return 2.0
    if kind in ("industrial", "warehouse", "factory", "manufacture"): return 3.0
    return 4.0


# Pre-war facade archetypes.  Shader themes 0..4 stay glass / brick / stone /
# metal / concrete; 5..8 add the older masonry city fabric (brick walk-up,
# beaux-arts limestone, art-deco pier strips, cast-iron loft front).
OLD_STYLE_KINDS = {
    "brownstone": 5, "tenement": 5, "rowhouse": 5, "row_house": 5,
    "civic": 6, "government": 6, "public": 6, "museum": 6, "train_station": 6,
    "castle": 6, "fort": 6, "monastery": 6, "cathedral": 6, "church": 6,
    "chapel": 6, "shrine": 6, "temple": 6, "mosque": 6, "synagogue": 6,
    "religious": 6,
    "tower": 7, "water_tower": 7,
    "factory": 8, "manufacture": 8, "brewery": 8, "warehouse": 8,
}
ARCHITECTURE_STYLES = {
    "art_deco": 7, "art_deco_style": 7, "deco": 7,
    "beaux_arts": 6, "beaux_arts_style": 6, "classical": 6, "neoclassical": 6,
    "neo_classical": 6, "romanesque": 6, "neo_romanesque": 6, "gothic": 6,
    "neo_gothic": 6, "baroque": 6, "renaissance": 6, "byzantine": 6,
    "victorian": 5, "edwardian": 5, "georgian": 5, "tudor": 5, "queen_anne": 5,
    "cast_iron": 8, "industrial": 8, "loft": 8,
}


def _built_year(tags):
    """First plausible construction year from the usual OSM date keys."""
    for key in ("year_built", "building:year_built", "start_date", "building:start_date",
                "construction_date", "opening_date"):
        match = re.search(r"(1[5-9]\d\d|20[0-2]\d)", str(tags.get(key) or ""))
        if match:
            return int(match[1])
    return None


def old_style_code(tags, kind: str, bid: int) -> int:
    """Pre-war facade archetype (5..8), or -1 to keep the modern theme.

    OSM rarely states a style, so tagged architecture wins over the mapped
    building type, which in turn wins over a dated construction year.  Untagged
    blocks only turn pre-war for a minority of apartment and commercial fabric,
    so a modern skyline is not repainted wholesale.
    """
    arch = re.sub(r"[^a-z]+", "_", (tags.get("building:architecture")
                                     or tags.get("architecture") or "").lower()).strip("_")
    if arch in ARCHITECTURE_STYLES:
        return ARCHITECTURE_STYLES[arch]
    code = OLD_STYLE_KINDS.get(kind)
    if code is not None:
        return code
    if tags.get("historic") or tags.get("heritage") or tags.get("building:heritage"):
        return 6
    year = _built_year(tags)
    if year is not None:
        if year < 1905:
            return 8 if kind in ("commercial", "retail", "warehouse", "industrial") else 5
        if year < 1945:
            return 7 if kind in ("office", "apartments", "hotel", "commercial") else 5
        return -1
    j = _jitter(bid * 3 + 11)
    if kind in ("apartments", "hotel") and j < .26:
        return 7
    if kind in ("commercial", "retail") and j < .32:
        return 8
    return -1


def classify_material(tags, kind, bid):
    mat = (tags.get("building:material") or tags.get("material") or "").lower()
    if "glass" in mat or "mirror" in mat or "window" in mat: return 0
    if "brick" in mat or "terracotta" in mat or "tile" in mat: return 1
    if any(w in mat for w in ("stone", "granite", "marble", "limestone", "plaster", "concrete")):
        return 2 if "concrete" not in mat else 4
    if any(w in mat for w in ("metal", "steel", "copper", "aluminium", "aluminum")): return 3
    j = _jitter(bid)
    if kind == "office": return 0 if j < 0.7 else 4
    if kind in ("apartments", "residential", "house", "detached", "semidetached_house",
                "terrace", "dormitory", "bungalow", "cabin", "hut"):
        return 1 if j < 0.75 else 4
    if kind in ("commercial", "retail", "hotel", "shop", "kiosk", "supermarket"):
        return 2 if j < 0.6 else 0
    if kind in ("industrial", "warehouse", "factory", "manufacture"): return 3
    return 4


_VALID_ROOFS = {"flat", "pyramidal", "hipped", "gabled", "dome", "skillion", "half-hipped"}


def parse_roof_tags(tags):
    shape = (tags.get("roof:shape") or "flat").lower()
    if shape not in _VALID_ROOFS: shape = "flat"
    rh = parse_length(tags.get("roof:height")) or 0.0
    ra = parse_length(tags.get("roof:angle")) or 0.0
    rd = parse_length(tags.get("roof:direction")) or 0.0
    return shape, float(rh), float(ra), float(rd)


def parse_color(val):
    if not val: return None
    val = val.lower()
    named = {
        "white": [0.95, 0.95, 0.95], "black": [0.1, 0.1, 0.1], "red": [0.8, 0.2, 0.2],
        "brown": [0.5, 0.3, 0.2], "grey": [0.5, 0.5, 0.5], "gray": [0.5, 0.5, 0.5],
        "yellow": [0.9, 0.8, 0.2], "green": [0.2, 0.6, 0.2], "blue": [0.2, 0.4, 0.8],
        "beige": [0.85, 0.80, 0.75], "cream": [0.9, 0.85, 0.75],
    }
    if val in named: return np.array(named[val], np.float32)
    if val.startswith("#") and len(val) >= 7:
        try: return np.array([int(val[i:i+2], 16) / 255.0 for i in (1, 3, 5)], np.float32)
        except ValueError: pass
    return None


def parse_length(value):
    if not value: return None
    m = re.match(r"\s*(\d+(?:[.,]\d+)?)\s*(m|ft|')?", str(value).lower())
    if not m: return None
    n = float(m.group(1).replace(",", "."))
    return n * 0.3048 if m.group(2) in ("ft", "'") else n


def parse_maxspeed(value):
    """OSM numeric maxspeed, including mph/knots and semicolon alternatives."""
    if value is None:
        return 0.0
    speeds = []
    for part in str(value).lower().split(';'):
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph|knots)?\s*', part)
        if match:
            number = float(match[1])
            factor = .44704 if match[2] == 'mph' else .514444 if match[2] == 'knots' else 1 / 3.6
            if number > 0:
                speeds.append(number * factor)
    return min(speeds) if speeds else 0.0


def _lane_count(value, default=0):
    try:
        return max(0, min(8, int(value)))
    except (TypeError, ValueError):
        return default


def _node_ids_at(road, points):
    """Retain original OSM identity through clipping and surface densification."""
    if not road.node_ids:
        return ()
    lookup = {tuple(np.round(p, 6)): int(node) for p, node in
              zip(road.geometry.coords, road.node_ids) if node}
    return tuple(lookup.get(tuple(np.round(p, 6)), 0) for p in points)


def _parts(geom):
    return list(getattr(geom, "geoms", [geom]))


def _clean_coords(coords):
    if len(coords) < 3: return coords
    keep = np.ones(len(coords), dtype=bool)
    keep[1:] = np.linalg.norm(coords[1:] - coords[:-1], axis=1) > 0.01
    coords = coords[keep]
    if len(coords) < 3: return coords
    p0 = coords
    p1 = np.roll(coords, -1, axis=0)
    p2 = np.roll(coords, -2, axis=0)
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p1[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p1[:, 0])
    keep = np.abs(cross) > 1e-4
    return coords[keep]


def building_heights(tags, kind, bid, default_height=9.0, level_height=3.2):
    h = parse_length(tags.get("height"))
    if h is None:
        lv = parse_length(tags.get("building:levels", tags.get("levels")))
        if lv: h = lv * level_height + (parse_length(tags.get("roof:levels")) or 0) * 1.5
    if h is None:
        base = KIND_DEFAULT_HEIGHT.get(kind, default_height)
        h = base * (0.85 + 0.30 * _jitter(bid))
    h0 = parse_length(tags.get("min_height"))
    if h0 is None:
        ml = parse_length(tags.get("building:min_level"))
        h0 = ml * level_height if ml else 0.0
    h = min(h, 900.0)
    h0 = min(max(h0, 0.0), h - 1.0)
    return max(h, h0 + 1.0), h0


# ---------------------------------------------------------------- geometry
def _geojson_rings(geom, out):
    if not isinstance(geom, dict): return
    gtype = geom.get("type")
    if gtype == "GeometryCollection":
        for sub in geom.get("geometries") or (): _geojson_rings(sub, out)
        return
    coords = geom.get("coordinates")
    if not coords: return
    if gtype == "LineString": out.append(np.asarray(coords, np.float64))
    elif gtype == "Polygon": out.extend(np.asarray(r, np.float64) for r in coords)
    elif gtype == "MultiPolygon":
        for poly in coords: out.extend(np.asarray(r, np.float64) for r in poly)


def _local_rings(rings, proj):
    out = []
    for r in rings:
        r = np.asarray(r, np.float64)
        if len(r) < 2: continue
        out.append(proj.to_local(r[:, 1], r[:, 0]))
    return out


def _polygonized_shells(rings):
    # OSM relation boundaries may be fragmented into two-point member ways.
    lines = [LineString(r) for r in rings if len(r) >= 2]
    if not lines: return []
    shells = {}
    for face in polygonize(_parts(unary_union(lines))):
        shell = Polygon(face.exterior)
        if not shell.is_empty:
            shells[shapely.normalize(shell).wkb] = shell
    return sorted(shells.values(), key=lambda p: p.area, reverse=True)


def _stitch_polygons(rings):
    """Role-less boundaries use nesting parity, never polygonized hole faces.

    GEOS emits both an annulus and its inner face. Returning both fills the
    courtyard again; classify the *shells* before subtracting direct children.
    """
    shells = _polygonized_shells(rings)
    parents, depths = [], []
    for i, shell in enumerate(shells):
        containing = [j for j in range(i) if shells[j].covers(shell)]
        parent = min(containing, key=lambda j: shells[j].area) if containing else -1
        parents.append(parent)
        depths.append(depths[parent] + 1 if parent >= 0 else 0)
    out = []
    for i, shell in enumerate(shells):
        if depths[i] % 2: continue
        holes = [q for j, q in enumerate(shells) if parents[j] == i]
        out.extend(polygons_of(shell.difference(unary_union(holes)) if holes else shell))
    return out


def _relation_polygons(members, proj):
    by_role = {"outer": [], "inner": [], "": []}
    for member in members:
        role = member.get("role", "")
        if role == "outline": role = "outer"
        if role not in by_role: continue  # building parts/labels are not outlines
        geom, rings = member.get("geometry"), []
        if isinstance(geom, dict): _geojson_rings(geom, rings)
        elif geom:
            rings.append(np.array([[p["lon"], p["lat"]] for p in geom], np.float64))
        by_role[role].extend(_local_rings(rings, proj))
    if not by_role["outer"] and not by_role["inner"]:
        return _stitch_polygons(by_role[""])
    outers = _polygonized_shells(by_role["outer"] + by_role[""])
    inners = _polygonized_shells(by_role["inner"])
    holes = [[] for _ in outers]
    for inner in inners:
        containing = [i for i, outer in enumerate(outers) if outer.covers(inner)]
        if containing:
            # Assign to the smallest containing outer: islands inside a courtyard
            # remain buildings, while a dangling inner never becomes a building.
            holes[min(containing, key=lambda i: outers[i].area)].append(inner)
    out = []
    for outer, inner in zip(outers, holes):
        out.extend(polygons_of(outer.difference(unary_union(inner)) if inner else outer))
    return out


FOOTPRINT_SIMPLIFY_TOL = 0.1


def polygons_of(geom):
    if isinstance(geom, WaterArea): geom = geom.geometry
    if geom is None or geom.is_empty: return []
    t = geom.geom_type
    if t == "Polygon": return [geom]
    if t in ("MultiPolygon", "GeometryCollection"):
        out = []
        for g in geom.geoms: out.extend(polygons_of(g))
        return out
    return []


def strip_ring(coords, min_edge=0.01, collinear_eps=1e-3):
    c = np.asarray(coords, np.float64)
    if c.ndim != 2 or len(c) == 0: return np.empty((0, 2), np.float64)
    c = c[:, :2]
    if len(c) > 1 and c[0, 0] == c[-1, 0] and c[0, 1] == c[-1, 1]: c = c[:-1]
    for _ in range(8):
        if len(c) < 3: return c
        nxt = np.roll(c, -1, axis=0)
        keep = np.linalg.norm(nxt - c, axis=1) > min_edge
        if not keep.all():
            c = c[keep]; continue
        prv = np.roll(c, 1, axis=0)
        e1, e2 = c - prv, nxt - c
        cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        chord = np.maximum(np.linalg.norm(nxt - prv, axis=1), 1e-12)
        keep = (np.abs(cross) / chord) > collinear_eps
        if keep.all(): break
        c = c[keep]
    return c


def _valid_geom(g):
    if g is None or g.is_empty: return Polygon()
    try:
        if g.is_valid: return g
        return make_valid(g)
    except Exception:
        try: return g.buffer(0)
        except Exception: return Polygon()


def _rebuild_polygon(poly):
    shell = strip_ring(poly.exterior.coords)
    if len(shell) < 3: return None
    holes = [h for h in (strip_ring(r.coords) for r in poly.interiors) if len(h) >= 3]
    return Polygon(shell, holes)


def sanitize_footprint(geom, simplify_tol=FOOTPRINT_SIMPLIFY_TOL, min_area=1.0):
    out = []
    for g in polygons_of(_valid_geom(geom)):
        try:
            sg = g.simplify(simplify_tol, preserve_topology=True) if simplify_tol > 0 else g
        except Exception:
            sg = g
        for sp in polygons_of(_valid_geom(sg)):
            rb = _rebuild_polygon(sp)
            if rb is None: continue
            for q in polygons_of(_valid_geom(rb)):
                if q.area >= min_area: out.append(orient(q, 1.0))
    return out


def _clean_polygons(polys, min_area=4.0):
    out = []
    for p in polys:
        out.extend(sanitize_footprint(p, FOOTPRINT_SIMPLIFY_TOL, min_area))
    return out


def _validate_polys(polys, max_dist=5000.0):
    out = []
    for p in polys:
        if p.is_empty: continue
        valid = True
        for ring in [p.exterior] + list(p.interiors):
            coords = np.array(ring.coords)
            if len(coords) == 0 or not np.isfinite(coords).all():
                valid = False; break
            if np.max(np.hypot(coords[:, 0], coords[:, 1])) > max_dist:
                valid = False; break
        if valid: out.append(p)
    return out


def item_polygons(el, proj):
    geom = el.get("geometry")
    if isinstance(geom, dict):
        gtype = geom.get("type")
        if gtype in ("Polygon", "MultiPolygon"):
            polys_rings = ([geom.get("coordinates") or ()] if gtype == "Polygon"
                           else list(geom.get("coordinates") or ()))
            built = []
            for poly_rings in polys_rings:
                rr = _local_rings(list(poly_rings), proj)
                if rr: built.append(Polygon(rr[0], rr[1:]))
            return _validate_polys(_clean_polygons(built))
        rings = []
        _geojson_rings(geom, rings)
        return _validate_polys(_clean_polygons(_stitch_polygons(_local_rings(rings, proj))))

    members = el.get("members")
    if members:
        return _validate_polys(_clean_polygons(_relation_polygons(members, proj)))

    if isinstance(geom, list):
        try:
            pts = np.array([[p["lon"], p["lat"]] for p in geom], np.float64)
            if len(pts) >= 4:
                local = proj.to_local(pts[:, 1], pts[:, 0])
                return _validate_polys(_clean_polygons([Polygon(local)]))
        except (KeyError, TypeError):
            pass
        return _validate_polys(_clean_polygons(_stitch_polygons(_local_rings(
            [np.array([[p["lon"], p["lat"]] for p in geom if "lon" in p and "lat" in p], np.float64)], proj))))

    return []


def owns_tile(poly, proj, z, tx, ty) -> bool:
    p = poly.representative_point()
    lat, lon = proj.to_latlon(p.x, p.y)
    return lonlat_to_tile(lat, lon, z) == (tx, ty)


# ---------------------------------------------------------------- tree scatter
def _icosphere(subdiv=1):
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], np.int64)
    for _ in range(subdiv):
        mid_cache, verts_list = {}, list(verts)
        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key in mid_cache: return mid_cache[key]
            m = verts[a] + verts[b]; m /= np.linalg.norm(m)
            idx = len(verts_list); verts_list.append(m); mid_cache[key] = idx
            return idx
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        verts = np.array(verts_list, np.float64); faces = np.array(nf, np.int64)
    return verts, faces


def build_tree_template(level=0):
    """Compatibility entry point for the shared branching broadleaf tree LODs."""
    from .tree_models import build_tree_template as build
    return build(level)


def scatter_trees_in_polygon(polygon, density=0.008, rng=None):
    if polygon.is_empty or polygon.area < 40.0: return np.empty((0, 2), np.float64)
    rng = rng or np.random.default_rng()
    minx, miny, maxx, maxy = polygon.bounds
    spacing = 1.0 / math.sqrt(max(density, 1e-5))
    xs = np.arange(minx + spacing * 0.5, maxx, spacing)
    ys = np.arange(miny + spacing * 0.5, maxy, spacing)
    if len(xs) == 0 or len(ys) == 0: return np.empty((0, 2), np.float64)
    xx, yy = np.meshgrid(xs, ys)
    xx = xx.ravel() + rng.uniform(-spacing * 0.35, spacing * 0.35, xx.size)
    yy = yy.ravel() + rng.uniform(-spacing * 0.35, spacing * 0.35, yy.size)
    if hasattr(shapely, "contains_xy"): inside = shapely.contains_xy(polygon, xx, yy)
    else: inside = np.array([polygon.contains(Point(x, y)) for x, y in zip(xx, yy)])
    return np.stack([xx[inside], yy[inside]], axis=1)


def _tree_inst(x, y, rng, z=0.0):
    # Linear dusty-green reflectance. The shader also corrects legacy cached
    # tints; leaf texture modulation must not be gamma-decoded or clamped away.
    base = FOLIAGE_PALETTE[rng.integers(0, len(FOLIAGE_PALETTE))]
    tint = base * rng.uniform(0.88, 1.10)
    tint *= np.array([rng.uniform(0.97, 1.06),
                      rng.uniform(0.98, 1.03),
                      rng.uniform(0.92, 1.04)])
    return (float(x), float(y), float(z), float(rng.uniform(0.85, 1.35)),
            float(rng.uniform(0.0, 2.0 * math.pi)),
            float(tint[0]), float(tint[1]), float(tint[2]))


def build_trees_mesh(trees, hf=None):
    if trees is None or len(trees) == 0: return None
    rng = np.random.default_rng(42)
    pts = np.asarray(trees, np.float64)
    if pts.ndim != 2 or len(pts) == 0: return None
    zs = hf.at(pts[:, 0], pts[:, 1]) if hf is not None else np.zeros(len(pts), np.float32)
    return np.array([_tree_inst(float(pts[i, 0]), float(pts[i, 1]), rng, float(zs[i]))
                     for i in range(len(pts))], np.float32)


# ---------------------------------------------------------------- ocean
def build_ocean(coast, bbox):
    x0, y0, x1, y1 = bbox
    tile = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    try:
        clipped, wx, wy, lx, ly = [], [], [], [], []
        for pts in coast:
            if len(pts) < 2: continue
            a, b = pts[:-1], pts[1:]
            d = b - a
            ln = np.hypot(d[:, 0], d[:, 1])
            ok = ln > 1e-3
            a, b, d, ln = a[ok], b[ok], d[ok], ln[ok]
            if len(a):
                mid = (a + b) * 0.5
                inside = ((mid[:, 0] > x0) & (mid[:, 0] < x1) &
                          (mid[:, 1] > y0) & (mid[:, 1] < y1))
                if inside.any():
                    m, dd, l = mid[inside], d[inside], ln[inside]
                    eps = np.minimum(0.75, 0.25 * l)
                    nx, ny = dd[:, 1] / l, -dd[:, 0] / l
                    wx.append(m[:, 0] + nx * eps); wy.append(m[:, 1] + ny * eps)
                    lx.append(m[:, 0] - nx * eps); ly.append(m[:, 1] - ny * eps)
            c = LineString(pts).intersection(tile)
            for g in getattr(c, "geoms", [c]):
                if g.geom_type == "LineString" and g.length > 0.01: clipped.append(g)
        if not clipped or not wx: return []
        wx, wy = np.concatenate(wx), np.concatenate(wy)
        lx, ly = np.concatenate(lx), np.concatenate(ly)
        pieces = list(polygonize(unary_union(clipped + [tile.boundary])))
        water = []
        for p in pieces:
            if p.is_empty or p.area < 1.0: continue
            w_votes = int(shapely.contains_xy(p, wx, wy).sum())
            l_votes = int(shapely.contains_xy(p, lx, ly).sum())
            if w_votes > l_votes: water.append(p)
        return water
    except Exception as exc:
        print(f"[ocean] could not build ocean polygons: {exc}")
        return []


# ---------------------------------------------------------------- trajectories
def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bridge_clearance(road, total=0.0, over_water=False) -> float:
    """Deck height above the ground/water for a bridge road.

    Depends only on the bridge's own tags (and its way id) so the deck height
    stays continuous across chunk-tile boundaries, where ways are clipped.
    """
    structure = (getattr(road, "structure", "") or "").lower()
    name = (getattr(road, "name", "") or "").lower()
    if "golden gate" in name:          return 60.0
    if "bay bridge" in name:           return 54.0
    if "george washington" in name:    return 52.0
    if "brooklyn" in name:             return 40.0
    if "suspension" in structure:      return 58.0
    if "cable" in structure:           return 50.0
    if "arch" in structure:            return 22.0
    if "viaduct" in structure:         return 18.0
    if total > 600.0 and road.highway in ("motorway", "trunk", "primary"):
        # Long major crossings sit well above the water.  Depends only on tags
        # and length class so parallel carriageways and tiles agree.
        return 26.0 + 2.0 * max(0, road.layer)
    if road.highway in ("motorway", "trunk"):
        return 9.0 if road.layer >= 1 else 7.0
    if road.highway in ("primary", "secondary"):
        return 7.5 if road.layer >= 1 else 6.0
    return 4.5 + 1.5 * max(0, road.layer)


def road_profile(pts, road, hf=None, sea_z=0.0, bbox=None):
    """Arc length + per-point road surface height for one alignment."""
    pts = np.asarray(pts, np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(arc[-1])
    if hf is None:
        base = np.zeros(len(pts), np.float64)
    else:
        base = np.asarray(hf.at(pts[:, 0], pts[:, 1]), np.float64)
    base = np.maximum(base, float(sea_z))
    zs = base + ROAD_SURFACE
    level = getattr(road, 'level', None)
    if road.tunnel or (level is not None and level < 0.):
        # Underground driving routes remain connected, but must not place cars
        # on top of the surface street/building. Floor depth is an approximation
        # when OSM supplies no surveyed elevation. Portals meet the surface.
        depth = max(4., -3.2 * level) if level is not None else max(4., -3.2 * road.layer)
        ramp = min(40., max(4., total * .3))
        a, b = _smoothstep(arc / ramp), _smoothstep((total - arc) / ramp)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            def on_boundary(p):
                return p[0] < x0 + 2. or p[0] > x1 - 2. or p[1] < y0 + 2. or p[1] > y1 - 2.
            if on_boundary(pts[0]): a = np.ones(len(pts))
            if on_boundary(pts[-1]): b = np.ones(len(pts))
        zs -= depth * a * b
        return arc, zs
    if road.bridge or road.layer > 0:
        clearance = bridge_clearance(road, total)
        # A short span over a canal or street must not sit 6-9 m up: that is what
        # made small bridges look like floating blocks with vertical ends.
        if not (getattr(road, "structure", "") or "") and total < 250.0:
            if road.highway in ("motorway", "trunk"):
                pass
            elif road.highway in ("primary", "secondary"):
                clearance = min(clearance, 3.5 + 0.05 * total)
            else:
                clearance = min(clearance, 2.4 + 0.04 * total)
        start_ok = end_ok = True
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            m = 2.0      # ways are clipped exactly at the tile edge
            p0, p1 = pts[0], pts[-1]
            start_ok = not (p0[0] < x0 + m or p0[0] > x1 - m or
                            p0[1] < y0 + m or p0[1] > y1 - m)
            end_ok = not (p1[0] < x0 + m or p1[0] > x1 - m or
                          p1[1] < y0 + m or p1[1] > y1 - m)
        ramp = float(np.clip(total * 0.30, 6.0, 200.0))
        a = _smoothstep(arc / ramp) if start_ok else np.ones(len(pts))
        b = _smoothstep((total - arc) / ramp) if end_ok else np.ones(len(pts))
        zs = np.maximum(zs, base + clearance * a * b)
    return arc, zs


def _traffic_points(road, shared_nodes):
    """Thin anonymous/shape samples without truncating a real OSM connection."""
    pts = np.asarray(road.geometry.coords, np.float64)
    ids = _node_ids_at(road, pts) or (0,) * len(pts)
    signals = set(road.signal_nodes)
    pinned = [i in (0, len(pts) - 1) or (nid and (nid in shared_nodes or nid in signals))
              for i, nid in enumerate(ids)]
    keep = [0]
    for i in range(1, len(pts)):
        if pinned[i]:
            # A short final segment used to drop the shared endpoint entirely.
            # Replace an insignificant nearby shape sample, not the endpoint or
            # another junction/signal. Distance is from the LAST retained point.
            while (len(keep) > 1 and not pinned[keep[-1]]
                   and np.linalg.norm(pts[i] - pts[keep[-1]]) <= .5):
                keep.pop()
        if pinned[i] or np.linalg.norm(pts[i] - pts[keep[-1]]) > .5:
            keep.append(i)
    return pts[keep]


def road_paths_from_roads(roads, min_length=2.0, hf=None, sea_z=0.0, bbox=None):
    roads = list(roads)
    owners = {}
    for r in roads:
        if r.highway not in DRIVABLE_HIGHWAYS or not r.drivable: continue
        for node in set(r.node_ids) - {0}:
            owners[node] = owners.get(node, 0) + 1
    shared_nodes = {node for node, count in owners.items() if count > 1}
    out, seen = [], set()
    for r in roads:
        if r.highway not in DRIVABLE_HIGHWAYS or not r.drivable: continue
        pts = np.asarray(r.geometry.coords, np.float64)
        if len(pts) < 2: continue
        pts = _traffic_points(r, shared_nodes)
        if len(pts) < 2: continue
        if r.tunnel or (r.level is not None and r.level < 0.):
            pts = _densify(pts)
        a = tuple(np.round(pts[0], 1)); b = tuple(np.round(pts[-1], 1))
        ends = (a, b) if a <= b else (b, a)
        key = (r.uid, r.layer, r.bridge, r.tunnel, r.oneway, ends,
               tuple(np.round(pts[len(pts)//2], 1)))
        if key in seen: continue
        seen.add(key)

        arc, zs = road_profile(pts, r, hf, sea_z, bbox)
        total = float(arc[-1])
        if total < min_length: continue

        out.append(RoadPath(pts, arc, total, r.highway, road_width(r),
                            oneway=r.oneway, lanes=r.lanes, z=zs, uid=r.uid,
                            layer=r.layer, bridge=r.bridge, tunnel=r.tunnel,
                            speed_limit=r.speed_limit, lanes_forward=r.lanes_forward,
                            lanes_backward=r.lanes_backward, turn_lanes=r.turn_lanes,
                            turn_lanes_backward=r.turn_lanes_backward,
                            node_ids=_node_ids_at(r, pts), signal_nodes=r.signal_nodes,
                            junction=r.junction, covered=r.covered,
                            clearance=r.clearance, level=r.level, sidewalks=getattr(r, 'sidewalks', ()),
                            name=r.name))
    return out


def parking_slots_from_paths(paths, spacing=6.5, side_clearance=1.0, sides=(1.0, -1.0)):
    """Kerb bays along `paths`; `sides` picks one kerb (1 = right of travel)."""
    out = []
    for p in paths:
        if p.highway not in PARKABLE_HIGHWAYS: continue
        if p.tunnel or p.covered or (p.level is not None and p.level < 0.): continue
        if p.width < 6.0: continue
        offset = p.width * 0.5 - side_clearance
        if offset < 1.5: continue
        s = spacing * 0.5
        last = p.total - spacing * 0.5
        while s < last:
            i = int(np.searchsorted(p.arc, s)) - 1
            i = max(0, min(i, len(p.points) - 2))
            seg_len = max(p.arc[i + 1] - p.arc[i], 1e-6)
            t = (s - p.arc[i]) / seg_len
            a, b = p.points[i], p.points[i + 1]
            pos = a + (b - a) * t
            d = b - a
            ln = np.linalg.norm(d)
            if ln < 1e-6:
                s += spacing; continue
            d /= ln
            n = np.array([d[1], -d[0]])
            z = sample_path_z(p, s)
            for sign in sides:
                sl = pos + n * offset * sign
                if abs(sl[0]) > 5000.0 or abs(sl[1]) > 5000.0: continue
                hd = d * sign
                out.append((float(sl[0]), float(sl[1]), float(hd[0]), float(hd[1]), z))
            s += spacing
    return out


def sample_path_position(path, arc):
    # Imported ways are open; total is the final endpoint, never a wrap to start.
    arc = float(np.clip(arc, 0., path.total))
    i = int(np.searchsorted(path.arc, arc)) - 1
    i = max(0, min(i, len(path.points) - 2))
    seg_len = max(path.arc[i + 1] - path.arc[i], 1e-6)
    t = (arc - path.arc[i]) / seg_len
    a, b = path.points[i], path.points[i + 1]
    pos = a + (b - a) * t
    d = b - a
    ln = np.linalg.norm(d)
    heading = math.atan2(d[1], d[0]) if ln > 1e-6 else 0.0
    return float(pos[0]), float(pos[1]), float(heading)


def sample_path_z(path, arc):
    zs = getattr(path, "z", None)
    if zs is None or len(zs) == 0: return 0.0
    if len(zs) == 1: return float(zs[0])
    arc = float(np.clip(arc, 0., path.total))
    i = int(np.searchsorted(path.arc, arc)) - 1
    i = max(0, min(i, len(zs) - 2))
    seg_len = max(path.arc[i + 1] - path.arc[i], 1e-6)
    t = (arc - path.arc[i]) / seg_len
    return float(zs[i] + (zs[i + 1] - zs[i]) * t)


# ---------------------------------------------------------------- parsing
def _densify(pts, step=8.0):
    """Insert intermediate vertices along long segments."""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2 or step <= 0.0: return pts
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        ln = float(np.hypot(d[0], d[1]))
        k = int(ln // step)
        if k > 1:
            t = (np.arange(1, k, dtype=np.float64) / k)[:, None]
            out.append(a + d * t)
        out.append(b[None, :])
    return np.vstack(out)


def clip_roads_to_bbox(roads, bbox, densify=8.0):
    """Clip road centrelines to the tile and resample long segments.

    Ways are fetched with a 5 km radius, so without clipping a ribbon can span
    kilometres in one quad and a bridge deck becomes a giant inclined plane
    over the neighbouring tiles.  Resampling keeps draped surfaces following
    the ground instead of interpolating across a whole segment.
    """
    if bbox is None: return roads
    x0, y0, x1, y1 = (float(v) for v in bbox)
    box = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    out = []
    for r in roads:
        pts = np.asarray(r.geometry.coords, np.float64)
        if len(pts) < 2: continue
        if (pts[:, 0].max() < x0 or pts[:, 0].min() > x1
                or pts[:, 1].max() < y0 or pts[:, 1].min() > y1):
            continue
        try:
            cut = LineString(pts).intersection(box)
        except Exception:
            cut = LineString(pts)
        for g in getattr(cut, "geoms", [cut]):
            if g.geom_type != "LineString" or g.length < 0.5: continue
            q = _densify(np.asarray(g.coords, np.float64), densify)
            if len(q) < 2: continue
            out.append(Road(r.highway, LineString(q), r.layer, r.bridge, r.tunnel,
                            r.oneway, r.lanes, name=r.name, structure=r.structure,
                            uid=r.uid, speed_limit=r.speed_limit,
                            lanes_forward=r.lanes_forward, lanes_backward=r.lanes_backward,
                            turn_lanes=r.turn_lanes, turn_lanes_backward=r.turn_lanes_backward,
                            node_ids=_node_ids_at(r, q), signal_nodes=r.signal_nodes,
                            junction=r.junction, drivable=r.drivable,
                            covered=r.covered, clearance=r.clearance, level=r.level,
                            width=getattr(r, 'width', 0.), sidewalks=getattr(r, 'sidewalks', ())))
    return out


def _points_clear_of(points, polys):
    """Boolean mask of points that are *not* inside any of the polygons."""
    pts = np.asarray(points, np.float64)
    ok = np.ones(len(pts), bool)
    if not polys or len(pts) == 0: return ok
    try:
        arr = np.empty(len(polys), dtype=object); arr[:] = polys
        tree = shapely.STRtree(arr)
        idx, _ = tree.query(shapely.points(pts[:, 0], pts[:, 1]), predicate="intersects")
        ok[np.unique(idx)] = False
    except Exception:
        for i, (x, y) in enumerate(pts):
            pt = Point(float(x), float(y))
            if any(p.contains(pt) for p in polys): ok[i] = False
    return ok


def _points_in_polys(points, polys):
    return ~_points_clear_of(points, polys)


def _avenue_trees(roads, spacing=17.0, offset=2.4):
    """Street trees on the verge beside drivable roads (never on the carriageway)."""
    out = []
    for r in roads:
        if r.highway not in ("primary", "secondary", "tertiary", "residential"): continue
        if r.bridge or r.tunnel or r.covered or (r.level is not None and r.level < 0.): continue
        pts = np.asarray(r.geometry.coords, np.float64)
        if len(pts) < 2: continue
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(seg)))
        total = float(arc[-1])
        if total < spacing * 2.0: continue
        s = np.arange(spacing, total - 1.0, spacing)
        i = np.clip(np.searchsorted(arc, s) - 1, 0, len(pts) - 2)
        d = pts[i + 1] - pts[i]
        ln = np.linalg.norm(d, axis=1)
        good = ln > 1e-6
        if not good.any(): continue
        i, d, ln = i[good], d[good], ln[good]
        t = (s[good] - arc[i]) / ln
        pos = pts[i] + d * t[:, None]
        nx, ny = d[:, 1] / ln, -d[:, 0] / ln
        side = np.where(np.arange(len(pos)) % 2 == 0, 1.0, -1.0)
        off = road_width(r) * 0.5 + offset
        out.append(np.stack([pos[:, 0] + nx * off * side,
                             pos[:, 1] + ny * off * side], axis=1))
    if not out: return np.empty((0, 2), np.float64)
    return np.vstack(out)


def parse_sidewalks(tags, highway):
    """Source-relative sides; separately mapped sidewalks already have ribbons."""
    default = highway not in ('motorway', 'motorway_link')
    common = str(tags.get('sidewalk:both', tags.get('sidewalk', ''))).strip().lower()
    sides = [default, default]
    if common in ('yes', 'both', 'left', 'right', 'no', 'none', 'separate'):
        sides = [common in ('yes', 'both', 'left'), common in ('yes', 'both', 'right')]
    for index, side in enumerate(('left', 'right')):
        value = str(tags.get(f'sidewalk:{side}', '')).strip().lower()
        if value in ('yes', 'no', 'none', 'separate'):
            sides[index] = value == 'yes'
    return tuple(sides)


def parse_tile_elements(raw, proj, owner=None, default_height=9.0, level_height=3.2):
    elements = raw.get("elements", ())
    bbox = tile = None
    if owner is not None:
        s, w, n, e = tile_bounds(*owner)
        sw, ne = proj.to_local(s, w), proj.to_local(n, e)
        bbox = (float(sw[0]), float(sw[1]), float(ne[0]), float(ne[1]))
        tile = shapely.box(*bbox)
    signal_nodes = {int(el['id']) for el in elements if el.get('type') == 'node'
                    and (el.get('tags') or {}).get('highway') == 'traffic_signals'}
    bridge_meta = {}
    building_members = set()
    relation_footprints = {}
    for el in elements:
        if el.get("type") != "relation": continue
        tags = el.get("tags") or {}
        if "building" in tags or "building:part" in tags:
            polys = item_polygons(el, proj)
            relation_footprints[el.get('id', 0)] = polys
            if polys:
                building_members.update(m['ref'] for m in el.get('members', ())
                                        if m.get('type') == 'way' and m.get('ref') is not None
                                        and m.get('role', '') in ('outer', 'outline', ''))
        if tags.get("man_made") != "bridge" and "bridge" not in tags: continue
        info = (tags.get("name", ""), tags.get("bridge:structure", ""))
        for m in el.get("members") or ():
            if m.get("type") == "way" and m.get("ref") is not None:
                bridge_meta[m["ref"]] = info
    buildings, roads, greens, waters, trees = [], [], [], [], []
    tree_buildings, scattered = [], []
    coast = []
    for el in elements:
        tags = el.get("tags") or el
        eid = el.get("id", 0)

        if "building" in tags or "building:part" in tags:
            if el.get('type') == 'way' and eid in building_members: continue
            # A subsurface station/garage footprint is not a surface podium.
            # Negative layer alone is not proof: retain sunken/elevated buildings.
            if tags.get('location') in ('underground', 'underwater') or tags.get('underground') == 'yes': continue
            is_part = tags.get("building:part", "no") != "no"
            kind = tags.get("building:part") or tags.get("building") or "yes"
            if kind in ("no",): continue
            polys = relation_footprints.get(eid, ()) if el.get('type') == 'relation' else item_polygons(el, proj)
            if not polys: continue
            # A neighboring tile may own the building mesh, but its footprint
            # can still cover tree roots on this side of the tile boundary.
            tree_buildings.extend(polys)
            c_tag = parse_color(tags.get("building:colour") or tags.get("building:color"))
            r_tag = parse_color(tags.get("roof:colour") or tags.get("roof:color"))
            if c_tag is None:
                h = _jitter(eid)
                if h < 0.20:  c_tag = np.array([0.85, 0.80, 0.75])
                elif h < 0.40: c_tag = np.array([0.75, 0.40, 0.30])
                elif h < 0.60: c_tag = np.array([0.20, 0.25, 0.30])
                elif h < 0.80: c_tag = np.array([0.60, 0.60, 0.65])
                else:          c_tag = np.array([0.95, 0.95, 0.95])
            if r_tag is None: r_tag = c_tag * 0.6
            roof_shape, roof_h, roof_angle, roof_dir = parse_roof_tags(tags)
            mat_kind = classify_material(tags, kind, eid)
            for i, poly in enumerate(polys):
                if owner is not None and not owns_tile(poly, proj, *owner): continue
                h, h0 = building_heights(tags, kind, eid + i, default_height, level_height)
                if roof_h <= 0.0 and roof_shape != "flat":
                    if roof_shape in ("pyramidal", "hipped", "gabled", "half-hipped"):
                        roof_h = max(2.5, min(12.0, h * 0.18))
                    elif roof_shape == "dome":
                        roof_h = max(3.0, min(18.0, h * 0.22))
                    elif roof_shape == "skillion":
                        roof_h = max(1.5, min(6.0, h * 0.08))
                buildings.append(Building(
                    eid * 16 + i, poly, h, h0, kind, c_tag, r_tag,
                    tags.get("building:material", ""), roof_shape=roof_shape,
                    roof_height=roof_h, roof_angle=roof_angle, roof_direction=roof_dir,
                    layer=1 if is_part else 0, material_kind=mat_kind,
                    style=old_style_code(tags, kind, eid + i),
                    shop=(tags.get("shop") or ""),
                    amenity=(tags.get("amenity") or ""),
                    name=(tags.get("name") or tags.get("brand") or ""),
                    commercial=commercial_score(tags, kind),
                ))

        elif "highway" in tags or "railway" in tags:
            hw = tags.get("highway") or tags.get("railway")
            if hw in ROAD_WIDTHS:
                geom = el.get("geometry")
                if isinstance(geom, list):
                    lat = [p["lat"] for p in geom]
                    lon = [p["lon"] for p in geom]
                    if len(lat) >= 2:
                        local = proj.to_local(np.array(lat), np.array(lon))
                        valid = np.isfinite(local).all(axis=1) & (np.hypot(local[:, 0], local[:, 1]) < 5000.0)
                        local = local[valid]
                        nodes = el.get('nodes') or ()
                        node_ids = tuple(np.asarray(nodes, np.int64)[valid]) if len(nodes) == len(valid) else ()
                        if len(local) >= 2:
                            try: layer = int(tags.get("layer", 0))
                            except ValueError: layer = 0
                            bridge = tags.get("bridge") in ("yes", "viaduct", "aqueduct")
                            tunnel = tags.get("tunnel") in ('yes', 'true', '1', 'culvert') or tags.get('location') == 'underground'
                            covered = tags.get('tunnel') == 'building_passage' or tags.get('covered') in ('yes', 'arcade', 'colonnade')
                            try: level = float(tags['level']) if 'level' in tags else None
                            except (ValueError, TypeError): level = None
                            junction = tags.get('junction', '')
                            reverse = tags.get('oneway') == '-1'
                            oneway = reverse or tags.get("oneway") in ("yes", "true", "1")
                            if tags.get('oneway') not in ('no', 'false', '0') and (junction == 'roundabout' or hw in ('motorway', 'motorway_link')):
                                oneway = True
                            lanes = _lane_count(tags.get('lanes'))
                            lf = _lane_count(tags.get('lanes:forward'))
                            lb = _lane_count(tags.get('lanes:backward'))
                            turns = tuple(tags.get('turn:lanes:forward', tags.get('turn:lanes', '')).split('|')) if tags.get('turn:lanes:forward', tags.get('turn:lanes')) else ()
                            turns_back = tuple(tags.get('turn:lanes:backward', '').split('|')) if tags.get('turn:lanes:backward') else ()
                            sidewalks = parse_sidewalks(tags, hw)
                            if reverse:
                                local = local[::-1]
                                node_ids = node_ids[::-1]
                                lf, lb = lb, lf
                                turns = turns_back or tuple(tags.get('turn:lanes', '').split('|')) if tags.get('turn:lanes') or turns_back else ()
                                turns_back = ()
                                sidewalks = sidewalks[::-1]
                            bname, bstruct = bridge_meta.get(eid, ("", ""))
                            roads.append(Road(hw if "highway" in tags else "railway", 
                                              LineString(local), layer, bridge, tunnel, oneway, lanes,
                                              name=tags.get("name", bname),
                                              structure=tags.get("bridge:structure", bstruct),
                                              uid=eid, speed_limit=parse_maxspeed(tags.get('maxspeed')),
                                              lanes_forward=lf, lanes_backward=lb,
                                              turn_lanes=turns, turn_lanes_backward=turns_back,
                                              node_ids=node_ids,
                                              signal_nodes=tuple(n for n in node_ids if n in signal_nodes),
                                              junction=junction,
                                              drivable=tags.get('motor_vehicle', tags.get('vehicle', tags.get('access', 'yes'))) not in ('no', 'private'),
                                              covered=covered, clearance=parse_length(tags.get('maxheight')) or 0.,
                                              level=level, width=parse_length(tags.get('width')) or 0.,
                                              sidewalks=sidewalks))

        elif tags.get("natural") == "coastline":
            geom = el.get("geometry")
            if isinstance(geom, list) and len(geom) >= 2:
                local = proj.to_local(np.array([p["lat"] for p in geom]),
                                      np.array([p["lon"] for p in geom]))
                local = local[np.isfinite(local).all(axis=1)]
                if len(local) >= 2: coast.append(local)

        elif tags.get("natural") == "water" or "waterway" in tags:
            if tags.get("natural") == "water" or tags.get("waterway") in ("riverbank", "dock"):
                for poly in item_polygons(el, proj):
                    # Unlike buildings, a water body is not owned by its centroid.
                    # Every intersected tile needs its own clipped, non-overlapping
                    # piece, retaining the original coastal/ele classification.
                    waters.append(water_area(poly, tags))
            elif tags.get("waterway") in ("river", "stream", "canal"):
                geom = el.get("geometry")
                if isinstance(geom, list):
                    lat = [p["lat"] for p in geom]
                    lon = [p["lon"] for p in geom]
                    if len(lat) >= 2:
                        local = proj.to_local(np.array(lat), np.array(lon))
                        valid = np.isfinite(local).all(axis=1) & (np.hypot(local[:, 0], local[:, 1]) < 5000.0)
                        local = local[valid]
                        if len(local) >= 2:
                            width = 20.0 if tags["waterway"] == "river" else 5.0
                            waters.append(water_area(LineString(local).buffer(width / 2), tags))

        elif (tags.get("landuse") in ("grass", "forest", "meadow", "railway",
                                       "residential", "commercial", "industrial",
                                       "retail", "farmland") or
              tags.get("leisure") in ("park", "garden", "pitch", "playground") or
              tags.get("natural") in ("wood", "scrub", "sand", "bare_rock", "beach") or
              tags.get("amenity") == "parking"):
            kind = (tags.get("landuse") or tags.get("leisure")
                    or tags.get("natural") or tags.get("amenity"))
            if kind in ("grass", "forest", "meadow", "wood", "scrub",
                        "park", "garden", "pitch", "playground", "residential"):
                mat = 1.0   # Grass
            elif kind in ("commercial", "retail", "industrial"):
                mat = 3.0   # Concrete
            elif kind == "parking":
                mat = 2.0   # Asphalt
            elif kind in ("farmland", "sand", "beach"):
                mat = 4.0   # Dirt / Sand
            else:
                mat = 1.0
            for i, poly in enumerate(item_polygons(el, proj)):
                local = poly.intersection(tile) if tile is not None else poly
                if local.is_empty: continue
                # Parks cover every intersected tile, not just the tile that
                # contains their representative point (e.g. Central Park).
                greens.extend(GreenSpace(kind, p, mat) for p in polygons_of(local) if p.area >= 1.)
                if mat == 1.0:
                    # Scatter on the original outline with a source-specific
                    # seed, then partition roots below. Clipping the scatter
                    # lattice or sharing an RNG across features shifts trees
                    # at seams and when Overpass changes its element order.
                    rng = np.random.default_rng((eid * 16 + i) ^ 90210)
                    density = 0.004 if kind in ("park", "garden") else 0.008
                    pts = scatter_trees_in_polygon(poly, density, rng)
                    if len(pts): scattered.append(pts)

        elif tags.get("natural") == "tree":
            lat = lon = None
            if "lat" in el and "lon" in el: lat, lon = el["lat"], el["lon"]
            else:
                g = el.get("geometry")
                if isinstance(g, dict) and g.get("type") == "Point":
                    c = g.get("coordinates") or ()
                    if len(c) >= 2: lon, lat = c[0], c[1]
            if lat is not None:
                xy = proj.to_local(lat, lon)
                if np.isfinite(xy).all() and np.hypot(xy[0], xy[1]) < 5000.0:
                    pt = Point(xy[0], xy[1])
                    if owner is None or owns_tile(pt, proj, *owner): trees.append(xy)

    if owner is not None:
        if coast: waters.extend(water_area(p, coastal=True) for p in build_ocean(coast, bbox))
        elif ASSUME_EMPTY_TILE_IS_SEA and not (buildings or roads or greens or waters or trees):
            waters.append(water_area(Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]),
                                               (bbox[2], bbox[3]), (bbox[0], bbox[3])]), coastal=True))

    water_polys = [p for w in waters for p in polygons_of(w)]

    avenue = _avenue_trees(roads)
    if len(avenue): scattered.append(avenue)
    if trees:
        scattered.append(np.asarray(trees, np.float64).reshape(-1, 2))

    if scattered:
        pts = np.vstack(scattered)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            # Web-Mercator ownership includes west/north, excludes east/south.
            # Out-of-tile roots would sample a clamped edge height, then float
            # over neighboring terrain (and duplicate its street trees).
            pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] < x1)
                      & (pts[:, 1] > y0) & (pts[:, 1] <= y1)]
        # Never plant inside a carriageway, a building or the water.
        blocked = [r.geometry.buffer(road_width(r) * 0.5 + 0.8)
                   for r in roads if not r.tunnel]
        blocked += tree_buildings
        keep = _points_clear_of(pts, blocked)
        if water_polys: keep &= _points_clear_of(pts, water_polys)
        trees = [tuple(p) for p in pts[keep]]
    else:
        trees = []

    return buildings, roads, greens, waters, trees


# ---------------------------------------------------------------- networking
class TileCancelled(Exception): pass

class RequestGate:
    def __init__(self, max_concurrent=3, min_interval=0.5):
        self._sem = threading.BoundedSemaphore(max(1, max_concurrent))
        self._lock = threading.Lock()
        self._next_start = 0.0
        self.min_interval = max(0.0, min_interval)
        self.in_flight = 0

    @contextmanager
    def slot(self):
        self._sem.acquire()
        try:
            with self._lock:
                now = time.monotonic()
                start = max(now, self._next_start)
                self._next_start = start + self.min_interval
            if start > now: time.sleep(start - now)
            with self._lock: self.in_flight += 1
            try: yield
            finally:
                with self._lock: self.in_flight -= 1
        finally: self._sem.release()

    def penalize(self, seconds):
        with self._lock:
            self._next_start = max(self._next_start, time.monotonic() + seconds)


def _decompress(blob, encoding):
    enc = (encoding or "").strip().lower()
    if not blob or not enc: return blob
    if "gzip" in enc: return gzip.decompress(blob)
    if "deflate" in enc:
        try: return zlib.decompress(blob)
        except zlib.error: return zlib.decompress(blob, -zlib.MAX_WBITS)
    return blob


def _close_quietly(conn):
    try: conn.close()
    except OSError: pass


class HttpPool:
    IDLE_TIMEOUT = 50.0

    def __init__(self, max_idle=4):
        self.max_idle = max(1, max_idle)
        self._idle: dict = {}
        self._lock = threading.Lock()
        self._ctx = None
        self._pm = None
        self._retry = None
        if urllib3 is not None:
            self._retry = urllib3.util.Retry(total=0, redirect=False, raise_on_status=False)
            self._pm = urllib3.PoolManager(num_pools=8, maxsize=self.max_idle,
                                           block=False, retries=self._retry)

    def request(self, method, url, body=None, headers=None,
                connect_timeout=6.0, read_timeout=45.0, on_bytes=None):
        if self._pm is not None:
            return self._urllib3(method, url, body, headers, connect_timeout, read_timeout, on_bytes)
        return self._stdlib(method, url, body, headers, connect_timeout, read_timeout, on_bytes)

    def close(self):
        with self._lock:
            conns = [item[1] for bucket in self._idle.values() for item in bucket]
            self._idle.clear()
        for c in conns: _close_quietly(c)
        if self._pm is not None: self._pm.clear()

    def _urllib3(self, method, url, body, headers, connect_timeout, read_timeout, on_bytes=None):
        resp = self._pm.urlopen(method, url, body=body, headers=dict(headers or {}),
                                timeout=urllib3.Timeout(connect=connect_timeout, read=read_timeout),
                                preload_content=False, decode_content=False, retries=self._retry)
        try:
            chunks, total = [], 0
            for chunk in resp.stream(65536, decode_content=False):
                chunks.append(chunk); total += len(chunk)
                if on_bytes is not None: on_bytes(total)
            blob = b"".join(chunks)
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            status = resp.status
        except BaseException:
            resp.close(); raise
        resp.release_conn()
        return status, hdrs, _decompress(blob, hdrs.get("content-encoding", ""))

    def _conn_new(self, parts, connect_timeout, read_timeout):
        host, port = parts.hostname, parts.port
        if parts.scheme == "https":
            if self._ctx is None: self._ctx = ssl.create_default_context()
            conn = http.client.HTTPSConnection(host, port, timeout=connect_timeout, context=self._ctx)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=connect_timeout)
        conn.connect()
        if conn.sock is not None: conn.sock.settimeout(read_timeout)
        return conn

    def _checkout(self, hostport, parts, connect_timeout, read_timeout):
        with self._lock: bucket = self._idle.get(hostport)
        now = time.monotonic()
        while bucket:
            item = bucket.pop()
            if now - item[0] < self.IDLE_TIMEOUT: return item[1]
            _close_quietly(item[1])
        return self._conn_new(parts, connect_timeout, read_timeout)

    def _release(self, hostport, conn):
        with self._lock:
            bucket = self._idle.setdefault(hostport, [])
            if len(bucket) < self.max_idle:
                bucket.append((time.monotonic(), conn)); return
        _close_quietly(conn)

    def _stdlib(self, method, url, body, headers, connect_timeout, read_timeout, on_bytes=None):
        parts = urllib.parse.urlsplit(url)
        path = parts.path + ("?" + parts.query if parts.query else "")
        default_port = 443 if parts.scheme == "https" else 80
        hostport = (parts.hostname, parts.port or default_port)
        hdrs = dict(headers or {})
        hdrs.setdefault("Connection", "keep-alive")
        for attempt in (0, 1):
            conn = self._checkout(hostport, parts, connect_timeout, read_timeout)
            try:
                conn.request(method, path, body=body, headers=hdrs)
                resp = conn.getresponse()
                chunks, total = [], 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk: break
                    chunks.append(chunk); total += len(chunk)
                    if on_bytes is not None: on_bytes(total)
                blob = b"".join(chunks)
                status = resp.status
                out = {k.lower(): v for k, v in resp.getheaders()}
                if resp.will_close: _close_quietly(conn)
                else: self._release(hostport, conn)
                return status, out, _decompress(blob, out.get("content-encoding", ""))
            except (http.client.RemoteDisconnected, http.client.BadStatusLine,
                    ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                _close_quietly(conn)
                if attempt: raise
            except BaseException:
                _close_quietly(conn); raise
        raise RuntimeError("unreachable")


class OSMDataFetcher:
    ENDPOINTS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.private.coffee/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]
    REQUEST_TIMEOUT = 120.0
    CONNECT_TIMEOUT = 10.0
    HEDGE_DELAY = 5.0
    MAX_RACE = 3
    ROUNDS = 3
    UA = {"User-Agent": "osm3d-viewer/1.0 (educational PyOpenGL demo)",
          "Accept-Encoding": "gzip, deflate"}

    def __init__(self, cache_dir, default_height=9.0, level_height=3.2,
                 max_requests=3, min_interval=0.5, http_timeout=None,
                 connect_timeout=None, hedge_delay=None, race=None):
        self.cache_dir = Path(cache_dir)
        self.default_height = default_height
        self.level_height = level_height
        self.gate = RequestGate(max_requests, min_interval)
        self.http = HttpPool()
        self.http_timeout = float(http_timeout or self.REQUEST_TIMEOUT)
        self.connect_timeout = float(connect_timeout or self.CONNECT_TIMEOUT)
        self.hedge_delay = float(hedge_delay if hedge_delay is not None else self.HEDGE_DELAY)
        self.max_race = max(1, int(self.MAX_RACE if race is None else race))
        self.refresh = False
        self._refreshed: set = set()
        self._rr = itertools.count()
        from .terrain import TerrainTiles
        self.terrain = TerrainTiles(self.cache_dir)
        self.terrain_datum = 0.0

    def close(self): self.http.close()

    def tile_terrain(self, z, x, y, samples=None):
        """Elevation grid for one chunk tile, relative to the world datum.

        Returns ``(heights, sea_z)`` or ``(None, 0.0)`` when terrain is off.
        """
        if not getattr(self.terrain, "enabled", False):
            return None, 0.0
        from ..core.config import TERRAIN_GRID
        s, w, n, e = tile_bounds(z, x, y)
        grid = self.terrain.grid(s, w, n, e, samples or TERRAIN_GRID)
        if grid is None:
            return None, 0.0
        datum = float(self.terrain_datum)
        return (grid - datum).astype(np.float32), -datum

    def geocode_full(self, place):
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
            {"q": place, "format": "json", "limit": 1})
        status, _, blob = self.http.request("GET", url,
                                            headers={"User-Agent": self.UA["User-Agent"]},
                                            connect_timeout=self.connect_timeout,
                                            read_timeout=20.0)
        if status != 200: raise RuntimeError(f"Nominatim HTTP {status}")
        res = json.loads(blob)
        if not res: raise RuntimeError(f"Could not find '{place}'")
        name = ", ".join(res[0].get("display_name", place).split(", ")[:2])
        return float(res[0]["lat"]), float(res[0]["lon"]), name

    def tile_cache_path(self, z, x, y):
        return self.cache_dir / f"overpass_v4_{z}_{x}_{y}.json.gz"

    @staticmethod
    def _cache_candidates(path):
        """Compressed cache first, then the legacy plain-JSON spelling."""
        path = Path(path)
        if path.suffix == ".gz":
            return path, path.with_suffix("")
        return path.with_name(path.name + ".gz"), path

    @classmethod
    def cache_path(cls, path):
        """Existing cache file for a canonical tile path, or the new .json.gz path."""
        candidates = cls._cache_candidates(path)
        return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    @classmethod
    def _read_cache(cls, path):
        for candidate in cls._cache_candidates(path):
            if not candidate.exists(): continue
            try:
                if candidate.suffix == ".gz":
                    with gzip.open(candidate, "rt", encoding="utf-8") as file:
                        data = json.load(file)
                else:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict) and "elements" in data:
                return data
            try: candidate.unlink()
            except OSError: pass
        return None

    @staticmethod
    def _write_cache(path, data):
        path = Path(path)
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            if path.suffix == ".gz":
                payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
                tmp.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
            else:
                tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, path)
            if path.suffix == ".gz":
                try: path.with_suffix("").unlink()
                except OSError: pass
        except (OSError, TypeError, ValueError) as exc:
            print(f"[data] cache write failed: {exc}")
            if tmp is not None:
                try: tmp.unlink()
                except OSError: pass

    @classmethod
    def tile_query(cls, s, w, n, e):
        bb = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
        return (
            "[out:json][timeout:60];\n(\n"
            f'  way["building"]({bb});\n'
            f'  relation["building"]({bb});\n'
            f'  way["building:part"]({bb});\n'
            f'  relation["building:part"]({bb});\n'
            f'  way["highway"]({bb});\n'
            f'  way["railway"]({bb});\n'
            f'  way["natural"="water"]({bb});\n'
            f'  way["waterway"]({bb});\n'
            f'  way["natural"="coastline"]({bb});\n'
            f'  relation["natural"="water"]({bb});\n'
            f'  relation["man_made"="bridge"]({bb});\n'
            f'  way["landuse"~"grass|forest|meadow|railway|residential|commercial|industrial|retail|farmland"]({bb});\n'
            f'  way["leisure"~"park|garden|pitch|playground"]({bb});\n'
            f'  way["natural"~"wood|scrub|sand|bare_rock|beach"]({bb});\n'
            f'  way["amenity"="parking"]({bb});\n'
            f'  node["natural"="tree"]({bb});\n'
            f'  node["highway"="traffic_signals"]({bb});\n'
            f'  nwr["amenity"~"^(cafe|restaurant|fast_food|food_court|bar|pub|biergarten|fuel|charging_station|parking|parking_entrance|hospital|clinic|doctors|dentist|pharmacy|school|university|college|library|bank|cinema|theatre|police|fire_station|place_of_worship|community_centre|townhall)$"]({bb});\n'
            f'  nwr["shop"]({bb});\n'
            f'  nwr["tourism"~"^(hotel|motel|hostel|guest_house|museum|attraction|viewpoint|gallery)$"]({bb});\n'
            f'  nwr["railway"~"^(station|halt|subway_entrance)$"]({bb});\n'
            f'  nwr["public_transport"="station"]({bb});\n'
            f'  nwr["leisure"~"^(park|garden|playground|stadium|sports_centre)$"]({bb});\n'
            f'  nwr["historic"~"^(monument|memorial|castle|ruins)$"]({bb});\n'
            ");\n"
            "out geom qt;"
        )

    def fetch_tile_raw(self, z, x, y, cancel=None):
        key = (z, x, y); tkey = (x, y); tag = f"{z}/{x}/{y}"
        cache = self.tile_cache_path(z, x, y)
        if self.refresh and key not in self._refreshed:
            self._refreshed.add(key)
        else:
            tm.BOARD.set(tkey, "cache", "reading OSM data cache")
            raw = self._read_cache(cache)
            if raw is not None:
                tm.LOG.info(f"tile {tag}: OSM data found in disk cache")
                return raw
        s, w, n, e = tile_bounds(z, x, y)
        body = urllib.parse.urlencode({"data": self.tile_query(s, w, n, e)}).encode()
        tm.BOARD.set(tkey, "download", "contacting Overpass")
        tm.LOG.info(f"tile {tag}: not cached, downloading from Overpass")
        t0 = time.monotonic()
        data = self._race(body, cancel, tkey, tag)
        tm.LOG.ok(f"tile {tag}: {len(data.get('elements', ())):,} OSM elements "
                  f"downloaded in {time.monotonic() - t0:.1f}s")
        self._write_cache(cache, data)
        return data

    def _attempt(self, url, body, out, cancel, tkey=None, tag=""):
        host = urllib.parse.urlparse(url).netloc
        t0 = time.monotonic()
        def on_bytes(n):
            tm.BOARD.update(tkey, nbytes=n, detail=f"{host} - {tm.fmt_bytes(n)} received")
        try:
            if cancel and cancel(): raise TileCancelled()
            tm.BOARD.update(tkey, detail=f"{host} - waiting for a request slot")
            with self.gate.slot():
                if cancel and cancel(): raise TileCancelled()
                tm.LOG.info(f"tile {tag}: asking {host}")
                tm.BOARD.update(tkey, host=host, detail=f"{host} - waiting for reply")
                status, headers, blob = self.http.request("POST", url, body=body,
                                                          headers=self.UA,
                                                          connect_timeout=self.connect_timeout,
                                                          read_timeout=self.http_timeout,
                                                          on_bytes=on_bytes)
            if status != 200:
                wait = 0.0
                try: wait = float(headers.get("retry-after", 0) or 0.0)
                except ValueError: pass
                if status == 429:
                    back = min(60.0, max(wait, 8.0))
                    self.gate.penalize(back)
                    tm.LOG.warn(f"{host}: rate limited (HTTP 429), backing off {back:.0f}s")
                raise RuntimeError(f"HTTP {status}")
            data = json.loads(blob)
            if "elements" not in data:
                raise RuntimeError(str(data.get("remark", "empty reply"))[:80])
            tm.LOG.info(f"tile {tag}: {host} answered with {tm.fmt_bytes(len(blob))} "
                        f"in {time.monotonic() - t0:.1f}s")
            out.put(("ok", data))
        except TileCancelled:
            out.put(("err", f"{host}: cancelled", True))
        except Exception as exc:
            msg = f"{host}: {str(exc)[:120]}"
            tm.LOG.warn(f"tile {tag}: {msg} (after {time.monotonic() - t0:.1f}s)")
            tm.BOARD.update(tkey, error=msg)
            out.put(("err", msg))

    def _race(self, body, cancel=None, tkey=None, tag=""):
        n_ep = len(self.ENDPOINTS)
        first = next(self._rr) % n_ep
        order = [self.ENDPOINTS[(first + i) % n_ep] for i in range(n_ep)]
        errors = []
        for rnd in range(self.ROUNDS):
            out = queue.Queue(); free = list(order); live = []; last_launch = -1e9
            if rnd:
                tm.LOG.warn(f"tile {tag}: all mirrors failed, starting round {rnd + 1}/{self.ROUNDS}")
                tm.BOARD.update(tkey, detail=f"retrying (round {rnd + 1}/{self.ROUNDS})")
            while True:
                now = time.monotonic()
                while free and len(live) < self.max_race and now - last_launch >= self.hedge_delay:
                    url = free.pop(0)
                    th = threading.Thread(target=self._attempt,
                                          args=(url, body, out, cancel, tkey, tag),
                                          name="overpass", daemon=True)
                    th.start(); live.append(th)
                    last_launch = time.monotonic(); now = last_launch
                if not live and not free: break
                try:
                    item = out.get(timeout=0.2)
                except queue.Empty:
                    if cancel and cancel(): raise TileCancelled()
                    live = [t for t in live if t.is_alive()]; continue
                tag_, payload = item[0], item[1]
                if tag_ == "ok": return payload
                if len(item) < 3: errors.append(payload)
                live = [t for t in live if t.is_alive()]
                last_launch = -1e9
            if rnd + 1 < self.ROUNDS:
                time.sleep(1.0)
                if cancel and cancel(): raise TileCancelled()
        raise RuntimeError("Overpass unavailable (" + "; ".join(errors[-3:] or ["no reply"]) + ")")


def demo_buildings(seed=7):
    rng = np.random.default_rng(seed)
    out, bid = [], 1
    for gx in range(-6, 6):
        for gy in range(-6, 6):
            if rng.random() < 0.15: continue
            cx, cy = gx * 55 + rng.uniform(-4, 4), gy * 55 + rng.uniform(-4, 4)
            w, d = rng.uniform(18, 40), rng.uniform(18, 40)
            h = float(rng.choice([8, 12, 18, 25, 40, 65, 95], p=[.2, .2, .2, .15, .12, .08, .05]))
            if rng.random() < 0.2:
                pts = [(0, 0), (w, 0), (w, d * .45), (w * .45, d * .45), (w * .45, d), (0, d)]
            else:
                pts = [(0, 0), (w, 0), (w, d), (0, d)]
            poly = Polygon([(cx + x - w / 2, cy + y - d / 2) for x, y in pts])
            if rng.random() < 0.15 and len(pts) == 4:
                hole = Polygon([(cx - w / 5, cy - d / 5), (cx + w / 5, cy - d / 5),
                                (cx + w / 5, cy + d / 5), (cx - w / 5, cy + d / 5)])
                poly = poly.difference(hole)
            c = np.array([0.8, 0.8, 0.8])
            out.append(Building(bid, poly, h, 0.0, "yes", c, c * 0.6, ""))
            bid += 1
    return out
