"""OSM data fetching, complex building relations, material parsing, and demo fallback."""
from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import linemerge, polygonize, unary_union

from .projection import LocalProjection


class TileCancelled(Exception):
    """Raised when a tile fetch or mesh job is cancelled by the UI."""
    pass


@dataclass
class Building:
    id: int
    footprint: Polygon
    height: float
    min_height: float
    kind: str
    color: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 0.7], np.float32))
    roof_color: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 0.7], np.float32))
    roof_shape: str = "flat"
    roof_height: float = 0.0
    parts: list = field(default_factory=list)
    is_part: bool = False


@dataclass
class Road:
    geometry: LineString
    highway: str


@dataclass
class GreenSpace:
    geometry: Polygon
    kind: str


def _jitter(i: int) -> float:
    x = math.sin(i * 12.9898) * 43758.5453
    return x - math.floor(x)


def theme_code(kind: str, seed: int) -> float:
    """Returns architectural theme: 0=Glass, 1=Masonry, 2=Brick, 3=Industrial"""
    if not kind:
        kind = "residential"
    k = kind.lower()
    
    if "glass" in k or "office" in k or "commercial" in k or "tower" in k:
        return 0.0
    if "industrial" in k or "warehouse" in k:
        return 3.0
    if "retail" in k or "shop" in k:
        return 1.5
    
    h = _jitter(seed * 7 + 3)
    if h < 0.4: return 1.0  # Masonry
    if h < 0.7: return 2.0  # Brick
    return 0.0              # Glass


def _parse_length(s):
    if not s: return None
    try:
        return float(str(s).split()[0])
    except (ValueError, TypeError):
        return None


def _color_for_theme(theme: float, seed: int) -> np.ndarray:
    rng_a = _jitter(seed * 7 + 3)
    rng_b = _jitter(seed * 11 + 5)
    
    if theme == 0.0:   # Glass
        base = np.array([0.15, 0.20, 0.25], np.float32)
    elif theme == 1.0: # Masonry
        base = np.array([0.65, 0.60, 0.55], np.float32)
    elif theme == 2.0: # Brick
        base = np.array([0.45, 0.22, 0.18], np.float32)
    else:              # Industrial
        base = np.array([0.55, 0.55, 0.50], np.float32)
        
    tint = 0.85 + 0.30 * rng_a
    return np.clip(base * tint + (rng_b - 0.5) * 0.05, 0, 1).astype(np.float32)


def _clean_coords(coords):
    if coords is None:
        return np.empty((0, 2), np.float64)
    arr = np.asarray(coords, np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if len(arr) == 0:
        return arr
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if len(arr) < 2:
        return arr
    keep = np.concatenate(([True], np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-6))
    return arr[keep]


def _parts(poly):
    if poly is None or poly.is_empty:
        return
    if poly.geom_type == "Polygon":
        yield poly
    elif hasattr(poly, "geoms"):
        for g in poly.geoms:
            if g.geom_type == "Polygon":
                yield g


def build_tree_template():
    verts, idx = [], []
    seg = 6
    trunk_r = 0.22
    z0, z1 = 0.0, 2.0
    trunk_col = (0.30, 0.20, 0.12)
    n0 = 0
    for i in range(seg + 1):
        a = 2.0 * math.pi * i / seg
        c, s = math.cos(a), math.sin(a)
        verts.append([trunk_r * c, trunk_r * s, z0, c, s, 0.0, *trunk_col, 0.0])
        verts.append([trunk_r * c, trunk_r * s, z1, c, s, 0.0, *trunk_col, 0.0])
    for i in range(seg):
        a = n0 + i * 2
        idx += [a, a + 1, a + 3, a, a + 3, a + 2]

    n_lat, n_lon = 5, 8
    lat0, lat1 = -0.15 * math.pi, 0.5 * math.pi
    r_c = 1.55
    z_c = 1.8
    canopy_col = (0.30, 0.55, 0.22)
    n1 = len(verts)
    for li in range(n_lat + 1):
        lat = lat0 + (lat1 - lat0) * li / n_lat
        for lo in range(n_lon):
            lon = 2.0 * math.pi * lo / n_lon
            nx = math.cos(lat) * math.cos(lon)
            ny = math.cos(lat) * math.sin(lon)
            nz = math.sin(lat)
            verts.append([r_c * nx, r_c * ny, z_c + r_c * nz, nx, ny, nz, *canopy_col, 1.0])
    for li in range(n_lat):
        for lo in range(n_lon):
            a = n1 + li * n_lon + lo
            b = n1 + li * n_lon + (lo + 1) % n_lon
            c = n1 + (li + 1) * n_lon + (lo + 1) % n_lon
            d = n1 + (li + 1) * n_lon + lo
            idx += [a, b, c, a, c, d]
    return np.asarray(verts, np.float32), np.asarray(idx, np.uint32)


def build_trees_mesh(trees):
    if not trees: return None
    arr = np.zeros((len(trees), 8), np.float32)
    for i, t in enumerate(trees):
        x, y, scl, r, g, b = t
        rot = _jitter(i * 13 + 7) * 2.0 * math.pi
        arr[i] = (x, y, 0.0, scl, rot, r, g, b)
    return arr


OVERPASS_QUERY = """
[out:json][timeout:90];
(
  way["building"]({bbox});
  relation["building"]({bbox});
  way["building:part"]({bbox});
  way["highway"]({bbox});
  way["railway"]({bbox});
  way["leisure"~"^(park|garden|nature_reserve)$"]({bbox});
  way["landuse"~"^(grass|meadow|forest|recreation_ground)$"]({bbox});
  way["natural"~"^(wood|scrub|grassland|heath|water|bay|strait|coastline)$"]({bbox});
  way["water"~"^(lake|reservoir|pond|river|basin|lagoon|harbour|bay)$"]({bbox});
  way["waterway"~"^(riverbank|dock|canal)$"]({bbox});
  relation["natural"~"^(water|bay|strait)$"]({bbox});
  relation["water"~"^(lake|reservoir|basin|harbour|bay|river)$"]({bbox});
  relation["waterway"~"^(riverbank|dock)$"]({bbox});
  way["man_made"~"^(pier|wharf|quay|breakwater|groyne|jetty)$"]({bbox});
  relation["man_made"~"^(pier|wharf|quay)$"]({bbox});
  way["landuse"~"^(industrial|port|harbour)$"]({bbox});
);
out body;
>;
out skel qt;
"""


def _bbox_of_tile(z, x, y):
    n = 2 ** z
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lon0 = x / n * 360.0 - 180.0
    lon1 = (x + 1) / n * 360.0 - 180.0
    return (min(lat0, lat1), lon0, max(lat0, lat1), lon1)


def fetch_raw_tile(z, x, y, timeout=90, cancel=None):
    if cancel is not None and cancel.is_set():
        raise TileCancelled("Cancelled before fetch")
        
    bbox = _bbox_of_tile(z, x, y)
    q = OVERPASS_QUERY.format(bbox=",".join(f"{v:.6f}" for v in bbox))
    req = urllib.request.Request(
        "https://overpass-api.de/api/interpreter",
        data=urllib.parse.urlencode({"data": q}).encode(),
        headers={"User-Agent": "PyOpenGL-CityViewer/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if cancel is not None and cancel.is_set():
            raise TileCancelled("Cancelled during fetch")
        payload = json.loads(r.read().decode())
    return payload.get("elements", [])


# --- Water / Coastline Logic ---
def _extend_to_bbox(p_end, p_toward, bbox):
    x, y = p_end
    dx, dy = x - p_toward[0], y - p_toward[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9: return (x, y)
    ts = []
    if abs(dx) > 1e-9: ts += [(bbox[0] - x) / dx, (bbox[2] - x) / dx]
    if abs(dy) > 1e-9: ts += [(bbox[1] - y) / dy, (bbox[3] - y) / dy]
    ts = [t for t in ts if t > 1e-9]
    if not ts: return (x, y)
    t = min(ts)
    return (x + dx * t, y + dy * t)

def _side_of_point(p, bbox):
    x, y = p
    tol = 1e-3
    if abs(x - bbox[0]) < tol: return 0
    if abs(y - bbox[3]) < tol: return 1
    if abs(x - bbox[2]) < tol: return 2
    if abs(y - bbox[1]) < tol: return 3
    return min([(abs(x - bbox[0]), 0), (abs(y - bbox[3]), 1), (abs(x - bbox[2]), 2), (abs(y - bbox[1]), 3)])[1]

def _snap_to_edge(p, side, bbox):
    x, y = p
    if side == 0: return (bbox[0], min(max(y, bbox[1]), bbox[3]))
    if side == 1: return (min(max(x, bbox[0]), bbox[2]), bbox[3])
    if side == 2: return (bbox[2], min(max(y, bbox[1]), bbox[3]))
    return (min(max(x, bbox[0]), bbox[2]), bbox[1])

def _cw_arc(p_start, side_s, p_end, side_e, bbox):
    corner_after = {0: (bbox[0], bbox[3]), 1: (bbox[2], bbox[3]), 2: (bbox[2], bbox[1]), 3: (bbox[0], bbox[1])}
    arc = [p_end]
    s = side_e
    for _ in range(4):
        if s == side_s: break
        arc.append(corner_after[s])
        s = (s + 1) % 4
    arc.append(p_start)
    out = [arc[0]]
    for p in arc[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out

def stitch_coastlines(coastline_ways, bbox):
    if not coastline_ways: return []
    try:
        merged = linemerge([LineString(c) for c in coastline_ways if len(c) >= 2])
    except Exception: return []
    lines = [merged] if merged.geom_type == "LineString" else list(merged.geoms)
    bbox_poly = box(*bbox)
    out = []
    for line in lines:
        if line.geom_type != "LineString" or len(line.coords) < 2: continue
        coords = list(line.coords)
        if coords[0] == coords[-1]: continue
        start = _extend_to_bbox(coords[0], coords[1], bbox)
        end = _extend_to_bbox(coords[-1], coords[-2], bbox)
        ss, se = _side_of_point(start, bbox), _side_of_point(end, bbox)
        start = _snap_to_edge(start, ss, bbox)
        end = _snap_to_edge(end, se, bbox)
        arc = _cw_arc(start, ss, end, se, bbox)
        ring = [start] + [tuple(p) for p in coords[1:-1]] + [end] + arc[1:]
        if len(ring) < 4: continue
        if ring[0] == ring[-1]: ring = ring[:-1]
        try:
            poly = Polygon(ring)
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_empty: continue
            poly = poly.intersection(bbox_poly)
            for part in (poly.geoms if hasattr(poly, "geoms") else [poly]):
                if part.geom_type == "Polygon" and part.area > 100.0: out.append(orient(part, 1.0))
        except Exception: continue
    return out


def _is_water_tag(tags):
    if tags.get("natural") in ("water", "bay", "strait"): return True
    if tags.get("water") in ("lake", "reservoir", "pond", "river", "basin", "lagoon", "harbour", "bay"): return True
    if tags.get("waterway") in ("riverbank", "dock", "canal"): return True
    return False

def _is_pier_tag(tags):
    return tags.get("man_made") in {"pier", "wharf", "quay", "breakwater", "groyne", "jetty"} or tags.get("landuse") in {"industrial", "port", "harbour"}

def _way_coords(way, nodes):
    return [nodes[nid] for nid in way.get("nodes", []) if nid in nodes]

def parse_water(ways, relations, nodes, bbox_xy):
    raw = []
    way_by_id = {w["id"]: w for w in ways}
    for w in ways:
        if not _is_water_tag(w.get("tags", {})): continue
        coords = _way_coords(w, nodes)
        if len(coords) < 4 or coords[0] != coords[-1]: continue
        try:
            p = Polygon(coords[:-1])
            if not p.is_valid: p = p.buffer(0)
            if not p.is_empty and p.area > 20.0: raw.append(p)
        except Exception: pass

    coastline_ways = [_way_coords(w, nodes) for w in ways if w.get("tags", {}).get("natural") == "coastline"]
    raw.extend(stitch_coastlines([c for c in coastline_ways if len(c) >= 2], bbox_xy))

    for rel in relations:
        if not _is_water_tag(rel.get("tags", {})): continue
        outers = [_way_coords(way_by_id[m["ref"]], nodes) for m in rel.get("members", []) if m.get("type") == "way" and m.get("role") != "inner" and m["ref"] in way_by_id]
        if not outers: continue
        try:
            merged = linemerge([LineString(c) for c in outers if len(c) >= 3])
            for p in polygonize(merged):
                if p.is_valid and p.area > 50.0: raw.append(p)
        except Exception: pass

    if not raw: return []
    try:
        # Simplify to prevent unary_union from hanging on massive overlapping coastline segments
        raw = [p.simplify(0.2).buffer(0) for p in raw if not p.is_empty]
        water = unary_union(raw)
    except Exception: return []

    piers = []
    for w in ways:
        if not _is_pier_tag(w.get("tags", {})): continue
        c = _way_coords(w, nodes)
        if len(c) < 4: continue
        try:
            p = Polygon(c if c[0] == c[-1] else c + [c[0]])
            if not p.is_valid: p = p.buffer(0)
            if not p.is_empty: piers.append(p)
        except Exception: pass
    if piers:
        try: water = water.difference(unary_union(piers))
        except Exception: pass

    if water.is_empty: return []
    return [orient(p, 1.0) for p in (water.geoms if hasattr(water, "geoms") else [water]) if p.geom_type == "Polygon" and p.area > 40.0]


# --- Main Parsing Logic ---
def parse_tile_elements(raw, proj, tile, default_height=12.0, level_height=3.5):
    nodes = {}
    ways, relations = [], []
    for el in raw:
        t = el.get("type")
        if t == "node": nodes[el["id"]] = proj.to_local(el["lat"], el["lon"])
        elif t == "way": ways.append(el)
        elif t == "relation": relations.append(el)

    way_by_id = {w["id"]: w for w in ways}
    z, x, y = tile
    lat0, lon0, lat1, lon1 = _bbox_of_tile(z, x, y)
    sw, ne = proj.to_local(lat0, lon0), proj.to_local(lat1, lon1)
    bbox_xy = (min(sw[0], ne[0]), min(sw[1], ne[1]), max(sw[0], ne[0]), max(sw[1], ne[1]))

    buildings, roads, greens, trees = [], [], [], []

    def _local(way):
        return _clean_coords(_way_coords(way, nodes))

    bld_map = {}
    for el in raw:
        tags = el.get("tags", {})
        if "building" not in tags and "building:part" not in tags: continue
        
        is_part = "building:part" in tags
        if el["type"] == "way":
            coords = _local(el)
            if len(coords) < 3: continue
            try:
                poly = Polygon(coords)
                if not poly.is_valid: poly = poly.buffer(0)
                if poly.is_empty or poly.area < 4.0: continue
                
                h = _parse_length(tags.get("height"))
                if h is None:
                    lv = _parse_length(tags.get("building:levels"))
                    h = lv * level_height if lv is not None else default_height * (0.6 + 1.2 * _jitter(el["id"]))
                minh = _parse_length(tags.get("min_height")) or _parse_length(tags.get("building:min_level")) or 0.0
                
                roof_h = _parse_length(tags.get("roof:height")) or 0.0
                roof_shape = tags.get("roof:shape", "flat")
                if roof_shape not in ("flat", "pyramidal", "pitched", "skillion", "dome"): roof_shape = "flat"
                
                kind = tags.get("building") or tags.get("building:part") or "yes"
                theme = theme_code(kind, el["id"])
                color = _color_for_theme(theme, el["id"])
                
                rc = tags.get("roof:colour")
                if rc:
                    try: roof_col = np.array([int(rc.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4)], np.float32)
                    except Exception: roof_col = np.clip(color * 1.25 + 0.02, 0, 1)
                else:
                    roof_col = np.clip(color * 1.25 + 0.02, 0, 1)
                    
                b = Building(
                    id=el["id"], footprint=poly, height=float(h), min_height=float(minh),
                    kind=kind, color=color, roof_color=roof_col.astype(np.float32),
                    roof_shape=roof_shape, roof_height=float(roof_h), is_part=is_part
                )
                bld_map[el["id"]] = b
                if not is_part:
                    buildings.append(b)
            except Exception: pass

    for el in raw:
        if el.get("type") != "relation" or "building" not in el.get("tags", {}): continue
        
        main_id = el["id"]
        main_h = _parse_length(el["tags"].get("height")) or default_height * 2
        main_theme = theme_code(el["tags"].get("building", "yes"), main_id)
        main_col = _color_for_theme(main_theme, main_id)
        
        main_bld = Building(
            id=main_id, footprint=Polygon(), height=main_h, min_height=0.0,
            kind="relation", color=main_col, roof_color=main_col*1.2
        )
        
        for m in el.get("members", []):
            if m.get("type") == "way" and m.get("role") == "part" and m["ref"] in bld_map:
                part = bld_map[m["ref"]]
                part.is_part = True
                main_bld.parts.append(part)
                
        if main_bld.parts:
            buildings.append(main_bld)

    for w in ways:
        tags = w.get("tags", {})
        hw, rw = tags.get("highway"), tags.get("railway")
        if not hw and rw not in ("rail", "light_rail", "subway", "tram"): continue
        coords = _local(w)
        if len(coords) < 2: continue
        try:
            if rw in ("rail", "light_rail", "subway", "tram"): roads.append(Road(LineString(coords), "railway"))
            else: roads.append(Road(LineString(coords), hw))
        except Exception: continue

    for w in ways:
        tags = w.get("tags", {})
        kind = None
        if tags.get("leisure") in ("park", "garden", "nature_reserve"): kind = tags["leisure"]
        elif tags.get("landuse") in ("grass", "meadow", "forest", "recreation_ground"): kind = tags["landuse"]
        elif tags.get("natural") in ("wood", "scrub", "grassland", "heath"): kind = "forest" if tags["natural"] == "wood" else "grass"
        if kind is None: continue
        coords = _local(w)
        if len(coords) < 3: continue
        try:
            poly = Polygon(coords)
            if not poly.is_valid: poly = poly.buffer(0)
            if poly.is_empty or poly.area < 10.0: continue
            greens.append(GreenSpace(geometry=poly, kind=kind))
            n_trees = min(int(poly.area / 200.0), 40)
            if n_trees:
                minx, miny, maxx, maxy = poly.bounds
                seed = w["id"] % 1000003
                for k in range(n_trees):
                    tx = minx + (maxx - minx) * _jitter(seed + k * 3)
                    ty = miny + (maxy - miny) * _jitter(seed + k * 3 + 1)
                    if not poly.contains(Point(tx, ty)): continue
                    trees.append((tx, ty, 0.7 + 0.6 * _jitter(seed + k * 3 + 2), 0.28, 0.45, 0.20))
        except Exception: continue

    waters = parse_water(ways, relations, nodes, bbox_xy)
    return buildings, roads, greens, waters, trees


class OSMDataFetcher:
    """Provides an object-oriented wrapper around the Overpass fetching logic."""
    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout", 90)
        self.default_height = kwargs.get("default_height", 12.0)
        self.level_height = kwargs.get("level_height", 3.5)

    def fetch_tile_raw(self, z, x, y, **kwargs):
        cancel = kwargs.get("cancel")
        try:
            return fetch_raw_tile(z, x, y, self.timeout, cancel)
        except TileCancelled:
            raise
        except Exception as e:
            raise TileCancelled(str(e))
            
    def fetch_raw(self, z, x, y, **kwargs):
        return self.fetch_tile_raw(z, x, y, **kwargs)
        
    def fetch(self, z, x, y, **kwargs):
        return self.fetch_tile_raw(z, x, y, **kwargs)


def demo_buildings(proj=None):
    """Generates a procedural city block for offline testing or fallback."""
    blds, roads, greens, waters, trees = [], [], [], [], []
    roads.append(Road(LineString([(-100, 0), (100, 0)]), "primary"))
    roads.append(Road(LineString([(0, -100), (0, 100)]), "secondary"))
    
    for i in range(4):
        for j in range(4):
            if i == 1 and j == 1: continue
            x, y = i * 30 - 40, j * 30 - 40
            if abs(x) < 10 or abs(y) < 10: continue
            
            h = 15.0 + _jitter(i * 10 + j) * 80.0
            theme_seed = i * 100 + j
            kind = "office" if h > 60 else ("commercial" if h > 30 else "residential")
            theme = theme_code(kind, theme_seed)
            color = _color_for_theme(theme, theme_seed)
            
            roof_shape, roof_h = "flat", 0.0
            if h > 20 and _jitter(theme_seed) > 0.6:
                roof_shape = "pyramidal" if _jitter(theme_seed) > 0.5 else "pitched"
                roof_h = 5.0 + _jitter(theme_seed) * 10.0
            
            poly = box(x, y, x + 15, y + 15)
            blds.append(Building(
                id=theme_seed, footprint=poly, height=h, min_height=0.0,
                kind=kind, color=color, roof_color=np.clip(color*1.2, 0, 1),
                roof_shape=roof_shape, roof_height=roof_h
            ))

    park = box(-15, -15, 15, 15)
    greens.append(GreenSpace(geometry=park, kind="park"))
    for k in range(10):
        trees.append((-10 + _jitter(k)*20, -10 + _jitter(k+1)*20, 1.0, 0.3, 0.5, 0.2))

    lake = box(60, -60, 90, -30)
    waters.append(orient(lake, 1.0))
    return blds, roads, greens, waters, trees