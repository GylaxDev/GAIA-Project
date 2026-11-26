"""Mesh extraction, cache and worker pool."""
from __future__ import annotations

import math
import os
import pickle
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import triangulate, unary_union, polygonize
from shapely.validation import make_valid
from .surface_geometry import TerrainDraper, terrain_path, vertex_normals

from ..core.config import MESH_CACHE_VERSION, ROAD_SURFACE, ROAD_WIDTHS, TERRAIN_GRID
from .data_fetcher import (Building, Road, GreenSpace, RoadPath, WaterArea,
                           build_trees_mesh, parse_tile_elements, clip_roads_to_bbox,
                           _densify,
                           _clean_coords, _jitter, theme_code, polygons_of,
                           _points_in_polys,
                           strip_ring, sanitize_footprint,
                           road_paths_from_roads, parking_slots_from_paths,
                           road_profile, pedestrian_bridge_pins,
                           sample_path_position, sample_path_z,
                           PEDESTRIAN_HIGHWAYS)
from ..core.projection import LocalProjection, tile_bounds
from .places import Place, parse_places
from .terrain import HeightField
from ..traffic.traffic_vec import TileRoads
from ..traffic.core import TrafficSimulator
from ..vehicles.vehicle_instances import VEHICLE_STRIDE
from .road_layout import RoadLayout, road_width, road_surface_lift
from .urban import (build_neon, build_props, build_sidewalks, build_crosswalks,
                    crosswalk_bands, crosswalk_cutouts,
                    build_crowns, merge as merge_urban)
from .streetscape import plan_streetscape, build_roadside_details, align_streetlight_bases

try:
    import mapbox_earcut as earcut
except ImportError:
    earcut = None

BLDG_STRIDE = 17
# Exact land-use colour sits just above the terrain facets: below every road
# ribbon (0.06 m) so streets still cover lawns, but far above depth noise.
LANDCOVER_LIFT = 0.02


@dataclass
class MeshData:
    vertices: np.ndarray
    indices: np.ndarray
    edges: np.ndarray
    radius: float
    max_height: float
    # Optional spatial index: rows of [first_index, index_count, min xyz, max xyz]
    # (mesh-local coords).  Triangles are stored sorted by cell, so each row is a
    # contiguous index range the renderer can cull and draw independently.
    cells: np.ndarray | None = None
    gpu_vertices: np.ndarray | None = None
    bounds: np.ndarray | None = None
    lod_indices: np.ndarray | None = None
    lod_cells: np.ndarray | None = None
    packed_edges: np.ndarray | None = None
    neon_samples: np.ndarray | None = None


@dataclass
class ChunkResult:
    key: tuple
    b_mesh: MeshData | None
    g_mesh: MeshData | None
    w_mesh: MeshData | None
    trees:  np.ndarray | None
    lights: np.ndarray | None = None
    parked: dict = None
    paths:  list = None
    coarse: bool = False
    s_mesh: MeshData | None = None
    p_mesh: MeshData | None = None
    roads: TileRoads | None = None
    driving: object = None       # worker-prepared position-only collision/support
    minimap: object = None       # worker-prepared flat footprints for the inset map
    places: tuple | None = None  # named POIs; None identifies an older cache


@dataclass
class TileResult:
    key: tuple
    b_mesh: tuple | None
    g_mesh: tuple | None
    w_mesh: tuple | None
    trees:  np.ndarray | None
    lights: np.ndarray | None
    parked: dict
    paths:  list
    s_mesh: tuple | None = None
    p_mesh: tuple | None = None
    places: tuple = ()


def roof_color(wall_col):
    return np.clip(wall_col * 1.25 + 0.02, 0, 1).astype(np.float32)


def edge_color(wall_col):
    # trim round every roof
    return np.clip(wall_col * 0.62, 0, 1).astype(np.float32)


_SUSPENSION_NAMES = ("golden gate", "bay bridge", "brooklyn bridge", "george washington",
                     "verrazano", "tacoma narrows", "bosporus", "akashi")


def _bridge_paint(road):
    """Steel / stone colour for a bridge, keyed off its name and structure."""
    name = (getattr(road, "name", "") or "").lower()
    structure = (getattr(road, "structure", "") or "").lower()
    if "golden gate" in name:  return np.array([0.74, 0.28, 0.11], np.float32)
    if "bay bridge" in name:   return np.array([0.56, 0.58, 0.60], np.float32)
    if "brooklyn" in name:     return np.array([0.46, 0.42, 0.36], np.float32)
    if "suspension" in structure or "cable" in structure:
        return np.array([0.58, 0.32, 0.18], np.float32)
    return np.array([0.53, 0.52, 0.50], np.float32)


def _water_surface(poly, hf, sea_z):
    """Resolve coastal/explicit datums, otherwise retain the local bank heuristic.

    ``sea_z`` is the world sea datum after the terrain's absolute offset. Coastal
    and tidal water must share it even when a bank DEM sample is noisy. Untagged
    inland rivers keep their local bank level rather than cutting a sea canyon.
    """
    sea_z = float(sea_z)
    if isinstance(poly, WaterArea):
        if poly.coastal:
            return sea_z
        if poly.elevation is not None and math.isfinite(poly.elevation):
            return float(poly.elevation) + sea_z
        poly = poly.geometry
    if hf is None or poly.is_empty:
        return sea_z
    try:
        c = poly.representative_point()
        bed = float(hf.point(c.x, c.y))
        # Offshore polygons have a bed below the sea datum.  Inland rivers
        # and lakes have a bed above it, even when the DEM cell is wet.
        if bed <= sea_z + 1.0:
            return sea_z
        coords = np.asarray(poly.exterior.coords, np.float64)
        if len(coords) < 3:
            return sea_z
        count = min(48, len(coords) - 1)
        ids = np.linspace(0, len(coords) - 2, count).astype(np.int64)
        bank = np.asarray(hf.at(coords[ids, 0], coords[ids, 1]), np.float64)
        bank = bank[np.isfinite(bank)]
        if len(bank) == 0:
            return sea_z
        # The low bank is the actual waterline; a small drop keeps the surface
        # under the shore without exposing a cliff.
        return max(sea_z, float(np.percentile(bank, 25)) - 0.35)
    except Exception:
        return sea_z


def _water_regions(waters, hf, sea_z, bbox=None):
    """Resolve datums before clipping; union duplicates and give each XY one owner."""
    groups = {}
    tile = shapely.box(*map(float, bbox)) if bbox is not None else None
    for w in waters or ():
        for p in polygons_of(w):
            if p.is_empty or p.area < .5:
                continue
            level = _water_surface(w if isinstance(w, WaterArea) else p, hf, sea_z)
            scale = float(w.wave_scale) if isinstance(w, WaterArea) else 1.
            try:
                p = p if p.is_valid else make_valid(p)
                if tile is not None:
                    p = p.intersection(tile)
                if not p.is_empty:
                    groups.setdefault(level, []).append((p, scale))
            except Exception:
                continue
    result, occupied = [], Polygon()
    # Sea/tidal coverage wins over duplicate inland DEM guesses at the coast.
    order = sorted(groups, key=lambda level: (abs(level - float(sea_z)) > .01, level))
    for level in order:
        records = groups[level]
        union = unary_union([p for p, _ in records]).difference(occupied)
        for p in polygons_of(union):
            if p.area < .5:
                continue
            # Keep isolated pools calm even when an unrelated river shares ele.
            scale = max(s for source, s in records if source.intersects(p))
            result.append((p, float(level), scale))
        occupied = unary_union([occupied, union])
    return result


def ring_vertices(coords):
    return strip_ring(coords)


def _tri_normal(p0, p1, p2):
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-9 else np.array([0.0, 0.0, 1.0])


# ================================================================ vehicles
VEHICLE_SEDAN, VEHICLE_SUV, VEHICLE_COMPACT, VEHICLE_BUS, VEHICLE_TRUCK = range(5)
VEHICLE_TAXI, VEHICLE_VAN, VEHICLE_PICKUP, VEHICLE_X1 = 5, 6, 7, 8
VEHICLE_KINDS = 9
# A 100-slot kerbside mix: ordinary traffic first, then a deliberate 12% of BMW
# X1 so the model is part of the normal street rather than a rare find.
_KIND_TABLE = np.array([0]*32 + [1]*13 + [2]*11 + [3]*2 + [4]*2 + [5]*9 + [6]*6 + [7]*13 + [8]*12,
                       np.int32)
# Real-world fleet colour mix: lots of white / silver / black / grey, a little colour.
_VEHICLE_PAINTS = np.array([
    (0.90, 0.90, 0.90), (0.90, 0.90, 0.90), (0.90, 0.90, 0.90),          # white
    (0.68, 0.69, 0.72), (0.68, 0.69, 0.72), (0.68, 0.69, 0.72),          # silver
    (0.04, 0.04, 0.045), (0.04, 0.04, 0.045), (0.04, 0.04, 0.045),       # black
    (0.30, 0.31, 0.33), (0.30, 0.31, 0.33),                              # graphite
    (0.10, 0.16, 0.34),                                                  # blue
    (0.62, 0.07, 0.07), (0.62, 0.07, 0.07),                              # red
    (0.08, 0.22, 0.14), (0.55, 0.45, 0.30), (0.30, 0.05, 0.08),          # green, tan, maroon
], np.float32)
_GLASS_COLOR   = (0.08, 0.10, 0.13)
_WHEEL_COLOR   = (0.02, 0.02, 0.02)
_HEAD_COLOR    = (0.90, 0.90, 0.80)
_TAIL_COLOR    = (0.60, 0.06, 0.06)
_GLOW_COLOR    = (1.00, 0.92, 0.72)
_HEADLIGHT_COLOR  = (1.00, 0.88, 0.65)
_TAILLIGHT_COLOR  = (1.00, 0.15, 0.08)
_GROUND_GLOW_COLOR= (1.00, 0.85, 0.55)


def _deterministic_hash(x, y, slot):
    h = ((int(x * 100) * 73856093) ^ (int(y * 100) * 19349663) ^ (int(slot) * 83492791))
    return h & 0xFFFFFFFF


def _pick_paint(x, y, slot):
    return _VEHICLE_PAINTS[_deterministic_hash(x, y, slot) % len(_VEHICLE_PAINTS)]


def _pick_kind(x, y, slot):
    return int(_KIND_TABLE[_deterministic_hash(x, y, slot + 17) % len(_KIND_TABLE)])


def build_vehicle_template(kind):
    from ..vehicles.vehicle_models import build_template
    return build_template(kind)


def build_all_vehicle_templates():
    return {k: build_vehicle_template(k) for k in range(VEHICLE_KINDS)}


def _instance_row(x, y, heading, color, flags, z=0.0):
    c, s = math.cos(heading), math.sin(heading)
    row = np.zeros(VEHICLE_STRIDE, np.float32)
    row[0:4]   = (c, s, 0.0, 0.0)
    row[4:8]   = (-s, c, 0.0, 0.0)
    row[8:12]  = (0.0, 0.0, 1.0, 0.0)
    row[12:16] = (x, y, z, 1.0)
    row[16:19] = color
    row[19]    = flags
    row[23]    = 1.0
    return row

# ================================================================ parked fleet
# One kerb bay every 8 m with a little over a quarter of them taken: a street
# reads as lived-in without the kerb turning into a solid line of bodies.
PARKED_SPACING = 8.0
PARKED_CHANCE = 0.26
PARKED_MIN_WIDTH = 6.0          # narrower leaves no room beside the carriageway
PARKED_BOTH_SIDES_WIDTH = 8.0   # below this only one kerb is ever used
PARKED_MIN_SEPARATION = 4.0     # never overlap a car placed from a crossing path
PARKED_MAX_PER_TILE = 2400      # hard stop so a junction-heavy tile stays cheap
_PARK_CELL = PARKED_MIN_SEPARATION
_PARK_LIMIT2 = PARKED_MIN_SEPARATION ** 2


def _parked_slots(paths):
    """Kerb bays for a tile: both sides on wide streets, one side on narrow ones."""
    wide, narrow = [], []
    for p in paths:
        if p.width >= PARKED_BOTH_SIDES_WIDTH:
            wide.append(p)
        elif p.width >= PARKED_MIN_WIDTH:
            narrow.append(p)
    # Two cars facing each other across a 6.5 m street leave ~2.6 m between their
    # doors, so a narrow street parks one kerb only and alternates which one.
    left = [p for i, p in enumerate(narrow) if i % 2 == 0]
    right = [p for i, p in enumerate(narrow) if i % 2 == 1]
    slots = parking_slots_from_paths(wide, spacing=PARKED_SPACING)
    slots += parking_slots_from_paths(left, spacing=PARKED_SPACING, sides=(1.0,))
    slots += parking_slots_from_paths(right, spacing=PARKED_SPACING, sides=(-1.0,))
    return slots


def _parked_blocked(grid, x, y):
    """True when an accepted car already sits inside PARKED_MIN_SEPARATION."""
    gx, gy = int(math.floor(x / _PARK_CELL)), int(math.floor(y / _PARK_CELL))
    for ix in (gx - 1, gx, gx + 1):
        for iy in (gy - 1, gy, gy + 1):
            for px, py in grid.get((ix, iy), ()):
                if (x - px) ** 2 + (y - py) ** 2 < _PARK_LIMIT2:
                    return True
    return False


def _guarantee_x1(by_kind):
    """A scene that parks anything parks at least one BMW X1."""
    if not by_kind or VEHICLE_X1 in by_kind:
        return
    donor = max(by_kind, key=lambda k: (len(by_kind[k]), -k))
    row = by_kind[donor].pop()
    if not by_kind[donor]:
        del by_kind[donor]
    by_kind.setdefault(VEHICLE_X1, []).append(row)


def build_parked_vehicles(paths, rng=None):
    rng = rng or np.random.default_rng(0xC0FFEE)
    by_kind, grid, placed = {}, {}, 0
    for i, (x, y, dx, dy, z) in enumerate(_parked_slots(paths)):
        if placed >= PARKED_MAX_PER_TILE:
            break
        if rng.random() > PARKED_CHANCE:
            continue
        # Jitter first: the separation guarantee has to hold for where the car
        # actually lands, not for the middle of its kerb bay.
        heading = math.atan2(dy, dx) + rng.uniform(-0.05, 0.05)
        jitter = rng.uniform(-0.1, 0.1)
        nx, ny = -dy, dx
        xp, yp = x + nx * jitter, y + ny * jitter
        if _parked_blocked(grid, xp, yp):
            continue
        grid.setdefault((int(math.floor(xp / _PARK_CELL)), int(math.floor(yp / _PARK_CELL))),
                        []).append((xp, yp))
        placed += 1

        kind = _pick_kind(x, y, i)
        row = _instance_row(xp, yp, heading, _pick_paint(x, y, i), 0.0, z=z)
        by_kind.setdefault(kind, []).append(row)
    _guarantee_x1(by_kind)
    return {k: np.stack(v) for k, v in by_kind.items()}


def build_streetlights_from_paths(paths):
    """Curb lamps with a local ground height, including short OSM road segments."""
    lights, seen = [], set()
    # Carriageways of every drivable road: a lamp that lands inside a crossing
    # street (junction corners, short arms) would stand in the middle of the road.
    _walk = ('footway', 'pedestrian', 'steps', 'path', 'cycleway', 'bridleway', 'track', 'railway')
    road_geoms, road_owner = [], []
    for q in paths:
        if (q.tunnel or q.covered or q.bridge or q.layer > 0 or q.highway in _walk
                or (q.level is not None and q.level < 0.) or q.width < 3.0 or len(q.points) < 2):
            continue
        try:
            road_geoms.append(shapely.geometry.LineString(q.points).buffer(q.width * .5 + .1, cap_style=2))
            road_owner.append(id(q))
        except Exception:
            continue
    road_tree = shapely.STRtree(road_geoms) if road_geoms else None

    def in_other_road(x, y, me):
        if road_tree is None: return False
        pt = shapely.geometry.Point(x, y)
        return any(road_owner[j] != me for j in road_tree.query(pt, predicate='intersects'))

    for p in paths:
        if p.tunnel or p.covered or (p.level is not None and p.level < 0.): continue
        if p.total < 12.0:
            continue
        setback = min(8.0, p.total * 0.25)
        count = max(1, int(math.ceil((p.total - 2.0 * setback) / 28.0)) + 1)
        arcs = np.linspace(setback, p.total - setback, count) if p.total >= 28.0 else [p.total * 0.5]
        for i, arc in enumerate(arcs):
            x, y, heading = sample_path_position(p, float(arc))
            base = sample_path_z(p, float(arc))
            height = 9.5 if p.width >= 10.0 else 7.5
            sides = (1.0, -1.0) if p.width >= 18.0 else ((1.0 if i % 2 == 0 else -1.0),)
            for side in sides:
                nx = -math.sin(heading) * side * (p.width * 0.5 + 0.4)
                ny = math.cos(heading) * side * (p.width * 0.5 + 0.4)
                lx, ly = x + nx, y + ny
                if p.layer <= 0 and not p.bridge and in_other_road(lx, ly, id(p)):
                    continue
                key = (round(lx / 2.0), round(ly / 2.0), round(base / 3.0))
                if key in seen:
                    continue
                seen.add(key)
                lights.append((lx, ly, base + height, math.atan2(-ny, -nx), base))
    return np.asarray(lights, np.float32) if lights else None


# ================================================================ extruder
class MeshExtruder:
    SIMPLIFY_TOL = 0.1
    OUTLINE_REMNANT_INSET = 0.25
    OPEN_KINDS = frozenset(("roof", "carport", "canopy", "pavilion", "gazebo", "shed",
                           "greenhouse", "stable", "hangar"))
    DOUBLE_SIDED_PITCHED = True
    SMOOTH_COS = np.cos(np.radians(35))
    SHARP_COS = np.cos(np.radians(25))
    # Synthetic wedding-cake setbacks read as floating roofs/canopies half way up a
    # tower, so they are off.  Real OSM building:part tiers are still honoured.
    GENERATE_SETBACKS = False
    SETBACK_MIN_HEIGHT = 40.0
    SETBACK_MIN_AREA = 200.0
    CROWN_MIN_HEIGHT = 60.0
    SPIRE_MIN_HEIGHT = 85.0

    @staticmethod
    def sea_piers(buildings, waters=(), hf=None, sea_z=0.0, bbox=None):
        """Buildings standing in the sea as ``(id, footprint, base_z, top_z)``.

        Bridges crossing one of these must not punch their own piers through
        it: in the water the building itself is the support.
        """
        if not buildings or not waters:
            return []
        sea_z = float(sea_z)
        try:
            tile = shapely.box(*map(float, bbox)) if bbox is not None else None
        except Exception:
            tile = None
        polys = []
        for w in waters:
            try:
                # Only water that sits at the sea datum: an upland lake is not
                # a place where a bridge is carried on buildings.
                if _water_surface(w, hf, sea_z) > sea_z + .5:
                    continue
                for p in polygons_of(w):
                    if p.is_empty or p.area < .5:
                        continue
                    if tile is not None:
                        p = p.intersection(tile)
                    if not p.is_empty:
                        polys.append(p)
            except Exception:
                continue
        if not polys:
            return []
        try:
            water = unary_union(polys)
        except Exception:
            return []
        out = []
        for b in buildings:
            foot = getattr(b, 'footprint', None)
            if foot is None or foot.is_empty or foot.area < 1.:
                continue
            base = MeshExtruder._ground_elevation(b, hf)
            try:
                at_sea = foot.intersection(water).area >= .5 * foot.area
            except Exception:
                at_sea = False
            if not at_sea and hf is not None:
                # A tile holding only a sliver of coastline still has the sea
                # bed under a pier building: trust the DEM there.
                at_sea = base <= sea_z - .4
            if not at_sea:
                continue
            out.append((b.id, foot, base, base + float(b.height)))
        return out

    def build(self, buildings, roads, greens, waters, trees, paths=None,
              hf=None, bbox=None, sea_z=0.0, sea_piers=None):
        if sea_piers is None:
            sea_piers = self.sea_piers(buildings, waters, hf, sea_z, bbox)
            buildings, sea_piers = MeshExtruder.drop_deck_buildings(
                buildings, roads, hf, bbox, sea_z, sea_piers)
        b_mesh = self._build_buildings(buildings, hf) if buildings else None
        deck_pins = (pedestrian_bridge_pins(roads, hf, sea_z, bbox, buildings)
                     if roads else {})
        bridges = (self._build_bridges(roads, hf, bbox, sea_z, deck_pins, sea_piers)
                   if roads else None)
        b_mesh = self._merge_meshes(b_mesh, bridges)
        terrain = self._build_terrain(hf, greens, waters, bbox, sea_z)
        cover = self._build_landcover(greens, waters, hf, bbox)
        roads_mesh = self._build_ground(roads, hf, bbox, sea_z, deck_pins) if roads else None
        g_mesh = self._merge_meshes(terrain, cover)
        g_mesh = self._merge_meshes(g_mesh, roads_mesh)
        w_mesh = self._build_water(waters, sea_z, hf, bbox=bbox) if waters else None
        tree_hf = hf
        if terrain is not None and trees is not None and len(trees):
            # Shoreline notches/clamps change the rendered ground from the DEM.
            # Roots must sample the grid actually drawn, including those cuts.
            n = math.isqrt(len(terrain.vertices))
            tree_hf = HeightField(terrain.vertices[:, 2].reshape(n, n), *bbox)
        t_mesh = build_trees_mesh(trees, tree_hf)
        l_mesh = self._build_streetlights(roads, hf, bbox, sea_z) if roads else None
        parked = build_parked_vehicles(paths) if paths else {}
        return b_mesh, g_mesh, w_mesh, t_mesh, l_mesh, parked

    def _roof_safe(self, poly):
        if poly.is_empty: return False
        try:
            if len(poly.interiors): return False
            rect = poly.minimum_rotated_rectangle
            if rect.is_empty or rect.geom_type != "Polygon" or rect.area < 1e-3: return False
            return (poly.area / rect.area) > 0.85
        except Exception:
            return False

    def _face_append(self, verts, idx, p0, p1, p2, n, color, style, part_kind, emissive=0.0):
        b = 3 * len(verts)
        f = np.zeros((3, BLDG_STRIDE), np.float32)
        f[0, 0:3] = p0; f[1, 0:3] = p1; f[2, 0:3] = p2
        f[:, 3:6] = n
        f[:, 6:9] = color
        f[:, 9] = 1.0
        f[:, 10:13] = style
        f[:, 15] = emissive
        f[:, 16] = part_kind
        verts.append(f)
        idx.append(np.array([b, b + 1, b + 2], np.uint32))

    def _face_append_orient(self, verts, idx, p0, p1, p2, color, style, part_kind,
                            emissive=0.0, center=None):
        n = _tri_normal(p0, p1, p2)
        if center is not None:
            mid = (p0 + p1 + p2) / 3.0
            if np.dot(n, mid - center) < 0.0:
                p1, p2 = p2, p1
                n = -n
        self._face_append(verts, idx, p0, p1, p2, n, color, style, part_kind, emissive)

    # ================================================================ buildings
    def _needs_double_sided(self, b, roof_shape):
        if b.layer > 0 or b.kind in self.OPEN_KINDS: return True
        return self.DOUBLE_SIDED_PITCHED and roof_shape != "flat"

    def _make_cap(self, pts, tris, z, up, color, style, part_kind=1.0):
        if not up: tris = tris[:, [0, 2, 1]]
        v = np.zeros((len(pts), BLDG_STRIDE), np.float32)
        v[:, 0:2] = pts; v[:, 2] = z
        v[:, 5] = 1.0 if up else -1.0
        v[:, 6:9] = color; v[:, 9] = 1.0
        v[:, 10:13] = style; v[:, 16] = part_kind
        return v, tris

    @staticmethod
    def _resolve_outline_overlap(buildings):
        """Drop / trim building outlines that building:part footprints already cover.

        OSM maps a tower as an outline plus parts that share its walls.  Meshing both puts two
        coplanar wall layers on the same plane, which z-fights at any distance (no depth
        buffer separates identical depths). Parts win only where they overlap:
        a fully covered outline is removed, but uncovered wings are retained.
        """
        import dataclasses
        try:
            outlines = {i for i, b in enumerate(buildings) if b.layer <= 0}
            part_ids = [i for i, b in enumerate(buildings) if b.layer > 0]
            if not outlines or not part_ids:
                return buildings
            geoms = np.empty(len(part_ids), dtype=object)
            geoms[:] = [buildings[i].footprint for i in part_ids]
            tree = shapely.STRtree(geoms)
            out = []
            for i, b in enumerate(buildings):
                fp = b.footprint
                if i not in outlines or fp.is_empty or fp.area < 1.0:
                    out.append(b); continue
                hit = tree.query(fp, predicate="intersects")
                if len(hit) == 0:
                    out.append(b); continue
                # Parts define the real solid volume, including raised spans.
                # Keeping a ground-level outline below raised parts fills passages.
                cover = shapely.union_all(geoms[hit])
                inter = fp.intersection(cover).area
                if inter < .1:
                    out.append(b); continue
                rest = fp.difference(cover.buffer(0.02))
                # Keep every real uncovered wing; even a small raised part must
                # override its outline. Emit individual polygons, never an
                # unexpected MultiPolygon in downstream footprint consumers.
                # Slight outline/part disagreement can leave long centimeter-
                # thick scraps whose area still exceeds the sanitizer's minimum.
                # Extruding them to the outline height creates paper-thin towers.
                # Filter only these derived remnants, never mapped thin parts.
                out.extend(dataclasses.replace(b, footprint=q) for q in sanitize_footprint(rest)
                           if not q.buffer(-MeshExtruder.OUTLINE_REMNANT_INSET).is_empty)
            return out
        except Exception as exc:                               # never let this break a tile
            print(f"[mesh] outline/part overlap pass skipped: {exc}")
            return buildings

    # 'drop' removes only thin hovering slabs; 'fill' also extends fully
    # unsupported parts down; 'off' disables the pass.
    FLOATING_PARTS = 'fill'

    @staticmethod
    def _ground_floating_parts(buildings):
        """Remove hovering slabs; fill a gap only when it cannot overlap anything.

        Extending a part downward must never create a second wall on top of
        another building's wall (that z-fights). So a part is only extended when
        nothing else occupies its plan area; partially supported parts are left
        exactly as mapped.
        """
        import dataclasses
        mode = MeshExtruder.FLOATING_PARTS
        if mode == 'off': return buildings
        try:
            raised = [i for i, b in enumerate(buildings)
                      if b.min_height > 3.0 and b.kind not in ('bridge', 'roof', 'canopy', 'carport')
                      and b.footprint is not None and not b.footprint.is_empty]
            if not raised: return buildings
            geoms = np.empty(len(buildings), dtype=object)
            geoms[:] = [b.footprint for b in buildings]
            tree = shapely.STRtree(geoms)
            drop, lowered = set(), {}
            for i in raised:
                b = buildings[i]
                fp = b.footprint
                if fp.area < 1.0: continue
                near = [j for j in tree.query(fp, predicate='intersects') if j != i]
                support = [buildings[j].footprint for j in near
                           if buildings[j].min_height < b.min_height - .5
                           and buildings[j].height >= b.min_height - .75]
                cover = shapely.union_all(support).buffer(.3) if support else None
                bare = fp.difference(cover).area if cover is not None else fp.area
                if bare < .9 * fp.area: continue          # mostly supported: leave as mapped
                if b.height - b.min_height <= 9.0 and b.min_height >= 8.0:
                    drop.add(i); continue                  # thin hovering slab
                if mode != 'fill': continue
                # Any neighbour sharing plan area would get coplanar walls: don't extend.
                crowd = sum(fp.intersection(buildings[j].footprint).area for j in near)
                if crowd > .02 * fp.area: continue
                lowered[i] = 0.0
            if not drop and not lowered: return buildings
            return [dataclasses.replace(b, min_height=lowered[i]) if i in lowered else b
                    for i, b in enumerate(buildings) if i not in drop]
        except Exception as exc:
            print(f"[mesh] floating-part pass skipped: {exc}")
            return buildings

    @staticmethod
    def prepare_buildings(buildings, roads=(), hf=None, bbox=None, sea_z=0.0):
        """Resolve parts and explicit surface passages without clipping roads.

        Only mapped covered passages carve the lower storeys; untagged overlaps
        are not evidence to delete buildings or reroute traffic. The original
        footprint/roof is retained above the passage clearance. Buildings a
        bridge deck runs through were already removed by
        :meth:`drop_deck_buildings`.
        """
        from dataclasses import replace
        if hf is not None:
            buildings = [replace(b, ground_elevation=MeshExtruder._ground_elevation(b, hf))
                         if b.ground_elevation is None else b for b in buildings]
        buildings = MeshExtruder._resolve_outline_overlap(buildings)
        buildings = MeshExtruder._ground_floating_parts(buildings)
        passages = [r for r in roads if r.covered and not r.tunnel
                    and not (r.level is not None and r.level < 0.)]
        if not passages or not buildings:
            return buildings
        corridors = [r.geometry.buffer(road_width(r) * .5 + .35,
                                       cap_style=2, join_style=2) for r in passages]
        tree = shapely.STRtree(corridors)
        out = []
        for b in buildings:
            hits = tree.query(b.footprint, predicate='intersects')
            if not len(hits):
                out.append(b); continue
            datum = MeshExtruder._ground_elevation(b, hf)
            ceiling = b.min_height
            cuts, roof_cuts = [], []
            for j in hits:
                r = passages[j]
                pts = _densify(np.asarray(r.geometry.coords))
                arc, zs = road_profile(pts, r, hf, sea_z, bbox)
                crossing = corridors[j].intersection(b.footprint)
                passage_ceiling = b.min_height
                for poly in polygons_of(crossing):
                    if poly.area < .01: continue
                    s = [r.geometry.project(Point(p)) for p in poly.exterior.coords]
                    samples = np.r_[min(s), arc[(arc > min(s)) & (arc < max(s))], max(s)]
                    height = float(np.max(np.interp(samples, arc, zs))) - datum
                    clearance = r.clearance or (2.8 if r.highway in PEDESTRIAN_HIGHWAYS else 4.5)
                    candidate = height + clearance + .15
                    if not r.clearance:
                        # A mapped low canopy is evidence against our generic
                        # clearance/DEM estimate, not permission to delete its
                        # entire roof when one terrain sample reaches above it.
                        # Explicit maxheight remains authoritative.
                        candidate = min(candidate, b.height - .25)
                    passage_ceiling = max(passage_ceiling, candidate)
                if passage_ceiling <= b.min_height + .1:
                    continue
                cuts.append(corridors[j])
                if passage_ceiling >= b.height - .1:
                    # Surveyed clearance above this roof opens only this
                    # corridor, not other covered paths under the same building.
                    roof_cuts.append(corridors[j])
                else:
                    ceiling = max(ceiling, passage_ceiling)
            if not cuts:
                out.append(b); continue
            if ceiling > b.min_height + .1:
                lower = b.footprint.difference(shapely.union_all(cuts))
                for poly in sanitize_footprint(lower):
                    out.append(replace(b, footprint=poly, height=ceiling,
                                       roof_shape='flat', roof_height=0., layer=1,
                                       ground_elevation=datum, street_details=False))
            if b.height > ceiling + .1:
                upper = (sanitize_footprint(b.footprint.difference(shapely.union_all(roof_cuts)))
                         if roof_cuts else [b.footprint])
                out.extend(replace(b, footprint=poly, min_height=ceiling, layer=1,
                                   ground_elevation=datum, street_details=False) for poly in upper)
        return out

    @staticmethod
    def drop_deck_buildings(buildings, roads=(), hf=None, bbox=None, sea_z=0.0,
                            sea_piers=None):
        """Remove sea buildings that a bridge deck runs through.

        A tower standing in the water with the roadway cutting through it reads
        as a broken building, and splitting or piercing it leaves a shape that
        is mapped nowhere.  Such a building is simply not drawn: the deck
        carries itself on its own piers and pylons there, as it does beside the
        old shoreline.  Buildings whose roof stays below the deck are kept:
        they can carry it (see ``_build_bridges``), so their ids also stay in
        the sea-pier list and the deck keeps its support.  Removed ids leave
        that list, or the deck would skip a pier where nothing stands.
        """
        if not buildings or not sea_piers or not roads:
            return buildings, sea_piers
        corridors = []
        for r in roads:
            if not (r.bridge or r.layer > 0) or r.tunnel:
                continue
            if r.level is not None and r.level < 0.:
                continue
            try:
                width = road_width(r)
                if width < 1.:
                    continue
                corridors.append((r, r.geometry.buffer(width * .5 + .6,
                                                       cap_style=2, join_style=2)))
            except Exception:
                continue
        if not corridors:
            return buildings, sea_piers
        by_id = {entry[0]: entry for entry in sea_piers}
        removed = set()
        for b in buildings:
            if by_id.get(b.id) is None:
                continue
            datum = MeshExtruder._ground_elevation(b, hf)
            roof = datum + float(b.height)
            for r, corridor in corridors:
                try:
                    if not corridor.intersects(b.footprint):
                        continue
                    pts = _densify(np.asarray(r.geometry.coords, np.float64), 4.0)
                    arc, zs = road_profile(pts, r, hf, sea_z, bbox)
                    if len(arc) < 2:
                        continue
                    crossing = corridor.intersection(b.footprint)
                    if crossing.is_empty:
                        continue
                    sample = crossing.representative_point()
                    arc_at = float(LineString(pts).project(Point(sample.x, sample.y)))
                    deck = float(np.interp(arc_at, arc, zs))
                except Exception:
                    continue
                if deck - datum <= roof - .5:
                    removed.add(b.id)
                    break
        if not removed:
            return buildings, sea_piers
        kept = [b for b in buildings if b.id not in removed]
        piers = [entry for entry in sea_piers if entry[0] not in removed]
        return kept, piers

    @staticmethod
    def _ground_elevation(b, hf=None):
        if b.ground_elevation is not None:
            return float(b.ground_elevation)
        if hf is None:
            return 0.
        try:
            ring = np.asarray(b.footprint.exterior.coords, np.float64)
            return float(np.min(hf.at(ring[:, 0], ring[:, 1]))) - .5
        except Exception:
            return float(hf.point(b.footprint.centroid.x, b.footprint.centroid.y)) - .5

    def _build_buildings(self, buildings, hf=None):
        buildings = self._resolve_outline_overlap(buildings)
        ground_z = np.array([self._ground_elevation(b, hf) for b in buildings], np.float64)
        parts, part_bld = [], []
        for i, b in enumerate(buildings):
            try:
                for q in sanitize_footprint(b.footprint, self.SIMPLIFY_TOL, 1.0):
                    parts.append(q); part_bld.append(i)
            except Exception as exc:
                print(f"[mesh] building {b.id}: footprint rejected ({exc})")
        parts, part_bld = self._drop_duplicate_footprints(parts, part_bld, buildings)

        roof_bias = np.array([0.15 * _jitter(b.id + 7919) for b in buildings], np.float64)
        ranks = self._conflict_ranks(buildings, parts, part_bld, roof_bias, ground_z)

        shells = []
        for pi, (part, bi_bld) in enumerate(zip(parts, part_bld)):
            b = buildings[bi_bld]
            z0 = b.min_height - 0.1 + ground_z[bi_bld]
            z1 = b.height + roof_bias[bi_bld] + ground_z[bi_bld]
            try:
                tiers = self._split_into_tiers(part, z0, z1, b)
            except Exception:
                tiers = [([part], z0, z1)]
            for ti, (tps, tz0, tz1) in enumerate(tiers):
                is_top = ti == len(tiers) - 1
                for tp in tps:
                    roof_shape = b.roof_shape if is_top else "flat"
                    if roof_shape != "flat" and not self._roof_safe(tp):
                        roof_shape = "flat"
                    shells.append((tp, tz0, tz1, is_top, bi_bld, roof_shape, ti, pi))

        split_shells = []
        base_z = 4.5
        for (poly, tz0, tz1, is_top, bi_bld, roof_shape, ti, pi) in shells:
            if tz1 > base_z:
                if tz0 < base_z:
                    split_shells.append((poly, tz0, base_z, False, bi_bld, "flat", 0, 6, pi, False))
                # tz0 < base_z means real material continues below the split, so the upper
                # shell's underside is an interior diaphragm, not a raised building bottom.
                split_shells.append((poly, max(base_z, tz0), tz1, is_top, bi_bld,
                                     roof_shape, ti, 0, pi, tz0 < base_z - 0.05))
            else:
                split_shells.append((poly, tz0, tz1, is_top, bi_bld, roof_shape, ti, 6, pi, False))

        rings, ring_bld, polys, roof_types = [], [], [], []
        z0s, z1s, wall_cols, roof_cols, edge_cols, tier_styles = [], [], [], [], [], []
        dside, first_tier, part_kinds = [], [], []
        is_tops, shell_open, hide_downs = [], [], []
        for (poly, tz0, tz1, is_top, bi_bld, roof_shape, ti, part_kind, pi, hide_down) in split_shells:
            b = buildings[bi_bld]
            bi = len(z0s)
            z0s.append(tz0); z1s.append(tz1)
            wall_cols.append(b.color)
            roof_cols.append(b.roof_color if is_top else b.roof_color * 0.85)
            edge_cols.append(edge_color(b.color))
            polys.append(poly)
            roof_types.append(roof_shape)
            dside.append(self._needs_double_sided(b, roof_shape))
            first_tier.append(ti == 0)
            part_kinds.append(part_kind)
            is_tops.append(is_top)
            shell_open.append(b.kind in self.OPEN_KINDS)
            hide_downs.append(hide_down)
            mk = (b.style if b.style >= 0 else
                  b.material_kind if b.material_kind >= 0 else int(theme_code(b.kind)))
            # style.x doubles as the depth-rank carrier (rank + jitter, jitter clamped
            # so floor(style.x) in the vertex shader can recover the rank exactly).
            jitter = min(_jitter(b.id + ti * 23), 0.98)
            tier_styles.append(np.array([float(ranks[pi]) + jitter, float(mk), float(tz1)], np.float32))
            for ring in (poly.exterior, *poly.interiors):
                r = ring_vertices(np.asarray(ring.coords))
                if len(r) >= 3:
                    rings.append(r); ring_bld.append(bi)

        if not rings:
            return None

        z0 = np.asarray(z0s, np.float64); z1 = np.asarray(z1s, np.float64)
        rb = np.asarray(ring_bld, np.int64)
        style_arr = np.stack(tier_styles)
        dside_arr = np.asarray(dside, bool)
        pk_arr = np.asarray(part_kinds, np.float32)

        wall_v, wall_i, lines = self._walls_batch(
            rings, z0[rb], z1[rb],
            np.asarray(wall_cols, np.float32)[rb],
            np.asarray(edge_cols, np.float32)[rb],
            style_arr[rb], dsided=dside_arr[rb],
            part_kinds=pk_arr[rb])
        V, I = [wall_v], [wall_i]; nv = len(wall_v)

        cap_V, cap_I, cap_nv = [], [], 0
        for bi, poly in enumerate(polys):
            try:
                pts, tris = self._triangulate(poly)
            except Exception as exc:
                print(f"[mesh] cap triangulation failed: {exc}")
                continue
            if len(tris) == 0: continue
            jobs = []
            # The base split seals every tall shell with an up-cap at base_z, but the
            # shell above always covers that plane: it is interior geometry (about 12%
            # of a dense tile's building triangles).  Open kinds keep it - you can see
            # into those - and genuine roofs (is_top / non-base heights) always emit.
            if is_tops[bi] or shell_open[bi] or abs(float(z1[bi]) - base_z) > 1e-9:
                jobs.append((float(z1[bi]), True, np.asarray(roof_cols[bi], np.float32), pk_arr[bi]))
            if dside_arr[bi]:
                jobs.append((float(z1[bi]), False, np.asarray(roof_cols[bi], np.float32) * 0.8, pk_arr[bi]))
            # The matching underside at base_z is only a real bottom when the building
            # is raised above the ground; hide_down marks the split case where solid
            # material continues below instead.
            if (first_tier[bi] and z0[bi] > 0.5 and not hide_downs[bi]) or dside_arr[bi]:
                jobs.append((float(z0[bi]), False, np.asarray(wall_cols[bi], np.float32) * 0.6, pk_arr[bi]))
            for zc, up, col, pk in jobs:
                v, t = self._make_cap(pts, tris, zc, up, col, style_arr[bi], pk)
                cap_V.append(v); cap_I.append((t + cap_nv).ravel()); cap_nv += len(v)
        if cap_V:
            I.append(np.concatenate(cap_I).astype(np.int64) + nv)
            V.append(np.vstack(cap_V)); nv += cap_nv

        for bi, poly, rt in zip(range(len(polys)), polys, roof_types):
            if rt == "flat": continue
            style = style_arr[bi].astype(np.float32)
            color = np.asarray(roof_cols[bi], np.float32)
            builder = {
                "pyramidal":   self._roof_pyramidal,
                "hipped":      self._roof_pyramidal,
                "half-hipped": self._roof_pyramidal,
                "gabled":      self._roof_gabled,
                "dome":        self._roof_dome,
                "skillion":    self._roof_skillion,
            }.get(rt, self._roof_pyramidal)
            try:
                rv, ri = builder(poly, float(z1[bi]), color, style)
            except Exception as exc:
                print(f"[mesh] {rt} roof skipped: {exc}")
                continue
            if rv is None or len(ri) == 0: continue
            I.append(ri.ravel() + nv); V.append(rv); nv += len(rv)

        top_of = {}
        for (poly, tz0, tz1, is_top, bld_idx, rt, ti, pk, pi, hide_down) in split_shells:
            if not is_top: continue
            cur = top_of.get(bld_idx)
            if cur is None or poly.area > cur[0].area:
                top_of[bld_idx] = (poly, float(tz1))
        for bi_bld, (poly_top, z_top) in top_of.items():
            b = buildings[bi_bld]
            if b.roof_shape != "flat" or b.kind in self.OPEN_KINDS or b.layer > 0:
                continue
            if b.height < 6.0:
                continue
            seed = _jitter(b.id + 7919)
            try:
                v, idx = self._rooftop(poly_top, z_top, b, seed)
                if v is not None:
                    I.append(idx + nv); V.append(v); nv += len(v)
            except Exception as exc:
                print(f"[mesh] roof-top detail skipped for {b.id}: {exc}")

        vertices = np.ascontiguousarray(np.vstack(V), dtype=np.float32)
        indices = np.ascontiguousarray(np.concatenate(I), dtype=np.uint32)
        edges = np.ascontiguousarray(lines, dtype=np.float32)
        radius = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
        return MeshData(vertices, indices, edges, radius, float(vertices[:, 2].max()))

    def _split_into_tiers(self, poly, z0, z1, b):
        whole = [([poly], z0, z1)]
        h = z1 - z0
        if not self.GENERATE_SETBACKS:
            return whole
        if (b.layer > 0 or b.roof_shape != "flat"
                or h < self.SETBACK_MIN_HEIGHT or poly.area < self.SETBACK_MIN_AREA):
            return whole
        try:
            n = 3 if h > 80 else 2
            base_area = poly.area
            tier_polys = []
            for i in range(n):
                frac = i / max(1, n - 1)
                inset = frac * min(10.0, math.sqrt(base_area) * 0.18)
                if inset < 0.3:
                    subs = [poly]
                else:
                    subs = sanitize_footprint(poly.buffer(-inset), self.SIMPLIFY_TOL, 1.0)
                if not subs or sum(sp.area for sp in subs) < base_area * 0.15:
                    break
                tier_polys.append(subs)
            if len(tier_polys) < 2:
                return whole
            fracs = ([(0.0, 0.7), (0.7, 1.0)] if len(tier_polys) == 2
                     else [(0.0, 0.55), (0.55, 0.82), (0.82, 1.0)])
            return [(subs, z0 + h * fa, z0 + h * fb) for subs, (fa, fb) in zip(tier_polys, fracs)]
        except Exception:
            return whole

    @staticmethod
    def _drop_duplicate_footprints(parts, part_bld, buildings, iou_min=0.85):
        n = len(parts)
        if n < 2: return parts, part_bld
        try:
            arr = np.empty(n, dtype=object); arr[:] = parts
            tree = shapely.STRtree(arr)
            left, right = tree.query(arr, predicate="intersects")
            m = left < right
            left, right = left[m], right[m]
            if len(left) == 0: return parts, part_bld
            inter = shapely.area(shapely.intersection(arr[left], arr[right]))
            areas = shapely.area(arr)
            iou = inter / np.maximum(areas[left] + areas[right] - inter, 1e-9)
            hit = iou > iou_min
            drop = np.zeros(n, bool)
            for a, b in zip(left[hit], right[hit]):
                if drop[a] or drop[b] or part_bld[a] == part_bld[b]: continue
                ba, bb = buildings[part_bld[a]], buildings[part_bld[b]]
                if abs(ba.min_height - bb.min_height) > 0.5: continue
                if (bb.height, areas[b]) > (ba.height, areas[a]): drop[a] = True
                else: drop[b] = True
            if not drop.any(): return parts, part_bld
            keep = ~drop
            return ([p for p, k in zip(parts, keep) if k],
                    [i for i, k in zip(part_bld, keep) if k])
        except Exception as exc:
            print(f"[mesh] duplicate-footprint pass skipped: {exc}")
            return parts, part_bld

    @staticmethod
    def _conflict_ranks(buildings, parts, part_bld, roof_bias, ground_z,
                        gap=0.25, z_min=1.5, ov_min=0.5):
        """Depth ranks that keep near-coplanar duplicate surfaces apart.

        OSM freely overlaps outlines, building:parts and neighbouring footprints, so
        two same-facing wall planes a few millimetres apart (or two roof caps within
        centimetres of height) tie in the depth buffer: the winner flips per pixel and
        the facade shimmers in a stipple pattern at any range.  A geometry gap only
        holds until it drops below the depth quantum with distance, so instead each
        footprint gets a small integer rank that the vertex shader turns into an
        eye-ward nudge of >= 2 quanta (see layerRank in shaders/building.py).
        Conflicting footprints must land in different ranks - a greedy colouring of
        the conflict graph, whose edges are wall pairs that are parallel, same-facing,
        within `gap` metres and sharing height, plus footprint pairs whose roof caps
        coincide.  The rank travels in style.x as rank + jitter.
        """
        n = len(parts)
        if n < 2:
            return np.zeros(n, np.int64)
        try:
            def z_range(pi):
                b = buildings[part_bld[pi]]
                gz = float(ground_z[part_bld[pi]]) if ground_z is not None else 0.0
                return (float(b.min_height) - 0.1 + gz,
                        float(b.height) + float(roof_bias[part_bld[pi]]) + gz)

            # ---- wall segments: position, outward facing, owning part ---------
            segs, lines, seg_pi = [], [], []
            for pi, poly in enumerate(parts):
                for ring in (poly.exterior, *poly.interiors):
                    c = np.asarray(ring.coords)[:-1]
                    m = len(c)
                    for k in range(m):
                        a, b2 = c[k], c[(k + 1) % m]
                        d = b2 - a
                        L = float(np.hypot(*d))
                        if L < 0.05:
                            continue
                        dn = d / L
                        nr = np.array([dn[1], -dn[0]])
                        if poly.contains(Point((a + b2) * 0.5 + 1e-3 * nr)):
                            nr = -nr
                        segs.append((a, b2, nr))
                        seg_pi.append(pi)
                        lines.append(LineString([a, b2]))

            edges = set()
            zr = [z_range(p) for p in range(n)]
            if lines:
                tree = shapely.STRtree(lines)
                li, ri = tree.query(lines, predicate="dwithin", distance=gap)
                keep = li < ri
                for i, j in zip(li[keep], ri[keep]):
                    pa, pb = seg_pi[i], seg_pi[j]
                    if pa == pb:
                        continue
                    a0, a1, na = segs[i]
                    b0, b1, nb = segs[j]
                    if float(np.dot(na, nb)) < 0.5:
                        continue                       # back-to-back: culling already
                    if min(zr[pa][1], zr[pb][1]) - max(zr[pa][0], zr[pb][0]) < z_min:
                        continue                       # no shared height range
                    la = float(np.linalg.norm(a1 - a0))
                    lb = float(np.linalg.norm(b1 - b0))
                    da = (a1 - a0) / la
                    db = (b1 - b0) / lb

                    def cr(d, p, q):
                        return float(d[0] * (p[1] - q[1]) - d[1] * (p[0] - q[0]))

                    sa0, sa1 = cr(db, a0, b0), cr(db, a1, b0)
                    sb0, sb1 = cr(da, b0, a0), cr(da, b1, a0)
                    if max(abs(sa0), abs(sa1), abs(sb0), abs(sb1)) > gap:
                        continue                       # not near-coplanar
                    # epsilon: an endpoint landing 1e-17 on the far side is dust, not a
                    # real crossing - those pairs are the most coplanar of all.
                    if sa0 * sa1 < -1e-12 or sb0 * sb1 < -1e-12:
                        continue                       # planes cross mid-segment
                    ta = sorted((float(np.dot(a0 - b0, db)), float(np.dot(a1 - b0, db))))
                    tb = sorted((float(np.dot(b0 - a0, da)), float(np.dot(b1 - a0, da))))
                    ov = min(max(0.0, min(ta[1], lb) - max(ta[0], 0.0)),
                             max(0.0, min(tb[1], la) - max(tb[0], 0.0)))
                    if ov < ov_min:
                        continue
                    edges.add((pa, pb) if pa < pb else (pb, pa))

            # ---- coincident roof caps ----------------------------------------
            arr = np.empty(n, dtype=object)
            arr[:] = parts
            tree2 = shapely.STRtree(arr)
            ci, cj = tree2.query(arr, predicate="intersects")
            for i, j in zip(ci, cj):
                i, j = int(i), int(j)
                if i >= j:
                    continue
                inter = shapely.area(shapely.intersection(arr[i], arr[j]))
                if inter < 0.5:
                    continue
                if abs(zr[i][1] - zr[j][1]) <= gap:
                    edges.add((i, j))

            # ---- greedy colouring --------------------------------------------
            adj = {}
            for i, j in edges:
                adj.setdefault(i, set()).add(j)
                adj.setdefault(j, set()).add(i)
            rank = np.zeros(n, np.int64)
            done = np.zeros(n, bool)
            for i in sorted(adj, key=lambda k: (-len(adj[k]), k)):
                used = {int(rank[j]) for j in adj[i] if done[j]}
                r = 0
                while r in used:
                    r += 1
                rank[i] = r
                done[i] = True
            if rank.max(initial=0) > 15:
                print(f"[mesh] conflict ranks dense (max {rank.max()}), clamped")
                np.minimum(rank, 15, out=rank)
            return rank
        except Exception as exc:                        # never let this break a tile
            print(f"[mesh] conflict-rank pass skipped: {exc}")
            return np.zeros(n, np.int64)

    def _walls_batch(self, rings, z0, z1, col, edge_col, style, dsided=None, want_lines=True, part_kinds=None):
        if dsided is None: dsided = np.zeros(len(rings), bool)
        if part_kinds is None: part_kinds = np.zeros(len(rings), np.float32)
        valid_rings, valid_z0, valid_z1, valid_col, valid_ecol, valid_style, valid_ds, valid_pk = [], [], [], [], [], [], [], []
        for r, z0v, z1v, cv, ev, sv, dv, pk in zip(rings, z0, z1, col, edge_col, style, dsided, part_kinds):
            if np.isfinite(r).all() and np.max(np.hypot(r[:, 0], r[:, 1])) < 5000.0:
                valid_rings.append(r); valid_z0.append(z0v); valid_z1.append(z1v)
                valid_col.append(cv); valid_ecol.append(ev); valid_style.append(sv); valid_ds.append(dv); valid_pk.append(pk)
        if not valid_rings:
            return (np.empty((0, BLDG_STRIDE), np.float32),
                    np.empty(0, np.uint32),
                    np.empty((0, 6), np.float32))
        rings = valid_rings
        z0 = np.array(valid_z0); z1 = np.array(valid_z1); ds_ring = np.array(valid_ds, bool)
        col = np.array(valid_col); edge_col = np.array(valid_ecol); style = np.array(valid_style); pk = np.array(valid_pk)

        counts = np.fromiter((len(r) for r in rings), np.int64, len(rings))
        starts = np.concatenate(([0], np.cumsum(counts)))
        v = np.concatenate(rings); n = len(v)
        rep = np.repeat(np.arange(len(counts)), counts)
        z0v, z1v = z0[rep], z1[rep]
        colv, ecolv = col[rep], edge_col[rep]; stylev = style[rep]; pkv = pk[rep]
        idx = np.arange(n); nxt = idx + 1
        nxt[starts[1:] - 1] = starts[:-1]
        prv = idx - 1; prv[starts[:-1]] = starts[1:] - 1
        p0, p1 = v, v[nxt]
        d = p1 - p0; d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        en = np.stack([d[:, 1], -d[:, 0]], axis=1); prev_n = en[prv]
        def blend(a, other):
            s = a + other; s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-9)
            return np.where(((a * other).sum(1) > self.SMOOTH_COS)[:, None], s, a)
        n_start, n_end = blend(en, prev_n), blend(en, en[nxt])
        dp, dn = d[prv], d[nxt]
        ao_s = np.where(dp[:, 0] * d[:, 1] - dp[:, 1] * d[:, 0] < -1e-3, 0.78, 1.0)
        ao_e = np.where(d[:, 0] * dn[:, 1] - d[:, 1] * dn[:, 0] < -1e-3, 0.78, 1.0)
        L = np.linalg.norm(p1 - p0, axis=1)

        w = np.zeros((n, 4, BLDG_STRIDE), np.float32)
        w[:, 0, 0:2], w[:, 3, 0:2] = p0, p0
        w[:, 1, 0:2], w[:, 2, 0:2] = p1, p1
        w[:, 0, 2], w[:, 1, 2] = z0v, z0v
        w[:, 2, 2], w[:, 3, 2] = z1v, z1v
        w[:, 0, 3:5], w[:, 3, 3:5] = n_start, n_start
        w[:, 1, 3:5], w[:, 2, 3:5] = n_end, n_end
        w[:, :, 5] = 0.0; w[:, :, 6:9] = colv[:, None, :]
        w[:, 0, 9], w[:, 3, 9] = ao_s, ao_s
        w[:, 1, 9], w[:, 2, 9] = ao_e, ao_e
        w[:, :, 10:13] = stylev[:, None, :]
        w[:, 0, 13] = 0; w[:, 1, 13] = L; w[:, 2, 13] = L; w[:, 3, 13] = 0
        w[:, :, 14] = L[:, None]
        w[:, :, 15] = 0.0
        w[:, :, 16] = pkv[:, None]

        quad = np.arange(n)[:, None] * 4
        wi = (quad + np.array([0, 1, 2, 0, 2, 3])).ravel()
        wv = w.reshape(-1, BLDG_STRIDE)
        ds_edge = ds_ring[rep]
        if ds_edge.any():
            inner = w[ds_edge].copy()
            inner[:, :, 3:6] *= -1.0
            k = len(inner)
            ibase = n * 4 + np.arange(k)[:, None] * 4
            wi = np.concatenate([wi, (ibase + np.array([0, 3, 2, 0, 2, 1])).ravel()])
            wv = np.concatenate([wv, inner.reshape(-1, BLDG_STRIDE)])
        if not want_lines:
            return wv, wi, np.empty((0, 6), np.float32)
        roof = np.empty((n, 2, 6), np.float32)
        roof[:, 0, 0:2], roof[:, 1, 0:2] = p0, p1
        roof[:, :, 2] = z1v[:, None]
        roof[:, :, 3:6] = ecolv[:, None, :]
        sharp = (en * prev_n).sum(1) < self.SHARP_COS
        vert = np.empty((int(sharp.sum()), 2, 6), np.float32)
        if sharp.any():
            vert[:, :, 0:2] = v[sharp][:, None, :]
            vert[:, 0, 2], vert[:, 1, 2] = z0v[sharp], z1v[sharp]
            vert[:, :, 3:6] = ecolv[sharp][:, None, :] * 0.55
        lines = np.concatenate([roof, vert]).reshape(-1, 6)
        return wv, wi, lines

    # ================================================================ roofs
    def _roof_pyramidal(self, poly, z_top, color, style):
        coords = np.asarray(poly.exterior.coords)[:-1]
        if len(coords) < 3: return None, None
        cx, cy = poly.centroid.x, poly.centroid.y
        span = math.sqrt(poly.area)
        h = max(2.5, min(15.0, span * 0.35))
        apex = np.array([cx, cy, z_top + h])
        verts, idx = [], []
        n_total = len(coords)
        for i in range(n_total):
            p0 = np.array([coords[i][0], coords[i][1], z_top])
            p1 = np.array([coords[(i + 1) % n_total][0], coords[(i + 1) % n_total][1], z_top])
            nv = _tri_normal(p0, p1, apex)
            self._face_append(verts, idx, p0, p1, apex, nv, color, style, 1)
        return np.vstack(verts), np.concatenate(idx)

    def _roof_gabled(self, poly, z_top, color, style):
        rect = poly.minimum_rotated_rectangle
        coords = np.asarray(rect.exterior.coords)[:-1]
        if len(coords) != 4: return self._roof_pyramidal(poly, z_top, color, style)
        e01 = coords[1] - coords[0]; e12 = coords[2] - coords[1]
        if np.linalg.norm(e01) >= np.linalg.norm(e12):
            l1, l2 = coords[0], coords[1]
            r1, r2 = coords[3], coords[2]
            short_len = np.linalg.norm(e12)
        else:
            l1, l2 = coords[1], coords[2]
            r1, r2 = coords[0], coords[3]
            short_len = np.linalg.norm(e01)
        h = max(2.0, min(10.0, short_len * 0.35))
        center = np.array([poly.centroid.x, poly.centroid.y, z_top])
        ridge_a = np.array([(l1[0] + r1[0]) * 0.5, (l1[1] + r1[1]) * 0.5, z_top + h])
        ridge_b = np.array([(l2[0] + r2[0]) * 0.5, (l2[1] + r2[1]) * 0.5, z_top + h])
        verts, idx = [], []
        p0 = np.array([l1[0], l1[1], z_top]); p1 = np.array([l2[0], l2[1], z_top])
        self._face_append_orient(verts, idx, p0, p1, ridge_b, color, style, 1, 0.0, center)
        self._face_append_orient(verts, idx, p0, ridge_b, ridge_a, color, style, 1, 0.0, center)
        p0 = np.array([r2[0], r2[1], z_top]); p1 = np.array([r1[0], r1[1], z_top])
        self._face_append_orient(verts, idx, p0, p1, ridge_a, color, style, 1, 0.0, center)
        self._face_append_orient(verts, idx, p0, ridge_a, ridge_b, color, style, 1, 0.0, center)
        p0 = np.array([l1[0], l1[1], z_top]); p1 = np.array([r1[0], r1[1], z_top])
        self._face_append_orient(verts, idx, p0, p1, ridge_a, color, style, 1, 0.0, center)
        p0 = np.array([l2[0], l2[1], z_top]); p1 = np.array([r2[0], r2[1], z_top])
        self._face_append_orient(verts, idx, p0, p1, ridge_b, color, style, 1, 0.0, center)
        return np.vstack(verts), np.concatenate(idx)

    def _roof_dome(self, poly, z_top, color, style):
        cx, cy = poly.centroid.x, poly.centroid.y
        minx, miny, maxx, maxy = poly.bounds
        rx = max(0.5, (maxx - minx) * 0.45)
        ry = max(0.5, (maxy - miny) * 0.45)
        hz = max(2.0, min(rx, ry) * 1.2)
        n_lon, n_lat = 16, 8
        verts, idx = [], []
        def pt(phi, th):
            return np.array([cx + rx * math.cos(phi) * math.cos(th),
                             cy + ry * math.cos(phi) * math.sin(th),
                             z_top + hz * math.sin(phi)])
        center = np.array([cx, cy, z_top])
        for i_lat in range(n_lat):
            phi0 = (i_lat / n_lat) * math.pi * 0.5
            phi1 = ((i_lat + 1) / n_lat) * math.pi * 0.5
            for i_lon in range(n_lon):
                th0 = (i_lon / n_lon) * 2 * math.pi
                th1 = ((i_lon + 1) / n_lon) * 2 * math.pi
                p00, p10 = pt(phi0, th0), pt(phi0, th1)
                p01, p11 = pt(phi1, th0), pt(phi1, th1)
                if i_lat + 1 == n_lat:
                    self._face_append_orient(verts, idx, p00, p10, p11, color, style, 1, 0.0, center)
                else:
                    self._face_append_orient(verts, idx, p00, p10, p11, color, style, 1, 0.0, center)
                    self._face_append_orient(verts, idx, p00, p11, p01, color, style, 1, 0.0, center)
        return np.vstack(verts), np.concatenate(idx)

    def _roof_skillion(self, poly, z_top, color, style):
        rect = poly.minimum_rotated_rectangle
        coords = np.asarray(rect.exterior.coords)[:-1]
        if len(coords) != 4: return self._roof_pyramidal(poly, z_top, color, style)
        e0 = coords[1] - coords[0]; e1 = coords[2] - coords[1]
        if np.linalg.norm(e0) >= np.linalg.norm(e1):
            low = (coords[0], coords[1]); high = (coords[3], coords[2])
            s1 = (coords[0], coords[3]); s2 = (coords[1], coords[2])
        else:
            low = (coords[1], coords[2]); high = (coords[0], coords[3])
            s1 = (coords[1], coords[0]); s2 = (coords[2], coords[3])
        h = max(1.5, min(5.0, np.linalg.norm(np.array(low[0]) - np.array(low[1])) * 0.12))
        pl_a = np.array([low[0][0], low[0][1], z_top]); pl_b = np.array([low[1][0], low[1][1], z_top])
        ph_a = np.array([high[0][0], high[0][1], z_top + h]); ph_b = np.array([high[1][0], high[1][1], z_top + h])
        center = np.array([poly.centroid.x, poly.centroid.y, z_top])
        verts, idx = [], []
        self._face_append_orient(verts, idx, pl_a, pl_b, ph_b, color, style, 1, 0.0, center)
        self._face_append_orient(verts, idx, pl_a, ph_b, ph_a, color, style, 1, 0.0, center)
        hb_a = np.array([high[0][0], high[0][1], z_top]); hb_b = np.array([high[1][0], high[1][1], z_top])
        self._face_append_orient(verts, idx, hb_a, hb_b, ph_b, color, style, 1, 0.0, center)
        self._face_append_orient(verts, idx, hb_a, ph_b, ph_a, color, style, 1, 0.0, center)
        s1_base_a = np.array([s1[0][0], s1[0][1], z_top]); s1_base_b = np.array([s1[1][0], s1[1][1], z_top])
        s1_top = np.array([s1[1][0], s1[1][1], z_top + h])
        self._face_append_orient(verts, idx, s1_base_a, s1_base_b, s1_top, color, style, 1, 0.0, center)
        s2_base_a = np.array([s2[0][0], s2[0][1], z_top]); s2_base_b = np.array([s2[1][0], s2[1][1], z_top])
        s2_top = np.array([s2[1][0], s2[1][1], z_top + h])
        self._face_append_orient(verts, idx, s2_base_a, s2_base_b, s2_top, color, style, 1, 0.0, center)
        return np.vstack(verts), np.concatenate(idx)

    def _band_from_polygon(self, poly, z0, z1, inset, color, style, part_kind, emissive):
        p = poly.buffer(-inset) if inset > 0.01 else poly
        cand = sanitize_footprint(p, 0.0, 0.5) if not p.is_empty else []
        if not cand: cand = sanitize_footprint(poly, 0.0, 0.5)
        if not cand: return None, None
        p = max(cand, key=lambda g: g.area)
        p = Polygon(p.exterior)
        coords = strip_ring(np.asarray(p.exterior.coords))
        if len(coords) < 3: return None, None
        verts, idx = [], []
        n = len(coords)
        for i in range(n):
            a, b = coords[i], coords[(i + 1) % n]
            p0 = np.array([a[0], a[1], z0]); p1 = np.array([b[0], b[1], z0])
            p2 = np.array([b[0], b[1], z1]); p3 = np.array([a[0], a[1], z1])
            nv = _tri_normal(p0, p1, p3)
            self._face_append(verts, idx, p0, p1, p2, nv, color, style, part_kind, emissive)
            self._face_append(verts, idx, p0, p2, p3, nv, color, style, part_kind, emissive)
        V = np.vstack(verts); Ix = np.concatenate(idx)
        tp, tt = self._triangulate(p)
        if len(tt):
            cv, ct = self._make_cap(tp, tt, z1, True, color, style, part_kind)
            cv[:, 15] = emissive
            Ix = np.concatenate([Ix, ct.ravel() + len(V)]); V = np.vstack([V, cv])
        return V, Ix

    # ================================================================ rooftops
    PARAPET_MIN_AREA = 30.0

    def _parapet(self, V, I, poly, z_top, t, ph, color, style):
        """A thin lip around the roof edge: outer face continues the facade, inner
        face and top strip close it off.  Returns False if the ring can't be offset
        cleanly (the roof is then left bare rather than risk a bad mesh)."""
        p = orient(Polygon(poly.exterior), 1.0)
        try:
            inner = p.buffer(-t, join_style=2, mitre_limit=3.0)
        except Exception:
            return False
        if inner.is_empty or inner.geom_type != "Polygon":
            return False
        oc = strip_ring(np.asarray(p.exterior.coords))
        ic = strip_ring(np.asarray(inner.exterior.coords))
        if len(oc) != len(ic) or not (3 <= len(oc) <= 48):
            return False
        k = int(np.argmin(np.linalg.norm(ic - oc[0], axis=1)))
        ic = np.roll(ic, -k, axis=0)
        if float(np.max(np.linalg.norm(ic - oc, axis=1))) > t * 4.0:
            return False
        z1 = z_top + ph
        up = np.array([0.0, 0.0, 1.0])
        n = len(oc)

        def P(xy, z):
            return np.array([xy[0], xy[1], z], np.float64)
        for i in range(n):
            a, b = oc[i], oc[(i + 1) % n]
            ai, bi = ic[i], ic[(i + 1) % n]
            d = b - a
            ln = float(np.hypot(d[0], d[1]))
            if ln < 1e-6:
                continue
            nout = np.array([d[1], -d[0], 0.0]) / ln
            self._face_append(V, I, P(a, z_top), P(b, z_top), P(b, z1), nout, color, style, 9.0)
            self._face_append(V, I, P(a, z_top), P(b, z1), P(a, z1), nout, color, style, 9.0)
            self._face_append(V, I, P(bi, z_top), P(ai, z_top), P(ai, z1), -nout, color, style, 9.0)
            self._face_append(V, I, P(bi, z_top), P(ai, z1), P(bi, z1), -nout, color, style, 9.0)
            self._face_append(V, I, P(a, z1), P(b, z1), P(bi, z1), up, color, style, 9.0)
            self._face_append(V, I, P(a, z1), P(bi, z1), P(ai, z1), up, color, style, 9.0)
        return True

    def _rooftop(self, poly, z_top, b, seed):
        """Parapet, a few plant rooms / AC units and, on tall towers, one slim mast
        with an aviation beacon.  Everything sits inside the footprint: nothing
        overhangs the facade (the old crown bands and spikes did)."""
        poly = Polygon(poly.exterior)
        if poly.is_empty or poly.area < self.PARAPET_MIN_AREA:
            return None, None
        V, I = [], []
        h = float(b.height)
        mk = (b.style if b.style >= 0 else
              b.material_kind if b.material_kind >= 0 else int(theme_code(b.kind)))
        style = np.array([seed, float(mk), z_top + 6.0], np.float32)
        rng = np.random.default_rng(int(b.id) & 0x7FFFFFFF)
        tone = 0.94 + 0.12 * seed
        concrete = np.clip(np.array([0.62, 0.61, 0.58], np.float32) * tone, 0, 1)

        t = 0.35 if h < 40.0 else 0.45
        ph = 0.6 if h < 15.0 else (1.0 if h < 80.0 else 1.3)
        lip = self._parapet(V, I, poly, z_top, t, ph, concrete, style)
        keep_out = poly.buffer(-(t + 0.7) if lip else -0.7)
        if keep_out.is_empty:
            return (np.vstack(V), np.concatenate(I)) if V else (None, None)
        if keep_out.geom_type != "Polygon":
            keep_out = max(keep_out.geoms, key=lambda g: g.area)

        # axis of the roof, so equipment lines up with the walls
        try:
            rc = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
            ang = math.atan2(rc[1][1] - rc[0][1], rc[1][0] - rc[0][0])
        except Exception:
            ang = 0.0
        ca, sa = math.cos(ang), math.sin(ang)
        minx, miny, maxx, maxy = keep_out.bounds
        placed = []

        def try_box(cx, cy, hx, hy, hz, col, emissive=0.0, part=9.0, pad=0.8):
            corners = [(cx + ca * sx * hx - sa * sy * hy, cy + sa * sx * hx + ca * sy * hy)
                       for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            fp = Polygon(corners)
            if not keep_out.contains(fp):
                return False
            if any(fp.buffer(pad).intersects(q) for q in placed):
                return False
            placed.append(fp)
            bot = [(x, y, z_top) for x, y in corners]
            top = [(x, y, z_top + hz) for x, y in corners]
            self._box(V, I, bot, top, col, style, part, emissive)
            return True

        def rand_point():
            return (float(rng.uniform(minx, maxx)), float(rng.uniform(miny, maxy)))

        span = math.sqrt(poly.area)
        # plant room / lift overrun
        if h >= 18.0:
            for n_room in range(2 if h >= 60.0 else 1):
                for _ in range(14):
                    cx, cy = rand_point()
                    w = float(np.clip(span * rng.uniform(0.08, 0.15), 3.0, 9.0))
                    d = float(np.clip(span * rng.uniform(0.06, 0.12), 2.5, 7.0))
                    hz = 2.8 + float(rng.uniform(0.0, 2.2)) + (1.8 if h >= 100.0 else 0.0)
                    col = np.clip(concrete * rng.uniform(0.92, 1.05), 0, 1)
                    if try_box(cx, cy, w * 0.5, d * 0.5, hz, col):
                        break
        # AC / chiller units
        for _ in range(int(min(7, poly.area / 170.0))):
            for _ in range(8):
                cx, cy = rand_point()
                w = float(rng.uniform(1.4, 2.6)); d = float(rng.uniform(1.0, 1.8))
                hz = float(rng.uniform(0.9, 1.6))
                if try_box(cx, cy, w * 0.5, d * 0.5, hz, np.clip(concrete * 0.84, 0, 1), pad=0.6):
                    break
        # slim mast + beacon on towers
        if h >= 90.0 and b.kind not in ("church", "cathedral"):
            c = keep_out.representative_point()
            cx, cy = float(c.x), float(c.y)
            tall = h >= 150.0
            mast_h = (14.0 + 14.0 * seed) if tall else (5.0 + 5.0 * seed)
            steel = np.array([0.74, 0.75, 0.77], np.float32)
            base_h = mast_h * 0.45 if tall else 0.0
            if tall and not try_box(cx, cy, 0.30, 0.30, base_h, steel, part=3.0, pad=0.0):
                base_h = 0.0
            z_a = z_top + base_h
            half = 0.11 if tall else 0.09
            corners = [(cx + half, cy + half), (cx - half, cy + half), (cx - half, cy - half), (cx + half, cy - half)]
            self._box(V, I, [(x, y, z_a) for x, y in corners],
                      [(x, y, z_top + mast_h) for x, y in corners], steel, style, 3.0)
            bh = 0.34
            cb = [(cx + bh, cy + bh), (cx - bh, cy + bh), (cx - bh, cy - bh), (cx + bh, cy - bh)]
            self._box(V, I, [(x, y, z_top + mast_h) for x, y in cb],
                      [(x, y, z_top + mast_h + 0.7) for x, y in cb], steel, style, 3.0, 1.0)
        if not V:
            return None, None
        return np.vstack(V), np.concatenate(I)

    def _triangulate(self, poly):
        empty = (np.empty((0, 2), np.float64), np.empty((0, 3), np.int64))
        shell = strip_ring(poly.exterior.coords)
        if len(shell) < 3: return empty
        rings = [shell] + [h for h in (strip_ring(r.coords) for r in poly.interiors) if len(h) >= 3]
        pts = np.ascontiguousarray(np.vstack(rings), dtype=np.float64)
        if not np.isfinite(pts).all(): return empty
        ends = np.cumsum([len(x) for x in rings]).astype(np.uint32)
        tris = None
        if earcut is not None:
            try:
                t = np.asarray(earcut.triangulate_float64(pts, ends), dtype=np.int64).reshape(-1, 3)
                t = t[(t >= 0).all(axis=1) & (t < len(pts)).all(axis=1)]
                if len(t): tris = self._filter_tris(poly, pts, t)
            except Exception: tris = None
        if tris is not None and len(tris) and self._coverage_ok(poly, pts, tris): return pts, tris
        fpts, ftris = self._triangulate_fallback(poly)
        if len(ftris):
            ftris = self._filter_tris(poly, fpts, ftris)
            if len(ftris): return fpts, ftris
        if tris is not None and len(tris): return pts, tris
        return empty

    @staticmethod
    def _filter_tris(poly, pts, tris):
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        tris = np.where((cross < 0)[:, None], tris[:, [0, 2, 1]], tris)
        keep = np.abs(cross) > 1e-6
        cen = (a + b + c) / 3.0
        try: keep &= shapely.contains_xy(poly.buffer(0.01), cen[:, 0], cen[:, 1])
        except Exception: pass
        return tris[keep]

    @staticmethod
    def _coverage_ok(poly, pts, tris):
        a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
        area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum()
        return abs(area - poly.area) <= max(0.02 * poly.area, 0.05)

    @staticmethod
    def _refine_uniform(pts, tris, max_edge=45.0, passes=7):
        """Split triangles into four while any edge exceeds max_edge.

        GEOS buffering and strip_ring both drop collinear vertices, so a 600 m
        straight footpath comes back as a four corner rectangle whatever we do
        to the ring beforehand.  Left alone those corners become kilometre long
        slivers that ignore the ground; splitting them afterwards keeps every
        ribbon vertex on the sampled surface.  Shared midpoints avoid cracks.
        """
        pts = [np.asarray(p, np.float64) for p in np.asarray(pts)]
        tris = np.asarray(tris, np.int64)
        for _ in range(passes):
            if len(tris) == 0: break
            p = np.vstack(pts)
            a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
            longest = max(np.linalg.norm(b - a, axis=1).max(),
                          np.linalg.norm(c - b, axis=1).max(),
                          np.linalg.norm(a - c, axis=1).max())
            if longest <= max_edge: break
            cache = {}

            def mid(i, j):
                key = (i, j) if i < j else (j, i)
                m = cache.get(key)
                if m is None:
                    m = len(pts)
                    pts.append((pts[i] + pts[j]) * 0.5)
                    cache[key] = m
                return m

            out = []
            for t in tris:
                i, j, k = int(t[0]), int(t[1]), int(t[2])
                m1, m2, m3 = mid(i, j), mid(j, k), mid(k, i)
                out.append((i, m1, m3)); out.append((m1, j, m2))
                out.append((m3, m2, k)); out.append((m1, m2, m3))
            tris = np.asarray(out, np.int64)
        return np.vstack(pts), tris

    @staticmethod
    def _triangulate_fallback(poly):
        tri_polys = []
        cdt = getattr(shapely, "constrained_delaunay_triangles", None)
        if cdt is not None:
            try: tri_polys = [t for t in cdt(poly).geoms if t.geom_type == "Polygon"]
            except Exception: tri_polys = []
        if not tri_polys:
            from shapely.ops import triangulate
            tri_polys = [t for t in triangulate(poly) if poly.covers(t.representative_point())]
        if not tri_polys:
            return np.empty((0, 2), np.float64), np.empty((0, 3), np.int64)

        pts, idx_of, tris = [], {}, []
        for tri in tri_polys:
            coords = list(tri.exterior.coords)[:3]
            if len(coords) < 3: continue
            tri_idx = []
            for c in coords:
                k = (round(c[0], 6), round(c[1], 6))
                if k not in idx_of:
                    idx_of[k] = len(pts); pts.append(c)
                tri_idx.append(idx_of[k])
            if len(set(tri_idx)) == 3:
                tris.append(tri_idx)
        if not tris:
            return np.empty((0, 2), np.float64), np.empty((0, 3), np.int64)
        return np.array(pts, np.float64), np.array(tris, np.int64)

    # ================================================================ terrain
    @staticmethod
    def _merge_meshes(a, b):
        if a is None: return b
        if b is None: return a
        verts = np.vstack([a.vertices, b.vertices])
        idx = np.concatenate([a.indices, b.indices + np.uint32(len(a.vertices))]).astype(np.uint32)
        if len(a.edges) or len(b.edges):
            edges = np.vstack([a.edges, b.edges]).astype(np.float32)
        else:
            edges = np.empty((0, 6), np.float32)
        radius = float(np.max(np.hypot(verts[:, 0], verts[:, 1])))
        return MeshData(verts, idx, edges, radius, float(verts[:, 2].max()))

    @staticmethod
    def _polyline_frame(line, arc, q):
        """Arc length + signed lateral offset of query points along a polyline."""
        line = np.asarray(line, np.float64)
        q = np.asarray(q, np.float64)
        if len(line) < 2 or len(q) == 0:
            return np.zeros(len(q)), np.zeros(len(q))
        a, b = line[:-1], line[1:]
        ab = b - a
        l2 = np.maximum((ab * ab).sum(1), 1e-12)
        seg = np.sqrt(l2)
        s_out = np.empty(len(q))
        o_out = np.empty(len(q))
        for s0 in range(0, len(q), 1024):
            qq = q[s0:s0 + 1024]
            d = qq[:, None, :] - a[None]
            t = np.clip((d * ab[None]).sum(2) / l2[None], 0.0, 1.0)
            proj = a[None] + ab[None] * t[:, :, None]
            dist2 = ((qq[:, None, :] - proj) ** 2).sum(2)
            k = dist2.argmin(1)
            r = np.arange(len(qq))
            s_out[s0:s0 + 1024] = arc[k] + t[r, k] * seg[k]
            vx = qq[:, 0] - proj[r, k, 0]
            vy = qq[:, 1] - proj[r, k, 1]
            tx = ab[k, 0] / seg[k]
            ty = ab[k, 1] / seg[k]
            o_out[s0:s0 + 1024] = tx * vy - ty * vx
        return s_out, o_out

    def _build_terrain(self, hf, greens, waters, bbox, sea_z, samples=None):
        if bbox is None: return None
        n = max(8, int(samples or TERRAIN_GRID))
        x0, y0, x1, y1 = (float(v) for v in bbox)
        if x1 - x0 < 1.0 or y1 - y0 < 1.0: return None
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        gx, gy = np.meshgrid(xs, ys)
        if hf is not None:
            H = np.asarray(hf.at(gx, gy), np.float64)
        else:
            H = np.zeros(gx.shape, np.float64)
        H = np.clip(H, sea_z - 500.0, 6000.0)
        # Land-use colour is drawn by an exact land-cover overlay (see
        # _build_landcover).  Painting the coarse height grid as well smears
        # every park/beach boundary across one ~23 m cell, which put tan sand
        # wedges through the boardwalk and lawn edges.
        mat = np.full(gx.shape, 5.0, np.float32)
        flat_h = H.reshape(-1)
        flat_m = mat.reshape(-1)
        if waters:
            for p, level, _ in _water_regions(waters, hf, sea_z, bbox):
                inside = _points_in_polys(
                    np.stack([gx.ravel(), gy.ravel()], 1), [p])
                if not inside.any():
                    continue
                # Only notch the shoreline by a shallow amount inland.  The
                # old sea-datum cut made narrow rivers into deep canyons;
                # open water keeps the old depth so oceans still read deep.
                depth = 2.25 if level <= sea_z + 0.01 else 0.90
                flat_h[inside] = np.minimum(flat_h[inside], level - depth)
                flat_m[inside] = 6.0
        hgt = flat_h.reshape(n, n)
        dx = max((x1 - x0) / (n - 1), 1e-3)
        dy = max((y1 - y0) / (n - 1), 1e-3)
        dzdx = np.zeros_like(hgt)
        dzdy = np.zeros_like(hgt)
        dzdx[:, 1:-1] = (hgt[:, 2:] - hgt[:, :-2]) / (2.0 * dx)
        dzdy[1:-1, :] = (hgt[2:, :] - hgt[:-2, :]) / (2.0 * dy)
        inv = 1.0 / np.sqrt(dzdx * dzdx + dzdy * dzdy + 1.0)
        normals = np.stack([-dzdx * inv, -dzdy * inv, inv], axis=-1)

        verts = np.zeros((n * n, BLDG_STRIDE), np.float32)
        verts[:, 0] = gx.ravel()
        verts[:, 1] = gy.ravel()
        verts[:, 2] = flat_h
        verts[:, 3:6] = normals.reshape(-1, 3)
        verts[:, 6] = flat_m
        verts[:, 9] = 1.0
        verts[:, 16] = 5.0
        grid = np.arange(n * n).reshape(n, n)
        a = grid[:-1, :-1].ravel(); b = grid[:-1, 1:].ravel()
        c = grid[1:, 1:].ravel(); d = grid[1:, :-1].ravel()
        idx = np.stack([a, b, c, a, c, d], axis=1).reshape(-1).astype(np.uint32)
        radius = float(np.max(np.hypot(verts[:, 0], verts[:, 1])))
        return MeshData(verts, idx, np.empty((0, 6), np.float32), radius, float(flat_h.max()))

    # ================================================================ land cover
    def _build_landcover(self, greens, waters=(), hf=None, bbox=None):
        """Exact, draped grass/sand/concrete cover, clipped by land-use priority.

        The terrain colour grid only has ~23 m between samples, so a park or a
        beach drawn into its vertices steps across whole cells.  A beach next
        to the Mid Beach Boardwalk therefore painted tan wedges through the
        pavement.  This overlay keeps the source polygon edges exact while
        staying below every road and sidewalk ribbon.
        """
        if bbox is None: return None
        try:
            clip = shapely.box(*map(float, bbox))
            water = unary_union([w.geometry for w in waters]) if waters else None
        except Exception:
            return None
        priority = {1.0: 3.0, 4.0: 2.0, 2.0: 2.0, 3.0: 1.0}
        entries = []
        for order, green in enumerate(greens or ()):
            material = float(green.material)
            rank = priority.get(material, 0.0)
            if rank <= 0.: continue
            for poly in polygons_of(green.geometry):
                if poly.is_empty or poly.area < 1.0: continue
                entries.append((rank, order, material, poly))
        if not entries: return None
        # Higher rank wins; among equal ranks the later OSM element wins, as
        # the old vertex painter did.  Paint the winners first, then cut them
        # out of everything below so boundaries never double-draw.
        entries.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        painted, V, I, nv = None, [], [], 0
        draper = TerrainDraper(hf)
        for _, _, material, poly in entries:
            try:
                piece = poly.intersection(clip)
                if water is not None: piece = piece.difference(water)
                if painted is not None: piece = piece.difference(painted)
                if piece.is_empty: continue
                painted = piece if painted is None else unary_union([painted, piece])
                parts = list(polygons_of(piece))
            except Exception:
                continue
            for part in parts:
                if part.is_empty or part.area < .05: continue
                try:
                    part = orient(part, sign=1.)
                    ext = _densify(np.asarray(part.exterior.coords, np.float64), 8.0)
                    rings = [ext] + [_densify(np.asarray(h.coords, np.float64), 8.0)
                                     for h in part.interiors]
                    pts, tris = self._triangulate(Polygon(rings[0], rings[1:]))
                    if len(tris) == 0: continue
                    if hf is not None:
                        pts, tris = draper.conform(pts, tris, max_edge=12.)
                    else:
                        pts, tris = self._refine_uniform(pts, tris, 45.)
                except Exception:
                    continue
                if len(tris) == 0: continue
                z = ((np.asarray(hf.at(pts[:, 0], pts[:, 1]), np.float64)
                      if hf is not None else np.zeros(len(pts), np.float64))
                     + LANDCOVER_LIFT)
                xyz = np.c_[pts, z]
                block = np.zeros((len(pts), BLDG_STRIDE), np.float32)
                block[:, :3] = xyz
                block[:, 3:6] = vertex_normals(xyz, tris)
                block[:, 6] = material
                block[:, 9] = 1.0
                block[:, 16] = 5.0
                V.append(block)
                I.append(tris.ravel().astype(np.uint32) + np.uint32(nv))
                nv += len(pts)
        if not V: return None
        vertices = np.ascontiguousarray(np.vstack(V), np.float32)
        indices = np.ascontiguousarray(np.concatenate(I), np.uint32)
        return MeshData(vertices, indices, np.empty((0, 6), np.float32),
                        float(np.max(np.hypot(vertices[:, 0], vertices[:, 1]))),
                        float(vertices[:, 2].max()))

    # ================================================================ ground
    def _build_ground(self, roads, hf=None, bbox=None, sea_z=0.0, deck_pins=None):
        V, I = [], []
        nv = 0          # running vertex count (V holds blocks, not vertices)
        stride = BLDG_STRIDE
        draper = TerrainDraper(hf)

        # Pedestrian surfaces are cut out of every carriageway below, so they
        # only need a shallow kerb: a tall lift made beach promenades float.
        carriageways = []
        for r in roads:
            if r.highway in PEDESTRIAN_HIGHWAYS: continue
            if r.tunnel or r.bridge or r.layer > 0: continue
            if r.level is not None and r.level < 0.: continue
            w = road_width(r)
            if w < 1.0: continue
            try:
                carriageways.append(r.geometry.buffer(w * 0.5, cap_style=2, join_style=2))
            except Exception:
                continue
        carriageways = unary_union(carriageways) if carriageways else None

        # roads & footpaths, draped over the terrain height field
        for ri, r in enumerate(roads):
            if r.tunnel or (r.level is not None and r.level < 0.): continue
            width = road_width(r)
            if width < 1.0: continue
            try:
                geom = r.geometry
                buf = geom.buffer(width * 0.5, cap_style=2, join_style=2)
            except Exception:
                continue
            pts_c = np.asarray(geom.coords, np.float64)
            if len(pts_c) < 2: continue
            keep = np.linalg.norm(np.diff(pts_c, axis=0), axis=1) > 0.05
            pts_c = pts_c[np.concatenate(([True], keep))]
            if len(pts_c) < 2: continue
            pins = deck_pins.get(id(r)) if deck_pins else None
            arc, zs = road_profile(pts_c, r, hf, sea_z, bbox, deck_pins=pins)
            total = float(arc[-1])
            if total < 0.5: continue
            is_footpath = r.highway in PEDESTRIAN_HIGHWAYS
            if is_footpath and carriageways is not None:
                # Pedestrians step off the kerb onto paint, not onto a slab that
                # spans the street; the zebra is the only surface crossing it.
                buf = buf.difference(carriageways)
            # Overlapping ribbons (junctions) get a deterministic order: bigger roads sit on top,
            # with a per-road tie-break.  The old 0-4 mm jitter was below depth resolution
            # beyond a few metres, so every junction shimmered.
            lift = road_surface_lift(r.highway, ri, footpath=is_footpath)

            for p in polygons_of(buf):
                if p.is_empty or p.area < 2.0: continue
                try:
                    # GEOS drops collinear vertices when buffering, so a long
                    # straight road comes back as one rectangle.  Resample the
                    # outline before triangulating, otherwise the ribbon is a
                    # plate-kilometre sliver that ignores the terrain.
                    ext = _densify(np.asarray(p.exterior.coords, np.float64), 8.0)
                    rings = [ext] + [_densify(np.asarray(h.coords, np.float64), 8.0)
                                     for h in p.interiors]
                    p = Polygon(rings[0], rings[1:])
                    pts, tris = self._triangulate(p)
                except Exception:
                    continue
                if len(tris) == 0: continue
                if hf is not None and not (r.bridge or r.layer > 0):
                    pts, tris = draper.conform(pts, tris)
                else:
                    pts, tris = self._refine_uniform(pts, tris, 45.0)
                if len(tris) == 0: continue
                q = np.asarray(pts, np.float64)
                s_along, off = self._polyline_frame(pts_c, arc, q)
                z_top = np.interp(s_along, arc, zs) + lift
                if hf is not None and not (r.bridge or r.layer > 0):
                    # Follow the ground under every ribbon vertex, not just the
                    # centreline: on a side slope a flat 20 m ribbon would
                    # otherwise sink under its uphill edge.
                    z_top = np.asarray(hf.at(q[:, 0], q[:, 1]), np.float64) + ROAD_SURFACE + lift
                u = np.clip(0.5 + off / width, 0.0, 1.0)
                base = nv
                block = np.zeros((len(q), stride), np.float32)
                block[:, 0] = q[:, 0]
                block[:, 1] = q[:, 1]
                block[:, 2] = z_top
                block[:, 3:6] = vertex_normals(np.c_[q, z_top], tris)
                block[:, 6] = 0.1 if is_footpath else 0.0
                block[:, 7] = width
                block[:, 9] = 1.0
                block[:, 10] = np.minimum(s_along, total - s_along)
                if not is_footpath:
                    layout = RoadLayout.from_road(r)
                    block[:, 11:13] = layout.paint_counts
                    # Paint class (blue channel): 0 unmarked, .5 city street, 1 highway.
                    # Stubs, alleys and service lanes carry no lane paint in reality.
                    if (r.highway in ('service', 'living_street', 'track', 'pedestrian')
                            or width < 5.2 or total < 18.0):
                        block[:, 8] = 0.0
                    elif r.highway in ('motorway', 'trunk', 'motorway_link', 'trunk_link'):
                        block[:, 8] = 1.0
                    else:
                        block[:, 8] = 0.5
                block[:, 13] = u
                block[:, 14] = s_along
                block[:, 16] = 5.0
                V.append(block)
                nv += len(q)
                I.append((tris.astype(np.int64) + base).ravel())

                if is_footpath:
                    coords = _densify(np.asarray(p.exterior.coords, np.float64), 8.0)
                    if not (r.bridge or r.layer > 0):
                        coords = terrain_path(coords, hf)
                    if len(coords) > 3:
                        cs, _ = self._polyline_frame(pts_c, arc, coords[:, :2])
                        cz = np.interp(cs, arc, zs) + lift
                        if hf is not None and not (r.bridge or r.layer > 0):
                            cz = np.asarray(hf.at(coords[:, 0], coords[:, 1]),
                                            np.float64) + ROAD_SURFACE + lift
                        drop = ROAD_SURFACE + lift
                        cbase = nv
                        for j in range(len(coords) - 1):
                            a, b = coords[j], coords[j + 1]
                            za, zb = cz[j], cz[j + 1]
                            dx = b[0] - a[0]; dy = b[1] - a[1]
                            nx, ny = dy, -dx
                            ln = math.hypot(nx, ny)
                            if ln > 1e-6: nx, ny = nx / ln, ny / ln
                            curb = np.zeros((4, stride), np.float32)
                            curb[0, 0:2] = a; curb[1, 0:2] = b
                            curb[2, 0:2] = b; curb[3, 0:2] = a
                            curb[0, 2] = za; curb[1, 2] = zb
                            # Reach the terrain, not just partway: a shallow face
                            # that stops above the ground reads as a floating lip.
                            curb[2, 2] = zb - drop; curb[3, 2] = za - drop
                            curb[:, 3] = nx; curb[:, 4] = ny
                            curb[:, 6] = 0.1; curb[:, 7] = width
                            curb[:, 9] = 1.0
                            curb[:, 13] = 0.5
                            curb[:, 16] = 5.0
                            V.append(curb)
                            nv += 4
                            o = cbase + j * 4
                            I.append(np.array([o, o + 1, o + 2, o, o + 2, o + 3], np.int64))

        if not V:
            return None
        vertices = np.ascontiguousarray(np.vstack(V), np.float32)
        indices = np.ascontiguousarray(np.concatenate(I).astype(np.uint32))
        radius = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
        return MeshData(vertices, indices,
                        np.empty((0, 6), np.float32), radius, float(vertices[:, 2].max()))

    # ================================================================ water
    def _build_water(self, waters, sea_z=0.0, hf=None, bbox=None):
        from .water_mesh import triangulate_water, WATER_BOUNDS_PAD
        V, I = [], []
        stride = BLDG_STRIDE
        nv = 0
        for p, z_water, scale in _water_regions(waters, hf, sea_z, bbox):
            try:
                pts, tris, spacing = triangulate_water(p, bbox, self._triangulate)
            except Exception:
                continue
            if len(tris) == 0: continue
            block = np.zeros((len(pts), stride), np.float32)
            block[:, :2] = pts
            block[:, 2] = z_water  # one datum: do not lower water by a second .35 m
            block[:, 5] = block[:, 9] = 1.
            block[:, 6:9] = (.05, .18, .25)
            block[:, 10] = scale
            block[:, 11] = spacing
            block[:, 16] = 5.
            V.append(block)
            I.append(tris.ravel() + np.uint32(nv))
            nv += len(pts)
        if not V:
            return None
        vertices = np.ascontiguousarray(np.vstack(V), np.float32)
        indices = np.ascontiguousarray(np.concatenate(I), np.uint32)
        radius = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
        return MeshData(vertices, indices, np.empty((0, 6), np.float32), radius,
                        float(vertices[:, 2].max()) + WATER_BOUNDS_PAD)

    # ================================================================ bridges
    def _plate(self, V, I, quad, color, style, part=1.0):
        """A quad emitted from both sides (thin bridge plates)."""
        p0, p1, p2, p3 = quad
        n = _tri_normal(p0, p1, p2)
        self._face_append(V, I, p0, p1, p2, n, color, style, part)
        self._face_append(V, I, p0, p2, p3, n, color, style, part)
        self._face_append(V, I, p0, p2, p1, -n, color, style, part)
        self._face_append(V, I, p0, p3, p2, -n, color, style, part)

    def _box(self, V, I, bottom, top, color, style, part=1.0, emissive=0.0):
        bottom = [np.asarray(p, np.float64) for p in bottom]
        top = [np.asarray(p, np.float64) for p in top]
        center = (sum(bottom) + sum(top)) / 8.0
        for i in range(4):
            a, b = bottom[i], bottom[(i + 1) % 4]
            c, d = top[(i + 1) % 4], top[i]
            self._face_append_orient(V, I, a, b, c, color, style, part, emissive=emissive, center=center)
            self._face_append_orient(V, I, a, c, d, color, style, part, emissive=emissive, center=center)
        self._face_append_orient(V, I, top[0], top[1], top[2], color, style, part, emissive=emissive, center=center)
        self._face_append_orient(V, I, top[0], top[2], top[3], color, style, part, emissive=emissive, center=center)
        self._face_append_orient(V, I, bottom[0], bottom[2], bottom[1], color, style, part, emissive=emissive, center=center)
        self._face_append_orient(V, I, bottom[0], bottom[3], bottom[2], color, style, part, emissive=emissive, center=center)

    def _box_between(self, V, I, center, normal, tangent, hw, hd,
                     z0, z1, color, style, part=1.0):
        cx, cy = float(center[0]), float(center[1])
        bottom, top = [], []
        for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            x = cx + normal[0] * a * hw + tangent[0] * b * hd
            y = cy + normal[1] * a * hw + tangent[1] * b * hd
            bottom.append((x, y, z0)); top.append((x, y, z1))
        self._box(V, I, bottom, top, color, style, part)

    def _build_bridges(self, roads, hf=None, bbox=None, sea_z=0.0, deck_pins=None,
                       sea_piers=None):
        V, I = [], []
        towers_seen = set()
        piers = list(sea_piers or ())
        for r in roads:
            if r.tunnel or not (r.bridge or r.layer > 0): continue
            if r.highway not in ROAD_WIDTHS: continue
            width = road_width(r)
            if width < 1.5: continue
            pts = np.asarray(r.geometry.coords, np.float64)
            if len(pts) < 2: continue
            keep = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.1
            pts = pts[np.concatenate(([True], keep))]
            if len(pts) < 2: continue
            pins = deck_pins.get(id(r)) if deck_pins else None
            arc, zs = road_profile(pts, r, hf, sea_z, bbox, deck_pins=pins)
            total = float(arc[-1])
            if total < 8.0: continue

            gx = np.gradient(pts[:, 0]); gy = np.gradient(pts[:, 1])
            ln = np.maximum(np.hypot(gx, gy), 1e-6)
            nvec = np.stack([gy / ln, -gx / ln], axis=1)
            tvec = np.stack([gx / ln, gy / ln], axis=1)
            deck = zs
            half = width * 0.5
            # every bridge gets a real deck and a railing so small crossings read as
            # proper bridges instead of thin floating slabs on flat ground
            thick = 1.7 if width >= 5.0 else 0.8
            rail_h = 1.15 if width >= 3.0 else 0.9
            steel = _bridge_paint(r)
            concrete = np.array([0.50, 0.49, 0.46], np.float32)
            structure = (r.structure or "").lower()
            name = (r.name or "").lower()
            over_water = False
            if hf is not None:
                try:
                    under = hf.at(pts[:, 0], pts[:, 1])
                    over_water = float((under < sea_z + 0.5).mean()) > 0.35
                except Exception:
                    over_water = False
            suspension = ("suspension" in structure or "cable" in structure
                          or any(k in name for k in _SUSPENSION_NAMES)
                          or (over_water and total > 700.0))
            style = np.array([0.37, 4.0, 0.0], np.float32)
            try:
                corridor = shapely.LineString(pts).buffer(width * .5 + .6,
                                                         cap_style=2, join_style=2)
            except Exception:
                corridor = None

            def on_pier(x, y, margin=0.6):
                """The sea building standing at this point, if any."""
                pt = Point(float(x), float(y))
                for entry in piers:
                    try:
                        if entry[1].distance(pt) <= margin:
                            return entry
                    except Exception:
                        continue
                return None

            ex_l = pts + nvec * half
            ex_r = pts - nvec * half
            for i in range(len(pts) - 1):
                # side girders (deck fascia) and the underside of the deck
                for ex in (ex_l, ex_r):
                    a0 = np.array([ex[i, 0], ex[i, 1], deck[i] - thick])
                    a1 = np.array([ex[i + 1, 0], ex[i + 1, 1], deck[i + 1] - thick])
                    b0 = np.array([ex[i, 0], ex[i, 1], deck[i]])
                    b1 = np.array([ex[i + 1, 0], ex[i + 1, 1], deck[i + 1]])
                    self._plate(V, I, (a0, a1, b1, b0), steel, style, 4.0)
                    # parapet / railing
                    if rail_h <= 0.0: continue
                    r0 = np.array([ex[i, 0], ex[i, 1], deck[i]])
                    r1 = np.array([ex[i + 1, 0], ex[i + 1, 1], deck[i + 1]])
                    r2 = np.array([ex[i + 1, 0], ex[i + 1, 1], deck[i + 1] + rail_h])
                    r3 = np.array([ex[i, 0], ex[i, 1], deck[i] + rail_h])
                    self._plate(V, I, (r0, r1, r2, r3), steel, style, 4.0)
                l0 = np.array([ex_l[i, 0], ex_l[i, 1], deck[i] - thick])
                l1 = np.array([ex_l[i + 1, 0], ex_l[i + 1, 1], deck[i + 1] - thick])
                q0 = np.array([ex_r[i, 0], ex_r[i, 1], deck[i] - thick])
                q1 = np.array([ex_r[i + 1, 0], ex_r[i + 1, 1], deck[i + 1] - thick])
                self._plate(V, I, (l0, q0, q1, l1), concrete, style, 1.0)

            # piers down to the ground / sea bed
            step = max(15.0, width * 3.4)
            pier_placed = False
            for s in np.arange(step * 0.7, total - step * 0.35, step):
                x = float(np.interp(s, arc, pts[:, 0]))
                y = float(np.interp(s, arc, pts[:, 1]))
                z_deck = float(np.interp(s, arc, deck))
                nx = float(np.interp(s, arc, nvec[:, 0]))
                ny = float(np.interp(s, arc, nvec[:, 1]))
                tx = float(np.interp(s, arc, tvec[:, 0]))
                ty = float(np.interp(s, arc, tvec[:, 1]))
                ground = hf.at(np.array([x]), np.array([y]))[0] if hf is not None else sea_z
                ground = max(float(ground), float(sea_z)) - 0.5
                if on_pier(x, y) is not None:
                    continue      # the sea building here is the support
                if z_deck - ground < 2.2: continue
                pier_placed = True
                if width >= 9.0:
                    for side in (1.0, -1.0):
                        cx = x + nx * half * 0.62 * side
                        cy = y + ny * half * 0.62 * side
                        self._box_between(V, I, (cx, cy), (nx, ny), (tx, ty),
                                          1.45, 1.9, ground, z_deck - 0.9,
                                          concrete, style, 1.0)
                    self._box_between(V, I, (x, y), (nx, ny), (tx, ty),
                                      half * 0.72, 1.1, z_deck - 1.7, z_deck - 0.9,
                                      concrete, style, 1.0)
                else:
                    self._box_between(V, I, (x, y), (nx, ny), (tx, ty),
                                      half * 0.55, 1.2, ground, z_deck - (thick - 0.1),
                                      concrete, style, 1.0)

            # A short span with no intermediate pier still needs a central support so
            # the deck does not read as an unsupported floating slab on flat ground.
            if not pier_placed and total > 8.0:
                sm = float(np.interp(total * 0.5, arc, pts[:, 0]))
                tm = float(np.interp(total * 0.5, arc, pts[:, 1]))
                nmx = float(np.interp(total * 0.5, arc, nvec[:, 0]))
                nmy = float(np.interp(total * 0.5, arc, nvec[:, 1]))
                tmx = float(np.interp(total * 0.5, arc, tvec[:, 0]))
                tmy = float(np.interp(total * 0.5, arc, tvec[:, 1]))
                zdeck_m = float(np.interp(total * 0.5, arc, deck))
                gnd = hf.at(np.array([sm]), np.array([tm]))[0] if hf is not None else sea_z
                gnd = max(float(gnd), float(sea_z)) - 0.5
                if on_pier(sm, tm) is None and zdeck_m - gnd >= 2.2:
                    self._box_between(V, I, (sm, tm), (nmx, nmy), (tmx, tmy),
                                      half * 0.5, 1.1, gnd, zdeck_m - (thick - 0.1),
                                      concrete, style, 1.0)

            # A sea building under the deck carries it: a shaft rises from its
            # roof to the soffit instead of a pier being driven through it.
            # Buildings the deck itself runs through were removed beforehand
            # (see ``drop_deck_buildings``), so they are not in this list.
            if piers and corridor is not None:
                for entry in piers:
                    foot, roof = entry[1], entry[3]
                    try:
                        if not foot.intersects(corridor):
                            continue
                        crossing = foot.intersection(corridor)
                        if crossing.is_empty:
                            continue
                        sample = crossing.representative_point()
                        s = float(LineString(pts).project(Point(sample.x, sample.y)))
                    except Exception:
                        continue
                    z_deck = float(np.interp(s, arc, deck))
                    if z_deck - max(thick - .1, .4) - roof < 0.6:
                        continue
                    nx = float(np.interp(s, arc, nvec[:, 0]))
                    ny = float(np.interp(s, arc, nvec[:, 1]))
                    tx = float(np.interp(s, arc, tvec[:, 0]))
                    ty = float(np.interp(s, arc, tvec[:, 1]))
                    try:
                        ring = np.asarray(foot.minimum_rotated_rectangle.exterior.coords,
                                          np.float64)[:-1]
                        hw = float(np.ptp(ring @ np.array([nx, ny]))) * .5
                        hd = float(np.ptp(ring @ np.array([tx, ty]))) * .5
                    except Exception:
                        hw = hd = 2.0
                    hw = max(1.0, min(hw * .9, half * .85))
                    hd = max(1.0, min(hd * .9, 5.0))
                    self._box_between(V, I, (float(sample.x), float(sample.y)),
                                      (nx, ny), (tx, ty), hw, hd, roof,
                                      z_deck - max(thick - .1, .4), concrete, style, 1.0)

            if not suspension:
                continue

            # ---- towers + main cables -------------------------------------
            tower_h = 108.0 if "suspension" in structure else 72.0
            # A main span has a few pylons, not one every 380 m: two over the
            # main span, one for a shorter crossing.  The rest of a long
            # crossing is viaduct carried on piers.
            if total >= 500.0:
                tower_arcs = [total * .35, total * .65]
            elif total >= 250.0:
                tower_arcs = [total * .5]
            else:
                tower_arcs = []
            placed = []
            for s in tower_arcs:
                x = float(np.interp(s, arc, pts[:, 0]))
                y = float(np.interp(s, arc, pts[:, 1]))
                z_deck = float(np.interp(s, arc, deck))
                if on_pier(x, y, 3.0) is not None:
                    continue      # a mapped building already stands as the pylon here
                nx = float(np.interp(s, arc, nvec[:, 0]))
                ny = float(np.interp(s, arc, nvec[:, 1]))
                tx = float(np.interp(s, arc, tvec[:, 0]))
                ty = float(np.interp(s, arc, tvec[:, 1]))
                key = (round(x / 45.0), round(y / 45.0))
                if key in towers_seen: continue
                towers_seen.add(key)
                ground = hf.at(np.array([x]), np.array([y]))[0] if hf is not None else sea_z
                ground = max(float(ground), float(sea_z)) - 0.5
                top = z_deck + tower_h
                for side in (1.0, -1.0):
                    cx = x + nx * (half + 2.4) * side
                    cy = y + ny * (half + 2.4) * side
                    self._box_between(V, I, (cx, cy), (nx, ny), (tx, ty),
                                      2.0, 2.6, ground, top, steel, style, 4.0)
                for frac in (0.42, 0.72, 0.965):
                    z = z_deck + (top - z_deck) * frac
                    self._box_between(V, I, (x, y), (nx, ny), (tx, ty),
                                      half + 2.4, 1.0, z - 0.55, z + 0.55, steel, style, 4.0)
                placed.append((s, top, (nx, ny)))
            if not placed:
                continue

            # main cable: anchor -> towers -> anchor, sagging toward the deck
            anchors = [(0.0, float(zs[0]) + 1.4)]
            anchors += [(s, top) for s, top, _ in placed]
            anchors += [(total, float(zs[-1]) + 1.4)]
            nrm = placed[0][2]
            for k in range(len(anchors) - 1):
                s0, t0 = anchors[k]
                s1, t1 = anchors[k + 1]
                span = s1 - s0
                if span < 6.0: continue
                d_mid = float(np.interp((s0 + s1) * 0.5, arc, deck))
                sag = max(0.0, min(t0, t1) - d_mid - 4.0) * 0.92
                segs = max(6, int(span / 24.0))
                prev = None
                for j in range(segs + 1):
                    f = j / segs
                    s = s0 + span * f
                    x = float(np.interp(s, arc, pts[:, 0]))
                    y = float(np.interp(s, arc, pts[:, 1]))
                    z = t0 + (t1 - t0) * f - sag * math.sin(math.pi * f)
                    p = np.array([x, y, z])
                    if prev is not None:
                        off = np.array([nrm[0] * 0.30, nrm[1] * 0.30, 0.0])
                        self._plate(V, I, (prev - off, p - off, p + off, prev + off),
                                    steel, style, 4.0)
                        voff = np.array([0.0, 0.0, 0.34])
                        self._plate(V, I, (prev - voff, p - voff, p + voff, prev + voff),
                                    steel, style, 4.0)
                    prev = p
                # vertical hangers
                for s in np.arange(s0 + 18.0, s1 - 8.0, 16.0):
                    f = (s - s0) / span
                    x = float(np.interp(s, arc, pts[:, 0]))
                    y = float(np.interp(s, arc, pts[:, 1]))
                    z_cable = t0 + (t1 - t0) * f - sag * math.sin(math.pi * f)
                    z_deck = float(np.interp(s, arc, deck)) + 1.0
                    if z_cable - z_deck < 2.0: continue
                    off = np.array([nrm[0] * 0.11, nrm[1] * 0.11, 0.0])
                    a = np.array([x, y, z_deck]) - off
                    b = np.array([x, y, z_deck]) + off
                    c = np.array([x, y, z_cable]) + off
                    d = np.array([x, y, z_cable]) - off
                    self._plate(V, I, (a, b, c, d), steel, style, 4.0)

        if not V:
            return None
        vertices = np.ascontiguousarray(np.vstack(V), np.float32)
        indices = np.ascontiguousarray(np.concatenate(I), np.uint32)
        radius = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1])))
        return MeshData(vertices, indices, np.empty((0, 6), np.float32),
                        radius, float(vertices[:, 2].max()))


    # ================================================================ streetlights
    def _build_streetlights(self, roads, hf=None, bbox=None, sea_z=0.0):
        try:
            paths = road_paths_from_roads([r for r in roads if not r.tunnel and not r.covered
                                          and not (r.level is not None and r.level < 0.)],
                                         min_length=12.0, hf=hf, sea_z=sea_z, bbox=bbox)
        except Exception:
            paths = []
        return build_streetlights_from_paths(paths)


# ================================================================ mesh cache
class MeshCache:
    """Versioned on-disk cache for tile mesh data.

    Mesh arrays live in a compressed ``.npz``; paths live in a companion
    pickle (they contain dataclasses with numpy arrays that npz can't hold).
    """
    def __init__(self, cache_dir, enabled=True, tag=""):
        self.dir = Path(cache_dir) / "mesh"
        self.enabled = bool(enabled)
        # Meshes are built relative to the terrain datum, so tiles cached for
        # one start elevation must not be reused at another.
        self.tag = str(tag or "")
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                print(f"[mesh cache] cannot create {self.dir}: {exc}")
                self.enabled = False
        self.version = MESH_CACHE_VERSION

    def _path(self, z, key):
        tag = f"_{self.tag}" if self.tag else ""
        return self.dir / f"v{self.version}{tag}_{z}_{key[0]}_{key[1]}.npz"

    def load(self, z, key):
        if not self.enabled: return None
        path = self._path(z, key)
        if not path.exists(): return None
        try:
            data = np.load(path, allow_pickle=False)
            def unpack(prefix):
                if f"{prefix}_v" not in data: return None
                e = data[f"{prefix}_e"] if f"{prefix}_e" in data else np.empty((0, 6), np.float32)
                r = float(data[f"{prefix}_r"]) if f"{prefix}_r" in data else 0.0
                mh = float(data[f"{prefix}_h"]) if f"{prefix}_h" in data else 0.0
                cl = data[f"{prefix}_c"] if f"{prefix}_c" in data else None
                return MeshData(data[f"{prefix}_v"], data[f"{prefix}_i"], e, r, mh, cl)
            b_mesh, g_mesh, w_mesh = unpack("b"), unpack("g"), unpack("w")
            s_mesh, p_mesh = unpack("s"), unpack("p")
            trees = data["trees"] if "trees" in data else None
            lights = data["lights"] if "lights" in data else None
            parked = {k: data[f"parked_{k}"] for k in range(VEHICLE_KINDS) if f"parked_{k}" in data}
            paths = []
            places = None
            if "place_xy" in data:
                places = tuple(Place(str(key_), str(name), str(category), float(xy[0]), float(xy[1]))
                               for key_, name, category, xy in zip(data["place_ids"], data["place_names"],
                                                                  data["place_categories"], data["place_xy"]))
            pkl = path.with_suffix(".pkl")
            if pkl.exists():
                try:
                    with open(pkl, "rb") as f:
                        paths = [RoadPath(*entry) for entry in pickle.load(f)]
                except Exception:
                    paths = []
            # Older lamp arrays omitted short segments and the pole's ground height.
            # Refresh just the lamp layout from cached road profiles; meshes stay cached.
            if paths and (lights is None or lights.shape[1] < 5):
                lights = build_streetlights_from_paths(paths)
            from ..render.lod import prepare_mesh, align_neon
            align_neon(b_mesh, s_mesh)
            for mesh, kind in ((b_mesh, "building"), (g_mesh, "terrain"), (w_mesh, "water"),
                               (s_mesh, "neon"), (p_mesh, "props")):
                prepare_mesh(mesh, kind)
            data.close()
            return ChunkResult(key=key, b_mesh=b_mesh, g_mesh=g_mesh, w_mesh=w_mesh,
                               trees=trees, lights=lights, parked=parked, paths=paths,
                               s_mesh=s_mesh, p_mesh=p_mesh, places=places)
        except Exception as exc:
            print(f"[mesh cache] load {z}/{key[0]}/{key[1]} failed: {exc}")
            return None
        finally:
            if 'data' in locals(): data.close()

    def store(self, z, key, b_mesh, g_mesh, w_mesh, trees,
              lights=None, parked=None, paths=None,
              s_mesh=None, p_mesh=None, places=None):
        if not self.enabled: return
        path = self._path(z, key)
        tmp = path.parent / f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            arrays = {}
            def pack(m, prefix):
                if m is None: return
                arrays[f"{prefix}_v"] = m.vertices
                arrays[f"{prefix}_i"] = m.indices
                if m.edges is not None and len(m.edges) > 0:
                    arrays[f"{prefix}_e"] = m.edges
                arrays[f"{prefix}_r"] = np.float32(m.radius)
                arrays[f"{prefix}_h"] = np.float32(m.max_height)
                if getattr(m, "cells", None) is not None:
                    arrays[f"{prefix}_c"] = m.cells
            pack(b_mesh, "b"); pack(g_mesh, "g"); pack(w_mesh, "w")
            pack(s_mesh, "s"); pack(p_mesh, "p")
            if places is not None:
                arrays["place_xy"] = np.asarray([p.position for p in places], np.float64).reshape(-1, 2)
                arrays["place_ids"] = np.asarray([p.key for p in places], dtype=str)
                arrays["place_names"] = np.asarray([p.name for p in places], dtype=str)
                arrays["place_categories"] = np.asarray([p.category for p in places], dtype=str)
            if trees is not None and len(trees) > 0:   arrays["trees"]  = trees
            if lights is not None and len(lights) > 0: arrays["lights"] = lights
            if parked:
                for k, v in parked.items():
                    if v is not None and len(v) > 0:
                        arrays[f"parked_{k}"] = v
            if not arrays: return
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, path)
            if paths:
                pkl = path.with_suffix(".pkl")
                tmp_pkl = path.parent / f"{path.stem}.{os.getpid()}.{threading.get_ident()}.pkl.tmp"
                with open(tmp_pkl, "wb") as f:
                    pickle.dump([(p.points, p.arc, p.total, p.highway,
                                  p.width, p.oneway, p.lanes, p.z,
                                  *[getattr(p, item.name) for item in fields(RoadPath)[8:]])
                                 for p in paths], f)
                os.replace(tmp_pkl, pkl)
        except Exception as exc:
            print(f"[mesh cache] store {z}/{key[0]}/{key[1]} failed: {exc}")
            try:
                if tmp.exists(): tmp.unlink()
            except OSError: pass


# ================================================================ worker pool
class _MeshExecutor:
    """Wraps an executor so submit(payload) runs mesh_tile_job(payload).

    ChunkStreamer._run_job calls pool.submit(payload) with a single tuple.
    Standard executors want submit(fn, *args); this shim keeps the existing
    app.py working unchanged.
    """
    def __init__(self, executor): self._executor = executor
    def submit(self, payload):   return self._executor.submit(mesh_tile_job, payload)
    def submit_task(self, fn, *args): return self._executor.submit(fn, *args)
    def shutdown(self, wait=True, cancel_futures=False):
        try:    return self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
        except TypeError: return self._executor.shutdown(wait=wait)


_mesh_pool = None
_mesh_pool_lock = threading.Lock()


def shared_mesh_pool(workers=3, process_pool=True):
    global _mesh_pool
    with _mesh_pool_lock:
        if _mesh_pool is None:
            n = max(1, int(workers))
            ex = ProcessPoolExecutor(max_workers=n) if process_pool else ThreadPoolExecutor(max_workers=n)
            _mesh_pool = _MeshExecutor(ex)
        return _mesh_pool


def shutdown_mesh_pool():
    global _mesh_pool
    with _mesh_pool_lock:
        if _mesh_pool is not None:
            try:    _mesh_pool.shutdown(wait=False, cancel_futures=True)
            except TypeError: _mesh_pool.shutdown(wait=False)
            _mesh_pool = None


# ---------------------------------------------------------------- spatial index
CELL_GRID = 4          # grid x grid cells per mesh
CELL_MIN_TRIS = 4000   # smaller meshes are not worth splitting


def build_cells(vertices, indices, grid=CELL_GRID, min_tris=CELL_MIN_TRIS):
    """Sort triangles into a grid x grid spatial index.

    The vertex buffer is untouched; only the index order changes, so a chunk is
    still ONE VBO + ONE EBO and still ONE draw call when it is fully visible.
    When it is only partly visible the renderer draws just the index ranges of
    the cells that survive frustum culling.

    Returns (indices, cells).  ``cells`` is None for small meshes, else rows of
    [first_index, index_count, minx, miny, minz, maxx, maxy, maxz].  The sort is
    stable, so triangle order inside a cell (decal-over-terrain layering) is kept.
    """
    idx = np.ascontiguousarray(indices, np.uint32)
    ntri = len(idx) // 3
    if ntri < min_tris or len(vertices) == 0:
        return idx, None
    tri = idx[:ntri * 3].reshape(-1, 3)
    pos = np.asarray(vertices[:, :3], np.float32)
    tv = pos[tri]                                        # (T, 3, 3)
    cen = tv.mean(axis=1)[:, :2]
    lo, hi = pos[:, :2].min(axis=0), pos[:, :2].max(axis=0)
    if grid is None:
        grid = int(np.clip(np.ceil(np.max(hi - lo) / 128.0), 2, 32))
    size = np.maximum(hi - lo, 1e-3) / grid
    cxy = np.clip(((cen - lo) / size).astype(np.int64), 0, grid - 1)
    cid = cxy[:, 1] * grid + cxy[:, 0]
    order = np.argsort(cid, kind="stable")
    tri, tv, cid = tri[order], tv[order], cid[order]
    counts = np.bincount(cid, minlength=grid * grid)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    keep = counts > 0
    s = starts[keep]
    cmin = np.minimum.reduceat(tv.min(axis=1), s, axis=0)
    cmax = np.maximum.reduceat(tv.max(axis=1), s, axis=0)
    cells = np.column_stack((s * 3, counts[keep] * 3, cmin, cmax)).astype(np.float64)
    return np.ascontiguousarray(tri.reshape(-1), np.uint32), cells



def _meshdata(pair):
    """Wrap an (vertices, indices) pair from urban.py as MeshData."""
    if pair is None or len(pair[1]) == 0:
        return None
    v, i = pair
    return MeshData(np.ascontiguousarray(v, np.float32),
                    np.ascontiguousarray(i, np.uint32),
                    np.empty((0, 6), np.float32),
                    float(np.max(np.hypot(v[:, 0], v[:, 1]))),
                    float(v[:, 2].max()))


# ================================================================ tile job
def mesh_tile_job(payload):
    """Build one tile's full 3D payload. Top-level -> picklable for process pool."""
    (raw, lat0, lon0, z, tx, ty, default_height, level_height,
     heights, sea_z) = payload
    proj = LocalProjection(lat0, lon0)
    s_, w_, n_, e_ = tile_bounds(z, tx, ty)
    sw, ne = proj.to_local(s_, w_), proj.to_local(n_, e_)
    bbox = (float(sw[0]), float(sw[1]), float(ne[0]), float(ne[1]))
    hf = HeightField(heights, *bbox) if heights is not None else None
    places = parse_places(raw, proj, owner=(z, tx, ty))
    buildings, roads, greens, waters, trees = parse_tile_elements(
        raw, proj, owner=(z, tx, ty),
        default_height=default_height, level_height=level_height,
    )
    sea_piers = MeshExtruder.sea_piers(buildings, waters, hf, sea_z, bbox)
    buildings, sea_piers = MeshExtruder.drop_deck_buildings(
        buildings, roads, hf, bbox, sea_z, sea_piers)
    buildings = MeshExtruder.prepare_buildings(buildings, roads, hf, bbox, sea_z)
    roads = clip_roads_to_bbox(roads, bbox)
    paths = road_paths_from_roads(roads, hf=hf, sea_z=sea_z, bbox=bbox)
    b_mesh, g_mesh, w_mesh, t_mesh, l_mesh, parked = MeshExtruder().build(
        buildings, roads, greens, waters, trees, paths=paths,
        hf=hf, bbox=bbox, sea_z=sea_z, sea_piers=sea_piers,
    )

    # ---- street-level detail: signage, furniture, sidewalks ----------------
    bands = crosswalk_bands(paths)
    cutouts = crosswalk_cutouts(bands)
    streetscape = plan_streetscape(roads, paths, buildings, greens, waters, trees, bbox, cutouts)
    l_mesh = align_streetlight_bases(l_mesh, streetscape, hf, sea_z)
    neon, neon_solid = build_neon(buildings, hf, paths)
    props, prop_neon = build_props(roads, paths, hf, bbox, sea_z, buildings=buildings,
                                  pavement=streetscape.pavement, trees=trees)
    roadside, pit_soil = build_roadside_details(streetscape, roads, paths, hf, sea_z)
    crosswalks = _meshdata(build_crosswalks(paths, bands, hf))
    spires, spire_neon = build_crowns(buildings, hf)
    try:
        from .times_square import build_times_square
        ts_glow, ts_solid = build_times_square(buildings, proj, bbox, hf, paths)
    except Exception as exc:                      # decoration must never break a tile
        print(f"[mesh] times square skipped: {exc}")
        ts_glow = ts_solid = None
    s_mesh = _meshdata(merge_urban([neon, prop_neon, spire_neon, ts_glow]))
    p_mesh = _meshdata(merge_urban([props, roadside, spires, neon_solid, ts_solid]))
    if crosswalks is not None:
        g_mesh = MeshExtruder._merge_meshes(g_mesh, crosswalks)
    sidewalks = _meshdata(build_sidewalks(roads, paths, hf, sea_z,
                                         cutouts=cutouts, aprons=streetscape.aprons,
                                         exclusions=streetscape.exclusions, pits=streetscape.pits,
                                         clip=streetscape.clip, pavement=streetscape.pavement))
    if sidewalks is not None:
        g_mesh = MeshExtruder._merge_meshes(g_mesh, sidewalks)
    g_mesh = MeshExtruder._merge_meshes(g_mesh, _meshdata(pit_soil))

    def pack(m, sort=False):
        if m is None: return None
        idx, cells = m.indices, None
        if sort:                      # done here, in the worker, never on the render thread
            idx, cells = build_cells(m.vertices, m.indices)
        return (m.vertices, idx, m.edges, m.radius, m.max_height, cells)
    return TileResult(key=(tx, ty), b_mesh=pack(b_mesh, True), g_mesh=pack(g_mesh, True),
                      w_mesh=pack(w_mesh), trees=t_mesh, lights=l_mesh,
                      parked=parked or {}, paths=paths,
                      s_mesh=pack(s_mesh), p_mesh=pack(p_mesh), places=places)


