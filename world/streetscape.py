"""Worker-only pavement infill and restrained, terrain-grounded roadside detail.

The layout is temporary: only ordinary terrain/prop meshes leave the worker.
Mapped lawns, water, buildings, separate footways and sidewalk tags win over
inferred front aprons. Decorations add neither textures nor render passes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon

from ..core.config import ROAD_SURFACE
from .road_layout import road_width, road_surface_lift
from .surface_geometry import TerrainDraper, vertex_normals
from .data_fetcher import DRIVABLE_HIGHWAYS, sample_path_position
from .urban import (_Sink, _frontages, _hash, _sidewalk_band, _sidewalk_sides,
                    SIDEWALK_CLEARANCE, SIDEWALK_HEIGHT)

PAVEMENT_LIFT = ROAD_SURFACE + .064 + SIDEWALK_HEIGHT
MAX_FRONT_GAP = 6.5
MAX_TREE_PITS = 160
MAX_DRAINS = 96
MAX_LEAVES = 1400
LEAF_COLOURS = np.array([(.43, .34, .21), (.53, .40, .24), (.35, .37, .23), (.34, .27, .18)])
_FRONT_KINDS = frozenset(('yes', 'retail', 'office', 'commercial', 'apartments', 'residential',
                          'mixed', 'hotel', 'supermarket', 'public', 'brownstone', 'terrace', 'tenement'))


def _union(polygons):
    return shapely.union_all([p for p in polygons if p is not None and not p.is_empty])


def _subtract(surface, blockers):
    # Do not re-union the entire road/footway network with every cutout merely
    # to subtract it. Sequential differences have the same set coverage.
    for p in blockers:
        if p is not None and not p.is_empty and not surface.is_empty:
            surface = surface.difference(p)
    return surface


def _ground(way):
    level = getattr(way, 'level', None)
    return (not way.tunnel and not way.bridge and way.layer == 0
            and (level is None or abs(float(level)) < 1e-6))


def _usable(path):
    return (_ground(path) and not path.covered and path.width >= 5. and path.total >= 30.
            and len(path.points) >= 2 and any(_sidewalk_sides(path)))


def _position_seed(point):
    return (int(round(point[0]*100))*73856093) ^ (int(round(point[1]*100))*19349663)


def _rectangle_points(center, tangent, half_x, half_y):
    normal = np.array([-tangent[1], tangent[0]])
    return np.array([center+tangent*x+normal*y for x, y in
                     ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))])


@dataclass(frozen=True)
class StreetscapeLayout:
    pavement: object
    aprons: tuple
    exclusions: object
    pits: object
    tree_pits: tuple
    carriageways: object
    cutouts: object
    clip: object
    ground_paths: tuple


def plan_streetscape(roads, paths, buildings=(), greens=(), waters=(), trees=None,
                    bbox=None, cutouts=None):
    """Plan close street-front infill and tree openings, never pave whole blocks."""
    usable = [p for p in paths if _usable(p)]
    bands = [_sidewalk_band(p) for p in usable]
    carriageways = _union([r.geometry.buffer(road_width(r)*.5, cap_style=2, join_style=2)
                           for r in roads if _ground(r)] +
                          [LineString(p.points).buffer(p.width*.5, cap_style=2, join_style=2)
                           for p in paths if _ground(p)])
    footprints = [b.footprint.buffer(.025, join_style=2) for b in buildings
                  if b.min_height < 2.5 and b.layer >= 0 and b.footprint is not None
                  and not b.footprint.is_empty]
    exclusions = _union(footprints + [getattr(w, 'geometry', w) for w in waters])
    landscape = _union([g.geometry for g in greens if g.material in (1., 4.)])
    guard = _union([exclusions, landscape])
    clip = shapely.box(*bbox) if bbox is not None else None
    blocked = (carriageways.buffer(SIDEWALK_CLEARANCE, join_style=2), exclusions,
               cutouts.buffer(SIDEWALK_CLEARANCE, join_style=2) if cutouts is not None else None)
    aprons, pit_records = [], []
    if usable:
        lines = [LineString(p.points) for p in usable]
        tree = shapely.STRtree(lines)
        front_reach = max(p.width for p in usable)*.5 + MAX_FRONT_GAP
        for b in buildings:
            if (b.min_height > .5 or b.layer < 0 or not b.street_details or b.footprint is None
                    or b.footprint.is_empty or not (b.kind in _FRONT_KINDS or b.shop or b.amenity)):
                continue
            if not len(tree.query(b.footprint, predicate='dwithin', distance=front_reach)):
                continue
            for a, end, tangent, _, _ in _frontages(b.footprint, tree):
                mid = (a+end)*.5
                hit = int(tree.nearest(shapely.Point(*mid)))
                p, line = usable[hit], lines[hit]
                s = float(line.project(shapely.Point(*mid)))
                x, y, heading = sample_path_position(p, s)
                along = np.array([math.cos(heading), math.sin(heading)])
                across = np.array([-along[1], along[0]])
                delta = mid-np.array([x, y])
                side = 0 if float(delta @ across) > 0. else 1
                gap = float(np.linalg.norm(delta))-p.width*.5
                if (not _sidewalk_sides(p)[side] or gap > MAX_FRONT_GAP or gap < .10
                        or abs(float(tangent @ along)) < .75):
                    continue
                near = []
                for point in (a, end):
                    q = line.interpolate(line.project(shapely.Point(*point)))
                    q = np.array([q.x, q.y])
                    d = point-q
                    length = float(np.linalg.norm(d))
                    near.append(q+d/max(length, 1e-6)*(p.width*.5+SIDEWALK_CLEARANCE))
                apron = Polygon([near[0], near[1], end, a])
                if not apron.is_valid: apron = shapely.make_valid(apron)
                apron = apron.difference(guard)
                if clip is not None: apron = apron.intersection(clip)
                if apron.area > .5:
                    aprons.append((apron, p))
        surface = _subtract(_union(bands+[a for a, _ in aprons]), blocked)
        # Avenue trees just outside a narrow band get a small connected surround,
        # not an isolated slab or a continuous invented strip across a lawn.
        points = np.asarray(trees, float) if trees is not None and len(trees) else np.empty((0, 2))
        points = points[np.isfinite(points[:, :2]).all(axis=1), :2]
        if clip is not None: points = points[shapely.contains_xy(clip, points[:, 0], points[:, 1])]
        # Stable spatial hashing spreads the cap across the tile; an x-sort
        # otherwise gave every tree surround to the westernmost few streets.
        for point in sorted(points.tolist(), key=lambda q: (_hash(_position_seed(q), 191), *q)):
            if len(pit_records) >= MAX_TREE_PITS: break
            point = np.asarray(point)
            if not np.isfinite(point).all() or surface.distance(shapely.Point(*point)) > .75: continue
            hit = int(tree.nearest(shapely.Point(*point)))
            p, line = usable[hit], lines[hit]
            s = float(line.project(shapely.Point(*point)))
            x, y, heading = sample_path_position(p, s)
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            offset = float((point-np.array([x, y])) @ normal)
            if not _sidewalk_sides(p)[0 if offset > 0. else 1]: continue
            pit = Polygon(_rectangle_points(point, tangent, .70, .58))
            pad = pit.buffer(.13, join_style=2)
            if (pad.intersects(guard) or any(pad.intersects(g) for g in blocked if g is not None)
                    or (clip is not None and not clip.covers(pad))
                    or any(pad.intersects(record[0]) for record in pit_records)):
                continue
            if not surface.covers(pad): aprons.append((pad, p))
            pit_records.append((pit, point, tangent, _position_seed(point)))
    pits = _union([record[0] for record in pit_records])
    pavement = _subtract(_union(bands+[a for a, _ in aprons]), (*blocked, pits))
    if clip is not None: pavement = pavement.intersection(clip)
    return StreetscapeLayout(pavement, tuple(aprons), exclusions, pits, tuple(pit_records),
                             carriageways, cutouts, clip, tuple(usable))


def _heights(q, hf, sea_z, lift, *, clamp=True):
    base = np.asarray(hf.at(q[:, 0], q[:, 1]), float) if hf is not None else np.full(len(q), sea_z)
    if clamp: base = np.maximum(base, sea_z)
    return base+lift


def align_streetlight_bases(lights, layout, hf=None, sea_z=0.):
    """Seat cast bases without moving the head or changing the original arm.

    An optional sixth source column retains the original mount height above
    the centreline datum. The renderer still emits nine-float GPU instances.
    """
    if lights is None or not len(lights) or lights.shape[1] < 5: return lights
    lights = lights.copy()
    q = lights[:, :2]
    z = _heights(q, hf, sea_z, PAVEMENT_LIFT-.002)
    mask = shapely.contains_xy(layout.pavement, q[:, 0], q[:, 1])
    # Compare to the source centreline profile, not the sidewalk elevation:
    # a wide street on a side slope can put its curb more than a metre higher.
    if layout.ground_paths:
        lines = [LineString(p.points) for p in layout.ground_paths]
        owners = shapely.STRtree(lines).nearest(shapely.points(q))
        source_z = np.full(len(q), np.inf)
        for owner in np.unique(owners):
            selected = owners == owner
            path = layout.ground_paths[owner]
            arc = shapely.line_locate_point(lines[owner], shapely.points(q[selected]))
            source_z[selected] = np.interp(arc, path.arc, path.z) if path.z is not None else sea_z
        mask &= np.abs(source_z-lights[:, 4]) < .10  # preserve elevated bridge lamps
    else:
        mask[:] = False
    if mask.any() and lights.shape[1] == 5:
        lights = np.column_stack((lights, lights[:, 2]-lights[:, 4]))
    lights[mask, 4] = z[mask]
    return lights


def _metal_rectangle(sink, center, tangent, half_x, half_y, lift, color, hf, sea_z,
                     seed, *, depth=0., clamp=True):
    q = _rectangle_points(center, tangent, half_x, half_y)
    h = _heights(q, hf, sea_z, lift, clamp=clamp)
    points = np.c_[q, h]
    uv = ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))
    style = (_hash(seed, 11), 4., 0.)
    sink.quad(*points, color, 7., style, uv, emissive=0., ao=.92)
    if depth:
        for j in range(4):
            k = (j+1) % 4
            a, b = points[j].copy(), points[k].copy()
            a[2] -= depth; b[2] -= depth
            sink.quad(a, b, points[k], points[j], color*.80, 7., style, emissive=0.)


def build_roadside_details(layout, roads, paths, hf=None, sea_z=0.):
    """Return (opaque detail, pit soil) with hard per-tile geometry budgets."""
    props, soil, draper = _Sink(), _Sink(), TerrainDraper(hf)
    leaves = drains = 0
    iron = np.array([.27, .285, .26])
    road_indices = {r.uid: i for i, r in enumerate(roads) if r.uid}

    def leaf(center, seed, k, pit=None):
        nonlocal leaves
        if leaves >= MAX_LEAVES: return
        angle = _hash(seed, k+7)*math.tau
        tangent = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-tangent[1], tangent[0]])
        length = .055+_hash(seed, k+9)*.070
        width = length*(.35+_hash(seed, k+17)*.20)
        q = np.array([center-tangent*length*.5, center-normal*width*.5,
                      center+tangent*length*.5, center+normal*width*.5])
        footprint = Polygon(q)
        in_pit = pit is not None and pit.covers(footprint)
        if not in_pit and not layout.pavement.covers(footprint): return
        lift = .009 if in_pit else PAVEMENT_LIFT+.003
        h = _heights(q, hf, sea_z, lift)
        ridge = np.r_[center, _heights(center[None], hf, sea_z, lift+.005)[0]]
        points = np.c_[q, h]
        tone = _hash(seed, k+31)
        color = LEAF_COLOURS[int(tone*len(LEAF_COLOURS)) % len(LEAF_COLOURS)]
        for j in range(4):
            props.tri(ridge, points[j], points[(j+1) % 4], color, 7.,
                      style=(tone, 7., 0.), emissive=0., ao=.86)
        leaves += 1

    pit_leaf_samples = min(18, MAX_LEAVES // max(2*len(layout.tree_pits), 1))
    for pit, center, tangent, seed in layout.tree_pits:
        q = np.asarray(pit.exterior.coords)[:-1]
        pts, tris = draper.conform(q, np.array([[0, 1, 2], [0, 2, 3]]))
        xyz = np.c_[pts, _heights(pts, hf, sea_z, .006)]
        block = np.zeros((len(pts), 17), np.float32)
        block[:, :3], block[:, 3:6] = xyz, vertex_normals(xyz, tris)
        block[:, 6], block[:, 9], block[:, 16] = 4., 1., 5.
        block[:, 13:15] = pts
        soil.v.append(block); soil.i.append(tris.ravel().astype(np.uint32)+soil.n); soil.n += len(pts)
        normal = np.array([-tangent[1], tangent[0]])
        for side in (-1., 1.):
            _metal_rectangle(props, center+normal*side*.56, tangent, .70, .022,
                             PAVEMENT_LIFT+.002, iron, hf, sea_z, seed, depth=.034)
            _metal_rectangle(props, center+tangent*side*.68, normal, .54, .022,
                             PAVEMENT_LIFT+.002, iron, hf, sea_z, seed, depth=.034)
        if _hash(seed, 5) < .48:
            for y in np.linspace(-.48, .48, 7):
                spans = ((-.65, -.21), (.21, .65)) if abs(y) < .22 else ((-.65, .65),)
                for lo, hi in spans:
                    _metal_rectangle(props, center+tangent*(lo+hi)*.5+normal*y, tangent,
                                     (hi-lo)*.5, .014, PAVEMENT_LIFT-.003,
                                     iron*.83, hf, sea_z, seed, depth=.022)
        for j in range(pit_leaf_samples):
            delta = tangent*(_hash(seed, j+37)-.5)*2.2+normal*(_hash(seed, j+73)-.5)*1.8
            leaf(center+delta, seed, j*43, pit)

    # Reserve gutter litter even in tree-heavy tiles and avoid spending the
    # entire roadside budget on the first few OSM ways in the payload.
    order = sorted(((i, p) for i, p in enumerate(paths) if _usable(p)),
                   key=lambda item: _hash(item[1].uid or _position_seed(item[1].points[0]), 211))
    for pi, p in order:
        seed = p.uid or _position_seed(p.points[0])
        for si, s in enumerate(np.arange(10.+_hash(seed, 79)*14., p.total-5., 23.)):
            if leaves >= MAX_LEAVES: break
            x, y, heading = sample_path_position(p, float(s))
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            for side in (-1., 1.):
                if not _sidewalk_sides(p)[0 if side > 0. else 1]: continue
                for j in range(6):
                    k = si*151+j*23+(0 if side > 0. else 61)
                    center = (np.array([x, y])+tangent*(_hash(seed, k+101)-.5)*2.8
                              +normal*side*(p.width*.5+.12+_hash(seed, k+109)*.58))
                    leaf(center, seed, k)
        # Gutters belong to carriageways. A wide pedestrian street or shared
        # path has no road-surface drain, and its footpath lift is 10 cm higher.
        if drains >= MAX_DRAINS or p.highway not in DRIVABLE_HIGHWAYS: continue
        line = LineString(p.points).buffer(p.width*.5, cap_style=2, join_style=2)
        for si, s in enumerate(np.arange(24.+_hash(seed, 83)*13., p.total-12., 86.)):
            if drains >= MAX_DRAINS: break
            x, y, heading = sample_path_position(p, float(s))
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            side = 1. if si % 2 == 0 else -1.
            if not _sidewalk_sides(p)[0 if side > 0. else 1]: side = -side
            center = np.array([x, y])+normal*side*(p.width*.5-.27)
            footprint = Polygon(_rectangle_points(center, tangent, .48, .22))
            if (not line.covers(footprint) or layout.carriageways.boundary.distance(shapely.Point(*center)) > .6
                    or (layout.clip is not None and not layout.clip.covers(footprint))
                    or (layout.cutouts is not None and layout.cutouts.intersects(footprint))):
                continue
            lift = ROAD_SURFACE+road_surface_lift(p.highway, road_indices.get(p.uid, pi))+.002
            _metal_rectangle(props, center, tangent, .48, .22, lift,
                             np.array([.105, .115, .103]), hf, sea_z, seed, clamp=False)
            for y in (-.205, .205):
                _metal_rectangle(props, center+normal*y, tangent, .48, .017, lift+.006,
                                 iron, hf, sea_z, seed, clamp=False)
            for x in (-.461, .461):
                _metal_rectangle(props, center+tangent*x, normal, .188, .017, lift+.006,
                                 iron, hf, sea_z, seed, clamp=False)
            for x in np.linspace(-.38, .38, 8):
                _metal_rectangle(props, center+tangent*x, normal, .187, .018, lift+.004,
                                 iron*.87, hf, sea_z, seed, clamp=False)
            drains += 1
    return props.mesh(), soil.mesh()
