"""Resident, mesh-accurate driving support and collision; no GL or network calls.

Small position-only triangle indexes are prepared in the chunk worker. Queries
sample the rendered ground, not a fresh DEM download, and respect bridge decks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..world.road_layout import RoadLayout
from .vehicle_instances import LENGTHS, WHEELBASES, WIDTHS, HEIGHTS, WHEEL_RADII


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _mesh_triangles(mesh):
    if mesh is None or not len(getattr(mesh, 'indices', ())):
        return np.empty((0, 3, 3), np.float32)
    packed = getattr(mesh, 'gpu_vertices', None)
    xyz = packed['position'] if packed is not None else mesh.vertices[:, :3]
    return np.ascontiguousarray(xyz[mesh.indices.reshape(-1, 3)], np.float32)


class TriangleIndex:
    CELL = 32.

    def __init__(self, triangles):
        t = np.asarray(triangles, np.float32).reshape(-1, 3, 3)
        t = t[np.isfinite(t).all(axis=(1, 2))]
        # Keep shared positions indexed: expanding an ordinary water grid would
        # triple its resident payload and force otherwise-small uploads to mmap.
        vertices, indices = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
        self.vertices = np.ascontiguousarray(vertices)
        self.indices = np.ascontiguousarray(indices.reshape(-1, 3), np.int32)
        self.bounds = np.concatenate((t[:, :, :2].min(axis=1), t[:, :, :2].max(axis=1)), axis=1)
        low = np.floor(self.bounds[:, :2] / self.CELL).astype(np.int64)
        high = np.floor(self.bounds[:, 2:] / self.CELL).astype(np.int64)
        buckets, large = {}, []
        for i, ((x0, y0), (x1, y1)) in enumerate(zip(low, high)):
            # Long, older cached triangles must not create unbounded grid rows.
            if (x1 - x0 + 1) * (y1 - y0 + 1) > 128:
                large.append(i)
                continue
            for x in range(int(x0), int(x1) + 1):
                for y in range(int(y0), int(y1) + 1):
                    buckets.setdefault((x, y), []).append(i)
        self.buckets = {k: np.asarray(v, np.int32) for k, v in buckets.items()}
        self.large = np.asarray(large, np.int32)

    @property
    def triangles(self):
        return self.vertices[self.indices]

    def detach_buffers(self):
        """Persistent indexes must not retain the upload's large mapped file."""
        self.vertices = self.vertices.copy()
        self.indices = self.indices.copy()
        self.bounds = self.bounds.copy()
        self.large = self.large.copy()
        self.buckets = {k: v.copy() for k, v in self.buckets.items()}

    def query(self, points, pad=0.):
        points = np.asarray(points, np.float64).reshape(-1, 2)
        lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
        low, high = np.floor(lo / self.CELL).astype(int), np.floor(hi / self.CELL).astype(int)
        rows = [self.large] if len(self.large) else []
        for x in range(low[0], high[0] + 1):
            for y in range(low[1], high[1] + 1):
                ids = self.buckets.get((x, y))
                if ids is not None:
                    rows.append(ids)
        if not rows:
            return np.empty((0, 3, 3), np.float64)
        ids = np.unique(np.concatenate(rows))
        b = self.bounds[ids]
        keep = (b[:, :2] <= hi).all(axis=1) & (b[:, 2:] >= lo).all(axis=1)
        return self.vertices[self.indices[ids[keep]]].astype(np.float64)

    def sample(self, points, reference=None):
        """Highest compatible facet at each XY; NaN means no support."""
        points = np.asarray(points, np.float64).reshape(-1, 2)
        t = self.query(points)
        result = np.full(len(points), np.nan)
        if not len(t):
            return result
        a, b = t[:, 1, :2] - t[:, 0, :2], t[:, 2, :2] - t[:, 0, :2]
        den = _cross2(a, b)
        good = np.abs(den) > 1e-8
        t, a, b, den = t[good], a[good], b[good], den[good]
        if not len(t):
            return result
        q = points[:, None] - t[None, :, 0, :2]
        u, v = _cross2(q, b) / den, _cross2(a, q) / den
        inside = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1. + 1e-6)
        z = t[None, :, 0, 2] + u * (t[:, 1, 2] - t[:, 0, 2]) + v * (t[:, 2, 2] - t[:, 0, 2])
        if reference is not None:
            # A car below an overpass cannot jump onto it, or fall off its deck.
            inside &= (z <= reference + .65) & (z >= reference - .85)
        heights = np.max(np.where(inside, z, -np.inf), axis=1)
        result[np.isfinite(heights)] = heights[np.isfinite(heights)]
        return result


@dataclass
class DrivingTile:
    ground: TriangleIndex
    walls: TriangleIndex
    water: TriangleIndex

    @classmethod
    def from_meshes(cls, building, ground, water):
        g, b, w = (_mesh_triangles(m) for m in (ground, building, water))
        gn = np.cross(g[:, 1] - g[:, 0], g[:, 2] - g[:, 0])
        bn = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
        # Vertical kerbs do not support tyres. Roofs are not facade barriers.
        g = g[np.abs(gn[:, 2]) > .45 * np.maximum(np.linalg.norm(gn, axis=1), 1e-8)]
        b = b[(np.abs(bn[:, 2]) < .75 * np.linalg.norm(bn, axis=1))
              & (np.ptp(b[:, :, 2], axis=1) > .15)]
        return cls(TriangleIndex(g), TriangleIndex(b), TriangleIndex(w))

    def detach_buffers(self):
        for index in (self.ground, self.walls, self.water):
            index.detach_buffers()


# A tyre is a circle, not a point: it starts climbing a kerb before its axle reaches
# the edge and rolls off a drop along its rim. Ground is sampled along each wheel's
# footprint and lifted onto the wheel's circle to get the real hub-support height.
ENVELOPE_SAMPLES = 9
ENVELOPE_REACH = .85        # fraction of the wheel radius sampled ahead/behind the axle
SUPPORT_LIFT = .015         # same clearance the body plane has always carried


@dataclass
class _Placement:
    tile: DrivingTile
    paths: tuple
    origin: np.ndarray
    scale: np.ndarray
    minimap: object = None
    places: tuple = ()

    def local(self, points):
        return (points - self.origin) / self.scale


class DrivingScene:
    def __init__(self):
        self.tiles = {}
        self.claimed = set()
        self.generation = 0

    def add_tile(self, key, tile, paths=(), origin=(0., 0.), scale=(1., 1.),
                 minimap=None, places=()):
        self.tiles[key] = _Placement(tile, tuple(paths or ()), np.asarray(origin[:2], np.float64),
                                     np.asarray(scale, np.float64), minimap, tuple(places or ()))
        self.generation += 1

    def remove_tile(self, key):
        if self.tiles.pop(key, None) is not None:
            self.generation += 1

    def clear(self):
        self.tiles.clear()
        self.claimed.clear()
        self.generation += 1
        self._map_places = None

    @staticmethod
    def parked_token(chunk, kind, row):
        return (chunk.key, kind, *np.round(row[12:15].astype(np.float64), 2))

    def filter_parked(self, chunk):
        """Keep a claimed curb car hidden even after its source tile reloads."""
        changed = False
        for kind, rows in enumerate(getattr(chunk, 'parked_cpu', None) or ()):
            if rows is None or not len(rows):
                continue
            keep = [self.parked_token(chunk, kind, row) not in self.claimed for row in rows]
            if not all(keep):
                chunk.parked_cpu[kind] = rows[np.asarray(keep)]
                chunk.v_counts[kind] = int(sum(keep))
                changed = True
        return changed

    def contact(self, position, heading, kind):
        """Body support plus the four actual wheel-surface heights.

        The body plane is fitted to the same wheel/centre samples used by
        ``support``; the returned wheel order is front-left, front-right,
        rear-left, rear-right.
        """
        c, s = math.cos(heading), math.sin(heading)
        axle, track = WHEELBASES[kind] * .5, WIDTHS[kind] * .42
        offsets = np.array([[0, 0], [axle, track], [axle, -track],
                            [-axle, track], [-axle, -track]])
        points = position[:2] + offsets @ np.array([[c, s], [-s, c]])
        heights, water = np.full(5, -np.inf), np.full(5, -np.inf)
        for placed in self.tiles.values():
            local = placed.local(points)
            heights = np.fmax(heights, placed.tile.ground.sample(local, float(position[2])))
            water = np.fmax(water, placed.tile.water.sample(local))
        if not np.isfinite(heights).all() or (water > heights + .025).any():
            return None
        pitch = math.atan2(float(heights[1:3].mean() - heights[3:5].mean()), 2 * axle)
        roll = math.atan2(float(heights[[1, 3]].mean() - heights[[2, 4]].mean()), 2 * track)
        if abs(pitch) > .60 or abs(roll) > .45:
            return None
        support = max(float(heights[1:].mean()), float(heights[0])) + .015
        return (support, pitch, roll), heights[1:].copy()

    def ground_contact(self, position, heading, kind):
        """Body support plus the tyre-envelope ground height under each wheel.

        Same pose conventions as ``contact`` (order FL, FR, RL, RR), but the wheel
        heights are absolute, include the body plane's clearance and are enveloped
        by the wheel circle, so they rise and fall continuously across a kerb or dip
        instead of stepping when the axle centre crosses the edge.
        """
        c, s = math.cos(heading), math.sin(heading)
        axle, track = WHEELBASES[kind] * .5, WIDTHS[kind] * .42
        radius = float(WHEEL_RADII[kind])
        centres = np.array([[axle, track], [axle, -track], [-axle, track], [-axle, -track]])
        reach = radius * np.linspace(-ENVELOPE_REACH, ENVELOPE_REACH, ENVELOPE_SAMPLES)
        rim = (centres[:, None, :] + np.stack((reach, np.zeros_like(reach)), axis=1)[None]).reshape(-1, 2)
        offsets = np.vstack(([[0., 0.]], centres, rim))
        points = position[:2] + offsets @ np.array([[c, s], [-s, c]])
        heights, water = np.full(len(points), -np.inf), np.full(5, -np.inf)
        for placed in self.tiles.values():
            local = placed.local(points)
            heights = np.fmax(heights, placed.tile.ground.sample(local, float(position[2])))
            water = np.fmax(water, placed.tile.water.sample(local[:5]))
        base = heights[:5]
        if not np.isfinite(base).all() or (water > base + .025).any():
            return None
        pitch = math.atan2(float(base[1:3].mean() - base[3:5].mean()), 2 * axle)
        roll = math.atan2(float(base[[1, 3]].mean() - base[[2, 4]].mean()), 2 * track)
        if abs(pitch) > .60 or abs(roll) > .45:
            return None
        support = max(float(base[1:].mean()), float(base[0])) + SUPPORT_LIFT
        arc = np.sqrt(np.maximum(radius * radius - reach * reach, 0.)) - radius
        rim_heights = heights[5:].reshape(4, ENVELOPE_SAMPLES)
        envelope = np.where(np.isfinite(rim_heights), rim_heights + arc, -np.inf).max(axis=1)
        ground = np.maximum(envelope, base[1:]) + SUPPORT_LIFT
        return (support, pitch, roll), ground.astype(np.float64)

    def support(self, position, heading, kind):
        result = self.contact(position, heading, kind)
        return None if result is None else result[0]

    def blocked(self, position, heading, kind):
        """SAT against actual facade triangles, not coarse renderer cell boxes."""
        length, width, height = LENGTHS[kind] + .12, WIDTHS[kind] + .10, HEIGHTS[kind]
        c, s = math.cos(heading), math.sin(heading)
        radius = math.hypot(length, width) * .5
        for placed in self.tiles.values():
            local = placed.local(np.asarray(position[:2]).reshape(1, 2))
            tri = placed.tile.walls.query(local, radius / float(placed.scale.min()))
            if not len(tri):
                continue
            z = (tri[:, :, 2].max(axis=1) > position[2] + .20) & (tri[:, :, 2].min(axis=1) < position[2] + height)
            tri = tri[z]
            if not len(tri):
                continue
            xy = tri[:, :, :2] * placed.scale + placed.origin - position[:2]
            p = xy @ np.array([[c, -s], [s, c]])
            half = np.array([length, width]) * .5
            hit = (p.min(axis=1) <= half).all(axis=1) & (p.max(axis=1) >= -half).all(axis=1)
            for edge in range(3):
                d = p[:, (edge + 1) % 3] - p[:, edge]
                n = np.column_stack((-d[:, 1], d[:, 0]))
                dots = np.einsum('nij,nj->ni', p, n)
                reach = np.abs(n) @ half
                hit &= (dots.min(axis=1) <= reach) & (dots.max(axis=1) >= -reach)
            if hit.any():
                return True
        return False

    def spawn_candidates(self, point, forward, kind=0, max_distance=600.):
        """Nearby legal lane poses, checked for clearance by the controller."""
        candidates = []
        for placed in self.tiles.values():
            for path in placed.paths:
                source = np.asarray(path.points, np.float64)
                p = source * placed.scale + placed.origin
                if len(p) < 2 or not np.isfinite(p).all():
                    continue
                a, d = p[:-1], np.diff(p, axis=0)
                l2 = np.sum(d * d, axis=1)
                lengths = np.sqrt(l2)
                arc = np.r_[0., np.cumsum(lengths)]
                if arc[-1] < LENGTHS[kind] + 4:
                    continue
                t = np.clip(np.sum((point[:2] - a) * d, axis=1) / np.maximum(l2, 1e-9), 0., 1.)
                z = np.zeros(len(p)) if path.z is None else np.asarray(path.z)
                if z.shape != (len(p),) or not np.isfinite(z).all():
                    continue
                distance2 = np.sum((a + t[:, None] * d - point[:2]) ** 2, axis=1)
                valid = np.flatnonzero(l2 >= 1e-6)
                nearest = valid[np.argsort(distance2[valid])[:8]]
                # Margins belong to the road's ends, not each OSM shape segment.
                # Interpolate shifted arc positions across short/repeated segments.
                margin = LENGTHS[kind] * .5 + .5
                along = {float(np.clip(arc[i] + t[i] * lengths[i] + shift,
                                       margin, arc[-1] - margin))
                         for i in nearest
                         for shift in (0., 8., -8., 16., -16.)}
                layout = RoadLayout.from_road(path)
                for direction in (True, False):
                    sign = 1 if direction else -1
                    for offset in layout.offsets(direction):
                        for distance in sorted(along):
                            i = int(np.searchsorted(arc, distance, side='right') - 1)
                            f = d[i] * sign / math.sqrt(l2[i])
                            u = (distance - arc[i]) / lengths[i]
                            local_d = source[i + 1] - source[i]
                            right = np.array([local_d[1], -local_d[0]]) * sign / np.linalg.norm(local_d)
                            xy = a[i] + u * d[i] + offset * right * placed.scale
                            dist = float(np.linalg.norm(xy - point[:2]))
                            if dist > max_distance:
                                continue
                            pose = np.r_[xy, z[i] + u * (z[i + 1] - z[i])]
                            score = dist + (1. - float(f @ forward[:2])) * 3.
                            candidates.append((score, pose, math.atan2(f[1], f[0])))
        return sorted(candidates, key=lambda item: item[0])
