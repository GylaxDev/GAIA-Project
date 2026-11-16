"""Historical 3D grass helpers, retained for old artifacts and isolated CPU tests.

The application no longer imports this module, builds placement samplers or
draws blades. Flat ground textures live in ground_textures.py instead.

The former renderer retained a compact triangle sampler, not millions of grass instances. Only
4-metre patches around the camera are populated, with a bounded build budget
and LRU. The sampler uses the rendered terrain's height/material interpolation
and topmost road/path overlays, including in old mesh caches.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np

GRASS_SPACING = .125
GRASS_PATCH_SIZE = 4.
GRASS_DISTANCE = 24.
GRASS_BLADES = 8
GRASS_RADIUS = .75          # tuft + maximum bend, wind, height and .85 terrain slope
GRASS_CENTER_Z = .12
GRASS_MEAN_HEIGHT = .18
GRASS_MAX_TUFTS = 65536
GRASS_PATCH_CACHE = 192
GRASS_PATCH_BUDGET = 2
GRASS_STRIDE = 8            # root xyz, slope xy, seed, coverage, height scale
GRASS_ROOT_CLEARANCE = .085
# Cached root AABB, active root AABB, and active start/end row range per patch.
GRASS_PATCH_META_BYTES = 2 * 6 * 4 + 2 * 4


def grass_hash(x, y, salt=0):
    """Integer world-cell hashing; stable across tiles, order and negative XY."""
    shape = np.broadcast_shapes(np.shape(x), np.shape(y))
    x = np.atleast_1d(np.asarray(x, np.int64)).astype(np.uint64)
    y = np.atleast_1d(np.asarray(y, np.int64)).astype(np.uint64)
    # Adding the coordinate products avoids XOR's mirrored (x,y)/(-x,-y)
    # collisions along diagonals. Overflow is intentional unsigned arithmetic.
    h = x * np.uint64(0x9E3779B185EBCA87) + y * np.uint64(0xC2B2AE3D27D4EB4F)
    h ^= np.uint64((int(salt) * 0x165667B19E3779F9) & ((1 << 64) - 1))
    h ^= h >> np.uint64(30)
    h = h * np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h = h * np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    return ((h & np.uint64(0xFFFFFF)).astype(np.float64) / 16777216.).astype(np.float32).reshape(shape)


def _cell_pairs(box, origin, size, shape):
    """Triangle AABBs -> (cell, triangle) pairs, in bounded temporary batches."""
    ny, nx = shape
    lo = np.floor((box[:, :2] - origin) / size).astype(np.int64)
    hi = np.floor((box[:, 2:] - origin) / size).astype(np.int64)
    lo = np.clip(lo, 0, [nx - 1, ny - 1])
    hi = np.clip(hi, 0, [nx - 1, ny - 1])
    widths = hi - lo + 1
    counts = widths[:, 0] * widths[:, 1]
    first = 0
    while first < len(box):
        end = min(first + 2048, len(box))
        # Long legacy triangles can cover many cells. Avoid a huge broadcast.
        cumulative = np.cumsum(counts[first:end])
        end = first + max(1, int(np.searchsorted(cumulative, 200000, side='right')))
        n = counts[first:end]
        tri = np.repeat(np.arange(first, end), n)
        offset = np.arange(int(n.sum())) - np.repeat(np.r_[0, np.cumsum(n)[:-1]], n)
        cx = lo[tri, 0] + offset % widths[tri, 0]
        cy = lo[tri, 1] + offset // widths[tri, 0]
        yield cy * nx + cx, tri
        first = end


@dataclass
class GrassSurface:
    """Pickle/staging-safe CSR spatial sampler; all large data is plain NumPy."""
    triangles: np.ndarray   # a.xyz, (b-a).xyz, (c-a).xyz, vertex material IDs
    starts: np.ndarray
    ids: np.ndarray
    origin: np.ndarray
    cell_size: float
    shape: tuple
    bounds: np.ndarray      # bounds of potentially grassy triangles only

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.triangles, self.starts, self.ids, self.origin, self.bounds))

    def detached(self):
        """Do not pin an entire staged city file after the GPU upload completes."""
        return GrassSurface(self.triangles.copy(), self.starts.copy(), self.ids.copy(),
                            self.origin.copy(), self.cell_size, self.shape, self.bounds.copy())

    def sample(self, xy):
        """Return topmost (height, material ID, height slope) at mesh-local XY.

        Unsupported points have NaN height. Non-grass overlays are returned too,
        so callers can reject them instead of growing blades through pavement.
        """
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        height = np.full(len(xy), np.nan)
        material = np.full(len(xy), 5.)
        slope = np.zeros((len(xy), 2))
        ny, nx = self.shape
        cell = np.floor((xy - self.origin) / self.cell_size).astype(np.int64)
        inside = (cell[:, 0] >= 0) & (cell[:, 0] < nx) & (cell[:, 1] >= 0) & (cell[:, 1] < ny)
        points = np.flatnonzero(inside)
        cid = cell[points, 1] * nx + cell[points, 0]
        order = np.argsort(cid, kind='stable')
        points, cid = points[order], cid[order]
        unique, start, count = np.unique(cid, return_index=True, return_counts=True)
        for key, first, n in zip(unique, start, count):
            tid = self.ids[int(self.starts[key]):int(self.starts[key + 1])]
            if not len(tid):
                continue
            t = self.triangles[tid].astype(np.float64)
            a, e, f = t[:, :3], t[:, 3:6], t[:, 6:9]
            det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
            dx = xy[points[first:first+n], 0, None] - a[:, 0]
            dy = xy[points[first:first+n], 1, None] - a[:, 1]
            u = (dx * f[:, 1] - dy * f[:, 0]) / det
            v = (dy * e[:, 0] - dx * e[:, 1]) / det
            hit = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1. + 1e-8)
            z = a[:, 2] + u * e[:, 2] + v * f[:, 2]
            # Later coplanar overlays win, just as in the ground depth pass.
            top = len(t) - 1 - np.argmax(np.where(hit, z, -np.inf)[:, ::-1], axis=1)
            row = np.flatnonzero(hit.any(axis=1))
            chosen = top[row]
            dest = points[first:first+n][row]
            height[dest] = z[row, chosen]
            material[dest] = (t[chosen, 9] + u[row, chosen] * (t[chosen, 10] - t[chosen, 9])
                              + v[row, chosen] * (t[chosen, 11] - t[chosen, 9]))
            slope[dest, 0] = (e[chosen, 2] * f[chosen, 1] - f[chosen, 2] * e[chosen, 1]) / det[chosen]
            slope[dest, 1] = (f[chosen, 2] * e[chosen, 0] - e[chosen, 2] * f[chosen, 0]) / det[chosen]
        return height, material, slope


def build_grass_surface(mesh):
    """Worker-side compact placement index; no cache version change required."""
    if mesh is None or not len(mesh.indices):
        return None
    if mesh.vertices.shape[1] >= 9:
        pos, material = mesh.vertices[:, :3], mesh.vertices[:, 6]
    else:
        packed = getattr(mesh, 'gpu_vertices', None)
        if packed is None or 'color' not in (packed.dtype.names or ()):
            return None
        pos, material = packed['position'], packed['color'][:, 0]
    tri = np.asarray(mesh.indices).reshape(-1, 3)
    mats = material[tri]
    potential = (mats.min(axis=1) < 1.5) & (mats.max(axis=1) >= .5)
    if not potential.any():
        return None
    p = np.asarray(pos[tri], np.float32)
    e, f = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = e[:, 0].astype(np.float64) * f[:, 1] - e[:, 1].astype(np.float64) * f[:, 0]
    valid = (np.abs(area) > 1e-7) & np.isfinite(p).all(axis=(1, 2)) & np.isfinite(mats).all(axis=1)
    p, e, f, mats, potential = p[valid], e[valid], f[valid], mats[valid], potential[valid]
    if not potential.any():
        return None
    bounds = np.r_[p[potential].min(axis=(0, 1)), p[potential].max(axis=(0, 1))].astype(np.float64)
    origin = np.floor(bounds[:2] / 4.) * 4.
    span = bounds[3:5] - origin
    # The inclusive last cell needs one slot too, including exact boundaries.
    size = max(4., math.ceil(float(span.max()) / (255. * 4.)) * 4.)
    nx, ny = np.maximum(1, np.floor(span / size).astype(int) + 1)
    boxes = np.c_[p[:, :, :2].min(axis=1), p[:, :, :2].max(axis=1)]
    overlap = (boxes[:, :2] <= bounds[3:5]).all(axis=1) & (boxes[:, 2:] >= bounds[:2]).all(axis=1)
    p, e, f, mats, boxes, potential = p[overlap], e[overlap], f[overlap], mats[overlap], boxes[overlap], potential[overlap]
    active = np.zeros(int(nx * ny), bool)
    for cells, _ in _cell_pairs(boxes[potential], origin, size, (int(ny), int(nx))):
        active[cells] = True
    cells_out, ids_out = [], []
    for cells, ids in _cell_pairs(boxes, origin, size, (int(ny), int(nx))):
        keep = active[cells]
        cells_out.append(cells[keep])
        ids_out.append(ids[keep])
    cells, ids = np.concatenate(cells_out), np.concatenate(ids_out)
    order = np.argsort(cells, kind='stable')
    counts = np.bincount(cells, minlength=int(nx * ny))
    starts = np.r_[0, np.cumsum(counts)].astype(np.uint32)
    # The grassy AABB can span large paved areas between disconnected lawns.
    # Retain only triangles referenced by active cells, including every road/
    # path overlay in those cells. Monotone remapping preserves coplanar order.
    referenced = np.zeros(len(p), bool)
    referenced[ids] = True
    remap = np.cumsum(referenced, dtype=np.uint32)
    compact_ids = remap[ids[order]] - np.uint32(1)
    data = np.ascontiguousarray(np.c_[p[referenced, 0], e[referenced], f[referenced], mats[referenced]], np.float32)
    return GrassSurface(data, starts, np.ascontiguousarray(compact_ids, np.uint32),
                        origin, size, (int(ny), int(nx)), bounds)


def build_grass_template(level=0):
    """Eight tapered ribbons: five/three triangles per blade, no alpha cards.

    Both LODs retain every blade, root and tip. Vertices encode (side, curve t,
    blade index); the shared scene/caster shader evaluates the actual geometry.
    """
    if level not in (0, 1):
        raise ValueError('grass LOD must be 0 or 1')
    segments = (3, 2)[level]
    vertices, indices = [], []
    for blade in range(GRASS_BLADES):
        base = len(vertices)
        for j in range(segments):
            vertices.extend(((-1., j / segments, blade), (1., j / segments, blade)))
        vertices.append((0., 1., blade))
        for j in range(segments - 1):
            a = base + j * 2
            indices.extend((a, a + 1, a + 3, a, a + 3, a + 2))
        a = base + (segments - 1) * 2
        indices.extend((a, a + 1, a + 2))
    return np.ascontiguousarray(vertices, np.float32), np.asarray(indices, np.uint32)


def _smoothstep(lo, hi, value):
    t = np.clip((value - lo) / (hi - lo), 0., 1.)
    return t * t * (3. - 2. * t)


def grass_coverage(rows, eye, pixel_scale, distance=GRASS_DISTANCE):
    """Smooth, world-seeded density/height reduction into the distant turf."""
    rows = np.asarray(rows, np.float32)
    delta = rows[:, :3] - np.asarray(eye, np.float32)
    d = np.sqrt(np.einsum('ij,ij->i', delta, delta))
    ratio = np.float32(8.) / np.maximum(d, np.float32(8.))
    probability = ratio * np.sqrt(ratio)
    # Independent selection from the blade seed, not a screen-space dissolve.
    seed = rows[:, 5] * np.float32(13.731)
    pick = seed - np.floor(seed)
    density = _smoothstep(pick - .07, pick + .07, probability)
    density[d <= 8.] = 1.
    pixels = np.float32(pixel_scale * GRASS_MEAN_HEIGHT) * rows[:, 7] / np.maximum(d, .2)
    coverage = density * (1. - _smoothstep(max(distance, .001) * .78, max(distance, .001), d)) * _smoothstep(2., 4., pixels)
    if distance <= 0.:
        coverage[:] = 0.
    return coverage, np.where(pixels > 13., np.uint8(0), np.uint8(1)), d


def grass_patch_mask(bounds, eye, distance, planes=None):
    """Conservative broadphase for root AABBs, including the entire bent tuft.

    Only the small patch table uses float64. Its rounding guard also covers the
    float32 narrowphase's plane arithmetic at large world coordinates.
    """
    b = np.asarray(bounds, np.float32).reshape(-1, 6).copy()
    b[:, (2, 5)] += np.float32(GRASS_CENTER_Z)
    b = b.astype(np.float64)
    eye = np.asarray(eye, np.float32).astype(np.float64)
    delta = np.maximum(np.maximum(b[:, :3] - eye, eye - b[:, 3:]), 0.)
    mask = np.einsum('ij,ij->i', delta, delta) <= (distance + GRASS_RADIUS + .001) ** 2
    if planes is not None:
        center, half = (b[:, :3] + b[:, 3:]) * .5, (b[:, 3:] - b[:, :3]) * .5
        magnitude = np.maximum(np.abs(b[:, :3]), np.abs(b[:, 3:]))
        for p in np.asarray(planes, np.float32).astype(np.float64):
            reach = half @ np.abs(p[:3])
            guard = .001 + 8. * np.finfo(np.float32).eps * (magnitude @ np.abs(p[:3]) + abs(p[3]))
            mask &= center @ p[:3] + p[3] + reach >= -GRASS_RADIUS - guard
    return mask


class GrassField:
    def __init__(self, capacity=GRASS_PATCH_CACHE):
        self.capacity = max(1, int(capacity))
        self.patches = OrderedDict()
        self.sources = ()
        self._source_key = None
        self.revision = 0
        self.pending = self.last_built = 0
        self._rows_key = None
        self._rows = np.empty((0, GRASS_STRIDE), np.float32)
        self._patch_bounds = {}
        self.patch_bounds = np.empty((0, 6), np.float32)
        self.patch_ranges = np.empty((0, 2), np.uint32)
        self._update_key = None

    def clear(self):
        self.patches.clear()
        self.sources = ()
        self._source_key = self._rows_key = None
        self._update_key = None
        self._rows = np.empty((0, GRASS_STRIDE), np.float32)
        self._patch_bounds.clear()
        self.patch_bounds = np.empty((0, 6), np.float32)
        self.patch_ranges = np.empty((0, 2), np.uint32)
        self.pending = self.last_built = 0
        self.revision += 1

    @property
    def nbytes(self):
        return (sum(a.nbytes for a in self.patches.values()) + self._rows.nbytes
                + sum(a.nbytes for a in self._patch_bounds.values())
                + self.patch_bounds.nbytes + self.patch_ranges.nbytes)

    def candidate_rows(self, patches):
        """Copy just selected patch ranges, in the full field's stable row order."""
        blocks = [self._rows[int(first):int(end)] for first, end in self.patch_ranges[patches]]
        return np.concatenate(blocks) if blocks else np.empty((0, GRASS_STRIDE), np.float32)

    def nearest_root(self, eye):
        """Exact nearest root index, without scanning the entire active field.

        Root-AABB lower bounds reject patches that cannot improve the current
        result. Float32 distances and stable row-order ties match coverage.
        """
        if not len(self._rows):
            return None
        eye = np.asarray(eye, np.float32)
        b = self.patch_bounds.astype(np.float64)
        delta = np.maximum(np.maximum(b[:, :3] - eye, eye - b[:, 3:]), 0.)
        lower2 = np.einsum('ij,ij->i', delta, delta)
        best, index = math.inf, len(self._rows)
        for patch in np.argsort(lower2, kind='stable'):
            if lower2[patch] > best * best * (1. + 8. * np.finfo(np.float32).eps) + .0001:
                break
            first, end = map(int, self.patch_ranges[patch])
            delta = self._rows[first:end, :3] - eye
            distances = np.sqrt(np.einsum('ij,ij->i', delta, delta))
            root = int(np.argmin(distances))
            d = float(distances[root])
            if d < best or (d == best and first + root < index):
                best, index = d, first + root
        return index

    def sync(self, sources):
        """Sources are (GrassSurface, world offset xy, tile xy scale)."""
        key = tuple((id(s), tuple(o), tuple(scale)) for s, o, scale in sources)
        if key != self._source_key:
            self.clear()
            self.sources = tuple(sources)
            self._source_key = key

    def sample(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        z, mat, slope = np.full(len(xy), np.nan), np.full(len(xy), 5.), np.zeros((len(xy), 2))
        for surface, offset, scale in self.sources:
            offset, scale = np.asarray(offset), np.asarray(scale)
            local = (xy - offset) / scale
            b = surface.bounds
            mask = (local >= b[:2]).all(axis=1) & (local <= b[3:5]).all(axis=1)
            ids = np.flatnonzero(mask)
            if not len(ids):
                continue
            h, m, g = surface.sample(local[ids])
            take = np.isfinite(h) & (~np.isfinite(z[ids]) | (h >= z[ids]))
            dest = ids[take]
            z[dest], mat[dest], slope[dest] = h[take], m[take], g[take] / scale
        return z, mat, slope

    def _patch(self, key):
        n = int(round(GRASS_PATCH_SIZE / GRASS_SPACING))
        xx, yy = np.meshgrid(np.arange(n) + key[0] * n, np.arange(n) + key[1] * n)
        x, y = xx.ravel(), yy.ravel()
        xy = np.c_[(x + .08 + .84 * grass_hash(x, y, 1)) * GRASS_SPACING,
                   (y + .08 + .84 * grass_hash(x, y, 2)) * GRASS_SPACING]
        c = GRASS_ROOT_CLEARANCE
        offsets = np.array([[0., 0.], [-c, -c], [-c, c], [c, -c], [c, c]])
        z, material, slope = self.sample((xy[:, None] + offsets).reshape(-1, 2))
        z, material, slope = z.reshape(-1, 5), material.reshape(-1, 5), slope.reshape(-1, 5, 2)
        predicted = z[:, :1] + np.sum(slope[:, :1] * offsets, axis=2)
        supported = np.isfinite(z).all(axis=1) & (material >= .5).all(axis=1) & (material < 1.5).all(axis=1)
        supported &= (np.sum(slope[:, 0] ** 2, axis=1) <= .85 ** 2)
        supported &= (np.abs(z - predicted) < .025).all(axis=1)
        ids = np.flatnonzero(supported)
        rows = np.ones((len(ids), GRASS_STRIDE), np.float32)
        rows[:, :2] = xy[ids]
        rows[:, 2] = z[ids, 0]
        rows[:, 3:5] = slope[ids, 0]
        rows[:, 5] = grass_hash(x[ids], y[ids], 3)
        macro = grass_hash(np.floor(xy[ids, 0] * .20), np.floor(xy[ids, 1] * .20), 7)
        rows[:, 7] = .80 + .40 * macro
        return rows

    def update(self, eye, distance=GRASS_DISTANCE, budget=GRASS_PATCH_BUDGET):
        """Populate at most `budget` patches. High/aerial views generate none."""
        eye = np.asarray(eye, np.float64)
        radius = max(0., min(float(distance), GRASS_DISTANCE))
        update_key = (eye.tobytes(), radius)
        if update_key == self._update_key and not self.pending:
            self.last_built = 0
            return self._rows
        self._update_key = update_key
        r = radius + GRASS_PATCH_SIZE * .71
        lo, hi = np.floor((eye[:2] - r) / GRASS_PATCH_SIZE).astype(int), np.floor((eye[:2] + r) / GRASS_PATCH_SIZE).astype(int)
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        grid = np.c_[xx.ravel(), yy.ravel()]
        center = (grid + .5) * GRASS_PATCH_SIZE
        d2 = np.sum((center - eye[:2]) ** 2, axis=1)
        a, b = center - GRASS_PATCH_SIZE * .5, center + GRASS_PATCH_SIZE * .5
        supported = np.zeros(len(grid), bool)
        for surface, offset, scale in self.sources:
            offset, scale = np.asarray(offset), np.asarray(scale)
            bounds = surface.bounds
            corners = np.array([bounds[:2] * scale + offset, bounds[3:5] * scale + offset])
            low, high = corners.min(axis=0), corners.max(axis=0)
            dz = max(bounds[2] - eye[2], eye[2] - bounds[5], 0.)
            if dz < radius:
                supported |= (a <= high).all(axis=1) & (b >= low).all(axis=1)
        selected = np.flatnonzero(supported & (d2 <= r * r))
        selected = selected[np.lexsort((grid[selected, 0], grid[selected, 1], d2[selected]))[:self.capacity]]
        active = [tuple(map(int, grid[i])) for i in selected]
        missing = [key for key in active if key not in self.patches]
        self.last_built = 0
        for key in missing[:max(0, int(budget))]:
            rows = self._patch(key)
            self.patches[key] = rows
            self._patch_bounds[key] = (np.r_[rows[:, :3].min(axis=0), rows[:, :3].max(axis=0)]
                                       if len(rows) else np.zeros(6, np.float32))
            self.last_built += 1
            self.revision += 1
        for key in active:
            if key in self.patches:
                self.patches.move_to_end(key)
        while len(self.patches) > self.capacity:
            key, _ = self.patches.popitem(last=False)
            self._patch_bounds.pop(key)
            self.revision += 1
        self.pending = len(missing) - self.last_built
        # Camera movement can reorder nearest-first build priorities without
        # changing the active patch set. Do not concatenate the same field again.
        row_key = (self.revision, tuple(sorted(active)))
        if row_key != self._rows_key:
            keys = [key for key in row_key[1] if key in self.patches and len(self.patches[key])]
            blocks = [self.patches[key] for key in keys]
            self._rows = np.concatenate(blocks) if blocks else np.empty((0, GRASS_STRIDE), np.float32)
            self.patch_bounds = np.asarray([self._patch_bounds[key] for key in keys], np.float32).reshape(-1, 6)
            ends = np.cumsum([len(a) for a in blocks], dtype=np.uint32)
            self.patch_ranges = np.column_stack((np.r_[np.uint32(0), ends[:-1]], ends)) if len(ends) else np.empty((0, 2), np.uint32)
            self._rows_key = row_key
        return self._rows
