"""Baked sixth-generation Explorer-style SUV (2020 model year).

The supplied Ford Explorer ST OBJ is converted to a versioned runtime asset,
``assets/vehicles/explorer_2020.npz``: three exterior LODs with baked vertex
colours and aligned wheel pivot/role metadata. Metre-scale hard points follow
Ford's 2020 specifications: 5.050 m length, 3.025 m wheelbase, 1.699 m track,
2.004 m body / 2.268 m mirror width, 1.783 m height, 255/55R20 tyres.

The shared curve/surface coachwork helpers below still supply the procedural
reference fleet (Camry, F-150 and Express). All materials use the existing
ten-float, single-draw instanced format.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .vehicle_models import VehicleTemplate, ONE, P_PAINT, P_MATTE
from .vehicle_lamps import bake_lamp_coords
from .vehicle_coachwork import (
    _Coach, _unique, _emit_grid, _const_hint, MATTE,
    EXPLORER_LEN, EXPLORER_W, EXPLORER_H, EXPLORER_CLEARANCE,
    EXPLORER_AXLE_F, EXPLORER_AXLE_R, EXPLORER_TRACK,
    EXPLORER_MIRROR_W, EXPLORER_TYRE_WIDTH, EXPLORER_TYRE_RADIUS,
    EXPLORER_RIM_RADIUS,
)

AXLES = (EXPLORER_AXLE_F, EXPLORER_AXLE_R)
WR = EXPLORER_TYRE_RADIUS
ARCH = 0.452
_MODEL = Path(__file__).resolve().parents[1] / 'assets/vehicles/explorer_2020.npz'


@lru_cache(maxsize=1)
def _templates():
    """Load and validate the baked Explorer exterior at all three LODs."""
    with np.load(_MODEL, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata['version'] not in (1, 2):
            raise ValueError('Unsupported Explorer model asset version')
        expected = {
            'length': EXPLORER_LEN, 'width': EXPLORER_W, 'height': EXPLORER_H,
            'wheelbase': EXPLORER_AXLE_F - EXPLORER_AXLE_R,
            'front_axle': EXPLORER_AXLE_F, 'track': EXPLORER_TRACK,
            'tyre_width': EXPLORER_TYRE_WIDTH, 'tyre_radius': EXPLORER_TYRE_RADIUS,
            'rim_radius': EXPLORER_RIM_RADIUS, 'mirror_width': EXPLORER_MIRROR_W,
            'clearance': EXPLORER_CLEARANCE,
        }
        for key, value in expected.items():
            if abs(metadata['dimensions'][key] - value) > 1e-7:
                raise ValueError('Explorer hard points changed; rebuild with '
                                 'tools/bake_explorer_model.py')
        glass = metadata.get('glass', [0, 0, 0])
        result = tuple(VehicleTemplate(bake_lamp_coords(data[f'v{lod}'], data[f'i{lod}']), data[f'i{lod}'],
                                       len(data[f'i{lod}']) - glass[lod], data[f'a{lod}'], glass[lod])
                       for lod in range(3))
    for v, idx, _ in result:
        v.setflags(write=False)
        idx.setflags(write=False)
    for template in result:
        template.animation.setflags(write=False)
    return result


def _explorer(m, lod):
    """Append one baked Explorer LOD to a mesh without copying its arrays."""
    if lod not in (0, 1, 2):
        raise ValueError('Explorer LOD must be 0, 1 or 2')
    template = _templates()[lod]
    base = len(m.V)
    m.V.extend(template[0].tolist())
    m.A.extend(template.animation.tolist())
    m.I.extend((template[1].astype(np.int64) + base).tolist())
    m.glass_indices = template.glass_count   # lenses are the tail of the index list


class _Curve:
    """Shape-preserving C1 Hermite curve for measured coachwork hard points."""
    def __init__(self, x, y=None):
        if y is None:
            points = np.asarray(x, np.float64)
            x, y = points[:, 0], points[:, 1]
        self.x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        self.scalar = y.ndim == 1
        self.y = y.reshape(len(self.x), -1)
        h = np.diff(self.x)
        d = np.diff(self.y, axis=0) / h[:, None]
        self.m = np.zeros_like(self.y)
        if len(self.x) == 2:
            self.m[:] = d[0]
            return
        for k in range(1, len(self.x) - 1):
            same = d[k - 1] * d[k] > 0
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            self.m[k, same] = (w1 + w2) / (w1 / d[k - 1, same] + w2 / d[k, same])
        for end, i, j in ((0, 0, 1), (-1, -1, -2)):
            slope = ((2 * h[i] + h[j]) * d[i] - h[i] * d[j]) / (h[i] + h[j])
            slope = np.where(slope * d[i] > 0, slope, 0.)
            limit = (d[i] * d[j] <= 0) & (np.abs(slope) > 3 * np.abs(d[i]))
            self.m[end] = np.where(limit, 3 * d[i], slope)

    def __call__(self, x):
        q = np.clip(np.asarray(x, np.float64), self.x[0], self.x[-1])
        k = np.clip(np.searchsorted(self.x, q, side='right') - 1, 0, len(self.x) - 2)
        h = self.x[k + 1] - self.x[k]
        t = ((q - self.x[k]) / h)[..., None]
        values = ((2 * t ** 3 - 3 * t ** 2 + 1) * self.y[k]
                  + (t ** 3 - 2 * t ** 2 + t) * h[..., None] * self.m[k]
                  + (-2 * t ** 3 + 3 * t ** 2) * self.y[k + 1]
                  + (t ** 3 - t ** 2) * h[..., None] * self.m[k + 1])
        return values[..., 0] if self.scalar else values


def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _clip(poly, boundary):
    """Sutherland-Hodgman intersection of two convex planar polygons."""
    out = list(poly)
    sign = 1 if _area2(boundary[0], boundary[1], boundary[2]) >= 0 else -1
    for a, b in zip(boundary, np.roll(boundary, -1, axis=0)):
        src, out = out, []
        if not src:
            break
        prev = src[-1]
        dp = sign * _area2(a, b, prev)
        for cur in src:
            dc = sign * _area2(a, b, cur)
            if (dc >= -1e-10) != (dp >= -1e-10):
                out.append(prev + (cur - prev) * (dp / (dp - dc)))
            if dc >= -1e-10:
                out.append(cur)
            prev, dp = cur, dc
    return np.asarray(out, np.float64)


class _Surface:
    """Project a convex detail onto a snapshot of a tessellated surface."""
    def __init__(self, m, indices, axes, hint, select=None):
        self.m, self.hint = m, np.asarray(hint, np.float64)
        v = np.asarray(m.V, np.float64)
        tri = v[np.asarray(indices, np.int64).reshape(-1, 3)]
        if select is not None:
            tri = tri[select(tri)]
        uv = axes(tri[:, :, :3]) if callable(axes) else tri[:, :, list(axes)]
        edges = np.stack((uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]), axis=-1)
        keep = np.abs(np.linalg.det(edges)) > 1e-10
        self.tri, self.uv = tri[keep], uv[keep]
        self.inverse = np.linalg.inv(edges[keep])
        self.lo, self.hi = self.uv.min(axis=1), self.uv.max(axis=1)

    def patch(self, poly, part, col, off, normal_offset=False):
        poly = np.asarray(poly, np.float64)
        if len(poly) < 3:
            return
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        candidates = np.flatnonzero(((self.hi >= lo - 1e-9) & (self.lo <= hi + 1e-9)).all(axis=1))
        cache = self.m.__dict__.setdefault('_detail_weld', {})
        for k in candidates:
            clipped = _clip(poly, self.uv[k])
            if len(clipped) < 3:
                continue
            bc = (clipped - self.uv[k, 0]) @ self.inverse[k].T
            weights = np.column_stack((1 - bc.sum(axis=1), bc))
            normals = weights @ self.tri[k, :, 3:6]
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
            points = weights @ self.tri[k, :, :3] + off * (normals if normal_offset else self.hint)
            ids = []
            for point, normal in zip(points, normals):
                key = (tuple(np.round(point, 6)), tuple(np.round(normal, 4)), part, tuple(col))
                i = cache.get(key)
                if i is None:
                    i = cache[key] = self.m._v(point, normal, col, part)
                ids.append(i)
            for j in range(1, len(ids) - 1):
                if np.linalg.norm(np.cross(points[j] - points[0], points[j + 1] - points[0])) > 1e-11:
                    self.m.I.extend((ids[0], ids[j], ids[j + 1]))

    def line(self, points, width, part, col, off, clip_to=None, normal_offset=False):
        points = np.asarray(points, np.float64)
        for a, b in zip(points[:-1], points[1:]):
            d = b - a
            length = np.linalg.norm(d)
            if length < 1e-9:
                continue
            n = np.array([-d[1], d[0]]) * (width / (2 * length))
            q = np.asarray([a - n, b - n, b + n, a + n])
            if clip_to is not None:
                q = _clip(q, clip_to)
            self.patch(q, part, col, off, normal_offset=normal_offset)

    def ellipse(self, center, rx, ry, part, col, off, n=16, normal_offset=False):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        self.patch(np.column_stack((rx * np.cos(a), ry * np.sin(a))) + center, part, col, off,
                   normal_offset=normal_offset)


def _side_selection(s):
    def select(tri):
        p = tri[:, :, :3]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        return (p[:, :, 1].mean(axis=1) * s > 0.1) & (np.abs(n[:, 1]) > np.abs(n[:, 2]) * 0.45)
    return select


class _ExplorerCoach(_Coach):
    """Rounded, aperture-cut coachwork; reference variants supply hard points."""
    def __init__(self, m, xs, T, lod=0, *, axles=AXLES, radius=WR, arch=ARCH,
                 faces=None, sweep=(2.160, 2.130), open_top=None):
        self.lod = lod
        self.axles, self.radius, self.arch = axles, radius, arch
        self.sweep, self.open_top = sweep, open_top
        self.samples = ((1, 2, 3, 2, 2, 3, 3, 2, 2, 2),
                        (1, 1, 1, 1, 1, 2, 2, 1, 1, 1), (1,) * 10)[lod]
        self.edge = sum(self.samples[:6])
        super().__init__(m, xs, T)
        self.faces = {s: _ExplorerFace(self, s, **(faces[s] if faces else {})) for s in (1, -1)}

    def profile(self, x):
        P = super().profile(x)
        wm, wb, zbe = (self.T[k](x) for k in ('Wm', 'Wb', 'zbe'))
        P[3, 0] = wm - .003
        P[5] = (wb - .014, zbe + min(.021, .45 * (P[6, 1] - zbe)))
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        curve = _Curve(distance, P)
        samples = np.concatenate([np.linspace(distance[k], distance[k + 1], n, endpoint=False)
                                  for k, n in enumerate(self.samples)] + [distance[-1:]])
        return curve(samples)

    def side_zmax(self, x):
        return float(self.prof_at(x)[self.edge, 1]) - .004

    def side_y(self, x, z):
        P = self.prof_at(x)[:self.edge + 1]
        return float(np.interp(min(z, P[-1, 1] - .004), P[:, 1], P[:, 0]))

    def top_z(self, x, y):
        P = self.prof_at(x)[self.edge:]
        return float(np.interp(abs(y), P[::-1, 0], P[::-1, 1]))

    def emit(self):
        start = len(self.m.I)
        zcen = .5 * (self.P[:, 0, 1] + self.P[:, -1, 1])
        zc_q = .5 * (zcen[:-1] + zcen[1:])[:, None]
        hint = lambda c: np.stack((np.zeros_like(c[..., 0]), c[..., 1], c[..., 2] - zc_q), -1)
        parts = np.full((len(self.xs) - 1, self.P.shape[1] - 1), P_PAINT)
        parts[:, 0] = P_MATTE
        if self.open_top is not None:
            xm = .5 * (self.xs[:-1] + self.xs[1:])
            parts[(xm > self.open_top[0]) & (xm < self.open_top[1]), self.edge:] = -1
        cols = np.broadcast_to(ONE, (*parts.shape, 3)).copy()
        cols[:, 0] = MATTE
        self.grids = {}
        for s in (1, -1):
            G = np.empty((*self.P.shape[:2], 3))
            G[:, :, 0] = self.xs[:, None]
            G[:, :, 1] = s * self.P[:, :, 0]
            G[:, :, 2] = self.P[:, :, 1]
            # Sweep the leading hood edge back at the corners. The complete
            # nose uses the same mapping, instead of meeting a flat end plate
            # at a 90-degree seam. Start beyond the measured wheel apertures.
            for face in self.faces.values():
                start_x = self.sweep[0 if face.s > 0 else 1]
                t = np.clip((face.s * self.xs - start_x) / (face.s * face.x0 - start_x), 0, 1)
                G[:, :, 0] += t[:, None] * (face.pos(G[:, :, 1], G[:, :, 2])[..., 0] - face.x0)
            for k, x in enumerate(self.xs):
                for ax in self.axles:
                    if abs(x - ax) <= self.arch:
                        floor = self.radius + np.sqrt(max(self.arch ** 2 - (x - ax) ** 2, 0.0))
                        for j in range(self.edge + 1):
                            if G[k, j, 2] < floor:
                                G[k, j, 1] = s * self.side_y(x, floor)
                                G[k, j, 2] = floor
            self.grids[s] = G
            _emit_grid(self.m, G, parts, cols, hint, cos_thr=-.5, center_col=self.P.shape[1] - 1)
        self.indices = self.m.I[start:]
        self._surfaces()

    def _surfaces(self):
        self.sides = {s: _Surface(self.m, self.indices, (0, 2), (0, s, 0), _side_selection(s)) for s in (1, -1)}
        def roof(tri):
            p = tri[:, :, :3]
            n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            return np.abs(n[:, 2]) > np.abs(n[:, 1]) * 1.5
        self.roof = _Surface(self.m, self.indices, (0, 1), (0, 0, 1), roof)

    def side(self, poly, part, col, off, normal_offset=False):
        for surface in self.sides.values():
            surface.patch(poly, part, col, off, normal_offset=normal_offset)

    def top(self, poly, part, col, off, mirror=False):
        self.roof.patch(poly, part, col, off)
        if mirror:
            self.roof.patch([(x, -y) for x, y in poly], part, col, off)


class _ExplorerFace:
    """Rounded fascia sharing the coachwork's exact curved end boundary."""
    def __init__(self, C, s, *, base=None, exponent=None, bulge=None, peak=None):
        self.C, self.s = C, s
        self.x0 = float(C.xs[-1] if s > 0 else C.xs[0])
        self.base = base if base is not None else (2.220 if s > 0 else -2.310)
        self.p = exponent if exponent is not None else (4.0 if s > 0 else 3.4)
        self.peak = peak if peak is not None else (.54 if s > 0 else .69)
        self.b = _Curve(bulge if bulge is not None else ([ (.330, .250), (.40, .278), (.54, .295), (.68, .288),
                         (.79, .274), (.95, .274), (1.035, .272), (1.080, .260),
                         (1.105, .247), (1.125, .225), (1.142, .190)] if s > 0 else
                        [(.370, .130), (.45, .166), (.58, .201), (.69, .215),
                         (.80, .176), (1.0, .100), (1.25, .070), (1.352, 0.)]))
        P = C.prof_at(self.x0)
        self.zr, self.yr = P[:C.edge + 1, 1], P[:C.edge + 1, 0]
        self.ty, self.tz = P[C.edge:][::-1].T

    def w(self, z):
        return np.interp(z, self.zr, self.yr)

    def ztop(self, y):
        return np.interp(np.abs(y), self.ty, self.tz)

    def pos(self, y, z):
        y, z = np.broadcast_arrays(y, z)
        u = np.clip(np.abs(y) / np.maximum(self.w(z), 1e-6), 0, 1)
        x = self.base + self.s * self.b(z) * np.sqrt(np.maximum(0., 1 - u ** self.p))
        return np.stack((x, y, z), axis=-1)

    def emit(self, matte_below=None):
        start = len(self.C.m.I)
        end = -1 if self.s > 0 else 0
        boundary = np.concatenate((self.C.grids[1][end], self.C.grids[-1][end, -2::-1]))
        # Horizontal rows follow bumper curvature changes at the same height.
        # A radial fan interpolates them at different heights, producing a
        # star-shaped reflection even with analytic normals.
        rim = self.C.grids[1][end]
        levels = _unique(rim[:, 2], self.b.x, [self.peak])
        levels = levels[(levels >= rim[0, 2]) & (levels <= rim[-1, 2])]
        gap = (.055, .14, .30)[self.C.lod]
        zs = _unique(levels, *[np.linspace(a, b, int(np.ceil((b - a) / gap)) + 1)
                               for a, b in zip(levels[:-1], levels[1:])])
        u = np.sin(np.linspace(-np.pi / 2, np.pi / 2, (33, 17, 11)[self.C.lod]))
        widths = np.interp(zs, rim[:, 2], rim[:, 1])
        G = self.pos(widths[:, None] * u, zs[:, None])
        # Cut points on the boundary use the actual swept skin edges, not a
        # second evaluation of the curved surface; this keeps the join closed.
        edge_x = np.interp(zs, rim[:, 2], rim[:, 0])
        G[:, 0, 0] = G[:, -1, 0] = edge_x
        _emit_grid(self.C.m, G, P_PAINT, ONE, _const_hint((self.s, 0, 0)), cos_thr=-.5)
        indices = self.C.m.I[start:]
        # Smooth normals from the same height field as the fascia geometry.
        used = np.unique(indices)
        points = np.asarray(self.C.m.V, np.float64)[used, :3]
        y, z, eps = points[:, 1], points[:, 2], .0002
        dy = (self.pos(y + eps, z)[:, 0] - self.pos(y - eps, z)[:, 0]) / (2 * eps)
        dz = (self.pos(y, z + eps)[:, 0] - self.pos(y, z - eps)[:, 0]) / (2 * eps)
        normals = self.s * np.column_stack((np.ones(len(used)), -dy, -dz))
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        # Round the transition into the bonnet/roof across a uniform narrow
        # strip, rather than only at its differently spaced boundary vertices.
        t = np.clip((self.ztop(y) - z) / .055, 0, 1)
        top_blend = .5 * (1 - t * t * (3 - 2 * t))
        normals *= (1 - top_blend[:, None])
        normals[:, 2] += top_blend
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        for i, normal in zip(used, normals):
            v = self.C.m.V[i]
            self.C.m.V[i] = (*v[:3], *normal, *v[6:])
        self._join_normals(boundary, indices)
        if matte_below is not None:
            # Clip at the actual cladding edge, not the centre of radial cells;
            # otherwise diagonal paint/plastic teeth appear across the bumper.
            self.C.m.I[start:] = self._split_cladding(indices, matte_below)
            indices = self.C.m.I[start:]
        self.indices = indices
        self.C._surfaces()
        self.surface = _Surface(self.C.m, indices, (1, 2), (self.s, 0, 0))

    def _split_cladding(self, indices, height):
        m, result, cache = self.C.m, [], {}
        v = np.asarray(m.V, np.float64)
        for ids in np.asarray(indices, np.int64).reshape(-1, 3):
            vertices = v[ids]
            if (vertices[:, 2] >= height).all():
                result.extend(ids.tolist())
                continue
            for below, part, col in ((True, P_MATTE, MATTE), (False, P_PAINT, ONE)):
                poly, prev = [], vertices[-1]
                inside_prev = prev[2] <= height if below else prev[2] >= height
                for cur in vertices:
                    inside = cur[2] <= height if below else cur[2] >= height
                    if inside != inside_prev:
                        t = (height - prev[2]) / (cur[2] - prev[2])
                        poly.append(prev + t * (cur - prev))
                    if inside:
                        poly.append(cur)
                    prev, inside_prev = cur, inside
                out = []
                for point in poly:
                    normal = point[3:6] / max(np.linalg.norm(point[3:6]), 1e-9)
                    key = (tuple(np.round(point[:3], 6)), tuple(np.round(normal, 4)), part)
                    i = cache.get(key)
                    if i is None:
                        i = cache[key] = m._v(point[:3], normal, col, part)
                    out.append(i)
                for k in range(1, len(out) - 1):
                    result.extend((out[0], out[k], out[k + 1]))
        return result

    def _join_normals(self, boundary, indices):
        """Remove a shading seam between the separately tessellated panels."""
        keys = {tuple(np.round(p, 6)): [] for p in boundary}
        used = np.unique(np.r_[self.C.indices, indices])
        for i in used:
            v = self.C.m.V[i]
            group = keys.get(tuple(np.round(v[:3], 6)))
            if group is not None:
                group.append(int(i))
        for group in keys.values():
            if len(group) < 2:
                continue
            normals = np.unique(np.round([self.C.m.V[i][3:6] for i in group], 6), axis=0)
            normal = normals.sum(axis=0)
            normal /= max(np.linalg.norm(normal), 1e-9)
            for i in group:
                v = self.C.m.V[i]
                self.C.m.V[i] = (*v[:3], *normal, *v[6:])
        # Horizontal tessellation also introduces cut points along skin edges.
        # Interpolate their joined endpoint normals; leaving analytic normals
        # between averaged endpoints makes a scalloped roof/fascia highlight.
        face_ids = np.unique(indices)
        positions = np.asarray(self.C.m.V, np.float64)[face_ids, :3]
        for a, b in zip(boundary[:-1], boundary[1:]):
            edge = b - a
            length2 = float(edge @ edge)
            if length2 < 1e-12:
                continue
            ga, gb = keys[tuple(np.round(a, 6))], keys[tuple(np.round(b, 6))]
            if not ga or not gb:
                continue
            t = (positions - a) @ edge / length2
            distance = np.linalg.norm(positions - a - t[:, None] * edge, axis=1)
            selected = (t > 1e-6) & (t < 1 - 1e-6) & (distance < 2e-6)
            na, nb = np.asarray(self.C.m.V[ga[0]][3:6]), np.asarray(self.C.m.V[gb[0]][3:6])
            for i, u in zip(face_ids[selected], t[selected]):
                normal = (1 - u) * na + u * nb
                normal /= max(np.linalg.norm(normal), 1e-9)
                v = self.C.m.V[i]
                self.C.m.V[i] = (*v[:3], *normal, *v[6:])

    def poly(self, poly, part, col, off, mirror=True, normal_offset=False):
        self.surface.patch(poly, part, col, off, normal_offset=normal_offset)
        if mirror:
            self.surface.patch([(-y, z) for y, z in poly], part, col, off, normal_offset=normal_offset)


def _annulus(m, x, y, z, ri, ro, s, col, part, n=24):
    """Flat annulus used by the procedural reference fleet's wheels."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    rings = [np.column_stack((x + r * np.sin(a), np.full(n + 1, y), z + r * np.cos(a)))
             for r in (ri, ro)]
    _emit_grid(m, np.stack(rings), part, col, _const_hint((0, s, 0)))


def build_explorer(m, lod=0):
    """Append the baked Explorer exterior and return its target length."""
    _explorer(m, lod)
    return EXPLORER_LEN


def build_explorer_lod(level):
    """Return a baked distance LOD without the translucent ground-glow quads."""
    if level not in (1, 2):
        raise ValueError('Explorer LOD level must be 1 or 2')
    return _templates()[level]
