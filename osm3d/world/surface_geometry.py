"""Worker-only tessellation on the same diagonal facets as the terrain mesh.

Corner height samples alone cannot keep a triangle above intervening ridges.
Intersecting the two triangulations preserves the footprint and puts each new
face on one terrain plane. No temporary facet index survives the mesh build.
"""
from __future__ import annotations

import numpy as np
import shapely


def refine_triangles(points, triangles, max_edge, passes=12):
    """Bound long slab edges without multiplying already-small triangles."""
    points = np.asarray(points, np.float64)
    triangles = np.asarray(triangles, np.int64)
    for _ in range(passes):
        if not len(triangles): break
        p = points[triangles]
        split = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2).max(axis=1) > max_edge
        if not split.any(): break
        a, b, c = triangles[split].T
        edges = np.sort(np.stack([a, b, b, c, c, a], axis=1).reshape(-1, 2), axis=1)
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        ab, bc, ca = (inverse.reshape(-1, 3) + len(points)).T
        points = np.vstack([points, points[unique].mean(axis=1)])
        children = np.stack([a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca], axis=1).reshape(-1, 3)
        triangles = np.vstack([triangles[~split], children])
    return points, triangles


def vertex_normals(points, triangles):
    p = np.asarray(points, np.float64)[triangles]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out = np.zeros_like(points, dtype=np.float64)
    for j in range(3): np.add.at(out, triangles[:, j], normals)
    lengths = np.linalg.norm(out, axis=1)
    out[lengths < 1e-12, 2] = 1.
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def terrain_path(points, hf):
    """Split a polyline at grid edges and diagonals; retain exact endpoints."""
    points = np.asarray(points, np.float64)
    if hf is None or not hasattr(hf, 'sx') or len(points) < 2:
        return points
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        uv0 = np.array([(a[0] - hf.x0) * hf.sx, (a[1] - hf.y0) * hf.sy])
        uv1 = np.array([(b[0] - hf.x0) * hf.sx, (b[1] - hf.y0) * hf.sy])
        cuts = []
        for start, end, lo, hi in ((uv0[0], uv1[0], 0, hf.w - 1),
                                   (uv0[1], uv1[1], 0, hf.h - 1),
                                   (uv0[0] - uv0[1], uv1[0] - uv1[1], 1 - hf.h, hf.w - 1)):
            if abs(end - start) < 1e-12: continue
            grid = np.arange(max(lo, int(np.ceil(min(start, end)))),
                             min(hi, int(np.floor(max(start, end)))) + 1)
            t = (grid - start) / (end - start)
            cuts.extend(t[(t > 1e-9) & (t < 1. - 1e-9)].tolist())
        if cuts:
            t = np.unique(np.round(cuts, 12))
            out.append(a + (b - a) * t[:, None])
        out.append(b[None])
    return np.vstack(out)


class TerrainDraper:
    """One reusable terrain-facet index for a tile mesh build."""
    def __init__(self, hf):
        self.hf = hf
        self.tree = None
        self.bounds = None
        if hf is not None and getattr(hf, 'w', 0) >= 2 and getattr(hf, 'h', 0) >= 2:
            self._index((hf.x0, hf.y0, hf.x1, hf.y1))

    def _index(self, bounds):
        hf = self.hf
        # Include the clamped outside strips: buffered roads overhang tile edges.
        x0, y0 = min(bounds[0], hf.x0) - 64., min(bounds[1], hf.y0) - 64.
        x1, y1 = max(bounds[2], hf.x1) + 64., max(bounds[3], hf.y1) + 64.
        xs = np.r_[x0, np.linspace(hf.x0, hf.x1, hf.w), x1]
        ys = np.r_[y0, np.linspace(hf.y0, hf.y1, hf.h), y1]
        gx, gy = np.meshgrid(xs, ys)
        q = np.stack([gx, gy], axis=-1)
        a, b, c, d = q[:-1, :-1], q[:-1, 1:], q[1:, 1:], q[1:, :-1]
        faces = np.stack([a, b, c, a, c, d], axis=2).reshape(-1, 3, 2)
        self.facets = shapely.polygons(faces)
        self.tree = shapely.STRtree(self.facets)
        self.bounds = (x0, y0, x1, y1)

    def conform(self, points, triangles, max_edge=None):
        points, triangles = np.asarray(points, np.float64), np.asarray(triangles, np.int64)
        if self.tree is not None and len(triangles):
            bounds = (*points.min(axis=0), *points.max(axis=0))
            if (bounds[0] < self.bounds[0] or bounds[1] < self.bounds[1]
                    or bounds[2] > self.bounds[2] or bounds[3] > self.bounds[3]):
                self._index(bounds)
            source = shapely.polygons(points[triangles])
            rows, columns = self.tree.query(source, predicate='intersects')
            pieces = shapely.intersection(source[rows], self.facets[columns])
            pieces = pieces[shapely.area(pieces) > 1e-10]
            # Triangle intersections are convex: a fan is exact, even for slivers.
            coords, owners = shapely.get_coordinates(shapely.get_exterior_ring(pieces), return_index=True)
            counts = np.bincount(owners, minlength=len(pieces))
            starts = np.cumsum(counts) - counts
            indices = np.arange(len(coords))
            interior = (indices - starts[owners] >= 1) & (indices - starts[owners] < counts[owners] - 2)
            fans = np.stack([starts[owners[interior]], indices[interior], indices[interior] + 1], axis=1)
            points, inverse = np.unique(np.round(coords, 7), axis=0, return_inverse=True)
            triangles = inverse[fans]
            p = points[triangles]
            cross = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                     - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0]))
            triangles = np.where((cross < 0.)[:, None], triangles[:, [0, 2, 1]], triangles)
            triangles = triangles[np.abs(cross) > 1e-10]
        if max_edge is not None:
            points, triangles = refine_triangles(points, triangles, max_edge)
        return points, triangles
