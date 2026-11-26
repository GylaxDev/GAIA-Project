"""Conforming tile-lattice water triangulation (CPU worker only).

Interior cells share indexed vertices; only shoreline/hole cells go through
GEOS clipping and polygon triangulation. Adjacent tiles use the same number of
edge intervals, including their boundary vertices. Refining arbitrary polygon
triangles independently would create displaced T-junctions at tile joins.
"""
from __future__ import annotations

import numpy as np
import shapely

WATER_GRID = 512
WATER_BOUNDS_PAD = 2.0  # conservative world metres for the shared wave spectrum


def _conform_grid_edges(points, triangles, xs, ys):
    """Insert clipped edge samples into adjacent full-cell triangles as well.

    A shore can follow a grid edge for part of a cell and then turn away. The
    clipped neighbour retains that turning point, but a full cell has only its
    corner vertices. Welding alone does not fix the resulting T-junction.
    Only triangles touching these extra edge samples need to be split.
    """
    tolerance = 1e-6
    origin = np.array([xs[0], ys[0]])
    step = np.array([xs[1] - xs[0], ys[1] - ys[0]])
    grid = (points - origin) / step
    integer = np.rint(grid).astype(np.int64)
    on_grid = np.abs(grid - integer) * step <= tolerance
    corners = on_grid.all(axis=1)
    cuts = {}
    for axis in (0, 1):
        ids = np.flatnonzero(on_grid[:, axis] & ~on_grid[:, 1 - axis])
        for index in ids:
            key = (axis, int(integer[index, axis]), int(np.floor(grid[index, 1 - axis])))
            cuts.setdefault(key, []).append(int(index))
    if not cuts:
        return points, triangles
    corner_ids = np.full((len(ys), len(xs)), -1, np.int64)
    corner_ids[integer[corners, 1], integer[corners, 0]] = np.flatnonzero(corners)
    touched = np.zeros(len(points), bool)
    for (axis, line, cell), ids in cuts.items():
        ids.sort(key=lambda index: points[index, 1 - axis])
        touched[ids] = True
        ends = (corner_ids[cell:cell + 2, line] if axis == 0 else corner_ids[line, cell:cell + 2])
        touched[ends[ends >= 0]] = True
    candidates = np.flatnonzero(touched[triangles].any(axis=1))
    keep = np.ones(len(triangles), bool)
    centers, split = [], []
    for index in candidates:
        triangle = triangles[index]
        ring, changed = [], False
        for a, b in zip(triangle, np.roll(triangle, -1)):
            ring.append(int(a))
            pa, pb = points[a], points[b]
            for axis in (0, 1):
                if not on_grid[a, axis] or abs(pa[axis] - pb[axis]) > tolerance:
                    continue
                key = (axis, int(integer[a, axis]), int(np.floor((grid[a, 1 - axis] + grid[b, 1 - axis]) * .5)))
                ids = cuts.get(key, ())
                low, high = sorted((pa[1 - axis], pb[1 - axis]))
                inside = [i for i in ids if low + 1e-7 < points[i, 1 - axis] < high - 1e-7]
                if pb[1 - axis] < pa[1 - axis]: inside.reverse()
                ring.extend(inside)
                changed |= bool(inside)
                break
        if not changed:
            continue
        keep[index] = False
        center = len(points) + len(centers)
        centers.append(points[triangle].mean(axis=0))
        # A center fan preserves all collinear boundary vertices; a corner fan
        # would discard them again through degenerate triangles on that edge.
        split.extend((center, a, b) for a, b in zip(ring, ring[1:] + ring[:1]))
    if not split:
        return points, triangles
    return np.vstack((points, centers)), np.vstack((triangles[keep], np.asarray(split, np.uint32)))


def triangulate_water(poly, bbox, triangulate, divisions=WATER_GRID):
    if poly.is_empty:
        return np.empty((0, 2), np.float64), np.empty((0, 3), np.uint32), 0.
    if bbox is None:
        # Standalone fixtures use a metric lattice. Production always supplies
        # tile bounds; never restart a grid at a water polygon's bounding box.
        lo = np.floor(np.asarray(poly.bounds[:2]) / 3.) * 3.
        hi = np.ceil(np.asarray(poly.bounds[2:]) / 3.) * 3.
        xs = np.linspace(lo[0], hi[0], max(2, int(round((hi[0] - lo[0]) / 3.)) + 1))
        ys = np.linspace(lo[1], hi[1], max(2, int(round((hi[1] - lo[1]) / 3.)) + 1))
    else:
        x0, y0, x1, y1 = map(float, bbox)
        xs, ys = np.linspace(x0, x1, divisions + 1), np.linspace(y0, y1, divisions + 1)
    spacing = max(float(np.diff(xs).max()), float(np.diff(ys).max()))
    # Restrict the transient grid to this connected component's bounds.
    x0, y0, x1, y1 = poly.bounds
    ix0 = max(0, int(np.searchsorted(xs, x0, side='right')) - 1)
    iy0 = max(0, int(np.searchsorted(ys, y0, side='right')) - 1)
    ix1 = min(len(xs) - 1, int(np.searchsorted(xs, x1, side='left')))
    iy1 = min(len(ys) - 1, int(np.searchsorted(ys, y1, side='left')))
    xs, ys = xs[ix0:ix1 + 1], ys[iy0:iy1 + 1]
    if len(xs) < 2 or len(ys) < 2:
        return np.empty((0, 2), np.float64), np.empty((0, 3), np.uint32), spacing
    gx, gy = np.meshgrid(xs, ys)
    points = [np.column_stack((gx.ravel(), gy.ravel()))]
    indices, count = [], gx.size
    stride = len(xs)
    shapely.prepare(poly)
    for row0 in range(0, len(ys) - 1, 32):
        row1 = min(row0 + 32, len(ys) - 1)
        cx, cy = np.meshgrid(np.arange(len(xs) - 1), np.arange(row0, row1))
        cx, cy = cx.ravel(), cy.ravel()
        cells = shapely.box(xs[cx], ys[cy], xs[cx + 1], ys[cy + 1])
        full = shapely.covers(poly, cells)
        ids = cy[full] * stride + cx[full]
        if len(ids):
            indices.append(np.stack((ids, ids + 1, ids + stride + 1,
                                     ids, ids + stride + 1, ids + stride), axis=1).reshape(-1, 3))
        edge = ~full & shapely.intersects(poly, cells)
        for cut in shapely.intersection(poly, cells[edge]):
            parts = [cut] if cut.geom_type == 'Polygon' else getattr(cut, 'geoms', ())
            for p in parts:
                if p.geom_type != 'Polygon' or p.area < 1e-8:
                    continue
                q, t = triangulate(p)
                if not len(t):
                    continue
                points.append(q)
                indices.append(t + count)
                count += len(q)
    if not indices:
        return np.empty((0, 2), np.float64), np.empty((0, 3), np.uint32), spacing
    pts, tri = np.vstack(points), np.vstack(indices)
    # Weld shoreline cells to their interior neighbours, remove unused lattice
    # points, and make every shared rest position bit-identical in the VBO.
    used = np.unique(tri)
    keys = np.round(pts[used], 6)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    remap = np.zeros(len(pts), np.uint32)
    remap[used] = inverse
    pts, tri = pts[used[first]], remap[tri]
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri = tri[np.abs(area2) > 1e-8]
    pts, tri = _conform_grid_edges(pts, tri, xs, ys)
    return pts, np.ascontiguousarray(tri, np.uint32), spacing
