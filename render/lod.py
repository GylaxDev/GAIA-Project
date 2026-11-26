"""Compact mesh staging and object-level visibility, independent of OpenGL."""
from __future__ import annotations

import zlib
import numpy as np


def align_neon(building, neon):
    """Attach cached/generated crown strips to the actual extruded facade tops.

    Terrain datum, overlapping outlines and procedural setbacks can all differ
    from OSM's original height/footprint. Use the rendered walls as the authority.
    """
    if building is None or neon is None or not len(neon.vertices): return
    import shapely
    from shapely.strtree import STRtree
    v = building.vertices
    tri = building.indices.reshape(-1, 3)
    tri = tri[(np.abs(v[tri[:, 0], 16]) < .1) & (np.abs(v[tri[:, 0], 5]) < .1)]
    if not len(tri): return
    pts = v[tri, :3]
    segments, heights, normals, ranks = [], [], [], []
    for a, b, other in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        good = (np.abs(pts[:, a, 2] - pts[:, b, 2]) < .02) & (pts[:, a, 2] > pts[:, other, 2] + .1)
        segments.append(pts[good][:, [a, b], :2])
        heights.append(pts[good, a, 2]); normals.append(v[tri[good, 0], 3:5])
        ranks.append(np.floor(v[tri[good, 0], 10] + .001))
    seg, top, nrm = np.concatenate(segments), np.concatenate(heights), np.concatenate(normals)
    if not len(seg): return
    rank = np.concatenate(ranks)
    edge = seg[:, 1] - seg[:, 0]
    face = np.column_stack((edge[:, 1], -edge[:, 0]))
    face /= np.maximum(np.linalg.norm(face, axis=1, keepdims=True), 1e-8)
    face *= np.where(np.sum(face*nrm, axis=1) < 0, -1, 1)[:, None]
    sv = neon.vertices
    it = neon.indices.reshape(-1, 3)
    crown = (np.abs(sv[it[:, 0], 16] - 9) < .1) & (sv[it[:, 0], 11] == 5)
    sel = np.flatnonzero(crown)
    if not len(sel): return
    # Both triangles of a strip have the same XY endpoints. Match their shared
    # midpoint once, never each vertex: independent nearest vertices could snap
    # to different podium/tower heights and stretch a ribbon up an entire facade.
    p = sv[it[sel], :3]
    mid = (p[:, :, :2].min(axis=1) + p[:, :, :2].max(axis=1)) * .5
    keys = np.column_stack((mid, sv[it[sel, 0], 10:13], sv[it[sel, 0], 3:5]))
    _, first, group = np.unique(np.round(keys, 3), axis=0, return_index=True, return_inverse=True)
    tree = STRtree(shapely.linestrings(seg))
    q, hit = tree.query(shapely.points(mid[first]), predicate='dwithin', distance=8.0)
    # Reject perpendicular neighbouring walls and strips that overhang the wall.
    src = p[first[q], :, :2]
    ab = edge[hit]
    length = np.maximum(np.linalg.norm(ab, axis=1), 1e-8)
    along = np.sum((src - seg[hit, 0, None])*ab[:, None], axis=2) / length[:, None]
    perpendicular = np.abs(np.sum((src-seg[hit, 0, None])*face[hit, None], axis=2))
    lo, hi = along.min(axis=1), along.max(axis=1)
    # Buffered footprint corners can extend a few centimetres past an edge.
    # A percentage tolerance on a long facade admitted wholly unsupported short
    # strips metres beyond its endpoint. Bound the allowance in metres and
    # require real overlap; the final endpoints are clamped to the finite wall.
    supported = (lo >= -.25) & (hi <= length + .25) & (perpendicular.max(axis=1) < 8)
    supported &= np.minimum(hi, length) - np.maximum(lo, 0.0) >= .02
    supported &= np.ptp(perpendicular, axis=1) < .15
    q, hit = q[supported], hit[supported]
    dist = np.linalg.norm(mid[first[q]] - seg[hit].mean(axis=1), axis=1)
    score = np.abs(top[hit]-sv[it[sel[first[q]], 0], 12]) + dist*.05
    order = np.lexsort((hit, score, q)); q, hit = q[order], hit[order]
    unique = np.r_[True, q[1:] != q[:-1]] if len(q) else np.empty(0, bool)
    chosen = np.full(len(first), -1, np.int64); chosen[q[unique]] = hit[unique]
    keep = np.ones(len(it), bool); keep[sel] = chosen[group] >= 0
    for g in np.flatnonzero(chosen >= 0):
        h = chosen[g]; ids = np.unique(it[sel[group == g]])
        along = np.sum((sv[ids, :2]-seg[h, 0])*edge[h], axis=1) / max(float(edge[h] @ edge[h]), 1e-8)
        sv[ids, :2] = seg[h, 0] + np.clip(along, 0.0, 1.0)[:, None]*edge[h] + .06*face[h]
        sv[ids, 2] += top[h]-sv[ids, 12]
        sv[ids, 12] = top[h]
        sv[ids, 9] = min(1.0, rank[h]/15.0)
    neon.indices = it[keep].reshape(-1).copy()

# Positions, UVs and procedural seeds stay float32. Only normal/color/AO are
# quantized. This cuts the city vertex stream from 68 to 48 bytes per vertex.
CITY_VERTEX = np.dtype([("position", "<f4", 3), ("normal", "i1", 4),
                        ("color", "u1", 4), ("style", "<f4", 3),
                        ("uv", "<f4", 2), ("emission", "<f4"), ("part", "<f4")])
TERRAIN_VERTEX = np.dtype([("position", "<f4", 3), ("normal", "i1", 4),
                           ("color", "<f4", 3), ("ao", "u1", 4),
                           ("style", "<f4", 3), ("uv", "<f4", 2)])
WATER_VERTEX = np.dtype([("position", "<f4", 3), ("style", "<f4", 3)])


def pack_vertices(v, terrain=False, *, water=False):
    if water:
        # Displaced water evaluates its own normal/material. Dense lattices do
        # not need the unused city colour, UV, emission and part attributes.
        out = np.empty(len(v), WATER_VERTEX)
        out['position'], out['style'] = v[:, :3], v[:, 10:13]
        return out
    out = np.zeros(len(v), TERRAIN_VERTEX if terrain else CITY_VERTEX)
    out["position"] = v[:, :3]
    out["normal"][:, :3] = np.rint(np.clip(v[:, 3:6], -1, 1) * 127).astype(np.int8)
    if terrain:
        # These are NOT RGB: red encodes surface IDs up to 6, green road width
        # in metres. Quantizing/clamping them turns roads and seabeds into grass.
        out['color'] = v[:, 6:9]
        out['ao'][:, 0] = np.rint(np.clip(v[:, 9], 0, 1) * 255).astype(np.uint8)
    else:
        out["color"][:, :3] = np.rint(np.clip(v[:, 6:9], 0, 1) * 255).astype(np.uint8)
        out["color"][:, 3] = np.rint(np.clip(v[:, 9], 0, 1) * 255).astype(np.uint8)
    out["style"] = v[:, 10:13]
    out["uv"] = v[:, 13:15]
    if not terrain:
        out["emission"] = v[:, 15]
        out["part"] = v[:, 16]
    return out


NEON_SAMPLE_FLOATS = 6           # xyz, rgb
NEON_SAMPLES_PER_MESH = 24


def neon_light_samples(mesh):
    """Small mesh-local source table; index scanning belongs in the worker.

    Only ordinary signs contribute point samples. LED advertising screens (part 8,
    style.y 4) are deliberately excluded: each one used to become a one-sided area
    emitter, and every ground/building fragment paid for a 48-light screen loop.
    Dense billboard blocks made the night pass lag. The screens still render as
    emissive surfaces; they just no longer cast light onto the street or facades.
    """
    v = mesh.vertices
    if not len(v) or not len(mesh.indices):
        return np.empty((0, NEON_SAMPLE_FLOATS), np.float32)
    screen = (v[:, 16] == 8.0) & (v[:, 11] == 4.0)
    ids = np.unique(mesh.indices)
    ids = ids[~screen[ids]]
    if not len(ids):
        return np.empty((0, NEON_SAMPLE_FLOATS), np.float32)
    ids = ids[::max(1, len(ids) // NEON_SAMPLES_PER_MESH)][:NEON_SAMPLES_PER_MESH]
    samples = np.zeros((len(ids), NEON_SAMPLE_FLOATS), np.float32)
    samples[:, :3] = v[ids, :3]
    samples[:, 3:6] = np.clip(v[ids, 6:9] * 2.0, 0.0, 2.0)
    return samples


def prepare_mesh(mesh, kind):
    """Worker-side staging, also applied to old disk caches without re-downloading."""
    if mesh is None or getattr(mesh, "gpu_vertices", None) is not None:
        return mesh
    from ..world.mesh_extruder import build_cells
    v, idx = mesh.vertices, mesh.indices
    if kind == "props" and len(idx):
        tri = idx.reshape(-1, 3)
        p = v[tri, :3]
        span = np.ptp(p, axis=1)
        part = v[tri[:, 0], 16]
        dark = v[tri[:, 0], 6:9].max(axis=1) < .20
        small = (part > 10.5) & dark & (span[:, :2].max(axis=1) < 1.1)
        area2 = np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1)
        wire = (part == 7) & (span[:, :2].max(axis=1) > 15) & (np.sum(span*span, axis=1) > 120*np.maximum(area2, 1e-6))
        idx = tri[~(small | wire)].reshape(-1)
        used, remap = np.unique(idx, return_inverse=True)
        v = np.ascontiguousarray(v[used]); idx = remap.astype(np.uint32)
        mesh.vertices = v
    if kind in ("building", "terrain", "props", "neon", "water"):
        idx, mesh.cells = build_cells(v, idx, grid=None, min_tris=600)
        if kind == 'water' and mesh.cells is not None:
            from ..world.water_mesh import WATER_BOUNDS_PAD
            mesh.cells[:, 2:5] -= WATER_BOUNDS_PAD
            mesh.cells[:, 5:8] += WATER_BOUNDS_PAD
    mesh.indices = idx
    mesh.bounds = (np.concatenate((v[:, :3].min(axis=0), v[:, :3].max(axis=0)))
                   if len(v) else np.zeros(6, np.float32))
    mesh.gpu_vertices = pack_vertices(v, terrain=kind == 'terrain', water=kind == 'water')
    mesh.lod_indices = mesh.lod_cells = None
    if kind == "building" and len(idx):
        tri = idx.reshape(-1, 3)
        # Small rooftop equipment is invisible from distant streets. Keep the
        # shell, parapets, real roof shape, bridges and obstruction beacons.
        keep = np.any(np.abs(v[tri, 16] - 9.0) > .1, axis=1)
        if not keep.all():
            mesh.lod_indices = np.ascontiguousarray(tri[keep].reshape(-1), np.uint32)
            if mesh.cells is not None:
                # Both LODs use identical cell bounds, so a near/far boundary
                # cannot leave holes when rooftop equipment changes the AABB.
                cells = mesh.cells.copy()
                start = cells[:, 0].astype(np.int64)//3
                end = start + cells[:, 1].astype(np.int64)//3
                prefix = np.r_[0, np.cumsum(keep)]
                cells[:, 0] = prefix[start]*3
                cells[:, 1] = (prefix[end]-prefix[start])*3
                mesh.lod_cells = cells[cells[:, 1] > 0]
    mesh.packed_edges = (np.frombuffer(zlib.compress(np.ascontiguousarray(mesh.edges, np.float32).tobytes(), 1), np.uint8)
                         if mesh.edges is not None and mesh.edges.size else None)
    if kind == 'neon': mesh.neon_samples = neon_light_samples(mesh)
    return mesh


def sphere_mask(positions, radii, eye, max_distance, planes=None):
    delta = positions - np.asarray(eye, dtype=np.result_type(positions.dtype, np.float32))
    mask = np.einsum("ij,ij->i", delta, delta) <= (max_distance + radii) ** 2
    if planes is not None:
        # Cull by distance first: only a small fraction of a large tile's forest
        # needs the six plane tests. Avoid waking BLAS for a narrow matrix.
        selected = np.flatnonzero(mask)
        pxyz = positions[selected]
        radius = radii[selected] if np.ndim(radii) else radii
        visible = np.ones(len(selected), bool)
        for p in planes:
            visible &= (pxyz[:, 0]*p[0] + pxyz[:, 1]*p[1] + pxyz[:, 2]*p[2] + p[3]) >= -radius
        mask[selected] = visible
    return mask
