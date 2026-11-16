"""Per-lamp optical coordinates for the baked vehicle meshes.

The vehicle shader used to give every lamp material one constant emission, so a
whole headlamp or tail-lamp rendered as a single flat, clipped-white shape.  A
real lamp is a *stack*: a bright LED light-guide / projector core, a dimmer
metallised reflector with facets, a dark housing rim, and a tinted lens over the
lot.  The shader can only build that if each fragment knows where it sits in its
own lamp, so this module bakes that into the (otherwise unused) vertex colour.

Encoding, per vertex, written into the colour slot ``(r, g, b)``:

  emitters  (3 head, 4 tail, 11 LED signature, 14 LED projector, 16/17 indicator)
      r = u      0 inboard ... 1 outboard (mirrored, so both lamps share a layout)
      g = v      0 bottom  ... 1 top
      b = 2 + w  w = 0 at the outer lens plane ... 1 at the back of the housing
  lenses    (22 clear head/reverse cover, 23 red tail cover)
      r = near   proximity to the nearest emitter behind the lens (tight)
      g = wide   the same, as a broad halo
      b = 2 + w  as above

``b > 1.5`` therefore flags a baked vertex.  Procedural boxes (the reference fleet)
keep their plain colours and the shader falls back to its old flat emission for
them.  Nothing here changes geometry, normals, part ids or index order.
"""
from collections import defaultdict

import numpy as np

EMITTER_PARTS = (3, 4, 11, 14, 16, 17)
LENS_PARTS = (22, 23)
# Emitters only merge with their own material group, so two projectors that sit
# 2 cm apart stay two lamps instead of one wide one.
_GROUPS = ((3, 14), (4,), (11,), (16,), (17,))
MERGE_GAP = 0.025          # metres between bounding boxes that still count as one lamp
MIN_EXTENT = 0.02          # avoid dividing by a hair-thin cluster
NEAR_SIGMA, WIDE_SIGMA = 0.030, 0.085


def _weld(positions, tol=1e-3):
    key = np.round(positions / tol).astype(np.int64)
    ids, out = {}, np.empty(len(key), np.int64)
    for i, k in enumerate(map(tuple, key)):
        out[i] = ids.setdefault(k, len(ids))
    return out


def _components(tri, weld, n_vertices):
    """Triangle-connected components -> list of vertex index arrays."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in weld[tri]:
        for x, y in ((a, b), (b, c)):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
    members = defaultdict(set)
    for t in tri:
        members[find(weld[t[0]])].update(int(i) for i in t)
    return [np.fromiter(m, np.int64) for m in members.values()]


def _lamp_frame(pts):
    """(side, x_end) for a cluster: which flank it is on and which end of the car."""
    return (1.0 if pts[:, 1].mean() >= 0.0 else -1.0), (1.0 if pts[:, 0].mean() >= 0.0 else -1.0)


def _merge_boxes(boxes, gap):
    """Union-find over bounding boxes (lo, hi) that overlap once grown by `gap`."""
    n = len(boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j):
                    continue
                (alo, ahi), (blo, bhi) = boxes[i], boxes[j]
                if np.all(alo - gap <= bhi) and np.all(blo - gap <= ahi):
                    parent[find(j)] = find(i)
                    boxes[i] = (np.minimum(alo, blo), np.maximum(ahi, bhi))
                    boxes[j] = boxes[i]
                    changed = True
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def bake_lamp_coords(vertices, indices):
    """Return a copy of ``vertices`` with lamp coordinates written into the colour slot."""
    v = np.array(vertices, np.float32, copy=True)
    idx = np.asarray(indices, np.int64)
    part = v[:, 9].astype(np.int32)
    pos = v[:, :3].astype(np.float64)
    tris = idx[: len(idx) // 3 * 3].reshape(-1, 3)
    weld = _weld(pos)
    emitter_vertices = []                         # (vertex ids, part group) per baked cluster

    for group in _GROUPS:
        sel = np.isin(part[tris[:, 0]], group)
        if not sel.any():
            continue
        comps = _components(tris[sel], weld, len(v))
        # a component is keyed by flank and end so mirrored lamps never merge across the car
        by_frame = defaultdict(list)
        for ids in comps:
            by_frame[_lamp_frame(pos[ids])].append(ids)
        for (side, end), items in by_frame.items():
            boxes = []
            for ids in items:
                p = pos[ids]
                boxes.append((np.array([p[:, 0].min(), (side * p[:, 1]).min(), p[:, 2].min()]),
                              np.array([p[:, 0].max(), (side * p[:, 1]).max(), p[:, 2].max()])))
            for cluster in _merge_boxes(list(boxes), MERGE_GAP):
                ids = np.unique(np.concatenate([items[k] for k in cluster]))
                p = pos[ids]
                yy = side * p[:, 1]
                lo = np.array([p[:, 0].min(), yy.min(), p[:, 2].min()])
                hi = np.array([p[:, 0].max(), yy.max(), p[:, 2].max()])
                ext = np.maximum(hi - lo, MIN_EXTENT)
                u = np.clip((yy - lo[1]) / ext[1], 0.0, 1.0)
                w = (hi[0] - p[:, 0]) / ext[0] if end > 0 else (p[:, 0] - lo[0]) / ext[0]
                vv = np.clip((p[:, 2] - lo[2]) / ext[2], 0.0, 1.0)
                v[ids, 6], v[ids, 7], v[ids, 8] = u, vv, 2.0 + np.clip(w, 0.0, 1.0)
                emitter_vertices.append((ids, group, side, end))

    # ---- lenses: proximity to the emitters that actually sit behind them -------------
    lens = np.flatnonzero(np.isin(part, LENS_PARTS))
    if len(lens):
        sources = {23: [], 22: []}
        for ids, group, side, end in emitter_vertices:
            if 4 in group:
                sources[23].append(ids)            # red cover  <- tail emitters
            elif end > 0 and (3 in group or 11 in group):
                sources[22].append(ids)            # clear head cover <- projectors, signature
        for kind in LENS_PARTS:
            ids = lens[part[lens] == kind]
            if not len(ids):
                continue
            src = np.unique(np.concatenate(sources[kind])) if sources[kind] else np.empty(0, np.int64)
            lp = pos[ids]
            near = np.zeros(len(ids))
            if len(src):
                sp = pos[src]
                for a in range(0, len(ids), 512):
                    # lateral/vertical distance only: depth does not matter behind a thin cover
                    same_end = np.sign(lp[a:a + 512, 0:1]) == np.sign(sp[None, :, 0])
                    d = np.hypot(lp[a:a + 512, None, 1] - sp[None, :, 1], lp[a:a + 512, None, 2] - sp[None, :, 2])
                    d = np.where(same_end, d, 1e3)
                    near[a:a + 512] = d.min(axis=1)
            sigma_n, sigma_w = NEAR_SIGMA, WIDE_SIGMA
            v[ids, 6] = np.exp(-(near / sigma_n) ** 2)
            v[ids, 7] = np.exp(-(near / sigma_w) ** 2)
            # depth into the cover shell along the car axis, per flank/end cluster
            comps = _components(tris[np.isin(part[tris[:, 0]], (kind,))], weld, len(v))
            v[ids, 8] = 2.0
            for cids in comps:
                p = pos[cids]
                end = 1.0 if p[:, 0].mean() >= 0 else -1.0
                ext = max(np.ptp(p[:, 0]), MIN_EXTENT)
                w = (p[:, 0].max() - p[:, 0]) / ext if end > 0 else (p[:, 0] - p[:, 0].min()) / ext
                v[cids, 8] = 2.0 + np.clip(w, 0.0, 1.0)
    return v
