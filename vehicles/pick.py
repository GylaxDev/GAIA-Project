"""Ray picking for the focus key: buildings first, then moving vehicles."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .driving import DRIVABLE_KINDS
from .vehicle_instances import LENGTHS, WIDTHS, HEIGHTS, normalize_instances

VEHICLE_HIT_RADIUS = 5.0
MAX_FOCUS_DIST = 1600.0


def ray_boxes(o, d, boxes):
    """Slab test against (N, 6) AABBs; returns the near hit t per box (inf = miss)."""
    inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
    t0 = (boxes[:, 0:3] - o) * inv
    t1 = (boxes[:, 3:6] - o) * inv
    tmin = np.minimum(t0, t1).max(axis=1)
    tmax = np.maximum(t0, t1).min(axis=1)
    hit = tmax >= np.maximum(tmin, 0.0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


def _vehicle_hit(o, d, traffic):
    """Nearest vehicle whose hull is within VEHICLE_HIT_RADIUS of the ray."""
    rows = [t for t in traffic if t is not None and len(t)] if traffic else []
    if not rows:
        return None, None
    r = np.concatenate(rows).astype(np.float64)
    p = r[:, 12:15]
    p[:, 2] += 0.9
    rel = p - o
    t = rel @ d
    closest = o + t[:, None] * d
    gap = np.linalg.norm(p - closest, axis=1)
    ok = (t > 0.0) & (t < MAX_FOCUS_DIST) & (gap < VEHICLE_HIT_RADIUS)
    if not ok.any():
        return None, None
    t = np.where(ok, t, np.inf)
    i = int(np.argmin(t))
    return float(t[i]), p[i]


def pick_focus(cam, chunks, traffic, sx, sy, w, h):
    """Return (anchor, radius, label) for the thing under the cursor, or None."""
    o, d = cam.ray(sx, sy, w, h)
    best_t, best = np.inf, None

    for c in chunks:
        table = getattr(c, "cell_b", None)
        if table is None:
            continue
        boxes = table[2]
        if boxes is None or len(boxes) == 0:
            continue
        t = ray_boxes(o, d, boxes)
        i = int(np.argmin(t))
        ti = float(t[i])
        if not np.isfinite(ti) or ti > MAX_FOCUS_DIST or ti >= best_t:
            continue
        b = boxes[i]
        height = float(b[5] - b[2])
        best_t = ti
        best = (np.array([(b[0] + b[3]) * 0.5, (b[1] + b[4]) * 0.5,
                          (b[2] + b[5]) * 0.5]),
                float(np.clip(height * 0.9 + 12.0, 16.0, 150.0)), "building")

    tv, pv = _vehicle_hit(o, d, traffic)
    if tv is not None and tv < best_t:
        best = (pv, 15.0, "vehicle")
    return best


@dataclass
class DrivePick:
    chunk: object
    kind: int
    index: int
    row: np.ndarray
    car: object = None


def _drive_candidates(chunks, cars, origin=None, limit=None):
    """Alive drivable cars; `origin`/`limit` drop far rows before the per-car tests."""
    for car in cars:
        yield DrivePick(None, car.kind, -1, car.instance(), car)
    reach = None if origin is None else float(limit) ** 2
    for chunk in chunks:
        for kind, rows in enumerate(chunk.parked_cpu or ()):
            if kind not in DRIVABLE_KINDS or rows is None or not len(rows):
                continue
            a = normalize_instances(rows)
            alive = a[:, 23] > .5
            if reach is not None:
                d = a[:, 12:14] - np.asarray(origin[:2])
                alive &= np.sum(d * d, axis=1) <= reach
            for i in np.flatnonzero(alive):
                yield DrivePick(chunk, kind, int(i), a[i])


def pick_drive_vehicle(cam, chunks, cars, sx, sy, w, h):
    """Pick the actual oriented car body, including parked/player-owned cars."""
    o, d = cam.ray(sx, sy, w, h)
    best, best_t = None, 600.
    # A hit at t <= 600 comes from a body whose centre is within a few metres of
    # that, so anything further from the ray origin can never be selected.
    for candidate in _drive_candidates(chunks, cars, o, 610.):
        row, kind = candidate.row, candidate.kind
        basis = np.column_stack((row[:3], row[4:7], row[8:11])).astype(np.float64)
        if abs(float(np.linalg.det(basis))) < 1e-8:
            continue
        local = np.linalg.solve(basis, np.column_stack((o - row[12:15], d)))
        box = np.array([[-LENGTHS[kind] * .5, -WIDTHS[kind] * .5, .03,
                         LENGTHS[kind] * .5, WIDTHS[kind] * .5, HEIGHTS[kind]]])
        t = float(ray_boxes(local[:, 0], local[:, 1], box)[0])
        if t < best_t:
            best, best_t = candidate, t
    return best


def nearest_drive_vehicle(chunks, cars, point, max_distance=100.):
    best, best_distance = None, max_distance
    for candidate in _drive_candidates(chunks, cars, point, max_distance):
        distance = float(np.linalg.norm(candidate.row[12:14] - point[:2]))
        if distance < best_distance:
            best, best_distance = candidate, distance
    return best
