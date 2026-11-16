"""Tyre marks: ribbons laid on the road behind sliding wheels.

CPU side only (numpy ring buffer). The renderer uploads the dirty range of
``buf`` and draws it as alpha-blended triangles with ``shaders/skid.py``.
Each segment is two triangles (6 vertices of x, y, z, u, intensity, birth).
"""
from __future__ import annotations

import math

import numpy as np

from .vehicle_instances import WHEELBASES, WIDTHS

FLOATS = 6                  # x, y, z, u (across the strip 0..1), intensity, birth time
SEGMENT_VERTS = 6


class SkidMarks:
    CAPACITY = 12000        # segments kept; the oldest are overwritten
    MIN_STEP = 0.30         # metres between laid points
    MAX_GAP = 4.0           # a larger jump (teleport / dropped frames) starts a new strip
    ON_LEVEL = 0.14         # slide intensity above which a wheel marks the road
    LIFT = 0.012            # metres above the road surface

    def __init__(self):
        self.buf = np.zeros((self.CAPACITY * SEGMENT_VERTS, FLOATS), np.float32)
        self.head = 0
        self.count = 0
        self.clock = 0.     # seconds; shared with the shader to fade marks with age
        self._tracks = [None] * 4
        self._dirty = None  # (first, last+1) segment indices written since the last upload
        self.generation = 0

    # ------------------------------------------------------------------ state
    def clear(self):
        self.buf[:] = 0.
        self.head = self.count = 0
        self._tracks = [None] * 4
        self._dirty = (0, self.CAPACITY)
        self.generation += 1

    def release(self):
        """The car stopped sliding / was left: finish every open strip."""
        self._tracks = [None] * 4

    def take_dirty(self):
        d, self._dirty = self._dirty, None
        return d

    # ----------------------------------------------------------------- update
    def update(self, car, position, heading, dt):
        """Lay marks for one frame from the car's slide state and (interpolated) pose."""
        self.clock += float(dt)
        kind = car.kind
        wb, width = float(WHEELBASES[kind]), float(WIDTHS[kind])
        half_track, half_tyre = .437 * width, .5 * (.20 + .02 * width)
        c, s = math.cos(heading), math.sin(heading)
        z_ok = bool(getattr(car, 'suspension_ready', False))
        front, rear = float(car.skid_front), float(car.skid_rear)
        # FL FR RL RR; local +x forward, +y left
        layout = ((.5 * wb, half_track, front), (.5 * wb, -half_track, front),
                  (-.5 * wb, half_track, rear), (-.5 * wb, -half_track, rear))
        for i, (ax, ay, level) in enumerate(layout):
            if level < self.ON_LEVEL:
                self._tracks[i] = None
                continue
            x = position[0] + c * ax - s * ay
            y = position[1] + s * ax + c * ay
            z = (float(car.wheel_z[i]) if z_ok else float(position[2])) + self.LIFT
            self._lay(i, np.array([x, y, z]), level, half_tyre)

    def _lay(self, i, p, level, half_w):
        tr = self._tracks[i]
        if tr is None:
            self._tracks[i] = {'p': p, 'L': None, 'R': None, 'level': 0.}
            return
        d = p[:2] - tr['p'][:2]
        dist = math.hypot(d[0], d[1])
        if dist < self.MIN_STEP:
            return
        if dist > self.MAX_GAP:
            self._tracks[i] = {'p': p, 'L': None, 'R': None, 'level': 0.}
            return
        n = np.array([-d[1], d[0], 0.]) / dist * half_w
        new_l, new_r = p + n, p - n
        if tr['L'] is None:                     # first segment of a strip fades in from nothing
            old_l, old_r = tr['p'] + n, tr['p'] - n
        else:                                   # later ones reuse the previous end: no gaps
            old_l, old_r = tr['L'], tr['R']
        b = self.clock
        a0, a1 = tr['level'], level
        v = self.buf[self.head * SEGMENT_VERTS:(self.head + 1) * SEGMENT_VERTS]
        v[0] = (*old_l, 0., a0, b); v[1] = (*old_r, 1., a0, b); v[2] = (*new_r, 1., a1, b)
        v[3] = (*old_l, 0., a0, b); v[4] = (*new_r, 1., a1, b); v[5] = (*new_l, 0., a1, b)
        lo, hi = self._dirty if self._dirty else (self.head, self.head)
        self._dirty = (min(lo, self.head), max(hi, self.head + 1)) if self._dirty else (self.head, self.head + 1)
        if self.head < lo and self._dirty[1] - self._dirty[0] > self.CAPACITY // 2:
            self._dirty = (0, self.CAPACITY)    # ring wrapped: upload everything
        self.head = (self.head + 1) % self.CAPACITY
        self.count = min(self.count + 1, self.CAPACITY)
        tr.update(p=p, L=new_l, R=new_r, level=level)
