"""Vectorised road-path sampling for the traffic simulator.

Moving every vehicle in a Python loop each frame (two ``np.searchsorted`` calls, a
few tiny array operations and one ``np.empty`` row per vehicle) costs several
milliseconds for ~1000 cars, more than the GPU needs to draw them.

Here every path of a tile is concatenated into one polyline table whose arc
lengths are made globally increasing (each path is shifted by a running base), so a
single ``np.searchsorted`` locates the segment of *all* vehicles at once.  The
maths per vehicle is identical to ``sample_path_position`` / ``sample_path_z`` in
``data_fetcher`` (agreement is ~1e-16 on random roads).

This module only needs numpy, so it can be tested without the GL / shapely stack.
"""
from __future__ import annotations

import numpy as np
from .traffic_dynamics import speed_limit, idm_acceleration, integrate_idm
from ..vehicles.vehicle_instances import VEHICLE_STRIDE, WHEEL_RADII, LAMPS, BRAKE


class TileRoads:
    """Concatenated polylines of one tile's paths, for batched sampling."""

    def __init__(self, paths):
        n = len(paths)
        self.n = n
        self.total = np.ones(n, np.float64)
        self.base = np.zeros(n, np.float64)
        self.start = np.zeros(n, np.int64)
        self.last_seg = np.zeros(n, np.int64)       # highest valid segment index (global)
        pts, loc, glob, zs = [], [], [], []
        off, base = 0, 0.0
        for i, p in enumerate(paths):
            pp = np.asarray(p.points, np.float64)
            aa = np.asarray(p.arc, np.float64)
            m = len(pp)
            if m < 2 or len(aa) != m:
                # unusable polyline: leave a harmless 2-point stub (never spawned on)
                pp = np.zeros((2, 2)); aa = np.array([0.0, 1.0]); m = 2
            z = getattr(p, "z", None)
            zz = (np.zeros(m) if z is None or len(z) != m
                  else np.asarray(z, np.float64))
            tot = max(float(p.total), 1e-6)
            self.total[i] = tot
            self.base[i] = base
            self.start[i] = off
            self.last_seg[i] = off + m - 2
            pts.append(pp); loc.append(aa); glob.append(aa + base); zs.append(zz)
            off += m
            base += max(float(aa[-1]), tot) + 16.0      # keeps the global arcs strictly increasing
        if n:
            self.pts = np.vstack(pts)
            self.loc = np.concatenate(loc)
            self.glob = np.concatenate(glob)
            self.z = np.concatenate(zs)
        else:
            self.pts = np.zeros((2, 2)); self.loc = np.zeros(2)
            self.glob = np.zeros(2); self.z = np.zeros(2)
        self.lane = np.array([min(1.75, max(.75, p.width*.23)) for p in paths])
        self.oneway = np.array([p.oneway for p in paths], bool)
        self.limit = np.array([speed_limit(p.highway, getattr(p, 'speed_limit', 0.)) for p in paths])
        self.links = np.full((n, 2), -1, np.int64)
        self.link_forward = np.ones((n, 2), bool)
        self.signal = np.zeros((n, 2), bool)
        # Connect road endpoints once, rather than wrapping each car independently.
        entries = {}
        for i, p in enumerate(paths):
            if p.total <= 20 or len(p.points) < 2: continue
            for forward, k in ((True, 0), (False, -1)):
                if not forward and p.oneway: continue
                xy = self.pts[self.start[i] if forward else self.last_seg[i]+1]
                bucket = tuple(np.floor(xy / 8).astype(int))
                entries.setdefault(bucket, []).append((i, forward, xy))
        for i, p in enumerate(paths):
            if len(p.points) < 2: continue
            for direction in (0, 1):
                end = self.last_seg[i]+1 if direction else self.start[i]
                xy, z = self.pts[end], self.z[end]
                tangent = (self.pts[end] - self.pts[end-1]) if direction else (self.pts[end] - self.pts[end+1])
                tangent /= max(np.linalg.norm(tangent), 1e-6)
                bx, by = np.floor(xy/8).astype(int)
                candidates = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for j, forward, q in entries.get((bx+dx, by+dy), ()):
                            if j == i or np.linalg.norm(q-xy) > 5: continue
                            start = self.start[j] if forward else self.last_seg[j]+1
                            if abs(self.z[start]-z) > 2: continue
                            d = self.pts[start+1] - q if forward else self.pts[start-1] - q
                            d /= max(np.linalg.norm(d), 1e-6)
                            candidates.append((float(d @ tangent), j, forward))
                if candidates:
                    _, j, forward = max(candidates)
                    self.links[i, direction] = j; self.link_forward[i, direction] = forward
                    self.signal[i, direction] = len(candidates) >= 2 and p.highway not in ('motorway', 'trunk')

    def sample(self, pidx, arc):
        """Position, heading and surface height for vehicles at ``arc`` on paths ``pidx``.

        Returns ``(x, y, heading, z)`` float64 arrays, matching
        ``sample_path_position`` + ``sample_path_z`` for each vehicle.
        """
        a = np.mod(arc, self.total[pidx])
        key = self.base[pidx] + a
        i = np.searchsorted(self.glob, key) - 1
        i = np.minimum(np.maximum(i, self.start[pidx]), self.last_seg[pidx])
        seg = np.maximum(self.loc[i + 1] - self.loc[i], 1e-6)
        t = (a - self.loc[i]) / seg
        p0, p1 = self.pts[i], self.pts[i + 1]
        d = p1 - p0
        x = p0[:, 0] + d[:, 0] * t
        y = p0[:, 1] + d[:, 1] * t
        ln = np.hypot(d[:, 0], d[:, 1])
        heading = np.where(ln > 1e-6, np.arctan2(d[:, 1], d[:, 0]), 0.0)
        z = self.z[i] + (self.z[i + 1] - self.z[i]) * t
        return x, y, heading, z


class Fleet:
    """Legacy polyline fixture/preview fleet; runtime uses traffic.DriverFleet."""

    __slots__ = ("pidx", "arc", "speed", "desired", "lane", "kind", "color", "forward", "n", "clock", "wheel", "acceleration")

    def __init__(self, pidx, arc, speed, lane, kind, color, forward):
        self.pidx = np.asarray(pidx, np.int64)
        self.arc = np.asarray(arc, np.float64)
        self.speed = np.asarray(speed, np.float64)
        self.desired = self.speed.copy()
        self.lane = np.asarray(lane, np.float64)
        self.kind = np.asarray(kind, np.int64)
        self.color = np.asarray(color, np.float32).reshape(-1, 3)
        self.forward = np.asarray(forward, bool)
        self.n = len(self.pidx)
        self.clock = 0.0
        self.wheel = np.zeros(self.n)
        self.acceleration = np.zeros(self.n)

    @classmethod
    def from_vehicles(cls, vehicles):
        """Pack a list of objects with path_idx/arc/speed/lane_offset/kind/color/forward."""
        if not vehicles:
            return cls([], [], [], [], [], np.zeros((0, 3), np.float32), [])
        return cls([v.path_idx for v in vehicles], [v.arc for v in vehicles],
                   [v.speed for v in vehicles], [v.lane_offset for v in vehicles],
                   [v.kind for v in vehicles], np.stack([v.color for v in vehicles]),
                   [v.forward for v in vehicles])

    def take(self, indices):
        other = Fleet(self.pidx[indices], self.arc[indices], self.speed[indices], self.lane[indices],
                      self.kind[indices], self.color[indices], self.forward[indices])
        other.desired = self.desired[indices].copy(); other.clock = self.clock
        other.wheel = self.wheel[indices].copy()
        other.acceleration = self.acceleration[indices].copy()
        return other

    def advance(self, dt, roads):
        """IDM compatibility helper; city routing/control live in TrafficCore."""
        if not self.n: return
        dt = min(max(float(dt), 0), .1)
        self.clock += dt
        totals = roads.total[self.pidx]
        progress = np.where(self.forward, self.arc, totals - self.arc)
        lane_id = self.pidx*2 + self.forward.astype(np.int64)
        order = np.lexsort((progress, lane_id))
        ahead = np.roll(order, -1)
        gap = np.full(self.n, np.inf)
        leader_speed = np.zeros(self.n)
        same = lane_id[order] == lane_id[ahead]
        separation = progress[ahead] - progress[order]
        valid = same & (separation > 0)
        gap[order[valid]] = separation[valid]
        leader_speed[order[valid]] = self.speed[ahead[valid]]
        # Heavy vehicles need more space; the gap is centre-to-centre.
        clearance = np.where(np.isin(self.kind, (3, 4, 7, 8)), 11.0, 6.5)
        desired = np.minimum(self.desired, roads.limit[self.pidx])
        direction = self.forward.astype(np.int64)
        red = roads.signal[self.pidx, direction] & ((self.clock + self.pidx*7.13) % 52 < 22)
        remaining = totals-progress
        stop_gap = np.where(red, np.maximum(0, remaining-3), np.inf)
        bumper_gap = gap - (clearance - 2.)
        stopping = stop_gap < bumper_gap
        self.acceleration = idm_acceleration(self.speed, desired, np.minimum(bumper_gap, stop_gap),
                                             np.where(stopping, 0., leader_speed))
        self.speed, move = integrate_idm(self.speed, self.acceleration, dt,
                                         np.minimum(np.maximum(0., gap-clearance), stop_gap))
        self.wheel = np.mod(self.wheel + move / WHEEL_RADII[self.kind], 2 * np.pi)
        progress += move
        end = np.flatnonzero(progress >= totals)
        self.arc = np.where(self.forward, progress, totals-progress)
        for i in end:
            old, direction = self.pidx[i], int(self.forward[i])
            nxt = roads.links[old, direction]
            forward = roads.link_forward[old, direction] if nxt >= 0 else (self.forward[i] if roads.oneway[old] else not self.forward[i])
            target = nxt if nxt >= 0 else old
            others = (self.pidx == target) & (self.forward == forward)
            others[i] = False
            start_gap = self.arc[others] if forward else roads.total[target] - self.arc[others]
            if len(start_gap) and start_gap.min() < 13:
                self.arc[i] = totals[i] - .001 if direction else .001
                self.speed[i] = 0
                continue
            if nxt >= 0:
                self.pidx[i] = nxt; self.forward[i] = roads.link_forward[old, direction]
                self.lane[i] = roads.lane[nxt]
            else:
                # Dead-end two-way roads turn back; one-way boundaries recycle.
                self.forward[i] = forward
            total = roads.total[self.pidx[i]]
            overshoot = min(progress[i]-totals[i], total*.5)
            self.arc[i] = overshoot if self.forward[i] else total-overshoot


def instance_rows(fleet, roads, origin, scale, cam_xy, max_d2, prediction=0.0):
    """Instance rows (``VEHICLE_STRIDE`` floats, world coords) for the vehicles of
    one tile within ``sqrt(max_d2)`` of the camera.  Returns ``(rows, kinds)``."""
    if fleet.n == 0:
        return None, None
    arc = fleet.arc
    if prediction:
        arc = np.clip(arc + np.where(fleet.forward, 1., -1.)*fleet.speed*prediction,
                      0, roads.total[fleet.pidx]-1e-6)
    lx, ly, heading, pz = roads.sample(fleet.pidx, arc)
    dx, dy = np.cos(heading), np.sin(heading)
    sgn = np.where(fleet.forward, 1.0, -1.0)
    fx, fy = dx * sgn, dy * sgn
    # right-of-travel perpendicular, pushed out by the vehicle's lane offset
    lx = lx + fy * fleet.lane
    ly = ly + (-fx) * fleet.lane
    sx = float(scale[0])
    wx = lx * sx + float(origin[0])
    wy = ly + float(origin[1])
    near = (wx - cam_xy[0]) ** 2 + (wy - cam_xy[1]) ** 2 <= max_d2
    if not near.any():
        return None, None
    wh = np.arctan2(fy, fx * sx) if sx > 1e-6 else np.arctan2(fy, fx)
    c, s = np.cos(wh[near]), np.sin(wh[near])
    m = int(near.sum())
    rows = np.zeros((m, VEHICLE_STRIDE), np.float32)
    rows[:, 0] = c;  rows[:, 1] = s
    rows[:, 4] = -s; rows[:, 5] = c
    rows[:, 10] = 1.0
    rows[:, 12] = wx[near]; rows[:, 13] = wy[near]; rows[:, 14] = pz[near]; rows[:, 15] = 1.0
    rows[:, 16:19] = fleet.color[near]
    rows[:, 19] = LAMPS + np.where(fleet.acceleration[near] < -.4, BRAKE, 0)
    rows[:, 20] = np.mod(fleet.wheel[near] + fleet.speed[near]*prediction / WHEEL_RADII[fleet.kind[near]], 2*np.pi)
    rows[:, 22] = fleet.speed[near]
    rows[:, 23] = 1.
    rows[:, 11] = WHEEL_RADII[fleet.kind[near]]
    return rows, fleet.kind[near]
