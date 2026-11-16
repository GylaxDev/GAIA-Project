"""The player on foot: camera-relative walking, jogging, sprinting and jumping.

Everything here is independent of Pygame and OpenGL so it can be tested headless.
Ground, wall and water queries run against the same ``DrivingScene`` triangle
indexes the cars use, so the character stands on pavement, steps up kerbs, is
stopped by facades and parked cars, and cannot wade into the harbour.

World axes: x/y ground plane, z up, metres; heading 0 faces +x.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pedestrian_gait import gait
from .pedestrian_models import (ACC_COAT, ACC_HAIR_SHORT, ACC_CAP, ACC_IDLE, ACC_PLAYER, ACC_BEARD, ACC_WATCH, HEIGHT)
from .human_assets import ACC_MASK, PLAYER
from ..vehicles.vehicle_instances import LENGTHS, WIDTHS

PED_STRIDE = 28
# instance-row columns (shared with pedestrians.py and shaders/pedestrian.py)
C_POS, C_HEADING, C_PHASE, C_AMP, C_SPEED, C_FLAGS = 0, 3, 4, 5, 6, 7
C_HEIGHT, C_WIDTH, C_HEADSCALE, C_SEED = 8, 9, 10, 11
C_SKIN, C_HAIR, C_TOP, C_BOTTOM, C_SHOE, C_ACCENT, C_TOPSTYLE, C_BOTSTYLE = 12, 13, 14, 15, 16, 17, 18, 19
C_BULK, C_VIS, C_HEADYAW, C_NOD, C_TURN = 20, 21, 22, 23, 24
C_VARIANT = 25


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SceneQuery:
    """Tile-aware point queries over a DrivingScene (cached per scene generation)."""

    def __init__(self):
        self._generation = -1
        self._boxes = {}

    def _refresh(self, scene):
        if scene.generation == self._generation:
            return
        boxes = {}
        for key, placed in scene.tiles.items():
            bounds = placed.tile.ground.bounds
            if not len(bounds):
                continue
            lo = bounds[:, :2].min(axis=0) * placed.scale + placed.origin
            hi = bounds[:, 2:].max(axis=0) * placed.scale + placed.origin
            boxes[key] = (lo, hi)
        self._boxes, self._generation = boxes, scene.generation

    def placements(self, scene, x, y, pad=1.):
        self._refresh(scene)
        for key, (lo, hi) in self._boxes.items():
            if lo[0] - pad <= x <= hi[0] + pad and lo[1] - pad <= y <= hi[1] + pad:
                placed = scene.tiles.get(key)
                if placed is not None:
                    yield placed

    def ground(self, scene, x, y, ref=None, rise=.5):
        """-> (ground z or None, water z or None) under (x, y).

        With a reference height only surfaces within ``rise`` above and ~1 m below it
        count, so an overpass above or a deck below is never mistaken for the pavement.
        """
        best, water = -np.inf, -np.inf
        point = np.array([[x, y]], np.float64)
        for placed in self.placements(scene, x, y):
            local = placed.local(point)
            g = placed.tile.ground.sample(local, None if ref is None else float(ref) + rise - .65)[0]
            if np.isfinite(g):
                best = max(best, float(g))
            w = placed.tile.water.sample(local)[0]
            if np.isfinite(w):
                water = max(water, float(w))
        return (best if np.isfinite(best) else None), (water if np.isfinite(water) else None)

    def push_out_walls(self, scene, xy, z, radius, prev=None, height=1.7, step=.30):
        """Slide a vertical cylinder out of facade triangles; -> corrected xy, touched."""
        xy = np.asarray(xy, np.float64).copy()
        touched = False
        for placed in self.placements(scene, xy[0], xy[1], pad=radius + 1.):
            local = placed.local(xy.reshape(1, 2))
            tri = placed.tile.walls.query(local, (radius + .6) / float(placed.scale.min()))
            if not len(tri):
                continue
            keep = (tri[:, :, 2].max(axis=1) > z + step) & (tri[:, :, 2].min(axis=1) < z + height)
            if not keep.any():
                continue
            poly = tri[keep][:, :, :2] * placed.scale + placed.origin           # (n, 3, 2)
            a, b = poly, np.roll(poly, -1, axis=1)
            ab = b - a
            ab2 = np.maximum((ab * ab).sum(-1), 1e-9)
            rows = np.arange(len(poly))
            for _ in range(4):
                t = np.clip(((xy - a) * ab).sum(-1) / ab2, 0., 1.)
                closest = a + ab * t[..., None]
                d = xy - closest
                dist = np.hypot(d[..., 0], d[..., 1])
                edge = dist.argmin(axis=1)
                dmin = dist[rows, edge]
                hit = dmin < radius
                if not hit.any():
                    break
                k = int(np.argmin(np.where(hit, dmin, np.inf)))
                vec = d[k, edge[k]]
                if dmin[k] > 1e-5:
                    direction = vec / dmin[k]
                else:                                  # centre exactly on the wall: go back where we came from
                    back = (np.asarray(prev, np.float64) - closest[k, edge[k]]) if prev is not None else np.array([1., 0.])
                    direction = back / max(float(np.hypot(*back)), 1e-6)
                xy = xy + direction * (radius - dmin[k] + 1e-3)
                touched = True
        return xy, touched

    def ray_distance(self, scene, origin, direction, max_distance):
        """Distance along a ray to the first facade triangle (Moller-Trumbore), or max_distance."""
        origin = np.asarray(origin, np.float64)
        direction = np.asarray(direction, np.float64)
        end = origin + direction * max_distance
        best = float(max_distance)
        for placed in self.placements(scene, *(origin[:2] + end[:2]) * .5, pad=max_distance * .5 + 2.):
            local = placed.local(np.array([origin[:2], end[:2]]))
            tri = placed.tile.walls.query(local, 1. / float(placed.scale.min()))
            if not len(tri):
                continue
            tri = tri.copy()
            tri[:, :, :2] = tri[:, :, :2] * placed.scale + placed.origin
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            p = np.cross(direction, e2)
            det = np.einsum('ij,ij->i', e1, p)
            ok = np.abs(det) > 1e-9
            inv = np.where(ok, 1. / np.where(ok, det, 1.), 0.)
            s = origin - tri[:, 0]
            u = np.einsum('ij,ij->i', s, p) * inv
            q = np.cross(s, e1)
            v = (q @ direction) * inv
            t = np.einsum('ij,ij->i', e2, q) * inv
            good = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > .05)
            if good.any():
                best = min(best, float(t[good].min()))
        return best


def push_out_cars(xy, radius, rows, kinds, reach=12.):
    """Slide a cylinder out of oriented car boxes. rows are 24-float vehicle instances."""
    if rows is None or not len(rows):
        return np.asarray(xy, np.float64), False
    xy = np.asarray(xy, np.float64).copy()
    near = np.flatnonzero(np.hypot(rows[:, 12] - xy[0], rows[:, 13] - xy[1]) < reach)
    touched = False
    for i in near:
        row, kind = rows[i], int(kinds[i])
        fwd = row[0:2].astype(np.float64)
        n = float(np.hypot(*fwd))
        if n < 1e-6:
            continue
        fwd /= n
        left = np.array([-fwd[1], fwd[0]])
        d = xy - row[12:14]
        lx, ly = float(d @ fwd), float(d @ left)
        hl, hw = LENGTHS[kind] * .5, WIDTHS[kind] * .5
        cx, cy = min(max(lx, -hl), hl), min(max(ly, -hw), hw)
        dx, dy = lx - cx, ly - cy
        dist = math.hypot(dx, dy)
        if dist >= radius:
            continue
        if dist > 1e-6:
            px, py = dx / dist * (radius - dist), dy / dist * (radius - dist)
        else:                                          # centre inside the box: leave by the nearest face
            gaps = (hl - abs(lx), hw - abs(ly))
            if gaps[0] < gaps[1]:
                px, py = math.copysign(gaps[0] + radius, lx), 0.
            else:
                px, py = 0., math.copysign(gaps[1] + radius, ly)
        xy += px * fwd + py * left
        touched = True
    return xy, touched


@dataclass
class WalkControls:
    forward: float = 0.
    side: float = 0.
    sprint: bool = False
    slow: bool = False
    jump: bool = False


class Walker:
    """Aiden-style trench-coat character: look, pose and physics."""
    RADIUS = .30
    WALK, JOG, SPRINT = 1.05, 1.9, 5.6      # stroll (Ctrl), brisk walk (default), run (Shift)
    JUMP_SPEED, GRAVITY = 4.4, 17.
    STEP_UP = .50

    def __init__(self):
        self.active = False
        self.position = np.zeros(3)
        self.heading = 0.
        self.velocity = np.zeros(2)
        self.vz = 0.
        self.grounded = True
        self.phase = 0.
        self.amp = self.sf = self.speed = 0.
        self.head_yaw = 0.
        self.height = 1.80
        self.face_covering = False
        self.query = SceneQuery()
        self._airborne_time = 0.
        self._safe = None
        self.jump_held = False
        self.blocked = False
        self._prev_heading = 0.
        self.turn_lean = 0.

    # ---------------------------------------------------------------- placement
    def place(self, scene, x, y, heading, ref=None):
        ground, water = self.query.ground(scene, x, y, ref, rise=3. if ref is not None else .5)
        if ground is None or (water is not None and water > ground + .025):
            return False
        self.position = np.array([x, y, ground + .01])
        self.heading, self.velocity, self.vz = float(heading), np.zeros(2), 0.
        self.grounded, self.speed, self.amp, self.sf = True, 0., 0., 0.
        self._prev_heading, self.turn_lean = float(heading), 0.
        self._airborne_time = 0.
        self._safe = self.position.copy()
        self.active = True
        return True

    def place_near(self, scene, x, y, heading, ref=None, radius=45.):
        """Spiral outward from (x, y) to the first supported, dry spot."""
        if self.place(scene, x, y, heading, ref):
            return True
        for r in np.arange(3., radius, 3.):
            for a in np.linspace(0, 2 * math.pi, max(8, int(r * 1.6)), endpoint=False):
                if self.place(scene, x + r * math.cos(a), y + r * math.sin(a), heading, ref):
                    return True
        return False

    def exit_car(self, scene, car, obstacles=None):
        """Step out beside a car: driver's side first, then the other side, then behind."""
        c, s = math.cos(car.heading), math.sin(car.heading)
        left = np.array([-s, c])
        fwd = np.array([c, s])
        half_w, half_l = WIDTHS[car.kind] * .5, LENGTHS[car.kind] * .5
        spots = [car.position[:2] + left * (half_w + .75),
                 car.position[:2] - left * (half_w + .75),
                 car.position[:2] - fwd * (half_l + .9),
                 car.position[:2] + fwd * (half_l + .9)]
        for spot in spots:
            if not self.place(scene, float(spot[0]), float(spot[1]), car.heading, float(car.position[2])):
                continue
            xy, touched = self.query.push_out_walls(scene, self.position[:2], self.position[2], self.RADIUS)
            if touched and np.hypot(*(xy - spot)) > .5:
                continue
            if obstacles is not None:
                xy2, touched = push_out_cars(self.position[:2], self.RADIUS, *obstacles)
                if touched and np.hypot(*(xy2 - spot)) > .6:
                    continue
            return True
        return self.place_near(scene, float(car.position[0]), float(car.position[1]), car.heading,
                               float(car.position[2]), radius=14.)

    # ------------------------------------------------------------------- update
    def update(self, dt, scene, controls, cam_yaw, obstacles=None):
        dt = float(min(max(dt, 1e-4), .1))
        fwd = np.array([math.cos(cam_yaw), math.sin(cam_yaw)])
        right = np.array([fwd[1], -fwd[0]])
        want = fwd * controls.forward + right * controls.side
        mag = float(np.hypot(*want))
        if mag > 1.:
            want /= mag
            mag = 1.
        top = self.SPRINT if controls.sprint else self.WALK if controls.slow else self.JOG
        target = want * top
        # snappier to start than to stop; less authority in the air
        authority = 1. if self.grounded else .25
        delta = target - self.velocity
        dist = float(np.hypot(*delta))
        accel = (18. if mag > .05 else 24.) * authority * dt
        self.velocity = target if dist <= accel else self.velocity + delta / dist * accel

        if controls.jump and not self.jump_held and self.grounded:
            self.vz, self.grounded = self.JUMP_SPEED, False
        self.jump_held = controls.jump

        moved = 0.
        steps = max(1, int(math.ceil(float(np.hypot(*self.velocity)) * dt / .18)))
        for _ in range(steps):
            moved += self._step(dt / steps, scene, obstacles)
        self.speed = moved / dt if dt > 0 else 0.

        # turn toward where we are going; fast when a new direction is demanded
        if mag > .08 and self.speed > .25:
            goal = math.atan2(self.velocity[1], self.velocity[0])
            rate = 14. if self.speed < 3.5 else 9.
            self.heading += wrap(goal - self.heading) * (1. - math.exp(-rate * dt))
        self.heading = wrap(self.heading)
        rel = wrap(cam_yaw - self.heading)
        self.head_yaw += (max(-1., min(1., rel)) * .75 - self.head_yaw) * (1. - math.exp(-6. * dt))

        if self.grounded:
            amp, self.sf, rate = gait(self.speed, self.height)
            self.amp += (amp - self.amp) * (1. - math.exp(-12. * dt))
            self.phase = (self.phase + rate * dt) % (2 * math.pi)
        else:                                           # airborne: legs trail, stride frozen
            self.amp += (.30 - self.amp) * (1. - math.exp(-8. * dt))
            self.sf = .55
        turn = wrap(self.heading - self._prev_heading) / dt if dt > 0 else 0.
        self._prev_heading = self.heading
        goal = max(-.10, min(.10, -turn * .018 * min(self.speed / 3., 1.)))     # lean into the turn
        self.turn_lean += (goal - self.turn_lean) * (1. - math.exp(-6. * dt))
        if self.grounded and self.speed > .5:
            self._safe = self.position.copy()

    def _step(self, dt, scene, obstacles):
        old = self.position.copy()
        xy = old[:2] + self.velocity * dt
        xy, hit_wall = self.query.push_out_walls(scene, xy, old[2], self.RADIUS, prev=old[:2])
        hit_car = False
        if obstacles is not None:
            xy, hit_car = push_out_cars(xy, self.RADIUS, *obstacles)
            if hit_car:
                xy, _ = self.query.push_out_walls(scene, xy, old[2], self.RADIUS, prev=old[:2])
        self.blocked = hit_wall or hit_car

        ref = old[2] if self.grounded else old[2] + .35
        ground, water = self.query.ground(scene, xy[0], xy[1], ref, rise=self.STEP_UP if self.grounded else .1)
        if ground is None or (water is not None and water > ground + .025):
            # no support or open water ahead: refuse the move and kill the momentum
            self.velocity[:] = 0.
            xy = old[:2]
            ground, _ = self.query.ground(scene, xy[0], xy[1], ref, rise=self.STEP_UP)
            if ground is None:
                ground = old[2]
        z = old[2]
        if self.grounded:
            if ground < z - .45:                         # stepped off a ledge
                self.grounded, self.vz = False, 0.
            else:
                z += (ground - z) * (1. - math.exp(-20. * dt))
                if abs(ground - z) < .002:
                    z = ground
        if not self.grounded:
            self.vz -= self.GRAVITY * dt
            z += self.vz * dt
            self._airborne_time += dt
            if z <= ground and self.vz <= 0.:
                z, self.vz, self.grounded, self._airborne_time = ground, 0., True, 0.
            if self._airborne_time > 2.5 and self._safe is not None and z < ground - 8.:
                xy, z = self._safe[:2].copy(), float(self._safe[2])   # fell out of the world
                self.grounded, self.vz, self._airborne_time = True, 0., 0.
        self.position = np.array([xy[0], xy[1], z])
        return float(np.hypot(*(xy - old[:2])))

    # ------------------------------------------------------------------ render
    def instance_row(self):
        row = np.zeros(PED_STRIDE, np.float32)
        row[0:3] = self.position
        row[C_HEADING] = self.heading
        row[C_PHASE], row[C_AMP], row[C_SPEED] = self.phase, self.amp, self.sf
        flags = ACC_COAT | ACC_HAIR_SHORT | ACC_CAP | ACC_PLAYER | ACC_BEARD | ACC_WATCH
        if self.face_covering:
            flags |= ACC_MASK
        if self.speed < .12 and self.grounded:
            flags |= ACC_IDLE
        row[C_FLAGS] = flags
        row[C_HEIGHT], row[C_WIDTH], row[C_HEADSCALE], row[C_SEED] = self.height, 1.0, 1.0, .31
        # Tanned skin, dark stubble, brown leather, dark jeans and a black cap.
        row[C_SKIN], row[C_HAIR], row[C_TOP], row[C_BOTTOM] = 2, 7, 15, 3
        row[C_SHOE], row[C_ACCENT], row[C_TOPSTYLE], row[C_BOTSTYLE] = 7, 15, 1, 0
        row[C_BULK], row[C_VIS], row[C_HEADYAW], row[C_NOD] = 1.04, 1., self.head_yaw, 0.
        row[C_TURN] = self.turn_lean
        row[C_VARIANT] = PLAYER
        return row
