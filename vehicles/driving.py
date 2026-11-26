"""Player-owned arcade cars: fixed-step bicycle motion and packed render streams.

The player stays on the render thread; AI traffic ownership/mailboxes are never
modified. Controls are independent of Pygame so physics and input routing can be
tested without a window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from .vehicle_instances import (VEHICLE_STRIDE, LENGTHS, WIDTHS, HEIGHTS, WHEELBASES,
                                WHEEL_RADII, MASSES, LAMPS, BRAKE, LEFT, RIGHT,
                                SUSPENSION, VehicleFrame, normalize_instances)

DRIVABLE_KINDS = (0, 1, 2, 5, 6, 7, 8)
# Quarter-car suspension x4 under a sprung 3-DOF body.
#   sprung body  : heave / pitch / roll integrated in the world (not slaved to the road)
#   four wheels  : each an unsprung mass on a tyre spring, so a wheel can climb a kerb
#                  or drop into a dip on its own while the body follows through the
#                  spring + damper, and the wheel visibly extends / compresses
# Spring rates follow a ~1.25 Hz ride frequency. Rebound damping is stiffer than bump
# damping (as on real dampers), which keeps a drop-off from slamming back up.
RIDE_HZ = 1.25
BUMP_DAMPING = .30          # damping ratio while compressing
REBOUND_DAMPING = .58       # damping ratio while extending
ANTI_ROLL = .35             # anti-roll bar rate as a fraction of the corner spring
UNSPRUNG_RATIO = .035       # wheel + hub + brake mass per corner / total mass
TIRE_DEFLECTION = .012      # static tyre deflection (m) -> tyre spring rate
TIRE_DAMPING = .20
STOP_STIFFNESS = 14.        # bump / droop stop rate as a multiple of the corner spring
BUMP_TRAVEL = .09           # compression available before the bump stop
DROOP_TRAVEL = .11          # extension available before the droop strap
RENDER_TRAVEL = .15         # wheel offset limit sent to the GPU
SUB_STEPS = 4               # tyre hop (~13 Hz) needs a finer step than the 120 Hz sim
PITCH_TRANSFER = .55
ROLL_TRANSFER = .62
RELAX_SECONDS = 4.          # a car you step out of keeps settling this long
GRAVITY = 9.81


@dataclass(frozen=True)
class DriveControls:
    throttle: float = 0.    # positive forward; negative brakes, then reverses
    steer: float = 0.       # positive left, including when backing up
    brake: bool = False


@dataclass(eq=False)
class PlayerCar:
    """``position``/``pitch``/``roll`` are the *ground* pose (collision, spawning).

    The rendered chassis comes from the sprung-body state below, which lags and
    rebounds against that ground pose; each wheel has its own height.
    """
    position: np.ndarray
    heading: float = 0.
    kind: int = 0
    color: np.ndarray = field(default_factory=lambda: np.array([.025, .22, .65], np.float32))
    pitch: float = 0.
    roll: float = 0.
    speed: float = 0.
    steering: float = 0.
    wheel: float = 0.
    distance: float = 0.
    lights: bool = True
    braking: bool = False
    indicator: int = 0
    suspension_ready: bool = False
    body_z: float = 0.          # sprung body origin height (world)
    body_pitch: float = 0.      # sprung body angles (world, absolute)
    body_roll: float = 0.
    body_rates: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))   # heave, pitch, roll
    wheel_z: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float64))      # tyre contact height, FL FR RL RR
    wheel_rate: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float64))
    ground_z: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float64))     # last valid road height per wheel
    suspension_travel: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float64))  # body - wheel: <0 bump, >0 droop
    relax_time: float = 0.
    vlat: float = 0.            # lateral (sideways) velocity in the car frame, +left (m/s)
    yaw_rate: float = 0.        # rad/s
    skid_front: float = 0.      # 0..1 tyre-slide intensity per axle (drives marks and squeal)
    skid_rear: float = 0.
    crush_front: float = 0.
    crush_rear: float = 0.
    crush_left: float = 0.
    crush_right: float = 0.
    crash_impulse: float = 0.
    impact_severity: float = 0.
    drivetrain_damage: float = 0.

    @property
    def slip_angle(self):
        """Angle between where the car points and where it is travelling (rad)."""
        return math.atan2(self.vlat, max(abs(self.speed), 1.)) if abs(self.speed) > 1. else 0.

    def __post_init__(self):
        self.position = np.asarray(self.position, np.float64).reshape(3).copy()
        self.color = np.asarray(self.color, np.float32).reshape(3).copy()
        self.body_rates = np.asarray(self.body_rates, np.float64).reshape(3).copy()
        for name in ('wheel_z', 'wheel_rate', 'ground_z', 'suspension_travel'):
            setattr(self, name, np.asarray(getattr(self, name), np.float64).reshape(4).copy())
        if self.kind not in DRIVABLE_KINDS:
            raise ValueError('only passenger cars, taxis, vans and pickups are player drivable')
        state = (self.position, self.color, self.heading, self.pitch, self.roll, self.speed,
                 self.steering, self.wheel, self.body_z, self.body_pitch, self.body_roll,
                 self.body_rates, self.wheel_z, self.wheel_rate, self.ground_z,
                 self.suspension_travel)
        if not all(np.isfinite(v).all() for v in state):
            raise ValueError('car pose, paint and suspension must be finite')
        self.home = (self.position.copy(), self.heading)

    def settle_suspension(self):
        """Drop the sprung state; the next driven step re-seeds it from the road."""
        self.suspension_ready = False
        self.relax_time = 0.
        self.body_rates.fill(0.)
        self.wheel_rate.fill(0.)
        self.suspension_travel.fill(0.)

    @property
    def body_moving(self):
        return bool(self.suspension_ready and (np.abs(self.body_rates).max() > 1e-4
                                               or np.abs(self.wheel_rate).max() > 1e-3))

    @property
    def body_position(self):
        """Sprung-body origin: what a camera should follow (no kerb snaps)."""
        if not self.suspension_ready:
            return self.position
        return np.array((self.position[0], self.position[1], self.body_z))

    @classmethod
    def from_instance(cls, row, kind):
        row = normalize_instances(row)[0]
        car = cls(row[12:15], math.atan2(float(row[1]), float(row[0])), kind, row[16:19])
        car.pitch = math.atan2(float(row[2]), math.hypot(float(row[0]), float(row[1])))
        car.roll = math.atan2(float(row[6]), float(row[10]))
        car.wheel, car.steering = float(row[20]), float(row[21])
        flags = int(row[19])
        car.lights, car.braking = bool(flags & LAMPS), bool(flags & BRAKE)
        car.indicator = flags & (LEFT | RIGHT)
        if len(row) >= 28:
            car.crush_front = float(row[24])
            car.crush_rear = float(row[25])
            car.crush_left = float(row[26])
            car.crush_right = float(row[27])
        return car

    @property
    def velocity(self):
        return self.speed * np.array([math.cos(self.heading), math.sin(self.heading), math.tan(self.pitch)])

    @property
    def gear(self):
        return 'R' if self.speed < -.05 else 'D' if self.speed > .05 else 'N'

    def instance(self, pose=None):
        """Packed render row.

        ``pose`` overrides ``(position, heading)`` with the interpolated pose, so the
        chassis a frame draws and the pose its camera chases are the same point in
        time instead of two different fixed-step quanta.
        """
        position, heading = (self.position, self.heading) if pose is None else pose
        if self.suspension_ready:
            z, pitch, roll = self.body_z, self.body_pitch, self.body_roll
        else:
            z, pitch, roll = position[2], self.pitch, self.roll
        c, s = math.cos(heading), math.sin(heading)
        cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
        row = np.zeros(VEHICLE_STRIDE, np.float32)
        row[:3] = (c * cp, s * cp, sp)
        row[4:7] = (-s * cr - c * sp * sr, c * cr - s * sp * sr, cp * sr)
        row[8:11] = (s * sr - c * sp * cr, -c * sr - s * sp * cr, cp * cr)
        row[11] = WHEEL_RADII[self.kind]
        row[12:16] = (position[0], position[1], z, 1.)
        row[16:19] = self.color
        flags = (LAMPS if self.lights else 0) | (BRAKE if self.braking else 0) | self.indicator
        if self.suspension_ready:
            # The matrix is the sprung body. Model w entries are unused by the affine
            # transform, so they carry each wheel's vertical offset in the body frame
            # (FL, FR, RL, RR); the flag keeps them apart from AI prediction rates.
            row[3], row[7], row[11], row[15] = -np.clip(self.suspension_travel,
                                                        -RENDER_TRAVEL, RENDER_TRAVEL)
            flags |= SUSPENSION
        row[19] = flags
        row[20:24] = (self.wheel, self.steering, self.speed, 1.)
        row[24:28] = (self.crush_front, self.crush_rear, self.crush_left, self.crush_right)
        return row


def vehicle_obstacles(chunks=(), traffic=None, cars=(), exclude=None, excluded_parked=None):
    """A local, display-only obstacle table, never an edit to the AI worker frame."""
    arrays, kinds = [], []
    # With an owner the table is distance-limited anyway, so drop the far rows
    # per source instead of concatenating every parked car in the city first.
    origin = None if exclude is None else exclude.position[:2]

    def _near(a):
        if origin is None:
            return a
        d = a[:, 12:14] - origin
        return a[np.sum(d * d, axis=1) < 100. ** 2]

    for chunk in chunks:
        for kind, rows in enumerate(chunk.parked_cpu or ()):
            if rows is None or not len(rows):
                continue
            a = normalize_instances(rows)
            if excluded_parked is not None and chunk is excluded_parked[0] and kind == excluded_parked[1]:
                a = np.delete(a, excluded_parked[2], axis=0)
            a = _near(a)
            if len(a):
                arrays.append(a); kinds.append(np.full(len(a), kind, np.int32))
    for kind, rows in enumerate(traffic or ()):
        if rows is not None and len(rows):
            a = _near(normalize_instances(rows))
            if len(a):
                arrays.append(a); kinds.append(np.full(len(a), kind, np.int32))
    for car in cars:
        if car is not exclude:
            arrays.append(car.instance()[None]); kinds.append(np.array([car.kind], np.int32))
    if not arrays:
        return np.empty((0, VEHICLE_STRIDE), np.float32), np.empty(0, np.int32)
    rows, kinds = np.concatenate(arrays), np.concatenate(kinds)
    keep = rows[:, 23] > .5
    if exclude is not None:
        keep &= np.sum((rows[:, 12:14] - exclude.position[:2]) ** 2, axis=1) < 100. ** 2
    return rows[keep], kinds[keep]


def cars_overlap(position, heading, kind, obstacles):
    rows, kinds = obstacles
    if not len(rows):
        return False
    f = np.array([math.cos(heading), math.sin(heading)])
    left = np.array([-f[1], f[0]])
    other = rows[:, :2].astype(np.float64)
    other /= np.maximum(np.linalg.norm(other, axis=1, keepdims=True), 1e-8)
    other_left = np.column_stack((-other[:, 1], other[:, 0]))
    delta = rows[:, 12:14] - position[:2]
    half = np.array([LENGTHS[kind], WIDTHS[kind]]) * .5 + .06
    length, width = LENGTHS[kinds] * .5, WIDTHS[kinds] * .5
    hit = (rows[:, 14] + HEIGHTS[kinds] > position[2] + .2) & (rows[:, 14] < position[2] + HEIGHTS[kind])
    for axis, extent in ((f, half[0]), (left, half[1])):
        hit &= np.abs(delta @ axis) < extent + length * np.abs(other @ axis) + width * np.abs(other_left @ axis)
    for axes, extent in ((other, length), (other_left, width)):
        hit &= np.abs(np.sum(delta * axes, axis=1)) < extent + half[0] * np.abs(axes @ f) + half[1] * np.abs(axes @ left)
    return bool(hit.any())


class DrivingController:
    STEP = 1. / 120.
    MAX_SPEED = 160. / 3.6
    REVERSE_SPEED = 30. / 3.6
    _PARAMS = {}

    def __init__(self):
        self.cars = []
        self.car = None
        self.active = False
        self.controls = DriveControls()
        self.clock = self._pending = self._idle_pending = 0.
        self.blocked = False
        self._last_step = None      # (pose before, pose after) the latest fixed step

    def enter(self, car):
        if car not in self.cars:
            self.cars.append(car)
        self.car, self.active = car, True
        car.speed, car.lights, car.braking = 0., True, False
        car.relax_time = 0.     # a still-settling car keeps its sprung state: no snap on entry
        self.controls, self._pending, self.blocked = DriveControls(), 0., False
        self._last_step = None

    def pause(self):
        self.controls, self._pending = DriveControls(), 0.
        self._last_step = None
        if self.car is not None:
            self.car.speed = 0.
            self.car.vlat = self.car.yaw_rate = self.car.skid_front = self.car.skid_rear = 0.
            self.car.braking = self.active

    def leave(self):
        self.pause()
        if self.car is not None:
            self.car.lights, self.car.braking, self.car.indicator = False, False, 0
            self.car.relax_time = RELAX_SECONDS     # let the body finish rocking to rest
        self.active, self.blocked = False, False

    def toggle_indicator(self, flag):
        if self.active and flag in (LEFT, RIGHT):
            self.car.indicator = 0 if self.car.indicator == flag else flag

    def place(self, car, scene, obstacles):
        support = scene.support(car.position, car.heading, car.kind)
        if support is None:
            return False
        position = car.position.copy(); position[2] = support[0]
        if scene.blocked(position, car.heading, car.kind) or cars_overlap(position, car.heading, car.kind, obstacles):
            return False
        car.position, car.pitch, car.roll = position, support[1], support[2]
        car.settle_suspension()
        return True

    def spawn(self, scene, point, forward, obstacles, kind=0, color=None):
        for _, position, heading in scene.spawn_candidates(point, forward, kind):
            car = PlayerCar(position, heading, kind)
            if color is not None:
                car.color = np.asarray(color, np.float32)
            if self.place(car, scene, obstacles):
                car.home = (car.position.copy(), car.heading)
                return car
        return None

    def reset(self, scene, obstacles):
        if self.car is None:
            return False
        self.pause()
        car = self.car
        recovered = self.spawn(scene, car.position, np.array([math.cos(car.heading), math.sin(car.heading)]),
                               obstacles, car.kind)
        if recovered is None:
            recovered = PlayerCar(car.home[0], car.home[1], car.kind, car.color)
            if not self.place(recovered, scene, obstacles):
                return False
        car.position, car.heading = recovered.position, recovered.heading
        car.pitch, car.roll = recovered.pitch, recovered.roll
        car.wheel, car.steering, car.indicator = 0., 0., 0
        car.settle_suspension()
        self.blocked = False
        return True

    def update(self, dt, scene, obstacles, wetness=0.):
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.:
            return
        dt = min(dt, .25)
        self.clock += dt
        if not self.active:
            self._relax(dt)
            return
        self._pending += dt
        steps = int((self._pending + 1e-10) / self.STEP)
        self._pending = max(0., self._pending - steps * self.STEP)
        self.blocked = False
        wetness = float(np.clip(wetness, 0., 1.))
        car = self.car
        for _ in range(steps):
            before = self._pose(car)
            self._step(scene, obstacles, wetness)
            self._last_step = (before, self._pose(car))

    @staticmethod
    def _pose(car):
        """Sprung-body ``(position, heading)``: what a chase camera follows."""
        z = float(car.body_z) if car.suspension_ready else float(car.position[2])
        return (np.array((car.position[0], car.position[1], z), np.float64),
                float(car.heading))

    def render_pose(self):
        """Pose to draw and to chase on this frame, or ``None`` when not moving.

        The simulation advances in whole 1/120 s steps, but frames arrive at whatever
        rate the display runs at, so the stepped pose jumps by 0–3 quanta per frame.
        Aiming the camera at that stepped pose is what made the chase view judder.
        Rendering one step back, blended by the accumulator remainder, gives the car
        and the camera a continuous motion at any refresh rate.
        """
        step = self._last_step
        if self.car is None or step is None:
            return None
        (p0, h0), (p1, h1) = step
        t = self._pending / self.STEP
        turn = math.atan2(math.sin(h1 - h0), math.cos(h1 - h0))
        return (p0 + (p1 - p0) * t, h0 + turn * t)

    def _relax(self, dt):
        """Cars the player has left keep settling on their springs, with no scene queries."""
        live = [c for c in self.cars if c.suspension_ready and c.relax_time > 0.]
        if not live:
            self._idle_pending = 0.
            return
        self._idle_pending += dt
        steps = int((self._idle_pending + 1e-10) / self.STEP)
        self._idle_pending -= steps * self.STEP
        for car in live:
            for _ in range(steps):
                self._update_suspension(car, 0., 0., car.ground_z, self.STEP)
            car.relax_time = max(0., car.relax_time - steps * self.STEP)

    @staticmethod
    def _contact(scene, position, heading, kind):
        """Return ``((support, pitch, roll), wheel_ground)`` for any scene.

        ``wheel_ground`` is the absolute road height under each wheel (FL, FR, RL, RR),
        in the same reference as ``support``.
        """
        enveloped = getattr(scene, 'ground_contact', None)
        if enveloped is not None:
            return enveloped(position, heading, kind)
        contact = getattr(scene, 'contact', None)
        if contact is not None:
            return contact(position, heading, kind)
        support = scene.support(position, heading, kind)
        if support is None:
            return None
        axle, track = WHEELBASES[kind] * .5, WIDTHS[kind] * .42
        offsets = np.array([[axle, track], [axle, -track],
                            [-axle, track], [-axle, -track]])
        wheels = (support[0] + offsets[:, 0] * math.tan(support[1])
                  + offsets[:, 1] * math.tan(support[2]))
        return support, wheels.astype(np.float64)

    @classmethod
    def _suspension_parameters(cls, kind):
        p = cls._PARAMS.get(kind)
        if p is not None:
            return p
        mass = float(MASSES[kind])
        axle, track = WHEELBASES[kind] * .5, WIDTHS[kind] * .42
        unsprung = UNSPRUNG_RATIO * mass
        sprung = mass - 4. * unsprung
        spring = (sprung / 4.) * (2. * math.pi * RIDE_HZ) ** 2
        corner_load = (sprung / 4. + unsprung) * GRAVITY
        tyre = corner_load / TIRE_DEFLECTION
        p = {
            'x': np.array([axle, axle, -axle, -axle]),
            'y': np.array([track, -track, track, -track]),
            'm_s': sprung, 'm_u': unsprung,
            'k': spring, 'k_arb': ANTI_ROLL * spring, 'k_stop': STOP_STIFFNESS * spring,
            'c_bump': BUMP_DAMPING * math.sqrt(spring * sprung),
            'c_reb': REBOUND_DAMPING * math.sqrt(spring * sprung),
            'k_tyre': tyre, 'c_tyre': 2. * TIRE_DAMPING * math.sqrt(tyre * unsprung),
            'f_tyre': corner_load,
            'i_pitch': sprung * (LENGTHS[kind] ** 2 + HEIGHTS[kind] ** 2) / 12.,
            'i_roll': sprung * (WIDTHS[kind] ** 2 + HEIGHTS[kind] ** 2) / 12.,
            'cg': HEIGHTS[kind] * .38,
        }
        cls._PARAMS[kind] = p
        return p

    def _begin_suspension(self, car, support, ground):
        """Seed the sprung state at rest on the road (entering, reset, big pose jumps)."""
        p = self._suspension_parameters(car.kind)
        car.body_z, car.body_pitch, car.body_roll = float(support[0]), float(support[1]), float(support[2])
        car.body_rates.fill(0.)
        car.wheel_z = np.asarray(ground, np.float64).copy()
        car.ground_z = car.wheel_z.copy()
        car.wheel_rate.fill(0.)
        body = car.body_z + p['x'] * math.tan(car.body_pitch) + p['y'] * math.tan(car.body_roll)
        car.suspension_travel[:] = np.clip(body - car.wheel_z, -RENDER_TRAVEL, RENDER_TRAVEL)
        car.suspension_ready = True

    def _update_suspension(self, car, acceleration, yaw_rate, ground, dt):
        """Advance the sprung body and the four unsprung wheels by ``dt``.

        Per corner i:  s = body_i - wheel_i  (<0 compressed, >0 extended)
          force on body   D = -k s - c(sign) s' - anti-roll + bump/droop stops
          wheel           m_u a = (tyre - tyre_static) - D      tyre = max(F0 + k_t pen - c_t v, 0)
        so a wheel that loses the road falls away under its own weight and the damper
        (rebound) while the body stays up; a wheel that meets a kerb is pushed up by the
        tyre, compresses the spring, and the body rises smoothly behind it. Braking,
        acceleration and cornering enter as pitch / roll moments on the body.
        """
        p = self._suspension_parameters(car.kind)
        x, y = p['x'], p['y']
        goal = np.asarray(ground, np.float64)
        start = car.ground_z
        z, pitch, roll = car.body_z, car.body_pitch, car.body_roll
        zd, pd, rd = (float(v) for v in car.body_rates)
        u, ud = car.wheel_z.copy(), car.wheel_rate.copy()
        longitudinal = float(np.clip(acceleration, -15., 10.))
        lateral = float(np.clip(car.speed * yaw_rate, -12., 12.))
        ext_pitch = p['m_s'] * longitudinal * p['cg'] * PITCH_TRANSFER
        ext_roll = p['m_s'] * lateral * p['cg'] * ROLL_TRANSFER
        h = dt / SUB_STEPS
        for i in range(SUB_STEPS):
            road = start + (goal - start) * ((i + 1) / SUB_STEPS)
            tp, tr = math.tan(pitch), math.tan(roll)
            body = z + x * tp + y * tr
            body_rate = zd + x * pd * (1. + tp * tp) + y * rd * (1. + tr * tr)
            s, sd = body - u, body_rate - ud
            force = -p['k'] * s - np.where(sd < 0., p['c_bump'], p['c_reb']) * sd
            front, rear = s[0] - s[1], s[2] - s[3]
            force -= p['k_arb'] * np.array((front, -front, rear, -rear))
            force += p['k_stop'] * (np.maximum(-s - BUMP_TRAVEL, 0.) - np.maximum(s - DROOP_TRAVEL, 0.))
            tyre = np.maximum(p['f_tyre'] + p['k_tyre'] * (road - u) - p['c_tyre'] * ud, 0.) - p['f_tyre']
            ud = np.clip(ud + (tyre - force) / p['m_u'] * h, -15., 15.)
            u = u + ud * h
            zd = float(np.clip(zd + float(force.sum()) / p['m_s'] * h, -6., 6.))
            pd = float(np.clip(pd + (float(force @ x) + ext_pitch) / p['i_pitch'] * h, -6., 6.))
            rd = float(np.clip(rd + (float(force @ y) + ext_roll) / p['i_roll'] * h, -6., 6.))
            z, pitch, roll = z + zd * h, pitch + pd * h, roll + rd * h
        values = (z, pitch, roll, zd, pd, rd)
        if not (all(math.isfinite(v) for v in values) and np.isfinite(u).all() and np.isfinite(ud).all()
                and abs(pitch) < 1.2 and abs(roll) < 1.0):
            car.settle_suspension()
            return
        car.body_z, car.body_pitch, car.body_roll = z, pitch, roll
        car.body_rates[:] = (zd, pd, rd)
        car.wheel_z, car.wheel_rate = u, ud
        car.ground_z = goal.copy()
        body = z + x * math.tan(pitch) + y * math.tan(roll)
        car.suspension_travel[:] = np.clip(body - u, -RENDER_TRAVEL, RENDER_TRAVEL)

    def _step(self, scene, obstacles, wetness):
        car, controls, dt = self.car, self.controls, self.STEP
        throttle = float(np.clip(controls.throttle, -1., 1.))
        steer = float(np.clip(controls.steer, -1., 1.))
        old_speed = car.speed
        braking = controls.brake or (throttle * old_speed < 0. and abs(old_speed) > .05)
        grip = 1. - .35 * wetness
        # Space is a handbrake at speed: it locks the rear axle (loses lateral grip) and
        # decelerates less than the sticky low-speed stop, which is what lets the tail slide.
        handbrake = bool(controls.brake) and abs(old_speed) > 4.
        drag = .28 + .0015 * old_speed * old_speed
        if braking:
            hard = 7.5 if handbrake else (12. if controls.brake else 8.5)
            acceleration = -math.copysign(hard * grip + drag, old_speed) if old_speed else 0.
        else:
            acceleration = throttle * (4.6 if throttle >= 0. else 3.2)
            if old_speed:
                acceleration -= math.copysign(drag, old_speed)
        speed = old_speed + acceleration * dt
        if (braking or not throttle) and old_speed * speed <= 0.:
            speed = 0.
        speed = float(np.clip(speed, -self.REVERSE_SPEED, self.MAX_SPEED))
        wb = WHEELBASES[car.kind]
        # Speed-sensitive steering. Normal driving is capped just inside the tyres' grip so
        # it does not skid on its own; the handbrake gives full lock, and wet roads lower the
        # grip (and therefore the cap) so they slide earlier.
        limit = .60 / (1. + .045 * abs(speed))
        if not handbrake:
            limit = min(limit, math.atan(7.8 * (1. - .20 * wetness) * wb / max(speed * speed, 1.)))
        goal = steer * limit
        car.steering = goal + (car.steering - goal) * math.exp(-9. * dt)
        delta = car.steering
        vx = .5 * (old_speed + speed)
        vy, w = car.vlat, car.yaw_rate
        # Kinematic response (low speed / reversing) ...
        w_kin = vx * math.tan(delta) / wb
        # ... and a dynamic bicycle model with saturating tyres (tanh friction curve).
        dyn = float(np.clip((abs(vx) - 2.5) / 2.5, 0., 1.)) if vx > 0. else 0.
        skid_f = skid_r = 0.
        if dyn > 0.:
            g = 9.81
            mu = .95 * grip * (1. - .10 * wetness)      # standing water: slightly less than the steering cap assumes
            along = (speed - old_speed) / dt
            share_f = float(np.clip(.5 - .10 * along / g, .35, .65))      # braking loads the nose
            lock = 1. - .6 * min(1., (along / (mu * g)) ** 2)             # friction circle
            fmax_f = mu * g * share_f * 2. * lock * .5
            fmax_r = mu * g * (1. - share_f) * 2. * lock * .5 * (.22 if handbrake else 1.)
            vxs = max(vx, 2.5)
            half = .5 * wb
            alpha_f = math.atan2(vy + half * w, vxs) - delta
            alpha_r = math.atan2(vy - half * w, vxs)
            f_f = -fmax_f * math.tanh(6.5 * alpha_f / max(mu, .2))
            f_r = -fmax_r * math.tanh(7.5 * alpha_r / max(mu, .2))
            vy_d = vy + (f_f * math.cos(delta) + f_r - vx * w) * dt
            w_d = w + (f_f * math.cos(delta) - f_r) * (2. / wb) * dt
            # sliding scrubs speed: lateral tyre force times the sine of its slip angle
            speed = max(0., speed - (abs(f_f) * abs(math.sin(alpha_f))
                                    + abs(f_r) * abs(math.sin(alpha_r))) * dt)
            vx = .5 * (old_speed + speed)
            skid_f = dyn * float(np.clip((abs(alpha_f) - .16) / .14, 0., 1.))
            skid_r = dyn * float(np.clip((abs(alpha_r) - .16) / .14, 0., 1.))
            if handbrake:
                skid_r = max(skid_r, float(np.clip((vx - 4.) / 5., 0., 1.)))
        else:
            vy_d, w_d = vy, w
        k = min(1., 12. * dt)
        w_k = w + (w_kin - w) * k
        # ... whose sideways velocity relaxes to the value the dynamic model has in a steady
        # turn (rear axle tracking), so crossing 2.5-5 m/s does not itself cause a slide.
        vy_k = vy + (.5 * wb * w_kin - vy) * k
        w = w_d * dyn + w_k * (1. - dyn)
        vy = vy_d * dyn + vy_k * (1. - dyn)
        yaw = w * dt
        move = vx * dt
        mid = car.heading + yaw * .5
        cm, sm = math.cos(mid), math.sin(mid)
        lateral = vy * dt
        position = car.position + np.array([cm * move - sm * lateral, sm * move + cm * lateral, 0.])
        heading = math.atan2(math.sin(car.heading + yaw), math.cos(car.heading + yaw))
        contact = self._contact(scene, position, heading, car.kind)
        support, ground = contact if contact is not None else (None, None)
        if support is not None:
            position[2] = support[0]
        collision = (support is None or scene.blocked(position, heading, car.kind)
                     or cars_overlap(position, heading, car.kind, obstacles))
        if collision:
            impulse = abs(old_speed) + abs(car.vlat)
            car.crash_impulse = impulse
            car.impact_severity = impulse
            if old_speed > 3.: car.crush_front = min(car.crush_front + impulse * 0.02, 1.0)
            elif old_speed < -3.: car.crush_rear = min(car.crush_rear + impulse * 0.02, 1.0)
            elif car.vlat > 3.: car.crush_left = min(car.crush_left + impulse * 0.02, 1.0)
            elif car.vlat < -3.: car.crush_right = min(car.crush_right + impulse * 0.02, 1.0)
            
            car.speed = -old_speed * 0.3
            car.vlat = -car.vlat * 0.3
            car.yaw_rate = car.yaw_rate * 0.5 + (1.0 - 0.5) * impulse * 0.1
            car.braking, self.blocked = True, True
            decel = -math.copysign(min(abs(old_speed) / dt, 25.), old_speed) if old_speed else 0.
            if car.suspension_ready:
                self._update_suspension(car, decel, 0., car.ground_z, dt)
            return
        car.position, car.heading = position, heading
        car.pitch, car.roll = support[1], support[2]
        car.speed, car.braking = speed, bool(braking)
        car.vlat, car.yaw_rate = vy, w
        blend = min(1., 25. * dt)
        car.skid_front += (skid_f - car.skid_front) * blend
        car.skid_rear += (skid_r - car.skid_rear) * blend
        car.distance += abs(move)
        car.wheel = (car.wheel + move / WHEEL_RADII[car.kind]) % (2 * math.pi)
        ground = np.asarray(ground, np.float64)
        if (not car.suspension_ready or abs(car.body_z - support[0]) > .8
                or abs(car.body_pitch - support[1]) > .6 or abs(car.body_roll - support[2]) > .6):
            self._begin_suspension(car, support, ground)    # first step, or a teleport / overpass jump
        else:
            self._update_suspension(car, (speed - old_speed) / dt, yaw / dt, ground, dt)

    def frame(self, traffic):
        """Merge player poses without mutating worker arrays, clocks or signals."""
        if not self.cars:
            return traffic
        arrays = list(traffic) if traffic is not None else []
        arrays.extend([None] * max(0, len(LENGTHS) - len(arrays)))
        pose = self.render_pose()
        for kind in range(len(LENGTHS)):
            own = [car.instance(pose if car is self.car else None)
                   for car in self.cars if car.kind == kind]
            if not own:
                continue
            rows = np.stack(own)
            if arrays[kind] is not None and len(arrays[kind]):
                rows = np.concatenate((normalize_instances(arrays[kind]), rows))
            arrays[kind] = np.ascontiguousarray(rows)
        # This is a render-only view. AI identity/odometer side tables are not
        # copied onto a differently-sized stream or fed back into presentation.
        return VehicleFrame(arrays, clock=getattr(traffic, 'clock', self.clock),
                            generation=getattr(traffic, 'generation', 0),
                            sequence=getattr(traffic, 'sequence', 0),
                            counts=getattr(traffic, 'counts', None), signals=getattr(traffic, 'signals', None))
