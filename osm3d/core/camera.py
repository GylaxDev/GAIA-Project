"""Camera rig: free-fly, spring orbit, car chase and cinematic focus.

Two control families share one yaw/pitch rig:

``fly``    WASD/QE move the eye along its own basis, the view pokes around a
           target that always rides ahead of the eye. Residency and the shadow
           box follow the eye, not the target's movement when looking around.
``orbit``  the eye rides a sphere around a pivot; dragging spins it with flick
           inertia, the wheel changes the radius.

Every axis is driven by two cascaded exponential dampers, which is what gives
the rig weight: an input first feeds a smoothed *goal* (velocity for fly, angle
for orbit) and the visible value chases that goal at its own rate.  Both use
``1 - exp(-rate * dt)`` so the response is identical at 30 or 240 fps, unlike a
raw ``mix(a, b, k * dt)`` which changes behaviour with the frame rate.

Pitch is clamped to (-89, +89) degrees and the view basis is rebuilt from the
clamped yaw/pitch with orthonormal cross products, so the up vector can never
collapse and the rig has no gimbal singularity.
"""
from __future__ import annotations

import math

import numpy as np

PITCH_LIMIT = math.radians(89.0)      # never quite vertical: no degenerate basis
ORBIT_PITCH_MIN = math.radians(1.0)   # the orbit eye stays above its pivot


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
    n = float(np.linalg.norm(f))
    if n < 1e-6:
        f = np.array([1.0, 0.0, 0.0])
    else:
        f = f / n
    s = np.cross(f, up)
    ns = float(np.linalg.norm(s))
    if ns < 1e-6:                                  # view parallel to `up`
        s = np.cross(f, np.array([0.0, 1.0, 0.0]))
        ns = float(np.linalg.norm(s)) or 1.0
    s = s / ns
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.asarray(eye, np.float64)
    return m


def damp(current, goal, rate, dt):
    """Frame-rate independent exponential smoothing of a scalar."""
    return goal + (current - goal) * math.exp(-rate * dt)


def damp3(current, goal, rate, dt):
    """Same smoother for a vector, in place-free form."""
    k = math.exp(-rate * dt)
    return goal + (current - goal) * k


def spring(x, v, goal, omega, dt):
    """Exact critically damped spring step -> (x, v).

    Unlike a plain exponential, the velocity is carried between frames, so a
    moving goal is tracked without the velocity steps that read as judder.
    """
    d = x - goal
    e = math.exp(-omega * dt)
    t = v + omega * d
    return goal + (d + t * dt) * e, (v - omega * t * dt) * e


def smootherstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Camera:
    ORBIT, FLY, DRIVE, WALK = "orbit", "fly", "drive", "walk"

    # damper rates (1/s).  Higher = snappier, lower = more cinematic weight.
    ORBIT_RATE = 16.0       # actual angle -> goal angle
    GLIDE_DECAY = 4.5       # flick inertia after the mouse is released
    DIST_RATE = 9.0         # orbit radius -> goal radius
    PAN_RATE = 12.0         # pivot -> pan goal
    ACCEL_RATE = 11.0       # fly stage 1: input -> desired velocity
    DRAG_RATE = 8.0         # fly stage 2: velocity -> applied motion

    FOCUS_TIME = 1.3        # seconds for a focus swoop

    # chase rig (GTA IV style): low and close, swings out in turns, widens with speed
    BASE_FOV = 55.0
    FOV_BOOST = 16.0        # extra degrees at full speed
    FOV_FULL_SPEED = 38.0   # m/s at which the boost is complete
    CHASE_DIST = 6.2        # metres behind the pivot (scaled by vehicle length)
    CHASE_ELEV = math.radians(11.0)

    # on-foot rig (over-the-shoulder): the character sits left of centre
    WALK_DIST = 2.9
    WALK_ELEV = math.radians(9.0)
    WALK_SHOULDER = 0.55    # metres the rig is shifted to the character's right
    WALK_PIVOT = 0.80       # fraction of body height the look target rides at

    def __init__(self, scene_radius, home_distance=None):
        self.scene_radius = max(scene_radius, 50.0)
        self.home_distance = home_distance
        self.speed = 60.0               # free-fly base speed (m/s, wheel adjusts)
        self.sprint, self.crawl = 4.0, 0.22
        self._last_dt = 1.0 / 60.0
        self.fov_effect = True          # speed-dependent field of view (settings toggle)
        self.sensitivity = 1.0          # mouse-look multiplier (settings)
        self.invert_y = False           # settings: move the mouse up to look down
        self._fov = self.BASE_FOV
        self._fov_k = 0.0
        self.collide = None             # fn(origin, direction, max_distance) -> free distance
        self._glide = np.zeros(2)
        self._pan_vel = np.zeros(3)
        self._fly_want = np.zeros(3)
        self._dragging = False
        self._trans = None
        self._focused = False
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.target = np.zeros(3, np.float64)
        self._tgt_goal = self.target.copy()
        self.mode = self.ORBIT
        self._focused = False
        self._trans = None
        self._fly = None
        self.distance = self.home_distance or self.scene_radius * 1.5
        self.azimuth = math.radians(-50)
        self.elevation = math.radians(35)
        self._dist_goal = self.distance
        self._az_goal = self.azimuth
        self._el_goal = self.elevation
        self._base = (self.distance, self.azimuth, self.elevation)
        self.arr_t = 1.0
        self.vel = np.zeros(3)
        self._vel_goal = np.zeros(3)
        self._fly_want = np.zeros(3)
        self._eye = self._orbit_eye()
        self._fly_dist = max(60.0, self.distance)
        self._glide[:] = 0.0
        self._pan_vel[:] = 0.0
        self._drive_position = None
        self._drive_heading = self._drive_speed = self._drive_yaw = 0.0
        self._drive_look_age = 0.0
        self._drive_height, self._drive_len, self._drive_steer = 1.0, 1.0, 0.0
        self._drive_z = 0.0
        self._drive_slip = 0.0
        self._walk_pos = None
        self._walk_speed = 0.0
        self._walk_height = 1.8
        self._walk_z = 0.0
        self._walk_clear = self.WALK_DIST
        self._walk_shoulder = self.WALK_SHOULDER
        self._az_vel = self._el_vel = self._dist_vel = self._z_vel = 0.0

    def begin_arrival(self):
        self.reset()
        self.arr_t = 0.0
        self.update(1.0 / 60.0)

    def fly_to(self, point):
        end = np.array([point[0], point[1], 0.0], np.float64)
        if self.mode == self.FLY:
            # in flight the eye stays put and the view turns onto the point
            d = self._eye - end
            n = float(np.linalg.norm(d)) or 1.0
            self._az_goal = math.atan2(d[1], d[0])
            el = math.asin(float(np.clip(d[2] / n, -1.0, 1.0)))
            self._el_goal = float(np.clip(el, -PITCH_LIMIT, PITCH_LIMIT))
            return
        dur = float(np.clip(np.linalg.norm(end - self.target) / 1500.0 + 0.6, 0.6, 2.5))
        self._fly = (self.target.copy(), end, 0.0, dur)
        self._focused = False
        self._trans = None

    # ------------------------------------------------------------------ basis
    def _offset_dir(self):
        """Unit vector from the pivot / eye toward the eye (the rig's pole)."""
        ce = math.cos(self.elevation)
        return np.array([ce * math.cos(self.azimuth),
                         ce * math.sin(self.azimuth),
                         math.sin(self.elevation)])

    def forward(self):
        """Unit view direction."""
        return -self._offset_dir()

    def _orbit_eye(self):
        return self.target + self.distance * self._offset_dir()

    @property
    def position(self):
        if self._trans is not None or self.mode in (self.FLY, self.DRIVE, self.WALK):
            return self._eye
        return self._orbit_eye()

    def _pitch_limits(self):
        if self.mode == self.FLY:
            return -PITCH_LIMIT, PITCH_LIMIT
        return ORBIT_PITCH_MIN, PITCH_LIMIT

    # ------------------------------------------------------------------ input
    def orbit(self, dx, dy):
        """Look / spin.  Live while dragging, then glides to a stop."""
        self.arr_t = 1.0
        step = 0.005 * self.sensitivity
        if self.invert_y:
            dy = -dy
        if self.mode == self.DRIVE:
            self._drive_yaw -= dx * step
            self._el_goal = float(np.clip(self._el_goal + dy * step, math.radians(2), math.radians(55)))
            self._drive_look_age = 0.0
            return
        if self.mode == self.WALK:
            # direct mouse-look: the player walks relative to this heading
            self._az_goal -= dx * step * 0.8
            self._el_goal = float(np.clip(self._el_goal + dy * step * 0.8, math.radians(-14), math.radians(68)))
            return
        self._az_goal -= dx * step
        lo, hi = self._pitch_limits()
        self._el_goal = float(np.clip(self._el_goal + dy * step, lo, hi))
        # remember the drag speed so releasing the mouse carries a little inertia
        dt = max(self._last_dt, 1e-4)
        self._glide[0] = -dx * step / dt * 0.55
        self._glide[1] = dy * step / dt * 0.55
        self._dragging = True

    def pan(self, dx, dy):
        self.arr_t = 1.0
        self._fly = None
        right = np.array([-math.sin(self.azimuth), math.cos(self.azimuth), 0.0])
        fwd = np.array([-math.cos(self.azimuth), -math.sin(self.azimuth), 0.0])
        delta = (-right * dx + fwd * dy) * (self.distance * 0.0015)
        self._tgt_goal += delta
        self._pan_vel += delta / max(self._last_dt, 1e-4) * 0.5
        self._dragging = True
        if self.mode == self.FLY:
            self._eye += delta
            self.target = self._eye + self.forward() * self._fly_dist

    def zoom(self, scroll):
        """Wheel: orbit radius when orbiting, base speed when flying."""
        self.arr_t = 1.0
        if self.mode == self.FLY:
            self.speed = float(np.clip(self.speed * 1.18 ** scroll, 3.0, 4000.0))
        elif self.mode == self.DRIVE:
            self._dist_goal = float(np.clip(self._dist_goal * 0.88 ** scroll, 3.5, 22.0))
        elif self.mode == self.WALK:
            self._dist_goal = float(np.clip(self._dist_goal * 0.88 ** scroll, 1.4, 7.0))
        else:
            self._dist_goal = float(np.clip(self._dist_goal * 0.88 ** scroll,
                                            5.0, self.scene_radius * 12))

    wheel = zoom

    def fly(self, forward=0.0, strafe=0.0, climb=0.0, boost=1.0):
        """Free-fly input in [-1, 1]; any press takes the rig out of orbit."""
        if self.mode != self.FLY:
            self.enter_fly()
        f = self.forward()
        right = np.array([f[1], -f[0], 0.0])
        n = float(np.linalg.norm(right))
        right = right / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        speed = self.speed * boost
        self._fly_want = (f * forward + right * strafe) * speed
        self._fly_want[2] += climb * speed

    def enter_fly(self):
        self.arr_t = 1.0                     # taking control cancels the arrival swoop
        self._eye = np.asarray(self.position, np.float64).copy()
        self._fly_dist = float(np.clip(np.linalg.norm(self.target - self._eye), 40.0, 400.0))
        self.vel[:] = 0.0
        self._vel_goal[:] = 0.0
        self._fly_want = np.zeros(3)
        self.mode = self.FLY
        self._trans = None
        self._focused = False
        self.distance = self._fly_dist

    # ------------------------------------------------------------------ chase
    def follow_vehicle(self, position, heading, speed=0.0, *, snap=False, height=1.5,
                       steer=0.0, length=4.6, slip=0.0):
        """Set a confirmed player pose. Camera lag never moves the stream anchor."""
        entering = self.mode != self.DRIVE
        self._drive_position = np.asarray(position, np.float64).copy()
        self._drive_heading, self._drive_speed = float(heading), float(speed)
        self._drive_height = max(1., float(height))
        self._drive_len = float(np.clip(float(length) / 4.6, 0.85, 1.4))
        self._drive_slip = float(np.clip(slip, -1.0, 1.0))
        if entering or snap:
            self.mode, self.arr_t = self.DRIVE, 1.0
            self._trans = self._fly = None
            self._focused = False
            self.vel[:] = self._vel_goal[:] = self._fly_want[:] = 0.0
            self._glide[:] = self._pan_vel[:] = 0.0
            self._drive_yaw, self._drive_look_age = 0.0, 0.0
            self._drive_steer = 0.0
            self._drive_z = float(self._drive_position[2])
            self._az_vel = self._el_vel = self._dist_vel = self._z_vel = 0.0
            self.azimuth = self._az_goal = heading + math.pi
            self.elevation = self._el_goal = self.CHASE_ELEV
            self.distance = self._dist_goal = self.CHASE_DIST
            self._fov_k = 0.0
            self.target, self._eye = self._chase_rig()
            self._tgt_goal = self.target.copy()
        else:
            self._drive_steer = damp(self._drive_steer, float(steer), 6.0, self._last_dt)

    # ------------------------------------------------------------------ on foot
    def follow_walker(self, position, heading, speed=0.0, *, snap=False, height=1.8):
        """Third-person rig behind and above the shoulder of a character on foot."""
        entering = self.mode != self.WALK
        self._walk_pos = np.asarray(position, np.float64).copy()
        self._walk_speed = float(speed)
        self._walk_height = max(1.0, float(height))
        if entering or snap:
            self.mode, self.arr_t = self.WALK, 1.0
            self._trans = self._fly = None
            self._focused = False
            self.vel[:] = self._vel_goal[:] = self._fly_want[:] = 0.0
            self._glide[:] = self._pan_vel[:] = 0.0
            self._walk_z = float(self._walk_pos[2])
            self._az_vel = self._el_vel = self._dist_vel = self._z_vel = 0.0
            self.azimuth = self._az_goal = heading + math.pi
            self.elevation = self._el_goal = self.WALK_ELEV
            self.distance = self._dist_goal = self.WALK_DIST
            self._walk_clear = self.WALK_DIST
            self._walk_shoulder = self.WALK_SHOULDER
            self._fov_k = 0.0
            self.target, self._eye = self._walk_rig()
            self._tgt_goal = self.target.copy()

    def _walk_rig(self):
        """Look target and eye from the current spring state (pure, no integration)."""
        pivot = np.array([self._walk_pos[0], self._walk_pos[1],
                          self._walk_z + self._walk_height * self.WALK_PIVOT])
        right = np.array([-math.sin(self.azimuth), math.cos(self.azimuth), 0.0])
        target = pivot + right * self._walk_shoulder
        eye = target + self._walk_clear * self._offset_dir()
        eye[2] = max(float(eye[2]), self._walk_z + 0.35)         # never dip into the pavement
        return target, eye

    def _step_walk(self, dt):
        # mouse-look is direct (no glide): a tight damper only removes frame-to-frame noise
        self.azimuth = damp(self.azimuth, self._az_goal, 26.0, dt)
        self.elevation = damp(self.elevation, self._el_goal, 26.0, dt)
        k = min(abs(self._walk_speed) / 6.2, 1.0)
        self.distance, self._dist_vel = spring(self.distance, self._dist_vel,
                                               self._dist_goal + 0.35 * k, 9.0, dt)
        self._walk_z, self._z_vel = spring(self._walk_z, self._z_vel, float(self._walk_pos[2]), 14.0, dt)
        self._fov_k = k
        # pull the eye in front of any wall between it and the character, ease back out slowly
        target, _ = self._walk_rig()
        want = self.distance
        if self.collide is not None:
            try:
                free = float(self.collide(target, self._offset_dir(), self.distance + 0.3))
                want = max(0.6, min(self.distance, free - 0.3))
            except Exception:
                want = self.distance
        rate = 30.0 if want < self._walk_clear else 3.0
        self._walk_clear = damp(self._walk_clear, want, rate, dt)
        self.target, self._eye = self._walk_rig()
        self._tgt_goal = self.target.copy()

    def recenter_drive(self):
        self._drive_yaw, self._drive_look_age = 0.0, 0.0
        self._el_goal, self._dist_goal = self.CHASE_ELEV, self.CHASE_DIST

    def _chase_rig(self):
        """Look target and eye from the current spring state (pure, no integration)."""
        v = abs(self._drive_speed)
        h = self._drive_height
        fwd = np.array([math.cos(self._drive_heading), math.sin(self._drive_heading), 0.])
        left = np.array([-fwd[1], fwd[0], 0.])
        # Anchored on the centre of the car: the look target IS the car's centre, so the
        # chassis stays pinned mid-frame and only the orbit angles ever lag.
        pivot = np.array([self._drive_position[0], self._drive_position[1], self._drive_z])
        target = pivot + np.array([0., 0., h * 0.55])
        eye = target + self.distance * self._offset_dir()
        eye[2] = max(float(eye[2]), float(pivot[2]) + 0.9)      # never dip into the road
        return target, eye

    def _step_drive(self, dt):
        self._drive_look_age += dt
        if self._drive_look_age > 2.0:
            self._drive_yaw = damp(self._drive_yaw, 0., 1.8, dt)
        v = abs(self._drive_speed)
        k = min(v / 20.0, 1.0)
        # Orbit angle: a critically damped spring on the heading.  Softer at speed, so
        # the camera trails a corner and shows the car's flank, tighter when crawling.
        # While sliding, swing toward the direction of travel so the flank of the car shows (GTA IV).
        yaw = self._drive_heading + math.pi + self._drive_yaw + 0.55 * self._drive_slip
        yaw = self.azimuth + math.atan2(math.sin(yaw - self.azimuth), math.cos(yaw - self.azimuth))
        self.azimuth, self._az_vel = spring(self.azimuth, self._az_vel, yaw, 11.0 - 4.0 * k, dt)
        self._az_goal = self.azimuth
        # Pitch drops and the rig pulls back as the car speeds up.
        el_goal = self._el_goal - math.radians(2.5) * k
        self.elevation, self._el_vel = spring(self.elevation, self._el_vel, el_goal, 10.0, dt)
        dist_goal = (self._dist_goal * self._drive_len) + min(v * 0.04, 1.2)
        self.distance, self._dist_vel = spring(self.distance, self._dist_vel, dist_goal, 10.0, dt)
        # The eye is locked to the (already interpolated) car pose in x/y: lag exists only
        # in the angles, so frame-time noise cannot change the camera-to-car distance.
        # Height is softened so kerbs and suspension bounce do not bob the view.
        self._drive_z, self._z_vel = spring(self._drive_z, self._z_vel,
                                            float(self._drive_position[2]), 12.0, dt)
        self._fov_k = min(v / self.FOV_FULL_SPEED, 1.0)
        self.target, self._eye = self._chase_rig()
        self._tgt_goal = self.target.copy()

    # ------------------------------------------------------------------ focus
    @property
    def focused(self):
        return self._focused or self._trans is not None

    def focus_on(self, anchor, radius=None, dur=None):
        """Swoop to `anchor` over ~1.3 s, then spring-orbit around it."""
        anchor = np.asarray(anchor, np.float64)
        eye0 = np.asarray(self.position, np.float64).copy()
        tgt0 = np.asarray(self.target, np.float64).copy()
        if radius is None:
            radius = float(np.clip(np.linalg.norm(anchor - eye0) * 0.45, 14.0, 95.0))
        d = eye0 - anchor
        if math.hypot(d[0], d[1]) < 1e-3:
            d = np.array([-1.0, -0.7, 0.35])
        el = float(np.clip(math.atan2(d[2], math.hypot(d[0], d[1])),
                           math.radians(10.0), math.radians(52.0)))
        az = math.atan2(d[1], d[0])
        ce = math.cos(el)
        eye1 = anchor + float(radius) * np.array(
            [ce * math.cos(az), ce * math.sin(az), math.sin(el)])
        self._trans = (eye0, eye1, tgt0, anchor.copy(), 0.0,
                       dur or self.FOCUS_TIME)
        self.arr_t = 1.0
        self._focused = True
        self._glide[:] = 0.0
        self._pan_vel[:] = 0.0

    def release(self):
        """Drop the focus anchor and keep flying from where the orbit ended."""
        if self.mode in (self.ORBIT, self.DRIVE, self.WALK):
            self.enter_fly()
        self._focused = False
        self._trans = None

    # ----------------------------------------------------------------- update
    def update(self, dt):
        dt = float(dt)
        if not (dt > 1e-6):
            dt = 1e-4
        self._last_dt = dt = min(dt, 0.1)
        if self._trans is not None:
            self._step_focus(dt)
        elif self.mode == self.DRIVE:
            self._step_drive(dt)
        elif self.mode == self.WALK:
            self._step_walk(dt)
        elif self.mode == self.FLY:
            self._step_fly(dt)
        else:
            if self.arr_t < 1.0:
                self._step_arrival(dt)
            self._step_orbit(dt)
        self._dragging = False
        # Speed FOV: widen with speed while chasing; relax back (also when the switch is off).
        boost = self.FOV_BOOST * smootherstep(self._fov_k) if (self.mode == self.DRIVE and self.fov_effect) else 0.0
        if self.mode == self.WALK and self.fov_effect:
            boost = 9.0 * smootherstep(self._fov_k)      # a little rush when sprinting
        self._fov = damp(self._fov, self.BASE_FOV + boost, 3.5, dt)

    def _apply_angles(self, dt):
        lo, hi = self._pitch_limits()
        if not self._dragging and (self._glide[0] or self._glide[1]):
            self._glide *= math.exp(-self.GLIDE_DECAY * dt)
            self._az_goal += float(self._glide[0]) * dt
            self._el_goal = float(np.clip(self._el_goal + float(self._glide[1]) * dt, lo, hi))
        self.azimuth = damp(self.azimuth, self._az_goal, self.ORBIT_RATE, dt)
        self.elevation = damp(self.elevation, self._el_goal, self.ORBIT_RATE, dt)

    def _step_orbit(self, dt):
        if self._fly is not None:
            start, end, t, dur = self._fly
            t = min(1.0, t + dt / dur)
            self.target = start + (end - start) * (t * t * (3 - 2 * t))
            self._tgt_goal = self.target.copy()
            self._fly = None if t >= 1.0 else (start, end, t, dur)
        self._apply_angles(dt)
        self.distance = damp(self.distance, self._dist_goal, self.DIST_RATE, dt)
        if self._pan_vel.any():
            self._tgt_goal += self._pan_vel * dt
            self._pan_vel *= math.exp(-6.0 * dt)
        self.target = damp3(self.target, self._tgt_goal, self.PAN_RATE, dt)

    def _step_fly(self, dt):
        self._apply_angles(dt)
        self._vel_goal = damp3(self._vel_goal, self._fly_want, self.ACCEL_RATE, dt)
        self.vel = damp3(self.vel, self._vel_goal, self.DRAG_RATE, dt)
        self._eye = self._eye + self.vel * dt
        self._eye[2] = max(float(self._eye[2]), 1.8)     # never below street level
        self.target = self._eye + self.forward() * self._fly_dist
        self._tgt_goal = self.target.copy()
        self.distance = self._fly_dist

    def _step_focus(self, dt):
        eye0, eye1, tgt0, tgt1, t, dur = self._trans
        t = min(1.0, t + dt / dur)
        e = smootherstep(t)
        self._eye = eye0 + (eye1 - eye0) * e
        self.target = tgt0 + (tgt1 - tgt0) * e
        self._tgt_goal = self.target.copy()
        self._trans = (eye0, eye1, tgt0, tgt1, t, dur)
        if t >= 1.0:
            self._trans = None
            d = eye1 - tgt1
            dist = float(np.linalg.norm(d))
            dh = math.hypot(d[0], d[1])
            self.mode = self.ORBIT
            self.distance = self._dist_goal = max(dist, 1.0)
            self.azimuth = self._az_goal = math.atan2(d[1], d[0])
            self.elevation = self._el_goal = float(np.clip(
                math.atan2(d[2], max(dh, 1e-6)), ORBIT_PITCH_MIN, PITCH_LIMIT))

    def _step_arrival(self, dt):
        self.arr_t = min(1.0, self.arr_t + dt / 3.0)
        e = 1 - (1 - self.arr_t) ** 3
        d, a, el = self._base
        self.distance = d * (1 + 4.0 * (1 - e))
        self.azimuth = a - 1.4 * (1 - e)
        self.elevation = el + (math.radians(78) - el) * (1 - e)
        self._dist_goal, self._az_goal, self._el_goal = (self.distance, self.azimuth,
                                                        self.elevation)

    # ------------------------------------------------------------------- view
    def view(self):
        return look_at(self.position, self.target, np.array([0.0, 0.0, 1.0]))

    def projection(self, aspect, reverse_z=False):
        # Fly focus distance is not clearance from nearby walls/cars.
        near = 0.20
        # Everything fades to fog by distance + stream radius, and the light field reaches
        # 2.2 km; a 50 km far plane only burned depth precision (z-fighting at range).
        far = max(self.distance + self.scene_radius * 1.5, 3000.0)
        p = perspective(self._fov, aspect, near, far)
        if reverse_z:
            # Zero-to-one clip space + float depth: millimetre-scale precision
            # across the city, without pushing the near plane through the street.
            p[2, 2] = near / (far - near)
            p[2, 3] = far * near / (far - near)
        return p

    def ray(self, sx, sy, w, h):
        """World-space pick ray through a screen pixel."""
        inv = np.linalg.inv((self.projection(w / h) @ self.view()).astype(np.float64))

        def unproject(ndc_z):
            p = inv @ np.array([2 * sx / w - 1, 1 - 2 * sy / h, ndc_z, 1.0])
            return p[:3] / p[3]

        o = unproject(-1.0)
        d = unproject(1.0) - o
        n = float(np.linalg.norm(d))
        return o, (d / n if n > 1e-9 else np.array([1.0, 0.0, 0.0]))

    def ground_hit(self, sx, sy, w, h):
        o, d = self.ray(sx, sy, w, h)
        if d[2] >= -1e-9:
            return None
        return o + d * (-o[2] / d[2])
