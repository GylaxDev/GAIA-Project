"""Orbit / pan / zoom camera with ground picking."""
from __future__ import annotations
import math
import numpy as np


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Camera:
    def __init__(self, scene_radius, home_distance=None):
        self.scene_radius = max(scene_radius, 50.0)
        self.home_distance = home_distance
        self._fly = None
        self.reset()

    def reset(self):
        self.target = np.zeros(3, np.float64)
        self._fly = None
        self.distance = self.home_distance or self.scene_radius * 1.5
        self.azimuth = math.radians(-50)
        self.elevation = math.radians(35)
        self._base = (self.distance, self.azimuth, self.elevation)
        self.arr_t = 1.0

    def begin_arrival(self):
        self.reset(); self.arr_t = 0.0; self.update(0.0)

    def fly_to(self, point):
        end = np.array([point[0], point[1], 0.0], np.float64)
        dur = float(np.clip(np.linalg.norm(end - self.target) / 1500.0 + 0.6, 0.6, 2.5))
        self._fly = (self.target.copy(), end, 0.0, dur)

    def update(self, dt):
        if self._fly is not None:
            start, end, t, dur = self._fly
            t = min(1.0, t + dt / dur)
            self.target = start + (end - start) * (t * t * (3 - 2 * t))
            self._fly = None if t >= 1.0 else (start, end, t, dur)
        if self.arr_t >= 1.0: return
        self.arr_t = min(1.0, self.arr_t + dt / 3.0)
        e = 1 - (1 - self.arr_t) ** 3
        d, a, el = self._base
        self.distance = d * (1 + 4.0 * (1 - e))
        self.azimuth = a - 1.4 * (1 - e)
        self.elevation = el + (math.radians(78) - el) * (1 - e)

    @property
    def position(self):
        ce = math.cos(self.elevation)
        return self.target + self.distance * np.array(
            [ce * math.cos(self.azimuth), ce * math.sin(self.azimuth), math.sin(self.elevation)])

    def orbit(self, dx, dy):
        self.arr_t = 1.0
        self.azimuth -= dx * 0.005
        self.elevation = float(np.clip(self.elevation + dy * 0.005,
                                       math.radians(3), math.radians(89)))

    def pan(self, dx, dy):
        self.arr_t = 1.0; self._fly = None
        right = np.array([-math.sin(self.azimuth), math.cos(self.azimuth), 0.0])
        fwd = np.array([-math.cos(self.azimuth), -math.sin(self.azimuth), 0.0])
        s = self.distance * 0.0015
        self.target += (-right * dx + fwd * dy) * s

    def zoom(self, scroll):
        self.arr_t = 1.0
        self.distance = float(np.clip(self.distance * 0.88 ** scroll, 5.0,
                                      self.scene_radius * 12))

    def view(self):
        return look_at(self.position, self.target, np.array([0.0, 0.0, 1.0]))

    def projection(self, aspect):
        # Depth precision is dominated by far/near, so keep both as tight as the scene allows.
        # Everything beyond fog_far (= distance + 0.9 * radius) is already pure fog colour.
        near = max(1.0, self.distance * 0.02)
        far = self.distance + self.scene_radius * 3.0
        return perspective(55.0, aspect, near, far)

    def ground_hit(self, sx, sy, w, h):
        inv = np.linalg.inv((self.projection(w / h) @ self.view()).astype(np.float64))
        def unproject(ndc_z):
            p = inv @ np.array([2 * sx / w - 1, 1 - 2 * sy / h, ndc_z, 1.0])
            return p[:3] / p[3]
        o = unproject(-1.0); d = unproject(1.0) - o
        if d[2] >= -1e-9: return None
        return o + d * (-o[2] / d[2])