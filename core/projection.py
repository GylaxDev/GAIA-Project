"""Local projection and tile grid."""
from __future__ import annotations
import math
import numpy as np

from .config import EARTH_RADIUS


class LocalProjection:
    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self.cos_lat0 = math.cos(math.radians(lat0))

    def to_local(self, lat, lon) -> np.ndarray:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        x = EARTH_RADIUS * np.radians(lon - self.lon0) * self.cos_lat0
        y = EARTH_RADIUS * np.radians(lat - self.lat0)
        return np.stack([x, y], axis=-1)

    def to_latlon(self, x, y) -> tuple[float, float]:
        lat = self.lat0 + math.degrees(y / EARTH_RADIUS)
        lon = self.lon0 + math.degrees(x / (EARTH_RADIUS * self.cos_lat0))
        return lat, lon


def lonlat_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 1 << z
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 1 << z
    def lat_at(row):
        return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * row / n))))
    return lat_at(y + 1), x / n * 360.0 - 180.0, lat_at(y), (x + 1) / n * 360.0 - 180.0


def tile_d2(a, b) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class ChunkGrid:
    def __init__(self, proj: LocalProjection, zoom: int = 14):
        self.proj, self.zoom = proj, zoom
        self._proj_cache: dict = {}

    def key_at(self, x, y):
        lat, lon = self.proj.to_latlon(x, y)
        return lonlat_to_tile(lat, lon, self.zoom)

    def bounds(self, key):
        return tile_bounds(self.zoom, key[0], key[1])

    def center(self, key):
        s, w, n, e = self.bounds(key)
        return (s + n) / 2, (w + e) / 2

    def origin(self, key):
        s, w, n, e = self.bounds(key)
        c = self.proj.to_local((s + n) / 2, (w + e) / 2)
        return float(c[0]), float(c[1])

    def tile_projection(self, key) -> LocalProjection:
        p = self._proj_cache.get(key)
        if p is None:
            lat, lon = self.center(key)
            p = self._proj_cache[key] = LocalProjection(lat, lon)
        return p

    def x_scale(self, key) -> float:
        lat, _ = self.center(key)
        c = math.cos(math.radians(lat))
        return float(self.proj.cos_lat0 / c) if abs(c) > 1e-6 else 1.0

    def size_m(self, key):
        s, w, n, e = self.bounds(key)
        sw, ne = self.proj.to_local(s, w), self.proj.to_local(n, e)
        return float(ne[0] - sw[0]), float(ne[1] - sw[1])

    def keys_within(self, center, r2):
        r, n = int(math.ceil(math.sqrt(r2))), 1 << self.zoom
        found = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                d2 = dx * dx + dy * dy
                x, y = center[0] + dx, center[1] + dy
                if d2 <= r2 and 0 <= x < n and 0 <= y < n:
                    found.append((d2, (x, y)))
        found.sort()
        return [k for _, k in found]