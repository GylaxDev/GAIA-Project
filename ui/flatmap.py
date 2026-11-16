"""Flat Web Mercator map for the map screen's 2D mode.

Tiles are drawn straight through the 2D overlay (the same inverted tile
shading the globe uses for its recolour), with parent tiles standing in while
the requested level is still downloading.
"""
from __future__ import annotations

import math

from ..core.config import ACCENT
from ..render.overlay import make_texture


class TileTextures:
    """Lazily turns downloaded tile pixels into GL textures."""

    def __init__(self, tiles, limit=200):
        import OpenGL.GL as gl
        self.gl, self.tiles, self.limit = gl, tiles, max(32, limit)
        self.tex = {}

    def get(self, key):
        tex = self.tex.get(key)
        if tex is not None:
            return tex
        data = self.tiles.px.get(key)
        if data is None:
            self.tiles.request(key)
            return None
        w, h, raw = data
        tex = make_texture(self.gl, w, h, raw)
        self.tex[key] = tex
        while len(self.tex) > self.limit:
            self.gl.glDeleteTextures([self.tex.pop(next(iter(self.tex)))])
        return tex


class FlatMap:
    """North-up Mercator map: pan, cursor-anchored zoom and fly-to."""

    T = 256
    scale = 1.0
    MIN_ZOOM, MAX_ZOOM = 1.0, 17.5

    def __init__(self, tiles):
        self.tiles = tiles
        self.texs = TileTextures(tiles)
        self.zoom = self.zoom_t = 4.2
        self.cx = self.cx_t = 0.5
        self.cy = self.cy_t = 0.40
        self.flying = False
        self.W, self.H = 1, 1
        self._clamp()

    # ------------------------------------------------------------- projection
    @staticmethod
    def to_world(lat, lon):
        s = math.sin(math.radians(max(-85.0511, min(85.0511, lat))))
        return ((lon + 180.0) / 360.0,
                0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi))

    @staticmethod
    def from_world(x, y):
        y = max(0.0, min(1.0, y))
        return (math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y)))),
                x * 360.0 - 180.0)

    @property
    def lat(self):
        return self.from_world(self.cx, self.cy)[0]

    @property
    def lon(self):
        return self.from_world(self.cx, self.cy)[1]

    def ppw(self):
        return self.T * 2.0 ** self.zoom

    def screen_to_latlon(self, sx, sy):
        p = self.ppw()
        return self.from_world(self.cx + (sx - self.W / 2) / p,
                               self.cy + (sy - self.H / 2) / p)

    def pick(self, sx, sy):
        if not (0 <= sx <= self.W and 0 <= sy <= self.H):
            return None
        return self.screen_to_latlon(sx, sy)

    def project(self, lat, lon):
        x, y = self.to_world(lat, lon)
        p = self.ppw()
        sx = (x - self.cx) * p + self.W / 2
        sy = (y - self.cy) * p + self.H / 2
        return sx, sy, (-80 <= sx <= self.W + 80 and -80 <= sy <= self.H + 80)

    def city_near(self, sx, sy, cities, px=13.0):
        best = None
        for name, lat, lon in cities:
            hit = self.project(lat, lon)
            if not hit[2]:
                continue
            d = math.hypot(hit[0] - sx, hit[1] - sy)
            if d <= px and (best is None or d < best[0]):
                best = (d, name, lat, lon)
        return None if best is None else (best[1], best[2], best[3])

    # ------------------------------------------------------------------- view
    def set_size(self, w, h):
        self.W, self.H = max(int(w), 1), max(int(h), 1)
        self._clamp()

    def _clamp(self):
        zmin = math.log2(max(self.W, self.H, self.T) / self.T)
        self.zoom = max(zmin, min(self.MAX_ZOOM, self.zoom))
        self.zoom_t = max(zmin, min(self.MAX_ZOOM, self.zoom_t))
        half = (self.H / 2) / self.ppw()
        self.cy = 0.5 if half >= 0.5 else max(half, min(1 - half, self.cy))
        self.cx = max(0.0, min(1.0, self.cx))

    def _sync(self):
        self.zoom_t, self.cx_t, self.cy_t, self.flying = self.zoom, self.cx, self.cy, False

    def pan(self, dx, dy):
        p = self.ppw()
        self.cx -= dx / p
        self.cy -= dy / p
        self._clamp()
        self._sync()

    def end_drag(self):
        self._sync()

    def zoom_by(self, steps, sx=None, sy=None):
        if sx is None:
            sx, sy = self.W / 2, self.H / 2
        p = self.ppw()
        wx, wy = self.cx + (sx - self.W / 2) / p, self.cy + (sy - self.H / 2) / p
        self.zoom += steps * 0.4
        self._clamp()
        p = self.ppw()
        self.cx, self.cy = wx - (sx - self.W / 2) / p, wy - (sy - self.H / 2) / p
        self._clamp()
        self._sync()

    def zoom_step(self, sign):
        self.zoom_by(sign)

    def jump_to(self, lat, lon, zoom=None):
        self.cx, self.cy = self.to_world(lat, lon)
        if zoom is not None:
            self.zoom = zoom
        self._clamp()
        self._sync()

    def fly_to(self, lat, lon, zoom=None, dur=0.9):
        self.cx_t, self.cy_t = self.to_world(lat, lon)
        self.zoom_t = float(self.zoom if zoom is None else zoom)
        self.flying = True

    def update(self, dt):
        if not self.flying:
            return
        k = 1 - math.exp(-4.5 * dt)
        self.zoom += (self.zoom_t - self.zoom) * k
        self.cx += (self.cx_t - self.cx) * k
        self.cy += (self.cy_t - self.cy) * k
        self._clamp()
        if (abs(self.zoom_t - self.zoom) < 0.02
                and abs(self.cx_t - self.cx) * self.ppw() < 0.5
                and abs(self.cy_t - self.cy) * self.ppw() < 0.5):
            self._sync()

    def compass_reset(self):
        pass

    # ---------------------------------------------------------------- drawing
    def draw(self, ov, invert=1.0, background=(0.545, 0.640, 0.470, 1.0)):
        ov.rect(0, 0, self.W, self.H, background)
        W, H, p = self.W, self.H, self.ppw()
        z = int(max(0, min(17, round(self.zoom))))
        n = 1 << z
        size = self.T * 2.0 ** (self.zoom - z)
        tx0 = max(0, int((self.cx - W / 2 / p) * n))
        tx1 = min(n - 1, int((self.cx + W / 2 / p) * n))
        ty0 = max(0, int((self.cy - H / 2 / p) * n))
        ty1 = min(n - 1, int((self.cy + H / 2 / p) * n))
        drawn = 0
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                px = (tx / n - self.cx) * p + W / 2
                py = (ty / n - self.cy) * p + H / 2
                S = self.scale
                px, py = round(px * S) / S, round(py * S) / S      # whole device pixels: no half-texel blur
                tex = self.texs.get((z, tx, ty))
                if tex is not None:
                    ov.tile(tex, px, py, size + 0.6, (0, 0, 1, 1), invert)
                    drawn += 1
                    continue
                for k in range(1, 6):
                    if z - k < 0:
                        break
                    parent = self.texs.get((z - k, tx >> k, ty >> k))
                    if parent is not None:
                        f = 1 << k
                        u0, v0 = (tx & (f - 1)) / f, (ty & (f - 1)) / f
                        ov.tile(parent, px, py, size + 0.6,
                                (u0, v0, u0 + 1 / f, v0 + 1 / f), invert)
                        drawn += 1
                        break
        if drawn == 0:
            self._graticule(ov)
        return drawn

    def _graticule(self, ov):
        ppd = self.ppw() / 360.0
        step = next((s for s in (0.1, 0.5, 1, 2, 5, 10, 15, 30) if s * ppd >= 70), 30)
        col = (0.30, 0.40, 0.32, 0.32)
        lat_top, lon_left = self.screen_to_latlon(0, 0)
        lat_bot, lon_right = self.screen_to_latlon(self.W, self.H)
        lon = math.floor(lon_left / step) * step
        while lon <= lon_right:
            x, _ = self.project(0.0, lon)[:2]
            ov.line(x, 0, x, self.H, 1.0, col)
            lon += step
        lat = max(-84.0, math.floor(lat_bot / step) * step)
        while lat <= min(84.0, lat_top):
            _, y = self.project(lat, 0.0)[:2]
            ov.line(0, y, self.W, y, 1.0, col)
            lat += step
        # Only claim "loading" while tiles are really on their way; an offline
        # map would otherwise say it forever (MapScreen shows that banner instead).
        if getattr(self.tiles, "pending", None):
            ov.text_c("Map data loading\u2026", self.W / 2, self.H / 2, 15,
                      (1.0, 1.0, 1.0, 0.75), anchor="center", weight="medium", halo=True)
