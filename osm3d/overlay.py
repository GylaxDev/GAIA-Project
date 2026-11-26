"""2D overlay, tile cache, world map."""
from __future__ import annotations

import ctypes
import hashlib
import math
import queue
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import ACCENT, CITIES
from .shaders import OVERLAY_VS, OVERLAY_FS


def make_texture(gl, w, h, rgba):
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE, rgba)
    return tex


def _quad(x, y, w, h, u0=0.0, v0=0.0, u1=1.0, v1=1.0):
    return np.array([[x, y, u0, v0], [x + w, y, u1, v0], [x + w, y + h, u1, v1],
                     [x, y, u0, v0], [x + w, y + h, u1, v1], [x, y + h, u0, v1]], np.float32)


class Overlay2D:
    def __init__(self):
        import OpenGL.GL as gl
        import pygame
        from OpenGL.GL import shaders
        self.gl, self.pg = gl, pygame
        self.prog = shaders.compileProgram(
            shaders.compileShader(OVERLAY_VS, gl.GL_VERTEX_SHADER),
            shaders.compileShader(OVERLAY_FS, gl.GL_FRAGMENT_SHADER), validate=False)
        self.u = {n: gl.glGetUniformLocation(self.prog, n)
                  for n in ("uScreen", "uMode", "uColor", "uInvert", "uTex")}
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        for loc, off in ((0, 0), (1, 8)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(off))
        gl.glBindVertexArray(0)
        self.fonts, self.text_cache = {}, {}

    def begin(self, w, h):
        gl = self.gl
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND); gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.prog)
        gl.glUniform2f(self.u["uScreen"], float(w), float(h))
        gl.glUniform1i(self.u["uTex"], 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self.vao)

    def end(self):
        self.gl.glDisable(self.gl.GL_BLEND)
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glBindVertexArray(0)

    def _set(self, mode, color=(1, 1, 1, 1), invert=0.0):
        gl = self.gl
        gl.glUniform1i(self.u["uMode"], mode)
        gl.glUniform4f(self.u["uColor"], *[float(c) for c in color])
        gl.glUniform1f(self.u["uInvert"], float(invert))

    def _submit(self, verts, prim):
        gl = self.gl
        data = np.ascontiguousarray(verts, np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STREAM_DRAW)
        gl.glDrawArrays(prim, 0, len(data))

    def rect(self, x, y, w, h, color):
        self._set(0, color); self._submit(_quad(x, y, w, h), self.gl.GL_TRIANGLES)

    def line(self, x0, y0, x1, y1, width, color):
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln * width / 2, dx / ln * width / 2
        v = np.array([[x0 + nx, y0 + ny, 0, 0], [x0 - nx, y0 - ny, 0, 0],
                      [x1 + nx, y1 + ny, 0, 0], [x1 - nx, y1 - ny, 0, 0]], np.float32)
        self._set(0, color); self._submit(v, self.gl.GL_TRIANGLE_STRIP)

    def circle(self, cx, cy, r, color, seg=28):
        a = np.linspace(0, 2 * math.pi, seg + 1)
        pts = np.stack([cx + r * np.cos(a), cy + r * np.sin(a),
                        np.zeros_like(a), np.zeros_like(a)], 1)
        self._set(0, color)
        self._submit(np.vstack([[cx, cy, 0, 0], pts]), self.gl.GL_TRIANGLE_FAN)

    def ring(self, cx, cy, r, thick, color, seg=36):
        a = np.linspace(0, 2 * math.pi, seg + 1)
        out = np.stack([cx + r * np.cos(a), cy + r * np.sin(a), 0 * a, 0 * a], 1)
        inn = np.stack([cx + (r - thick) * np.cos(a), cy + (r - thick) * np.sin(a),
                        0 * a, 0 * a], 1)
        strip = np.empty((2 * len(a), 4), np.float32)
        strip[0::2], strip[1::2] = out, inn
        self._set(0, color); self._submit(strip, self.gl.GL_TRIANGLE_STRIP)

    # ---- small widgets used by the HUD
    def panel(self, x, y, w, h, accent=ACCENT, bg=(0.030, 0.040, 0.062, 0.90)):
        """Card with a faint border and a coloured top edge."""
        self.rect(x - 1, y - 1, w + 2, h + 2, (1, 1, 1, 0.08))
        self.rect(x, y, w, h, bg)
        if accent is not None:
            self.rect(x, y, w, 2, (accent[0], accent[1], accent[2], 0.95))

    def bar(self, x, y, w, h, frac, color=ACCENT, shimmer=0.0):
        """Progress bar. shimmer > 0 sweeps a highlight over the fill (activity indicator)."""
        frac = max(0.0, min(1.0, float(frac)))
        self.rect(x, y, w, h, (1, 1, 1, 0.10))
        fw = w * frac
        if fw <= 0.5: return
        self.rect(x, y, fw, h, (color[0], color[1], color[2], 0.95))
        if shimmer > 0.0 and fw > 8:
            band = min(70.0, fw)
            bx = x + ((shimmer * 140.0) % (fw + band)) - band
            x0, x1 = max(bx, x), min(bx + band, x + fw)
            if x1 > x0: self.rect(x0, y, x1 - x0, h, (1, 1, 1, 0.28))

    def spinner(self, cx, cy, r, t, color=ACCENT, dots=10):
        for i in range(dots):
            a = t * 6.0 - i * 2.0 * math.pi / dots
            fade = 1.0 - i / dots
            self.circle(cx + r * math.cos(a), cy + r * math.sin(a),
                        1.2 + 1.6 * fade, (color[0], color[1], color[2], 0.15 + 0.85 * fade), 10)

    def tick(self, cx, cy, s, color, width=2.2):
        self.line(cx - s * 0.55, cy + s * 0.05, cx - s * 0.15, cy + s * 0.45, width, color)
        self.line(cx - s * 0.15, cy + s * 0.45, cx + s * 0.60, cy - s * 0.40, width, color)

    def tile(self, tex, x, y, size, uv=(0, 0, 1, 1), invert=1.0, alpha=1.0):
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
        self._set(2, (1, 1, 1, alpha), invert)
        self._submit(_quad(x, y, size, size, *uv), self.gl.GL_TRIANGLES)

    def _text_tex(self, s, size):
        key = (s, size)
        if key in self.text_cache: return self.text_cache[key]
        if len(self.text_cache) > 300:
            self.gl.glDeleteTextures([t[0] for t in self.text_cache.values()])
            self.text_cache.clear()
        font = self.fonts.get(size) or self.fonts.setdefault(
            size, self.pg.font.Font(None, size))
        surf = font.render(s, True, (255, 255, 255)).convert_alpha()
        w, h = surf.get_size()
        tex = make_texture(self.gl, w, h, self.pg.image.tostring(surf, "RGBA", False))
        self.text_cache[key] = (tex, w, h)
        return self.text_cache[key]

    def text(self, s, x, y, size=20, color=(1, 1, 1, 1), anchor="left", shadow=True):
        if not s: return 0
        tex, w, h = self._text_tex(s, size)
        if anchor == "center": x -= w / 2
        elif anchor == "right": x -= w
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
        if shadow:
            self._set(1, (0, 0, 0, 0.8 * color[3]))
            self._submit(_quad(x + 1, y + 1, w, h), self.gl.GL_TRIANGLES)
        self._set(1, color)
        self._submit(_quad(x, y, w, h), self.gl.GL_TRIANGLES)
        return w


class TileCache:
    UA = {"User-Agent": "osm3d-viewer/1.0 (educational PyOpenGL demo)"}

    def __init__(self, url, cache_dir, workers=4):
        import OpenGL.GL as gl
        import pygame
        self.gl, self.pg, self.url = gl, pygame, url
        self.dir = Path(cache_dir) / "tiles" / hashlib.sha1(url.encode()).hexdigest()[:8]
        self.tex = {}; self.pending = set(); self.failed = {}
        self.jobs = queue.Queue(); self.done = queue.Queue()
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()

    @property
    def offline(self): return bool(self.failed) and not self.tex

    def request(self, key):
        if key in self.tex or key in self.pending or len(self.pending) >= 48: return
        if time.time() - self.failed.get(key, -99) < 20: return
        self.pending.add(key); self.jobs.put(key)

    def _worker(self):
        while True:
            key = self.jobs.get()
            try: self.done.put((key, *self._load(key)))
            except Exception: self.done.put((key, None, None, None))

    def _load(self, key):
        z, x, y = key
        path = self.dir / str(z) / str(x) / f"{y}.png"
        if not path.exists():
            url = self.url.format(z=z, x=x, y=y, s="a")
            with urllib.request.urlopen(urllib.request.Request(url, headers=self.UA),
                                        timeout=15) as r:
                blob = r.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp"); tmp.write_bytes(blob); tmp.replace(path)
        pg = self.pg
        src = pg.image.load(str(path))
        out = pg.Surface(src.get_size(), pg.SRCALPHA, 32); out.blit(src, (0, 0))
        return (*out.get_size(), pg.image.tostring(out, "RGBA", False))

    def pump(self, limit=6):
        for _ in range(limit):
            try: key, w, h, data = self.done.get_nowait()
            except queue.Empty: break
            self.pending.discard(key)
            if data is None: self.failed[key] = time.time()
            else: self.tex[key] = make_texture(self.gl, w, h, data)
        if len(self.tex) > 600:
            for k in list(self.tex)[:150]:
                self.gl.glDeleteTextures([self.tex.pop(k)])


class WorldMap:
    T = 256
    PANEL_X, PANEL_Y, PANEL_W, ROW_H = 12, 12, 210, 24

    def __init__(self, tiles, invert=True):
        self.tiles = tiles
        self.invert = 1.0 if invert else 0.0
        self.zoom = self.zoom_t = 2.2
        self.cx = self.cx_t = 0.5
        self.cy = self.cy_t = 0.40
        self.marker = None
        self.flying = False
        self.W = self.H = 1

    @staticmethod
    def to_world(lat, lon):
        s = math.sin(math.radians(max(-85.0511, min(85.0511, lat))))
        return (lon + 180.0) / 360.0, 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)

    @staticmethod
    def from_world(x, y):
        y = max(0.0, min(1.0, y))
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y)))), x * 360.0 - 180.0

    def ppw(self): return self.T * 2.0 ** self.zoom

    def screen_to_latlon(self, sx, sy):
        p = self.ppw()
        return self.from_world(self.cx + (sx - self.W / 2) / p,
                               self.cy + (sy - self.H / 2) / p)

    def latlon_to_screen(self, lat, lon):
        x, y = self.to_world(lat, lon); p = self.ppw()
        return (x - self.cx) * p + self.W / 2, (y - self.cy) * p + self.H / 2

    def set_size(self, w, h):
        self.W, self.H = max(w, 1), max(h, 1); self._clamp()

    def _clamp(self):
        zmin = math.log2(max(self.H, 256) / self.T)
        self.zoom = max(zmin, min(17.5, self.zoom))
        half = (self.H / 2) / self.ppw()
        self.cy = 0.5 if half >= 0.5 else max(half, min(1 - half, self.cy))
        self.cx = max(0.0, min(1.0, self.cx))
        self.zoom_t = max(zmin, min(17.5, self.zoom_t))

    def _sync(self):
        self.zoom_t, self.cx_t, self.cy_t, self.flying = self.zoom, self.cx, self.cy, False

    def pan(self, dx, dy):
        p = self.ppw(); self.cx -= dx / p; self.cy -= dy / p
        self._clamp(); self._sync()

    def zoom_by(self, steps, sx, sy):
        p = self.ppw()
        wx, wy = self.cx + (sx - self.W / 2) / p, self.cy + (sy - self.H / 2) / p
        self.zoom += steps * 0.4; self._clamp(); p = self.ppw()
        self.cx, self.cy = wx - (sx - self.W / 2) / p, wy - (sy - self.H / 2) / p
        self._clamp(); self._sync()

    def jump_to(self, lat, lon, zoom):
        self.cx, self.cy = self.to_world(lat, lon)
        self.zoom = zoom; self._clamp(); self._sync()

    def fly_to(self, lat, lon, zoom):
        self.cx_t, self.cy_t = self.to_world(lat, lon)
        self.zoom_t = zoom; self.flying = True

    def update(self, dt):
        if not self.flying: return
        k = 1 - math.exp(-4.5 * dt)
        self.zoom += (self.zoom_t - self.zoom) * k
        self.cx += (self.cx_t - self.cx) * k
        self.cy += (self.cy_t - self.cy) * k
        self._clamp()
        if (abs(self.zoom_t - self.zoom) < 0.02
                and abs(self.cx_t - self.cx) * self.ppw() < 0.5
                and abs(self.cy_t - self.cy) * self.ppw() < 0.5):
            self._sync()

    def city_row_at(self, x, y, n):
        if self.PANEL_X <= x <= self.PANEL_X + self.PANEL_W and y >= self.PANEL_Y + 34:
            i = int((y - self.PANEL_Y - 34) // self.ROW_H)
            if 0 <= i < n: return i
        return None

    def city_near(self, sx, sy, cities, px=12):
        for name, lat, lon in cities:
            x, y = self.latlon_to_screen(lat, lon)
            if math.hypot(x - sx, y - sy) <= px: return name, lat, lon
        return None

    def draw(self, ov, mouse, cities, loading=False):
        W, H, p = self.W, self.H, self.ppw()
        z = int(max(0, min(17, math.floor(self.zoom + 0.5))))
        n = 1 << z; size = self.T * 2.0 ** (self.zoom - z)
        x0, x1 = self.cx - W / 2 / p, self.cx + W / 2 / p
        y0, y1 = self.cy - H / 2 / p, self.cy + H / 2 / p
        tx0, tx1 = max(0, int(math.floor(x0 * n))), min(n - 1, int(math.floor(x1 * n)))
        ty0, ty1 = max(0, int(math.floor(y0 * n))), min(n - 1, int(math.floor(y1 * n)))
        drawn = 0
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                px, py = (tx / n - self.cx) * p + W / 2, (ty / n - self.cy) * p + H / 2
                tex = self.tiles.tex.get((z, tx, ty))
                if tex is not None:
                    ov.tile(tex, px, py, size + 0.6, (0, 0, 1, 1), self.invert)
                    drawn += 1; continue
                self.tiles.request((z, tx, ty))
                self.tiles.request((max(z - 3, 0), tx >> min(3, z), ty >> min(3, z)))
                for k in range(1, 6):
                    if z - k < 0: break
                    parent = self.tiles.tex.get((z - k, tx >> k, ty >> k))
                    if parent is not None:
                        f = 1 << k
                        u0, v0 = (tx & (f - 1)) / f, (ty & (f - 1)) / f
                        ov.tile(parent, px, py, size + 0.6,
                                (u0, v0, u0 + 1 / f, v0 + 1 / f), self.invert)
                        drawn += 1; break
        if drawn == 0: self._graticule(ov)
        for i, (name, lat, lon) in enumerate(cities):
            x, y = self.latlon_to_screen(lat, lon)
            if -20 < x < W + 20 and -20 < y < H + 20:
                ov.circle(x, y, 4.5, (0.05, 0.07, 0.10, 0.9))
                ov.circle(x, y, 3.2, ACCENT)
                ov.text(name, x + 8, y - 7, 17, (0.85, 0.93, 1.0, 0.95))
        if self.marker:
            lat, lon, label = self.marker
            x, y = self.latlon_to_screen(lat, lon)
            pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)
            ov.ring(x, y, 14 + 4 * pulse, 2.0, (*ACCENT[:3], 0.9))
            ov.circle(x, y, 4.5, ACCENT)
            ov.line(x - 22, y, x - 8, y, 1.5, (*ACCENT[:3], 0.8))
            ov.line(x + 8, y, x + 22, y, 1.5, (*ACCENT[:3], 0.8))
            ov.line(x, y - 22, x, y - 8, 1.5, (*ACCENT[:3], 0.8))
            ov.line(x, y + 8, x, y + 22, 1.5, (*ACCENT[:3], 0.8))
            ov.text(label[:48], x, y - 44, 22, (1, 1, 1, 1), anchor="center")
        h = 40 + self.ROW_H * len(cities) + 10
        ov.panel(self.PANEL_X, self.PANEL_Y, self.PANEL_W, h)
        ov.text("TRAVEL", self.PANEL_X + 14, self.PANEL_Y + 12, 20, ACCENT)
        ov.text("click or press key", self.PANEL_X + self.PANEL_W - 12, self.PANEL_Y + 15, 14,
                (0.55, 0.65, 0.78, 0.8), anchor="right")
        sel = self.marker[2] if self.marker else None
        for i, (name, _, _) in enumerate(cities):
            y = self.PANEL_Y + 34 + i * self.ROW_H
            hover = (self.PANEL_X <= mouse[0] <= self.PANEL_X + self.PANEL_W
                     and y <= mouse[1] < y + self.ROW_H)
            if hover:
                ov.rect(self.PANEL_X + 4, y, self.PANEL_W - 8, self.ROW_H - 2,
                        (0.20, 0.38, 0.55, 0.55))
                ov.rect(self.PANEL_X + 4, y, 3, self.ROW_H - 2, ACCENT)
            elif name == sel:
                ov.rect(self.PANEL_X + 4, y, self.PANEL_W - 8, self.ROW_H - 2,
                        (0.36, 0.80, 1.0, 0.12))
            ov.rect(self.PANEL_X + 14, y + 3, 20, self.ROW_H - 8, (1, 1, 1, 0.08))
            ov.text(str((i + 1) % 10), self.PANEL_X + 24, y + 5, 16, (0.55, 0.72, 0.9, 1),
                    anchor="center", shadow=False)
            ov.text(name, self.PANEL_X + 44, y + 4, 19, (0.92, 0.96, 1, 1))
        lat, lon = self.screen_to_latlon(*mouse)
        ov.text(f"{lat:.4f}, {lon:.4f}", W - 12, H - 44, 18, (0.7, 0.8, 0.9, 0.9),
                anchor="right")
        ov.text("(c) OpenStreetMap contributors", W - 12, H - 24, 16,
                (0.6, 0.65, 0.72, 0.8), anchor="right")
        if self.tiles.offline:
            ov.text("Map tiles unavailable (offline?) - the grid and presets still work",
                    W / 2, 16, 20, (1.0, 0.75, 0.4, 1), anchor="center")
        ov.text("Drag: pan   Scroll: zoom   Click: pick   Double-click / Enter: travel   "
                "/: search   F2: log   M or Esc: back to 3D", 12, H - 24, 16, (0.65, 0.72, 0.8, 0.9))

    def _graticule(self, ov):
        ppd = self.ppw() / 360.0
        step = next((s for s in (0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 15, 30)
                     if s * ppd >= 70), 30)
        col = (0.25, 0.38, 0.55, 0.45)
        lat_top, lon_left = self.screen_to_latlon(0, 0)
        lat_bot, lon_right = self.screen_to_latlon(self.W, self.H)
        lon = math.floor(lon_left / step) * step
        while lon <= lon_right:
            x, _ = self.latlon_to_screen(0, lon)
            ov.line(x, 0, x, self.H, 1.0, col); lon += step
        lat = max(-84.0, math.floor(lat_bot / step) * step)
        while lat <= min(84.0, lat_top):
            _, y = self.latlon_to_screen(lat, 0)
            ov.line(0, y, self.W, y, 1.0, col); lat += step