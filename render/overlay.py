"""2D overlay primitives and the map tile cache.

Everything the HUD draws goes through ``Overlay2D``:

* shapes are signed-distance-field quads, so rounded boxes, circles, rings and
  lines are analytically anti-aliased (no polygon fans, no stair-steps);
* text is rasterised with Inter at the *physical* pixel size and drawn on exact
  pixel boundaries, so glyphs are never resampled (that resampling was the
  "blurry UI");
* coordinates are logical pixels; ``scale`` maps them onto the framebuffer so
  the layout is identical on 100 % and 200 % displays.
"""
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

from ..ui import theme as T
from ..core.config import ACCENT
from ..shaders import OVERLAY_VS, OVERLAY_FS

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_FILES = {"regular": "Inter-Regular.ttf", "medium": "Inter-Medium.ttf",
              "semibold": "Inter-SemiBold.ttf", "bold": "Inter-Bold.ttf"}
_FALLBACK_CHAIN = ("inter", "segoeui", "sfprotext", "helveticaneue", "arial", "dejavusans")


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


def _is_pair(c):
    """True for a (top, bottom) gradient pair, False for a single RGBA tuple."""
    return len(c) == 2 and hasattr(c[0], "__len__")


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
                  for n in ("uScreen", "uMode", "uColor", "uColor2", "uBorder", "uHalf", "uRad",
                            "uBw", "uSoft", "uInvert", "uTex", "uClip", "uClipRad")}
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        for loc, off in ((0, 0), (1, 8)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(off))
        self.cbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.cbo)
        gl.glVertexAttribPointer(2, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        # Only the triangle batch enables this array.  Leaving it on would make
        # every later primitive fetch colours beyond this buffer's last upload.
        gl.glDisableVertexAttribArray(2)
        gl.glBindVertexArray(0)
        self.fonts, self.text_cache = {}, {}
        self.font_name = None          # optional system-font override
        self.scale = 1.0               # logical -> physical pixel factor (HiDPI)
        self.lw, self.lh = 1280.0, 800.0
        self.pw, self.ph = 1280, 800
        self._clips = []
        self.now, self.dt, self._last = time.time(), 1 / 60, time.time()
        self._anim = {}
        self._spring = {}
        self._font_paths = {}

    # ---------------------------------------------------------------- frame
    def begin(self, w, h):
        """``w``/``h`` are the framebuffer size in physical pixels."""
        gl = self.gl
        self.now = time.time()
        self.dt = min(0.1, max(1e-3, self.now - self._last)); self._last = self.now
        S = self.scale = max(0.5, float(self.scale))
        self.lw, self.lh = w / S, h / S
        self.pw, self.ph = int(w), int(h)
        self._clips = []
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND); gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glUseProgram(self.prog)
        gl.glUniform2f(self.u["uScreen"], self.lw, self.lh)
        gl.glUniform1i(self.u["uTex"], 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(self.vao)

    # ------------------------------------------------------------- clipping
    def push_clip(self, x, y, w, h):
        """Restrict drawing to a logical-pixel rectangle (nests by intersection).

        Scrolling lists draw every row and let the scissor cut the ones that poke
        out of their viewport.  Always pair with ``pop_clip``.
        """
        S = self.scale
        x0, y0 = max(0, int(math.floor(x * S))), max(0, int(math.floor(y * S)))
        x1 = min(self.pw, int(math.ceil((x + w) * S)))
        y1 = min(self.ph, int(math.ceil((y + h) * S)))
        if self._clips:
            cx0, cy0, cx1, cy1 = self._clips[-1]
            x0, y0, x1, y1 = max(x0, cx0), max(y0, cy0), min(x1, cx1), min(y1, cy1)
        x1, y1 = max(x0, x1), max(y0, y1)
        self._clips.append((x0, y0, x1, y1))
        self._apply_clip()

    def pop_clip(self):
        if self._clips:
            self._clips.pop()
        self._apply_clip()

    def _apply_clip(self):
        gl = self.gl
        if not self._clips:
            gl.glDisable(gl.GL_SCISSOR_TEST)
            return
        x0, y0, x1, y1 = self._clips[-1]
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(x0, self.ph - y1, x1 - x0, y1 - y0)      # GL's origin is the bottom-left

    def end(self):
        while self._clips:
            self.pop_clip()
        self.gl.glDisable(self.gl.GL_BLEND)
        self.gl.glEnable(self.gl.GL_DEPTH_TEST)
        self.gl.glBindVertexArray(0)

    def anim(self, key, target, rate=16.0):
        """Exponentially smoothed value per widget key (hover fades, slide-ins)."""
        v = self._anim.get(key, target if key not in self._anim else 0.0)
        v += (target - v) * (1.0 - math.exp(-rate * self.dt))
        if abs(target - v) < 0.002: v = target
        self._anim[key] = v
        return v

    def anim_spring(self, key, target, damping=0.85, stiffness=180.0):
        """Spring-physics animation per widget key — natural, Apple-like motion.

        ``damping`` < 1 gives a slight overshoot; ``stiffness`` controls speed.
        State (position + velocity) is stored per key for critically-damped springs.
        """
        state = self._spring.get(key)
        if state is None:
            state = [target if key not in self._spring else 0.0, 0.0]  # [pos, vel]
            self._spring[key] = state
        pos, vel = state
        dt = self.dt
        # Semi-implicit Euler spring integration
        accel = stiffness * (target - pos) - 2.0 * damping * math.sqrt(stiffness) * vel
        vel += accel * dt
        pos += vel * dt
        # Snap when close enough and slow
        if abs(target - pos) < 0.002 and abs(vel) < 0.02:
            pos, vel = target, 0.0
        state[0], state[1] = pos, vel
        return pos

    # ------------------------------------------------------------ low level
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

    def _snap(self, v):
        """Round a logical coordinate onto a physical pixel boundary."""
        S = self.scale
        return round(v * S) / S

    # ----------------------------------------------------------- primitives
    def rect(self, x, y, w, h, color):
        self._set(0, color); self._submit(_quad(x, y, w, h), self.gl.GL_TRIANGLES)

    def line(self, x0, y0, x1, y1, width, color):
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1.0
        ux, uy = dx / ln, dy / ln
        m = 1.0 / self.scale                      # one device pixel of anti-aliasing fringe
        hw = width / 2.0
        h = hw + m
        nx, ny = -uy * h, ux * h
        ex, ey = ux * 0.5 * m, uy * 0.5 * m
        x0, y0, x1, y1 = x0 - ex, y0 - ey, x1 + ex, y1 + ey
        v = np.array([[x0 + nx, y0 + ny, h, 0], [x0 - nx, y0 - ny, -h, 0],
                      [x1 + nx, y1 + ny, h, 0], [x1 - nx, y1 - ny, -h, 0]], np.float32)
        gl = self.gl
        self._set(6, color)
        gl.glUniform2f(self.u["uHalf"], hw, 0.0)
        self._submit(v, gl.GL_TRIANGLE_STRIP)

    def box(self, x, y, w, h, r, fill=None, border=None, bw=1.0, shadow=0.0, shadow_dy=0.0,
            shadow_alpha=0.55):
        """The one rounded-box primitive: gradient fill, hairline border, soft shadow.

        ``fill`` is an RGBA tuple or a (top, bottom) pair; ``border`` an RGBA tuple.
        """
        if w <= 0 or h <= 0: return
        gl = self.gl
        r = max(0.0, min(float(r), min(w, h) * 0.5))
        cx, cy = x + w * 0.5, y + h * 0.5
        hx, hy = w * 0.5, h * 0.5
        if shadow > 0.0:
            m = shadow * 1.2
            sx0, sy0 = x - m, y - m + shadow_dy
            self._set(5, (0.0, 0.0, 0.0, shadow_alpha))
            gl.glUniform2f(self.u["uHalf"], hx, hy)
            gl.glUniform1f(self.u["uRad"], r)
            gl.glUniform1f(self.u["uSoft"], float(shadow))
            q = _quad(sx0, sy0, w + 2 * m, h + 2 * m)
            q[:, 2] = q[:, 0] - cx; q[:, 3] = q[:, 1] - (cy + shadow_dy)
            self._submit(q, gl.GL_TRIANGLES)
        if fill is None and border is None: return
        if fill is None: top = bot = (0.0, 0.0, 0.0, 0.0)
        elif _is_pair(fill): top, bot = fill
        else: top = bot = fill
        bcol = border if border is not None else (0.0, 0.0, 0.0, 0.0)
        m = 1.5 / self.scale
        q = _quad(x - m, y - m, w + 2 * m, h + 2 * m)
        q[:, 2] = q[:, 0] - cx; q[:, 3] = q[:, 1] - cy
        self._set(4, top)
        gl.glUniform4f(self.u["uColor2"], *[float(c) for c in bot])
        gl.glUniform4f(self.u["uBorder"], *[float(c) for c in bcol])
        gl.glUniform2f(self.u["uHalf"], hx, hy)
        gl.glUniform1f(self.u["uRad"], r)
        gl.glUniform1f(self.u["uBw"], float(bw) if border is not None else 0.0)
        self._submit(q, gl.GL_TRIANGLES)

    def circle(self, cx, cy, r, color, seg=0):
        self.box(cx - r, cy - r, 2 * r, 2 * r, r, color)

    def ring(self, cx, cy, r, thick, color, seg=0):
        self.box(cx - r, cy - r, 2 * r, 2 * r, r, None, color, bw=thick)

    def rrect(self, x, y, w, h, r, color, seg=0):
        """Anti-aliased rounded rectangle (``seg`` kept for call-site compatibility)."""
        self.box(x, y, w, h, r, color)

    def poly(self, pts, color):
        """Filled convex polygon; a hairline stroke in the same colour softens the edge."""
        arr = np.asarray(pts, np.float32)
        if len(arr) < 3: return
        c = arr.mean(0)
        v = np.zeros((len(arr) + 2, 4), np.float32)
        v[0, :2] = c
        v[1:1 + len(arr), :2] = arr
        v[-1, :2] = arr[0]
        self._set(0, color); self._submit(v, self.gl.GL_TRIANGLE_FAN)
        n = len(arr)
        for i in range(n):
            a, b = arr[i], arr[(i + 1) % n]
            self.line(a[0], a[1], b[0], b[1], 1.0, color)

    def tri(self, p0, p1, p2, color):
        self.poly([p0, p1, p2], color)

    def gradient(self, x, y, w, h, top, bottom):
        """Vertical two-colour gradient quad."""
        gl = self.gl
        self._set(3, top)
        gl.glUniform4f(self.u["uColor2"], *[float(c) for c in bottom])
        self._submit(_quad(x, y, w, h), gl.GL_TRIANGLES)

    def outline_poly(self, pts, width, color, closed=True):
        n = len(pts)
        for i in range(n if closed else n - 1):
            a, b = pts[i], pts[(i + 1) % n]
            self.line(a[0], a[1], b[0], b[1], width, color)

    # ----------------------------------------------------------- containers
    def container(self, x, y, w, h, kind="panel", r=None, hover=0.0, pressed=False,
                  active=False, focus=False, shadow=True):
        """A themed surface.  ``kind``: panel, popup, button, field, inset, accent.

        ``hover`` (0..1, from ``anim``) cross-fades the button gradient and border.
        Snapped to whole pixels so the 1 px hairline is always crisp.
        """
        x, y = self._snap(x), self._snap(y)
        w, h = self._snap(w), self._snap(h)
        if kind == "button":
            fill = T.BUTTON_FILL
            if pressed: fill = T.BUTTON_PRESS
            elif hover > 0.0:
                fill = tuple(tuple(a + (b - a) * hover for a, b in zip(f0, f1))
                             for f0, f1 in zip(T.BUTTON_FILL, T.BUTTON_HOVER))
            bc = T.BORDER_BUTTON
            if hover > 0.0: bc = tuple(a + (b - a) * hover for a, b in zip(T.BORDER_BUTTON, T.BORDER_BUTTON_HOVER))
            if active: fill, bc = T.ACCENT_FILL, (1.0, 1.0, 1.0, 0.95)
            if focus: bc = T.BORDER_FOCUS
            self.box(x, y, w, h, T.R_BUTTON if r is None else r, fill, bc,
                     shadow=10.0 if shadow else 0.0, shadow_dy=3.0, shadow_alpha=0.35)
        elif kind == "field":
            self.box(x, y, w, h, T.R_FIELD if r is None else r, T.FIELD_FILL,
                     T.BORDER_FOCUS if focus else T.BORDER_PANEL)
        elif kind == "inset":
            self.box(x, y, w, h, T.R_BUTTON if r is None else r, T.INSET_FILL,
                     T.BORDER_SOFT if not hover else T.BORDER_PANEL)
        elif kind == "accent":
            self.box(x, y, w, h, T.R_BUTTON if r is None else r, T.ACCENT_FILL,
                     (1.0, 1.0, 1.0, 0.9))
        else:
            fill = T.POPUP_FILL if kind == "popup" else T.PANEL_FILL
            self.box(x, y, w, h, T.R_PANEL if r is None else r, fill, T.BORDER_PANEL,
                     shadow=22.0 if shadow else 0.0, shadow_dy=8.0, shadow_alpha=0.50)

    def glass(self, x, y, w, h, r=T.R_PANEL, fill=None, border=None, shadow=True):
        """Backward-compatible card: now a graphite container."""
        self.container(x, y, w, h, "panel", r=r, shadow=shadow)

    def panel(self, x, y, w, h, accent=ACCENT, bg=None):
        self.container(x, y, w, h, "panel")

    def divider(self, x, y, w, color=T.HAIRLINE):
        self.rect(self._snap(x), self._snap(y), w, 1.0 / self.scale, color)

    def vdivider(self, x, y, h, color=T.HAIRLINE):
        self.rect(self._snap(x), self._snap(y), 1.0 / self.scale, h, color)

    def kbd(self, x, y, key, size=12, h=None):
        """A keycap chip.  Returns its width."""
        tw = self.text_w(key, size, "semibold")
        h = h or size + 10
        w = max(h, tw + 12)
        self.box(self._snap(x), self._snap(y), self._snap(w), self._snap(h), T.R_KEY,
                 ((0.26, 0.26, 0.27, 1.0), (0.15, 0.15, 0.16, 1.0)), (1.0, 1.0, 1.0, 0.38))
        self.text(key, x + w / 2, y + (h - self.measure("Ag", size)[1]) / 2, size, T.TEXT,
                  anchor="center", weight="semibold")
        return w

    # ------------------------------------------------------------- widgets
    def bar(self, x, y, w, h, frac, color=ACCENT, shimmer=0.0):
        """Progress bar. shimmer > 0 sweeps a highlight over the fill (activity indicator)."""
        frac = max(0.0, min(1.0, float(frac)))
        self.box(x, y, w, h, h / 2, (1, 1, 1, 0.14))
        fw = w * frac
        if fw <= 0.5: return
        self.box(x, y, max(fw, h), h, h / 2, (color[0], color[1], color[2], 1.0))
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
                        1.0 + 1.5 * fade, (color[0], color[1], color[2], 0.15 + 0.85 * fade))

    def tick(self, cx, cy, s, color, width=2.0):
        self.line(cx - s * 0.55, cy + s * 0.05, cx - s * 0.15, cy + s * 0.45, width, color)
        self.line(cx - s * 0.15, cy + s * 0.45, cx + s * 0.60, cy - s * 0.40, width, color)

    def tile(self, tex, x, y, size, uv=(0, 0, 1, 1), invert=1.0, alpha=1.0):
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
        self._set(2, (1, 1, 1, alpha), invert)
        self._submit(_quad(x, y, size, size, *uv), self.gl.GL_TRIANGLES)

    def image(self, tex, x, y, w, h, radius=0.0, alpha=1.0):
        """Draw a texture with an anti-aliased rounded mask (cover art, thumbnails)."""
        gl = self.gl
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        self._set(7, (1, 1, 1, alpha))
        gl.glUniform2f(self.u["uHalf"], w * 0.5, h * 0.5)
        gl.glUniform1f(self.u["uRad"], max(0.0, min(float(radius), min(w, h) * 0.5)))
        q = _quad(x, y, w, h)
        cx, cy = x + w * 0.5, y + h * 0.5
        q[:, 2] = q[:, 0] - cx; q[:, 3] = q[:, 1] - cy
        self._submit(q, gl.GL_TRIANGLES)

    def triangles(self, verts, colors, clip=None, clip_radius=0.0):
        """Batched per-vertex coloured triangles.

        ``verts`` is (N, 2) in logical pixels (three entries per triangle) and
        ``colors`` (N, 4) linear RGBA.  ``clip`` is ``(x, y, w, h)``; fragments
        outside that rounded rectangle are discarded, which lets a 3D inset
        draw its geometry without a separate framebuffer.
        """
        verts = np.ascontiguousarray(verts, np.float32).reshape(-1, 2)
        colors = np.ascontiguousarray(colors, np.float32).reshape(-1, 4)
        if len(verts) == 0 or len(verts) != len(colors):
            return
        gl = self.gl
        pos = np.zeros((len(verts), 4), np.float32)
        pos[:, :2] = verts
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, pos.nbytes, pos, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.cbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STREAM_DRAW)
        gl.glEnableVertexAttribArray(2)
        self._set(8, (1, 1, 1, 1))
        x, y, w, h = clip if clip is not None else (0.0, 0.0, self.lw, self.lh)
        gl.glUniform4f(self.u["uClip"], float(x + w * 0.5), float(y + h * 0.5),
                       float(w), float(h))
        gl.glUniform1f(self.u["uClipRad"], float(clip_radius))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(verts))
        gl.glDisableVertexAttribArray(2)

    # ----------------------------------------------------------------- text
    def _font_path(self, weight):
        p = self._font_paths.get(weight)
        if p is None:
            f = FONT_DIR / FONT_FILES.get(weight, FONT_FILES["regular"])
            if f.exists():
                p = str(f)
            else:                                   # not bundled: best installed approximation
                names = ([self.font_name] if self.font_name else []) + list(_FALLBACK_CHAIN)
                bold = weight in ("semibold", "bold")
                p = ""
                for n in names:
                    p = self.pg.font.match_font(n, bold=bold) or ""
                    if p: break
            self._font_paths[weight] = p
        return p

    @staticmethod
    def _weight(bold, weight):
        if weight: return weight
        return "semibold" if bold else "regular"

    def _font(self, size, bold=False, weight=None):
        wt = self._weight(bold, weight)
        px = max(6, int(round(size * self.scale)))
        key = (px, wt)
        font = self.fonts.get(key)
        if font is None:
            path = self._font_path(wt)
            font = self.pg.font.Font(path, px) if path else self.pg.font.Font(None, px)
            self.fonts[key] = font
        return font

    def measure(self, s, size, bold=False, weight=None):
        """Logical size of a string without allocating a texture."""
        w, h = self._font(size, bold, weight).size(s or " ")
        return w / self.scale, h / self.scale

    def text_w(self, s, size=14, bold=False, weight=None):
        return self._font(size, bold, weight).size(s)[0] / self.scale if s else 0

    def fit(self, s, size, max_w, bold=False, weight=None):
        """Ellipsise ``s`` so it fits in ``max_w`` logical pixels."""
        if not s or self.text_w(s, size, bold, weight) <= max_w: return s
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.text_w(s[:mid].rstrip() + "\u2026", size, bold, weight) <= max_w: lo = mid
            else: hi = mid - 1
        return s[:lo].rstrip() + "\u2026"

    def _text_tex(self, s, px, wt):
        key = (s, px, wt)
        hit = self.text_cache.pop(key, None)
        if hit is not None:
            self.text_cache[key] = hit          # re-insert: most recently used goes last
            return hit
        if len(self.text_cache) >= 500:
            # evict only the oldest few; wiping every texture at once was a periodic hitch
            for k in list(self.text_cache)[:60]:
                self.gl.glDeleteTextures([self.text_cache.pop(k)[0]])
        path = self._font_path(wt)
        font = self.fonts.get((px, wt))
        if font is None:
            font = self.pg.font.Font(path, px) if path else self.pg.font.Font(None, px)
            self.fonts[(px, wt)] = font
        surf = font.render(s, True, (255, 255, 255)).convert_alpha()
        w, h = surf.get_size()
        tex = make_texture(self.gl, w, h, self.pg.image.tostring(surf, "RGBA", False))
        self.text_cache[key] = (tex, w, h)
        return self.text_cache[key]

    def text(self, s, x, y, size=14, color=(1, 1, 1, 1), anchor="left", shadow=False,
             bold=False, weight=None, halo=False):
        """Draw ``s`` with its top-left (or centre/right per ``anchor``) at logical x, y.

        The glyph bitmap is blitted 1:1 onto whole device pixels - never resampled.
        ``shadow`` is accepted for compatibility and ignored; use ``halo`` for labels
        that sit directly on the map or the 3D scene.
        """
        if not s: return 0
        S = self.scale
        wt = self._weight(bold, weight)
        px = max(6, int(round(size * S)))
        tex, w, h = self._text_tex(s, px, wt)
        lw, lh = w / S, h / S
        if anchor == "center": x -= lw / 2
        elif anchor == "right": x -= lw
        x, y = round(x * S) / S, round(y * S) / S
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, tex)
        if halo:
            hc = (0.0, 0.0, 0.0, 0.55 * color[3]); d = 1.0 / S
            for ox, oy in ((-d, 0), (d, 0), (0, -d), (0, d), (-d, -d), (d, d), (-d, d), (d, -d)):
                self._set(1, hc); self._submit(_quad(x + ox, y + oy, lw, lh), self.gl.GL_TRIANGLES)
        self._set(1, color)
        self._submit(_quad(x, y, lw, lh), self.gl.GL_TRIANGLES)
        return lw

    def text_c(self, s, x, cy, size=14, color=(1, 1, 1, 1), anchor="left", weight=None, bold=False,
               halo=False):
        """Text vertically centred on ``cy`` (Inter's caps sit on the box centre)."""
        h = self.measure("Ag", size, bold, weight)[1]
        return self.text(s, x, cy - h / 2, size, color, anchor, bold=bold, weight=weight, halo=halo)


class TileCache:
    """Background tile downloader with an on-disk cache.

    Pixels stay on the CPU: the globe composites them into its mosaic, so no
    per-tile GL texture is needed any more.
    """

    UA = {"User-Agent": "osm3d-viewer/1.0 (educational PyOpenGL demo)"}

    def __init__(self, url, cache_dir, workers=6):
        import pygame
        self.pg, self.url = pygame, url
        self.dir = Path(cache_dir) / "tiles" / hashlib.sha1(url.encode()).hexdigest()[:8]
        self.px = {}; self.pending = set(); self.failed = {}
        self.jobs = queue.Queue(); self.done = queue.Queue()
        for _ in range(workers):
            threading.Thread(target=self._worker, daemon=True).start()

    @property
    def offline(self):
        return bool(self.failed) and not self.px

    def request(self, key):
        if key in self.px or key in self.pending or len(self.pending) >= 24:
            return
        if time.time() - self.failed.get(key, -99) < 20:
            return
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
        src = pg.image.load(str(path)).convert_alpha()
        return (*src.get_size(), pg.image.tostring(src, "RGBA", False))

    def pump(self, limit=8):
        for _ in range(limit):
            try: key, w, h, data = self.done.get_nowait()
            except queue.Empty: break
            self.pending.discard(key)
            if data is None: self.failed[key] = time.time()
            else: self.px[key] = (w, h, data)
        while len(self.px) > 320:
            self.px.pop(next(iter(self.px)))
