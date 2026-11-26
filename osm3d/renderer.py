"""Sky atmosphere + OpenGL renderer (buildings, water, terrain, trees, shadows)."""
from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass

import numpy as np

from .config import BG_COLOR, MOON_COLOR
from .data_fetcher import build_tree_template
from .mesh_extruder import MeshData
from .shaders import (
    SKY_VS, SKY_FS,
    BUILDING_VS, BUILDING_FS,
    LINE_VS, LINE_FS,
    GROUND_VS, GROUND_FS,
    TERRAIN_FS,
    WATER_VS, WATER_FS,
    TREE_VS, TREE_FS,
    SHADOW_VS, SHADOW_FS, SHADOW_TREE_VS,
)


# ---------------------------------------------------------------- sky state
def _smoothstep(e0, e1, x):
    t = 0.0 if e1 == e0 else (x - e0) / (e1 - e0)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def solar_position(lat_deg, hours, decl_deg=12.0):
    lat, dec = math.radians(lat_deg), math.radians(decl_deg)
    h = math.radians(15.0 * (hours - 12.0))
    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(h)
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(-math.cos(dec) * math.sin(h),
                    math.sin(dec) * math.cos(lat) - math.cos(dec) * math.sin(lat) * math.cos(h))
    return alt, az


def sky_vector(alt, az):
    ce = math.cos(alt)
    return np.array([ce * math.sin(az), ce * math.cos(az), math.sin(alt)], np.float64)


@dataclass
class SkyState:
    hours: float
    alt: float
    az: float
    sun_dir: np.ndarray
    moon_dir: np.ndarray
    light_dir: np.ndarray
    sky: np.ndarray
    fog: np.ndarray
    amb: np.ndarray
    light: np.ndarray
    shadows: bool
    strength: float
    moon_phase: float
    night: float
    sun_intensity: float
    moon_intensity: float

    @property
    def is_day(self) -> bool: return self.alt > 0.0
    @property
    def stamp(self) -> str:
        m = int(round(self.hours * 60.0)) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"


def compute_sky(hours, lat_deg=40.758, decl_deg=12.0) -> SkyState:
    hours = float(hours) % 24.0
    alt, az = solar_position(lat_deg, hours, decl_deg)
    malt, maz = solar_position(lat_deg, (hours + 12.0) % 24.0, -decl_deg)
    deg = math.degrees(alt)
    sun_vec = sky_vector(alt, az).astype(np.float32)
    moon_vec = sky_vector(malt, maz).astype(np.float32)
    light_dir = sun_vec if deg > -3.0 else moon_vec
    sun_gain = _smoothstep(-3.0, 5.0, deg)
    warm = _smoothstep(-3.0, 15.0, deg)
    sun_color = np.array([1.0, 0.6 + 0.4 * warm, 0.35 + 0.65 * warm], np.float32) * 4.0 * sun_gain
    moon_visible = _smoothstep(-3.0, -9.0, deg) * (1.0 if malt > 0.0 else 0.0)
    moon_color = np.array(MOON_COLOR, np.float32) * 0.4 * moon_visible
    cos_phase = float(np.clip(np.dot(sun_vec, moon_vec), -1.0, 1.0))
    moon_phase = max(0.06, 0.5 * (1.0 - cos_phase))
    night = _smoothstep(-2.0, -14.0, deg)

    w_day = _smoothstep(-2.0, 15.0, deg)
    w_night = 1.0 - _smoothstep(-18.0, -2.0, deg)
    w_sunset = max(0.0, 1.0 - w_day - w_night)

    horizon_day = np.array([0.65, 0.75, 0.90], np.float32)
    horizon_sunset = np.array([0.95, 0.55, 0.30], np.float32)
    horizon_night = np.array([0.03, 0.05, 0.10], np.float32)

    fog = horizon_day * w_day + horizon_sunset * w_sunset + horizon_night * w_night
    amb_day = np.array([0.45, 0.55, 0.65], np.float32)
    amb_sunset = np.array([0.55, 0.42, 0.32], np.float32)
    amb_night = np.array([0.03, 0.04, 0.08], np.float32)
    amb = amb_day * w_day + amb_sunset * w_sunset + amb_night * w_night
    light = sun_color if deg > -3.0 else moon_color
    sky_clear = fog * 0.9
    strength = _smoothstep(0.5, 7.0, deg)

    return SkyState(
        hours=hours, alt=alt, az=az,
        sun_dir=sun_vec, moon_dir=moon_vec, light_dir=light_dir.astype(np.float32),
        sky=sky_clear.astype(np.float32), fog=fog, amb=amb, light=light,
        shadows=strength > 0.002, strength=strength,
        moon_phase=moon_phase, night=night,
        sun_intensity=float(sun_gain), moon_intensity=float(moon_visible),
    )


# ---------------------------------------------------------------- light matrix
def ortho(l, r, b, t, n, f):
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1], m[2, 2] = 2.0 / (r - l), 2.0 / (t - b), -2.0 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    m[3, 3] = 1.0
    return m


@dataclass
class LightMatrix:
    view: np.ndarray
    proj: np.ndarray
    matrix: np.ndarray
    extent: float
    texel: float


def light_matrix(target, light_dir, extent, size):
    d = np.asarray(light_dir, np.float64)
    d = d / max(1e-9, np.linalg.norm(d))
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.98 else np.array([0.0, 1.0, 0.0])
    r = np.cross(up, d); r /= max(1e-9, np.linalg.norm(r))
    u = np.cross(d, r)
    target = np.asarray(target, np.float64)
    texel = 2.0 * extent / float(size)
    cx, cy = float(target @ r), float(target @ u)
    snap = target + (round(cx / texel) * texel - cx) * r + (round(cy / texel) * texel - cy) * u
    eye = snap + d * (extent * 2.0)
    view = np.identity(4, np.float32)
    view[0, :3], view[1, :3], view[2, :3] = r, u, d
    view[:3, 3] = -view[:3, :3] @ eye
    proj = ortho(-extent, extent, -extent, extent, 0.1, extent * 4.0)
    return LightMatrix(view, proj, (proj @ view).astype(np.float32), float(extent), float(texel))


def frustum_planes(m):
    p = np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1],
                  m[3] + m[2], m[3] - m[2]])
    return p / np.linalg.norm(p[:, :3], axis=1, keepdims=True)


# ---------------------------------------------------------------- shadow map
class ShadowMap:
    def __init__(self, gl, size=2048):
        self.gl, self.size = gl, int(size)
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24,
                        self.size, self.size, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR),
                     (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                     (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER),
                     (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER),
                     (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_COMPARE_REF_TO_TEXTURE),
                     (gl.GL_TEXTURE_COMPARE_FUNC, gl.GL_LEQUAL)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
        gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, np.ones(4, np.float32))
        self.fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                  gl.GL_TEXTURE_2D, self.tex, 0)
        gl.glDrawBuffer(gl.GL_NONE); gl.glReadBuffer(gl.GL_NONE)
        self.complete = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def begin(self):
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.size, self.size)
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND); gl.glClear(gl.GL_DEPTH_BUFFER_BIT)

    def end(self, w, h):
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0); gl.glViewport(0, 0, w, h)

    def bind_texture(self, unit=1):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)

    def release_texture(self, unit=1):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)


# ---------------------------------------------------------------- chunk VAO
@dataclass
class ChunkVAO:
    key: tuple
    vao_b: int
    vao_e: int
    vao_g: int
    vao_w: int
    vao_t: int
    bufs: list
    b_count: int
    e_count: int
    g_count: int
    w_count: int
    t_count: int
    offset: np.ndarray
    scale: np.ndarray
    sphere: np.ndarray
    nbytes: int
    born: float
    last_used: float


# ---------------------------------------------------------------- renderer
class Renderer:
    GROW_TIME = 0.7
    FOG_SPAN = 0.9

    def __init__(self, width, height, shadow_size=2048, shadows=True):
        import OpenGL.GL as gl
        self.gl = gl
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glClearColor(*BG_COLOR, 1.0)

        self.prog_sky = self._program(SKY_VS, SKY_FS)
        self.prog_building = self._program(BUILDING_VS, BUILDING_FS)
        self.prog_line = self._program(LINE_VS, LINE_FS)
        self.prog_ground = self._program(GROUND_VS, GROUND_FS)
        self.prog_terrain = self._program(BUILDING_VS, TERRAIN_FS)
        self.prog_water = self._program(WATER_VS, WATER_FS)
        self.prog_shadow = self._program(SHADOW_VS, SHADOW_FS)
        self.prog_tree = self._program(TREE_VS, TREE_FS)
        self.prog_shadow_tree = self._program(SHADOW_TREE_VS, SHADOW_FS)

        self._uloc = {}
        self.sky = compute_sky(13.0)
        self.sun_dir = self.sky.light_dir.copy()
        self.shadow = ShadowMap(gl, shadow_size) if shadows else None
        self.shadow_bias = 0.0011
        self.stream_radius = 1000.0
        self.last_drawn = self.last_tris = self.last_shadow_draws = 0

        # Base plane at sea level (Z = 0.00). Water is at 0.00 too, land at 0.04+;
        # the flat-layer pass relies on paint order for layering.
        quad = np.array([[-1, -1, 0.0], [ 1, -1, 0.0], [ 1,  1, 0.0],
                         [-1, -1, 0.0], [ 1,  1, 0.0], [-1,  1, 0.0]], np.float32)
        self.vao_ground, self._ground_bufs = self._vao(quad, [(0, 3)])

        sky_quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], np.float32)
        self.vao_sky, self._sky_bufs = self._vao(sky_quad, [(0, 2)])

        self.tree_vertices, self.tree_indices = build_tree_template()
        self.tree_indices_count = int(len(self.tree_indices))
        self.tree_mesh_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tree_mesh_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.tree_vertices.nbytes,
                        self.tree_vertices, gl.GL_STATIC_DRAW)
        self.tree_mesh_ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.tree_mesh_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.tree_indices.nbytes,
                        self.tree_indices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        self.resize(width, height)

    def _program(self, vs_src, fs_src):
        from OpenGL.GL import shaders
        gl = self.gl
        return shaders.compileProgram(
            shaders.compileShader(vs_src, gl.GL_VERTEX_SHADER),
            shaders.compileShader(fs_src, gl.GL_FRAGMENT_SHADER),
            validate=False)

    def _vao(self, data, layout, indices=None):
        gl = self.gl
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        vbo = gl.glGenBuffers(1); bufs = [vbo]
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
        stride, off = data.shape[1] * 4, 0
        for loc, size in layout:
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                     ctypes.c_void_p(off))
            off += size * 4
        if indices is not None:
            ebo = gl.glGenBuffers(1); bufs.append(ebo)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)
        return vao, bufs

    def _tree_vao(self, instances):
        gl = self.gl
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tree_mesh_vbo)
        stride = self.tree_vertices.shape[1] * 4
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                     ctypes.c_void_p(off))
        inst_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, inst_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, instances.nbytes, instances, gl.GL_STATIC_DRAW)
        ist = 32
        gl.glEnableVertexAttribArray(4)
        gl.glVertexAttribPointer(4, 3, gl.GL_FLOAT, gl.GL_FALSE, ist, ctypes.c_void_p(0))
        gl.glVertexAttribDivisor(4, 1)
        gl.glEnableVertexAttribArray(5)
        gl.glVertexAttribPointer(5, 2, gl.GL_FLOAT, gl.GL_FALSE, ist, ctypes.c_void_p(12))
        gl.glVertexAttribDivisor(5, 1)
        gl.glEnableVertexAttribArray(6)
        gl.glVertexAttribPointer(6, 3, gl.GL_FLOAT, gl.GL_FALSE, ist, ctypes.c_void_p(20))
        gl.glVertexAttribDivisor(6, 1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.tree_mesh_ebo)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        return vao, inst_vbo

    def _loc(self, prog, name):
        key = (prog, name)
        if key not in self._uloc:
            self._uloc[key] = self.gl.glGetUniformLocation(prog, name)
        return self._uloc[key]

    def _mat(self, prog, name, m):
        self.gl.glUniformMatrix4fv(self._loc(prog, name), 1, self.gl.GL_TRUE,
                                   np.ascontiguousarray(m, np.float32))

    def _vec3(self, prog, name, v):
        self.gl.glUniform3f(self._loc(prog, name), *[float(x) for x in v])

    def _float(self, prog, name, v):
        self.gl.glUniform1f(self._loc(prog, name), float(v))

    @staticmethod
    def _u32(m):
        return np.ascontiguousarray(m.indices, dtype=np.uint32)

    def upload_chunk(self, key, b_mesh, g_mesh, w_mesh, t_mesh, origin, scale=(1.0, 1.0)):
        vao_b = vao_e = vao_g = vao_w = vao_t = 0
        bufs = []; b_count = e_count = g_count = w_count = t_count = 0; nbytes = 0
        if b_mesh:
            vao_b, bb = self._vao(b_mesh.vertices, [(0, 3), (1, 3), (2, 3), (3, 1), (4, 3), (5, 2)], self._u32(b_mesh))
            vao_e, be = self._vao(b_mesh.edges, [(0, 3), (1, 3)])
            bufs += bb + be
            b_count = len(b_mesh.indices); e_count = len(b_mesh.edges)
            nbytes += b_mesh.vertices.nbytes + b_mesh.indices.nbytes + b_mesh.edges.nbytes
        if g_mesh:
            vao_g, bg = self._vao(g_mesh.vertices, [(0, 3), (1, 3), (2, 3), (3, 1), (4, 3), (5, 2)], self._u32(g_mesh))
            bufs += bg; g_count = len(g_mesh.indices)
            nbytes += g_mesh.vertices.nbytes + g_mesh.indices.nbytes
        if w_mesh:
            vao_w, bw = self._vao(w_mesh.vertices, [(0, 3), (1, 3), (2, 3), (3, 1), (4, 3), (5, 2)], self._u32(w_mesh))
            bufs += bw; w_count = len(w_mesh.indices)
            nbytes += w_mesh.vertices.nbytes + w_mesh.indices.nbytes
        if t_mesh is not None and len(t_mesh) > 0:
            vao_t, inst_vbo = self._tree_vao(t_mesh)
            bufs.append(inst_vbo); t_count = int(len(t_mesh)); nbytes += int(t_mesh.nbytes)
        now = time.time()
        sphere = np.array([origin[0], origin[1], 10.0, 1500.0], np.float64)
        return ChunkVAO(
            key=key, vao_b=vao_b, vao_e=vao_e, vao_g=vao_g, vao_w=vao_w, vao_t=vao_t, bufs=bufs,
            b_count=b_count, e_count=e_count, g_count=g_count, w_count=w_count, t_count=t_count,
            offset=np.array([origin[0], origin[1], 0.0], np.float32),
            scale=np.array(scale, np.float32), sphere=sphere, nbytes=int(nbytes),
            born=now, last_used=now)

    def free_chunk(self, chunk):
        gl = self.gl
        for vao in (chunk.vao_b, chunk.vao_e, chunk.vao_g, chunk.vao_w, chunk.vao_t):
            if vao: gl.glDeleteVertexArrays(1, [vao])
        for buf in chunk.bufs:
            if buf: gl.glDeleteBuffers(1, [buf])
        chunk.bufs = []

    def clear(self):
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT | self.gl.GL_DEPTH_BUFFER_BIT)

    def resize(self, w, h):
        self.width, self.height = max(w, 1), max(h, 1)
        self.gl.glViewport(0, 0, self.width, self.height)

    def _grow(self, chunk, now=None):
        t = min(1.0, ((time.time() if now is None else now) - chunk.born) / self.GROW_TIME)
        return 1.0 - (1.0 - t) ** 3

    def shadow_extent(self, cam):
        return float(np.clip(cam.distance * 1.4 + 150.0, 250.0, 2800.0))

    def _render_sky(self, cam, sky, view, proj):
        gl = self.gl
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE); gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        p = self.prog_sky
        gl.glUseProgram(p)
        inv_vp = np.linalg.inv((proj @ view).astype(np.float64)).astype(np.float32)
        self._mat(p, "uInvViewProj", inv_vp)
        self._vec3(p, "uCamPos", cam.position)
        self._vec3(p, "uSunDir", sky.sun_dir)
        self._vec3(p, "uMoonDir", sky.moon_dir)
        self._vec3(p, "uFogColor", sky.fog)
        self._float(p, "uSunIntensity", sky.sun_intensity)
        self._float(p, "uMoonIntensity", sky.moon_intensity)
        self._float(p, "uMoonPhase", sky.moon_phase)
        self._float(p, "uNightFactor", sky.night)
        self._float(p, "uTime", time.time())
        gl.glBindVertexArray(self.vao_sky)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)

    def _shadow_pass(self, casters, light):
        gl = self.gl
        self.shadow.begin()
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(2.0, 4.0)
        now = time.time()
        p = self.prog_shadow
        gl.glUseProgram(p)
        self._mat(p, "uLightSpaceMatrix", light.matrix)
        for c in casters:
            if c.b_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
            self._float(p, "uGrow", self._grow(c, now))
            gl.glBindVertexArray(c.vao_b)
            gl.glDrawElements(gl.GL_TRIANGLES, c.b_count, gl.GL_UNSIGNED_INT, None)
        p = self.prog_shadow_tree
        gl.glUseProgram(p)
        self._mat(p, "uLightSpaceMatrix", light.matrix)
        for c in casters:
            if c.t_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
            self._float(p, "uGrow", self._grow(c, now))
            gl.glBindVertexArray(c.vao_t)
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.tree_indices_count,
                                       gl.GL_UNSIGNED_INT, None, c.t_count)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glBindVertexArray(0)
        self.shadow.end(self.width, self.height)
        self.last_shadow_draws = len(casters)

    def _shadow_uniforms(self, p, light, sky):
        strength = sky.strength if (self.shadow is not None and light is not None) else 0.0
        if light is not None: self._mat(p, "uLightSpaceMatrix", light.matrix)
        self.gl.glUniform1i(self._loc(p, "uShadowMap"), 1)
        self._float(p, "uShadowTexel", 1.0 / self.shadow.size if self.shadow else 1.0)
        self._float(p, "uShadowTexelWorld", light.texel if light else 1.0)
        self._float(p, "uShadowBias", self.shadow_bias)
        self._float(p, "uShadowStrength", strength)

    def draw(self, cam, chunks, wireframe=False, show_edges=True, sky=None):
        gl = self.gl
        sky = sky or self.sky or compute_sky(13.0)
        self.sky, self.sun_dir = sky, sky.light_dir.copy()
        gl.glClearColor(*[float(c) for c in sky.sky], 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        view, proj = cam.view(), cam.projection(self.width / self.height)
        eye = cam.position
        fog_near = cam.distance * 1.2
        fog_far = cam.distance + self.stream_radius * self.FOG_SPAN
        night = float(sky.night)

        self._render_sky(cam, sky, view, proj)

        chunks = list(chunks)
        if chunks:
            planes = frustum_planes(proj.astype(np.float64) @ view.astype(np.float64))
            sph = np.array([c.sphere for c in chunks], np.float64)
            inside = (sph[:, :3] @ planes[:, :3].T + planes[:, 3] >= -sph[:, 3:4]).all(axis=1)
            vis = [c for c, ok in zip(chunks, inside) if ok]
        else:
            vis = []
        self.last_drawn = len(vis)
        self.last_tris = sum(c.b_count for c in vis) // 3
        now = time.time()

        light = None
        if self.shadow is not None and sky.shadows:
            light = light_matrix(np.asarray(cam.target, np.float64), sky.light_dir,
                                 self.shadow_extent(cam), self.shadow.size)
            reach = light.extent + 60.0
            tx, ty = float(cam.target[0]), float(cam.target[1])
            casters = [c for c in chunks
                       if abs(c.offset[0] - tx) <= reach + c.sphere[3]
                       and abs(c.offset[1] - ty) <= reach + c.sphere[3]]
            self._shadow_pass(casters, light)
        else:
            self.last_shadow_draws = 0
        if self.shadow is not None:
            self.shadow.bind_texture(1)

        def common(p):
            self._mat(p, "uView", view); self._mat(p, "uProj", proj)
            self._vec3(p, "uCamPos", eye); self._vec3(p, "uFogColor", sky.fog)
            self._float(p, "uFogNear", fog_near); self._float(p, "uFogFar", fog_far)
            self._vec3(p, "uSunDir", sky.light_dir); self._vec3(p, "uSunColor", sky.light)
            self._vec3(p, "uAmbColor", sky.amb); self._float(p, "uNightFactor", night)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_CULL_FACE)

        # Flat layers (base, water, terrain) - paint order, no depth write.
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE)

        # 1. Base sea-level plane at Z = 0.00
        gl.glUseProgram(self.prog_ground)
        common(self.prog_ground)
        self._shadow_uniforms(self.prog_ground, light, sky)
        gl.glUniform2f(self._loc(self.prog_ground, "uCenter"),
                       float(cam.target[0]), float(cam.target[1]))
        self._float(self.prog_ground, "uSize", (cam.distance + self.stream_radius) * 2.5)
        gl.glBindVertexArray(self.vao_ground)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # 2. Water at Z = 0.00
        p = self.prog_water
        gl.glUseProgram(p); common(p)
        self._float(p, "uTime", time.time())
        for c in vis:
            if c.w_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
            gl.glBindVertexArray(c.vao_w)
            gl.glDrawElements(gl.GL_TRIANGLES, c.w_count, gl.GL_UNSIGNED_INT, None)

        # 3. Terrain (parks/roads/rails) at Z >= 0.04
        p = self.prog_terrain
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        for c in vis:
            if c.g_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
            self._float(p, "uGrow", 1.0)
            gl.glBindVertexArray(c.vao_g)
            gl.glDrawElements(gl.GL_TRIANGLES, c.g_count, gl.GL_UNSIGNED_INT, None)

        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_LEQUAL)

        # 4. Buildings
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if wireframe else gl.GL_FILL)
        if not wireframe:
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
            gl.glPolygonOffset(2.0, 4.0)
        p = self.prog_building
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        for c in vis:
            if c.b_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
            self._float(p, "uGrow", self._grow(c, now))
            gl.glBindVertexArray(c.vao_b)
            gl.glDrawElements(gl.GL_TRIANGLES, c.b_count, gl.GL_UNSIGNED_INT, None)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

        # 5. Trees
        if not wireframe and any(c.t_count > 0 for c in vis):
            gl.glEnable(gl.GL_CULL_FACE)
            p = self.prog_tree
            gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
            for c in vis:
                if c.t_count == 0: continue
                self._vec3(p, "uOffset", c.offset)
                gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
                self._float(p, "uGrow", self._grow(c, now))
                gl.glBindVertexArray(c.vao_t)
                gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.tree_indices_count,
                                           gl.GL_UNSIGNED_INT, None, c.t_count)

        # 6. Edges
        if show_edges and not wireframe:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            p = self.prog_line
            gl.glUseProgram(p); common(p)
            for c in vis:
                if c.e_count == 0: continue
                self._vec3(p, "uOffset", c.offset)
                gl.glUniform2f(self._loc(p, "uScale"), float(c.scale[0]), float(c.scale[1]))
                self._float(p, "uGrow", self._grow(c, now))
                gl.glBindVertexArray(c.vao_e)
                gl.glDrawArrays(gl.GL_LINES, 0, c.e_count)
        gl.glBindVertexArray(0)
        if self.shadow is not None:
            self.shadow.release_texture(1)