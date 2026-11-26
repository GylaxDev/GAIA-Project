"""Sky atmosphere + OpenGL renderer."""
from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass

import numpy as np
import zlib
from .lod import CITY_VERTEX, WATER_VERTEX, pack_vertices, sphere_mask

from ..core.config import BG_COLOR, MOON_COLOR
from ..core.photometry import SUN_IRRADIANCE, NITS_PER_SCENE_UNIT, MOON_LUX, NIGHT_EXPOSURE_STOPS, DAY_EXPOSURE
from ..world.data_fetcher import build_tree_template
from ..world.foliage import build_leaf_textures, build_leaf_mipmaps
from ..world.tree_models import (TREE_CENTER_Z, TREE_HEIGHT, TREE_RADIUS,
                                 build_tree_billboard_template, tree_lod_levels)
from ..world.tree_impostors import (TREE_IMPOSTOR_ATLAS, TREE_IMPOSTOR_TILE,
                                    TREE_IMPOSTOR_VIEWS, build_tree_impostor_mipmaps)
from ..world.ground_textures import build_ground_textures, build_ground_mipmaps
from ..world.street_furniture import streetpole_template, LAMP_LENS_DROP
from ..world.water_ripples import build_water_ripples, build_water_ripple_mipmaps
from ..world.mesh_extruder import (MeshData, build_all_vehicle_templates,
                            _HEADLIGHT_COLOR, _TAILLIGHT_COLOR, _GROUND_GLOW_COLOR)
from .lights import LightField
from .billboard_textures import (TEXTURE_ASPECT as BILLBOARD_ASPECT,
                                 load_billboard_images)
from ..peds.pedestrian_renderer import PedestrianRenderer
from .spill import SpillGrid
from ..vehicles.vehicle_models import LED_COLOR as _LED_COLOR, animation_metadata
from ..vehicles.vehicle_instances import (VEHICLE_STRIDE, VEHICLE_BYTES, INSTANCE_ATTRIBUTES, normalize_instances,
                                          LENGTHS as _CAR_LENGTHS, WIDTHS as _CAR_WIDTHS, WHEELBASES as _CAR_WHEELBASES)
from ..traffic.traffic_signals import build_signal_template, SIGNAL_CENTER, SIGNAL_RADIUS
from ..world.weather_system import PrecipitationRenderer, RainOcclusionMap, wind_displacement, HEADLIGHT_BEAMS
from ..core.astronomy import REFERENCE_DATE, celestial_state, parse_date
from ..core.atmosphere import horizon_display
from .starfield import StarField
from ..shaders.cityhaze import HAZE_FS, HAZE_APPLY_FS, HAZE_DEFAULTS, HAZE_GLOW_TINT, HAZE_MAX_LAMPS
from ..shaders import (
    SKY_VS, SKY_FS, SKY_LUT_VS, SKY_LUT_FS,
    BUILDING_VS, BUILDING_FS,
    LINE_VS, LINE_FS,
    GROUND_VS, GROUND_FS,
    TERRAIN_FS,
    WATER_VS, WATER_FS,
    TREE_VS, TREE_FS, SHADOW_TREE_FS, TREE_IMPOSTOR_VS, TREE_IMPOSTOR_FS,
    SHADOW_VS, SHADOW_FS, SHADOW_TREE_VS,
    STREETPOLE_VS, STREETPOLE_FS,
    SKID_VS, SKID_FS,
    CONTACT_VS, CONTACT_FS,
    VEHICLE_VS, VEHICLE_FS, VEHICLE_GLASS_MAT_FS, VEHICLE_SHADOW_VS, VEHICLE_SHADOW_FS,
    NEON_VS, NEON_FS,
    BLOOM_VS, BLOOM_PREFILTER_FS, BLOOM_BLUR_FS, BLOOM_COMPOSITE_FS, STAR_FS, STAR_SRC_FS,
    STAR_STREAK_FS, SUN_FLARE_FS,
    AE_METER_FS, AE_ADAPT_FS, BLOOM_DOWN_FS,
    SSR_FS, mrt_fs, BUILDING_MAT, TERRAIN_MAT, WATER_MAT, VEHICLE_MAT, AO_FS,
)

# ---------------------------------------------------------------- street lamps
_LAMP_TYPES = np.array([
    [1.00, 0.40, 0.07],   # 0  high-pressure sodium
    [1.00, 0.58, 0.24],   # 1  halogen / metal-halide
    [0.95, 0.85, 0.72],   # 2  LED neutral
    [0.72, 0.84, 1.00],   # 3  LED cool
], np.float32)
_LAMP_GAIN = np.array([1.60, 1.25, 1.00, 1.00], np.float32)
_LAMP_ARM_LEN_TALL, _LAMP_ARM_LEN_SHORT = 2.4, 1.7
_LAMP_LENS_DROP = LAMP_LENS_DROP

def _hash01(x, y, salt):
    h = np.sin(x.astype(np.float64) * 12.9898 + y.astype(np.float64) * 78.233 + salt * 37.719) * 43758.5453
    return (h - np.floor(h)).astype(np.float32)

def lamp_colors(wx, wy):
    wx = np.asarray(wx, np.float64); wy = np.asarray(wy, np.float64)
    u = _hash01(np.floor(wx / 160.0), np.floor(wy / 160.0), 1.0)
    district = np.where(u < 0.30, 0, np.where(u < 0.55, 1, np.where(u < 0.80, 2, 3)))
    odd = _hash01(wx, wy, 2.0) < 0.15
    rnd = np.minimum((_hash01(wx, wy, 3.0) * 4.0).astype(np.int32), 3)
    kind = np.where(odd, rnd, district)
    col = _LAMP_TYPES[kind] * _LAMP_GAIN[kind][:, None]
    col = col * (0.88 + 0.24 * _hash01(wx, wy, 4.0))[:, None]
    tint = 1.0 + 0.08 * (np.stack([_hash01(wx, wy, 5.0), _hash01(wx, wy, 6.0),
                                   _hash01(wx, wy, 7.0)], axis=1) - 0.5)
    col = col * tint
    col[_hash01(wx, wy, 8.0) < 0.04] = 0.0
    return col.astype(np.float32)

def _box(cx, cy, cz, hx, hy, hz, part):
    faces = [((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
             ((-1, 0, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
             ((0, 1, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
             ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
             ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
             ((0, 0, -1), [(-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)])]
    V, I = [], []
    for n, quad in faces:
        b = len(V)
        for (sx, sy, sz) in quad:
            V.append([cx + sx * hx, cy + sy * hy, cz + sz * hz, n[0], n[1], n[2], part])
        I += [b, b + 1, b + 2, b, b + 2, b + 3]
    return np.asarray(V, np.float32), np.asarray(I, np.uint32)

def build_streetpole_template():
    return streetpole_template()

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
    sky_glow: float = 0.0
    exposure: float = 1.0
    atmosphere_intensity: float = 0.0
    date: str = "2026-04-22"
    julian_day: float = 2451545.0
    sidereal_angle: float = 0.0
    equatorial_to_world: np.ndarray = None
    moon_radius: float = .0045
    moon_brightness: float = 0.0
    star_visibility: float = 0.0
    star_limit: float = 6.5
    milky_way_visibility: float = 0.0
    light_pollution: float = .35
    observer_height: float = 100.0
    aerosol_density: float = 1.0
    cloud_sun: np.ndarray = None

    @property
    def is_day(self) -> bool: return self.alt > 0.0
    @property
    def stamp(self) -> str:
        m = int(round(self.hours * 60.0)) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"

def _atmosphere_horizon(sun_dir, sun_intensity, exposure, **options):
    return horizon_display(sun_dir, sun_intensity, exposure, **options)

def compute_sky(hours, lat_deg=40.758, decl_deg=12.0, *, date=None, lon_deg=-73.9855,
                utc_offset=None, altitude_m=100.0, light_pollution=.35) -> SkyState:
    """Date/location-aware sky. A date-less call preserves the legacy solar track.

    The app always supplies its calendar date. Old offline fixtures can retain
    their fixed 12-degree solar declination; their stars/Moon use REFERENCE_DATE.
    Without an explicit UTC offset, hours are local mean solar time.
    """
    if not math.isfinite(float(hours)):
        raise ValueError('sky hours must be finite')
    hours = float(hours) % 24.0
    calendar = parse_date(date) if date is not None else REFERENCE_DATE
    celestial = celestial_state(calendar, hours, lat_deg, lon_deg, utc_offset, altitude_m)
    if date is None and decl_deg is not None:
        alt, az = solar_position(lat_deg, hours, decl_deg)
        sun_vec = sky_vector(alt, az).astype(np.float32)
    else:
        sun_vec = celestial.sun_dir.astype(np.float32)
        alt = math.asin(float(np.clip(sun_vec[2], -1., 1.)))
        az = math.atan2(float(sun_vec[0]), float(sun_vec[1]))
    deg = math.degrees(alt)
    moon_vec = celestial.moon_dir.astype(np.float32)
    light_dir = sun_vec if deg > -.83 else moon_vec
    sun_gain = _smoothstep(-.83, 5.0, deg)
    # Earth's shadow, not a direct-light fade, determines which upper air is lit.
    atmosphere_gain = _smoothstep(-20., -16., deg)
    warm = _smoothstep(-3.0, 15.0, deg)
    sun_tint = np.array([1.0, 0.58 + 0.38 * warm, 0.30 + 0.56 * warm], np.float32)
    sun_color = sun_tint * SUN_IRRADIANCE * sun_gain
    moon_alt = math.degrees(math.asin(float(np.clip(moon_vec[2], -1., 1.))))
    moon_visible = _smoothstep(-.8, 1.8, moon_alt)
    moon_gain = celestial.moon_brightness * moon_visible
    moon_color = np.array(MOON_COLOR, np.float32) * (MOON_LUX / NITS_PER_SCENE_UNIT) * moon_gain
    moon_color *= _smoothstep(-1., -7., deg)
    night = 1.0 - _smoothstep(-6.0, 5.0, deg)
    # Night vision comes from camera exposure, rather than making every lamp / wall emit.
    exposure = DAY_EXPOSURE * 2.0 ** (NIGHT_EXPOSURE_STOPS * _smoothstep(0.0, -12.0, deg))
    pollution = float(np.clip(light_pollution, 0., 1.))
    if not math.isfinite(pollution):
        raise ValueError('light pollution must be finite')
    sky_glow = .3 * pollution
    height = float(np.clip(altitude_m, 2., 80000.))
    fog = _atmosphere_horizon(sun_vec, atmosphere_gain, exposure,
                              height=height, night=night, glow=sky_glow).astype(np.float32)
    amb_day    = np.array([0.17, 0.215, 0.285], np.float32)
    amb_sunset = np.array([0.29, 0.205, 0.18], np.float32)
    # Diffuse urban sky/street bounce, in scene radiance units. Keeps unlit
    # facades readable while retaining a large contrast to actual luminaires.
    amb_night  = np.array([8.0, 10.0, 12.0], np.float32) / NITS_PER_SCENE_UNIT
    
    w_day    = _smoothstep(-2.0, 18.0, deg)
    
    twilight = 2.0 ** (min(deg, 0.0) * 0.95)
    amb = (amb_day * w_day + amb_sunset * (1.0 - w_day)) * twilight + amb_night * (1.0 - twilight)
    # Clear daylight needs separation between direct sun and diffuse sky fill.
    # Fade the trim in after sunrise, retaining civil-twilight and night fill.
    amb *= 1.0 - .28 * _smoothstep(5.0, 22.0, deg)

    light = sun_color if deg > -.83 else moon_color
    # Bright stars emerge first; faint catalog stars wait for astronomical night.
    dark_limit = 6.5 - 3.5 * pollution - 1.5 * moon_gain
    twilight_limit = -2. + 8.5 * _smoothstep(-2., -18., deg)
    strength = _smoothstep(0.5, 7.0, deg)

    return SkyState(
        hours=hours, alt=alt, az=az,
        sun_dir=sun_vec, moon_dir=moon_vec, light_dir=light_dir.astype(np.float32),
        sky=fog.copy(), fog=fog, amb=amb, light=light,
        shadows=strength > 0.002, strength=strength,
        moon_phase=celestial.moon_phase, night=night,
        sun_intensity=float(sun_gain), moon_intensity=float(moon_visible),
        sky_glow=sky_glow, exposure=exposure, atmosphere_intensity=atmosphere_gain,
        date=calendar.isoformat(), julian_day=celestial.julian_day, sidereal_angle=celestial.sidereal_angle,
        equatorial_to_world=celestial.equatorial_to_world.astype(np.float32),
        moon_radius=celestial.moon_radius, moon_brightness=celestial.moon_brightness,
        star_visibility=_smoothstep(-4., -12., deg), star_limit=min(dark_limit, twilight_limit),
        milky_way_visibility=_smoothstep(-12., -18., deg) * (1. - pollution) ** 3 * (1. - .90 * moon_gain),
        light_pollution=pollution, observer_height=height,
        cloud_sun=sun_tint * SUN_IRRADIANCE * _smoothstep(-6., -1., deg),
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

    @property
    def depth_span(self):
        return 2.0 / float(np.linalg.norm(self.matrix[2, :3]))

def light_matrix(target, light_dir, extent, size, *, depth_extent=None):
    d = np.asarray(light_dir, np.float64)
    d = d / max(1e-9, np.linalg.norm(d))
    up = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.98 else np.array([0.0, 1.0, 0.0])
    r = np.cross(up, d); r /= max(1e-9, np.linalg.norm(r))
    u = np.cross(d, r)
    target = np.asarray(target, np.float64)
    texel = 2.0 * extent / float(size)
    # Re-centre the box in big whole-texel steps (~6 % of its width) instead of every texel:
    # flying around used to invalidate the cached shadow map almost every frame.  Whole
    # texels keep the shadows perfectly stable (no shimmer).
    quant = texel * max(1.0, round(extent * 0.0625 / texel))
    cx, cy, cd = float(target @ r), float(target @ u), float(target @ d)
    # Snap all three light-space axes.  Only snapping x/y left the box's depth position
    # following the camera continuously, so the matrix (the shadow cache key) changed on
    # every frame of camera movement and the whole shadow map was re-rendered each time.
    snap = (round(cx / quant) * quant) * r + (round(cy / quant) * quant) * u \
         + (round(cd / quant) * quant) * d
    # Close-map XY density must not clip tall/offscreen casters along the light
    # ray. Its depth coverage is independent of its small rooftop footprint.
    depth_extent = max(float(extent), float(depth_extent) if depth_extent is not None else float(extent))
    eye = snap + d * (depth_extent * 2.0)
    view = np.identity(4, np.float32)
    view[0, :3], view[1, :3], view[2, :3] = r, u, d
    view[:3, 3] = -view[:3, :3] @ eye
    proj = ortho(-extent, extent, -extent, extent, 0.1, depth_extent * 4.0)
    return LightMatrix(view, proj, (proj @ view).astype(np.float32), float(extent), float(texel))

def frustum_planes(m):
    p = np.array([m[3] + m[0], m[3] - m[0], m[3] + m[1], m[3] - m[1],
                  m[3] + m[2], m[3] - m[2]])
    return p / np.linalg.norm(p[:, :3], axis=1, keepdims=True)

# ---------------------------------------------------------------- shadow map
class ShadowMap:
    def __init__(self, gl, size=4096):
        self.gl, self.requested_size = gl, int(size)
        if self.requested_size <= 0:
            raise ValueError('shadow map size must be positive')
        self.tex = self.fbo = 0
        self.complete = False
        self.allocation_attempts = []
        previous_fbo = int(gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING))
        previous_tex = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        limit = max(1, int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)))
        self.size = min(self.requested_size, limit)
        minimum = min(512, self.size)
        error = None
        try:
            while True:
                self.allocation_attempts.append(self.size)
                try:
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
                    if self.complete: break
                    raise RuntimeError(f'incomplete {self.size}-pixel shadow framebuffer')
                except Exception as exc:
                    error = exc
                    self.close()
                    if self.size <= minimum:
                        raise RuntimeError('could not allocate a complete shadow map') from error
                    self.size = max(minimum, self.size // 2)
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, previous_fbo)
            gl.glBindTexture(gl.GL_TEXTURE_2D, previous_tex)

    @property
    def nbytes(self):
        # Conservative GPU accounting: D24 generally occupies a 32-bit texel.
        return self.size * self.size * 4 if self.complete else 0

    def close(self):
        if self.fbo: self.gl.glDeleteFramebuffers(1, [self.fbo])
        if self.tex: self.gl.glDeleteTextures([self.tex])
        self.fbo = self.tex = 0
        self.complete = False

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
    vao_v: list
    bufs: list
    b_count: int
    e_count: int
    g_count: int
    w_count: int
    t_count: int
    l_count: int
    v_counts: list
    offset: np.ndarray
    scale: np.ndarray
    sphere: np.ndarray
    nbytes: int
    born: float
    last_used: float
    lights_cpu: np.ndarray = None
    vao_p: int = 0
    vao_s: int = 0
    vao_prop: int = 0
    s_count: int = 0
    p_count: int = 0
    neon_cpu: np.ndarray = None
    aabb: np.ndarray = None      # tight world-space bounds (minxyz, maxxyz)
    cell_b: tuple = None         # (first, count, world boxes) per building cell
    cell_g: tuple = None
    cell_w: tuple = None         # displaced-water cells, conservatively padded
    edges_cpu: np.ndarray = None  # outline lines, uploaded on first use (see _ensure_edges)
    edges_packed: bool = False
    vao_b_lod: int = 0
    b_lod_count: int = 0
    cell_b_lod: tuple = None
    cell_prop: tuple = None
    cell_s: tuple = None
    tree_cpu: np.ndarray = None
    pole_cpu: np.ndarray = None
    parked_cpu: list = None

# ---------------------------------------------------------------- renderer
class Renderer:
    GROW_TIME = 0.7
    FOG_SPAN = 0.9

    VEHICLE_SHADOW_UNIT = 8
    CONTACT_FLOATS = 13         # origin3 axisX3 axisY3 (halfLen halfWidth wheelbase visibility)
    CONTACT_MAX = 1024
    NEAR_SHADOW_UNIT = 9
    GROUND_TEXTURE_UNIT = 10
    WATER_RIPPLE_UNIT = 12
    BILLBOARD_UNIT = 13

    def __init__(self, width, height, shadow_size=4096, shadows=True, *, vehicle_shadow_size=4096,
                  near_shadow_size=2048):
        import OpenGL.GL as gl
        self.gl = gl
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glClearColor(*BG_COLOR, 1.0)
        
        # Depth clamping turns geometry behind/inside the eye into full-screen
        # sheets. Real near clipping is required, especially in free flight.
        gl.glDisable(gl.GL_DEPTH_CLAMP)
        self.reverse_z = bool(getattr(gl, 'glClipControl', None))
        self._reverse_active = False

        self.prog_sky         = self._program(SKY_VS, SKY_FS)
        ambient = "max(0.0, dot(ambientLit, vec3(0.2126, 0.7152, 0.0722))) * uExposure * (1.0 - f)"
        self.prog_building    = self._program(BUILDING_VS, mrt_fs(BUILDING_FS, BUILDING_MAT, ambient))
        self.prog_line        = self._program(LINE_VS, LINE_FS)
        self.prog_ground      = self._program(GROUND_VS, mrt_fs(GROUND_FS))
        self.prog_terrain     = self._program(BUILDING_VS, mrt_fs(TERRAIN_FS, TERRAIN_MAT, ambient))
        self.prog_water       = self._program(WATER_VS, mrt_fs(WATER_FS, WATER_MAT))
        self.prog_shadow      = self._program(SHADOW_VS, SHADOW_FS)
        self.prog_tree        = self._program(TREE_VS, mrt_fs(TREE_FS, "N * 0.5 + 0.5, 0.0", ambient.replace("ambientLit", "ambient")))
        self.prog_shadow_tree = self._program(SHADOW_TREE_VS, SHADOW_TREE_FS)
        self.prog_shadow_vehicle = self._program(VEHICLE_SHADOW_VS, VEHICLE_SHADOW_FS)
        self.prog_streetpole  = self._program(STREETPOLE_VS, mrt_fs(STREETPOLE_FS, "N * 0.5 + 0.5, 0.0"))
        self.prog_vehicle     = self._program(VEHICLE_VS, mrt_fs(VEHICLE_FS, VEHICLE_MAT))
        self.prog_vehicle_glass = self._program(VEHICLE_VS, VEHICLE_GLASS_MAT_FS)
        self.last_ped_tris = 0
        try:                       # pedestrians are optional: a shader failure must not stop the city
            self.peds = PedestrianRenderer(self)
        except Exception as exc:
            self.peds = None
            print(f"[pedestrians] disabled: {exc}")
        self.prog_skid        = self._program(SKID_VS, SKID_FS)
        self.skid_marks = None        # SkidMarks owned by the app; uploaded lazily
        self._skid_vao = self._skid_vbo = None
        self._skid_generation = -1
        self.prog_contact     = self._program(CONTACT_VS, CONTACT_FS)
        self._contact_vao = self._contact_corner_vbo = self._contact_inst_vbo = None
        self._contact_rows = np.empty((0, self.CONTACT_FLOATS), np.float32)
        self._contact_count = 0
        self.prog_neon        = self._program(NEON_VS, NEON_FS)
        self.prog_bloom_pre   = self._program(BLOOM_VS, BLOOM_PREFILTER_FS)
        self.prog_bloom_blur  = self._program(BLOOM_VS, BLOOM_BLUR_FS)
        self.prog_bloom_comp  = self._program(BLOOM_VS, BLOOM_COMPOSITE_FS)
        self.prog_star        = self._program(BLOOM_VS, STAR_FS)
        self.prog_star_src    = self._program(BLOOM_VS, STAR_SRC_FS)
        self.prog_star_streak = self._program(BLOOM_VS, STAR_STREAK_FS)
        self.prog_sun_flare   = self._program(BLOOM_VS, SUN_FLARE_FS)
        self.prog_bloom_down  = self._program(BLOOM_VS, BLOOM_DOWN_FS)
        self.prog_ae_meter    = self._program(BLOOM_VS, AE_METER_FS)
        self.prog_ae_adapt    = self._program(BLOOM_VS, AE_ADAPT_FS)
        self.prog_ssr         = self._program(BLOOM_VS, SSR_FS)
        self.prog_ao          = self._program(BLOOM_VS, AO_FS)
        self.prog_haze        = self._program(BLOOM_VS, HAZE_FS)
        self.prog_haze_apply  = self._program(BLOOM_VS, HAZE_APPLY_FS)
        # Light sprites are drawn past the pole/geometry LOD so the skyline keeps
        # its lamps when the buildings have faded into fog, but the range is kept
        # inside the fog so the thousands of sprites beyond it are never packed.
        self.lights = LightField(gl, far_clip=1500.0)

        self._uloc = {}
        self._uniform_values = {}
        self._instance_cache = {}
        self._cell_batch_key = None
        self._cell_visibility = {}
        self._cell_draw_cache = {}
        self._parked_key = None
        self.leaf_tex, self.leaf_normal_tex = self._make_leaf_textures()
        self.ground_texture_enabled = True
        self.ground_tex = self._make_ground_texture()
        self.billboard_tex = 0
        self.billboard_count = 0
        self.billboard_layers = []
        self.billboard_texture_bytes = 0
        self.billboard_anisotropy = 1.
        self.billboard_error = None
        try:
            self.billboard_tex = self._make_billboard_texture()
        except Exception as exc:
            self.billboard_error = str(exc)
            print(f"[billboards] textured screens disabled: {exc}")
        self.sky = compute_sky(13.0)
        self.stars_enabled = True
        self.constellations_enabled = False
        self.starfield = None
        self.starfield_error = self.sky_lut_error = None
        self.sky_lut_refreshes = self.last_sky_lut_draws = 0
        self.sun_dir = self.sky.light_dir.copy()
        self.precip = PrecipitationRenderer(gl)
        self.precip_enabled = self.rain_splashes_enabled = self.weather_map_enabled = True
        self.water_waves_enabled = self.water_sss_enabled = True
        self.water_ripples_enabled = True
        self.water_ripple_tex = self.water_texture_bytes = 0
        self.water_texture_error = None
        self.water_texture_anisotropy = 1.
        self.rain_roof = None
        self.weather_map_error = None
        self._rain_roof_active = False
        self.weather_map_refreshes = self.last_weather_map_draws = 0
        self._weather_map_growing = False
        self._weather_map_age = 0
        self.shadow = self.vehicle_shadow = self.near_shadow = None
        self.shadow_error = None
        if shadows:
            try: self.shadow = ShadowMap(gl, shadow_size)
            except RuntimeError as exc: self.shadow_error = str(exc)
            if self.shadow is not None and vehicle_shadow_size > 0:
                try: self.vehicle_shadow = ShadowMap(gl, vehicle_shadow_size)
                except RuntimeError as exc: self.shadow_error = str(exc)
            if self.shadow is not None and near_shadow_size > 0:
                try: self.near_shadow = ShadowMap(gl, near_shadow_size)
                except RuntimeError as exc: self.shadow_error = f'near city shadow: {exc}'
        self.near_shadows_enabled = self.near_shadow is not None
        self._near_light = None
        self._near_shadow_key = None
        self.near_shadow_refreshes = 0
        self.last_near_shadow_draws = self.last_near_shadow_casters = 0
        self.vehicle_shadows_enabled = self.vehicle_shadow is not None
        self._vehicle_light = None
        self._vehicle_shadow_key = None
        self._vehicle_shadow_active = False
        self.vehicle_shadow_refreshes = 0
        self.last_vehicle_shadow_draws = self.last_vehicle_shadow_casters = self.last_vehicle_shadow_tris = 0
        self.last_shadow_casters = 0
        self.shadow_bias = 0.02      # metres, not a fraction of a kilometre-scale light box
        self.stream_radius = 1000.0
        self.last_drawn = self.last_tris = self.last_shadow_draws = 0

        # Depth texture for water depth absorption and shoreline foam
        self.depth_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT32F, width, height, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)

        quad = np.array([[-1, -1, -1.0], [1, -1, -1.0], [1, 1, -1.0],
                         [-1, -1, -1.0], [1, 1, -1.0], [-1, 1, -1.0]], np.float32)
        self.vao_ground, self._ground_bufs = self._vao(quad, [(0, 3)])

        sky_quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], np.float32)
        self.vao_sky, self._sky_bufs = self._vao(sky_quad, [(0, 2)])

        self.pole_vertices, self.pole_indices = build_streetpole_template()
        self.pole_indices_count = int(len(self.pole_indices))
        self.pole_mesh_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pole_mesh_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.pole_vertices.nbytes,
                        self.pole_vertices, gl.GL_STATIC_DRAW)
        self.pole_mesh_ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.pole_mesh_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.pole_indices.nbytes,
                        self.pole_indices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        self.spill = SpillGrid(gl)
        # Three geometric templates plus the 2D impostor card at level 3.
        self.tree_templates = [build_tree_template(level) for level in range(3)]
        self.tree_templates.append(build_tree_billboard_template())
        self.tree_template_vbos, self.tree_template_ebos = [], []
        self.tree_template_counts = []
        for vertices, indices in self.tree_templates:
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
            ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
            self.tree_template_vbos.append(vbo)
            self.tree_template_ebos.append(ebo)
            self.tree_template_counts.append(int(len(indices)))
        # The old inspection/VAO entry points continue to mean the near template.
        self.tree_vertices, self.tree_indices = self.tree_templates[0]
        self.tree_indices_count = self.tree_template_counts[0]
        self.tree_mesh_vbo, self.tree_mesh_ebo = self.tree_template_vbos[0], self.tree_template_ebos[0]
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        tree_batches = [self._tree_vao(np.empty((0, 8), np.float32), level)
                        for level in range(len(self.tree_templates))]
        self.tree_batch_vaos = [vao for vao, _ in tree_batches]
        self.tree_batch_vbos = [vbo for _, vbo in tree_batches]
        self.tree_batch_vao, self.tree_batch_vbo = tree_batches[0]
        self.tree_impostor_tex = 0
        self.tree_impostor_bytes = 0
        self.tree_impostor_error = None
        try:
            self.tree_impostor_tex = self._make_tree_impostor()
        except Exception as exc:
            self.tree_impostor_error = str(exc)
            print(f"[trees] 2D impostors disabled: {exc}")
        self.pole_batch_vao, pb = self._streetpole_vao(np.empty((0, 9), np.float32))
        self.pole_batch_vbo = pb[0]

        self.vehicle_templates = build_all_vehicle_templates()
        self.vehicle_kinds = len(self.vehicle_templates)
        self.lights.set_vehicle_templates(self.vehicle_templates)
        from ..vehicles.vehicle_models import build_vehicle_lod
        for level in (1, 2):
            for kind in range(self.vehicle_kinds):
                self.vehicle_templates[kind + level*self.vehicle_kinds] = build_vehicle_lod(self.vehicle_templates[kind], level, kind=kind)
        # Separate furniture batch; it is never counted, routed or LOD'd as a car.
        self.signal_kind = len(self.vehicle_templates)
        self.vehicle_templates[self.signal_kind] = build_signal_template()
        self.vehicle_template_vbos = []
        self.vehicle_template_ebos = []
        self.vehicle_animation_vbos = []
        self.vehicle_opaque_counts = []
        self.vehicle_glass_counts = []
        self.vehicle_total_counts  = []
        for kind in range(len(self.vehicle_templates)):
            verts, idx, opaque_count = self.vehicle_templates[kind]
            vbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
            animation = animation_metadata(self.vehicle_templates[kind])
            abo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, abo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, animation.nbytes, animation, gl.GL_STATIC_DRAW)
            ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
            self.vehicle_template_vbos.append(vbo)
            self.vehicle_template_ebos.append(ebo)
            self.vehicle_animation_vbos.append(abo)
            self.vehicle_opaque_counts.append(opaque_count)
            self.vehicle_glass_counts.append(getattr(self.vehicle_templates[kind], 'glass_count', 0))
            self.vehicle_total_counts.append(len(idx))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        self.moving_inst_vbos = [gl.glGenBuffers(1) for _ in self.vehicle_templates]
        self.moving_vaos      = [self._build_instance_vao(k) for k in self.vehicle_templates]
        self.moving_counts    = [0] * len(self.vehicle_templates)
        # Coarse, silhouette-preserving body/wheel meshes cast into the car map.
        # They share the scene batches' instance buffers, including their motion.
        self.vehicle_shadow_vaos = [self._make_vehicle_vao(k % self.vehicle_kinds + 2*self.vehicle_kinds,
                                                         self.moving_inst_vbos[k])
                                    for k in range(self.signal_kind)]
        self.vehicle_shadow_counts = [0] * len(self.vehicle_templates)
        self._vehicle_instance_offsets = {}
        self._vehicle_uploads = {}
        self.last_vehicle_upload_bytes = self.last_vehicle_uploads = 0

        self._neon_pos = np.zeros((48, 3), np.float32)
        self._neon_col = np.zeros((48, 3), np.float32)
        self._neon_scr = np.zeros((48, 4), np.float32)   # screens: normal xy, half width, half height
        self._neon_n = 0
        # Night neon sits at ~3-4 exposed HDR units, so the old 3.0 threshold (knee from 2.4) let almost
        # nothing bloom.  1.5 lets signs and lamps glow; strength is kept moderate.
        self.bloom_strength = 0.12
        self.bloom_threshold = 1.5  # 20% soft knee starts at 1.2 exposed HDR units
        self._t0 = time.time()
        # GTA-style grade (V toggles).  grade_strength scales the whole look.
        self.grade_enabled = True
        self.day_glow = 0.06         # optical scatter: only a small fraction of highlight energy
        self.day_threshold = 4.0     # onset at 3.2: ordinary sunlit materials stay below bloom
        self.grade_strength = 1.0
        # Film look: 0 neutral, 1 Colorful, 2 Cinematic.  Cycled with V.
        self.look = 4
        self.LOOK_NAMES = ("Neutral", "Colorful", "Cinematic", "Cyberpunk", "Natural")
        # Eye adaptation.  The scene buffer is linear HDR (already x sky.exposure); the meter
        # averages log luminance on the GPU and a 1x1 texture follows it with an
        # asymmetric time constant (the eye closes down faster than it opens up).
        self.ae_enabled = True
        self.ae_strength = self.AE_DEFAULT_STRENGTH  # 0 = fixed, 1 = full correction
        self.ae_brightness = self.AE_DEFAULT_BRIGHTNESS  # stops on the metered target
        self.ae_key_day = 0.13         # target mean luminance (post-exposure) in daylight
        self.ae_key_night = 0.045     # ... and at night, so dark streets stay dark
        self.ae_min_ev, self.ae_max_ev = -2.5, 3.2  # headroom to open up deep shade (and clip what is outside)
        # Cinematic timing: a slow settle (a few seconds) plus a quick partial reflex, and a dead-band
        # so tiny metering changes never make the exposure hunt.
        self.ae_speed_brighter = 1.2   # 1/s, exposure falling (stepping into the sun)
        self.ae_speed_darker = 0.45    # 1/s, exposure rising (stepping into shade)
        self.ae_dead_band = 0.12       # EV
        self.ae_fast_share = 0.35      # share of a sudden change the quick reflex answers at once
        self.ae_fast_rate = 5.0        # 1/s
        # Filmic contrast (linear light, pivoting on the adaptation key): 1 = none.
        self.contrast = 1.12
        self.ev_comp = 0.0             # manual exposure compensation (stops)
        self.wide_glow = 0.03          # wide-angle lens scatter / halation around bright signs
        self.streak_gain = 0.02        # anamorphic streak, Matrix look only
        self.letterbox = 0.0
        # Lens flare of a stopped-down lens (N toggles, Shift+N switches f/22 <-> f/14)
        self.flare_enabled = True      # lens-flare ghosts and star spikes are on by default (N toggles)
        self.flare_gain = 1.0
        self.flare_max_brightness = 0.25  # exposed-HDR luminance ceiling for the combined flare
        self.flare_fstop = 22.0
        self.flare_blades = 8          # 8 blades -> 8 spikes (odd counts give 2x as many)
        self.flare_star_threshold = 6.0   # exposed HDR level where a highlight starts to make spikes
        self.flare_star_clamp = 10.0      # cap per source so the sun does not drown the glints
        self.flare_threshold = 0.5     # highlight level (after the bloom knee) that starts to flare
        self.flare_star_gain = 1.0     # normalized starburst gain, shared by all diffraction axes
        self.flare_star_length = 0.5   # spike reach scale (1.0 = original length)
        self.flare_ghost_gain = 0.03   # ghosts, rainbow comet and halo
        self.flare_star_day = 0.0     # daylight starburst share: the sun blooms as a soft disc, not spikes
        # Soft circular sun bloom (post composite). 0 disables, 1 is the reference look.
        self.sun_glare = 0.35
        self.sun_glare_tint = (1.0, 0.93, 0.78)
        # Dedicated cinematic sun flare (SUN_FLARE_FS): hot disc, chromatic halo,
        # aperture ghosts and the anamorphic streak.  Independent of the opt-in
        # highlight starburst above and visible by default.
        self.sun_flare = 1.0
        self.sun_flare_tint = (1.0, 0.93, 0.78)
        self.sun_flare_max_brightness = 1.2   # exposed-HDR ceiling before tone mapping
        self.sun_flare_ghost_gain = 1.0
        self.sun_flare_streak_gain = 1.0
        # Floor on aerosol haze (0 = clear 25 km air, 1 = 1.2 km fog). A milky mid-distance
        # haze is a large part of Cinematic look; weather fog can only raise it.
        self.haze_floor = 0.24
        # Volumetric city haze (shaders/cityhaze.py): coloured, light-catching air with height fog,
        # drifting patches, neon halos and headlight shafts.  Tune live through ``city_haze``
        # (keys are HAZE_DEFAULTS); city_haze_enabled switches the pass off.
        self.city_haze_enabled = True
        self.city_haze = dict(HAZE_DEFAULTS)
        self.city_haze_floor = None       # world z of the street; None follows the camera target
        self._haze_floor_s = None
        self.haze_ok = False
        # Cinematic vignette (B toggles).  strength = darkening at the corners.
        self.vignette = 0.10
        self.vignette_start = 0.60
        # Camera motion blur (Settings panel).  Shutter length as a fraction of the frame
        # interval; the frame is gathered along each pixel's reprojection into the last one.
        self.motion_blur = self.MOTION_BLUR_DEFAULT
        self.motion_blur_max = 0.06   # longest smear, as a fraction of the frame
        self.motion_car = None        # player car pose (x, y, z, heading, hl, hw, h) or None
        self._cur_car = self._prev_car = None
        self.speed_fov = True         # chase-camera speed FOV effect (settings switch)
        self._cur_vp = None           # view-projection the current frame was drawn with
        self._cur_proj_z = (0.0, 1.0)
        self._prev_vp = None          # ... and the previous one, for the reprojection
        self._vp_time = None
        self._vp_eye = None
        # Footprint-filtered ground textures and analytic bark/material detail.
        # Lower values select coarser ground mips and fade small joints earlier.
        self.surface_detail = 1.0
        self._last_vp = None
        self._cloud = 0.0
        self.ae_ok = False
        self._ae_cur = 0
        self._ae_reset = True
        self._ae_t = None
        self._hdr_out = 0.0

        # Screen space reflections (toggle at runtime with Renderer.ssr_enabled)
        # Detail bounds in metres. Mesh cells and individual instances are culled
        # within large chunks; vehicle LOD additionally follows projected size.
        # Everything past these distances is either fogged out or drawn by the
        # simplified far meshes, so the cut is not visible from the street.
        # Two-triangle impostors make the far tree band cheap, so the tree draw
        # distance extends past the geometric LOD range.
        self.lod = {"trees": 400.0, "props": 200.0, "vehicles": 480.0,
                    "edges": 320.0, "neon": 340.0}
        # Building LOD transition distance: close cells keep rooftop equipment,
        # everything further uses the simplified shell drawn for the skyline.
        self.building_lod_distance = 100.0
        self.last_tree_lod_counts = [0, 0, 0, 0]
        self.last_tree_tris = self.last_tree_shadow_tris = 0
        self.last_tree_shadow_lod_counts = [0, 0, 0, 0]
        self._tree_pixel_scale = float(height)
        self._dist_cache = {}
        self._shadow_dir = None      # held light direction (see _held_light_dir)
        self._shadow_key = None      # what the shadow map currently contains
        self.last_draw_calls = 0
        # Layered automotive paint (shaders/vehicle.py, MATERIAL_LIGHTING.md):
        # clear-coat glossiness, orange-peel amplitude, metallic-flake density and
        # pearl/thin-film strength.  Sent per frame as uPaintCoat.
        self.paint_gloss = 0.965
        self.paint_orange_peel = 1.0
        self.paint_flakes = 1.0
        self.paint_pearl = 0.55
        self.ssr_enabled = True
        self.ssr_strength = 1.0     # overall reflection gain
        self.ssr_max_dist = 700.0   # max ray length in metres
        self.ao_enabled = True
        self.ao_radius = 3.0        # metres: contact under parapets, vehicles and in corners
        self.ao_strength = 1.0      # occludes ambient light only
        # GTA IV-style dark contact shadow under cars (shaders/contact.py).
        self.wheel_ao = 1.0         # 0..1 recess occlusion on the brake disc / hub (shaders/vehicle.py)
        self.contact_shadows = True
        self.contact_shadow_strength = 0.90   # 0..1, how black it gets under the chassis
        self.contact_shadow_margin = 0.55      # metres of soft falloff past the body
        self.contact_shadow_clip = 1.0         # 0 ignore the shadow map .. 1 fully clipped by it
        self.contact_shadow_distance = 140.0   # metres; fades out over the last 40%
        self._mrt_on = False        # scene FBO (2 colour targets) is bound
        self._mrt_n = 0
        # MSAA samples on the HDR scene target; 0 = off.  Off by default: the
        # interactive app opts in (F8 toggles it), while offline renders and the
        # pixel-exact regression suites stay deterministic.
        self.aa_samples = 0
        self.ms_fbo, self.ms_rbs = 0, []

        self.resize(width, height)

    def _program(self, vs_src, fs_src):
        from OpenGL.GL import shaders
        gl = self.gl
        stage, source = 'vertex', vs_src
        try:
            vs = shaders.compileShader(vs_src, gl.GL_VERTEX_SHADER)
            stage, source = 'fragment', fs_src
            fs = shaders.compileShader(fs_src, gl.GL_FRAGMENT_SHADER)
            stage = 'link'
            return shaders.compileProgram(vs, fs, validate=False)
        except Exception as exc:
            self._report_shader_failure(stage, source, exc)
            raise

    def _report_shader_failure(self, stage, source, exc):
        """Dump the GPU, the driver's full message and the offending source lines.

        PyOpenGL truncates the driver log in the traceback, and the same shader can
        compile on one GPU and be rejected by another (hybrid laptops switch GPU when
        a monitor is attached), so the log has to be captured at the point of failure.
        """
        import re, sys, traceback
        gl = self.gl

        def query(name):
            try:
                value = gl.glGetString(name)
                return value.decode(errors='replace') if value else '?'
            except Exception:
                return '?'

        log = exc.args[0] if exc.args else str(exc)
        if isinstance(log, bytes):
            log = log.decode(errors='replace')
        log = str(log).replace('\\r\\n', '\n').replace('\\n', '\n')
        lines = source.splitlines()
        out = [f"GPU: {query(gl.GL_VENDOR)} | {query(gl.GL_RENDERER)}",
               f"GL {query(gl.GL_VERSION)} | GLSL {query(gl.GL_SHADING_LANGUAGE_VERSION)}",
               f"{stage} shader failed to build", "", log, ""]
        seen = []
        for match in re.finditer(r"(?<![\w.])0[(:](\d+)", log, re.M):
            n = int(match.group(1))
            if n not in seen and 0 < n <= len(lines):
                seen.append(n)
        for n in seen[:6]:
            out.append(f"--- source around line {n} ---")
            for i in range(max(1, n - 3), min(len(lines), n + 2) + 1):
                out.append(f"{'>>' if i == n else '  '} {i:5d}  {lines[i - 1]}")
            out.append("")
        text = "\n".join(out)
        try:
            with open('shader_error.log', 'w', encoding='utf-8') as fh:
                fh.write(text)
        except OSError:
            pass
        print(text, file=sys.stderr)
        print("(also written to shader_error.log in the working directory)", file=sys.stderr)

    def _make_leaf_textures(self):
        gl = self.gl
        levels = build_leaf_mipmaps(*build_leaf_textures())
        self.vegetation_texture_bytes = sum(data.nbytes for texture in levels for data in texture)
        textures = list(gl.glGenTextures(2))
        try:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            for tex, mips in zip(textures, levels):
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                for level, data in enumerate(mips):
                    gl.glTexImage2D(gl.GL_TEXTURE_2D, level, gl.GL_RGBA8,
                                    data.shape[1], data.shape[0], 0,
                                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, len(mips) - 1)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        except BaseException:
            gl.glDeleteTextures(textures)
            raise
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return textures

    @staticmethod
    def _tree_bake_matrix(view):
        """Orthographic view of template space from one of the four atlas directions."""
        angle = view * math.pi * .5
        forward = np.array([math.cos(angle), math.sin(angle), 0.])
        center = np.array([0., 0., TREE_HEIGHT * .5])
        eye = center + forward * 32.
        f = center - eye
        f /= np.linalg.norm(f)
        s = np.cross(f, [0., 0., 1.])
        s /= np.linalg.norm(s)
        u = np.cross(s, f)
        view_m = np.eye(4)
        view_m[0, :3], view_m[0, 3] = s, -s @ eye
        view_m[1, :3], view_m[1, 3] = u, -u @ eye
        view_m[2, :3], view_m[2, 3] = -f, f @ eye
        half = TREE_HEIGHT * .5
        near, far = 8., 56.
        ortho = np.array([[1. / half, 0., 0., 0.],
                          [0., 1. / half, 0., 0.],
                          [0., 0., -2. / (far - near), -(far + near) / (far - near)],
                          [0., 0., 0., 1.]])
        return (ortho @ view_m).astype(np.float32)

    def _make_tree_impostor(self):
        """Bake the four-view tree atlas on the GPU, then build coverage-preserving mips.

        The geometric template is far too heavy for a Python rasteriser, and the
        atlas is only rebuilt once per renderer, so the production tree is drawn
        into four 256 px orthographic cells.  Lighting stays dynamic: the bake
        carries albedo, alpha and crown occlusion only.
        """
        gl = self.gl
        side, tile = TREE_IMPOSTOR_ATLAS, TREE_IMPOSTOR_TILE
        program = self._program(TREE_IMPOSTOR_VS, TREE_IMPOSTOR_FS)
        color, depth = gl.glGenTextures(1), gl.glGenRenderbuffers(1)
        fbo, vao = gl.glGenFramebuffers(1), gl.glGenVertexArrays(1)
        previous_fbo = int(gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING))
        previous_viewport = [int(v) for v in gl.glGetIntegerv(gl.GL_VIEWPORT)]
        previous_clear = [float(v) for v in gl.glGetFloatv(gl.GL_COLOR_CLEAR_VALUE)]
        previous_pack = int(gl.glGetIntegerv(gl.GL_PACK_ALIGNMENT))
        previous_scissor = bool(gl.glIsEnabled(gl.GL_SCISSOR_TEST))
        previous_box = [int(v) for v in gl.glGetIntegerv(gl.GL_SCISSOR_BOX)]
        previous_cull = bool(gl.glIsEnabled(gl.GL_CULL_FACE))
        previous_blend = bool(gl.glIsEnabled(gl.GL_BLEND))
        try:
            gl.glBindTexture(gl.GL_TEXTURE_2D, color)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, side, side, 0,
                            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
            for param, value in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR),
                                 (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                                 (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                                 (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, param, value)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth)
            gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, side, side)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_2D, color, 0)
            gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                         gl.GL_RENDERBUFFER, depth)
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f'incomplete tree impostor framebuffer (0x{status:x})')

            gl.glBindVertexArray(vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tree_template_vbos[1])
            stride = self.tree_templates[1][0].shape[1] * 4
            for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36),
                                   (7, 2, 40), (8, 1, 48)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride,
                                         ctypes.c_void_p(off))
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.tree_template_ebos[1])

            gl.glUseProgram(program)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_BLEND)
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDepthFunc(gl.GL_LESS)
            gl.glClearColor(0., 0., 0., 0.)
            gl.glEnable(gl.GL_SCISSOR_TEST)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.leaf_tex)
            self._int(program, 'uLeafTex', 0)
            for view in range(TREE_IMPOSTOR_VIEWS):
                x, y = (view % 2) * tile, (view // 2) * tile
                gl.glViewport(x, y, tile, tile)
                gl.glScissor(x, y, tile, tile)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                self._mat(program, 'uBakeMatrix', self._tree_bake_matrix(view))
                gl.glDrawElements(gl.GL_TRIANGLES, self.tree_template_counts[1],
                                  gl.GL_UNSIGNED_INT, None)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            raw = gl.glReadPixels(0, 0, side, side, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
            atlas = np.frombuffer(raw, np.uint8).reshape(side, side, 4).copy()
            levels = build_tree_impostor_mipmaps(atlas)
            gl.glBindTexture(gl.GL_TEXTURE_2D, color)
            for level, data in enumerate(levels):
                gl.glTexImage2D(gl.GL_TEXTURE_2D, level, gl.GL_RGBA8,
                                data.shape[1], data.shape[0], 0,
                                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                               gl.GL_LINEAR_MIPMAP_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, len(levels) - 1)
            self.tree_impostor_bytes = sum(data.nbytes for data in levels)
            return color
        except BaseException:
            gl.glDeleteTextures([color])
            raise
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, previous_fbo)
            gl.glViewport(*previous_viewport)
            gl.glClearColor(*previous_clear)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, previous_pack)
            gl.glScissor(*previous_box)
            if previous_scissor:
                gl.glEnable(gl.GL_SCISSOR_TEST)
            else:
                gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glUseProgram(0)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDeleteFramebuffers(1, [fbo])
            gl.glDeleteRenderbuffers(1, [depth])
            gl.glDeleteVertexArrays(1, [vao])
            gl.glDeleteProgram(program)
            if previous_cull:
                gl.glEnable(gl.GL_CULL_FACE)
            else:
                gl.glDisable(gl.GL_CULL_FACE)
            if previous_blend:
                gl.glEnable(gl.GL_BLEND)
            else:
                gl.glDisable(gl.GL_BLEND)

    def _bind_leaf_texture(self, p):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + 4)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.leaf_tex)
        self._int(p, "uLeafTex", 4)
        normal_loc = self._loc(p, 'uLeafNormalTex')
        if normal_loc >= 0:
            gl.glActiveTexture(gl.GL_TEXTURE0 + 3)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.leaf_normal_tex)
            gl.glUniform1i(normal_loc, 3)
        impostor_loc = self._loc(p, 'uTreeImpostorTex')
        if impostor_loc >= 0:
            gl.glActiveTexture(gl.GL_TEXTURE0 + 5)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tree_impostor_tex)
            gl.glUniform1i(impostor_loc, 5)
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def release_vegetation(self):
        """Release shared trees, leaf atlases and flat ground textures, idempotently."""
        gl = self.gl
        for vao in self.tree_batch_vaos:
            gl.glDeleteVertexArrays(1, [vao])
        for buf in self.tree_template_vbos + self.tree_template_ebos + self.tree_batch_vbos:
            gl.glDeleteBuffers(1, [buf])
        textures = [tex for tex in (self.leaf_tex, self.leaf_normal_tex,
                                    self.tree_impostor_tex) if tex]
        if textures:
            gl.glDeleteTextures(textures)
        self.tree_batch_vaos, self.tree_batch_vbos = [], []
        self.tree_template_vbos, self.tree_template_ebos = [], []
        self.tree_mesh_vbo = self.tree_mesh_ebo = self.tree_batch_vao = self.tree_batch_vbo = 0
        self.leaf_tex = self.leaf_normal_tex = self.tree_impostor_tex = 0
        self.vegetation_texture_bytes = self.tree_impostor_bytes = 0
        self.release_ground_texture()

    def _make_ground_texture(self):
        """One shared, mipmapped array; all CPU bake data dies after this upload."""
        from OpenGL.extensions import hasGLExtension
        gl = self.gl
        levels = build_ground_mipmaps(build_ground_textures())
        previous = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D_ARRAY))
        alignment = int(gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT))
        tex = gl.glGenTextures(1)
        self.ground_texture_bytes = 0
        self.ground_texture_anisotropy = 1.
        try:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tex)
            for level, data in enumerate(levels):
                gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, level, gl.GL_RGBA8,
                                data.shape[2], data.shape[1], data.shape[0], 0,
                                gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
            for param, value in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
                                 (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                                 (gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT),
                                 (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT),
                                 (gl.GL_TEXTURE_MAX_LEVEL, len(levels) - 1)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, param, value)
            if hasGLExtension('GL_EXT_texture_filter_anisotropic'):
                from OpenGL.GL.EXT.texture_filter_anisotropic import (
                    GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, GL_TEXTURE_MAX_ANISOTROPY_EXT)
                self.ground_texture_anisotropy = min(4., float(gl.glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)))
                gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                   self.ground_texture_anisotropy)
            self.ground_texture_bytes = sum(data.nbytes for data in levels)
        except BaseException:
            gl.glDeleteTextures([tex])
            raise
        finally:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, previous)
        return tex

    def _bind_ground_texture(self, p):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.GROUND_TEXTURE_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.ground_tex)
        self._int(p, "uGroundTex", self.GROUND_TEXTURE_UNIT)
        self._float(p, 'uGroundTextureOn', float(self.ground_texture_enabled and bool(self.ground_tex)))
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def release_ground_texture(self):
        if self.ground_tex:
            self.gl.glDeleteTextures([self.ground_tex])
        self.ground_tex = self.ground_texture_bytes = 0

    def _make_billboard_texture(self):
        """One shared 16:9 texture array of the drop-in Times Square creatives."""
        from OpenGL.extensions import hasGLExtension
        gl = self.gl
        frames, paths = load_billboard_images()
        self.billboard_layers = [path.name for path in paths]
        self.billboard_anisotropy = 1.
        if frames is None:
            return 0
        count, height, width, _ = frames.shape
        previous = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D_ARRAY))
        alignment = int(gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT))
        tex = gl.glGenTextures(1)
        try:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tex)
            # Image rows are top-down; OpenGL's texture origin is bottom-left.
            gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, 0, gl.GL_RGBA8,
                            width, height, count, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE,
                            np.ascontiguousarray(frames[:, ::-1]))
            for param, value in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
                                 (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                                 (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                                 (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, param, value)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D_ARRAY)
            # Screens stand at a steep angle to the street, exactly like the ground: filter
            # along the compressed axis or a panel's creative dissolves into a lattice.
            if hasGLExtension('GL_EXT_texture_filter_anisotropic'):
                from OpenGL.GL.EXT.texture_filter_anisotropic import (
                    GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, GL_TEXTURE_MAX_ANISOTROPY_EXT)
                self.billboard_anisotropy = min(8., float(gl.glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)))
                gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                   self.billboard_anisotropy)
            self.billboard_texture_bytes = int(frames.nbytes * 4 / 3)
            self.billboard_count = int(count)
            return int(tex)
        except BaseException:
            gl.glDeleteTextures([tex])
            raise
        finally:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, previous)

    def _bind_billboard_texture(self, p):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.BILLBOARD_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.billboard_tex)
        self._int(p, "uBillboards", self.BILLBOARD_UNIT)
        self._int(p, "uBillboardCount", self.billboard_count)
        self._float(p, "uBillboardAspect", BILLBOARD_ASPECT)
        self._float(p, "uBillboardOn", float(bool(self.billboard_tex and self.billboard_count)))
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def release_billboards(self):
        if self.billboard_tex:
            self.gl.glDeleteTextures([self.billboard_tex])
        self.billboard_tex = self.billboard_count = self.billboard_texture_bytes = 0
        self.billboard_anisotropy = 1.
        self.billboard_layers = []

    def _make_water_texture(self):
        """Four shared spectral fields; retain no CPU bake arrays or per-tile maps."""
        from OpenGL.extensions import hasGLExtension
        gl = self.gl
        levels = build_water_ripple_mipmaps(build_water_ripples())
        previous = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D_ARRAY))
        alignment = int(gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT))
        tex = gl.glGenTextures(1)
        try:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, tex)
            for level, data in enumerate(levels):
                gl.glTexImage3D(gl.GL_TEXTURE_2D_ARRAY, level, gl.GL_RGBA16F,
                                data.shape[2], data.shape[1], data.shape[0], 0,
                                gl.GL_RGBA, gl.GL_HALF_FLOAT, data)
            for param, value in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR),
                                 (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                                 (gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT),
                                 (gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT),
                                 (gl.GL_TEXTURE_MAX_LEVEL, len(levels) - 1)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, param, value)
            if hasGLExtension('GL_EXT_texture_filter_anisotropic'):
                from OpenGL.GL.EXT.texture_filter_anisotropic import (
                    GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, GL_TEXTURE_MAX_ANISOTROPY_EXT)
                self.water_texture_anisotropy = min(8., float(gl.glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)))
                gl.glTexParameterf(gl.GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                   self.water_texture_anisotropy)
            self.water_texture_bytes = sum(data.nbytes for data in levels)
        except BaseException:
            gl.glDeleteTextures([tex])
            raise
        finally:
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, alignment)
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, previous)
        return tex

    def _bind_water_texture(self, p, has_water):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.WATER_RIPPLE_UNIT)
        if has_water and self.water_ripples_enabled and not self.water_ripple_tex and self.water_texture_error is None:
            try:
                self.water_ripple_tex = self._make_water_texture()
            except Exception as exc:
                self.water_texture_error = str(exc)  # analytic detail remains usable
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.water_ripple_tex)
        self._int(p, "uWaterRippleTex", self.WATER_RIPPLE_UNIT)
        self._float(p, 'uWaterRippleOn', float(self.water_ripples_enabled and bool(self.water_ripple_tex)))
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def release_water_texture(self):
        """Idempotent explicit release; resize/toggles do not discard the shared map."""
        if self.water_ripple_tex:
            self.gl.glDeleteTextures([self.water_ripple_tex])
        self.water_ripple_tex = self.water_texture_bytes = 0
        self.water_texture_error = None

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

    def _city_vao_steps(self, mesh, owned, vertex_buffer=None, indices=None, terrain=False, water=False):
        """Upload at most 1 MiB per yield; cancellation owns every allocation."""
        gl = self.gl
        stride = WATER_VERTEX.itemsize if water else 52 if terrain else 48
        vao = gl.glGenVertexArrays(1); owned[0].append(vao)
        gl.glBindVertexArray(vao)
        bufs = []
        vbo = vertex_buffer
        if vbo is None:
            vbo = gl.glGenBuffers(1); bufs.append(vbo); owned[1].append(vbo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, len(mesh.vertices)*stride, None, gl.GL_STATIC_DRAW)
            yield
            for first in range(0, len(mesh.vertices), 16384):
                data = (mesh.gpu_vertices[first:first+16384] if mesh.gpu_vertices is not None
                        else pack_vertices(mesh.vertices[first:first+16384], terrain=terrain, water=water))
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, first*stride, data.nbytes, data)
                yield
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        attrs = ((0,3,gl.GL_FLOAT,False,0), (1,3,gl.GL_BYTE,True,12),
                (2,3,gl.GL_UNSIGNED_BYTE,True,16), (3,1,gl.GL_UNSIGNED_BYTE,True,19),
                (4,3,gl.GL_FLOAT,False,20), (5,2,gl.GL_FLOAT,False,32),
                (6,1,gl.GL_FLOAT,False,40), (7,1,gl.GL_FLOAT,False,44))
        if terrain:
            attrs = ((0,3,gl.GL_FLOAT,False,0), (1,3,gl.GL_BYTE,True,12),
                     (2,3,gl.GL_FLOAT,False,16), (3,1,gl.GL_UNSIGNED_BYTE,True,28),
                      (4,3,gl.GL_FLOAT,False,32), (5,2,gl.GL_FLOAT,False,44))
        elif water:
            attrs = ((0,3,gl.GL_FLOAT,False,0), (4,3,gl.GL_FLOAT,False,12))
        for loc, size, typ, norm, off in attrs:
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc,size,typ,norm,stride,ctypes.c_void_p(off))
        idx = mesh.indices if indices is None else indices
        ebo = gl.glGenBuffers(1); bufs.append(ebo); owned[1].append(ebo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, None, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)
        yield
        for first in range(0,len(idx),262144):
            # ARRAY_BUFFER binding uploads EBO storage without changing another VAO.
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER,ebo)
            block = idx[first:first+262144]
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER,first*4,block.nbytes,block)
            yield
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER,0)
        return vao, bufs

    def _tree_vao(self, instances, level=0):
        gl = self.gl
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.tree_template_vbos[level])
        stride = self.tree_templates[level][0].shape[1] * 4
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36),
                               (7, 2, 40), (8, 1, 48)):
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
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.tree_template_ebos[level])
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        return vao, inst_vbo

    def _streetpole_vao(self, inst):
        """Instanced pole + arm + luminaire.  The glow sprite lives in LightField."""
        gl = self.gl
        inst = np.ascontiguousarray(inst, np.float32)
        inst_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, inst_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, inst.nbytes, inst, gl.GL_STATIC_DRAW)

        vao_p = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao_p)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.pole_mesh_vbo)
        for loc, size, off in ((0, 3, 0), (1, 3, 12), (2, 1, 24)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, 28, ctypes.c_void_p(off))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, inst_vbo)
        for loc, off in ((3, 0), (4, 16)):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, gl.GL_FALSE, 36, ctypes.c_void_p(off))
            gl.glVertexAttribDivisor(loc, 1)
        gl.glEnableVertexAttribArray(5)
        gl.glVertexAttribPointer(5, 1, gl.GL_FLOAT, gl.GL_FALSE, 36, ctypes.c_void_p(32))
        gl.glVertexAttribDivisor(5, 1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.pole_mesh_ebo)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        return vao_p, [inst_vbo]

    def _prepare_lamps(self, lights, origin, scale):
        """Prepare the unchanged instance stream; grounded sources retain arm choice."""
        lights = np.asarray(lights, np.float32)
        n = len(lights)
        lx, ly, lz = lights[:, 0], lights[:, 1], lights[:, 2]
        base = lights[:, 4] if lights.shape[1] >= 5 else np.zeros(n, np.float32)
        has_arm = lights.shape[1] >= 4
        ang = lights[:, 3] if has_arm else np.zeros(n, np.float32)
        source_height = lights[:, 5] if lights.shape[1] >= 6 else lz-base
        arm = (np.where(source_height >= 8.5, _LAMP_ARM_LEN_TALL, _LAMP_ARM_LEN_SHORT).astype(np.float32)
               if has_arm else np.zeros(n, np.float32))
        sx, sy = float(scale[0]), float(scale[1])
        wx = lx * sx + float(origin[0]); wy = ly * sy + float(origin[1])
        col = lamp_colors(wx, wy)
        inst = np.empty((n, 9), np.float32)
        inst[:, 0], inst[:, 1], inst[:, 2], inst[:, 3] = lx, ly, lz, ang
        inst[:, 4] = arm
        inst[:, 5:8] = col
        inst[:, 8] = base
        world = np.empty((n, 6), np.float32)
        world[:, 0] = (lx + np.cos(ang) * arm) * sx + float(origin[0])
        world[:, 1] = (ly + np.sin(ang) * arm) * sy + float(origin[1])
        world[:, 2] = lz - _LAMP_LENS_DROP
        world[:, 3:6] = col
        return inst, world

    def _select_lamps(self, chunks, cam, night):
        """Rebuild the world-space lamp grid whenever the chunk set changes.

        Lamps are static, so the grid (and its textures) stay valid across
        frames; only the night factor gates the lookup in the shader.
        """
        if night <= .02: return
        lamp_chunks = [c for c in chunks if c.lights_cpu is not None and len(c.lights_cpu)]
        key = tuple(id(c) for c in lamp_chunks)
        self.spill.request(key, (c.lights_cpu for c in lamp_chunks))
        self.spill.update()

    def _lamp_uniforms(self, p):
        self.spill.bind(p)

    # Vertex layout shared by every car template (10 floats: pos3 normal3 colour3 material-id) and
    # 24-float instances and a separate pivot/role buffer preserve the geometry format.
    # One VAO per (template, instance buffer) pair; no draw call per wheel/driver.
    _VEH_ATTRS = ((0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 1, 36))
    _VEH_INST_ATTRS = INSTANCE_ATTRIBUTES

    def _make_vehicle_vao(self, kind, inst_vbo):
        gl = self.gl
        stride = self.vehicle_templates[kind][0].shape[1] * 4
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vehicle_template_vbos[kind])
        for loc, size, off in self._VEH_ATTRS:
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(off))
        animation_vbos = getattr(self, 'vehicle_animation_vbos', ())
        if animation_vbos:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, animation_vbos[kind])
            gl.glEnableVertexAttribArray(10)
            gl.glVertexAttribPointer(10, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        else:
            # Legacy inspection/test geometry without animation remains static.
            gl.glDisableVertexAttribArray(10)
            gl.glVertexAttrib4f(10, 0, 0, 0, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.vehicle_template_ebos[kind])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, inst_vbo)
        for loc, size, off in self._VEH_INST_ATTRS:
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, VEHICLE_BYTES, ctypes.c_void_p(off))
            gl.glVertexAttribDivisor(loc, 1)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        return vao

    def _build_instance_vao(self, kind):
        return self._make_vehicle_vao(kind, self.moving_inst_vbos[kind])

    def _build_chunk_vehicle_vao(self, kind, inst_vbo):
        return self._make_vehicle_vao(kind, inst_vbo)

    def _tshader(self):
        """Seconds since start.  Raw epoch time is ~1.7e9, which float32 can only
        represent in 128 s steps, so every animated shader froze between jumps."""
        return float((time.time() - self._t0) % 86400.0)

    def _loc(self, prog, name):
        key = (prog, name)
        if key not in self._uloc:
            self._uloc[key] = self.gl.glGetUniformLocation(prog, name)
        return self._uloc[key]

    def _mat(self, prog, name, m):
        self.gl.glUniformMatrix4fv(self._loc(prog, name), 1, self.gl.GL_TRUE,
                                   np.ascontiguousarray(m, np.float32))

    def _mat3(self, prog, name, m):
        self.gl.glUniformMatrix3fv(self._loc(prog, name), 1, self.gl.GL_TRUE,
                                   np.ascontiguousarray(m, np.float32))

    def _vec3(self, prog, name, v):
        key = (prog, name)
        value = tuple(float(x) for x in v)
        if self._uniform_values.get(key) == value: return
        self._uniform_values[key] = value
        loc = self._loc(prog, name)
        if loc >= 0: self.gl.glUniform3f(loc, *value)

    def _float(self, prog, name, v):
        key = (prog, name)
        value = float(v)
        if self._uniform_values.get(key) == value: return
        self._uniform_values[key] = value
        loc = self._loc(prog, name)
        if loc >= 0: self.gl.glUniform1f(loc, value)

    # Sampler units, resolutions and texel steps are re-sent by every pass even though
    # they only change on a resize or a new program.  Each PyOpenGL call costs tens of
    # microseconds of dispatch, and at street level the frame is submit-bound, so these
    # go through the same value cache as _float/_vec3 and cost a dict lookup once warm.
    def _int(self, prog, name, v):
        key = ("i", prog, name)
        value = int(v)
        if self._uniform_values.get(key) == value: return
        self._uniform_values[key] = value
        loc = self._loc(prog, name)
        if loc >= 0: self.gl.glUniform1i(loc, value)

    def _vec2(self, prog, name, v):
        key = ("v2", prog, name)
        value = (float(v[0]), float(v[1]))
        if self._uniform_values.get(key) == value: return
        self._uniform_values[key] = value
        loc = self._loc(prog, name)
        if loc >= 0: self.gl.glUniform2f(loc, value[0], value[1])

    def _stream_instances(self, vbo, rows):
        gl = self.gl
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        # Orphan storage so a previous frame never stalls a CPU upload.
        gl.glBufferData(gl.GL_ARRAY_BUFFER, rows.nbytes, None, gl.GL_STREAM_DRAW)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, rows.nbytes, rows)

    def _vehicle_rows(self, chunks, traffic):
        key = tuple(id(c) for c in chunks)
        if key != self._parked_key:
            arrays, kinds = [], []
            for c in chunks:
                for kind, rows in enumerate(c.parked_cpu or ()):
                    if rows is not None and len(rows):
                        arrays.append(normalize_instances(rows))
                        kinds.append(np.full(len(rows), kind, np.int32))
            self._parked_rows = np.concatenate(arrays) if arrays else np.empty((0, VEHICLE_STRIDE), np.float32)
            self._parked_kinds = np.concatenate(kinds) if kinds else np.empty(0, np.int32)
            self._parked_key = key
        arrays, kinds = [self._parked_rows], [self._parked_kinds]
        for kind, rows in enumerate(traffic or ()):
            if rows is not None and len(rows):
                arrays.append(normalize_instances(rows))
                kinds.append(np.full(len(rows), kind, np.int32))
        return np.concatenate(arrays), np.concatenate(kinds)

    def _vehicle_instance_offset(self, vao, vbo, instances):
        """GL 3.3 base-instance fallback, changed only when the caster prefix moves."""
        offset = int(instances) * VEHICLE_BYTES
        if self._vehicle_instance_offsets.get(vao, 0) == offset: return
        gl = self.gl
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        for loc, count, off in INSTANCE_ATTRIBUTES:
            gl.glVertexAttribPointer(loc, count, gl.GL_FLOAT, gl.GL_FALSE, VEHICLE_BYTES,
                                     ctypes.c_void_p(offset + off))
        self._vehicle_instance_offsets[vao] = offset

    def _prepare_vehicles(self, chunks, traffic, proj, planes, vehicle_light):
        """One upload per kind/LOD, shared by colour and light-space caster draws.

        Rows are [offscreen casters, visible casters, other visible cars]. Shadow
        VAOs read the caster prefix; scene VAOs read the visible suffix. Offscreen
        cars can cast onto visible roads without entering the scene draw count.
        """
        arr, kinds = self._vehicle_rows(chunks, traffic)
        alive = arr[:, 23] > 0.
        visible = sphere_mask(arr[:, 12:15], 8., self._eye, self.lod['vehicles'], planes) & alive
        self._prepare_contact_shadows(arr, kinds, visible)
        casters = np.zeros(len(arr), bool)
        if vehicle_light is not None:
            casters = sphere_mask(arr[:, 12:15], 8., self._eye, math.inf,
                                  frustum_planes(vehicle_light.matrix.astype(np.float64))) & alive
        self.last_vehicle_shadow_casters = int(casters.sum())
        self.moving_counts = [0] * len(self.vehicle_templates)
        self.vehicle_shadow_counts = [0] * len(self.vehicle_templates)
        self.last_vehicle_upload_bytes = self.last_vehicle_uploads = 0
        pixel_scale = self.height * abs(float(proj[1, 1])) * .5
        distance = np.linalg.norm(arr[:, 12:15] - self._eye, axis=1)
        pixels = pixel_scale * 4.5 / np.maximum(distance, 1.)
        # Optimized: more aggressive LOD transitions (was 100, 26)
        levels = np.where(pixels > 80., 0, np.where(pixels > 20., 1, 2))
        batches = kinds + levels*self.vehicle_kinds
        offscreen = casters & ~visible
        # Invisible casters only need the far scene kind's shared instance VBO.
        batches[offscreen] = kinds[offscreen] + 2*self.vehicle_kinds
        for k in np.unique(batches[visible | offscreen]):
            off = np.flatnonzero(offscreen & (batches == k))
            both = np.flatnonzero(visible & casters & (batches == k))
            view = np.flatnonzero(visible & ~casters & (batches == k))
            rows = np.ascontiguousarray(arr[np.r_[off, both, view]])
            previous = self._vehicle_uploads.get(int(k))
            if previous is None or previous.shape != rows.shape or not np.array_equal(previous, rows):
                self._stream_instances(self.moving_inst_vbos[k], rows)
                rows.flags.writeable = False
                self._vehicle_uploads[int(k)] = rows
                self.last_vehicle_upload_bytes += rows.nbytes
                self.last_vehicle_uploads += 1
            self._vehicle_instance_offset(self.moving_vaos[k], self.moving_inst_vbos[k], len(off))
            self.moving_counts[k] = len(both) + len(view)
            self.vehicle_shadow_counts[k] = len(off) + len(both)
        if not casters.any(): return None
        # Exact pose/animation/visibility identity, excluding paint, lamps and
        # safe prediction speed. Paused cars reuse their map; flashing does not
        # dirty it. This is independent of camera visibility/scene LOD ordering.
        pose = np.ascontiguousarray(arr[casters, :16]).tobytes()
        motion = np.ascontiguousarray(arr[casters][:, [20, 21, 23]]).tobytes()
        return (vehicle_light.matrix.tobytes(), kinds[casters].tobytes(), pose, motion)

    def _prepare_contact_shadows(self, arr, kinds, visible):
        """Pack one oriented footprint per nearby visible car (parked and moving)."""
        self._contact_count = 0
        if not self.contact_shadows or not len(arr):
            return
        idx = np.flatnonzero(visible)
        if not len(idx):
            return
        dist = np.linalg.norm(arr[idx, 12:15] - self._eye, axis=1)
        near = dist < self.contact_shadow_distance
        idx, dist = idx[near], dist[near]
        if not len(idx):
            return
        if len(idx) > self.CONTACT_MAX:
            idx = idx[np.argsort(dist)[:self.CONTACT_MAX]]
        a = arr[idx]
        k = np.minimum(kinds[idx], len(_CAR_LENGTHS) - 1)

        def unit(v):
            return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        rows = np.empty((len(idx), self.CONTACT_FLOATS), np.float32)
        rows[:, 0:3] = a[:, 12:15]                 # model translation: origin on the road
        rows[:, 3:6] = unit(a[:, 0:3])             # model +x: forward
        rows[:, 6:9] = unit(a[:, 4:7])             # model +y: left
        rows[:, 9] = _CAR_LENGTHS[k] * .5
        rows[:, 10] = _CAR_WIDTHS[k] * .5
        rows[:, 11] = _CAR_WHEELBASES[k]
        rows[:, 12] = np.clip(a[:, 23], 0., 1.)    # spawn/recycle fade
        self._contact_rows = rows
        self._contact_count = len(rows)

    def _draw_contact_shadows(self, common, light, sky):
        """Dark soft footprint under every car: multiplies the road, clipped by the sun shadow."""
        gl = self.gl
        n = self._contact_count
        if not self.contact_shadows or n == 0 or self.contact_shadow_strength <= 0.:
            return
        if self._contact_vao is None:
            self._contact_vao = gl.glGenVertexArrays(1)
            self._contact_corner_vbo = gl.glGenBuffers(1)
            self._contact_inst_vbo = gl.glGenBuffers(1)
            gl.glBindVertexArray(self._contact_vao)
            corners = np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._contact_corner_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, corners.nbytes, corners, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(0)
            gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, None)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._contact_inst_vbo)
            stride = self.CONTACT_FLOATS * 4
            for loc, size, off in ((1, 3, 0), (2, 3, 12), (3, 3, 24), (4, 4, 36)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(off))
                gl.glVertexAttribDivisor(loc, 1)
        rows = np.ascontiguousarray(self._contact_rows[:n])
        self._stream_instances(self._contact_inst_vbo, rows)
        p = self.prog_contact
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        self._float(p, "uMargin", max(0.05, self.contact_shadow_margin))
        self._float(p, "uContactStrength", float(np.clip(self.contact_shadow_strength, 0., 1.)))
        self._float(p, "uContactClip", float(np.clip(self.contact_shadow_clip, 0., 1.)))
        self._float(p, "uContactFadeNear", self.contact_shadow_distance * .6)
        self._float(p, "uContactFadeFar", self.contact_shadow_distance)
        self._set_mrt(1)                            # colour only: never touch the material buffer
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ZERO, gl.GL_SRC_COLOR)  # dst *= (1 - darkness)
        # Slope-scaled depth offset toward the eye (sign flips with reverse-Z) so the quad
        # survives tilt mismatch with the road instead of being cut into it.
        off = 1. if self._reverse_active else -1.
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(off * 2., off * 8.)
        gl.glBindVertexArray(self._contact_vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLE_STRIP, 0, 4, n)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        self.last_draw_calls += 1
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glBindVertexArray(0)

    def _draw_foliage_poles(self, chunks, planes, p, trees=True, *, tree_light=None):
        key = (trees, tuple(id(c) for c in chunks))
        cached = self._instance_cache.get(key)
        if cached is None:
            batches = [c.tree_cpu if trees else c.pole_cpu for c in chunks]
            batches = [a for a in batches if a is not None and len(a)]
            if not batches: return
            a = np.concatenate(batches)
            pos = a[:, :3].copy()
            if trees:
                radius = a[:, 3] * TREE_RADIUS
                pos[:, 2] += a[:, 3] * TREE_CENTER_Z
            else:
                radius = (a[:, 2] - a[:, 8]) * .5 + 3
                pos[:, 2] = (a[:, 2] + a[:, 8]) * .5
            if len(self._instance_cache) >= 8: self._instance_cache.clear()
            cached = self._instance_cache[key] = (a, pos, radius)
        a, pos, radius = cached
        # Apply fog-aware scaling to tree/prop culling distance
        fog_scale = getattr(self, '_fog_lod_scale', 1.0)
        max_distance = math.inf if tree_light is not None else self.lod['trees' if trees else 'props'] * fog_scale
        mask = sphere_mask(pos, radius, self._eye, max_distance, planes)
        if not mask.any(): return
        rows = np.ascontiguousarray(a[mask])
        self._vec3(p, 'uOffset', (0, 0, 0))
        self._vec2(p, "uScale", (1, 1))
        self._float(p, 'uGrow', 1)
        if trees:
            distance = np.linalg.norm(pos[mask] - self._eye, axis=1)
            levels = tree_lod_levels(rows[:, 3], pixel_scale=self._tree_pixel_scale, distances=distance,
                                     texel=tree_light.texel if tree_light is not None else None)
            # Shadow maps keep geometry: a camera-facing card is not a useful caster.
            if tree_light is not None or not self.tree_impostor_tex:
                levels = np.where(levels > 2, 2, levels)
            counts = [0, 0, 0, 0]
            for level in range(4):
                selected = np.ascontiguousarray(rows[levels == level])
                counts[level] = len(selected)
                if not len(selected): continue
                self._stream_instances(self.tree_batch_vbos[level], selected)
                self.gl.glBindVertexArray(self.tree_batch_vaos[level])
                self.gl.glDrawElementsInstanced(self.gl.GL_TRIANGLES, self.tree_template_counts[level],
                                               self.gl.GL_UNSIGNED_INT, None, len(selected))
                self.last_draw_calls += 1
            tris = sum(count * size // 3 for count, size in zip(counts, self.tree_template_counts))
            if tree_light is None:
                self.last_tree_lod_counts, self.last_tree_tris = counts, tris
            else:
                self.last_tree_shadow_tris += tris
                self.last_tree_shadow_lod_counts = [a + b for a, b in zip(self.last_tree_shadow_lod_counts, counts)]
        else:
            self._stream_instances(self.pole_batch_vbo, rows)
            self.gl.glBindVertexArray(self.pole_batch_vao)
            self.gl.glDrawElementsInstanced(self.gl.GL_TRIANGLES, self.pole_indices_count,
                                           self.gl.GL_UNSIGNED_INT, None, len(rows))
            self.last_draw_calls += 1

    @staticmethod
    def _u32(m):
        return np.ascontiguousarray(m.indices, dtype=np.uint32)

    def upload_chunk(self, *args, **kwargs):
        """Upload a whole chunk at once (blocking).  See upload_chunk_steps."""
        gen = self.upload_chunk_steps(*args, **kwargs)
        while True:
            try: next(gen)
            except StopIteration as stop: return stop.value

    def upload_chunk_steps(self, *args, **kwargs):
        owned = ([], [])
        complete = False
        try:
            chunk = yield from self._upload_chunk_steps(*args, **kwargs, _owned=owned)
            complete = True
            return chunk
        finally:
            if not complete:
                for vao in owned[0]: self.gl.glDeleteVertexArrays(1, [vao])
                for buf in owned[1]: self.gl.glDeleteBuffers(1, [buf])

    def _upload_chunk_steps(self, key, b_mesh, g_mesh, w_mesh, t_mesh, origin,
                           scale=(1.0, 1.0), lights=None, parked=None,
                           s_mesh=None, p_mesh=None, _owned=None):
        """Generator form of the upload: yields after each GPU buffer set so the caller can
        spread a big tile over several frames (a single 50-100 MB upload is a visible hitch).
        The finished ChunkVAO is the generator's return value."""
        gl = self.gl
        vao_b = vao_e = vao_g = vao_w = vao_t = 0
        edges_cpu = None
        vao_s = vao_prop = 0; s_count = p_count = 0
        vao_b_lod = 0; b_lod_count = 0; cell_b_lod = None
        trees_world = poles_world = None
        bufs = []; b_count = e_count = g_count = w_count = t_count = l_count = 0; nbytes = 0
        if b_mesh:
            vao_b, bb = yield from self._city_vao_steps(b_mesh, _owned)
            bufs += bb
            b_count = len(b_mesh.indices); e_count = len(b_mesh.edges)
            # Outlines are off by default, so their buffer (24 B per line vertex, often
            # a sizeable share of the chunk) stays in system RAM until they are first
            # drawn.  That cuts VRAM and the upload hitch when a tile streams in.
            edges_cpu = (b_mesh.packed_edges.copy() if b_mesh.packed_edges is not None else b_mesh.edges) if e_count else None
            nbytes += len(b_mesh.vertices)*48 + b_mesh.indices.nbytes
            if b_mesh.lod_indices is not None:
                vao_b_lod, lb = yield from self._city_vao_steps(b_mesh, _owned, bb[0], b_mesh.lod_indices)
                bufs += lb; b_lod_count = len(b_mesh.lod_indices)
                nbytes += b_mesh.lod_indices.nbytes
                if b_mesh.lod_cells is not None:
                    cell_b_lod = self._world_boxes(b_mesh.lod_cells, origin, scale)
            yield
        if g_mesh:
            vao_g, bg = yield from self._city_vao_steps(g_mesh, _owned, terrain=True)
            bufs += bg; g_count = len(g_mesh.indices)
            nbytes += len(g_mesh.vertices)*52 + g_mesh.indices.nbytes
            yield
        if w_mesh:
            vao_w, bw = yield from self._city_vao_steps(w_mesh, _owned, water=True)
            bufs += bw; w_count = len(w_mesh.indices)
            nbytes += len(w_mesh.vertices)*WATER_VERTEX.itemsize + w_mesh.indices.nbytes
            yield
        if t_mesh is not None and len(t_mesh) > 0:
            trees_world = t_mesh.copy()
            trees_world[:, :2] = trees_world[:, :2] * scale + np.asarray(origin)[:2]
            t_count = len(trees_world)
            yield
        vao_p = 0; lamps_world = None
        if lights is not None and len(lights) > 0:
            inst, lamps_world = self._prepare_lamps(lights, origin, scale)
            poles_world = inst
            poles_world[:, :2] = poles_world[:, :2] * scale + np.asarray(origin)[:2]
            l_count = len(inst)
            yield

        if s_mesh:
            vao_s, bs = yield from self._city_vao_steps(s_mesh, _owned)
            bufs += bs; s_count = len(s_mesh.indices)
            nbytes += len(s_mesh.vertices)*48 + s_mesh.indices.nbytes
            yield
        if p_mesh:
            vao_prop, bp = yield from self._city_vao_steps(p_mesh, _owned)
            bufs += bp; p_count = len(p_mesh.indices)
            nbytes += len(p_mesh.vertices)*48 + p_mesh.indices.nbytes
            yield

        neon_cpu = None
        if s_mesh is not None and len(s_mesh.vertices) > 0:
            samples = getattr(s_mesh, 'neon_samples', None)
            if samples is None:       # blocking/static uploads can use raw geometry
                from .lod import neon_light_samples
                samples = neon_light_samples(s_mesh)
            neon_cpu = samples.copy()
            neon_cpu[:, :2] = neon_cpu[:, :2] * scale + np.asarray(origin)[:2]

        vao_v = [0] * self.vehicle_kinds
        v_counts = [0] * self.vehicle_kinds
        parked_world = [None] * self.vehicle_kinds
        if parked:
            for kind in range(self.vehicle_kinds):
                arr = parked.get(kind)
                if arr is None or len(arr) == 0: continue
                arr = normalize_instances(arr).copy()
                arr[:, 12:14] = arr[:, 12:14] * scale + np.asarray(origin)[:2]
                # Model columns carry the tile's x/y scaling, as in VEHICLE_VS.
                arr[:, [0, 4, 8]] *= scale[0]
                arr[:, [1, 5, 9]] *= scale[1]
                parked_world[kind] = arr
                v_counts[kind] = int(len(arr))
                yield

        now = time.time()
        sphere = np.array([origin[0], origin[1], 10.0, 1500.0], np.float64)
        aabb, cell_b, cell_g = self._chunk_bounds(b_mesh, g_mesh, w_mesh, origin, scale)
        if aabb is not None:
            ctr = (aabb[:3] + aabb[3:]) * 0.5
            sphere = np.array([ctr[0], ctr[1], ctr[2],
                               0.5 * float(np.linalg.norm(aabb[3:] - aabb[:3])) + 15.0], np.float64)
        return ChunkVAO(
            key=key,
            vao_b=vao_b, vao_e=vao_e, vao_g=vao_g, vao_w=vao_w, vao_t=vao_t,
            vao_v=vao_v, bufs=bufs,
            b_count=b_count, e_count=e_count, g_count=g_count, w_count=w_count,
            t_count=t_count, l_count=l_count, v_counts=v_counts,
            offset=np.array([origin[0], origin[1], 0.0], np.float32),
            scale=np.array(scale, np.float32), sphere=sphere, nbytes=int(nbytes),
            born=now, last_used=now,
            lights_cpu=lamps_world, vao_p=vao_p,
            vao_s=vao_s, vao_prop=vao_prop, s_count=s_count, p_count=p_count,
            neon_cpu=neon_cpu,
            aabb=aabb, cell_b=cell_b, cell_g=cell_g, edges_cpu=edges_cpu,
            cell_w=self._world_boxes(w_mesh.cells, origin, scale) if w_mesh is not None and w_mesh.cells is not None else None,
            edges_packed=b_mesh is not None and b_mesh.packed_edges is not None,
            vao_b_lod=vao_b_lod, b_lod_count=b_lod_count, cell_b_lod=cell_b_lod,
            cell_prop=self._world_boxes(p_mesh.cells, origin, scale) if p_mesh is not None and p_mesh.cells is not None else None,
            cell_s=self._world_boxes(s_mesh.cells, origin, scale) if s_mesh is not None and s_mesh.cells is not None else None,
            tree_cpu=trees_world, pole_cpu=poles_world, parked_cpu=parked_world)

    def _ensure_edges(self, c):
        """Create the outline VAO the first time a chunk's edges are drawn."""
        if c.vao_e or c.edges_cpu is None:
            return
        data = (np.frombuffer(zlib.decompress(c.edges_cpu), np.float32).reshape(-1, 6)
                if c.edges_packed else c.edges_cpu)
        c.vao_e, eb = self._vao(data, [(0, 3), (1, 3)])
        c.nbytes += data.nbytes
        c.bufs += eb
        c.edges_cpu = None

    def free_chunk(self, chunk):
        gl = self.gl
        self._instance_cache.clear()
        self._cell_batch_key = self._parked_key = None
        self._cell_visibility.clear()
        self._cell_draw_cache.clear()
        for vao in (chunk.vao_b, chunk.vao_e, chunk.vao_g, chunk.vao_w,
                    chunk.vao_t, chunk.vao_p, chunk.vao_s, chunk.vao_prop, chunk.vao_b_lod):
            if vao: gl.glDeleteVertexArrays(1, [vao])
        for vao in chunk.vao_v:
            if vao: gl.glDeleteVertexArrays(1, [vao])
        for buf in chunk.bufs:
            if buf: gl.glDeleteBuffers(1, [buf])
        chunk.bufs = []
        chunk.edges_cpu = None
        chunk.tree_cpu = chunk.pole_cpu = chunk.parked_cpu = None

    def clear(self):
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(*BG_COLOR, 1.0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    def resize(self, w, h):
        self.width, self.height = max(w, 1), max(h, 1)
        self._prev_vp = None          # new aspect: no valid history to reproject into
        self.gl.glViewport(0, 0, self.width, self.height)
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.depth_tex)
        self.gl.glTexImage2D(self.gl.GL_TEXTURE_2D, 0, self.gl.GL_DEPTH_COMPONENT32F, self.width, self.height, 0, self.gl.GL_DEPTH_COMPONENT, self.gl.GL_FLOAT, None)
        self._create_post_targets()

    def _release_post_targets(self):
        gl = self.gl
        self._release_msaa_targets()
        for attr in ("scene_fbo", "bloom_fbo_a", "bloom_fbo_b", "ssr_fbo",
                     "bloom_fbo_c", "bloom_fbo_d", "ae_meter_fbo", "ao_fbo", "haze_fbo", "haze_half_fbo"):
            fbo = getattr(self, attr, 0)
            if fbo:
                gl.glDeleteFramebuffers(1, [fbo])
                setattr(self, attr, 0)
        for fbo in getattr(self, "ae_fbo", None) or ():
            if fbo: gl.glDeleteFramebuffers(1, [fbo])
        for t in getattr(self, "ae_tex", None) or ():
            if t: gl.glDeleteTextures([t])
        self.ae_fbo, self.ae_tex = [], []
        for t in getattr(self, "star_tex", None) or ():
            if t: gl.glDeleteTextures([t])
        self.star_tex = []
        for fbo in getattr(self, "star_fbo", None) or ():
            if fbo: gl.glDeleteFramebuffers(1, [fbo])
        self.star_fbo = []
        for attr in ("scene_tex", "bloom_tex_a", "bloom_tex_b", "mat_tex", "ssr_tex",
                     "scene_depth_tex", "bloom_tex_c", "bloom_tex_d", "ae_meter_tex", "ao_tex",
                     "haze_tex", "haze_half_tex"):
            tex = getattr(self, attr, 0)
            if tex:
                gl.glDeleteTextures([tex])
                setattr(self, attr, 0)
        rb = getattr(self, "scene_rb", 0)
        if rb:
            gl.glDeleteRenderbuffers(1, [rb])
            self.scene_rb = 0

    def _release_msaa_targets(self):
        gl = self.gl
        if getattr(self, "ms_fbo", 0):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glDeleteFramebuffers(1, [self.ms_fbo])
            self.ms_fbo = 0
        rbs = getattr(self, "ms_rbs", [])
        if rbs:
            gl.glDeleteRenderbuffers(len(rbs), rbs)
        self.ms_rbs = []

    def _build_msaa_targets(self, internal):
        """Multisampled twin of the scene target, resolved into it each frame.

        The window framebuffer stays single-sampled: the scene is HDR and only
        reaches the screen through the composite, so anti-aliasing must happen
        where the geometry is rasterised, not on the final blit.
        """
        gl = self.gl
        self._release_msaa_targets()
        samples = int(getattr(self, "aa_samples", 0) or 0)
        w, h = self.width, self.height
        if samples < 2 or w < 2 or h < 2 or internal is None:
            return
        fbo, rbs, ok = 0, [], False
        try:
            max_samples = int(gl.glGetIntegerv(gl.GL_MAX_SAMPLES))
            samples = min(samples, max(1, max_samples))
            if samples < 2: return
            fbo = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            for attach, fmt in ((gl.GL_COLOR_ATTACHMENT0, internal),
                                (gl.GL_COLOR_ATTACHMENT1, gl.GL_RGBA8),
                                (gl.GL_DEPTH_ATTACHMENT, gl.GL_DEPTH_COMPONENT32F)):
                rb = gl.glGenRenderbuffers(1)
                rbs.append(rb)
                gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, rb)
                gl.glRenderbufferStorageMultisample(gl.GL_RENDERBUFFER, samples, fmt, w, h)
                gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, attach,
                                             gl.GL_RENDERBUFFER, rb)
            gl.glDrawBuffers(2, np.array([gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1],
                                         dtype=np.uint32))
            ok = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
            gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
            if ok:
                gl.glEnable(gl.GL_MULTISAMPLE)
        except Exception:
            ok = False
        if ok:
            self.ms_fbo, self.ms_rbs = fbo, rbs
        else:
            for rb in rbs:
                gl.glDeleteRenderbuffers(1, [rb])
            if fbo: gl.glDeleteFramebuffers(1, [fbo])
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.scene_fbo)

    def _resolve_scene(self):
        """Copy the multisampled scene (colour, material, depth) into the textures
        the post chain samples.  Blitting resolves the samples in one go."""
        if not getattr(self, "ms_fbo", 0): return
        gl = self.gl
        w, h = self.width, self.height
        n = 2 if getattr(self, "_mrt_on", False) else 1
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.ms_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.scene_fbo)
        # A colour blit only pairs the framebuffer's single selected read buffer
        # with draw buffer 0, so attachment 1 (world normals / reflectivity) needs
        # its own pass; blitting once left the material target black.
        for i in range(n):
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 + i)
            gl.glDrawBuffers(1, np.array([gl.GL_COLOR_ATTACHMENT0 + i], dtype=np.uint32))
            gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
        # A depth blit needs a single draw buffer on the destination.
        gl.glDrawBuffers(1, np.array([gl.GL_NONE], dtype=np.uint32))
        gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.GL_DEPTH_BUFFER_BIT, gl.GL_NEAREST)
        bufs = [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1][:n]
        gl.glDrawBuffers(n, np.array(bufs, dtype=np.uint32))
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        self._mrt_n = n
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.scene_fbo)

    def set_aa(self, samples):
        """0 disables scene MSAA, 2 / 4 / 8 request that many samples."""
        samples = max(0, int(samples))
        if samples == getattr(self, "aa_samples", 0):
            return self.aa_samples
        self.aa_samples = samples
        if getattr(self, "scene_fbo", 0):
            self._build_msaa_targets(getattr(self, "_internal", None))
        return self.aa_samples

    def _build_post_targets(self, internal):
        gl = self.gl
        w, h = self.width, self.height
        bw, bh = max(1, w // 4), max(1, h // 4)      # quarter-res bloom chain
        self.bw, self.bh = bw, bh
        is_hdr = internal == gl.GL_RGBA16F
        pixel_type = gl.GL_FLOAT if is_hdr else gl.GL_UNSIGNED_BYTE

        def tex(tw, th):
            t = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, t)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, internal, tw, th, 0,
                            gl.GL_RGBA, pixel_type, None)
            for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR),
                         (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                         (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                         (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            return t

        self.scene_tex = tex(w, h)
        self.scene_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.scene_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.scene_tex, 0)
        # Depth is a texture (not a renderbuffer) so the SSR pass can sample it.
        self.scene_depth_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.scene_depth_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT32F, w, h, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        for pn, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST),
                      (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST),
                      (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                      (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, pn, v)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                  gl.GL_TEXTURE_2D, self.scene_depth_tex, 0)

        # Material target: world normal (rgb) + reflectivity (a), second colour output.
        self.mat_tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.mat_tex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        for pn, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST),
                      (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST),
                      (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                      (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
            gl.glTexParameteri(gl.GL_TEXTURE_2D, pn, v)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1,
                                  gl.GL_TEXTURE_2D, self.mat_tex, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glDrawBuffers(2, np.array([gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1],
                                     dtype=np.uint32))
        self._mrt_n = 2
        self.scene_complete = (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                               == gl.GL_FRAMEBUFFER_COMPLETE)
        self._internal = internal
        self._build_msaa_targets(internal)

        # SSR output: the finished scene (reflections mixed in); bloom reads this.
        self.ssr_tex = tex(w, h)
        self.ssr_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ssr_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.ssr_tex, 0)
        self.scene_complete = bool(self.scene_complete and
                                   gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                                   == gl.GL_FRAMEBUFFER_COMPLETE)

        # City haze output: the frame with the volumetric medium applied; bloom reads this.
        self.haze_tex = tex(w, h)
        self.haze_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.haze_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.haze_tex, 0)
        ok_full = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
        # The expensive ray-march runs at half resolution; the full-res apply pass upsamples it
        # depth-aware (see HAZE_APPLY_FS).
        self.haze_half_tex = tex(max(1, (w + 1) // 2), max(1, (h + 1) // 2))
        self.haze_half_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.haze_half_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self.haze_half_tex, 0)
        self.haze_ok = bool(is_hdr and ok_full and gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                            == gl.GL_FRAMEBUFFER_COMPLETE)

        self.aw, self.ah = max(1, (w + 1) // 2), max(1, (h + 1) // 2)
        self.ao_tex = tex(self.aw, self.ah)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ao_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        self.ao_fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ao_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.ao_tex, 0)
        self.ao_ok = is_hdr and gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE

        self.bloom_tex_a = tex(bw, bh)
        self.bloom_tex_b = tex(bw, bh)
        # starburst streak passes: source + ping-pong pair, same size as the near glow
        self.star_tex = [tex(bw, bh) for _ in range(3)]
        self.star_fbo = [gl.glGenFramebuffers(1) for _ in range(3)]
        for fbo, t in zip(self.star_fbo, self.star_tex):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_2D, t, 0)
        self.bloom_fbo_a = gl.glGenFramebuffers(1)
        self.bloom_fbo_b = gl.glGenFramebuffers(1)
        complete = self.scene_complete
        for fbo, t in ((self.bloom_fbo_a, self.bloom_tex_a),
                       (self.bloom_fbo_b, self.bloom_tex_b)):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_2D, t, 0)
            complete = (complete and gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                        == gl.GL_FRAMEBUFFER_COMPLETE)

        # Wide glow level: 1/16 resolution, blurred twice.  Gives the soft film halation
        # that the 1/4-res chain alone cannot (its blur radius is only ~30 px).
        cw, ch = max(1, w // 16), max(1, h // 16)
        self.cw, self.ch = cw, ch
        self.bloom_tex_c = tex(cw, ch)
        self.bloom_tex_d = tex(cw, ch)
        self.bloom_fbo_c = gl.glGenFramebuffers(1)
        self.bloom_fbo_d = gl.glGenFramebuffers(1)
        for fbo, t in ((self.bloom_fbo_c, self.bloom_tex_c),
                       (self.bloom_fbo_d, self.bloom_tex_d)):
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_2D, t, 0)
            complete = (complete and gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                        == gl.GL_FRAMEBUFFER_COMPLETE)

        # Eye adaptation targets (HDR scenes only): a 128x128 log-luminance meter whose
        # mip chain is averaged by the GPU, plus two 1x1 float textures that ping-pong.
        self.ae_ok = False
        if is_hdr:
            def tex_fmt(tw, th, fmt, mip):
                t = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, t)
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, fmt, tw, th, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER,
                                   gl.GL_LINEAR_MIPMAP_LINEAR if mip else gl.GL_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER,
                                   gl.GL_LINEAR if mip else gl.GL_NEAREST)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
                return t
            self.ae_meter_tex = tex_fmt(128, 128, gl.GL_RGBA16F, True)
            self.ae_meter_fbo = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ae_meter_fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_2D, self.ae_meter_tex, 0)
            ae_ok = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
            self.ae_tex, self.ae_fbo = [], []
            for _ in range(2):
                t = tex_fmt(1, 1, gl.GL_RGBA32F, False)
                f = gl.glGenFramebuffers(1)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, f)
                gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                          gl.GL_TEXTURE_2D, t, 0)
                gl.glViewport(0, 0, 1, 1)
                gl.glClearColor(0.0, 0.0, 0.0, 1.0)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                ae_ok = ae_ok and (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
                                   == gl.GL_FRAMEBUFFER_COMPLETE)
                self.ae_tex.append(t); self.ae_fbo.append(f)
            self.ae_ok = bool(ae_ok)
            self._ae_cur = 0
            self._ae_reset = True
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        # the setup above leaves a 1x1 viewport and a black clear colour behind;
        # restore both so the map / loading screens (2D overlay only) are visible
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(*BG_COLOR, 1.0)
        return bool(complete)

    def _create_post_targets(self):
        gl = self.gl
        self._release_post_targets()
        if self._build_post_targets(gl.GL_RGBA16F):
            self.hdr_scene = True
            # Preserve calibrated bloom settings across allocation / window resizing.
        else:
            # No float render target: render to 8-bit and disable bloom rather
            # than blooming the whole tonemapped image (which whites out day).
            self._release_post_targets()
            self._build_post_targets(gl.GL_RGBA8)
            self.hdr_scene = False
            self.ae_ok = False
            self.bloom_strength = 0.0
            self.bloom_threshold = 1.0

    def _neon_uniforms(self, p, chunks, cam):
        gl = self.gl
        # The nearest-48 selection only needs redoing when the chunk set changes or the
        # target moved ~10 m; it used to run (vstack + argpartition) twice every frame.
        key = (tuple(id(c) for c in chunks),
               round(float(cam.target[0]) / 10.0), round(float(cam.target[1]) / 10.0))
        if key != getattr(self, "_neon_key", None):
            self._neon_key = key
            pools = [c.neon_cpu for c in chunks
                     if c.neon_cpu is not None and len(c.neon_cpu)]
            n = 0
            self._neon_pos[:] = 0.0; self._neon_col[:] = 0.0; self._neon_scr[:] = 0.0
            if pools:
                allv = np.vstack(pools)
                tx, ty = float(cam.target[0]), float(cam.target[1])
                tz = float(cam.target[2]) if len(cam.target) > 2 else 0.0
                # Rows: xyz, rgb, then (screens only) normal xyz, half width, half height.
                width = allv.shape[1]
                nrm = allv[:, 6:9] if width >= 11 else np.zeros((len(allv), 3), np.float32)
                ext = allv[:, 9:11] if width >= 11 else np.zeros((len(allv), 2), np.float32)
                is_screen = ((nrm[:, :2] ** 2).sum(axis=1) > .25) & (ext[:, 0] > 0.) & (ext[:, 1] > 0.)
                d2 = (allv[:, 0] - tx) ** 2 + (allv[:, 1] - ty) ** 2
                # Big LED screens outrank small signs: shrink their effective distance by
                # their strength so a Times Square block is not crowded out by vertex samples.
                strength = np.maximum(allv[:, 3:6].max(axis=1), 1.0)
                # A screen only lights what is in front of its glass, so one that faces away
                # from the view target must not take a slot from one that faces it.
                facing = np.ones(len(allv), np.float64)
                if is_screen.any():
                    glass = allv[is_screen, 0:3] - nrm[is_screen] * 3.0   # SCREEN_LIGHT_SETBACK
                    to_t = np.stack((tx - glass[:, 0], ty - glass[:, 1], tz - glass[:, 2]), 1)
                    dist = np.maximum(np.linalg.norm(to_t, axis=1), 1.0)
                    cos_e = np.clip((to_t * nrm[is_screen]).sum(axis=1) / dist, 0., 1.)
                    facing[is_screen] = .15 + cos_e
                d2 = d2 / (strength * facing) ** 2
                k = min(48, len(allv))
                idx = (np.argpartition(d2, k - 1)[:k] if len(allv) > k
                       else np.arange(len(allv)))
                self._neon_pos[:k] = allv[idx, 0:3]
                col = allv[idx, 3:6].astype(np.float32)
                # Signs light their surroundings with a warm, mostly neutral spill, not their raw
                # saturated pink / cyan: keeps the street lit without coloured light blobs.
                lum = (col @ np.array([.2126, .7152, .0722], np.float32))[:, None]
                warm = lum * np.array([1., .86, .66], np.float32)
                scr = is_screen[idx][:, None]
                # Screens keep far more of their ad colour, so the pool of light on the street
                # and on the opposite facades is visibly the colour of what is on the screen.
                col = np.where(scr, col * .45 + warm * .55, col * .22 + warm * .78)
                self._neon_col[:k] = col
                sel = idx[is_screen[idx]]
                rows = np.flatnonzero(is_screen[idx])
                if len(sel):
                    xy = nrm[sel, :2]
                    xy = xy / np.maximum(np.linalg.norm(xy, axis=1, keepdims=True), 1e-6)
                    self._neon_scr[rows, 0:2] = xy
                    self._neon_scr[rows, 2:4] = ext[sel]
                n = int(k)
            self._neon_n = n
        n = self._neon_n
        gl.glUniform3fv(self._loc(p, "uNeonPos[0]"), 48, self._neon_pos.ravel())
        gl.glUniform3fv(self._loc(p, "uNeonCol[0]"), 48, self._neon_col.ravel())
        gl.glUniform4fv(self._loc(p, "uNeonScr[0]"), 48, self._neon_scr.ravel())
        self._int(p, "uNumNeon", n)

    def _draw_precip(self, cam, weather, sky, fog_near, fog_far, view, proj, eye):
        """World-space precipitation and roof-supported splashes into the bound target."""
        self.precip.air_extinction = self.air_extinction
        self.precip.camera_height = float(eye[2])
        self.precip.draw(cam, weather, sky, fog_near, fog_far, view, proj,
                         roof=self.rain_roof if self._rain_roof_active else None,
                         pixel_scale=self.height * abs(float(proj[1, 1])) * .5,
                         splashes=self.rain_splashes_enabled)
        self.last_draw_calls += sum(bool(n) for n in (self.precip.last_rain_count,
                                                     self.precip.last_snow_count,
                                                     self.precip.last_splash_count))

    def _draw_precip_after_reflections(self, cam, weather, sky, fog_near, fog_far,
                                       view, proj, eye):
        """Draw precipitation over the reflection result, clipped by scene depth.

        The scene depth texture is attached only for this draw: an attachment must
        not be sampled by the pass that produced the target.
        """
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ssr_fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                  gl.GL_TEXTURE_2D, self.scene_depth_tex, 0)
        gl.glViewport(0, 0, self.width, self.height)
        self._depth_state(self._reverse_active)
        try:
            self._draw_precip(cam, weather, sky, fog_near, fog_far, view, proj, eye)
        finally:
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                      gl.GL_TEXTURE_2D, 0, 0)

    def _draw_skid_marks(self, common, now_wetness):
        """Tyre-mark ribbons: depth-tested, alpha blended, colour-only."""
        gl, sk = self.gl, self.skid_marks
        if sk is None or sk.count == 0:
            return
        from ..vehicles.skidmarks import FLOATS
        stride = FLOATS * 4
        if self._skid_vao is None:
            self._skid_vao = gl.glGenVertexArrays(1)
            self._skid_vbo = gl.glGenBuffers(1)
            gl.glBindVertexArray(self._skid_vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._skid_vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, sk.buf.nbytes, None, gl.GL_DYNAMIC_DRAW)
            for loc, size, off in ((0, 3, 0), (1, 1, 12), (2, 1, 16), (3, 1, 20)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(off))
            self._skid_generation = -1
        gl.glBindVertexArray(self._skid_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._skid_vbo)
        if self._skid_generation != sk.generation:      # cleared: re-upload everything
            gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, sk.buf.nbytes, sk.buf)
            self._skid_generation = sk.generation
            sk.take_dirty()
        else:
            dirty = sk.take_dirty()
            if dirty is not None:
                lo, hi = dirty
                rows = np.ascontiguousarray(sk.buf[lo * 6:hi * 6])
                gl.glBufferSubData(gl.GL_ARRAY_BUFFER, lo * 6 * stride, rows.nbytes, rows)
        p = self.prog_skid
        gl.glUseProgram(p); common(p)
        self._float(p, "uNow", sk.clock)
        self._float(p, "uWet", float(np.clip(now_wetness, 0., 1.)))
        self._set_mrt(1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ZERO, gl.GL_ONE)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, sk.CAPACITY * 6)
        self.last_draw_calls += 1
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glBindVertexArray(0)

    def _set_mrt(self, n):
        """Select 2 draw buffers (colour + material) or 1 (colour only) on the scene FBO.

        Opaque passes write both; additive / blended passes must write colour
        only, otherwise their blend would corrupt the material buffer.
        """
        if not self._mrt_on or n == self._mrt_n:
            return
        gl = self.gl
        bufs = [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1][:n]
        gl.glDrawBuffers(n, np.array(bufs, dtype=np.uint32))
        self._mrt_n = n

    def _set_mrt_mat(self):
        """Route fragment output 0 into the material target only (colour stays untouched)."""
        if not self._mrt_on:
            return
        self.gl.glDrawBuffers(1, np.array([self.gl.GL_COLOR_ATTACHMENT1], dtype=np.uint32))
        self._mrt_n = 3      # 1 = colour only, 2 = colour + material, 3 = material only

    def _ao_pass(self, view, proj):
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ao_fbo)
        gl.glViewport(0, 0, self.aw, self.ah)
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_BLEND); gl.glDisable(gl.GL_CULL_FACE)
        p = self.prog_ao
        gl.glUseProgram(p)
        for unit, tex, name in ((0, self.scene_depth_tex, "uDepth"), (1, self.mat_tex, "uMat")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(self._loc(p, name), unit)
        self._mat(p, "uView", view); self._mat(p, "uProj", proj)
        self._mat(p, "uInvProj", np.linalg.inv(np.asarray(proj, np.float64)))
        self._vec2(p, "uResolution", (float(self.width), float(self.height)))
        self._float(p, "uRadius", max(0.1, self.ao_radius))
        self._float(p, "uReverseZ", float(self._reverse_active))
        gl.glBindVertexArray(self.vao_sky)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def _ssr_pass(self, view, proj, near):
        """Ray-march the depth buffer; result (scene + reflections) lands in ssr_tex."""
        gl = self.gl
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ssr_fbo)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        p = self.prog_ssr
        gl.glUseProgram(p)
        for unit, tex, name in ((0, self.scene_tex, "uScene"),
                                (1, self.scene_depth_tex, "uDepth"),
                                (2, self.mat_tex, "uMat"), (3, self.ao_tex, "uAO")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(self._loc(p, name), unit)
        self._mat(p, "uView", view)
        self._mat(p, "uProj", proj)
        self._mat(p, "uInvProj", np.linalg.inv(np.asarray(proj, np.float64)))
        self._vec2(p, "uResolution", (float(self.width), float(self.height)))
        self._float(p, "uNear", near)
        self._float(p, "uReverseZ", float(self._reverse_active))
        self._float(p, "uMaxDist", self.ssr_max_dist)
        self._float(p, "uStrength", self.ssr_strength if self.ssr_enabled else 0.0)
        self._float(p, "uAOOn", float(self.ao_enabled and self.ao_ok))
        self._float(p, "uAOStrength", self.ao_strength)
        gl.glBindVertexArray(self.vao_sky)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        gl.glActiveTexture(gl.GL_TEXTURE0)

    # ------------------------------------------------------------ eye adaptation
    MOTION_BLUR_DEFAULT = 0.60
    AE_DEFAULT_STRENGTH = 0.85      # follows the subject firmly: shade opens up, the sky behind it blows out
    AE_DEFAULT_BRIGHTNESS = 0.75

    def _ae_key(self):
        night = float(self.sky.night)
        key = self.ae_key_day * (1.0 - night) + self.ae_key_night * night
        # Bias the adaptation target, not the lights or display gamma. At the
        # default strength this lifts exposure by 0.45 stops, within the EV bounds.
        # A camera exposes an overcast day high-key and lets a thunderstorm stay dark; without
        # this the meter drags every weather back to the same mid-grey and the sky looks flat.
        storm = float(np.clip(getattr(self, '_storm', 0.0), 0.0, 1.0))
        cloud = float(np.clip(getattr(self, '_cloud', 0.0), 0.0, 1.0))
        rain = float(np.clip(getattr(self, '_rain', 0.0), 0.0, 1.0))
        weather_ev = 0.30 * cloud * (1.0 - storm) - 0.45 * rain * (1.0 - storm) - 1.35 * storm
        return key * 2.0 ** (self.ae_brightness + weather_ev * (1.0 - night))

    def reset_eye_adaptation_settings(self):
        self.ae_enabled = True
        self.ae_strength = self.AE_DEFAULT_STRENGTH
        self.ae_brightness = self.AE_DEFAULT_BRIGHTNESS
        self.reset_adaptation()

    def _auto_exposure(self, src):
        """Meter the HDR frame and move the 1x1 exposure texture toward the target."""
        gl = self.gl
        now = time.time()
        dt = 0.0 if self._ae_t is None else min(max(now - self._ae_t, 0.0), 0.1)
        self._ae_t = now
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_BLEND); gl.glDisable(gl.GL_CULL_FACE)
        gl.glBindVertexArray(self.vao_sky)

        p = self.prog_ae_meter
        gl.glUseProgram(p)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ae_meter_fbo)
        gl.glViewport(0, 0, 128, 128)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src)
        self._int(p, "uScene", 0)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ae_meter_tex)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)

        nxt = 1 - self._ae_cur
        p = self.prog_ae_adapt
        gl.glUseProgram(p)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ae_fbo[nxt])
        gl.glViewport(0, 0, 1, 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ae_meter_tex)
        self._int(p, "uMeter", 0)
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ae_tex[self._ae_cur])
        self._int(p, "uPrev", 1)
        self._float(p, "uDt", dt)
        self._float(p, "uKey", self._ae_key())
        self._float(p, "uStrength", self.ae_strength)
        self._float(p, "uMinEV", self.ae_min_ev)
        self._float(p, "uMaxEV", self.ae_max_ev)
        self._float(p, "uSpeedBrighter", self.ae_speed_brighter)
        self._float(p, "uSpeedDarker", self.ae_speed_darker)
        self._float(p, "uDead", self.ae_dead_band)
        self._float(p, "uFastK", self.ae_fast_share)
        self._float(p, "uFastRate", self.ae_fast_rate)
        self._float(p, "uReset", 1.0 if self._ae_reset else 0.0)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        self._ae_reset = False
        self._ae_cur = nxt
        gl.glActiveTexture(gl.GL_TEXTURE0)

    def ae_debug(self):
        """(EV, mean log2 luminance) the eye adaptation currently holds.  Stalls the GPU:
        for tools and tests, never call per frame."""
        gl = self.gl
        if not self.ae_ok or not self.ae_fbo:
            return None
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.ae_fbo[self._ae_cur])
        px = gl.glReadPixels(0, 0, 1, 1, gl.GL_RGBA, gl.GL_FLOAT)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        a = np.asarray(px, np.float32).reshape(-1)
        return float(a[0]), float(a[1])

    def reset_adaptation(self):
        self._ae_reset = True

    def _look_index(self):
        return int(self.look) if self.grade_enabled else 0

    def _composite_bloom(self, src=None):
        """Bloom chain + final composite (exposure, glow, tone-map, film look) to the screen."""
        src = self.scene_tex if src is None else src
        gl = self.gl
        hdr = 1.0 if self.hdr_scene else 0.0
        ae_on = bool(hdr and self.ae_enabled and self.ae_ok)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_BLEND)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glBindVertexArray(self.vao_sky)
        night_k = float(self.sky.night)
        ae_tex = self.ae_tex[self._ae_cur] if (ae_on and self.ae_tex) else self.scene_tex

        # ---- near glow: threshold in exposed HDR light, 4x downsample, one blur pair
        p = self.prog_bloom_pre
        gl.glUseProgram(p)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_a)
        gl.glViewport(0, 0, self.bw, self.bh)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, src)
        self._int(p, "uScene", 0)
        gl.glActiveTexture(gl.GL_TEXTURE3)
        gl.glBindTexture(gl.GL_TEXTURE_2D, ae_tex)
        self._int(p, "uAE", 3)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        # taps 2 px apart so the 4x downsample still averages a full 4x4 block
        self._vec2(p, "uTexel", (2.0 / self.width, 2.0 / self.height))
        thr = self.bloom_threshold * night_k + self.day_threshold * (1.0 - night_k)
        self._float(p, "uThreshold", thr)
        self._float(p, "uHdrIn", hdr)
        self._float(p, "uEVComp", self.ev_comp)
        self._float(p, "uAEOn", 1.0 if ae_on else 0.0)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        p = self.prog_bloom_blur
        gl.glUseProgram(p)
        self._int(p, "uImage", 0)
        self._vec2(p, "uTexel", (1.0 / self.bw, 1.0 / self.bh))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_b)       # horizontal: A -> B
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_a)
        self._vec2(p, "uDir", (1.0, 0.0))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_a)       # vertical: B -> A
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_b)
        self._vec2(p, "uDir", (0.0, 1.0))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # ---- wide glow: A -> 1/16 res, two blur pairs
        sun_uv, sun_base = self._sun_frame()
        sun_on = bool(hdr > 0.5 and self.sun_flare > 0.0 and sun_base > 0.0)
        flare_on = bool(hdr > 0.5 and self.flare_enabled and self.flare_gain > 0.0)
        wide_on = (self.wide_glow > 0.0 or flare_on) and hdr > 0.5
        if wide_on:
            p = self.prog_bloom_down
            gl.glUseProgram(p)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_c)
            gl.glViewport(0, 0, self.cw, self.ch)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_a)
            self._int(p, "uImage", 0)
            self._vec2(p, "uTexel", (1.0 / self.bw, 1.0 / self.bh))
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            p = self.prog_bloom_blur
            gl.glUseProgram(p)
            self._int(p, "uImage", 0)
            self._vec2(p, "uTexel", (1.0 / self.cw, 1.0 / self.ch))
            for _ in range(2):
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_d)   # C -> D
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_c)
                self._vec2(p, "uDir", (1.0, 0.0))
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_c)   # D -> C
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_d)
                self._vec2(p, "uDir", (0.0, 1.0))
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # ---- dedicated sun flare: disc, halo, aperture ghosts and anamorphic streak -> B
        if sun_on:
            p = self.prog_sun_flare
            gl.glUseProgram(p)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_b)
            gl.glViewport(0, 0, self.bw, self.bh)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.scene_depth_tex)
            self._int(p, "uDepth", 0)
            self._vec2(p, "uSunUV", sun_uv)
            self._float(p, "uSunFlare", sun_base * float(self.sun_flare))
            self._vec3(p, "uSunTint", self.sun_flare_tint)
            self._float(p, "uAspect", self.width / max(1.0, float(self.height)))
            self._float(p, "uNight", night_k)
            self._float(p, "uTime", self._tshader())
            self._float(p, "uReverseZ", float(self._reverse_active))
            self._float(p, "uGhostGain", self.sun_flare_ghost_gain)
            self._float(p, "uStreakGain", self.sun_flare_streak_gain)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # ---- screen-space lens flare: starburst + ghosts from the highlight textures -> B
        if flare_on:
            p = self.prog_star
            gl.glUseProgram(p)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.bloom_fbo_b)
            gl.glViewport(0, 0, self.bw, self.bh)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_a)
            self._int(p, "uNear", 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.bloom_tex_c)
            self._int(p, "uFar", 1)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            self._float(p, "uAspect", self.width / max(1.0, float(self.height)))
            self._float(p, "uThr", self.flare_threshold)
            self._float(p, "uFStop", self.flare_fstop)
            self._int(p, "uBlades", int(self.flare_blades))
            night_f = 1.0 - 0.75 * night_k          # higher night exposure needs quieter lamp flares
            self._float(p, "uStarGain", 0.0)
            self._float(p, "uGhostGain", self.flare_ghost_gain * self.flare_gain * night_f)
            # STAR_FS overwrites B, which would erase the sun flare; add instead.
            if sun_on:
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            if sun_on:
                gl.glDisable(gl.GL_BLEND)

            # starburst: 4 axes x 4 passes, summed additively into B (under the ghosts)
            star_gain = (self.flare_star_gain * self.flare_gain * night_f
                         * (night_k + (1.0 - night_k) * self.flare_star_day))
            if star_gain > 0.0 and len(self.star_tex) == 3:
                s0, t1, t2 = self.star_tex
                f1, f2 = self.star_fbo[1], self.star_fbo[2]
                p = self.prog_star_src
                gl.glUseProgram(p)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.star_fbo[0])
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, src)
                self._int(p, "uScene", 0)
                gl.glActiveTexture(gl.GL_TEXTURE3)
                gl.glBindTexture(gl.GL_TEXTURE_2D, ae_tex)
                self._int(p, "uAE", 3)
                gl.glActiveTexture(gl.GL_TEXTURE0)
                self._vec2(p, "uTexel", (1.0 / self.width, 1.0 / self.height))
                self._float(p, "uThr", self.flare_star_threshold)
                self._float(p, "uClamp", self.flare_star_clamp)
                self._float(p, "uHdrIn", hdr)
                self._float(p, "uEVComp", self.ev_comp)
                self._float(p, "uAEOn", 1.0 if ae_on else 0.0)
                gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

                p = self.prog_star_streak
                gl.glUseProgram(p)
                self._int(p, "uImage", 0)
                fn = float(np.clip((self.flare_fstop - 8.0) / 14.0, 0.0, 1.0))
                fn *= 1.0 - 0.75 * night_k         # keep night starbursts compact around each fixture
                decay = 0.945 + 0.025 * (fn - 0.43) / 0.57          # f/22 reaches further than f/14
                n_ax = 4
                for ax in range(n_ax):
                    ang = ax * math.pi / n_ax + 0.26                  # blade orientation
                    dx = math.cos(ang) * self.flare_star_length / self.bw
                    dy = math.sin(ang) * self.flare_star_length / self.bh
                    chain = ((s0, f1), (t1, f2), (t2, f1), (t1, self.bloom_fbo_b))
                    for k, (tin, fout) in enumerate(chain):
                        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fout)
                        gl.glBindTexture(gl.GL_TEXTURE_2D, tin)
                        scale = 4.0 ** k
                        self._vec2(p, "uStep", (dx * scale, dy * scale))
                        self._float(p, "uDecay", decay ** scale)
                        last = (k == 3)
                        self._float(p, "uGain", star_gain / n_ax if last else 1.0)
                        self._float(p, "uSharp", 0.02 if last else 0.0)  # normalized highlight units
                        if last:
                            gl.glEnable(gl.GL_BLEND)
                            gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE)
                        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
                        if last:
                            gl.glDisable(gl.GL_BLEND)

        # ---- composite
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self.width, self.height)
        p = self.prog_bloom_comp
        gl.glUseProgram(p)
        for unit, tex, name in ((0, src, "uScene"), (1, self.bloom_tex_a, "uBloom"),
                                (2, self.bloom_tex_c, "uBloomWide"), (3, ae_tex, "uAE"),
                                (4, self.scene_depth_tex, "uDepth"),
                                (5, self.bloom_tex_b, "uFlareTex")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(self._loc(p, name), unit)
        sun_uv, sun_glare = self._sun_glare_params()
        self._vec2(p, "uSunUV", (sun_uv[0], sun_uv[1]))
        self._float(p, "uSunGlare", sun_glare if hdr > 0.5 else 0.0)
        self._vec3(p, "uSunTint", self.sun_glare_tint)
        self._float(p, "uReverseZ", float(self._reverse_active))
        # Daylight has little HDR headroom beyond the sky and speculars, so the near glow
        # fades in with night; emissives carry it after dark.
        glow = self.bloom_strength * (night_k + (1.0 - night_k) * self.day_glow)
        look = self._look_index()
        self._float(p, "uStrength", glow)
        self._float(p, "uWide", (self.wide_glow * (0.55 + 0.45 * night_k)) if (wide_on and self.wide_glow > 0.0) else 0.0)
        self._float(p, "uFlareTexOn", 1.0 if (flare_on or sun_on) else 0.0)
        flare_ceiling = self.flare_max_brightness
        if sun_on:
            flare_ceiling = max(flare_ceiling, self.sun_flare_max_brightness)
        self._float(p, "uFlareMaxBrightness", flare_ceiling)
        self._float(p, "uStreak", self.streak_gain if (hdr > 0.5 and look == 2) else 0.0)
        self._vec2(p, "uResolution", (float(self.width), float(self.height)))
        self._float(p, "uNight", night_k)
        self._float(p, "uTime", self._tshader())
        self._float(p, "uGrade", self.grade_strength if look else 0.0)
        self._int(p, "uLook", look)
        self._float(p, "uHdrIn", hdr)
        self._float(p, "uEVComp", self.ev_comp)
        self._float(p, "uAEOn", 1.0 if ae_on else 0.0)
        self._float(p, "uContrast", self.contrast if self.grade_enabled else 1.0)
        # Contrast rotates around the exposure the eye adaptation is aiming for (mid-grey if it is off).
        self._float(p, "uPivot", self._ae_key() if ae_on else 0.18)
        # camera motion blur: reproject this frame's depth into the previous frame
        vp = self._cur_vp if self._cur_vp is not None else self._last_vp
        prev = self._prev_vp
        motion = float(self.motion_blur) if (prev is not None and vp is not None) else 0.0
        if motion > 0.0:
            self._mat(p, "uInvViewProj", np.linalg.inv(np.asarray(vp, np.float64)))
            self._mat(p, "uPrevViewProj", prev)
        # Player car: its pixels move rigidly with the car, not like static scenery.
        car, pcar = self._cur_car, self._prev_car
        car_on = motion > 0.0 and car is not None
        self._float(p, "uCarOn", 1.0 if car_on else 0.0)
        if car_on:
            pcar = pcar if pcar is not None else car
            gl.glUniform4f(self._loc(p, "uCarCur"), *car[:4])
            gl.glUniform4f(self._loc(p, "uCarPrev"), *pcar[:4])
            gl.glUniform3f(self._loc(p, "uCarBox"), *car[4:7])
        self._float(p, "uMotionBlur", motion)
        self._float(p, "uMotionMax", float(self.motion_blur_max))
        self._float(p, "uLetterbox", self.letterbox)
        self._float(p, "uVigStrength", self.vignette)
        self._float(p, "uVigStart", self.vignette_start)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        for unit in (5, 4, 3, 2, 1):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindVertexArray(0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)

    def _haze_floor(self, cam, eye):
        """World z of the street the haze layer rests on (smoothed so terrain steps do not pop)."""
        if self.city_haze_floor is not None:
            return float(self.city_haze_floor)
        try:
            z = float(cam.target[2])
        except Exception:
            z = 0.0
        z = min(z, float(eye[2]) - 0.5)
        prev = self._haze_floor_s
        self._haze_floor_s = z if (prev is None or abs(z - prev) > 60.0) else prev + (z - prev) * 0.05
        return self._haze_floor_s

    def _haze_lamps(self, eye, night):
        """Nearest street lamps to the camera, for the haze halos.  The selection is cached and
        only redone when the lamp set changes or the camera has moved ~8 m."""
        pos = np.zeros((HAZE_MAX_LAMPS, 3), np.float32)
        col = np.zeros((HAZE_MAX_LAMPS, 3), np.float32)
        lamps = getattr(self.lights, "_lamps", None)
        if night <= 0.05 or lamps is None or len(lamps) == 0:
            return pos, col, 0
        key = (id(lamps), len(lamps), round(float(eye[0]) / 8.0), round(float(eye[1]) / 8.0))
        if key != getattr(self, "_haze_lamp_key", None):
            d2 = (lamps[:, 0] - eye[0]) ** 2 + (lamps[:, 1] - eye[1]) ** 2
            k = min(HAZE_MAX_LAMPS, len(lamps))
            idx = np.argpartition(d2, k - 1)[:k] if len(lamps) > k else np.arange(len(lamps))
            n = len(idx)
            # normalise colour to luminance 1: the halo is warm sodium/LED white, not an emitter value
            c = np.maximum(lamps[idx, 4:7], 0.0)
            c = c / np.maximum(c @ np.array([.2126, .7152, .0722], np.float32), 1e-3)[:, None]
            self._haze_lamp_key = key
            self._haze_lamp_cache = (lamps[idx, 0:3].astype(np.float32), c.astype(np.float32), n)
        p, c, n = self._haze_lamp_cache
        pos[:n], col[:n] = p, c
        return pos, col, n

    def _haze_pass(self, src, cam, sky, weather, eye, night):
        """Ray-march the city haze through the finished HDR frame; returns the texture to bloom."""
        if (not self.city_haze_enabled or not self.haze_ok or not self.hdr_scene
                or self._hdr_out < 0.5 or self._cur_vp is None
                or not getattr(self, "haze_fbo", 0)):
            return src
        gl = self.gl
        cfg = self.city_haze
        ae_on = bool(self.ae_enabled and self.ae_ok and self.ae_tex)
        ae_tex = self.ae_tex[self._ae_cur] if ae_on else src
        night = float(np.clip(night, 0.0, 1.0))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.haze_half_fbo)
        gl.glViewport(0, 0, max(1, (self.width + 1) // 2), max(1, (self.height + 1) // 2))
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_BLEND); gl.glDisable(gl.GL_CULL_FACE)
        p = self.prog_haze
        gl.glUseProgram(p)
        for unit, tex, name in ((0, src, "uScene"), (1, self.scene_depth_tex, "uDepth"), (2, ae_tex, "uAE")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(self._loc(p, name), unit)
        self._mat(p, "uInvViewProj", np.linalg.inv(np.asarray(self._cur_vp, np.float64)))
        self._vec3(p, "uCamPos", eye)
        self._float(p, "uReverseZ", float(self._reverse_active))
        self._float(p, "uAEOn", 1.0 if ae_on else 0.0)
        self._float(p, "uEVComp", self.ev_comp)
        self._float(p, "uTime", self._tshader())
        wind = (wind_displacement(weather) if weather is not None else (0.0, 0.0))
        self._vec2(p, "uWindOffset", (float(wind[0]), float(wind[1])))
        self._float(p, "uExposure", sky.exposure)
        self._float(p, "uNight", night)
        self._vec3(p, "uSunDir", sky.light_dir)
        self._vec3(p, "uSunColor", sky.light)
        self._vec3(p, "uAmbColor", sky.amb)
        # Rain, storm, snow and overcast thicken the air (weather.fog is ~0.06 in clear weather).
        wfog = float(getattr(weather, "fog", 0.0)) if weather is not None else 0.0
        self._float(p, "uWeather", 1.0 + 1.2 * float(np.clip(wfog - 0.06, 0.0, 1.0)))
        # By day the weather system's own fog already carries rain and storms: keep only a share.
        self._float(p, "uDensityScale", cfg["DayDensity"] * (1.0 - night) + night)
        self._float(p, "uHazeFloor", self._haze_floor(cam, eye))
        for key in ("Density", "Height", "Noise", "NoiseScale", "Aniso", "Ground", "MaxDist",
                    "Glow", "Neon", "Beam", "Lamp"):
            self._float(p, "uHaze" + key, cfg[key])
        self._float(p, "uHazeAmbient", cfg["AmbientDay"] * (1.0 - night) + cfg["AmbientNight"] * night)
        self._int(p, "uHazeSteps", int(cfg["Steps"]))
        self._vec3(p, "uHazeGlowTint", HAZE_GLOW_TINT)
        # Light tables the surfaces already use: nearest neon emitters and vehicle headlights.
        gl.glUniform3fv(self._loc(p, "uNeonPos[0]"), 48, self._neon_pos.ravel())
        gl.glUniform3fv(self._loc(p, "uNeonCol[0]"), 48, self._neon_col.ravel())
        self._int(p, "uNumNeon", int(self._neon_n))
        lamp_pos, lamp_col, n_lamps = self._haze_lamps(eye, night)
        gl.glUniform3fv(self._loc(p, "uLampPos[0]"), HAZE_MAX_LAMPS, lamp_pos.ravel())
        gl.glUniform3fv(self._loc(p, "uLampCol[0]"), HAZE_MAX_LAMPS, lamp_col.ravel())
        self._int(p, "uNumLamps", int(n_lamps))
        heads = getattr(self.lights, "head_beams", None)
        n_beams = 0
        if heads is not None and night > 0.05:
            n_beams = min(len(heads), HEADLIGHT_BEAMS)
        beam_pos = np.zeros((HEADLIGHT_BEAMS, 4), np.float32)
        beam_dir = np.zeros((HEADLIGHT_BEAMS, 4), np.float32)
        if n_beams:
            beam_pos[:n_beams, :3] = heads[:n_beams, :3]
            beam_dir[:n_beams, :3] = heads[:n_beams, 3:6]
            beam_dir[:n_beams, 3] = heads[:n_beams, 6]
        gl.glUniform4fv(self._loc(p, "uBeamPos[0]"), HEADLIGHT_BEAMS, beam_pos.ravel())
        gl.glUniform4fv(self._loc(p, "uBeamDir[0]"), HEADLIGHT_BEAMS, beam_dir.ravel())
        self._int(p, "uBeamCount", int(n_beams))
        gl.glBindVertexArray(self.vao_sky)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # Full-resolution finish: depth-aware upsample + emitter exemption.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.haze_fbo)
        gl.glViewport(0, 0, self.width, self.height)
        q = self.prog_haze_apply
        gl.glUseProgram(q)
        for unit, tex, name in ((0, src, "uScene"), (1, self.scene_depth_tex, "uDepth"),
                                (2, ae_tex, "uAE"), (3, self.haze_half_tex, "uHalf")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glUniform1i(self._loc(q, name), unit)
        self._float(q, "uReverseZ", float(self._reverse_active))
        self._float(q, "uAEOn", 1.0 if ae_on else 0.0)
        self._float(q, "uEVComp", self.ev_comp)
        self._float(q, "uHazeEmitter", cfg["Emitter"])
        gl.glUniform2f(self._loc(q, "uProjZ"), *self._cur_proj_z)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        for unit in (3, 2, 1):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        return self.haze_tex

    def _sun_frame(self):
        """Sun position in the frame (uv) and its un-gained visibility.

        Strength follows the sun's own intensity (0 below the horizon) and is
        softened by overcast; it is 0 while the sun is behind the camera.
        """
        sky, vp = self.sky, self._last_vp
        if sky is None or vp is None:
            return (0.5, 0.5), 0.0
        d = np.asarray(sky.sun_dir, np.float64)
        clip = np.asarray(vp, np.float64) @ np.array([d[0], d[1], d[2], 0.0])
        if not np.isfinite(clip).all() or clip[3] <= 1e-4:
            return (0.5, 0.5), 0.0
        uv = clip[:2] / clip[3] * 0.5 + 0.5
        cloud = float(np.clip(self._cloud, 0.0, 1.0))
        gain = float(sky.sun_intensity) * (1.0 - 0.5 * cloud)
        return (float(uv[0]), float(uv[1])), gain

    def _sun_glare_params(self):
        """Sun position and soft-glare strength for the post composite."""
        uv, gain = self._sun_frame()
        return uv, gain * float(self.sun_glare)

    def _grow(self, chunk, now=None):
        t = min(1.0, ((time.time() if now is None else now) - chunk.born) / self.GROW_TIME)
        return 1.0 - (1.0 - t) ** 3

    def shadow_extent(self, cam):
        e = float(np.clip(max(cam.distance * 1.4 + 150.0,
                              float(cam.position[2]) * 2.5 + 200.0), 250.0, 2800.0))
        # Quantise to 25 % steps.  A continuously changing extent (every scroll-wheel tick,
        # every orbit nudge) changed the light matrix and forced a full shadow-map render
        # on every single frame.
        step = math.ceil(math.log(e / 250.0) / math.log(1.25) - 1e-9)
        return min(2800.0, 250.0 * 1.25 ** max(0, step))

    def vehicle_shadow_extent(self, cam):
        # A close light box resolves body/tyre silhouettes without redrawing the
        # 4K city map. Elevated/aerial views expand it, with snapped size changes.
        # Street level uses a 96 m half-extent (4.7 cm texels at 4096, 2.3 cm at 8192;
        # the old 128 m / 2048 box gave 12.5 cm).
        e = float(np.clip(max(cam.distance * .8 + 48., float(cam.position[2]) * 1.25 + 64.), 96., 768.))
        step = math.ceil(math.log(e / 96.) / math.log(1.5) - 1e-9)
        return min(768., 96. * 1.5 ** max(0, step))

    def near_shadow_extent(self, cam):
        # Fixed XY density around the eye/pivot, including on high rooftops.
        # The distant city map still grows to cover aerial views.
        return 128.0

    @property
    def shadow_memory_bytes(self):
        return sum(s.nbytes for s in (self.shadow, self.vehicle_shadow, getattr(self, 'near_shadow', None))
                   if s is not None)

    def release_shadows(self):
        for shadow in (self.shadow, self.vehicle_shadow, getattr(self, 'near_shadow', None)):
            if shadow is not None: shadow.close()
        self.shadow = self.vehicle_shadow = self.near_shadow = None
        self._shadow_key = self._vehicle_shadow_key = self._near_shadow_key = None
        self._near_light = None
        self.near_shadows_enabled = False
        self._vehicle_shadow_active = False

    def _render_sky(self, cam, sky, view, proj, weather=None):
        gl = self.gl
        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE); gl.glDisable(gl.GL_CULL_FACE)
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        p = self.prog_sky
        gl.glUseProgram(p)
        self._mat(p, 'uInvProjection', np.linalg.inv(proj.astype(np.float64)))
        self._mat3(p, 'uInvViewRotation', view[:3, :3].T)
        self._vec3(p, "uCamPos", cam.position)
        self._vec3(p, "uSunDir", sky.sun_dir)
        self._int(p, "uSkyLUT", self.SKY_LUT_UNIT)
        self._float(p, "uSkyLutOn", float(getattr(self, "sky_lut_ok", False)))
        self._vec3(p, "uFogColor", sky.fog)
        self._float(p, "uSunIntensity", sky.atmosphere_intensity)
        self._float(p, 'uSunDiscIntensity', sky.sun_intensity)
        self._float(p, "uNightFactor", sky.night)
        self._float(p, "uSkyGlow", sky.sky_glow)
        self._float(p, "uExposure", sky.exposure)
        self._float(p, "uHdrOut", self._hdr_out)
        self._float(p, "uTime", self._tshader())
        self._atmosphere_uniforms(p, sky)
        self._celestial_uniforms(p, sky)
        self._weather_uniforms(p, weather)
        gl.glBindVertexArray(self.vao_sky)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)
        if self.stars_enabled and sky.star_visibility > .001:
            if self.starfield is None and self.starfield_error is None:
                try:
                    self.starfield = StarField(gl)
                except Exception as exc:
                    self.starfield_error = str(exc)  # keep atmosphere / Moon without a retry every frame
            if self.starfield is not None:
                self.starfield.draw(sky, view, proj, self.height * abs(float(proj[1, 1])) * .5,
                    weather, hdr=self._hdr_out > .5, time=self._tshader(), guides=self.constellations_enabled,
                    origin=cam.position[:2])
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)

    def _atmosphere_uniforms(self, p, sky):
        for name, value in (('uAtmosphereHeight', sky.observer_height), ('uAerosolDensity', sky.aerosol_density),
                            ('uAtmosphereIntensity', sky.atmosphere_intensity), ('uSkyGlow', sky.sky_glow)):
            self._float(p, name, value)

    def _celestial_uniforms(self, p, sky):
        rotation = sky.equatorial_to_world if sky.equatorial_to_world is not None else np.eye(3, dtype=np.float32)
        self._mat3(p, 'uWorldToEquatorial', rotation.T)
        self._vec3(p, 'uSkyMoonDir', sky.moon_dir)
        for name, value in (('uMoonIntensity', sky.moon_intensity), ('uMoonPhase', sky.moon_phase),
                            ('uMoonRadius', sky.moon_radius), ('uMoonBrightness', sky.moon_brightness),
                            ('uMilkyWayVisibility', sky.milky_way_visibility if self.stars_enabled else 0.)):
            self._float(p, name, value)

    # ------------------------------------------------------------ spatial index
    @staticmethod
    def _world_boxes(cells, origin, scale):
        """Cell rows (mesh-local) -> (first, count, world AABBs)."""
        sx, sy = float(scale[0]), float(scale[1])
        ox, oy = float(origin[0]), float(origin[1])
        b = np.empty((len(cells), 6), np.float64)
        x0, x1 = cells[:, 2] * sx + ox, cells[:, 5] * sx + ox
        y0, y1 = cells[:, 3] * sy + oy, cells[:, 6] * sy + oy
        b[:, 0], b[:, 3] = np.minimum(x0, x1), np.maximum(x0, x1)
        b[:, 1], b[:, 4] = np.minimum(y0, y1), np.maximum(y0, y1)
        b[:, 2], b[:, 5] = cells[:, 4], cells[:, 7]
        return cells[:, 0].astype(np.int64), cells[:, 1].astype(np.int64), b

    def _chunk_bounds(self, b_mesh, g_mesh, w_mesh, origin, scale):
        """Tight world AABB of a chunk plus its per-cell tables (None if unavailable)."""
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
        sx, sy = float(scale[0]), float(scale[1])
        ox, oy = float(origin[0]), float(origin[1])
        for m in (b_mesh, g_mesh, w_mesh):
            if m is None or len(m.vertices) == 0: continue
            v = m.vertices
            bounds = m.bounds if m.bounds is not None else np.r_[v[:, :3].min(axis=0), v[:, :3].max(axis=0)]
            a, z = bounds[:3], bounds[3:]
            if m is w_mesh:
                from ..world.water_mesh import WATER_BOUNDS_PAD
                a, z = a - WATER_BOUNDS_PAD, z + WATER_BOUNDS_PAD
            xs = sorted((a[0] * sx + ox, z[0] * sx + ox)); ys = sorted((a[1] * sy + oy, z[1] * sy + oy))
            lo = np.minimum(lo, [xs[0], ys[0], a[2]]); hi = np.maximum(hi, [xs[1], ys[1], z[2]])
        if not np.isfinite(lo).all():
            return None, None, None
        def table(m):
            c = getattr(m, "cells", None) if m is not None else None
            return None if c is None or len(c) == 0 else self._world_boxes(c, origin, scale)
        return np.concatenate([lo, hi]), table(b_mesh), table(g_mesh)

    def _chunk_dist(self, c, eye):
        """Distance from the camera to the chunk's nearest bound (cached per frame)."""
        d = self._dist_cache.get(id(c))
        if d is None:
            if c.aabb is None:
                d = 0.0
            else:
                lo, hi = c.aabb[:3], c.aabb[3:]
                gap = np.maximum(np.maximum(lo - eye, eye - hi), 0.0)
                d = float(np.sqrt(gap @ gap))
            self._dist_cache[id(c)] = d
        return d

    @staticmethod
    def _boxes_visible(box, planes):
        # Six vector operations avoid an N x 6 x 3 temporary and its reductions.
        visible = np.ones(len(box), bool)
        for x, y, z, d in planes:
            visible &= (box[:, 3 if x >= 0 else 0]*x +
                        box[:, 4 if y >= 0 else 1]*y +
                        box[:, 5 if z >= 0 else 2]*z + d) >= 0
        return visible

    def _prepare_cell_visibility(self, chunks, planes):
        """Cull all resident mesh cells together, once per camera/light pass."""
        key = tuple(id(c) for c in chunks)
        if key != self._cell_batch_key:
            tables = [getattr(c, 'cell_' + kind, None) for c in chunks
                      for kind in ('b', 'b_lod', 'g', 'w', 'prop', 's')]
            tables = [t for t in tables if t is not None]
            self._cell_slices = []
            first = 0
            for table in tables:
                end = first + len(table[2])
                self._cell_slices.append((id(table), slice(first, end)))
                first = end
            self._cell_boxes = (np.concatenate([t[2] for t in tables]) if tables
                                else np.empty((0, 6)))
            self._cell_batch_key = key
            self._cell_draw_cache.clear()
        box = self._cell_boxes
        visible = self._boxes_visible(box, planes)
        gap = np.maximum(np.maximum(box[:, :3]-self._eye, self._eye-box[:, 3:]), 0)
        d2 = np.einsum('ij,ij->i', gap, gap)
        self._cell_visibility = {key: (visible[sl], d2[sl]) for key, sl in self._cell_slices}

    def _cell_ranges(self, table, planes, distance=None, minimum=None):
        """Index ranges of the cells that pass `planes`; None means 'draw everything'."""
        first, count, box = table
        cached = self._cell_visibility.get(id(table))
        if cached is None:
            vis = self._boxes_visible(box, planes)
            gap = np.maximum(np.maximum(box[:, :3] - self._eye, self._eye - box[:, 3:]), 0)
            d2 = np.einsum('ij,ij->i', gap, gap)
        else:
            vis, d2 = cached
        if distance is not None: vis = vis & (d2 <= distance*distance)
        if minimum is not None: vis = vis & (d2 > minimum*minimum)
        if vis.all():
            return None
        key = (id(table), distance, minimum)
        mask = vis.tobytes()
        previous = self._cell_draw_cache.get(key)
        if previous is not None and previous[0] == mask:
            return previous[1]
        f, n = first[vis], count[vis]
        if len(f) == 0:
            result = (np.empty(0, np.int32), np.empty(0, np.uintp))
            self._cell_draw_cache[key] = (mask, result)
            return result
        # adjacent cells merge into one draw: a new run starts wherever a cell does
        # not begin exactly where the previous one ended
        brk = np.empty(len(f), bool)
        brk[0] = True
        brk[1:] = f[1:] != f[:-1] + n[:-1]
        starts = np.flatnonzero(brk)
        ends = np.append(starts[1:], len(f))
        cs = np.concatenate(([0], np.cumsum(n)))
        result = (np.ascontiguousarray(cs[ends] - cs[starts], np.int32),
                  np.ascontiguousarray(f[starts]*4, np.uintp))
        self._cell_draw_cache[key] = (mask, result)
        return result

    def _draw_tris(self, c, which, planes, distance=None, minimum=None):
        """Draw indexed city/water meshes, skipping conservatively bounded cells."""
        gl = self.gl
        count = getattr(c, {"b":"b_count", "g":"g_count", "w":"w_count", "b_lod":"b_lod_count", "prop":"p_count", "s":"s_count"}[which])
        table = getattr(c, "cell_" + which)
        rng = None if (table is None or planes is None) else self._cell_ranges(table, planes, distance, minimum)
        if rng is None:
            gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)
            self.last_draw_calls += 1
            if which == 'w': self.last_water_tris += count // 3
            return
        counts, offsets = rng
        if len(counts):
            gl.glMultiDrawElements(gl.GL_TRIANGLES, counts, gl.GL_UNSIGNED_INT,
                                   offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p)), len(counts))
            self.last_draw_calls += 1
            if which == 'w': self.last_water_tris += int(counts.sum()) // 3

    # ------------------------------------------------------------ sky look-up table
    SKY_LUT_W, SKY_LUT_H, SKY_LUT_UNIT = 192, 96, 7
    # NOTE: unit 7, not 6 - spill.py binds the lamp-record texture to unit 6
    # (_UNIT_REC), and lamp uniforms are set *after* common() for the building
    # pass, which used to overwrite the LUT binding right before envSky() read
    # it: glass then sampled lamp records (garbage: huge R, negative G) as the
    # sky reflection = magenta banding above eye level.

    def _init_sky_lut(self):
        """Small RGBA16F table of the real atmosphere, sampled by envSky() for reflections."""
        gl = self.gl
        self.sky_lut_ok = False
        self._sky_lut_key = None
        self.sky_lut_tex = self.sky_lut_fbo = self.prog_sky_lut = 0
        draw_fbo = int(gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING))
        read_fbo = int(gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING))
        texture = int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D))
        try:
            self.prog_sky_lut = self._program(SKY_LUT_VS, SKY_LUT_FS)
            tex = self.sky_lut_tex = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, self.SKY_LUT_W, self.SKY_LUT_H, 0,
                            gl.GL_RGBA, gl.GL_FLOAT, None)
            for p, v in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR), (gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR),
                         (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE), (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, p, v)
            fbo = self.sky_lut_fbo = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, tex, 0)
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError('incomplete atmosphere lookup target')
            self.sky_lut_ok = True
        except Exception as exc:
            self.sky_lut_error = str(exc)
            self._release_sky_lut()
        finally:
            gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, draw_fbo)
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, read_fbo)
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

    def _render_sky_lut(self, sky):
        """Refresh the table when the sky changed, and bind it for the reflecting shaders."""
        gl = self.gl
        self.last_sky_lut_draws = 0
        if not hasattr(self, "sky_lut_ok"):
            self._init_sky_lut()
        if not self.sky_lut_ok:
            return
        sd = np.asarray(sky.sun_dir, np.float64)
        # The atmosphere is azimuthally symmetric about the Sun. Camera turns,
        # cloud wind and sidereal stars never dirty this scattering-only table.
        key = (round(float(sd[2]), 3) if sky.atmosphere_intensity > .001 else 0.,
               round(float(sky.atmosphere_intensity), 3), round(float(sky.night), 3),
               round(float(sky.sky_glow), 4), round(float(sky.observer_height) / 20.),
               round(float(sky.aerosol_density), 2))
        if key != self._sky_lut_key:
            self._sky_lut_key = key
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.sky_lut_fbo)
            gl.glViewport(0, 0, self.SKY_LUT_W, self.SKY_LUT_H)
            gl.glDisable(gl.GL_DEPTH_TEST); gl.glDisable(gl.GL_CULL_FACE); gl.glDisable(gl.GL_BLEND)
            p = self.prog_sky_lut
            gl.glUseProgram(p)
            self._vec3(p, "uSunDir", sd)
            self._float(p, "uSunIntensity", sky.atmosphere_intensity)
            self._float(p, "uNightFactor", sky.night)
            self._float(p, "uSkyGlow", sky.sky_glow)
            self._atmosphere_uniforms(p, sky)
            gl.glBindVertexArray(self.vao_sky)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            gl.glBindVertexArray(0)
            self.sky_lut_refreshes += 1
            self.last_sky_lut_draws = 1
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.SKY_LUT_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.sky_lut_tex)
        gl.glActiveTexture(gl.GL_TEXTURE0)

    @property
    def star_count(self):
        return self.starfield.count if self.starfield is not None else 0

    @property
    def atmosphere_memory_bytes(self):
        """Scoped LUT + catalog GPU payload, excluding programs / driver overhead."""
        lut = self.SKY_LUT_W * self.SKY_LUT_H * 8 if getattr(self, 'sky_lut_tex', 0) else 0
        return lut + (self.starfield.nbytes if self.starfield is not None else 0)

    def _release_sky_lut(self):
        gl = self.gl
        if getattr(self, 'sky_lut_fbo', 0):
            gl.glDeleteFramebuffers(1, [self.sky_lut_fbo])
        if getattr(self, 'sky_lut_tex', 0):
            gl.glDeleteTextures([self.sky_lut_tex])
        if getattr(self, 'prog_sky_lut', 0):
            program = self.prog_sky_lut
            gl.glDeleteProgram(program)
            for cache in (self._uloc, self._uniform_values):
                for key in [key for key in cache if key[0] == program]:
                    del cache[key]
        self.sky_lut_tex = self.sky_lut_fbo = self.prog_sky_lut = 0
        self.sky_lut_ok = False
        self._sky_lut_key = None

    def release_atmosphere(self):
        """Release the shared lookup and static star resources before context teardown."""
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.SKY_LUT_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        if self.starfield is not None:
            self.starfield.release()
            self.starfield = None
        self._release_sky_lut()
        self.last_sky_lut_draws = 0

    def _held_light_dir(self, d):
        """Shadow-map light direction, re-aimed only when the sun moved > 0.3 deg.

        While the sky clock runs the sun creeps every frame; re-rendering the
        shadow map each time would defeat caching.  0.3 deg is invisible.
        """
        d = np.asarray(d, np.float64); d = d / max(1e-9, np.linalg.norm(d))
        h = self._shadow_dir
        if h is None or float(h @ d) < 0.99998629:        # cos(0.3 deg)
            self._shadow_dir = d
        return self._shadow_dir

    def _shadow_pass(self, casters, light, chunks):
        self._city_shadow_pass(casters, light, chunks, self.shadow)
        self.last_shadow_draws = len(casters)

    def _near_shadow_pass(self, casters, light, chunks):
        self._city_shadow_pass(casters, light, chunks, self.near_shadow)
        self.last_near_shadow_draws = len(casters)
        self.near_shadow_refreshes += 1

    def _city_shadow_pass(self, casters, light, chunks, shadow):
        gl = self.gl
        self._depth_state(False)
        shadow.begin()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        # Receiver-plane comparisons handle slopes. A slope-scaled caster
        # offset adds whole texels of separation and erases small AC shadows.
        gl.glPolygonOffset(0.0, 2.0)
        now = time.time()
        
        p = self.prog_shadow
        gl.glUseProgram(p)
        self._mat(p, "uLightSpaceMatrix", light.matrix)
        planes = frustum_planes(light.matrix.astype(np.float64))   # cull cells vs the light box
        self._prepare_cell_visibility(chunks, planes)
        
        for c in casters:
            if c.b_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
            self._float(p, "uGrow", self._grow(c, now))
            gl.glBindVertexArray(c.vao_b)
            self._draw_tris(c, "b", planes)
            
        for c in casters:
            if c.g_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
            self._float(p, "uGrow", 1.)
            gl.glBindVertexArray(c.vao_g)
            self._draw_tris(c, "g", planes)

        p = self.prog_shadow_tree
        gl.glUseProgram(p)
        self._bind_leaf_texture(p)
        self._mat(p, "uLightSpaceMatrix", light.matrix)
        self._draw_foliage_poles(casters, planes, p, tree_light=light)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glBindVertexArray(0)
        shadow.end(self.width, self.height)
        self._depth_state(self._reverse_active)

    def _update_city_shadow(self, chunks, spheres, light, eye, now, *, near=False):
        planes = frustum_planes(light.matrix.astype(np.float64))
        selected = sphere_mask(spheres[:, :3], spheres[:, 3], eye, math.inf, planes) if chunks else ()
        casters = [c for c, ok in zip(chunks, selected) if ok]
        label = 'near_shadow' if near else 'shadow'
        setattr(self, f'last_{label}_casters', len(casters))
        key = (light.matrix.tobytes(), tuple(np.floor(eye / 64.0).astype(int)),
               tuple((id(c), c.b_count, c.g_count, c.t_count) for c in casters))
        growing = any(self._grow(c, now) < 1.0 for c in casters)
        age = getattr(self, f'_{label}_age', 0) + 1
        settled = getattr(self, f'_{label}_growing', False) and not growing
        setattr(self, f'_{label}_growing', growing)
        # Independent maps reuse static depth; refresh growth every third frame
        # and once at settlement, matching the existing city-cache policy.
        if key != getattr(self, f'_{label}_key') or settled or (growing and age >= 3):
            draw = self._near_shadow_pass if near else self._shadow_pass
            draw(casters, light, chunks)
            setattr(self, f'_{label}_key', key)
            age = 0
        else:
            setattr(self, f'last_{label}_draws', 0)
        setattr(self, f'_{label}_age', age)

    def _vehicle_shadow_pass(self, light):
        gl = self.gl
        self._depth_state(False)
        self.vehicle_shadow.begin()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glPolygonOffset(1., 1.)
        p = self.prog_shadow_vehicle
        gl.glUseProgram(p)
        self._mat(p, 'uLightSpaceMatrix', light.matrix)
        self._vec3(p, 'uOffset', (0., 0., 0.))
        self._vec2(p, "uScale", (1., 1.))
        for k, count in enumerate(self.vehicle_shadow_counts[:self.signal_kind]):
            if not count: continue
            template = k % self.vehicle_kinds + 2*self.vehicle_kinds
            opaque = self.vehicle_opaque_counts[template]
            gl.glBindVertexArray(self.vehicle_shadow_vaos[k])
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, opaque, gl.GL_UNSIGNED_INT, None, count)
            self.last_vehicle_shadow_draws += 1
            self.last_vehicle_shadow_tris += opaque // 3 * count
            self.last_draw_calls += 1
        if self.peds is not None and light is not None:
            self.peds.draw_shadow(light)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
        gl.glBindVertexArray(0)
        self.vehicle_shadow.end(self.width, self.height)
        self._depth_state(self._reverse_active)
        self.vehicle_shadow_refreshes += 1

    def _depth_state(self, reverse):
        gl = self.gl
        if self.reverse_z:
            gl.glClipControl(gl.GL_LOWER_LEFT, gl.GL_ZERO_TO_ONE if reverse else gl.GL_NEGATIVE_ONE_TO_ONE)
        gl.glClearDepth(0.0 if reverse else 1.0)
        gl.glDepthFunc(gl.GL_GEQUAL if reverse else gl.GL_LEQUAL)

    def _weather_uniforms(self, p, weather=None):
        gl = self.gl
        w = weather
        self._float(p, "uWetness", w.wetness if w is not None else 0.0)
        self._float(p, "uRain", w.rain if w is not None else 0.0)
        self._float(p, "uSnow", w.snow if w is not None else 0.0)
        self._float(p, "uCloudCover", w.cloud if w is not None else 0.0)
        self._float(p, "uStorm", w.storm if w is not None else 0.0)
        self._float(p, "uWeatherTime", w.time if w is not None else 0.0)
        self._float(p, "uLightning", getattr(w, 'lightning', 0.0))
        night = _smoothstep(-2., -12., math.degrees(self.sky.alt))
        night_fill = .04 + .40 * self.sky.light_pollution + .15 * self.sky.moon_brightness * self.sky.moon_intensity
        # Under cloud the city lights the deck from below: far brighter than the open night sky,
        # and warm (sodium / LED sky-glow) rather than the cool moonlit blue.
        cover = float(np.clip(self._cloud, 0., 1.))
        warm = cover * float(np.clip(self.sky.light_pollution / .35, 0., 1.)) * .8
        night_tint = np.array([.65, .72, .82]) * (1. - warm) + np.array([.90, .74, .58]) * warm
        cloud_light = (self.sky.amb * .52 * (1. - night)
                       + night_tint * (night_fill * (1. + 5. * cover) / NITS_PER_SCENE_UNIT) * night)
        self._vec3(p, 'uCloudLight', cloud_light)
        self._vec3(p, 'uCloudSunLight', self.sky.cloud_sun if self.sky.cloud_sun is not None else self.sky.light)
        self._vec3(p, 'uCloudSunDir', self.sky.sun_dir)
        origin = getattr(self, '_eye', np.zeros(3))
        self._vec3(p, 'uCloudOrigin', (float(origin[0]), float(origin[1]), self.sky.observer_height))
        if w is not None:
            self._vec2(p, "uWind", (float(w.wind[0]), float(w.wind[1])))
            gl.glUniform2f(self._loc(p, 'uWindOffset'), *map(float, wind_displacement(w)))
        else:
            self._vec2(p, "uWind", (0.0, 0.0))
            self._vec2(p, "uWindOffset", (0., 0.))
        roof = self.rain_roof
        active = self._rain_roof_active and roof is not None and roof.complete
        self._float(p, 'uRainRoofOn', float(active))
        self._int(p, "uRainRoof", RainOcclusionMap.UNIT)
        if active:
            self._mat(p, 'uRainViewProj', roof.matrix)
            self._mat(p, 'uRainInvViewProj', roof.inv_matrix)
            self._float(p, 'uRainRoofBias', roof.bias)
            self._vec2(p, "uRainRoofTexel", (1. / roof.size, 1. / roof.size))

    @property
    def weather_memory_bytes(self):
        """Scoped retained GPU buffer/depth payload, not total driver VRAM."""
        return self.precip.nbytes + (self.rain_roof.nbytes if self.rain_roof is not None else 0)

    def release_weather(self):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + RainOcclusionMap.UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        if self.rain_roof is not None:
            self.rain_roof.release()
            self.rain_roof = None
        self.precip.release()
        self._rain_roof_active = False
        self.last_weather_map_draws = 0

    def _update_weather_roof(self, chunks, cam, weather, now, wireframe=False):
        self.last_weather_map_draws = 0
        wet = max((getattr(weather, name, 0.) for name in ('rain', 'snow', 'wetness')), default=0.)
        active = self.weather_map_enabled and not wireframe and wet > .01
        self._rain_roof_active = False
        if active and self.rain_roof is None and self.weather_map_error is None:
            try:
                self.rain_roof = RainOcclusionMap(self.gl)
            except Exception as exc:
                self.weather_map_error = str(exc)  # particles retain the depth-tested fallback
        roof = self.rain_roof
        if active and roof is not None and roof.complete:
            eye = cam.position
            boxes = [c.aabb for c in chunks if c.aabb is not None]
            bottom = min([float(eye[2]) - 64.] + [float(b[2]) - 8. for b in boxes])
            top = max([float(eye[2]) + 128.] + [float(b[5]) + 8. for b in boxes])
            roof.configure(eye, bottom, top)
            planes = frustum_planes(roof.matrix.astype(np.float64))
            casters = [c for c in chunks if np.all(c.sphere[:3] @ planes[:, :3].T + planes[:, 3] >= -c.sphere[3])]
            key = (roof.matrix.tobytes(), tuple((id(c), c.b_count, c.g_count, c.w_count, c.p_count,
                                                c.offset.tobytes(), c.scale.tobytes()) for c in casters))
            growing = any(self._grow(c, now) < 1. for c in casters if c.b_count)
            self._weather_map_age += 1
            settled = self._weather_map_growing and not growing
            self._weather_map_growing = growing
            if roof.key != key or settled or (growing and self._weather_map_age >= 3):
                gl = self.gl
                self._depth_state(False)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, roof.fbo)
                gl.glViewport(0, 0, roof.size, roof.size)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_TRUE)
                gl.glDisable(gl.GL_BLEND)
                gl.glDisable(gl.GL_CULL_FACE)
                gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
                gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
                p = self.prog_shadow
                gl.glUseProgram(p)
                self._mat(p, 'uLightSpaceMatrix', roof.matrix)
                self._prepare_cell_visibility(chunks, planes)
                for c in casters:
                    self._vec3(p, 'uOffset', c.offset)
                    gl.glUniform2f(self._loc(p, 'uScale'), *map(float, c.scale))
                    for vao, count, kind in ((c.vao_b, c.b_count, 'b'), (c.vao_g, c.g_count, 'g'),
                                              (c.vao_w, c.w_count, 'w'), (c.vao_prop, c.p_count, 'prop')):
                        if not count:
                            continue
                        self._float(p, 'uGrow', self._grow(c, now) if kind == 'b' else 1.)
                        gl.glBindVertexArray(vao)
                        if kind == 'w':
                            gl.glDrawElements(gl.GL_TRIANGLES, count, gl.GL_UNSIGNED_INT, None)
                            self.last_draw_calls += 1
                        else:
                            self._draw_tris(c, kind, planes)
                        self.last_weather_map_draws += 1
                gl.glBindVertexArray(0)
                gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.scene_fbo if self._mrt_on else 0)
                gl.glViewport(0, 0, self.width, self.height)
                self._depth_state(self._reverse_active)
                roof.key = key
                self.weather_map_refreshes += 1
                self._weather_map_age = 0
            self._rain_roof_active = True
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0 + RainOcclusionMap.UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, roof.texture if self._rain_roof_active else 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)

    @staticmethod
    def _shadow_depth_bias(light, metres):
        # Orthographic world -> [0,1] depth scale. Also supports inspection
        # matrices without LightMatrix metadata and close maps with long depth.
        return float(metres) * .5 * float(np.linalg.norm(light.matrix[2, :3])) if light is not None else 0.

    def _shadow_uniforms(self, p, light, sky):
        strength = sky.strength if (self.shadow is not None and light is not None) else 0.0
        if light is not None: self._mat(p, "uLightSpaceMatrix", light.matrix)
        self._int(p, "uShadowMap", 1)
        self._float(p, "uShadowTexel", 1.0 / self.shadow.size if self.shadow else 1.0)
        self._float(p, "uShadowTexelWorld", light.texel if light else 1.0)
        self._float(p, "uShadowBias", self._shadow_depth_bias(light, self.shadow_bias))
        self._float(p, "uShadowStrength", strength)
        near = getattr(self, '_near_light', None)
        near_map = getattr(self, 'near_shadow', None)
        if near is not None: self._mat(p, 'uNearLightSpaceMatrix', near.matrix)
        self._int(p, "uNearShadowMap", self.NEAR_SHADOW_UNIT)
        self._float(p, 'uNearShadowTexel', 1. / near_map.size if near_map else 1.)
        self._float(p, 'uNearShadowBias', self._shadow_depth_bias(near, self.shadow_bias))
        self._float(p, 'uNearShadowStrength', sky.strength if near_map is not None and near is not None else 0.)
        vehicle = getattr(self, '_vehicle_light', None)
        vehicle_map = getattr(self, 'vehicle_shadow', None)
        active = bool(getattr(self, '_vehicle_shadow_active', False))
        if vehicle is not None: self._mat(p, 'uVehicleLightSpaceMatrix', vehicle.matrix)
        self._int(p, "uVehicleShadowMap", self.VEHICLE_SHADOW_UNIT)
        self._float(p, 'uVehicleShadowTexel', 1. / vehicle_map.size if vehicle_map else 1.)
        self._float(p, 'uVehicleShadowTexelWorld', vehicle.texel if vehicle else 1.)
        self._float(p, 'uVehicleShadowBias', self._shadow_depth_bias(vehicle, .025))
        self._float(p, 'uVehicleShadowStrength', sky.strength if active and vehicle is not None else 0.)

    def draw(self, cam, chunks, wireframe=False, show_edges=False, sky=None,
             traffic=None, weather=None, pedestrians=None):
        gl = self.gl
        sky = sky or self.sky or compute_sky(13.0)
        self.sky, self.sun_dir = sky, sky.light_dir.copy()
        self._render_sky_lut(sky)          # before the scene target is bound
        use_post = bool(self.scene_complete) and not wireframe
        self._reverse_active = self.reverse_z and use_post
        self._depth_state(self._reverse_active)
        self._mrt_on = use_post
        # The sky is cleared and drawn first, so it must land in the same target
        # the geometry uses; otherwise the resolve replaces it with a bare frame.
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, (self.ms_fbo or self.scene_fbo) if use_post else 0)
        self._mrt_n = 2 if use_post else 0
        self._hdr_out = 1.0 if (use_post and self.hdr_scene) else 0.0
        self.lights.hdr_out = self._hdr_out
        self.lights.exposure = sky.exposure
        self.precip.hdr_out = self._hdr_out
        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(*[float(c) for c in sky.sky], 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_GEQUAL if self._reverse_active else gl.GL_LEQUAL)
        gl.glDepthMask(gl.GL_TRUE)
        if use_post:
            self._mrt_n = 0; self._set_mrt(2)       # re-assert both targets
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if use_post:
            # material target: flat normal, not reflective (sky stays this way)
            gl.glClearBufferfv(gl.GL_COLOR, 1, np.array([0.5, 0.5, 1.0, 0.0], np.float32))
            self._set_mrt(1)                        # sky writes colour only
        view, proj = cam.view(), cam.projection(self.width / self.height)
        cull_proj = proj
        # Motion blur reprojects into the previous frame; a cut (teleport, resize, long
        # stall) starts a fresh history so the whole frame does not smear across it.
        cut = (self._vp_time is None or time.time() - self._vp_time > 0.25
               or self._vp_eye is None
               or float(np.linalg.norm(np.asarray(cam.position, np.float64) - self._vp_eye)) > 40.0)
        self._prev_vp = None if cut else self._cur_vp
        self._prev_car = None if cut else self._cur_car
        self._last_vp = np.asarray(proj, np.float64) @ np.asarray(view, np.float64)
        self._cloud = float(getattr(weather, "cloud", 0.0)) if weather is not None else 0.0
        self._storm = float(getattr(weather, "storm", 0.0)) if weather is not None else 0.0
        self._rain = float(getattr(weather, "rain", 0.0)) if weather is not None else 0.0
        eye = cam.position
        self._eye = np.asarray(eye, np.float64)
        eye = self._eye
        self._dist_cache = {}
        self.last_draw_calls = 0
        self.last_water_tris = 0
        self.last_tree_lod_counts = [0, 0, 0, 0]
        self.last_tree_shadow_lod_counts = [0, 0, 0, 0]
        self.last_tree_tris = self.last_tree_shadow_tris = 0
        self._tree_pixel_scale = self.height * abs(float(proj[1, 1])) * .5
        # Traffic/signals can arrive before any mesh is resident. They still
        # need the same frustum culling and directional lamp field as the city.
        planes = frustum_planes(cull_proj.astype(np.float64) @ view.astype(np.float64))
        
        # Representative urban visibility: 25 km clear, down to 1.2 km in heavy weather.
        haze = float(np.clip(max(getattr(weather, "fog", 0.0), self.haze_floor), 0.0, 1.0))
        self.air_extinction = 3.912 / (25000.0 * (1200.0 / 25000.0) ** haze)
        # The fog now ends just past the loaded tiles: the far edge of the streamed
        # city dissolves into haze instead of drawing detail nobody can resolve,
        # which keeps the chunk boundary hidden without a visible wall of fog.
        fog_near = max(900.0, (cam.distance + self.stream_radius) * 0.50)
        fog_far = max(1500.0, (cam.distance + self.stream_radius) * 0.80)
        night = float(sky.night)
        # Fog-aware LOD scaling: in heavy fog, reduce detail distances further
        # since objects are obscured anyway. 0.5x at max haze, 1.0x at clear.
        self._fog_lod_scale = 1.0 - 0.5 * haze

        # Extract near and far from projection matrix for linearization
        near = 2.0 * proj[2, 3] / (2.0 * proj[2, 2] - 2.0)
        far = 2.0 * proj[2, 3] / (2.0 * proj[2, 2] + 2.0)
        depth_q = 1.0 / (max(near, 0.05) * 16777216.0)

        self._render_sky(cam, sky, view, proj, weather)
        if self._reverse_active:
            proj = cam.projection(self.width / self.height, reverse_z=True)
            # The float reverse buffer needs no distance-squared world offset.
            depth_q = 0.0
        # The depth buffer and the post pass must share the exact projection used below.
        self._cur_vp = np.asarray(proj, np.float64) @ np.asarray(view, np.float64)
        self._cur_proj_z = (float(proj[2][2]), float(proj[2][3]))
        self._vp_eye = np.asarray(eye, np.float64).copy()
        self._vp_time = time.time()
        self._cur_car = self.motion_car

        chunks = list(chunks)
        if chunks:
            sph = np.array([c.sphere for c in chunks], np.float64)
            inside = (sph[:, :3] @ planes[:, :3].T + planes[:, 3] >= -sph[:, 3:4]).all(axis=1)
            vis = [c for c, ok in zip(chunks, inside) if ok]
        else:
            vis = []
        self.last_drawn = len(vis)
        self.last_tris = sum(c.b_count for c in vis) // 3
        now = time.time()
        light = vehicle_light = near_light = None
        self.last_vehicle_shadow_draws = self.last_vehicle_shadow_tris = self.last_shadow_casters = 0
        self.last_near_shadow_draws = self.last_near_shadow_casters = 0
        if self.shadow is not None and sky.shadows:
            anchor = cam.position if cam.mode == cam.FLY else cam.target
            light = light_matrix(np.asarray(anchor, np.float64),
                                 self._held_light_dir(sky.light_dir),
                                  self.shadow_extent(cam), self.shadow.size)
            if self.near_shadow is not None and self.near_shadows_enabled and not wireframe:
                near_light = light_matrix(np.asarray(anchor, np.float64), self._shadow_dir,
                                          self.near_shadow_extent(cam), self.near_shadow.size,
                                          depth_extent=max(1024., light.extent))
            if self.vehicle_shadow is not None and self.vehicle_shadows_enabled and not wireframe:
                vehicle_light = light_matrix(np.asarray(anchor, np.float64), self._shadow_dir,
                                             self.vehicle_shadow_extent(cam), self.vehicle_shadow.size)
            spheres = sph if chunks else np.empty((0, 4), np.float64)
            self._update_city_shadow(chunks, spheres, light, eye, now)
            if near_light is not None:
                self._update_city_shadow(chunks, spheres, near_light, eye, now, near=True)
        else:
            self.last_shadow_draws = 0
        self._near_light = near_light
        self._vehicle_light = vehicle_light
        vehicle_key = self._prepare_vehicles(chunks, traffic, proj, planes, vehicle_light)
        ped_key = None
        if self.peds is not None:
            ped_key = self.peds.prepare(pedestrians, eye, planes, vehicle_light)
        # People and cars share one dynamic shadow map; either one changing redraws it.
        shadow_key = None if (vehicle_key is None and ped_key is None) else (vehicle_key, ped_key)
        self._vehicle_shadow_active = shadow_key is not None
        if shadow_key is not None and shadow_key != self._vehicle_shadow_key:
            self._vehicle_shadow_pass(vehicle_light)
        self._vehicle_shadow_key = shadow_key
        if chunks:
            self._prepare_cell_visibility(chunks, planes)
        if self.shadow is not None:
            self.shadow.bind_texture(1)
        if self.vehicle_shadow is not None:
            self.vehicle_shadow.bind_texture(self.VEHICLE_SHADOW_UNIT)
        if self.near_shadow is not None:
            self.near_shadow.bind_texture(self.NEAR_SHADOW_UNIT)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self._update_weather_roof(chunks, cam, weather, now, wireframe)
        if chunks:
            self._prepare_cell_visibility(chunks, planes)
        if use_post:
            # ShadowMap.end() binds the default framebuffer.  The scene was
            # already bound before the shadow pass, so restore it before
            # drawing the city; otherwise shadows-on frames composite only
            # the pre-shadow sky.
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ms_fbo or self.scene_fbo)
            gl.glViewport(0, 0, self.width, self.height)
            self._mrt_n = 0; self._set_mrt(2)       # opaque passes write colour + material

        def common(p):
            self._mat(p, "uView", view); self._mat(p, "uProj", proj)
            self._vec3(p, "uCamPos", eye); self._vec3(p, "uFogColor", sky.fog)
            self._float(p, "uFogNear", fog_near); self._float(p, "uFogFar", fog_far)
            self._float(p, "uAirExtinction", self.air_extinction)
            self._float(p, "uAirCameraHeight", float(eye[2]))
            self._vec3(p, "uSunDir", sky.light_dir); self._vec3(p, "uSunColor", sky.light)
            self._float(p, "uSunIntensity", sky.sun_intensity)
            self._vec3(p, "uAmbColor", sky.amb); self._float(p, "uNightFactor", night)
            self._float(p, "uHours", sky.hours)
            self._float(p, "uTime", self._tshader())
            self._float(p, "uExposure", sky.exposure)
            self._float(p, "uHdrOut", self._hdr_out)
            self._float(p, "uDepthQ", depth_q)
            self._float(p, "uReverseZ", float(self._reverse_active))
            self._int(p, "uSkyLUT", self.SKY_LUT_UNIT)
            self._float(p, "uSkyLutOn", 1.0 if getattr(self, "sky_lut_ok", False) else 0.0)
            self._vec3(p, "uSkySunDir", sky.sun_dir)
            self._atmosphere_uniforms(p, sky)
            self._celestial_uniforms(p, sky)
            self._weather_uniforms(p, weather)

        self._select_lamps(chunks, cam, night)

        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glDisable(gl.GL_CULL_FACE)

        gl.glDisable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_FALSE)

        gl.glUseProgram(self.prog_ground)
        common(self.prog_ground)
        self._bind_ground_texture(self.prog_ground)
        self._float(self.prog_ground, 'uSurfDetail', self.surface_detail)
        self._shadow_uniforms(self.prog_ground, light, sky)
        gl.glUniform2f(self._loc(self.prog_ground, "uCenter"),
                       float(cam.target[0]), float(cam.target[1]))
        self._float(self.prog_ground, "uSize", (cam.distance + self.stream_radius) * 2.5)
        # This quad is only the far backdrop behind the streamed tiles.  Fade it
        # into the haze early, otherwise its green plane shows as a wall above
        # the sea at the edge of the loaded area.
        self._float(self.prog_ground, "uFogNear", 0.0)
        self._float(self.prog_ground, "uFogFar", max(500.0, self.stream_radius * 0.8))
        gl.glBindVertexArray(self.vao_ground)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # 1. Terrain (elevation, parks, roads) - depth ON so it occludes water
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_GEQUAL if self._reverse_active else gl.GL_LEQUAL)
        p = self.prog_terrain
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        self._bind_ground_texture(p)
        self._lamp_uniforms(p); self._neon_uniforms(p, vis, cam)
        self._float(p, "uTime", self._tshader())
        self._float(p, "uGroundPass", 1.0)
        self._float(p, "uSurfDetail", self.surface_detail)
        for c in vis:
            if c.g_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
            self._float(p, "uGrow", 1.0)
            gl.glBindVertexArray(c.vao_g)
            self._draw_tris(c, "g", planes)

        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)
        gl.glDepthFunc(gl.GL_GEQUAL if self._reverse_active else gl.GL_LEQUAL)

        # 2. Buildings
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE if wireframe else gl.GL_FILL)
        if not wireframe:
            gl.glEnable(gl.GL_CULL_FACE)
            # Per-shell depth ranks resolve coincident faces. A slope-based
            # polygon offset here displaced entire grazing facades through roofs.
        p = self.prog_building
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        self._lamp_uniforms(p); self._neon_uniforms(p, vis, cam)
        for c in vis:
            if c.b_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
            self._float(p, "uGrow", self._grow(c, now))
            # Apply fog-aware scaling to LOD distance
            lod_dist = self.building_lod_distance * getattr(self, '_fog_lod_scale', 1.0)
            if c.vao_b_lod and c.cell_b is not None and c.cell_b_lod is not None:
                gl.glBindVertexArray(c.vao_b)
                self._draw_tris(c, "b", planes, distance=lod_dist)
                gl.glBindVertexArray(c.vao_b_lod)
                self._draw_tris(c, "b_lod", planes, minimum=lod_dist)
            else:
                distant = c.vao_b_lod and self._chunk_dist(c, eye) > lod_dist
                gl.glBindVertexArray(c.vao_b_lod if distant else c.vao_b)
                self._draw_tris(c, "b_lod" if distant else "b", planes)
        gl.glDisable(gl.GL_POLYGON_OFFSET_FILL)

        # 2b. Street furniture: signal gantries, signs and broad road repairs
        if not wireframe:
            gl.glDisable(gl.GL_CULL_FACE)
            props_dist = self.lod["props"] * getattr(self, '_fog_lod_scale', 1.0)
            for c in vis:
                if c.p_count == 0 or self._chunk_dist(c, eye) > props_dist: continue
                self._vec3(p, "uOffset", c.offset)
                self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
                self._float(p, "uGrow", 1.0)
                gl.glBindVertexArray(c.vao_prop)
                self._draw_tris(c, "prop", planes, props_dist)

        # 3. Trees
        if not wireframe and any(c.t_count > 0 for c in vis):
            gl.glDisable(gl.GL_CULL_FACE)
            p = self.prog_tree
            gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
            self._bind_leaf_texture(p)
            self._float(p, 'uSurfDetail', self.surface_detail)
            self._draw_foliage_poles(chunks, planes, p)

        # ==========================================================
        # DEPTH BUFFER COPY (Required for Water Soft Edge / Absorption)
        # ==========================================================
        # Skipped when no visible chunk has water (most inland views), and copied
        # into the existing texture instead of reallocating it every frame.
        has_water = any(c.w_count > 0 for c in vis)
        if has_water:
            # glCopyTexSubImage2D cannot read a multisampled framebuffer, so resolve
            # the scene so far into the single-sample target, take both copies, and
            # then resume drawing into the multisampled target for the water pass.
            # Only frames actually rendered into the multisampled target resolve;
            # a direct-to-window frame has nothing in it to copy.
            ms = bool(use_post and self.ms_fbo)
            mrt = self._mrt_n
            if ms:
                self._resolve_scene()
            # Diagnostic/material readbacks can leave MRT attachment 1 selected.
            # Refraction must always copy opaque scene colour, never its normals.
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0 if use_post else gl.GL_BACK)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_tex)
            # Copying from the conventional default buffer must not replace the
            # float storage: the next HDR frame uses reversed depth again.
            gl.glCopyTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, 0, 0, self.width, self.height)
            if use_post:
                # Reuse the existing SSR scratch target for pre-water opaque colour.
                # Sampling the attached scene target itself would be undefined feedback.
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.ssr_tex)
                gl.glCopyTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, 0, 0, self.width, self.height)
                if ms:
                    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.ms_fbo)
                    gl.glViewport(0, 0, self.width, self.height)
                    self._mrt_n = 0; self._set_mrt(mrt or 2)

        # 4. Water
        self._set_mrt(2)
        gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)
        p = self.prog_water
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        self._float(p, "uTime", self._tshader())
        self._float(p, 'uWaterWaves', float(self.water_waves_enabled))
        self._float(p, 'uWaterSSS', float(self.water_sss_enabled))
        self._bind_water_texture(p, has_water)
        self._float(p, 'uRefractionOn', float(use_post and has_water))
        if has_water:
            self._mat(p, 'uInvViewProj', np.linalg.inv(proj.astype(np.float64) @ view.astype(np.float64)))
        gl.glActiveTexture(gl.GL_TEXTURE0 + 3)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.ssr_tex if use_post else 0)
        self._int(p, "uRefractionMap", 3)
        
        gl.glActiveTexture(gl.GL_TEXTURE0 + 2)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_tex)
        self._int(p, "uDepthMap", 2)
        
        self._vec2(p, "uResolution", (float(self.width), float(self.height)))
        self._float(p, "uNear", near)
        self._float(p, "uFar", far)

        for c in vis:
            if c.w_count == 0: continue
            self._vec3(p, "uOffset", c.offset)
            self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
            gl.glBindVertexArray(c.vao_w)
            self._draw_tris(c, 'w', planes)

        gl.glActiveTexture(gl.GL_TEXTURE0)

        # 5. Edges
        if show_edges and not wireframe:
            self._set_mrt(1)
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            p = self.prog_line
            gl.glUseProgram(p); common(p)
            edges_dist = self.lod["edges"] * getattr(self, '_fog_lod_scale', 1.0)
            for c in vis:
                if c.e_count == 0 or self._chunk_dist(c, eye) > edges_dist: continue
                self._ensure_edges(c)
                if not c.vao_e: continue
                self._vec3(p, "uOffset", c.offset)
                self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
                self._float(p, "uGrow", self._grow(c, now))
                gl.glBindVertexArray(c.vao_e)
                gl.glDrawArrays(gl.GL_LINES, 0, c.e_count)

        # 6. Streetlights (Poles and Glow Sprites)
        if not wireframe and any(c.l_count > 0 for c in vis):
            self._set_mrt(2)
            gl.glEnable(gl.GL_DEPTH_TEST); gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_CULL_FACE)
            p = self.prog_streetpole
            gl.glUseProgram(p); common(p)
            self._lamp_uniforms(p)
            self._draw_foliage_poles(chunks, planes, p, trees=False)

        # 6b. Light sprites (lamps, facade emissives, vehicle lamps) out to 1.5 km.
        #     One instanced batch for the whole scene: the field culls behind and
        #     off-frustum sources on the CPU, the vertex shader repeats the test.
        if not wireframe and night > 0.05 and planes is not None:
            # All resident chunks, not just the visible ones: the per-light frustum cull
            # does the rejecting, and a stable chunk set means the record arrays are
            # rebuilt only when tiles stream, not whenever the camera turns.
            self.lights.sync_chunks(chunks)
            self.lights.cull(view, planes, eye, night, traffic)
            if self.lights.count:
                self._set_mrt(1)
                gl.glEnable(gl.GL_DEPTH_TEST)
                gl.glDepthMask(gl.GL_FALSE)
                gl.glEnable(gl.GL_BLEND)
                gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE, gl.GL_ZERO, gl.GL_ONE)
                gl.glDisable(gl.GL_CULL_FACE)
                self.lights.air_extinction = self.air_extinction
                self.lights.camera_height = float(eye[2])
                self.lights.draw(view, proj, night, fog_near, fog_far)
                gl.glDisable(gl.GL_BLEND)
                gl.glDepthMask(gl.GL_TRUE)

        # 6b. Tyre marks (decals under the cars, before them so cars cover their own marks)
        if not wireframe:
            self._draw_skid_marks(common, getattr(weather, 'wetness', 0.0) if weather is not None else 0.0)

        # 6c. Dark contact shadow under the cars (also before them, so the bodies cover it)
        if not wireframe:
            self._draw_contact_shadows(common, light, sky)

        # 7. Vehicles
        self._set_mrt(2)
        p = self.prog_vehicle
        gl.glUseProgram(p); common(p); self._shadow_uniforms(p, light, sky)
        self._lamp_uniforms(p)
        gl.glUniform3f(self._loc(p, "uHeadlightColor"),  *_HEADLIGHT_COLOR)
        gl.glUniform3f(self._loc(p, "uTaillightColor"),  *_TAILLIGHT_COLOR)
        gl.glUniform3f(self._loc(p, "uGlowColor"),       *_GROUND_GLOW_COLOR)
        gl.glUniform3f(self._loc(p, "uLedColor"),        *_LED_COLOR)      # LED signature / DRL (material 11)
        self._float(p, 'uWheelAO', self.wheel_ao)
        self._float(p, 'uTrafficClock', getattr(traffic, 'clock', self._tshader()))
        gl.glUniform4f(self._loc(p, "uPaintCoat"), self.paint_gloss, self.paint_orange_peel,
                       self.paint_flakes, self.paint_pearl)
        gl.glDisable(gl.GL_CULL_FACE)
        
        self._vec3(p, "uOffset", (0, 0, 0))
        self._vec2(p, "uScale", (1, 1))
        for k, count in enumerate(self.moving_counts):
            if not count: continue
            gl.glBindVertexArray(self.moving_vaos[k])
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.vehicle_opaque_counts[k], gl.GL_UNSIGNED_INT, None, count)
            self.last_draw_calls += 1
        # Lamp lenses/cabin windows: pass 0 transmits the actual opaque scene/cabin;
        # pass 1 adds Fresnel sky/sun reflection. Depth-tested, no depth writes.
        if any(self.vehicle_glass_counts[k] and self.moving_counts[k] for k in range(len(self.moving_counts))):
            self._set_mrt(1)
            gl.glEnable(gl.GL_BLEND)
            gl.glDepthMask(gl.GL_FALSE)
            lens_loc = self._loc(p, "uLensPass")
            for lens_pass, blend in ((0, (gl.GL_ZERO, gl.GL_SRC_COLOR)), (1, (gl.GL_ONE, gl.GL_ONE))):
                gl.glBlendFunc(*blend)
                gl.glUniform1i(lens_loc, lens_pass)
                for k, count in enumerate(self.moving_counts):
                    if not count or not self.vehicle_glass_counts[k]: continue
                    gl.glBindVertexArray(self.moving_vaos[k])
                    gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.vehicle_glass_counts[k], gl.GL_UNSIGNED_INT,
                                               ctypes.c_void_p(self.vehicle_opaque_counts[k]*4), count)
                    self.last_draw_calls += 1
            gl.glUniform1i(lens_loc, 0)
            gl.glDepthMask(gl.GL_TRUE); gl.glDisable(gl.GL_BLEND)
            self._set_mrt(2)
        # Light pools need just one blend/MRT transition for all near cars.
        if night > .05:
            self._set_mrt(1)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE, gl.GL_ZERO, gl.GL_ONE)
            gl.glDepthMask(gl.GL_FALSE)
            for k in range(self.vehicle_kinds):
                if not self.moving_counts[k]: continue
                gl.glBindVertexArray(self.moving_vaos[k])
                gl.glDrawElementsInstanced(gl.GL_TRIANGLES, 12, gl.GL_UNSIGNED_INT,
                                          ctypes.c_void_p((self.vehicle_opaque_counts[k] + self.vehicle_glass_counts[k])*4), self.moving_counts[k])
                self.last_draw_calls += 1
            gl.glDepthMask(gl.GL_TRUE); gl.glDisable(gl.GL_BLEND)

        # 7c. Pedestrians: instanced, shader-animated, lit like the cars
        if self.peds is not None and self.peds.count and not wireframe:
            self._set_mrt(2)
            self.peds.draw(common, light, sky)

        signals = getattr(traffic, 'signals', None)
        if signals is not None and len(signals):
            centers = (signals[:, 12:15] + signals[:, :3] * SIGNAL_CENTER[0]
                       + signals[:, 4:7] * SIGNAL_CENTER[1] + signals[:, 8:11] * SIGNAL_CENTER[2])
            props_dist = self.lod["props"] * getattr(self, '_fog_lod_scale', 1.0)
            visible = sphere_mask(centers, SIGNAL_RADIUS, eye, props_dist, planes) & (signals[:, 23] > 0.)
            if visible.any():
                self._set_mrt(2)
                rows = np.ascontiguousarray(signals[visible])
                k = self.signal_kind
                self._stream_instances(self.moving_inst_vbos[k], rows)
                gl.glBindVertexArray(self.moving_vaos[k])
                gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.vehicle_opaque_counts[k], gl.GL_UNSIGNED_INT, None, len(rows))
                self.last_draw_calls += 1

        # 8. Neon signage (additive, depth-tested)
        self._set_mrt(1)
        if not wireframe and any(c.s_count > 0 for c in vis):
            p = self.prog_neon
            gl.glUseProgram(p); common(p)
            self._bind_billboard_texture(p)
            self._float(p, "uTime", self._tshader())
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glEnable(gl.GL_BLEND)
            gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE, gl.GL_ZERO, gl.GL_ONE)
            gl.glDisable(gl.GL_CULL_FACE)
            neon_dist = self.lod["neon"] * getattr(self, '_fog_lod_scale', 1.0)
            for c in vis:
                if c.s_count == 0 or self._chunk_dist(c, eye) > neon_dist: continue
                self._vec3(p, "uOffset", c.offset)
                self._vec2(p, "uScale", (float(c.scale[0]), float(c.scale[1])))
                self._float(p, "uGrow", self._grow(c, now))
                gl.glBindVertexArray(c.vao_s)
                self._draw_tris(c, "s", planes, neon_dist)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_BLEND)

        # 9. World-space rain/snow and roof-supported splashes, never a screen overlay.
        self.precip.last_count = self.precip.last_splash_count = 0
        self.precip.last_rain_count = self.precip.last_snow_count = 0
        precip_ready = not wireframe and weather is not None and self.precip_enabled
        # Reflections ray-march the scene colour, so streaks drawn into it returned as
        # bright smears across wet ground.  Once reflections run, precipitation goes
        # into their finished target instead, where the road can never reflect it.
        reflect = use_post and ((self.ssr_enabled and self.ssr_strength > 0.0)
                                or (self.ao_enabled and self.ao_ok))
        # 7e. Pane surface for reflections: cabin windows and lamp covers are drawn by the
        # two-pass transmission path, which writes colour only, so the SSR march had no
        # finish weight (and no depth) on them and skipped every car window.  This pass
        # adds the pane's depth and material without touching colour, and runs after
        # pedestrians, signals and precipitation so see-through glazing still shows them.
        if (self._mrt_on and use_post and not wireframe and self.ssr_enabled
                and self.ssr_strength > 0.0
                and any(self.vehicle_glass_counts[k] and self.moving_counts[k]
                        for k in range(len(self.moving_counts)))):
            self._set_mrt_mat()
            gp = self.prog_vehicle_glass
            gl.glUseProgram(gp)
            common(gp)
            self._vec3(gp, "uOffset", (0, 0, 0))
            self._vec2(gp, "uScale", (1, 1))
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_CULL_FACE)
            for k, count in enumerate(self.moving_counts):
                if not count or not self.vehicle_glass_counts[k]: continue
                gl.glBindVertexArray(self.moving_vaos[k])
                gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.vehicle_glass_counts[k], gl.GL_UNSIGNED_INT,
                                           ctypes.c_void_p(self.vehicle_opaque_counts[k]*4), count)
                self.last_draw_calls += 1
            self._set_mrt(2)

        if precip_ready and not reflect:
            self._draw_precip(cam, weather, sky, fog_near, fog_far, view, proj, eye)

        gl.glBindVertexArray(0)

        if use_post:
            self._resolve_scene()
            src = self.scene_tex
            if self.ao_enabled and self.ao_ok:
                self._ao_pass(view, proj)
            if reflect:
                self._ssr_pass(view, proj, near)
                src = self.ssr_tex
                if precip_ready:
                    self._draw_precip_after_reflections(cam, weather, sky, fog_near, fog_far,
                                                        view, proj, eye)
            if self._hdr_out > 0.5 and self.ae_enabled and self.ae_ok:
                self._auto_exposure(src)
            # Volumetric haze runs after metering (the eye adapts to the scene, not to the mist
            # in front of it) and before bloom, so fog-lit neon glow still blooms.
            src = self._haze_pass(src, cam, sky, weather, eye, night)
            self._composite_bloom(src)
        else:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        if self.shadow is not None:
            self.shadow.release_texture(1)
        if self.vehicle_shadow is not None:
            self.vehicle_shadow.release_texture(self.VEHICLE_SHADOW_UNIT)
        if self.near_shadow is not None:
            self.near_shadow.release_texture(self.NEAR_SHADOW_UNIT)
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.GROUND_TEXTURE_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.WATER_RIPPLE_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.BILLBOARD_UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        self._depth_state(False)   # map, UI and default framebuffer use standard depth
