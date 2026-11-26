"""Weather transitions and GPU-instanced precipitation.

The system owns the slowly changing weather scalars.  Shaders only read the
uniforms supplied by the renderer, which keeps cloud cover, wetness, fog and
precipitation from drifting out of phase with one another.
"""
from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.photometry import NITS_PER_SCENE_UNIT
from ..core.atmosphere import horizon_display
from ..shaders.weather import HEADLIGHT_BEAMS


def _smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / max(b - a, 1e-9)))
    return t * t * (3.0 - 2.0 * t)


@dataclass
class WeatherState:
    label: str = "Clear"
    cloud: float = 0.16
    rain: float = 0.0
    snow: float = 0.0
    fog: float = 0.05
    wetness: float = 0.0
    storm: float = 0.0
    time: float = 0.0
    wind: np.ndarray = None
    wind_offset: np.ndarray = None
    lightning: float = 0.0

    def __post_init__(self):
        if self.wind is None:
            self.wind = np.zeros(2, np.float32)
        else:
            self.wind = np.asarray(self.wind, np.float32).copy()
        if self.wind_offset is not None:
            self.wind_offset = np.asarray(self.wind_offset, np.float32).copy()


def wind_displacement(weather):
    """Manual snapshots can omit integrated advection; running weather never does."""
    offset = getattr(weather, 'wind_offset', None)
    return np.asarray(offset if offset is not None else weather.wind * weather.time, np.float32)


class WeatherSystem:
    """Coherent cloud/precipitation fronts, integrated gusts and lingering wetness."""

    MODES = ("clear", "overcast", "rain", "storm", "snow")
    LABELS = {"clear": "Clear", "overcast": "Overcast", "rain": "Rain",
              "storm": "Thunderstorm", "snow": "Snow"}
    PRESETS = {
        # cloud, rain, snow, storm, wind-x, wind-y
        "clear":    (0.16, 0.00, 0.00, 0.00, 0.8, 0.25),
        "overcast": (0.78, 0.00, 0.00, 0.05, 2.1, 0.65),
        "rain":     (0.92, 0.85, 0.00, 0.38, 3.8, 1.10),
        "storm":    (1.00, 1.00, 0.00, 1.00, 6.2, 2.10),
        "snow":     (0.86, 0.00, 1.00, 0.12, 1.5, 0.55),
    }

    def __init__(self, start="clear"):
        self.mode = start if start in self.PRESETS else "clear"
        self.cloud, self.rain, self.snow, self.storm, wx, wy = self.PRESETS[self.mode]
        self.live = False            # True while following real reported conditions
        self.live_fog = None         # fog target from the report, when live
        self.custom_label = None     # "Partly cloudy · 24 °C" while live
        self.live_state = None
        self.wetness = .55 if self.rain > .02 else 0.0
        self.fog = self._fog_target()
        self.time = 0.0
        self.wind = np.array([wx, wy], np.float32)
        self.wind_offset = np.zeros(2, np.float64)
        self.lightning = 0.0
        self._rng = np.random.default_rng(0x6A1A)
        self._next_lightning = float(self._rng.uniform(9., 19.))
        self._flash_age = math.inf
        self._preset = self.PRESETS[self.mode]
        self.state = self._snapshot()

    @property
    def label(self):
        return self.custom_label or self.LABELS[self.mode]

    @property
    def temperature(self):
        state = self.live_state or {}
        return state.get("temp_c")

    def set_mode(self, mode):
        if mode not in self.PRESETS:
            return
        if mode == "storm" and self.mode != mode:
            self._next_lightning = self.time + float(self._rng.uniform(9., 19.))
        # A manual choice always leaves live mode, otherwise the next refresh
        # would silently overwrite what the user just picked.
        self.live, self.live_fog, self.custom_label, self.live_state = False, None, None, None
        self.mode = mode
        self._preset = self.PRESETS[mode]

    def apply_live(self, state):
        """Follow a real observation (see ``world.live_weather``).

        Only the targets change; the existing transition rates blend into them, so
        switching cities or refreshing the report never snaps the sky.
        """
        if not state:
            return
        kind = state.get("kind", "cloudy")
        self.mode = ("storm" if kind == "storm" else
                     "snow" if kind == "snow" else
                     "rain" if kind in ("rain", "drizzle") else
                     "overcast" if kind in ("overcast", "cloudy", "fog") else "clear")
        wx, wy = state.get("wind", (0.0, 0.0))
        self._preset = (float(state.get("cloud", 0.3)), float(state.get("rain", 0.0)),
                        float(state.get("snow", 0.0)), float(state.get("storm", 0.0)),
                        float(wx), float(wy))
        self.live_fog = float(state.get("fog", 0.05))
        temp = state.get("temp_c")
        text = state.get("text") or self.LABELS[self.mode]
        self.custom_label = f"{text}  \u00b7  {round(temp):d} \u00b0C" if temp is not None else text
        self.live_state = dict(state)
        self.live = True

    def clear_live(self):
        self.live, self.live_fog, self.custom_label, self.live_state = False, None, None, None

    def cycle(self):
        i = (self.MODES.index(self.mode) + 1) % len(self.MODES)
        self.set_mode(self.MODES[i])
        return self.mode

    def update(self, dt, sky=None):
        dt = float(dt)
        dt = max(0.0, min(dt, .25)) if math.isfinite(dt) else 0.0
        self.time += dt
        p = self._preset
        target_cloud, target_rain, target_snow, target_storm = p[0], p[1], p[2], p[3]

        # Clouds and wind settle over a few seconds; rain starts/stops faster
        # than the roads dry, which is what produces the wet-after-rain look.
        k = 1.0 - math.exp(-dt * 0.45)
        self.cloud += (target_cloud - self.cloud) * k
        self.storm += (target_storm - self.storm) * k
        front = _smoothstep(.38, .82, self.cloud)
        self.rain += (target_rain * front - self.rain) * (1.0 - math.exp(-dt * 1.25))
        self.snow += (target_snow * front - self.snow) * (1.0 - math.exp(-dt * .90))
        self.fog += (self._fog_target() - self.fog) * (1.0 - math.exp(-dt * .45))

        wx, wy = p[4], p[5]
        gust = (.32 * math.sin(self.time * .73) + .18 * math.sin(self.time * 1.91 + 1.3)) * (.25 + self.storm)
        old_wind = self.wind.copy()
        self.wind += (np.array([wx + gust, wy + gust * .55], np.float32) - self.wind) * k
        self.wind_offset += (old_wind + self.wind) * (.5 * dt)
        if self.rain > .015:
            self.wetness = min(1.0, self.wetness + dt * (.035 + .16 * self.rain))
        else:
            # Cloudy roads dry much more slowly than sunlit roads.
            self.wetness = max(0.0, self.wetness - dt * (.006 + .014 * (1. - self.cloud)))

        self._flash_age += dt
        if self.storm > .65 and self.mode == 'storm' and self.time >= self._next_lightning:
            self._flash_age = 0.0
            self._next_lightning = self.time + float(self._rng.uniform(10., 24.))
        age = self._flash_age
        self.lightning = ((.80 * math.exp(-age / .065) + .32 * math.exp(-((age - .16) / .045) ** 2))
                          * _smoothstep(.55, .85, self.storm) if age < .5 else 0.0)
        self.state = self._snapshot()
        return self.state

    def _fog_target(self):
        if self.live and self.live_fog is not None:
            return min(1., float(self.live_fog))
        return min(1., .04 + .22 * self.cloud + .44 * self.rain + .24 * self.storm + .22 * self.snow)

    def _snapshot(self):
        return WeatherState(
            label=self.label, cloud=float(np.clip(self.cloud, 0.0, 1.0)),
            rain=float(np.clip(self.rain, 0.0, 1.0)),
            snow=float(np.clip(self.snow, 0.0, 1.0)),
            fog=float(np.clip(self.fog, 0.0, 1.0)),
            wetness=float(np.clip(self.wetness, 0.0, 1.0)),
            storm=float(np.clip(self.storm, 0.0, 1.0)),
            time=self.time, wind=self.wind, wind_offset=self.wind_offset, lightning=self.lightning,
        )

    # Display-referred colour of the cloud deck seen from the street, at local noon, per weather
    # type.  Reference: real overcast cities are bright, neutral and only slightly cool; rain
    # darkens and cools the deck; a thunderstorm deck is dark slate with a blue-violet cast;
    # snow clouds are the brightest and the most neutral.
    DECK_OVERCAST = np.array([0.600, 0.628, 0.664], np.float32)
    DECK_RAIN = np.array([0.440, 0.476, 0.528], np.float32)
    DECK_STORM = np.array([0.235, 0.255, 0.330], np.float32)
    DECK_SNOW = np.array([0.720, 0.748, 0.792], np.float32)
    # Cloud base lit from below by the city: warm, neutral-grey sodium / LED sky-glow.
    DECK_NIGHT = np.array([0.092, 0.080, 0.071], np.float32)

    def deck_colour(self, sky, clear_fog):
        """Colour of the overcast deck and the fog beneath it, for the current sun and weather."""
        luma = np.array([0.2126, 0.7152, 0.0722], np.float32)
        storm = float(np.clip(self.storm, 0., 1.))
        rain = float(np.clip(self.rain, 0., 1.)) * (1. - storm)
        snow = float(np.clip(self.snow, 0., 1.)) * (1. - storm)
        plain = max(0., 1. - storm - rain - snow)
        day_col = (self.DECK_OVERCAST * plain + self.DECK_RAIN * rain
                   + self.DECK_STORM * storm + self.DECK_SNOW * snow)
        # The deck is dimmer when the sun is low, and picks up a trace of its colour at the horizon.
        alt = math.degrees(float(sky.alt))
        elevation = _smoothstep(-4., 38., alt)
        day_col = day_col * (0.42 + 0.58 * elevation)
        low_sun = (1. - _smoothstep(1., 20., alt)) * _smoothstep(-6., 2., alt)
        tint = np.asarray(clear_fog, np.float32) / max(float(np.asarray(clear_fog, np.float32) @ luma), 1e-4)
        warm = (0.42 * low_sun) * (1. - 0.65 * storm)
        day_col = day_col * (1. - warm + warm * tint)
        night = float(sky.night)
        glow = 0.55 + 0.45 * float(np.clip(sky.light_pollution / .35, 0., 1.))
        night_col = self.DECK_NIGHT * glow * (1. - 0.30 * storm)
        return (day_col * (1. - night) + night_col * night).astype(np.float32)

    def effective_sky(self, sky):
        """Shade surface lighting; leave sky gaps / Moon for directional cloud extinction."""
        cloud = float(np.clip(self.cloud, 0.0, 1.0))
        storm = float(np.clip(self.storm, 0.0, 1.0))
        night = float(sky.night)
        aerosol = float(np.clip(sky.aerosol_density * (1. + 1.8 * self.fog + .7 * storm), .1, 8.))
        exposure = float(sky.exposure * (1.0 + 0.06 * cloud))
        clear_fog = horizon_display(sky.sun_dir, sky.atmosphere_intensity, exposure,
                                   height=sky.observer_height, aerosol=aerosol,
                                   night=night, glow=sky.sky_glow)

        # Fog tint is display-referred; ambient illumination stays in scene light.
        deck = self.deck_colour(sky, clear_fog)
        # Cloud cover below ~15% leaves the clear horizon alone; a full deck replaces it entirely
        # (the old blend always kept a yellow strip of clear-sky horizon under a storm).
        deck_fog = 0.94 * _smoothstep(.12, .85, cloud)
        deck_sky = 0.97 * _smoothstep(.08, .78, cloud)
        fog = clear_fog * (1.0 - deck_fog) + deck * deck_fog
        sky_col = clear_fog * (1.0 - deck_sky) + deck * deck_sky
        sky_luma = float(sky.amb @ np.array([0.2126, 0.7152, 0.0722]))
        amb = sky.amb * (1.0 - 0.55 * cloud) + np.array([0.97, 0.99, 1.0]) * sky_luma * (0.80 * cloud)
        amb = amb * (1.0 - 0.30 * storm)
        # A dense rain deck hides the solar disc and glints completely. Diffuse
        # daylight remains, instead of retaining a bright sun with disabled shadows.
        # Scattered fair-weather cumulus barely dims the sun; a thick deck removes it.
        transmission = math.exp(-3.2 * max(cloud - .14, 0.)) * (1. - _smoothstep(.55, .90, cloud))
        light = sky.light * transmission
        sun_i = sky.sun_intensity * transmission
        flash = np.array([.80, .88, 1.0]) * self.lightning * (.28 * (1. - night) + 65. / NITS_PER_SCENE_UNIT * night)
        amb += flash
        return replace(
            sky, sky=sky_col.astype(np.float32), fog=fog.astype(np.float32),
            amb=amb.astype(np.float32), light=light.astype(np.float32),
            shadows=bool(sky.shadows and transmission > .06),
            strength=float(sky.strength * transmission),
            sun_intensity=float(sun_i), aerosol_density=aerosol,
            exposure=exposure,
        )


class RainOcclusionMap:
    """Lazy, cached top-down surface depth: roofs shelter rain even with shadows off."""

    UNIT = 11

    def __init__(self, gl, size=1024, radius=128.):
        self.gl = gl
        self.size = min(int(size), int(gl.glGetIntegerv(gl.GL_MAX_TEXTURE_SIZE)))
        self.radius = float(radius)
        self.texture = self.fbo = 0
        self.complete = False
        self.matrix = np.eye(4, dtype=np.float32)
        self.inv_matrix = self.matrix.copy()
        self.bias = 0.
        self.key = None
        try:
            self.texture = gl.glGenTextures(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT32F, self.size, self.size,
                            0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
            for name, value in ((gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST),
                                (gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST),
                                (gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE),
                                (gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE),
                                (gl.GL_TEXTURE_COMPARE_MODE, gl.GL_NONE)):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, name, value)
            self.fbo = gl.glGenFramebuffers(1)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.texture, 0)
            gl.glDrawBuffer(gl.GL_NONE)
            gl.glReadBuffer(gl.GL_NONE)
            self.complete = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) == gl.GL_FRAMEBUFFER_COMPLETE
            if not self.complete:
                raise RuntimeError('incomplete rain occlusion depth target')
        except Exception:
            self.release()
            raise
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    @property
    def nbytes(self):
        return self.size * self.size * 4 if self.texture else 0

    def configure(self, eye, bottom, top):
        # Eight-metre snapping prevents turning/standing still from re-rendering depth.
        center = np.floor(np.asarray(eye[:2], np.float64) / 8.) * 8.
        bottom = math.floor(float(bottom) / 32.) * 32.
        top = max(bottom + 32., math.ceil(float(top) / 32.) * 32.)
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 0] = matrix[1, 1] = 1. / self.radius
        matrix[:2, 3] = -center / self.radius
        matrix[2, 2] = -2. / (top - bottom)
        matrix[2, 3] = (top + bottom) / (top - bottom)
        self.matrix = matrix
        self.inv_matrix = np.asarray(np.linalg.inv(matrix.astype(np.float64)), np.float32)
        self.bias = .08 / (top - bottom)

    def release(self):
        gl = self.gl
        if self.fbo:
            gl.glDeleteFramebuffers(1, [self.fbo])
            self.fbo = 0
        if self.texture:
            gl.glDeleteTextures([self.texture])
            self.texture = 0
        self.complete = False
        self.key = None


class PrecipitationRenderer:
    """Layered world-space falling drops/flakes and depth-backed secondary splashes."""

    def __init__(self, gl, count=16000):
        from OpenGL.GL import shaders
        from ..shaders.weather import PRECIP_VS, PRECIP_FS, SPLASH_VS, SPLASH_FS

        self.gl = gl
        self.count = max(1, int(count))
        self.splash_capacity = min(4500, self.count)
        self.program = shaders.compileProgram(
            shaders.compileShader(PRECIP_VS, gl.GL_VERTEX_SHADER),
            shaders.compileShader(PRECIP_FS, gl.GL_FRAGMENT_SHADER),
            validate=False)
        self.splash_program = shaders.compileProgram(
            shaders.compileShader(SPLASH_VS, gl.GL_VERTEX_SHADER),
            shaders.compileShader(SPLASH_FS, gl.GL_FRAGMENT_SHADER), validate=False)
        self._locations = {}

        rng = np.random.default_rng(0xBADC0FFE)
        seeds = rng.random((self.count, 4), dtype=np.float32)
        self.quad = np.array([[-1, -1], [1, -1], [1, 1],
                              [-1, -1], [1, 1], [-1, 1]], np.float32)
        self.vao = gl.glGenVertexArrays(1)
        self.quad_vbo = gl.glGenBuffers(1)
        self.seed_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.quad.nbytes, self.quad, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.seed_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, seeds.nbytes, seeds, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 4, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glVertexAttribDivisor(1, 1)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        self.last_count = self.last_splash_count = self.last_rain_count = self.last_snow_count = 0
        self.hdr_out = self.air_extinction = self.camera_height = 0.
        # Set by the renderer: binds the street-lamp grid for the current program.
        self.lamp_binder = None
        # (x, y, z, beam dir xyz, power) rows for the nearest vehicle headlights.
        self.headlights = np.zeros((0, 7), np.float32)
        self._beam_pos = np.zeros((HEADLIGHT_BEAMS, 4), np.float32)
        self._beam_dir = np.zeros((HEADLIGHT_BEAMS, 4), np.float32)

    @property
    def nbytes(self):
        return self.quad.nbytes + self.count * 16 if self.vao else 0

    def _loc(self, program, name):
        key = program, name
        if key not in self._locations:
            self._locations[key] = self.gl.glGetUniformLocation(program, name)
        return self._locations[key]

    def draw(self, cam, weather, sky, fog_near, fog_far, view=None, proj=None, *, roof=None, pixel_scale=None, splashes=True):
        self.last_count = self.last_splash_count = self.last_rain_count = self.last_snow_count = 0
        if weather is None:
            return
        rain, snow = float(np.clip(weather.rain, 0., 1.)), float(np.clip(weather.snow, 0., 1.))
        if max(rain, snow) < .015 or not self.vao:
            return
        gl = self.gl
        view = cam.view() if view is None else view
        proj = cam.projection(1.0) if proj is None else proj
        if pixel_scale is None:
            pixel_scale = float(gl.glGetIntegerv(gl.GL_VIEWPORT)[3]) * abs(float(proj[1, 1])) * .5
        right = np.array(view[0, :3], np.float32)
        up = np.array(view[1, :3], np.float32)
        right /= max(1e-6, float(np.linalg.norm(right)))
        up /= max(1e-6, float(np.linalg.norm(up)))
        roof_on = roof is not None and roof.complete
        gl.glActiveTexture(gl.GL_TEXTURE0 + RainOcclusionMap.UNIT)
        gl.glBindTexture(gl.GL_TEXTURE_2D, roof.texture if roof_on else 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)

        def common(program):
            gl.glUseProgram(program)
            # Precipitation is lit by the nearest street lamps; the spill grid has
            # to be bound once the program is current, like every surface pass.
            lamp_binder = getattr(self, 'lamp_binder', None)
            if lamp_binder is not None:
                lamp_binder(program)
            for name, mat in (("uView", view), ("uProj", proj)):
                gl.glUniformMatrix4fv(self._loc(program, name), 1, gl.GL_TRUE,
                                      np.ascontiguousarray(mat, np.float32))
            for name, value in (("uCamPos", cam.position), ("uCamRight", right), ("uCamUp", up),
                                ("uFogColor", sky.fog), ("uSunColor", sky.light), ("uAmbColor", sky.amb)):
                gl.glUniform3f(self._loc(program, name), *map(float, value))
            for name, value in (("uTime", weather.time), ("uExposure", sky.exposure), ("uHdrOut", self.hdr_out),
                                ("uFogNear", fog_near), ("uFogFar", fog_far), ("uPixelScale", pixel_scale),
                                ("uAirExtinction", self.air_extinction), ("uAirCameraHeight", self.camera_height),
                                ("uRainRoofOn", float(roof_on)),
                                ("uNightFactor", float(getattr(sky, 'night', 0.)))):
                gl.glUniform1f(self._loc(program, name), float(value))
            gl.glUniform1i(self._loc(program, 'uRainRoof'), RainOcclusionMap.UNIT)
            beams = min(len(getattr(self, 'headlights', ())), HEADLIGHT_BEAMS)
            if beams:
                self._beam_pos[:beams, :3] = self.headlights[:beams, :3]
                self._beam_pos[:beams, 3] = 1.
                self._beam_dir[:beams, :3] = self.headlights[:beams, 3:6]
                self._beam_dir[:beams, 3] = self.headlights[:beams, 6]
                gl.glUniform4fv(self._loc(program, 'uBeamPos[0]'), HEADLIGHT_BEAMS, self._beam_pos)
                gl.glUniform4fv(self._loc(program, 'uBeamDir[0]'), HEADLIGHT_BEAMS, self._beam_dir)
            gl.glUniform1i(self._loc(program, 'uBeamCount'), beams)
            if roof_on:
                for name, mat in (('uRainViewProj', roof.matrix), ('uRainInvViewProj', roof.inv_matrix)):
                    gl.glUniformMatrix4fv(self._loc(program, name), 1, gl.GL_TRUE, mat)
                gl.glUniform1f(self._loc(program, 'uRainRoofBias'), roof.bias)
                gl.glUniform2f(self._loc(program, 'uRainRoofTexel'), 1. / roof.size, 1. / roof.size)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA, gl.GL_ZERO, gl.GL_ONE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glBindVertexArray(self.vao)
        common(self.program)
        gl.glUniform2f(self._loc(self.program, 'uWind'), *map(float, weather.wind))
        gl.glUniform2f(self._loc(self.program, 'uWindOffset'), *map(float, wind_displacement(weather)))
        for snow_mix, amount in ((0., rain), (1., snow)):
            if amount < .015:
                continue
            count = max(1, int(self.count * amount * (.55 if snow_mix else 1.)))
            gl.glUniform1f(self._loc(self.program, 'uSnowMix'), snow_mix)
            gl.glUniform1f(self._loc(self.program, 'uAmount'), amount)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, count)
            self.last_count += count
            if snow_mix:
                self.last_snow_count = count
            else:
                self.last_rain_count = count
        if roof_on and rain >= .015 and splashes:
            common(self.splash_program)
            self.last_splash_count = max(1, int(self.splash_capacity * rain))
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, 6, self.last_splash_count)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)

    def release(self):
        gl = self.gl
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao]); self.vao = 0
        for name in ('quad_vbo', 'seed_vbo'):
            buf = getattr(self, name)
            if buf:
                gl.glDeleteBuffers(1, [buf])
                setattr(self, name, 0)
        for name in ('program', 'splash_program'):
            program = getattr(self, name)
            if program:
                gl.glDeleteProgram(program)
                setattr(self, name, 0)
        self._locations.clear()
        self.last_count = self.last_splash_count = self.last_rain_count = self.last_snow_count = 0
