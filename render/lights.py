"""Long-range light sprites: CPU view-space culling, one instanced draw call.

Every resident chunk contributes street-lamp positions and facade emissive
samples; the traffic simulator contributes head/tail lights.  Each frame the
whole set is transformed into view space, the behind-camera and off-frustum
sources are dropped, and only the survivors are uploaded as a compact instance
array (two vec4 per light) for a single ``glDrawElementsInstanced``.

The vertex shader repeats the rejection per instance, so a light that slipped
through (or whose sprite has grown past the screen edge) costs no raster work.

Backends
    GL 4.3+   shader-storage buffer, indexed by ``gl_InstanceID``
    GL 3.3    instanced vertex attributes with a divisor of one
"""
from __future__ import annotations

import ctypes

import numpy as np

from ..core.config import NEON_EXPOSURE_STOPS
from ..core.photometry import LUMINANCE_NITS
from ..shaders.lightfield import LIGHT_RECORD_STRIDE, lightfield_shaders
from ..shaders.weather import HEADLIGHT_BEAMS, HEADLIGHT_REACH
from ..vehicles.vehicle_models import SUV, LED_COLOR, P_LEFT, P_RIGHT
from ..vehicles.vehicle_instances import (normalize_instances, indicator_brightness,
                                LAMPS, BRAKE, LEFT, RIGHT, SUSPENSION)
from ..vehicles.vehicle_coachwork import EXPLORER_HEAD_SPRITE, EXPLORER_TAIL_SPRITE
from ..vehicles.vehicle_reference import LAMP_ANCHORS
from ..traffic.traffic_signals import signal_lamps

_MODEL_LAMPS = {SUV: (EXPLORER_HEAD_SPRITE, EXPLORER_TAIL_SPRITE, True), **LAMP_ANCHORS}

# Sprite growth: a lamp is ~0.6 m wide up close but must stay visible at 2 km,
# where a fixed-size quad would be a fraction of a pixel.
SIZE_GAIN = 0.009
SIZE_FLOOR = 0.30
MAX_SPRITE = 44.0

_LAMP_RADIUS = 0.18
# Low-beam cut-off: the projector is aimed a few degrees below the horizon.
_BEAM_DROP = .055
_EMISSIVE_INTENSITY = LUMINANCE_NITS["CROWN"] * 2.0 ** NEON_EXPOSURE_STOPS
_VEHICLE_FAR = 1200.0


def supports_ssbo(gl):
    """True when the context understands shader-storage buffers (GL 4.3)."""
    try:
        ver = gl.glGetString(gl.GL_VERSION)
        if isinstance(ver, bytes):
            ver = ver.decode("ascii", "ignore")
        major, minor = (int(x) for x in ver.split(" ")[0].split(".")[:2])
        return (major, minor) >= (4, 3)
    except Exception:
        return False


class LightField:
    """Holds the candidate lights and draws the surviving ones in one batch."""

    def __init__(self, gl, capacity=24000, far_clip=2200.0, fade_ratio=0.78):
        self.gl = gl
        self.capacity = int(capacity)
        self.far_clip = float(far_clip)
        self.fade_start = float(far_clip) * float(fade_ratio)
        self.using_ssbo = supports_ssbo(gl)

        vs, fs = lightfield_shaders(self.using_ssbo)
        from OpenGL.GL import shaders as glshaders
        self.program = glshaders.compileProgram(
            glshaders.compileShader(vs, gl.GL_VERTEX_SHADER),
            glshaders.compileShader(fs, gl.GL_FRAGMENT_SHADER),
            validate=False)

        self._uloc = {}
        self._records = np.zeros((self.capacity, LIGHT_RECORD_STRIDE), np.float32)
        self.count = 0
        self.gathered = 0
        self.culled = 0

        self._chunk_key = None
        self._lamps = np.zeros((0, LIGHT_RECORD_STRIDE), np.float32)
        self._emissive = np.zeros((0, LIGHT_RECORD_STRIDE), np.float32)
        self._indicator_anchors = {}
        # Rain scatters these projectors; the precipitation pass reads them.
        self.head_beams = np.zeros((0, 7), np.float32)

        # ---- geometry: one quad, four vertices, six indices
        quad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.quad_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad.nbytes, quad, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        self.ibo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ibo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)

        # ---- instance data
        self.buf = gl.glGenBuffers(1)
        self._target = (gl.GL_SHADER_STORAGE_BUFFER
                        if self.using_ssbo and hasattr(gl, "GL_SHADER_STORAGE_BUFFER")
                        else gl.GL_ARRAY_BUFFER)
        gl.glBindBuffer(self._target, self.buf)
        gl.glBufferData(self._target, self.capacity * LIGHT_RECORD_STRIDE * 4, None,
                        gl.GL_STREAM_DRAW)
        if self.using_ssbo:
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.buf)
        else:
            for loc, off in ((1, 0), (2, 16)):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, 4, gl.GL_FLOAT, gl.GL_FALSE,
                                         LIGHT_RECORD_STRIDE * 4, ctypes.c_void_p(off))
                gl.glVertexAttribDivisor(loc, 1)
        gl.glBindVertexArray(0)

    # ------------------------------------------------------------------ upload
    def _loc(self, name):
        key = name
        if key not in self._uloc:
            self._uloc[key] = self.gl.glGetUniformLocation(self.program, name)
        return self._uloc[key]

    def _float(self, name, value):
        self.gl.glUniform1f(self._loc(name), float(value))

    # ------------------------------------------------------------- candidates
    def sync_chunks(self, chunks):
        """Rebuild the static lamp / emissive record arrays when the world changed."""
        key = tuple(id(c) for c in chunks)
        if key == self._chunk_key:
            return
        self._chunk_key = key

        lamps = [c.lights_cpu for c in chunks
                 if c.lights_cpu is not None and len(c.lights_cpu)]
        if lamps:
            a = np.concatenate(lamps).astype(np.float32)
            a = a[a[:, 3:6].sum(axis=1) > 0.01]          # skip dead lamps
        else:
            a = np.zeros((0, 6), np.float32)
        lamp = np.empty((len(a), LIGHT_RECORD_STRIDE), np.float32)
        lamp[:, 0:3] = a[:, 0:3]
        lamp[:, 3] = _LAMP_RADIUS
        lamp[:, 4:7] = a[:, 3:6]
        lamp[:, 7] = LUMINANCE_NITS["STREET_LAMP"]
        self._lamps = lamp

        emitters = [c.neon_cpu for c in chunks
                    if c.neon_cpu is not None and len(c.neon_cpu)]
        if emitters:
            e = np.concatenate(emitters).astype(np.float32)
        else:
            e = np.zeros((0, 6), np.float32)
        em = np.empty((len(e), LIGHT_RECORD_STRIDE), np.float32)
        em[:, 0:3] = e[:, 0:3]
        # stable per-sign size so facade glow varies instead of pulsing
        jitter = np.abs(np.sin(e[:, 0] * 3.1 + e[:, 1] * 1.7)) if len(e) else 0.0
        em[:, 3] = 0.15 + 0.10 * jitter   # emitting strip width, not a multi-metre glowing facade
        em[:, 4:7] = e[:, 3:6]
        em[:, 7] = _EMISSIVE_INTENSITY
        self._emissive = em

    def set_vehicle_templates(self, templates):
        """Fit signal sprite centres to the same emitting geometry as the mesh."""
        anchors = {}
        for kind, (v, idx, opaque) in templates.items():
            tri = v[idx[:opaque].reshape(-1, 3)]
            p = tri[:, :, :3].astype(np.float64)
            centers = p.mean(axis=1)
            area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
            lamps = []
            for part, flag in ((P_LEFT, LEFT), (P_RIGHT, RIGHT)):
                for front in (True, False):
                    mask = ((tri[:, 0, 9] == part) & ((centers[:, 0] > 0) == front)
                            & (np.abs(tri[:, :, 3].mean(axis=1)) > .2) & (area > 1e-10))
                    if mask.any():
                        lamps.append((np.average(centers[mask], weights=area[mask], axis=0), flag))
            anchors[kind] = lamps
        self._indicator_anchors = anchors

    @staticmethod
    def _head_beams(positions, fx, fy, power):
        """(x, y, z, beam dir xyz, power) rows for the rain scattering pass."""
        out = np.empty((len(positions), 7), np.float32)
        out[:, 0:3] = positions
        axis = np.column_stack((fx, fy, np.full(len(fx), _BEAM_DROP, np.float32)))
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-6)
        out[:, 3:6] = axis
        out[:, 6] = power
        return out

    def _store_beams(self, parts, eye):
        """Keep only the nearest lit projectors; farther beams cannot reach the rain."""
        self.head_beams = np.zeros((0, 7), np.float32)
        if not parts:
            return
        b = np.concatenate(parts)
        b = b[b[:, 6] > .02]
        if not len(b):
            return
        d2 = (b[:, 0] - eye[0]) ** 2 + (b[:, 1] - eye[1]) ** 2
        near = d2 < HEADLIGHT_REACH ** 2
        b, d2 = b[near], d2[near]
        if len(b) > HEADLIGHT_BEAMS:
            b = b[np.argpartition(d2, HEADLIGHT_BEAMS - 1)[:HEADLIGHT_BEAMS]]
        self.head_beams = b if len(b) else np.zeros((0, 7), np.float32)

    def _traffic_lights(self, traffic, eye):
        """Head and tail light sprites for the moving vehicles near the camera."""
        beams = []
        rows = [(kind, t) for kind, t in enumerate(traffic) if t is not None and len(t)]
        if not rows:
            return None
        r = np.concatenate([normalize_instances(t) for _, t in rows])
        kinds = np.concatenate([np.full(len(t), kind, np.int32) for kind, t in rows])
        x, y, z = r[:, 12], r[:, 13], r[:, 14]
        m = ((x - eye[0]) ** 2 + (y - eye[1]) ** 2 < _VEHICLE_FAR ** 2) & (r[:, 23] > 0.)
        if not m.any():
            return None
        r = r[m]
        kinds = kinds[m]
        x, y, z = r[:, 12], r[:, 13], r[:, 14]
        fx, fy = r[:, 0], r[:, 1]
        n = np.hypot(fx, fy)
        n[n < 1e-4] = 1.0
        fx, fy = fx / n, fy / n
        flags = r[:, 19].astype(np.int32)
        power, brake = (flags & LAMPS) != 0, (flags & BRAKE) != 0

        out = np.empty((len(r) * 2, LIGHT_RECORD_STRIDE), np.float32)
        head = out[:len(r)]
        head[:, 0] = x + fx * 2.2
        head[:, 1] = y + fy * 2.2
        head[:, 2] = z + 0.75
        head[:, 3] = 0.09
        head[:, 4:7] = (1.00, 0.90, 0.68)
        head[:, 7] = LUMINANCE_NITS["HEADLIGHT"] * power * r[:, 23]
        beams.append(self._head_beams(head[:, 0:3], fx, fy, power * r[:, 23]))
        tail = out[len(r):]
        tail[:, 0] = x - fx * 2.2
        tail[:, 1] = y - fy * 2.2
        tail[:, 2] = z + 0.70
        tail[:, 3] = 0.11
        tail[:, 4:7] = (1.00, 0.16, 0.09)
        tail[:, 7] = (LUMINANCE_NITS["TAILLIGHT"] * power + LUMINANCE_NITS['BRAKELIGHT'] * brake) * r[:, 23]
        reference = np.isin(kinds, tuple(_MODEL_LAMPS))
        groups = [out[np.tile(~reference, 2)]]
        for kind, (head_anchor, tail_anchor, led) in _MODEL_LAMPS.items():
            cars = r[kinds == kind]
            if not len(cars):
                continue
            # Separate clusters rotate, pitch and scale with the same instance
            # matrix as the mesh, including road elevation and parked yaw.
            hx, hy, hz = head_anchor
            tx, ty, tz = tail_anchor
            anchors = np.array([(hx, hy, hz), (hx, -hy, hz),
                                (tx, ty, tz), (tx, -ty, tz)], np.float32)
            local = np.broadcast_to(anchors, (len(cars), 4, 3)).copy()
            car_flags = cars[:, 19].astype(np.int32)
            moving = (car_flags & SUSPENSION) != 0
            if moving.any():
                # Match the body-only suspension transform in the vehicle
                # shader so lamp sprites stay attached during dive/roll.
                pitch = cars[moving, 3]
                roll = cars[moving, 7]
                cp, sp = np.cos(pitch), np.sin(pitch)
                cr, sr = np.cos(roll), np.sin(roll)
                x, y, z = local[moving, :, 0].copy(), local[moving, :, 1].copy(), local[moving, :, 2].copy()
                z -= .55
                local[moving, :, 0] = cp[:, None] * x - sp[:, None] * z
                z = sp[:, None] * x + cp[:, None] * z
                local[moving, :, 1] = cr[:, None] * y - sr[:, None] * z
                local[moving, :, 2] = sr[:, None] * y + cr[:, None] * z + .55 + cars[moving, 15][:, None]
            lamps = np.empty((len(cars), 4, LIGHT_RECORD_STRIDE), np.float32)
            lamps[:, :, :3] = (cars[:, None, 12:15]
                               + cars[:, None, :3] * local[:, :, 0:1]
                               + cars[:, None, 4:7] * local[:, :, 1:2]
                               + cars[:, None, 8:11] * local[:, :, 2:3])
            lamps[:, :, 3] = (.045, .045, .062, .062)
            lamps[:, :2, 4:7] = LED_COLOR if led else (1.00, .90, .68)
            lamps[:, 2:, 4:7] = (1.00, .15, .08)
            lamps[:, :, 7] = (LUMINANCE_NITS["HEADLIGHT"], LUMINANCE_NITS["HEADLIGHT"],
                              LUMINANCE_NITS["TAILLIGHT"], LUMINANCE_NITS["TAILLIGHT"])
            lamps[:, :, 7] *= ((car_flags & LAMPS) != 0)[:, None]
            lamps[:, 2:, 7] += (LUMINANCE_NITS['BRAKELIGHT'] * ((car_flags & BRAKE) != 0))[:, None]
            lamps[:, :, 7] *= cars[:, None, 23]
            cfx, cfy = cars[:, 0].copy(), cars[:, 1].copy()
            norm = np.hypot(cfx, cfy)
            norm[norm < 1e-4] = 1.
            beams.append(self._head_beams(lamps[:, :2, :3].mean(axis=1), cfx / norm, cfy / norm,
                                          ((car_flags & LAMPS) != 0) * cars[:, 23]))
            groups.append(lamps.reshape(-1, LIGHT_RECORD_STRIDE))
        clock = getattr(traffic, 'clock', 0.)
        flash = indicator_brightness(clock, flags)
        for kind, anchors in getattr(self, '_indicator_anchors', {}).items():
            for anchor, flag in anchors:
                mask = (kinds == kind) & ((flags & flag) != 0) & (flash > 0.)
                cars = r[mask]
                if not len(cars):
                    continue
                lamps = np.empty((len(cars), LIGHT_RECORD_STRIDE), np.float32)
                lamps[:, :3] = (cars[:, 12:15] + cars[:, :3] * anchor[0]
                                + cars[:, 4:7] * anchor[1] + cars[:, 8:11] * anchor[2])
                lamps[:, 3], lamps[:, 4:7] = .050, (1., .32, .025)
                lamps[:, 7] = LUMINANCE_NITS['INDICATOR'] * flash[mask] * cars[:, 23]
                groups.append(lamps)
        self._store_beams(beams, eye)
        result = np.concatenate(groups)
        result = result[result[:, 7] > 0.]
        return result if len(result) else None

    # ------------------------------------------------------------------- cull
    def cull(self, view, planes, eye, night, traffic=None):
        """Keep the lights in front of the camera and inside the frustum."""
        self.count = 0
        self.gathered = self.culled = 0
        self.head_beams = np.zeros((0, 7), np.float32)
        if night < 0.04:
            return 0

        eye = np.asarray(eye, np.float32)
        basis = np.asarray(view, np.float32)[:3, :3]
        trans = np.asarray(view, np.float32)[:3, 3]
        nrm = np.asarray(planes, np.float32)[:, :3]
        off = np.asarray(planes, np.float32)[:, 3]

        parts = [self._lamps, self._emissive]
        if traffic:
            veh = self._traffic_lights(traffic, eye)
            if veh is not None:
                parts.append(veh)
            signals = signal_lamps(getattr(traffic, 'signals', None), eye, self.far_clip)
            if signals is not None:
                parts.append(signals)

        far2 = self.far_clip * self.far_clip
        keep, dists = [], []
        for arr in parts:
            if arr is None or len(arr) == 0:
                continue
            self.gathered += len(arr)
            p = arr[:, :3]
            rad = arr[:, 3]
            # world -> view space.  ``trans`` already carries -R * eye, so the
            # origin must not be subtracted again (doing both squashed the
            # whole field once the camera climbed a few hundred metres).
            v = p @ basis.T + trans
            d2 = (v * v).sum(axis=1)
            # behind the camera: the view basis looks down -Z, so anything whose
            # nearest surface reaches Z = 0 cannot contribute a pixel
            front = v[:, 2] < -rad
            inside = (p @ nrm.T + off >= -rad[:, None]).all(axis=1)
            near = d2 < far2
            m = front & inside & near
            self.culled += len(arr) - int(m.sum())
            if m.any():
                keep.append(arr[m])
                dists.append(d2[m])

        if not keep:
            return 0
        data = np.concatenate(keep) if len(keep) > 1 else keep[0]
        if len(data) > self.capacity:
            # over budget: keep the closest sources, they are the ones that read
            d = np.concatenate(dists)
            sel = np.argpartition(d, self.capacity - 1)[:self.capacity]
            data = data[sel]
        n = len(data)
        self._records[:n] = data
        self.count = int(n)
        self._upload(n)
        return self.count

    def _upload(self, n):
        gl = self.gl
        if n == 0:
            return
        gl.glBindBuffer(self._target, self.buf)
        gl.glBufferData(self._target, self._records[:n].nbytes,
                        np.ascontiguousarray(self._records[:n]), gl.GL_STREAM_DRAW)
        gl.glBindBuffer(self._target, 0)
        if self.using_ssbo:
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 0, self.buf)

    # ------------------------------------------------------------------- draw
    def draw(self, view, proj, night, fog_near, fog_far, gain=1.0):
        if self.count == 0:
            return
        gl = self.gl
        gl.glUseProgram(self.program)
        gl.glUniformMatrix4fv(self._loc("uView"), 1, gl.GL_TRUE,
                              np.ascontiguousarray(view, np.float32))
        gl.glUniformMatrix4fv(self._loc("uProj"), 1, gl.GL_TRUE,
                              np.ascontiguousarray(proj, np.float32))
        self._float("uFarClip", self.far_clip)
        self._float("uFarFadeStart", self.fade_start)
        self._float("uSizeGain", SIZE_GAIN)
        self._float("uSizeFloor", SIZE_FLOOR)
        self._float("uMaxSprite", MAX_SPRITE)
        self._float("uNightFactor", night)
        self._float("uFogNear", fog_near)
        self._float("uFogFar", fog_far)
        self._float("uGain", gain)
        self._float("uHdrOut", getattr(self, "hdr_out", 0.0))
        self._float("uExposure", getattr(self, "exposure", 1.0))
        self._float("uAirExtinction", getattr(self, "air_extinction", 0.0))
        self._float("uAirCameraHeight", getattr(self, "camera_height", 0.0))
        gl.glBindVertexArray(self.vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None, self.count)
        gl.glBindVertexArray(0)
