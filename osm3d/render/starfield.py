"""Offline HYG catalog and one reusable GPU star / optional constellation batch."""
from __future__ import annotations

import csv
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..core.astronomy import J2000, equatorial_vector
from ..world.weather_system import wind_displacement

ASSETS = Path(__file__).resolve().parents[1] / 'assets' / 'sky'
STAR_STRIDE = 11
# Common Western stick figures, specified as polylines of real Hipparcos IDs.
# They are optional recognition aids, not official constellation boundaries.
CONSTELLATIONS = {
    'Orion': ((27989, 25336, 25930, 26311, 26727, 27366, 24436, 25930), (27989, 26727), (24436, 27366), (26311, 26207)),
    'Ursa Major': ((54061, 53910, 58001, 59774, 54061), (59774, 62956, 65378, 67301)),
    'Ursa Minor': ((11767, 85822, 82080, 77055, 79822, 75097, 72607, 77055),),
    'Cassiopeia': ((746, 3179, 4427, 6686, 8886),),
    'Cygnus': ((102098, 100453, 95947), (97165, 100453, 104732)),
    'Lyra': ((91262, 91971, 92420, 93194, 92791, 91971),),
    'Aquila': ((97278, 97649, 98036), (97649, 95501)),
    'Leo': ((49669, 49583, 50583, 50335, 48455, 47908), (50583, 54872, 57632, 54879, 49669)),
    'Scorpius': ((78820, 78401, 80112, 80763, 82396, 82514, 82729, 84143, 86228, 85927, 85696), (78401, 78265)),
    'Crux': ((61084, 60718), (62434, 59747)),
    'Pegasus': ((113963, 113881, 677, 1067, 113963), (113963, 107315)),
    'Andromeda': ((677, 5447, 9640),),
    'Gemini': ((36850, 32246, 30343, 29655), (37826, 35550, 31681, 32362), (36850, 37826), (32246, 35550)),
    'Sagittarius': ((88635, 90185, 90496, 90887, 92041, 92855, 93506, 90185), (92855, 93864, 93506)),
    'Cepheus': ((105199, 106032, 116727, 110991, 110988, 105199),),
    'Auriga': ((24608, 28360, 28380, 25428, 23015, 24608),),
    'Perseus': ((14328, 15863, 17358, 18532, 18246, 14576, 15863),),
    'Bootes': ((69673, 72105, 73555, 71075, 67927, 69673), (73555, 72125, 72105)),
}


def star_colors(bv):
    """Approximate linear blackbody colors from measured B-V, luminance normalized."""
    bv = np.clip(np.asarray(bv, np.float64), -.4, 2.)
    temp = np.clip(4600. * (1. / (.92 * bv + 1.7) + 1. / (.92 * bv + .62)), 2800., 22000.) / 100.
    red = np.where(temp <= 66., 255., 329.698727 * np.maximum(temp - 60., 1.) ** -.13320476)
    green = np.where(temp <= 66., 99.4708026 * np.log(temp) - 161.119568,
                     288.12217 * np.maximum(temp - 60., 1.) ** -.07551485)
    blue = np.where(temp >= 66., 255., np.where(temp <= 19., 0., 138.517731 * np.log(np.maximum(temp - 10., 1.)) - 305.044793))
    srgb = np.clip(np.stack((red, green, blue), axis=-1) / 255., 0., 1.)
    linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
    return (linear / np.maximum(linear @ [.2126, .7152, .0722], 1e-5)[..., None]).astype(np.float32)


@dataclass(frozen=True)
class StarCatalog:
    vertices: np.ndarray
    guides: np.ndarray
    hips: dict
    names: dict

    @property
    def nbytes(self):
        return self.vertices.nbytes + self.guides.nbytes


@lru_cache(maxsize=1)
def load_star_catalog():
    payload = (ASSETS / 'stars.csv').read_bytes()
    manifest = json.loads((ASSETS / 'catalog.json').read_text(encoding='utf-8'))
    if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('offline star catalog identity mismatch')
    rows = list(csv.DictReader(payload.decode('utf-8').splitlines()))
    if len(rows) != manifest['rows'] or len(rows) < 4000:
        raise ValueError('incomplete naked-eye star catalog')
    ra = np.array([float(row['ra']) for row in rows])
    dec = np.array([float(row['dec']) for row in rows])
    mag = np.array([float(row['mag']) for row in rows])
    if not (np.isfinite(ra).all() and np.isfinite(dec).all() and np.isfinite(mag).all()
            and np.all((ra >= 0.) & (ra < 24.)) and np.all(np.abs(dec) <= 90.) and np.all(mag <= 6.5)):
        raise ValueError('invalid naked-eye star coordinates / magnitudes')
    vertices = np.empty((len(rows), STAR_STRIDE), np.float32)
    vertices[:, :3] = equatorial_vector(ra, dec)
    rr, dd = np.radians(ra * 15.), np.radians(dec)
    east = np.column_stack((-np.sin(rr), np.cos(rr), np.zeros(len(rows))))
    north = np.column_stack((-np.sin(dd) * np.cos(rr), -np.sin(dd) * np.sin(rr), np.cos(dd)))
    pmra = np.array([float(row['pmra'] or 0.) for row in rows])
    pmdec = np.array([float(row['pmdec'] or 0.) for row in rows])
    vertices[:, 3:6] = (east * pmra[:, None] + north * pmdec[:, None]) * (math.pi / (180. * 3600000.))
    bv = [float(row['ci']) if row['ci'] else .45 for row in rows]
    vertices[:, 6:9] = star_colors(bv)
    vertices[:, 9] = mag
    vertices[:, 10] = [(int(row['id']) * .61803398875) % 1. for row in rows]
    hips, names = {}, {}
    for index, row in enumerate(rows):
        if row['hip']:
            hips.setdefault(int(row['hip']), index)
        if row['proper']:
            names.setdefault(row['proper'], index)
    indices = []
    for polylines in CONSTELLATIONS.values():
        for line in polylines:
            for a, b in zip(line, line[1:]):
                if a not in hips or b not in hips:
                    raise ValueError(f'constellation star missing from catalog: {a}, {b}')
                indices.extend((hips[a], hips[b]))
    guides = np.asarray(indices, np.uint32)
    vertices.setflags(write=False)
    guides.setflags(write=False)
    return StarCatalog(vertices, guides, hips, names)


class StarField:
    """One static vertex/index upload, no CPU star transformation per frame."""

    def __init__(self, gl):
        from OpenGL.GL import shaders
        from ..shaders.stars import STARFIELD_VS, STARFIELD_FS, CONSTELLATION_FS
        self.gl = gl
        self.vao = self.vbo = self.ebo = self.program = self.guide_program = 0
        self.count = self.guide_count = self._nbytes = 0
        self._locations = {}
        try:
            catalog = load_star_catalog()
            self.program = shaders.compileProgram(shaders.compileShader(STARFIELD_VS, gl.GL_VERTEX_SHADER),
                shaders.compileShader(STARFIELD_FS, gl.GL_FRAGMENT_SHADER), validate=False)
            self.guide_program = shaders.compileProgram(shaders.compileShader(STARFIELD_VS, gl.GL_VERTEX_SHADER),
                shaders.compileShader(CONSTELLATION_FS, gl.GL_FRAGMENT_SHADER), validate=False)
            self.vao, self.vbo, self.ebo = gl.glGenVertexArrays(1), gl.glGenBuffers(1), gl.glGenBuffers(1)
            gl.glBindVertexArray(self.vao)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, catalog.vertices.nbytes, catalog.vertices, gl.GL_STATIC_DRAW)
            offset = 0
            for location, size in enumerate((3, 3, 3, 1, 1)):
                gl.glEnableVertexAttribArray(location)
                gl.glVertexAttribPointer(location, size, gl.GL_FLOAT, gl.GL_FALSE, STAR_STRIDE * 4, ctypes.c_void_p(offset * 4))
                offset += size
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, catalog.guides.nbytes, catalog.guides, gl.GL_STATIC_DRAW)
            self.count, self.guide_count, self._nbytes = len(catalog.vertices), len(catalog.guides), catalog.nbytes
        except Exception:
            self.release()
            raise
        finally:
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    @property
    def nbytes(self):
        return self._nbytes

    def _loc(self, program, name):
        key = program, name
        if key not in self._locations:
            self._locations[key] = self.gl.glGetUniformLocation(program, name)
        return self._locations[key]

    def draw(self, sky, view, proj, pixel_scale, weather=None, *, hdr=True, time=0., guides=False, origin=(0., 0.)):
        if not self.vao or sky.star_visibility <= .001:
            return
        gl = self.gl

        def common(program):
            gl.glUseProgram(program)
            for name, matrix in (('uView', view), ('uProj', proj)):
                gl.glUniformMatrix4fv(self._loc(program, name), 1, gl.GL_TRUE, np.ascontiguousarray(matrix, np.float32))
            rotation = sky.equatorial_to_world if sky.equatorial_to_world is not None else np.eye(3, dtype=np.float32)
            gl.glUniformMatrix3fv(self._loc(program, 'uEquatorialToWorld'), 1, gl.GL_TRUE, rotation)
            values = dict(uExposure=sky.exposure, uHdrOut=float(hdr), uNightFactor=sky.night,
                uStarVisibility=sky.star_visibility, uStarLimit=sky.star_limit, uStarExtinction=sky.aerosol_density,
                uTime=time, uEpochYears=(sky.julian_day - J2000) / 365.25, uPixelScale=pixel_scale,
                uCloudCover=getattr(weather, 'cloud', 0.), uStorm=getattr(weather, 'storm', 0.))
            for name, value in values.items():
                gl.glUniform1f(self._loc(program, name), float(value))
            gl.glUniform3f(self._loc(program, 'uCloudOrigin'), float(origin[0]), float(origin[1]), sky.observer_height)
            wind = getattr(weather, 'wind', (0., 0.))
            offset = wind_displacement(weather) if weather is not None else (0., 0.)
            gl.glUniform2f(self._loc(program, 'uWind'), *map(float, wind))
            gl.glUniform2f(self._loc(program, 'uWindOffset'), *map(float, offset))

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFuncSeparate(gl.GL_ONE, gl.GL_ONE, gl.GL_ZERO, gl.GL_ONE)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glBindVertexArray(self.vao)
        common(self.program)
        gl.glDrawArrays(gl.GL_POINTS, 0, self.count)
        if guides:
            common(self.guide_program)
            gl.glLineWidth(1.)
            gl.glDrawElements(gl.GL_LINES, self.guide_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def release(self):
        gl = self.gl
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
            self.vao = 0
        for name in ('vbo', 'ebo'):
            value = getattr(self, name)
            if value:
                gl.glDeleteBuffers(1, [value])
                setattr(self, name, 0)
        for name in ('program', 'guide_program'):
            value = getattr(self, name)
            if value:
                gl.glDeleteProgram(value)
                setattr(self, name, 0)
        self.count = self.guide_count = self._nbytes = 0
        self._locations.clear()
