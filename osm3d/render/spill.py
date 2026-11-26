"""World-space buckets of street lamps for the surface spill shader.

The surface pass used to receive the 256 lamps nearest the camera target through
uniform arrays, which lit roughly one block and then went dark a street away.
Each lamp only matters within ``LAMP_REACH`` (100 m, see shaders/streetspill.py), so
the whole resident world is bucketed instead.

Lamps never move, so they are bucketed once per chunk-set change into a coarse
grid: every cell stores the lamps that can reach any point inside it, nearest to
the cell centre first.  A fragment then fetches its own cell's list and loops
only those, which lights every street in the resident world at a cost per pixel
close to the old fixed 256-iteration loop.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

import numpy as np

# Reach and per-cell cap come from the shader module, so the CPU buckets and the
# GLSL cut-off cannot drift apart (they used to be duplicated by hand).
from ..shaders.streetspill import LAMP_REACH, MAX_CELL_LIGHTS

CELL_SIZE = 48.0          # grid pitch in metres
INFLUENCE = LAMP_REACH    # how far one lamp reaches (matches the shader cutoff)
MAX_PER_CELL = MAX_CELL_LIGHTS   # candidates kept per cell, nearest to the cell centre
TEX_WIDTH = 1024          # texels per row in the index/record textures
MAX_GRID = 1024           # cap on cells per axis; the pitch grows past that
UPLOAD_BYTES = 256 * 1024 # maximum texture payload in one render-thread step

_UNIT_CELL, _UNIT_REC = 5, 6


@dataclass
class SpillData:
    origin: tuple
    cell: float
    lamps: int
    records: int
    cells: np.ndarray
    rec: np.ndarray


def prepare_grid(lamps):
    """Pack a lamp snapshot without OpenGL; large NumPy operations run in a worker."""
    lamps = np.asarray(lamps, np.float32)
    if not lamps.size:
        return SpillData((0.0, 0.0), CELL_SIZE, 0, 0,
                         np.zeros((1, 1, 4), np.float32), np.zeros((1, 1, 4), np.float32))

    px, py = lamps[:, 0], lamps[:, 1]
    margin = INFLUENCE + CELL_SIZE * 0.7072
    cell = CELL_SIZE
    x0, y0 = float(px.min() - margin), float(py.min() - margin)
    while True:
        nx = int(np.ceil((float(px.max() + margin) - x0) / cell)) + 1
        ny = int(np.ceil((float(py.max() + margin) - y0) / cell)) + 1
        if max(nx, ny) <= MAX_GRID: break
        cell *= 1.5

    ci = np.floor((px - x0) / cell).astype(np.int32)
    cj = np.floor((py - y0) / cell).astype(np.int32)
    ids = np.arange(len(lamps))
    reach = int(np.ceil(margin / cell)) + 1
    # Stable ring order preserves the existing approximately nearest-first cap.
    offs = [(di, dj) for dj in range(-reach, reach + 1)
            for di in range(-reach, reach + 1)
            if di * di + dj * dj <= (reach + 0.5) ** 2]
    offs.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])

    pxc, pyc = (px - x0) / cell, (py - y0) / cell
    cids, lids = [], []
    for di, dj in offs:
        ii, jj = ci + di, cj + dj
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        dx, dy = (pxc - ii - 0.5) * cell, (pyc - jj - 0.5) * cell
        ok &= dx * dx + dy * dy <= margin * margin
        if ok.any():
            cids.append(jj[ok] * nx + ii[ok])
            lids.append(ids[ok])

    if cids:
        cid, lid = np.concatenate(cids), np.concatenate(lids)
        order = np.argsort(cid, kind="stable")
        cid, lid = cid[order], lid[order]
        counts = np.bincount(cid, minlength=nx * ny)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        keep = (np.arange(len(cid)) - starts[cid]) < MAX_PER_CELL
        cid, lid = cid[keep], lid[keep]
    else:
        cid = lid = np.empty(0, np.int64)

    counts = np.bincount(cid, minlength=nx * ny).astype(np.float32)
    starts = np.concatenate(([0.0], np.cumsum(counts)[:-1])).astype(np.float32)
    cells = np.zeros((ny * nx, 4), np.float32)
    cells[:, 0], cells[:, 1] = starts, counts
    # Downward fixtures cannot reach receivers above this cell's tallest source.
    cells[:, 2] = -np.finfo(np.float32).max
    np.maximum.at(cells[:, 2], cid, lamps[lid, 2])

    total = len(lid)
    rows = max(1, (total + TEX_WIDTH - 1) // TEX_WIDTH)
    rec = np.zeros((rows, TEX_WIDTH, 4), np.float32)
    flat = rec.reshape(-1, 4)
    flat[:total, :3] = lamps[lid, :3]
    if total:
        # Chromaticity only: candela and luminance normalization remain in GLSL.
        rgb = np.maximum(lamps[lid, 3:6], 0.0)
        rgb /= np.maximum(rgb.max(axis=1, keepdims=True), 1e-5)
        flat[:total, 3] = (np.round(rgb[:, 0] * 255.0) * 65536.0
                           + np.round(rgb[:, 1] * 255.0) * 256.0
                           + np.round(rgb[:, 2] * 255.0))
    return SpillData((x0, y0), cell, len(lamps), total, cells.reshape(ny, nx, 4), rec)


def prepare_chunks(tables):
    """Gather/filter resident lamps off-thread as well as packing their grid."""
    lamps = np.concatenate(tables) if tables else np.empty((0, 6), np.float32)
    lamps = lamps[lamps[:, 3:6].sum(axis=1) > 0.01]
    return prepare_grid(lamps)


class SpillGrid:
    """Builds and owns the lamp textures sampled by ``streetLights``."""

    def __init__(self, gl):
        self.gl = gl
        self.origin = (0.0, 0.0)
        self.cell = CELL_SIZE
        self.lamps = self.records = 0
        self._loc = {}
        self.key = None
        self._desired = self._future = self._upload = None
        self._pool = None
        self._closed = False
        self.tex_cell = self._new_tex()
        self.tex_rec = self._new_tex()

    # ------------------------------------------------------------------ GPU
    def _new_tex(self, width=1, height=1, owned=None):
        gl = self.gl
        gl.glActiveTexture(gl.GL_TEXTURE0)
        tex = gl.glGenTextures(1)
        if owned is not None: owned.append(tex)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        initial = np.zeros((1, 1, 4), np.float32) if owned is None else None
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0,
                        gl.GL_RGBA, gl.GL_FLOAT, initial)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return tex

    def _texture_steps(self, data, owned):
        gl = self.gl
        h, w = data.shape[:2]
        tex = self._new_tex(w, h, owned)
        yield
        rows = max(1, UPLOAD_BYTES // (w * 16))
        for y in range(0, h, rows):
            block = data[y:y+rows]
            # Unit 0 only: do not disturb the shadow texture left on unit 1.
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, y, w, len(block),
                             gl.GL_RGBA, gl.GL_FLOAT, block)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            yield
        return tex

    # ----------------------------------------------------------- preparation
    def request(self, key, tables):
        """Coalesce chunk changes; never queue more than one CPU preparation."""
        if self._closed or (self._desired is not None and self._desired[0] == key): return
        self._desired = (key, tuple(tables))
        if self._upload is not None and self._upload[0] != key:
            self._upload[1].close()
            self._upload = None
        if self._future is not None and self._future[0] != key and self._future[1].cancel():
            self._future = None

    def update(self, max_ms=0.65):
        """Consume completed CPU work and bound uploads; old textures remain bound."""
        if self._closed: return
        started = time.perf_counter()
        if self._future is not None and self._future[1].done():
            key, future = self._future
            self._future = None
            data = future.result()       # done() above: this cannot wait on a worker
            if self._desired is not None and key == self._desired[0]:
                self._upload = (key, self.build_steps(data))
        if self._upload is not None:
            key, steps = self._upload
            while (time.perf_counter() - started) * 1000 < max_ms:
                try: next(steps)
                except StopIteration:
                    self.key = key
                    self._upload = None
                    break
        if (self._desired is not None and self._desired[0] != self.key
                and self._future is None and self._upload is None):
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='lamp-grid')
            key, tables = self._desired
            self._future = (key, self._pool.submit(prepare_chunks, tables))

    def build(self, lamps):
        """Blocking convenience for isolated/static probes; streaming uses request/update."""
        self._desired = None
        if self._upload is not None: self._upload[1].close()
        self._upload = None
        self.key = None
        for _ in self.build_steps(prepare_grid(lamps)): pass

    def build_steps(self, data):
        """Upload prepared textures, then atomically publish the pair and metadata."""
        owned, complete = [], False
        try:
            cell = yield from self._texture_steps(data.cells, owned)
            rec = yield from self._texture_steps(data.rec, owned)
            old = [self.tex_cell, self.tex_rec]
            self.tex_cell, self.tex_rec = cell, rec
            self.origin, self.cell = data.origin, data.cell
            self.lamps, self.records = data.lamps, data.records
            complete = True
            self.gl.glDeleteTextures(old)
        finally:
            if not complete and owned: self.gl.glDeleteTextures(owned)

    def close(self):
        """Cancel partial GL uploads before context teardown; CPU work has no GL access."""
        if self._closed: return
        self._closed = True
        self._desired = None
        if self._upload is not None: self._upload[1].close()
        self._upload = None
        if self._future is not None: self._future[1].cancel()
        self._future = None
        if self._pool is not None: self._pool.shutdown(wait=False, cancel_futures=True)
        self._pool = None
        self.gl.glDeleteTextures([self.tex_cell, self.tex_rec])
        self.tex_cell = self.tex_rec = 0

    # --------------------------------------------------------------- binding
    def _uniforms(self, program, names):
        locs = self._loc.get(program)
        if locs is None:
            get = self.gl.glGetUniformLocation
            locs = {k: get(program, k) for k in
                    ("uLampCell", "uLampRec", "uLampOrigin", "uLampCellSize")}
            self._loc[program] = locs
        return locs

    def bind(self, program):
        gl = self.gl
        locs = self._uniforms(program, None)
        for unit, tex, name in ((_UNIT_CELL, self.tex_cell, "uLampCell"),
                                (_UNIT_REC, self.tex_rec, "uLampRec")):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
            loc = locs[name]
            if loc != -1:
                gl.glUniform1i(loc, unit)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        if locs["uLampOrigin"] != -1:
            gl.glUniform2f(locs["uLampOrigin"], self.origin[0], self.origin[1])
        if locs["uLampCellSize"] != -1:
            gl.glUniform1f(locs["uLampCellSize"], self.cell)
