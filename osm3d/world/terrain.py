"""Terrain elevation: terrarium DEM tiles, PNG decode and height sampling.

Elevation comes from the public AWS "elevation-tiles-prod" terrain tiles
(RGB-encoded, metre resolution).  Tiles are cached on disk next to the OSM
data so a city only pays for them once.
"""
from __future__ import annotations

import math
import struct
import threading
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from ..core.config import TERRAIN_URL, TERRAIN_ZOOM

_UA = {"User-Agent": "osm3d-viewer/1.0 (educational PyOpenGL demo)"}
_TILE = 256


def decode_png(blob: bytes) -> np.ndarray:
    """Decode an 8-bit, non-interlaced PNG into an (h, w, channels) uint8 array."""
    if not blob or blob[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, ihdr, idat = 8, None, []
    while pos + 8 <= len(blob):
        (length,) = struct.unpack(">I", blob[pos:pos + 4])
        kind = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", data)
        elif kind == b"IDAT":
            idat.append(data)
        elif kind == b"IEND":
            break
    if ihdr is None or not idat:
        raise ValueError("incomplete PNG")
    width, height, depth, color, _comp, _filt, interlace = ihdr
    if depth != 8 or interlace != 0 or color not in (0, 2, 6):
        raise ValueError(f"unsupported PNG {width}x{height} depth={depth} color={color}")
    bpp = {0: 1, 2: 3, 6: 4}[color]
    raw = zlib.decompress(b"".join(idat))
    stride = width * bpp
    if len(raw) < height * (stride + 1):
        raise ValueError("truncated PNG data")

    rows = np.frombuffer(raw, np.uint8, height * (stride + 1)).reshape(height, stride + 1)
    out = np.empty((height, width, bpp), np.uint8)
    prev = np.zeros(stride, np.int32)
    for y in range(height):
        filt = int(rows[y, 0])
        line = rows[y, 1:].astype(np.int32)
        if filt == 0:
            pass
        elif filt == 1:
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif filt == 2:
            line = (line + prev) & 255
        elif filt == 3:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
        elif filt == 4:
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        else:
            raise ValueError(f"bad PNG filter {filt}")
        out[y, :, :] = line.reshape(width, bpp)
        prev = line
    return out


def _rgb_to_height(px: np.ndarray) -> np.ndarray:
    """Terrarium encoding: (R * 256 + G + B / 256) - 32768 metres."""
    if px.shape[2] == 1:
        return px[:, :, 0].astype(np.float32) - 128.0
    r = px[:, :, 0].astype(np.float32)
    g = px[:, :, 1].astype(np.float32)
    b = px[:, :, 2].astype(np.float32)
    return (r * 256.0 + g + b / 256.0) - 32768.0


class HeightField:
    """Piecewise-linear lookup over the terrain's diagonal facets."""

    __slots__ = ("data", "x0", "y0", "x1", "y1", "sx", "sy", "h", "w")

    def __init__(self, data, x0, y0, x1, y1):
        self.data = np.ascontiguousarray(data, np.float32)
        self.x0, self.y0, self.x1, self.y1 = float(x0), float(y0), float(x1), float(y1)
        self.h, self.w = self.data.shape
        self.sx = (self.w - 1) / max(self.x1 - self.x0, 1e-6)
        self.sy = (self.h - 1) / max(self.y1 - self.y0, 1e-6)

    @classmethod
    def flat(cls, n=2):
        return cls(np.zeros((n, n), np.float32), 0.0, 0.0, 1.0, 1.0)

    def at(self, x, y):
        """Height of the *rendered* terrain surface.

        The terrain mesh is two triangles per cell, split along the (0,0)-(1,1)
        diagonal.  Sampling that same piecewise-linear surface (rather than
        bilinear) keeps roads, cars and building bases exactly on the ground
        that is actually drawn; bilinear heights can sit a metre below the
        coarse mesh on slopes and leave streets buried.
        """
        gx = np.clip((np.asarray(x, np.float64) - self.x0) * self.sx, 0.0, self.w - 1.0)
        gy = np.clip((np.asarray(y, np.float64) - self.y0) * self.sy, 0.0, self.h - 1.0)
        ix = np.floor(gx).astype(np.int64)
        iy = np.floor(gy).astype(np.int64)
        jx = np.minimum(ix + 1, self.w - 1)
        jy = np.minimum(iy + 1, self.h - 1)
        tx = gx - ix
        ty = gy - iy
        d = self.data
        h00, h01 = d[iy, ix], d[iy, jx]
        h10, h11 = d[jy, ix], d[jy, jx]
        tri_ab = h00 + tx * (h01 - h00) + ty * (h11 - h01)
        tri_ac = h00 + ty * (h10 - h00) + tx * (h11 - h10)
        return np.where(tx >= ty, tri_ab, tri_ac).astype(np.float32)

    def point(self, x, y) -> float:
        return float(self.at(x, y))

    def slope(self, x, y) -> float:
        """Terrain gradient magnitude (rise over run) around a point."""
        d = 2.0
        dzdx = (self.point(x + d, y) - self.point(x - d, y)) / (2.0 * d)
        dzdy = (self.point(x, y + d) - self.point(x, y - d)) / (2.0 * d)
        return float(math.hypot(dzdx, dzdy))


class TerrainTiles:
    """Downloads / caches terrarium elevation tiles and samples them."""

    def __init__(self, cache_dir, zoom: int = TERRAIN_ZOOM, enabled: bool = True,
                 timeout: float = 8.0):
        self.dir = Path(cache_dir) / "terrain"
        self.zoom = int(zoom)
        self.enabled = bool(enabled)
        self.timeout = float(timeout)
        self._tiles: dict = {}
        self._bad: set = set()
        self._lock = threading.Lock()
        self.fetches = 0
        self.failures = 0
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.enabled = False

    def _path(self, x, y):
        return self.dir / f"{self.zoom}_{x}_{y}.png"

    def _load(self, x, y):
        key = (x, y)
        with self._lock:
            hit = self._tiles.get(key)
            if hit is not None:
                return hit if hit is not False else None
            if key in self._bad:
                return None
        if not self.enabled:
            return None
        path = self._path(x, y)
        blob = None
        try:
            if path.exists():
                blob = path.read_bytes()
            else:
                url = TERRAIN_URL.format(z=self.zoom, x=x, y=y)
                req = urllib.request.Request(url, headers=_UA)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    blob = resp.read()
                if blob:
                    try:
                        tmp = path.with_suffix(".tmp")
                        tmp.write_bytes(blob)
                        tmp.replace(path)
                    except OSError:
                        pass
                with self._lock:
                    self.fetches += 1
            px = _rgb_to_height(decode_png(blob))
            with self._lock:
                self._tiles[key] = px
            return px
        except Exception:
            with self._lock:
                self._bad.add(key)
                self.failures += 1
            return None

    def heights_at(self, lats, lons):
        """Absolute elevation (metres) for arrays of latitude / longitude."""
        lats = np.asarray(lats, np.float64)
        lons = np.asarray(lons, np.float64)
        shape = np.broadcast(lats, lons).shape
        out = np.zeros(shape, np.float32)
        if not self.enabled or out.size == 0:
            return out
        n = 1 << self.zoom
        span = _TILE * n
        px = (lons + 180.0) / 360.0 * span
        lat_c = np.clip(lats, -85.05, 85.05)
        py = (1.0 - np.arcsinh(np.tan(np.radians(lat_c))) / np.pi) / 2.0 * span
        px = np.broadcast_to(px, shape)
        py = np.broadcast_to(py, shape)
        tx = np.clip(np.floor(px / _TILE).astype(np.int64), 0, n - 1)
        ty = np.clip(np.floor(py / _TILE).astype(np.int64), 0, n - 1)
        lx = px - tx * _TILE
        ly = py - ty * _TILE
        if out.ndim == 0:
            tx, ty, lx, ly = tx.reshape(1), ty.reshape(1), lx.reshape(1), ly.reshape(1)
        tx_f, ty_f = tx.ravel(), ty.ravel()
        lx_f, ly_f = lx.ravel(), ly.ravel()
        out_f = out.reshape(-1)
        for ux, uy in set(zip(tx_f.tolist(), ty_f.tolist())):
            arr = self._load(ux, uy)
            if arr is None:
                continue
            mask = (tx_f == ux) & (ty_f == uy)
            xs = np.clip(lx_f[mask], 0.0, _TILE - 1.001)
            ys = np.clip(ly_f[mask], 0.0, _TILE - 1.001)
            ix = xs.astype(np.int64)
            iy = ys.astype(np.int64)
            fx = xs - ix
            fy = ys - iy
            h, w = arr.shape
            jx = np.minimum(ix + 1, w - 1)
            jy = np.minimum(iy + 1, h - 1)
            top = arr[iy, ix] * (1.0 - fx) + arr[iy, jx] * fx
            bot = arr[jy, ix] * (1.0 - fx) + arr[jy, jx] * fx
            out_f[mask] = top * (1.0 - fy) + bot * fy
        return out

    def height_at(self, lat, lon) -> float:
        return float(self.heights_at(np.asarray([lat]), np.asarray([lon]))[0])

    def grid(self, s, w, n, e, samples):
        """Elevation grid over a lat/lon box: row 0 is the southern edge."""
        samples = max(2, int(samples))
        lats = np.linspace(s, n, samples)
        lons = np.linspace(w, e, samples)
        lon_g, lat_g = np.meshgrid(lons, lats)
        return self.heights_at(lat_g, lon_g)
