"""Dot-matrix backdrop: a world globe, concentric dotted rings and a sparse grid.

The globe is built from map data.  When the host hands over its OpenStreetMap tile manager
(``tiles`` with ``px`` / ``request`` like the one the world map uses) the land / water split
is read straight from the low-zoom OSM tiles, so the globe literally is the OpenStreetMap map
wrapped round a sphere, drawn as dots.  Until those tiles are downloaded (or when offline) a
built-in coastline outline is used instead, so the screen is never empty.

Everything is vectorised and drawn with a single ``ui.triangles`` call per layer.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import theme as T
from .ui_kit import dot_grid, dots, prepare

# Coarse coastlines, (longitude, latitude) outlines.  Enough to read as the real continents at
# dot resolution; replaced by OpenStreetMap data when tiles are available.
LAND_POLYGONS = (
    # North America
    ((-168, 65.5), (-165, 68.5), (-156, 71.3), (-143, 70), (-130, 70), (-117, 68.5), (-108, 68), (-96, 68),
     (-90, 69.2), (-84, 68.5), (-82, 66), (-86, 64), (-93, 61.5), (-94.5, 58.8), (-90, 57), (-85, 55.3), (-82, 53),
     (-79.5, 51.5), (-78, 56), (-77, 60), (-78, 62.5), (-74, 62.3), (-70, 61), (-65, 60), (-62, 57), (-58, 54),
     (-56, 52), (-60, 50), (-66, 50), (-71, 47), (-64, 48.5), (-61, 46), (-65, 43.5), (-70, 43), (-70.5, 41.8),
     (-74, 40.3), (-76, 37), (-75.5, 35), (-78, 33.8), (-81, 31.5), (-80, 27), (-80.3, 25.2), (-82, 26.8),
     (-83, 29.5), (-86, 30.3), (-89.5, 30.2), (-91, 29.2), (-94, 29.5), (-97, 27.8), (-97.5, 24), (-97.8, 21),
     (-96, 19), (-94.5, 18.2), (-91, 18.7), (-90.5, 21), (-87, 21.5), (-87.5, 18), (-88.5, 16), (-84, 15.8),
     (-83.2, 12), (-83.8, 10), (-81.5, 8.8), (-79.5, 9.5), (-77.4, 8.6), (-78, 7.5), (-80.5, 7.3), (-82.8, 8.2),
     (-85.7, 10), (-87.5, 13), (-91.5, 14), (-94.5, 16), (-98, 16), (-101, 17.3), (-105.5, 20), (-110, 22.8),
     (-115, 27.5), (-117.2, 32.5), (-120.5, 34.5), (-122.5, 37.5), (-124.2, 40.5), (-124, 46.2), (-123, 48.4),
     (-127, 50.5), (-130, 54.5), (-134, 58), (-140, 59.7), (-148, 60.3), (-152, 59), (-158, 56.5), (-163, 54.8),
     (-158, 58.5), (-162, 60), (-166, 61.5), (-165, 63.5)),
    ((-80, 73.5), (-72, 71.5), (-62, 66.5), (-65, 63), (-72, 64), (-77, 65), (-85, 67), (-88, 70)),     # Baffin
    ((-118, 72.5), (-105, 73), (-100, 70), (-112, 69)),                                                   # Victoria
    ((-90, 80), (-72, 83), (-62, 82), (-75, 78), (-90, 76)),                                              # Ellesmere
    ((-85, 22), (-82, 23.1), (-77, 21.8), (-74.2, 20.2), (-77.5, 19.9), (-80, 21.7)),                     # Cuba
    ((-74.4, 18.4), (-72, 19.9), (-68.4, 18.6), (-71, 17.8)),                                             # Hispaniola
    ((-73, 78), (-60, 82), (-30, 83.5), (-18, 80), (-20, 72), (-22, 70), (-32, 68), (-41, 64), (-43.5, 60),
     (-49, 61), (-53, 66), (-55, 70), (-66, 76)),                                                         # Greenland
    ((-24, 65.5), (-22, 66.5), (-16, 66.5), (-13.5, 65), (-18, 63.4), (-22.5, 63.8)),                     # Iceland
    # South America
    ((-77.4, 8.6), (-75.5, 10.8), (-72, 12), (-70, 12), (-67, 10.6), (-62, 10.7), (-60, 8.5), (-57, 6), (-52, 5),
     (-50, 1.8), (-48, -1), (-44, -2.5), (-40, -3), (-35, -5.2), (-35, -9), (-38.5, -13), (-39, -17.5), (-41, -22),
     (-44, -23), (-48, -25.5), (-48.8, -28.5), (-52, -32), (-54, -34.5), (-57, -35), (-58, -38.2), (-62, -39),
     (-62, -41), (-65, -42), (-64.5, -45), (-67.5, -46.5), (-66, -48), (-69, -51), (-68.5, -52.4), (-71, -53.8),
     (-74, -52), (-75.5, -47), (-74, -43), (-73.5, -37), (-71.5, -30), (-70.5, -22), (-70.2, -18.3), (-76, -14),
     (-79.5, -7.5), (-81.2, -5), (-80, -2.5), (-80.7, 0.5), (-78.8, 1.8), (-77.3, 4), (-77.8, 7)),
    # Africa
    ((-17, 21), (-16, 24), (-13, 27.7), (-10, 29.8), (-9.8, 31.5), (-6.5, 35.5), (-2, 35.2), (3, 36.8), (10, 37.3),
     (11, 35), (10.2, 33.3), (15, 32.3), (20, 32.8), (20.3, 31), (25, 31.7), (30, 31.2), (32.5, 30), (34.2, 27.8),
     (35.5, 24), (37.3, 21), (38.6, 18), (40, 15), (42.7, 12.5), (43.4, 11.7), (51, 11.8), (51.2, 10.4), (48, 5),
     (44, 1.5), (41.5, -1.5), (40.2, -3.5), (39, -6.5), (39.4, -10.5), (40.5, -14.5), (40.7, -16.8), (35.2, -21),
     (35.5, -24), (32.8, -26), (32.8, -28.5), (31, -30), (28, -33), (25.5, -34), (20, -34.8), (18.4, -34),
     (17.5, -30.5), (15, -26.5), (14.5, -22.5), (12, -18), (11.8, -14.5), (13.5, -11.5), (13, -8.8), (12.2, -6),
     (9.5, -2), (9.5, 0.5), (9.8, 3.5), (8.5, 4.5), (5, 5.8), (1.5, 6), (-2, 4.8), (-4.8, 5.2), (-7.8, 4.4),
     (-11.5, 6.8), (-13.5, 9.5), (-15, 11.5), (-17, 14.7), (-16.5, 19)),
    ((49.3, -12), (50.4, -15.2), (47.5, -24.5), (45, -25.4), (43.3, -22.5), (44.4, -16.5)),               # Madagascar
    # Eurasia
    ((-9.5, 37), (-9.5, 43), (-1.8, 43.5), (-1.2, 46), (-4.5, 48.4), (-1.5, 48.7), (2, 51), (4.5, 52), (8.5, 54.5),
     (8.2, 57), (10.5, 57.5), (10.5, 55), (12.5, 54.3), (18, 54.8), (21, 55), (21, 57.5), (24, 57.2), (24.2, 59.4),
     (23, 60), (21.5, 61), (21.5, 63), (25, 65), (24.5, 65.8), (21.5, 65.7), (17.5, 62.5), (18.8, 60.5), (16.5, 57),
     (14.2, 55.5), (12.8, 56), (11.5, 58.8), (10.5, 59), (8, 58), (5.5, 59), (5, 61.5), (7, 63), (11, 64.5),
     (14, 67.5), (18, 69.7), (24, 71), (31, 70.3), (33, 69.4), (41, 67), (41, 66), (36, 64.5), (40, 64.5),
     (44, 66.5), (44, 68.5), (53, 68.8), (60, 69.5), (67, 69), (69, 73), (73, 72.5), (80, 73.5), (87, 75),
     (100, 76.5), (105, 77.7), (113, 73.8), (128, 72), (140, 72.5), (150, 71.3), (160, 69.7), (170, 70),
     (180, 68.8), (180, 65.5), (177, 64.5), (179, 62.3), (171, 60), (163, 59.8), (163, 56), (156, 51),
     (155.5, 57.5), (160, 61), (150, 59.4), (143, 59.3), (137, 54), (141, 53), (140, 48.5), (135, 43.3),
     (131, 42.5), (129.5, 41), (127.5, 39.5), (129.4, 37), (129.2, 35.2), (126.5, 34.5), (126.8, 37.8), (125, 39.7),
     (121.5, 39), (122, 40.8), (118, 39.2), (119, 37.2), (122.5, 37), (120.2, 35.5), (121.8, 31.8), (121, 28),
     (119, 25), (116.5, 22.8), (113.5, 22), (110.5, 21.2), (108.5, 21.8), (106.5, 20), (105.7, 18.5), (108.8, 15.5),
     (109.3, 12), (107, 10.4), (105, 8.6), (104.8, 10.5), (103, 10.8), (100.8, 13.3), (99.5, 9.8), (101, 6.8),
     (103.4, 4.5), (104.2, 1.4), (101.3, 2.7), (100.3, 5.5), (98.3, 8), (98.5, 12.5), (97.5, 16.5), (94.5, 16),
     (94.2, 19), (91.8, 22.5), (90.5, 22.2), (87, 21.5), (86.5, 19.8), (84.5, 18.2), (80.2, 15.5), (80.2, 13),
     (79.8, 10.3), (77.5, 8.1), (76.2, 10), (74.8, 13), (73.2, 17.5), (72.8, 20.5), (70.5, 20.8), (68.8, 22.5),
     (70.2, 23.2), (67, 24.8), (66.5, 25.5), (61.5, 25.2), (57.5, 25.7), (56.7, 27), (54, 26.7), (51, 28.8),
     (48.5, 30), (48, 29.5), (50, 26.5), (51.5, 24.8), (51.8, 26), (53, 24.2), (56, 24.8), (56.4, 26.3), (57, 24),
     (59.7, 22.5), (58.5, 20.5), (55.5, 17.5), (52, 16), (48, 14), (44.5, 12.8), (43.2, 13), (42.8, 15), (41, 19),
     (39.1, 21.5), (37.2, 25), (35, 28), (34.9, 29.5), (34.2, 31.3), (35, 33), (35.8, 35.5), (36.2, 36.8),
     (33, 36.2), (30, 36.3), (28, 36.8), (27.2, 38.2), (26.2, 39.3), (26.5, 40), (29, 41), (30.5, 41.1), (33, 42),
     (36, 41.5), (38, 41), (41.5, 41.5), (41.7, 43), (39, 44.5), (37.5, 45.2), (38, 46.5), (35, 45.3), (33.5, 44.5),
     (32.5, 45.5), (31.5, 46.6), (30.2, 45.8), (28.7, 44.5), (28, 43.2), (27.7, 42), (26.5, 40.8), (24, 40.8),
     (23.5, 40), (22.8, 39), (24, 37.8), (22.5, 36.4), (21.6, 37.2), (21, 38.8), (19.4, 40.4), (19.5, 42),
     (18.5, 42.6), (15.5, 44.8), (13.6, 45.6), (12.4, 44.3), (14, 42.3), (16, 41.2), (18.5, 40.2), (16.5, 38.8),
     (15.7, 38), (15.7, 40.2), (14.5, 40.7), (12, 42), (10.5, 43), (8.8, 44.4), (7.3, 43.8), (4.5, 43.4), (3.2, 42.2),
     (0.7, 40.5), (-0.3, 39.2), (-0.7, 37.6), (-2.2, 36.7), (-5.6, 36), (-6.3, 36.8), (-8.8, 37)),
    ((-5.7, 50), (1.4, 51.2), (1.7, 52.8), (0, 53.5), (-1.5, 55), (-2, 57.5), (-4, 57.7), (-3.2, 58.6), (-5, 58.6),
     (-6, 56.5), (-5, 55), (-3, 54.8), (-3, 53.4), (-4.7, 52.8), (-5.2, 51.7), (-3.2, 51.4)),             # Great Britain
    ((-10, 51.6), (-6, 52.2), (-6, 54.2), (-6.2, 55.2), (-8.5, 55), (-10, 53.7)),                         # Ireland
    ((130.9, 34), (135, 33.5), (139.8, 35), (141, 38), (142, 40.5), (141.2, 41.4), (139.8, 39.7), (137, 37),
     (135.5, 35.5), (131.5, 34.5)),                                                                       # Honshu
    ((140, 42), (141.5, 45.4), (145.5, 43.3), (143, 42)),                                                 # Hokkaido
    ((129.7, 33), (131.8, 34), (132, 31.5), (130.5, 31)),                                                 # Kyushu
    ((120, 23), (121.5, 25.2), (121.8, 24), (120.8, 22)),                                                 # Taiwan
    ((120, 18.5), (122.3, 18.4), (124, 13), (121, 13.7), (120, 16)),                                      # Luzon
    ((122, 7), (126.5, 9), (126.3, 6), (124, 6.5)),                                                       # Mindanao
    ((95.3, 5.6), (98, 4), (104, -1), (106, -5.8), (102, -4), (97, 1.5)),                                 # Sumatra
    ((105.2, -6.8), (114.5, -7.8), (114, -8.7), (106, -7.5)),                                             # Java
    ((109, 1.5), (111, 2), (113, 3.5), (117, 7), (119, 5), (117.8, 0.5), (116.5, -3.5), (114, -4),
     (110.5, -3), (109, -0.5)),                                                                           # Borneo
    ((119, -5.5), (120.5, -2.8), (119.8, 0.8), (125, 1.5), (121, -0.8), (123, -4.8), (120.5, -5.8)),      # Sulawesi
    ((131, -0.8), (135, -3.3), (141, -2.6), (147, -6), (150, -10.5), (147, -10), (143, -9), (141, -9),
     (138, -8), (135, -4.5), (132.5, -2.8)),                                                              # New Guinea
    ((113.3, -22), (114, -26), (115, -34), (118, -35), (123.5, -34), (129, -31.7), (134, -32.8), (137.8, -33),
     (138, -35), (140, -37.8), (144, -38.5), (147, -38.3), (150, -37.5), (151.2, -33.8), (153.5, -28.5),
     (153, -25), (150, -22.5), (146.3, -19), (145.4, -15), (143.5, -14), (142.5, -10.7), (141.5, -13),
     (141.2, -17), (139.5, -17.5), (136, -15), (135.5, -12), (132, -11.3), (130.5, -12.5), (129.5, -15),
     (126, -14), (122.5, -17.5), (121, -19.5), (117, -20.7)),                                             # Australia
    ((144.7, -40.7), (148.3, -40.9), (148, -43.2), (146, -43.6)),                                         # Tasmania
    ((172.7, -34.5), (175.5, -37), (178.5, -37.7), (177, -39.5), (175, -41.5), (174.8, -39.5), (173.8, -39),
     (174.5, -37.5)),                                                                                     # NZ north
    ((172.7, -40.5), (174.3, -41.7), (171.5, -44.5), (170.5, -46.3), (166.5, -46), (168, -44)),           # NZ south
    ((79.8, 9.8), (81.8, 7.5), (81.2, 6), (80, 6)),                                                       # Sri Lanka
    ((11, 79.5), (20, 80.3), (27, 79.5), (17, 77)),                                                       # Svalbard
    ((52, 71), (57, 75), (68, 77), (60, 74.5), (55, 71)),                                                 # Novaya Zemlya
)
WATER_POLYGONS = (
    ((47, 45), (53, 46.5), (53.5, 42), (54, 37.5), (49, 37.5), (49.5, 40.5), (47.5, 43)),                 # Caspian
)


def _inside(polygon, lon, lat):
    """Vectorised even-odd point-in-polygon."""
    poly = np.asarray(polygon, np.float64)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    inside = np.zeros(lon.shape, bool)
    for ax, ay, bx, by in zip(x0, y0, x1, y1):
        if ay == by:
            continue
        crosses = (ay > lat) != (by > lat)
        xi = ax + (lat - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (lon < xi)
    return inside


def polygon_land_mask(lat, lon):
    land = lat < -71.0
    for poly in LAND_POLYGONS:
        land |= _inside(poly, lon, lat)
    for poly in WATER_POLYGONS:
        land &= ~_inside(poly, lon, lat)
    return land


def osm_land_mask(px, lat, lon, z=2):
    """Land / water from downloaded OpenStreetMap tiles (``px[(z, x, y)] = (w, h, raw)``).

    Returns a boolean array, or ``None`` if the tiles aren't all there yet or have no usable
    water / land contrast (an unexpected map style): the caller then keeps the built-in outline.
    """
    n = 1 << z
    tiles = {}
    for x in range(n):
        for y in range(n):
            d = px.get((z, x, y))
            if d is None:
                return None
            tiles[(x, y)] = d
    w, h, _ = tiles[(0, 0)]
    mosaic = np.zeros((n * h, n * w, 3), np.uint8)
    for (x, y), (tw, th, raw) in tiles.items():
        a = np.frombuffer(raw, np.uint8)
        if tw != w or th != h or a.size % (tw * th):
            return None
        mosaic[y * h:(y + 1) * h, x * w:(x + 1) * w] = a.reshape(th, tw, a.size // (tw * th))[..., :3]
    s = np.sin(np.radians(np.clip(lat, -85.0511, 85.0511)))
    u = (lon + 180.0) / 360.0
    v = 0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)
    ix = np.clip((u * n * w).astype(int), 0, n * w - 1)
    iy = np.clip((v * n * h).astype(int), 0, n * h - 1)
    rgb = mosaic[iy, ix].astype(np.float32)
    f = rgb[:, 2] - rgb[:, 0]                       # OSM water is blue-ish, land is warm grey / beige
    lo, hi = np.percentile(f, (8, 92))
    if hi - lo < 14:
        return None
    land = f < (lo + hi) / 2
    frac = float(land.mean())
    return land if 0.12 < frac < 0.55 else None


class DotGlobe:
    """A rotating globe of dots on a lat/lon grid."""

    STEP = 2.6                                       # degrees between rows

    def __init__(self):
        lats, lons = [], []
        for lat in np.arange(-86.0, 86.1, self.STEP):
            c = max(0.12, math.cos(math.radians(lat)))
            count = max(6, int(round(360.0 / (self.STEP / c))))
            lon = (np.arange(count) + (0.5 if int((lat + 90) / self.STEP) % 2 else 0.0)) * (360.0 / count) - 180.0
            lats.append(np.full(count, lat))
            lons.append(lon)
        self.lat = np.concatenate(lats)
        self.lon = np.concatenate(lons)
        self.land = polygon_land_mask(self.lat, self.lon)
        self.source = "built-in"
        self._next_try = 0.0
        phi, lam = np.radians(self.lat), np.radians(self.lon)
        self._cp, self._sp = np.cos(phi), np.sin(phi)
        self._lam = lam

    def use_tiles(self, tiles):
        """Pull land / water from OSM tiles (cheap no-op until they've all arrived)."""
        if tiles is None or self.source == "OpenStreetMap":
            return
        now = time.time()
        if now < self._next_try:
            return
        self._next_try = now + 1.0
        try:
            px = tiles.px
            for x in range(4):
                for y in range(4):
                    if (2, x, y) not in px:
                        tiles.request((2, x, y))
            mask = osm_land_mask(px, self.lat, self.lon, 2)
        except Exception:
            return
        if mask is not None:
            self.land, self.source = mask, "OpenStreetMap"

    def draw(self, ui, cx, cy, r, t, dim=1.0, tilt=0.38, speed=0.10, view=None):
        """``view`` = (lat, lon) turns the globe to face that place instead of spinning."""
        if view is None:
            lam = self._lam + t * speed
        else:
            lam, tilt = self._lam - math.radians(view[1]), math.radians(view[0])
        x0 = self._cp * np.sin(lam)
        z0 = self._cp * np.cos(lam)
        y0 = self._sp
        ct, st = math.cos(tilt), math.sin(tilt)
        y = y0 * ct - z0 * st
        z = y0 * st + z0 * ct
        x = x0
        vis = z > 0.02
        base = r * math.radians(self.STEP) * 0.56
        k = (0.30 + 0.70 * z[vis])
        land = self.land[vis]
        px, py = cx + x[vis] * r, cy - y[vis] * r
        a = (0.25 + 0.75 * z[vis]) * dim
        # Land: bright dots
        col = np.empty((int(land.sum()), 4), np.float32)
        col[:, :3] = 1.0
        col[:, 3] = np.clip(a[land], 0, 1)
        dots(ui, px[land], py[land], base * k[land] * 0.5, None, col)
        # Water: barely-there pin pricks keep the sphere readable
        sea = ~land
        sea_col = np.empty((int(sea.sum()), 4), np.float32)
        sea_col[:, :3] = 1.0
        sea_col[:, 3] = 0.10 * dim * (0.3 + 0.7 * z[vis][sea])
        dots(ui, px[sea], py[sea], np.maximum(0.7, base * 0.10), None, sea_col)
        # Limb: a ring of dots
        ang = np.linspace(0, math.tau, 220, endpoint=False, dtype=np.float32)
        dots(ui, cx + np.cos(ang) * (r + 3), cy + np.sin(ang) * (r + 3), 0.9, None, (1.0, 1.0, 1.0, 0.45 * dim))


def soft_glow(ui, cx, cy, radius, t, dim=1.0):
    """A large, very soft two-colour aurora — the Apple splash, without a blur pass.

    Stacked translucent discs approximate a gaussian falloff; the hue drifts slowly
    between blue and violet so the background is alive but never busy.
    """
    wobble = 0.035 * math.sin(t * 0.23)
    layers = 26
    for i in range(layers, 0, -1):
        f = i / layers
        rr = radius * (1.85 + wobble) * f
        a = (0.016 + 0.020 * (1.0 - f)) * dim
        ui.circle(cx, cy, rr, (0.06, 0.34, 1.0, a * 1.05))
        ui.circle(cx + rr * 0.30, cy + rr * 0.18, rr * 0.80,
                  (0.42, 0.20, 0.92, a * 0.55))
        ui.circle(cx - rr * 0.26, cy - rr * 0.12, rr * 0.62,
                  (0.05, 0.55, 0.62, a * 0.35))


def dotted_rings(ui, cx, cy, radii, t, dim=1.0):
    """Legacy hook kept for call sites: now a soft glow rather than dotted rings."""
    soft_glow(ui, cx, cy, max(radii) * 0.62, t, dim)


class Backdrop:
    """Black field, sparse dot grid, dotted rings and the dot globe."""

    def __init__(self):
        self.globe = DotGlobe()

    @property
    def source(self):
        return self.globe.source

    def draw(self, ui, w, h, t, dim=1.0, globe=None, tiles=None, rings=True, grid=True, show_globe=True,
             view=None):
        """``globe`` = (cx, cy, radius) in pixels; default is centred."""
        prepare(ui)
        ui.gradient(0, 0, w, h, (0.055, 0.072, 0.110, 1.0), (0.008, 0.010, 0.018, 1.0))
        if globe is None:
            globe = (w / 2, h * 0.46, min(w, h) * 0.27)
        cx, cy, r = globe
        if rings:
            soft_glow(ui, cx, cy, r, t, dim)
        if grid:
            dot_grid(ui, 0, 0, w, h, 34.0, (1.0, 1.0, 1.0, 0.05 * dim), 0.6)
        self.globe.use_tiles(tiles)
        if show_globe:
            self.globe.draw(ui, cx, cy, r, t, dim, view=view)
