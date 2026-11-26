"""Global constants, StreamConfig, SkyClock and time parsing."""
from __future__ import annotations
from dataclasses import dataclass
import math
from datetime import timedelta

from .astronomy import parse_date

EARTH_RADIUS = 6_378_137.0

BG_COLOR = (0.45, 0.60, 0.85)

KIND_DEFAULT_HEIGHT = {
    "house": 6.0, "detached": 6.0, "semidetached_house": 7.0, "terrace": 8.0,
    "bungalow": 4.0, "garage": 3.0, "garages": 3.0, "shed": 2.8, "hut": 2.8,
    "cabin": 4.0, "roof": 3.5, "carport": 3.0, "kiosk": 3.0, "apartments": 15.0,
    "residential": 10.0, "commercial": 14.0, "office": 20.0, "retail": 8.0,
    "industrial": 9.0, "warehouse": 8.0, "church": 16.0, "cathedral": 32.0,
    "hotel": 24.0, "hospital": 18.0, "university": 14.0, "school": 10.0,
    # Older / pre-war fabric.  These kinds used to fall through to the generic
    # default, so tenements, lofts and civic halls read like modern blocks.
    "brownstone": 13.0, "tenement": 16.0, "dormitory": 13.0, "cottage": 5.0,
    "farmhouse": 7.0, "barn": 7.0, "farm": 6.0, "stable": 5.0,
    "greenhouse": 4.0, "cowshed": 5.0, "brewery": 12.0, "factory": 10.0,
    "manufacture": 12.0, "civic": 14.0, "government": 20.0, "public": 12.0,
    "museum": 14.0, "train_station": 13.0, "transportation": 10.0,
    "theatre": 12.0, "cinema": 9.0, "stadium": 20.0, "sports_hall": 10.0,
    "chapel": 9.0, "shrine": 6.0, "temple": 12.0, "mosque": 14.0,
    "synagogue": 12.0, "monastery": 14.0, "religious": 12.0,
    "castle": 14.0, "fort": 9.0, "ruins": 5.0, "bunker": 4.0,
    "tower": 30.0, "water_tower": 22.0,
}

ROAD_WIDTHS = {
    "motorway": 14.0, "trunk": 14.0, "primary": 14.0, "secondary": 10.0,
    "tertiary": 8.0, "residential": 6.5, "unclassified": 6.5,
    "service": 4.0, "footway": 2.0, "path": 2.0, "pedestrian": 2.5,
    "steps": 2.0, "track": 2.5, "cycleway": 2.0, "bridleway": 2.0,
    "railway": 3.0,
    "motorway_link": 7.0, "trunk_link": 7.0, "primary_link": 7.0,
    "secondary_link": 6.5, "tertiary_link": 6.5, "living_street": 5.0,
}

DEFAULT_TRAFFIC_COUNT = 3600

CITIES = [
    ("New York", 40.7580, -73.9855), ("London", 51.5138, -0.0984),
    ("Paris", 48.8584, 2.2945), ("Tokyo", 35.6595, 139.7005),
    ("Sydney", -33.8688, 151.2093), ("Singapore", 1.2834, 103.8607),
    ("San Francisco", 37.7946, -122.3999), ("Dubai", 25.1972, 55.2744),
    ("Rio de Janeiro", -22.9068, -43.1729), ("Chicago", 41.8827, -87.6233),
]

DEFAULT_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ACCENT = (0.04, 0.52, 1.0, 1.0)      # Apple system blue
MESH_CACHE_VERSION = 70  # v70: vehicle kind table changed and parked kinds renumbered (invalidates v69)

# Terrain elevation (AWS open terrain tiles, terrarium encoding, no key needed).
TERRAIN_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRAIN_ZOOM = 13
TERRAIN_GRID = 257         # samples across a chunk tile (native 256 px terrain tile + 1)
ROAD_SURFACE = 0.06        # road surface lift above sampled terrain
MOON_COLOR = (0.55, 0.65, 0.95)
NEON_EXPOSURE_STOPS = 0.0  # optional trim on the calibrated cd/m² values in photometry.py
# A tile with no coastline AND no mapped features at all is almost always open sea
# (Overpass has no polygon for the ocean). Set False if you view empty inland deserts.
ASSUME_EMPTY_TILE_IS_SEA = True


@dataclass
class StreamConfig:
    zoom: int = 14
    load_radius: int = 1
    retain_radius: int | None = None
    workers: int = 3
    vram_mb: float = 512.0
    mesh_cache: bool = True
    process_pool: bool = True
    prefetch: bool = True
    prefetch_tiles: float = 2.0
    prefetch_lead: float = 1.0
    max_fps: int = 60

    @property
    def load_r2(self) -> float:
        return (self.load_radius + 0.5) ** 2

    @property
    def retain_r2(self) -> float:
        floor = self.load_radius + (2 if self.prefetch else 1)
        r = self.retain_radius if self.retain_radius is not None else floor
        return (max(r, self.load_radius + 1) + 0.5) ** 2


class SkyClock:
    def __init__(self, hours: float = 13.0, speed: float = 0.05, running: bool = True, *, date=None):
        self.date = parse_date(date)
        self.hours = 0.0
        self.shift(float(hours))
        self.speed = float(speed)
        if not math.isfinite(self.speed):
            raise ValueError('sky speed must be finite')
        self.running = bool(running)

    def advance(self, dt: float):
        dt = float(dt)
        if self.running and math.isfinite(dt):
            self.shift(self.speed * max(0.0, dt))

    def shift(self, hours: float):
        total = self.hours + float(hours)
        if not math.isfinite(total):
            raise ValueError('sky time must be finite')
        days = math.floor(total / 24.0)
        self.date += timedelta(days=days)
        self.hours = total % 24.0

    def toggle(self):
        self.running = not self.running

    @property
    def stamp(self) -> str:
        m = int(round(self.hours * 60.0)) % 1440
        return f"{m // 60:02d}:{m % 60:02d}"


def parse_clock(text: str) -> float:
    t = str(text).strip()
    if ":" in t:
        h, _, m = t.partition(":")
        return (float(h or 0) + float(m or 0) / 60.0) % 24.0
    return float(t) % 24.0
