"""Global constants, StreamConfig, SkyClock and time parsing."""
from __future__ import annotations
from dataclasses import dataclass

EARTH_RADIUS = 6_378_137.0

BG_COLOR = (0.45, 0.60, 0.85)

KIND_DEFAULT_HEIGHT = {
    "house": 6.0, "detached": 6.0, "semidetached_house": 7.0, "terrace": 8.0,
    "bungalow": 4.0, "garage": 3.0, "garages": 3.0, "shed": 2.8, "hut": 2.8,
    "cabin": 4.0, "roof": 3.5, "carport": 3.0, "kiosk": 3.0, "apartments": 15.0,
    "residential": 10.0, "commercial": 14.0, "office": 20.0, "retail": 8.0,
    "industrial": 9.0, "warehouse": 8.0, "church": 16.0, "cathedral": 32.0,
    "hotel": 24.0, "hospital": 18.0, "university": 14.0, "school": 10.0,
}

ROAD_WIDTHS = {
    "motorway": 14.0, "trunk": 12.0, "primary": 10.0, "secondary": 8.0,
    "tertiary": 7.0, "residential": 6.0, "unclassified": 5.0,
    "service": 4.0, "footway": 2.5, "path": 2.0, "pedestrian": 4.0,
    "railway": 3.0,  # Added railways
}

CITIES = [
    ("New York", 40.7580, -73.9855), ("London", 51.5138, -0.0984),
    ("Paris", 48.8584, 2.2945), ("Tokyo", 35.6595, 139.7005),
    ("Sydney", -33.8688, 151.2093), ("Singapore", 1.2834, 103.8607),
    ("San Francisco", 37.7946, -122.3999), ("Dubai", 25.1972, 55.2744),
    ("Rio de Janeiro", -22.9068, -43.1729), ("Chicago", 41.8827, -87.6233),
]

DEFAULT_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
ACCENT = (0.36, 0.80, 1.0, 1.0)
MESH_CACHE_VERSION = 11  # v11: ocean polygons built from natural=coastline
MOON_COLOR = (0.55, 0.65, 0.95)
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

    @property
    def load_r2(self) -> float:
        return (self.load_radius + 0.5) ** 2

    @property
    def retain_r2(self) -> float:
        floor = self.load_radius + (2 if self.prefetch else 1)
        r = self.retain_radius if self.retain_radius is not None else floor
        return (max(r, self.load_radius + 1) + 0.5) ** 2


class SkyClock:
    def __init__(self, hours: float = 13.0, speed: float = 0.05, running: bool = True):
        self.hours = float(hours) % 24.0
        self.speed = float(speed)
        self.running = bool(running)

    def advance(self, dt: float):
        if self.running:
            self.hours = (self.hours + self.speed * max(0.0, dt)) % 24.0

    def shift(self, hours: float):
        self.hours = (self.hours + hours) % 24.0

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