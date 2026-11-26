"""Real current weather, fetched from Open-Meteo (no API key) on a background thread.

The values that drive the sky are derived from what a weather report actually
says: total cloud cover, precipitation rate, snowfall, WMO condition code,
visibility, wind speed and direction.  Temperatures are carried through for the
HUD so the interface reports the same conditions the renderer is showing.

Network work never runs on the render thread: ``request`` only records where we
are and wakes a worker, and the host reads ``snapshot`` once per frame.
"""
from __future__ import annotations

import json
import math
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://api.open-meteo.com/v1/forecast"
UA = {"User-Agent": "osm3d-viewer/1.0 (educational PyOpenGL demo)"}

# WMO 4677 present-weather codes: (text, kind)
WMO = {
    0: ("Clear", "clear"),
    1: ("Mainly clear", "clear"),
    2: ("Partly cloudy", "cloudy"),
    3: ("Overcast", "overcast"),
    45: ("Fog", "fog"), 48: ("Freezing fog", "fog"),
    51: ("Light drizzle", "drizzle"), 53: ("Drizzle", "drizzle"), 55: ("Heavy drizzle", "drizzle"),
    56: ("Freezing drizzle", "drizzle"), 57: ("Freezing drizzle", "drizzle"),
    61: ("Light rain", "rain"), 63: ("Rain", "rain"), 65: ("Heavy rain", "rain"),
    66: ("Freezing rain", "rain"), 67: ("Freezing rain", "rain"),
    71: ("Light snow", "snow"), 73: ("Snow", "snow"), 75: ("Heavy snow", "snow"), 77: ("Snow grains", "snow"),
    80: ("Light showers", "rain"), 81: ("Showers", "rain"), 82: ("Heavy showers", "rain"),
    85: ("Snow showers", "snow"), 86: ("Heavy snow showers", "snow"),
    95: ("Thunderstorm", "storm"), 96: ("Thunderstorm with hail", "storm"),
    99: ("Severe thunderstorm", "storm"),
}


def describe(code):
    return WMO.get(int(code), ("Cloudy", "cloudy"))


def utc_offset(payload, cur):
    """UTC offset of the observed place, in seconds.  The top-level field is
    authoritative; the local timestamp in ``current`` is the fallback for older
    responses.  Returns ``None`` when neither is usable."""
    offset = (payload or {}).get("utc_offset_seconds")
    if offset is None:
        try:
            local = datetime.fromisoformat(str((cur or {}).get("time")))
            offset = (local - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        except (TypeError, ValueError):
            return None
    try:
        return float(offset)
    except (TypeError, ValueError):
        return None


def local_hours(offset, now=None):
    """Wall-clock hour (0..24) at a place, from its UTC offset in seconds.

    Open-Meteo reports the offset for the observed location; adding it to the
    epoch and taking the hour of the day is the exact local time, DST included.
    """
    if offset is None:
        return None
    try:
        return ((time.time() if now is None else float(now)) + float(offset)) / 3600.0 % 24.0
    except (TypeError, ValueError):
        return None


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def state_from_current(cur):
    """Map an Open-Meteo ``current`` block onto renderer weather values.

    Returns a plain dict; ``WeatherSystem.apply_live`` consumes it.  The mapping
    is physical rather than decorative: cloud cover is the reported fraction,
    rain uses the reported mm/h against a heavy-rain reference, snow uses the
    reported cm/h, fog follows visibility, and wind is the reported vector.
    """
    code = int(cur.get("weather_code", 3) or 3)
    text, kind = describe(code)
    cloud = _clamp(float(cur.get("cloud_cover", 60) or 0) / 100.0)
    precip = max(0.0, float(cur.get("precipitation", 0.0) or 0.0))      # mm in the last hour
    snow_cm = max(0.0, float(cur.get("snowfall", 0.0) or 0.0))          # cm in the last hour
    visibility = float(cur.get("visibility", 24000.0) or 24000.0)       # metres
    wind_kmh = max(0.0, float(cur.get("wind_speed_10m", 0.0) or 0.0))
    wind_dir = float(cur.get("wind_direction_10m", 0.0) or 0.0)         # degrees the wind comes FROM

    rain = _clamp(precip / 6.0) if kind in ("rain", "drizzle", "storm") else 0.0
    if kind == "drizzle":
        rain = max(rain, 0.22)
    if kind == "rain" and precip <= 0.0:
        rain = 0.45
    snow = _clamp(snow_cm / 1.2) if kind == "snow" else 0.0
    if kind == "snow" and snow <= 0.0:
        snow = 0.6
    storm = 0.0
    if kind == "storm":
        storm = 0.75 + 0.25 * _clamp((precip - 2.0) / 8.0)
    elif kind in ("rain", "snow") and precip > 4.0:
        storm = 0.2

    # Haze/fog: 12 km reads as clear air, ~2 km as thick haze.  Kept below the
    # manual rain preset so live conditions rarely swamp the scene.
    fog = _clamp((12000.0 - visibility) / 12000.0) * 0.55 + 0.03
    if kind == "fog":
        fog = max(fog, 0.55)
        cloud = max(cloud, 0.9)

    # Meteorological direction is where it blows from; the vector points where it goes.
    a = math.radians(wind_dir + 180.0)
    scale = 0.30                                    # km/h -> the renderer's wind units
    wx, wy = math.cos(a) * wind_kmh * scale, math.sin(a) * wind_kmh * scale

    temp = cur.get("temperature_2m")
    feels = cur.get("apparent_temperature")
    return {
        "kind": kind, "text": text, "code": code,
        "label": text,
        "cloud": cloud, "rain": rain, "snow": snow, "storm": storm,
        "fog": fog, "wind": (wx, wy), "wind_kmh": wind_kmh, "wind_dir": wind_dir,
        "temp_c": None if temp is None else float(temp),
        "feels_c": None if feels is None else float(feels),
        "humidity": cur.get("relative_humidity_2m"),
        "visibility_km": visibility / 1000.0,
        "is_day": bool(cur.get("is_day", 1)),
        "precipitation_mm": precip,
        "time": cur.get("time"),
        "stamp": time.time(),
    }


class LiveWeather:
    """A cached, self-refreshing current-conditions feed for one location."""

    REFRESH = 600.0          # seconds between automatic refreshes
    MIN_MOVE = 0.01          # degrees (~1 km) before a new place triggers a fetch

    def __init__(self):
        self._lock = threading.Lock()
        self._state = None
        self._error = None
        self._latlon = None
        self._last_try = 0.0
        self._thread = None

    # ------------------------------------------------------------------ api
    def request(self, lat, lon, force=False):
        """Ask for conditions at ``lat``/``lon``.  Never blocks."""
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            return
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return
        now = time.time()
        with self._lock:
            moved = (self._latlon is None
                     or abs(self._latlon[0] - lat) > self.MIN_MOVE
                     or abs(self._latlon[1] - lon) > self.MIN_MOVE)
            stale = now - self._last_try >= self.REFRESH
            if not (force or moved or stale):
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._latlon = (lat, lon)
            self._last_try = now
            self._thread = threading.Thread(target=self._fetch, args=(lat, lon), daemon=True)
            self._thread.start()

    @property
    def state(self):
        with self._lock:
            return dict(self._state) if self._state else None

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def place(self):
        with self._lock:
            return self._latlon

    # --------------------------------------------------------------- worker
    def _fetch(self, lat, lon):
        params = {
            "latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}",
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,"
                       "precipitation,rain,snowfall,weather_code,cloud_cover,wind_speed_10m,"
                       "wind_direction_10m,visibility",
            "timezone": "auto", "wind_speed_unit": "kmh",
        }
        url = API + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=12) as response:
                payload = json.loads(response.read().decode("utf-8"))
            cur = payload.get("current") or {}
            if not cur:
                raise ValueError("no current conditions in the response")
            state = state_from_current(cur)
            # The clock follows the observed place, so carry its UTC offset.
            state["utc_offset"] = utc_offset(payload, cur)
            with self._lock:
                self._state, self._error = state, None
        except Exception as exc:                    # offline, rate-limited, malformed: keep the last good state
            with self._lock:
                self._error = str(exc)[:160]
                self._last_try = time.time() - self.REFRESH + 60.0   # retry in a minute
