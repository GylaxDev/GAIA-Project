"""Offline solar/lunar ephemerides and J2000 -> local east/north/up rotation.

Sun: Meeus/NOAA low-precision solar coordinates. Moon: Keplerian elements with
the principal Schlyter perturbations and topocentric parallax. Suitable for a
visual sky (roughly sub-degree lunar accuracy), not navigation or eclipse work.
No timezone, network, system-clock or rendering dependencies enter the math.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date, datetime, timedelta
import math

import numpy as np

J2000 = 2451545.0
EARTH_RADIUS_KM = 6378.137
REFERENCE_DATE = Date(2026, 4, 22)


def parse_date(value=None):
    if value is None:
        return Date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, Date):
        return value
    return Date.fromisoformat(str(value).strip())


def julian_day(date, hours=0., lon_deg=0., utc_offset=None):
    """Local date/hour -> UTC JD. No offset means *local mean solar time*.

    Longitude / 15 is intentionally not guessed as a political timezone or DST.
    Supply the actual UTC offset for civil-time astronomy.
    """
    date = parse_date(date)
    hours, lon_deg = float(hours), float(lon_deg)
    offset = lon_deg / 15. if utc_offset is None else float(utc_offset)
    if not all(math.isfinite(x) for x in (hours, lon_deg, offset)):
        raise ValueError('time, longitude and UTC offset must be finite')
    return date.toordinal() + 1721424.5 + (hours - offset) / 24.


def local_sidereal_angle(jd, lon_deg=0.):
    """Greenwich mean sidereal time + east-positive longitude, in radians."""
    t = (float(jd) - J2000) / 36525.
    degrees = (280.46061837 + 360.98564736629 * (float(jd) - J2000)
               + .000387933 * t * t - t * t * t / 38710000. + float(lon_deg))
    return math.radians(degrees % 360.)


def equatorial_vector(ra_hours, dec_deg):
    ra, dec = np.radians(np.asarray(ra_hours, np.float64) * 15.), np.radians(np.asarray(dec_deg, np.float64))
    c = np.cos(dec)
    return np.stack((c * np.cos(ra), c * np.sin(ra), np.sin(dec)), axis=-1)


def equatorial_to_enu(lat_deg, sidereal_angle):
    lat, theta = math.radians(float(lat_deg)), float(sidereal_angle)
    sl, cl, st, ct = math.sin(lat), math.cos(lat), math.sin(theta), math.cos(theta)
    return np.array([[-st, ct, 0.], [-sl * ct, -sl * st, cl], [cl * ct, cl * st, sl]], np.float64)


def precession_matrix(jd):
    """IAU 1976 precession: mean J2000 coordinates to mean equator of date."""
    t = (float(jd) - J2000) / 36525.
    scale = math.pi / (180. * 3600.)
    zeta = (2306.2181 * t + .30188 * t * t + .017998 * t ** 3) * scale
    z = (2306.2181 * t + 1.09468 * t * t + .018203 * t ** 3) * scale
    theta = (2004.3109 * t - .42665 * t * t - .041833 * t ** 3) * scale

    def rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    c, s = math.cos(theta), math.sin(theta)
    ry = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])
    return rz(z) @ ry @ rz(zeta)


def _sin(degrees):
    return math.sin(math.radians(degrees % 360.))


def _cos(degrees):
    return math.cos(math.radians(degrees % 360.))


def _obliquity(jd):
    t = (jd - J2000) / 36525.
    return math.radians(23. + (26. + (21.448 - t * (46.815 + t * (.00059 - .001813 * t))) / 60.) / 60.)


def solar_equatorial(jd):
    t = (float(jd) - J2000) / 36525.
    mean_long = (280.46646 + t * (36000.76983 + .0003032 * t)) % 360.
    anomaly = (357.52911 + t * (35999.05029 - .0001537 * t)) % 360.
    center = ((1.914602 - t * (.004817 + .000014 * t)) * _sin(anomaly)
              + (.019993 - .000101 * t) * _sin(2. * anomaly) + .000289 * _sin(3. * anomaly))
    omega = 125.04 - 1934.136 * t
    longitude = math.radians(mean_long + center - .00569 - .00478 * _sin(omega))
    obliquity = _obliquity(jd) + math.radians(.00256 * _cos(omega))
    return np.array([math.cos(longitude), math.cos(obliquity) * math.sin(longitude),
                     math.sin(obliquity) * math.sin(longitude)])


def lunar_equatorial(jd):
    """Geocentric Moon position in equatorial-of-date Earth radii."""
    d = float(jd) - 2451543.5
    node, inclination = 125.1228 - .0529538083 * d, 5.1454
    periapsis, anomaly = 318.0634 + .1643573223 * d, (115.3654 + 13.0649929509 * d) % 360.
    eccentricity = .054900
    m = math.radians(anomaly)
    e = m + eccentricity * math.sin(m) * (1. + eccentricity * math.cos(m))
    for _ in range(5):
        e -= (e - eccentricity * math.sin(e) - m) / (1. - eccentricity * math.cos(e))
    x, y = math.cos(e) - eccentricity, math.sqrt(1. - eccentricity ** 2) * math.sin(e)
    radius = 60.2666 * math.hypot(x, y)
    v = math.degrees(math.atan2(y, x)) + periapsis
    ex = _cos(node) * _cos(v) - _sin(node) * _sin(v) * _cos(inclination)
    ey = _sin(node) * _cos(v) + _cos(node) * _sin(v) * _cos(inclination)
    ez = _sin(v) * _sin(inclination)
    longitude, latitude = math.degrees(math.atan2(ey, ex)), math.degrees(math.atan2(ez, math.hypot(ex, ey)))
    solar_anomaly = (356.0470 + .9856002585 * d) % 360.
    solar_long = 282.9404 + .0000470935 * d + solar_anomaly
    moon_long = node + periapsis + anomaly
    elongation, argument = moon_long - solar_long, moon_long - node
    longitude += (-1.274 * _sin(anomaly - 2. * elongation) + .658 * _sin(2. * elongation)
                  - .186 * _sin(solar_anomaly) - .059 * _sin(2. * anomaly - 2. * elongation)
                  - .057 * _sin(anomaly - 2. * elongation + solar_anomaly)
                  + .053 * _sin(anomaly + 2. * elongation) + .046 * _sin(2. * elongation - solar_anomaly)
                  + .041 * _sin(anomaly - solar_anomaly) - .035 * _sin(elongation)
                  - .031 * _sin(anomaly + solar_anomaly) - .015 * _sin(2. * argument - 2. * elongation)
                  + .011 * _sin(anomaly - 4. * elongation))
    latitude += (-.173 * _sin(argument - 2. * elongation) - .055 * _sin(anomaly - argument - 2. * elongation)
                 - .046 * _sin(anomaly + argument - 2. * elongation) + .033 * _sin(argument + 2. * elongation)
                 + .017 * _sin(2. * anomaly + argument))
    radius += -.58 * _cos(anomaly - 2. * elongation) - .46 * _cos(2. * elongation)
    xe, ye, ze = radius * _cos(longitude) * _cos(latitude), radius * _sin(longitude) * _cos(latitude), radius * _sin(latitude)
    obliquity = _obliquity(jd)
    return np.array([xe, ye * math.cos(obliquity) - ze * math.sin(obliquity),
                     ye * math.sin(obliquity) + ze * math.cos(obliquity)])


def lunar_brightness(phase):
    """Full-moon-relative integrated brightness (the Moon is not Lambertian)."""
    angle = math.degrees(math.acos(np.clip(2. * float(phase) - 1., -1., 1.)))
    return 10. ** (-.4 * (.026 * angle + 4e-9 * angle ** 4))


@dataclass(frozen=True)
class CelestialState:
    julian_day: float
    sidereal_angle: float
    equatorial_to_world: np.ndarray
    sun_dir: np.ndarray
    moon_dir: np.ndarray
    moon_phase: float
    moon_brightness: float
    moon_radius: float


def celestial_state(date, hours, lat_deg, lon_deg=0., utc_offset=None, altitude_m=0.):
    latitude, longitude, altitude = float(lat_deg), float(lon_deg), float(altitude_m)
    if not all(math.isfinite(x) for x in (latitude, longitude, altitude)) or not -90. <= latitude <= 90.:
        raise ValueError('observer coordinates must be finite, with latitude in [-90, 90]')
    jd = julian_day(date, hours, longitude, utc_offset)
    theta = local_sidereal_angle(jd, longitude)
    rotation = equatorial_to_enu(latitude, theta)
    sun_eq, moon_eq = solar_equatorial(jd), lunar_equatorial(jd)
    phase = .5 * (1. - float(np.clip(sun_eq @ (moon_eq / np.linalg.norm(moon_eq)), -1., 1.)))
    observer = rotation[2] * (1. + max(-500., altitude) / (EARTH_RADIUS_KM * 1000.))
    moon_local = rotation @ (moon_eq - observer)
    distance = float(np.linalg.norm(moon_local))
    return CelestialState(jd, theta, rotation @ precession_matrix(jd), rotation @ sun_eq,
                          moon_local / distance, phase, lunar_brightness(phase),
                          math.asin(1737.4 / (distance * EARTH_RADIUS_KM)))
