"""The weather card: what the sky is doing, in detail, in the glass style.

Live conditions (Open-Meteo, see ``world.live_weather``) show temperature, feels-like,
today's range, an hourly outlook, a sunrise-to-sunset arc and a grid of readings
(wind with direction and gusts, humidity and dew point, pressure, visibility, cloud cover,
UV, precipitation, air quality).  With manual weather it shows the simulated sky instead,
and offers to switch to live.  Everything is optional: a reading the feed did not supply
is simply left out.
"""
from __future__ import annotations

import math
from datetime import datetime

from . import theme as T
from . import icons
from .ui_kit import button, inside, mix, spring, surface

PANEL_W = 352
PAD = 16
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")

# condition kind -> (day icon, night icon, tint)
_KIND = {
    "clear": ("sun", "moon", T.SYS_YELLOW),
    "cloudy": ("cloud-sun", "cloud-moon", T.SYS_ORANGE),
    "overcast": ("cloudy", "cloudy", T.TEXT_SECONDARY),
    "fog": ("cloud-fog", "cloud-fog", T.TEXT_SECONDARY),
    "drizzle": ("cloud-drizzle", "cloud-drizzle", T.SYS_TEAL),
    "rain": ("cloud-rain", "cloud-rain", T.SYS_BLUE),
    "snow": ("cloud-snow", "cloud-snow", (0.80, 0.90, 1.0, 1.0)),
    "storm": ("cloud-lightning", "cloud-lightning", T.SYS_PURPLE),
}


def condition_icon(kind, is_day=True):
    day, night, tint = _KIND.get(kind, _KIND["cloudy"])
    return (day if is_day else night), tint


def compass_point(deg):
    return COMPASS[int(((deg % 360.0) + 11.25) // 22.5) % 16]


# ------------------------------------------------------------------ units
class Units:
    """Metric by default; imperial switches temperature, speed, distance and pressure together."""

    def __init__(self, imperial=False):
        self.imperial = imperial

    def temp(self, c, digits=0):
        if c is None:
            return "--"
        v = c * 9.0 / 5.0 + 32.0 if self.imperial else c
        return f"{round(v):d}\u00b0" if digits == 0 else f"{v:.{digits}f}\u00b0"

    def temp_unit(self):
        return "\u00b0F" if self.imperial else "\u00b0C"

    def speed(self, kmh):
        v = kmh * 0.621371 if self.imperial else kmh
        return f"{round(v):d}", "mph" if self.imperial else "km/h"

    def distance(self, km):
        if self.imperial:
            mi = km * 0.621371
            return (f"{mi:.0f}" if mi >= 10 else f"{mi:.1f}"), "mi"
        return (f"{km:.0f}" if km >= 10 else f"{km:.1f}"), "km"

    def pressure(self, hpa):
        return (f"{hpa * 0.02953:.2f}", "inHg") if self.imperial else (f"{hpa:.0f}", "hPa")

    def rain(self, mm):
        return (f"{mm / 25.4:.2f}", "in/h") if self.imperial else (f"{mm:.1f}", "mm/h")


# ------------------------------------------------------------------ descriptors
def uv_label(uv):
    return ("Low" if uv < 3 else "Moderate" if uv < 6 else "High" if uv < 8 else "Very high" if uv < 11 else "Extreme")


def aqi_label(aqi):
    return ("Good" if aqi <= 50 else "Moderate" if aqi <= 100 else "Unhealthy for sensitive groups" if aqi <= 150
            else "Unhealthy" if aqi <= 200 else "Very unhealthy" if aqi <= 300 else "Hazardous")


def aqi_color(aqi):
    return (T.SYS_GREEN if aqi <= 50 else T.SYS_YELLOW if aqi <= 100 else T.SYS_ORANGE if aqi <= 150 else T.SYS_RED
            if aqi <= 200 else T.SYS_PURPLE)


def pressure_label(hpa):
    return "Low" if hpa < 1005 else "High" if hpa > 1022 else "Steady"


def visibility_label(km):
    return "Clear" if km >= 10 else "Good" if km >= 5 else "Hazy" if km >= 2 else "Poor" if km >= 1 else "Very poor"


def humidity_label(h):
    return "Dry" if h < 30 else "Comfortable" if h <= 60 else "Humid" if h <= 80 else "Very humid"


def wind_label(kmh):
    return ("Calm" if kmh < 2 else "Light" if kmh < 20 else "Breezy" if kmh < 40 else "Windy" if kmh < 62 else "Gale")


def _hm(iso):
    """Minutes since local midnight from an Open-Meteo ``YYYY-MM-DDTHH:MM`` string, or None."""
    try:
        d = datetime.fromisoformat(str(iso))
        return d.hour * 60 + d.minute
    except (TypeError, ValueError):
        return None


def _clock(iso):
    m = _hm(iso)
    return "--:--" if m is None else f"{m // 60:02d}:{m % 60:02d}"


# ------------------------------------------------------------------ data
def build_info(weather, live_feed, clock_stamp):
    """Everything the pill and panel need, from the ``WeatherSystem`` and ``LiveWeather``.

    Returns a plain dict; ``live`` tells whether real readings back it.  ``clock_stamp`` is the
    scene clock ("HH:MM"), used for the day/night icon when the feed is not live.
    """
    live = bool(weather.live and weather.live_state)
    st = dict(weather.live_state) if live else {}
    mode = weather.mode
    sim_kind = {"clear": "clear", "overcast": "overcast", "rain": "rain", "storm": "storm", "snow": "snow"}.get(mode, "clear")
    try:
        hh = int(clock_stamp[:2])
        sim_day = 6 <= hh < 19
    except (TypeError, ValueError):
        sim_day = True
    wind = weather.wind
    info = {
        "live": live,
        "kind": st.get("kind", sim_kind), "text": st.get("text") or weather.label.split("  \u00b7")[0],
        "is_day": st.get("is_day", sim_day),
        "error": getattr(live_feed, "error", None) if live_feed is not None else None,
        "fetching": bool(weather.live and not weather.live_state),
        "sim": {"cloud": float(weather.cloud), "rain": float(weather.rain), "snow": float(weather.snow),
                "storm": float(weather.storm), "fog": float(weather.fog), "wet": float(weather.wetness),
                "wind_kmh": float(math.hypot(float(wind[0]), float(wind[1]))) / 0.30},
    }
    info.update({k: st.get(k) for k in (
        "temp_c", "feels_c", "temp_max_c", "temp_min_c", "humidity", "dew_c", "pressure_hpa", "wind_kmh", "gust_kmh",
        "wind_dir", "visibility_km", "cloud_pct", "uv", "precipitation_mm", "pop_max", "aqi", "pm25", "sunrise",
        "sunset", "hourly", "time")})
    return info


def pill_text(info, units):
    """Short text for the HUD pill: ``"7\u00b0  Partly cloudy"`` (live) or the simulated label."""
    if info["live"] and info.get("temp_c") is not None:
        return units.temp(info["temp_c"]), info["text"]
    return None, info["text"] + ("  \u00b7  live\u2026" if info["fetching"] else "")


# ------------------------------------------------------------------ drawing
def _tile(ui, x, y, w, h, icon, label, value, unit="", sub="", sub_color=T.TEXT_TERTIARY, angle=0.0, tint=None):
    ui.box(x, y, w, h, T.R_TILE + 2, T.GROUP, (1.0, 1.0, 1.0, 0.06))
    icons.draw(ui, icon, x + 17, y + 17, 15, tint or T.TEXT_TERTIARY, angle=angle)
    ui.text_c(label, x + 33, y + 17, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")
    vw = ui.text(value, x + 12, y + 29, T.T_TITLE_2 - 2, T.TEXT_PRIMARY, weight="semibold")
    if unit:
        ui.text(unit, x + 12 + vw + 4, y + 35, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")
    if sub:
        ui.text(ui.fit(sub, T.T_CAPTION_2, w - 24), x + 12, y + h - 22, T.T_CAPTION_2, sub_color, weight="regular")


def _live_tiles(info, u):
    tiles = []
    if info.get("wind_kmh") is not None:
        v, unit = u.speed(info["wind_kmh"])
        d = info.get("wind_dir")
        sub = wind_label(info["wind_kmh"])
        if d is not None:
            sub = f"{compass_point(d)}  \u00b7  {sub}"
        if info.get("gust_kmh") and info["gust_kmh"] > info["wind_kmh"] + 3:
            g, _ = u.speed(info["gust_kmh"])
            sub = f"{compass_point(d) if d is not None else 'Gusts'}  \u00b7  gusts {g}"
        tiles.append(dict(icon="navigation-2", label="Wind", value=v, unit=unit, sub=sub,
                          angle=(d + 180.0) if d is not None else 0.0))
    if info.get("humidity") is not None:
        sub = humidity_label(info["humidity"]) + (f"  \u00b7  dew point {u.temp(info['dew_c'])}" if info.get("dew_c") is not None else "")
        tiles.append(dict(icon="droplets", label="Humidity", value=f"{info['humidity']:.0f}", unit="%", sub=sub))
    if info.get("pressure_hpa") is not None:
        v, unit = u.pressure(info["pressure_hpa"])
        tiles.append(dict(icon="gauge", label="Pressure", value=v, unit=unit, sub=pressure_label(info["pressure_hpa"])))
    if info.get("visibility_km") is not None:
        v, unit = u.distance(info["visibility_km"])
        tiles.append(dict(icon="eye", label="Visibility", value=v, unit=unit, sub=visibility_label(info["visibility_km"])))
    if info.get("cloud_pct") is not None:
        tiles.append(dict(icon="cloud", label="Cloud cover", value=f"{info['cloud_pct']:.0f}", unit="%",
                          sub="Clear" if info["cloud_pct"] < 20 else "Partly cloudy" if info["cloud_pct"] < 70 else "Mostly cloudy"))
    if info.get("uv") is not None:
        tiles.append(dict(icon="sun-medium", label="UV index", value=f"{info['uv']:.0f}" if info["uv"] >= 10 else f"{info['uv']:.1f}",
                          sub=uv_label(info["uv"]) + (f"  \u00b7  peak {info['uv_max']:.0f}" if info.get("uv_max") is not None else "")))
    if info.get("precipitation_mm") is not None:
        v, unit = u.rain(info["precipitation_mm"])
        sub = f"{info['pop_max']:.0f}% chance today" if info.get("pop_max") is not None else "Last hour"
        tiles.append(dict(icon="umbrella", label="Precipitation", value=v, unit=unit, sub=sub))
    if info.get("aqi") is not None:
        tiles.append(dict(icon="leaf", label="Air quality", value=f"{info['aqi']:.0f}", unit="AQI",
                          sub=aqi_label(info["aqi"]), sub_color=aqi_color(info["aqi"])))
    return tiles


def _sim_tiles(info, u):
    s = info["sim"]
    v, unit = u.speed(s["wind_kmh"])
    tiles = [dict(icon="cloud", label="Cloud cover", value=f"{s['cloud'] * 100:.0f}", unit="%", sub="Simulated"),
             dict(icon="navigation-2", label="Wind", value=v, unit=unit, sub=wind_label(s["wind_kmh"]), angle=135.0)]
    if s["snow"] > 0.02:
        tiles.append(dict(icon="snowflake", label="Snowfall", value=f"{s['snow'] * 100:.0f}", unit="%", sub="Intensity"))
    else:
        tiles.append(dict(icon="cloud-rain", label="Rainfall", value=f"{s['rain'] * 100:.0f}", unit="%", sub="Intensity"))
    tiles.append(dict(icon="cloud-fog", label="Haze", value=f"{s['fog'] * 100:.0f}", unit="%", sub="Fog density"))
    tiles.append(dict(icon="droplet", label="Wet surfaces", value=f"{s['wet'] * 100:.0f}", unit="%", sub="Roads and ground"))
    tiles.append(dict(icon="zap", label="Storm", value=f"{s['storm'] * 100:.0f}", unit="%", sub="Lightning risk"))
    return tiles


def _hourly(ui, x, y, w, info, u):
    rows = info.get("hourly") or []
    if len(rows) < 3:
        return False
    picks = rows[0:12:2][:6]
    cw = w / len(picks)
    for i, r in enumerate(picks):
        cx = x + cw * (i + 0.5)
        label = "Now" if i == 0 else _clock(r["time"])[:2]
        ui.text_c(label, cx, y + 8, T.T_CAPTION_2, T.TEXT_PRIMARY if i == 0 else T.TEXT_TERTIARY,
                  anchor="center", weight="semibold" if i == 0 else "medium")
        name, tint = condition_icon(r["kind"], r["is_day"])
        icons.draw(ui, name, cx, y + 34, 22, tint)
        ui.text_c(u.temp(r["temp_c"]), cx, y + 59, T.T_BODY, T.TEXT_PRIMARY, anchor="center", weight="semibold")
        pop = r.get("pop")
        if pop is not None and pop >= 10:
            ui.text_c(f"{pop:.0f}%", cx, y + 76, T.T_CAPTION_1, T.SYS_TEAL, anchor="center", weight="medium")
    return True


def _sun_arc(ui, x, y, w, h, info):
    rise, sset = _hm(info.get("sunrise")), _hm(info.get("sunset"))
    if rise is None or sset is None or sset <= rise:
        return False
    now = _hm(info.get("time"))
    cx, base, rx, ry = x + w / 2, y + h - 26, w / 2 - 34, h - 44
    pts = [(cx - rx * math.cos(math.pi * i / 40), base - ry * math.sin(math.pi * i / 40)) for i in range(41)]
    frac = None if now is None else (now - rise) / float(sset - rise)
    for i in range(40):
        lit = frac is not None and 0.0 <= frac and (i / 40.0) <= frac
        ui.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], 2.0, T.SYS_YELLOW if lit else (1, 1, 1, 0.20))
    ui.line(x + 12, base, x + w - 12, base, 1.0, T.SEPARATOR)
    if frac is not None and 0.0 <= frac <= 1.0:
        px, py = cx - rx * math.cos(math.pi * frac), base - ry * math.sin(math.pi * frac)
        ui.circle(px, py, 9, (T.SYS_YELLOW[0], T.SYS_YELLOW[1], T.SYS_YELLOW[2], 0.28))
        ui.circle(px, py, 5, T.SYS_YELLOW)
    icons.draw(ui, "sunrise", x + 24, base + 14, 14, T.TEXT_TERTIARY)
    ui.text_c(_clock(info["sunrise"]), x + 38, base + 14, T.T_CAPTION_2, T.TEXT_SECONDARY, weight="medium")
    icons.draw(ui, "sunset", x + w - 24, base + 14, 14, T.TEXT_TERTIARY)
    ui.text_c(_clock(info["sunset"]), x + w - 38, base + 14, T.T_CAPTION_2, T.TEXT_SECONDARY, anchor="right", weight="medium")
    mins = sset - rise
    ui.text_c(f"{mins // 60}h {mins % 60:02d}m of daylight", cx, base + 14, T.T_CAPTION_2, T.TEXT_TERTIARY,
              anchor="center", weight="medium")
    return True


def draw_panel(ui, x, y, info, units, mouse, max_h):
    """Draw the weather card with its top-right corner at (x + PANEL_W, y).

    Returns ``(rect, hits)``; ``hits`` is ``[(action, rect), ...]`` for the chrome's hit testing.
    Sections that do not fit in ``max_h`` are dropped in this order: sun arc, hourly, extra tiles.
    """
    w = PANEL_W
    live = info["live"]
    tiles = _live_tiles(info, units) if live else _sim_tiles(info, units)
    has_hourly = live and len(info.get("hourly") or []) >= 3
    has_sun = live and info.get("sunrise") and info.get("sunset")
    head_h, hour_h, sun_h, foot_h, gap = 92, 84, 92, 48, 8
    tile_h, tile_gap = 74, 8

    def total(n_tiles, hourly, sun):
        rows = (n_tiles + 1) // 2
        h = PAD + head_h + (gap + hour_h if hourly else 0) + (gap + sun_h if sun else 0)
        h += gap + rows * tile_h + max(0, rows - 1) * tile_gap + gap + foot_h
        return h

    n = len(tiles)
    show_sun, show_hourly = bool(has_sun), bool(has_hourly)
    while total(n, show_hourly, show_sun) > max_h:
        if show_sun:
            show_sun = False
        elif show_hourly:
            show_hourly = False
        elif n > 2:
            n -= 2
        else:
            break
    tiles = tiles[:n]
    H = total(n, show_hourly, show_sun)
    x = x - w
    surface(ui, x, y, w, H, T.R_PANEL, T.SHEET, T.BORDER_DEFAULT, T.SHADOW_3, specular=True)
    hits = [("weather_panel", (x, y, w, H))]
    ix, iw = x + PAD, w - 2 * PAD

    # header ---------------------------------------------------------------
    name, tint = condition_icon(info["kind"], info["is_day"])
    icons.draw(ui, name, ix + 22, y + PAD + 28, 40, tint)
    tx = ix + 58
    if info.get("temp_c") is not None:
        tw = ui.text(units.temp(info["temp_c"]), tx, y + PAD + 2, T.T_DISPLAY + 6, T.TEXT_PRIMARY, weight="semibold")
        ui.text(units.temp_unit().lstrip("\u00b0"), tx + tw + 2, y + PAD + 10, T.T_BODY, T.TEXT_TERTIARY, weight="medium")
        ui.text(ui.fit(info["text"], T.T_CALLOUT, iw - 58 - 70), tx, y + PAD + 50, T.T_CALLOUT, T.TEXT_PRIMARY, weight="semibold")
    else:
        title = "Fetching\u2026" if info["fetching"] else info["text"]
        ui.text(ui.fit(title, T.T_TITLE_3, iw - 58 - 70), tx, y + PAD + 10, T.T_TITLE_3, T.TEXT_PRIMARY, weight="semibold")
    sub = []
    if info.get("feels_c") is not None:
        sub.append(f"Feels like {units.temp(info['feels_c'])}")
    if info.get("temp_max_c") is not None and info.get("temp_min_c") is not None:
        sub.append(f"H {units.temp(info['temp_max_c'])}  L {units.temp(info['temp_min_c'])}")
    if sub:
        ui.text("  \u00b7  ".join(sub), tx, y + PAD + 72, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")
    elif not live:
        ui.text("Manual weather \u00b7 T cycles presets", tx, y + PAD + 50, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")

    # close + unit toggle (top right)
    cr = (x + w - PAD - 28, y + PAD - 2, 28, 28)
    hv = spring(ui, ("weather", "close"), float(inside(cr, *mouse)), T.SPRING_GENTLE)
    ui.box(*cr, 14, mix((1, 1, 1, 0.10), (1, 1, 1, 0.26), hv))
    icons.draw(ui, "x", cr[0] + 14, cr[1] + 14, 14, T.TEXT_PRIMARY)
    hits.append(("weather_close", cr))
    ur = (cr[0] - 8 - 52, cr[1], 52, 28)
    hv = spring(ui, ("weather", "units"), float(inside(ur, *mouse)), T.SPRING_GENTLE)
    ui.box(*ur, 14, mix((1, 1, 1, 0.10), (1, 1, 1, 0.22), hv), (1, 1, 1, 0.10))
    ui.text_c(f"{units.temp_unit()}", ur[0] + 26, ur[1] + 14, T.T_CAPTION_2, T.TEXT_PRIMARY, anchor="center", weight="semibold")
    hits.append(("weather_units", ur))
    chip_txt = "Live" if live else "Simulated"
    cw = ui.text_w(chip_txt, T.T_CAPTION_1, weight="semibold") + 22
    col = T.SYS_GREEN if live else T.TEXT_TERTIARY
    cx0, cy0 = x + w - PAD - cw, y + PAD + 28
    ui.box(cx0, cy0, cw, 22, 11, (col[0], col[1], col[2], 0.16), (col[0], col[1], col[2], 0.30))
    ui.circle(cx0 + 10, cy0 + 11, 3, col)
    ui.text_c(chip_txt, cx0 + 17, cy0 + 11, T.T_CAPTION_1, col, weight="semibold")

    cy = y + PAD + head_h
    # hourly ---------------------------------------------------------------
    if show_hourly:
        cy += gap
        ui.box(ix, cy, iw, hour_h, T.R_TILE + 2, T.GROUP, (1, 1, 1, 0.06))
        _hourly(ui, ix + 6, cy + 4, iw - 12, info, units)
        cy += hour_h
    if show_sun:
        cy += gap
        ui.box(ix, cy, iw, sun_h, T.R_TILE + 2, T.GROUP, (1, 1, 1, 0.06))
        _sun_arc(ui, ix, cy + 4, iw, sun_h - 8, info)
        cy += sun_h
    # tiles ----------------------------------------------------------------
    cy += gap
    tw = (iw - tile_gap) / 2
    for i, t in enumerate(tiles):
        tx = ix + (i % 2) * (tw + tile_gap)
        ty = cy + (i // 2) * (tile_h + tile_gap)
        _tile(ui, tx, ty, tw, tile_h, t["icon"], t["label"], t["value"], t.get("unit", ""), t.get("sub", ""),
              t.get("sub_color", T.TEXT_TERTIARY), t.get("angle", 0.0))
    rows = (len(tiles) + 1) // 2
    cy += rows * tile_h + max(0, rows - 1) * tile_gap + gap
    # footer: live toggle ----------------------------------------------------
    br = (ix, cy + 6, iw, 36)
    if live:
        button(ui, br, "Use manual weather", mouse, key="wx_live", kind="secondary", icon="rotate-ccw", size=T.T_FOOTNOTE)
    else:
        button(ui, br, "Use live weather", mouse, key="wx_live", kind="primary", icon="cloud-sun", size=T.T_FOOTNOTE)
    hits.insert(0, ("weather_live", br))
    hits.insert(0, ("weather_units", ur))
    hits.insert(0, ("weather_close", cr))
    if info.get("error") and weather_err(info):
        ui.text_c("Offline \u00b7 showing last reading" if live else "Offline", x + w / 2, y + H - 8, T.T_CAPTION_1,
                  T.WARN_AMBER, anchor="center", weight="medium")
    return (x, y, w, H), hits


def weather_err(info):
    return bool(info.get("error"))
