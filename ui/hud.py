"""Small, self-contained HUD cards in the glass style.

Stand-alone HUD cards (one class per card, simple ``draw`` methods), built on
the same widgets as the rest of the interface.  The map HUD (``MapsChrome``) already provides
the key-hint bar, status card and speed card; these wrappers just expose them as stand-alone
objects for code that prefers a class per card.
"""
from __future__ import annotations

import math

from . import theme as T
from .maps_ui import MapsChrome
from .ui_kit import glass, progress, toast


class SpeedCard:
    """Speedometer: big number, unit, gear badge, turn signals and a load bar."""

    def __init__(self):
        self._chrome = MapsChrome()

    def draw(self, ui, x, y, vehicle_name, speed, gear, state, signal=0, braking=False, rpm_frac=0.0,
             unit="km/h"):
        return self._chrome.speed_card(ui, x, y, vehicle_name, speed, gear, state, signal, braking, rpm_frac, unit)


class StatusCard:
    """Stacked text rows with an optional progress bar, anchored by its bottom edge."""

    def __init__(self):
        self._chrome = MapsChrome()

    def draw(self, ui, x, bottom, rows, progress=None, shimmer=0.0, min_w=220):
        return self._chrome.status_card(ui, x, bottom, rows, progress, shimmer, min_w)


class Compass:
    """A round compass: ticks rotate with the heading, a red N marks north."""

    def draw(self, ui, cx, cy, r, heading):
        glass(ui, cx - r, cy - r, 2 * r, 2 * r, r, "hud")
        for i in range(24):
            a = math.radians(i * 15.0) - heading
            r0, r1 = r - 5.0, r - (9.0 if i % 6 == 0 else 7.0)
            ui.line(cx + r0 * math.sin(a), cy - r0 * math.cos(a), cx + r1 * math.sin(a), cy - r1 * math.cos(a),
                    1.4 if i % 6 == 0 else 1.0, (1, 1, 1, 0.7 if i % 6 == 0 else 0.28))
        a = -heading
        ui.text_c("N", cx + (r - 17) * math.sin(a), cy - (r - 17) * math.cos(a), T.T_CAPTION_2, T.SYS_RED,
                  anchor="center", weight="bold")


class TimeWeatherCard:
    """Clock with a one-line conditions summary."""

    def draw(self, ui, x, y, w, time_str, weather_label, temperature=None):
        sub = weather_label if temperature is None else f"{weather_label} · {temperature}"
        glass(ui, x, y, w, 56, 20, "hud")
        ui.text(time_str, x + 18, y + 8, T.T_HEAD, T.TEXT_PRIMARY, weight="semibold")
        ui.text(ui.fit(sub, T.T_CAPTION_2, w - 36), x + 18, y + 33, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")
        return w, 56


class Toast:
    """Transient capsule notification, fading in and out with ``progress`` (0 -> 1 over its life)."""

    @staticmethod
    def draw(ui, w, h, text, color=None, progress=0.0):
        k = min(1.0, progress * 8.0, (1.0 - progress) * 5.0) if 0.0 < progress < 1.0 else (0.0 if progress >= 1.0 else progress * 8.0)
        if k <= 0.0:
            return
        toast(ui, w / 2, 24 + (1.0 - k) * -10.0, text, ok=color in (None, T.OK_GREEN))
