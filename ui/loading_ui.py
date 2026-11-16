"""Full-screen start-up and city-loading screens.

Both share an animated night-skyline backdrop (three parallax layers, twinkling windows, a
star field) so the app never opens onto a blank window.  ``draw_boot`` is used while the
window, shaders, audio and panels are created; ``draw_travel`` while the first city streams
in.  They only draw: the host supplies progress, step states and log lines.
"""
from __future__ import annotations

import math
import random

from . import theme as T
from .ui_kit import alpha, ease_out, keycaps, logo_mark, mix, prompt_bar

TIPS = (
    "Press Tab to open the Garage and call any car to the road beside you.",
    "Hold Q while on foot to profile the people around you.",
    "Press T to cycle the weather and K / L to move the time of day.",
    "Press Esc at any time for the menu: settings, travel and the Garage.",
    "Walk up to any parked car and press Enter to get in. Enter again to park.",
    "Press M to open the world map and double-click anywhere to travel there.",
    "Press F6 for nearby destinations and turn-by-turn driving directions.",
    "Press F9 to hide the interface for clean screenshots.",
    "Downloaded cities are cached, so they load almost instantly next time.",
    "The Settings menu has nine pages: graphics, post-processing, camera, world and more.",
)


class Backdrop:
    """Procedural skyline: deterministic, so it never pops between frames."""

    LAYERS = (  # (speed px/s, base colour, min h, max h, min w, max w, windows)
        (5.0, (0.075, 0.105, 0.17), 0.10, 0.34, 34, 80, False),
        (12.0, (0.045, 0.066, 0.108), 0.14, 0.46, 40, 96, True),
        (26.0, (0.024, 0.034, 0.058), 0.18, 0.56, 50, 118, True),
    )
    PATTERN = 2200.0

    def __init__(self):
        rng = random.Random(0x05A3D)
        self.layers = []
        for speed, col, h0, h1, w0, w1, windows in self.LAYERS:
            blds, x = [], 0.0
            while x < self.PATTERN:
                bw = rng.uniform(w0, w1)
                bh = rng.uniform(h0, h1)
                lights = ([(rng.random(), rng.random(), rng.random()) for _ in range(int(bw * bh * 60 / 40))]
                          if windows else [])
                blds.append((x, bw, bh, lights))
                x += bw + rng.uniform(0, 8)
            self.layers.append((speed, col, windows, blds))
        self.stars = [(rng.random(), rng.random() * 0.55, rng.uniform(0.5, 1.6), rng.uniform(0, 6.28))
                      for _ in range(70)]

    def draw(self, ui, w, h, t, dim=1.0):
        horizon = h * 0.80
        ui.gradient(0, 0, w, h, (0.012, 0.022, 0.048, 1.0), (0.05, 0.085, 0.15, 1.0))
        for sx, sy, size, ph in self.stars:
            tw = 0.55 + 0.45 * math.sin(t * 1.3 + ph)
            ui.circle(sx * w, sy * h, size, (0.8, 0.88, 1.0, 0.45 * tw * dim))
        # horizon glow
        ui.gradient(0, horizon - h * 0.30, w, h * 0.30, (T.ACCENT[0], T.ACCENT[1], T.ACCENT[2], 0.0),
                    (T.ACCENT[0], T.ACCENT[1], T.ACCENT[2], 0.16 * dim))
        for speed, col, windows, blds in self.layers:
            off = (t * speed) % self.PATTERN
            for rep in (-1, 0, 1):
                base = rep * self.PATTERN - off
                if base > w or base + self.PATTERN < 0:
                    continue
                for bx, bw, bh, lights in blds:
                    x = base + bx
                    if x > w or x + bw < 0:
                        continue
                    top = horizon - bh * h
                    ui.rect(x, top, bw, horizon - top + 2, (col[0], col[1], col[2], 1.0))
                    if windows:
                        self._windows(ui, x, top, bw, horizon - top, lights, t)
        ui.rect(0, horizon, w, h - horizon, (0.012, 0.018, 0.03, 1.0))
        ui.gradient(0, horizon, w, 26, (0.0, 0.0, 0.0, 0.25), (0.0, 0.0, 0.0, 0.0))

    @staticmethod
    def _windows(ui, x, top, bw, bh, lights, t):
        for a, b, c in lights[:int(bw * bh / 220)]:
            wx, wy = x + 4 + a * (bw - 10), top + 6 + b * (bh - 14)
            blink = 0.5 + 0.5 * math.sin(t * (0.4 + c) + c * 40)
            if blink > 0.35:
                ui.rect(wx, wy, 2.4, 3.2, (1.0, 0.82, 0.45, 0.10 + 0.34 * blink))


class LoadingScreen:
    def __init__(self):
        self.backdrop = Backdrop()
        self.shown = 0.0            # eased progress so the bar glides between host updates
        self.t0 = None
        self._tip0 = 0

    # ------------------------------------------------------------------ boot
    def draw_boot(self, ui, w, h, frac, label, version, t):
        self.backdrop.draw(ui, w, h, t)
        ui.rect(0, 0, w, h, (0.0, 0.0, 0.0, 0.18))
        cx, cy = w / 2, h * 0.40
        e = ease_out(t / 0.6)
        logo_mark(ui, cx - 36, cy - 108 + (1 - e) * 14, 72.0, t)
        ui.text("OSM 3D", cx, cy - 18 + (1 - e) * 14, 46, alpha(T.TEXT, e), anchor="center", weight="bold")
        ui.text("City explorer", cx, cy + 40, T.T_LABEL, alpha(T.TEXT_3, e), anchor="center", weight="medium")
        bw = min(420.0, w - 80)
        bx, by = cx - bw / 2, cy + 104
        self.shown += (frac - self.shown) * 0.18
        ui.bar(bx, by, bw, 5, self.shown, color=T.SIGNAL, shimmer=t)
        ui.text(label, bx, by + 16, T.T_BODY, T.TEXT_2, weight="medium")
        ui.text(f"{int(self.shown * 100)}%", bx + bw, by + 16, T.T_BODY, T.TEXT_3, anchor="right", weight="semibold")
        ui.text(f"v{version}", cx, h - 34, T.T_SMALL, T.TEXT_4, anchor="center", weight="medium")

    # ---------------------------------------------------------------- travel
    def draw_travel(self, ui, w, h, t, *, title, coords, elapsed, frac, steps, detail="", hint=None,
                    lines=(), level_colors=None, cancel=True):
        """``steps``: [(label, state)] with state done / active / todo."""
        self.backdrop.draw(ui, w, h, t)
        ui.gradient(0, 0, w, h, (0.0, 0.0, 0.0, 0.55), (0.0, 0.0, 0.0, 0.10))
        wide = w >= 900
        x = 72.0 if wide else 30.0
        pw = min(560.0, w - 2 * x)
        y = max(40.0, h * 0.14)
        e = ease_out(t / 0.5)
        x -= (1 - e) * 24
        ui.text("TRAVELLING TO", x, y, T.T_CAPTION, T.SIGNAL, weight="semibold")
        name = title if ui.text_w(title, 44, weight="bold") <= pw else ui.fit(title, 34, pw, weight="bold")
        size = 44 if name is title else 34
        ui.text(name, x, y + 20, size, T.TEXT, weight="bold")
        ui.text(coords, x, y + 20 + size + 10, T.T_BODY, T.TEXT_3, weight="medium")
        self.shown += (frac - self.shown) * 0.12
        by = y + 20 + size + 52
        ui.bar(x, by, pw - 60, 6, self.shown, color=T.SIGNAL, shimmer=t)
        ui.text_c(f"{int(self.shown * 100)}%", x + pw, by + 3, T.T_LABEL, T.TEXT, anchor="right", weight="semibold")
        ui.text(f"{int(elapsed)} s", x, by + 18, T.T_SMALL, T.TEXT_4, weight="medium")
        sy = by + 48
        row = 30.0
        for i, (label, state) in enumerate(steps):
            cy = sy + i * row + row / 2
            if state == "done":
                ui.tick(x + 10, cy, 9, T.OK)
                col, wt = T.TEXT_2, "regular"
            elif state == "active":
                ui.spinner(x + 10, cy, 6, t, color=T.SIGNAL, dots=8)
                col, wt = T.TEXT, "semibold"
            else:
                ui.circle(x + 10, cy, 3, (1, 1, 1, 0.25))
                col, wt = T.TEXT_4, "regular"
            ui.text_c(label, x + 34, cy, T.T_BASE + 1, col, weight=wt)
            if state == "active" and detail:
                ui.text_c(ui.fit(detail, T.T_SMALL, pw * 0.45), x + pw, cy, T.T_SMALL, T.TEXT_3,
                          anchor="right", weight="medium")
        ly = sy + len(steps) * row + 10
        if hint:
            ui.text_c(ui.fit(hint, T.T_SMALL, pw), x, ly + 8, T.T_SMALL, T.WARN, weight="medium")
            ly += 26
        if lines and h - ly > 150:
            ui.divider(x, ly + 4, pw)
            ly += 14
            colors = level_colors or {}
            for stamp, level, msg in lines:
                ui.text(ui.fit(f"{stamp}  {msg}", T.T_SMALL, pw), x, ly, T.T_SMALL,
                        colors.get(level, T.TEXT_3), weight="regular")
                ly += 18
        # rotating tip
        tip = TIPS[int(elapsed // 7 + self._tip0) % len(TIPS)]
        ui.container(x, h - 128, min(pw + 60, w - 2 * x), 52, "panel", r=T.R_PANEL - 2, shadow=False)
        ui.text("TIP", x + 18, h - 128 + 19, T.T_CAPTION, T.SIGNAL, weight="semibold")
        ui.text_c(ui.fit(tip, T.T_BODY, min(pw + 60, w - 2 * x) - 76), x + 52, h - 128 + 26, T.T_BODY, T.TEXT_2,
                  weight="medium")
        items = ([("Esc", "Cancel")] if cancel else []) + [("F2", "Full log")]
        prompt_bar(ui, w, h, items)
