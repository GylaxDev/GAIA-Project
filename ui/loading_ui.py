"""Full-screen start-up and city-loading screens.

Both share an animated "living Earth" backdrop: a slowly turning globe over a soft, colourful
aurora and a twinkling star field, so the app never opens onto a blank window.  ``draw_boot``
is used while the window, shaders, audio and panels are created; ``draw_travel`` while the
first city streams in.  They only draw: the host supplies progress, step states and log lines.
"""
from __future__ import annotations

import math
import random

from . import theme as T
from .backdrop import Backdrop
from .ui_kit import (activity, alpha, caps, caps_w, check_circle, divider, ease_out_expo, frame, glass, hsv,
                     keycaps, logo_mark, mix, progress, prompt_bar)

APP_NAME = "Project GAIA"
TAGLINE = "Explore the living Earth"

TIPS = (
    "Press Tab to open the Garage and call any car to the road beside you.",
    "Press U to step out and walk the streets on foot.",
    "Press T to cycle the weather and K / L to move the time of day.",
    "Press Esc at any time for the menu: settings, travel and the Garage.",
    "Walk up to any parked car and press Enter to get in. Enter again to park.",
    "Press M to open the world map and double-click anywhere to travel there.",
    "Press F6 for nearby destinations and turn-by-turn driving directions.",
    "Press F9 to hide the interface for clean screenshots.",
    "Downloaded cities are cached, so they load almost instantly next time.",
    "The Settings menu has nine pages: graphics, post-processing, camera, world and more.",
)


# Rough land masses as (latitude, longitude, radius in degrees) blobs - enough to read as Earth.
class LoadingScreen:
    def __init__(self):
        self.backdrop = Backdrop()
        self.shown = 0.0            # eased progress so the bar glides between host updates
        self.t0 = None
        self._tip0 = 0

    # ------------------------------------------------------------------ boot
    def draw_boot(self, ui, w, h, frac, label, version, t, tiles=None):
        from . import ui_kit
        ui_kit.CHROME["version"] = f"v{version}" if version else ""
        r = min(w, h) * 0.24
        self.backdrop.draw(ui, w, h, t, globe=(w / 2, h * 0.38, r), tiles=tiles)
        e = ease_out_expo(t / 0.6)
        cx, ty = w / 2, h * 0.38 + r * 1.28
        caps(ui, APP_NAME, cx, ty + (1 - e) * 10, 34, alpha(T.TEXT_PRIMARY, e), "bold", anchor="center", track=0.18)
        caps(ui, TAGLINE, cx, ty + 48, T.T_CAPTION_2 + 1, alpha(T.TEXT_SECONDARY, e), "medium", anchor="center")

        bw = min(360.0, w - 80)
        bx, by = cx - bw / 2, ty + 84
        self.shown += (frac - self.shown) * 0.18
        progress(ui, bx, by, bw, 4, self.shown, T.SYS_BLUE, shimmer=(t * 0.7) % 1.0)
        caps(ui, label, bx, by + 12, T.T_CAPTION_1, T.TEXT_SECONDARY, "medium")
        caps(ui, f"{int(self.shown * 100)} %", bx + bw, by + 12, T.T_CAPTION_1, T.SYS_BLUE, "medium", anchor="right")
        prompt_bar(ui, w, h, ())

    def draw_travel(self, ui, w, h, t, *, title, coords, elapsed, frac, steps, detail="", hint=None,
                    lines=(), level_colors=None, cancel=True, tiles=None):
        """``steps``: [(label, state)] with state done / active / todo."""
        wide = w >= 900
        gr = min(w * 0.26, h * 0.36)
        self.backdrop.draw(ui, w, h, t, globe=(w - gr - w * 0.08, h * 0.47, gr) if wide else None, tiles=tiles,
                           dim=1.0 if wide else 0.5)

        x = 72.0 if wide else 30.0
        pw = min(560.0, w - 2 * x)
        y = max(40.0, h * 0.13)
        e = ease_out_expo(t / 0.5)
        x -= (1 - e) * 24

        caps(ui, "Travelling to", x, y, T.T_CAPTION_2 + 1, T.SYS_BLUE, "medium")
        size = 44 if caps_w(ui, title, 44, "bold", 0.06) <= pw else 30
        caps(ui, title, x, y + 24, size, T.TEXT_PRIMARY, "bold", track=0.06)
        ui.text(coords, x, y + 24 + size + 12, T.T_BODY, T.TEXT_TERTIARY)

        self.shown += (frac - self.shown) * 0.12
        by = y + 24 + size + 58

        bar_w = pw - 70
        progress(ui, x, by, bar_w, 4, self.shown, T.SYS_BLUE, shimmer=(t * 0.6) % 1.0)
        caps(ui, f"{int(self.shown * 100)} %", x + pw, by - 4, T.T_SUBHEAD, T.SYS_BLUE, "bold", anchor="right")
        caps(ui, f"{int(elapsed)} s elapsed", x, by + 12, T.T_CAPTION_1, T.TEXT_TERTIARY, "medium")

        sy = by + 48
        row = 32.0
        for i, (label, state) in enumerate(steps):
            cy = sy + i * row + row / 2
            if state == "done":
                check_circle(ui, x + 10, cy, 8.5, T.SYS_BLUE)
                col, wt = T.TEXT_SECONDARY, "regular"
            elif state == "active":
                activity(ui, x + 10, cy, 8, t, T.TEXT_PRIMARY)
                col, wt = T.TEXT_PRIMARY, "semibold"
            else:
                ui.box(x + 3, cy - 7, 14, 14, 0.0, None, (1, 1, 1, 0.25), bw=1.0)
                col, wt = T.TEXT_QUATERNARY, "regular"
            ui.text_c(label, x + 34, cy, T.T_SUBHEAD, col, weight=wt)
            if state == "active" and detail:
                lw = ui.text_w(label, T.T_SUBHEAD, weight=wt)
                ui.text_c(ui.fit(detail, T.T_CAPTION_2, pw * 0.45), x + 34 + lw + 14, cy, T.T_CAPTION_2,
                          T.TEXT_SECONDARY, weight="medium")

        ly = sy + len(steps) * row + 10
        if hint:
            ui.text_c(ui.fit(hint, T.T_CAPTION_2, pw), x, ly + 8, T.T_CAPTION_2, T.WARN_AMBER, weight="medium")
            ly += 26
        if lines and h - ly > 170:
            divider(ui, x, ly + 4, pw, T.HAIRLINE)
            ly += 14
            colors = level_colors or {}
            for stamp, level, msg in lines:
                ui.text(ui.fit(f"{stamp}  {msg}", T.T_CAPTION_2, pw), x, ly, T.T_CAPTION_2,
                        colors.get(level, T.TEXT_TERTIARY), weight="regular")
                ly += 18

        # Rotating tip in a framed panel
        tip = TIPS[int(elapsed // 7 + self._tip0) % len(TIPS)]
        tip_w = min(pw + 60, w - 2 * x)
        ty = h - T.HINT_BAR_H - 84
        glass(ui, x, ty, tip_w, 52, T.R_CARD)
        frame(ui, x, ty, tip_w, 52)
        caps(ui, "Tip", x + 18, ty + 26, T.T_CAPTION_2, T.TEXT_TERTIARY, "semibold", vcenter=True)
        ui.text_c(ui.fit(tip, T.T_BODY, tip_w - 90), x + 62, ty + 26, T.T_BODY, T.TEXT_SECONDARY)

        items = ([("Esc", "Cancel")] if cancel else []) + [("F2", "Full log")]
        prompt_bar(ui, w, h, items)
