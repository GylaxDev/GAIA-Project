"""Interface chrome shared by the world-map and 3D views, in the graphite design language.

* Map screen: a floating sidebar container (brand, Search / Guides / Directions,
  an expandable list of places, a travel call-to-action), a grouped control
  cluster (map style / locate / full screen), a compass and a zoom pair.
* 3D screen: a place card with search, a time + conditions card, the same
  controls, plus the HUD building blocks (key-hint bar, status card, speed card).

Every surface is an ``Overlay2D.container`` (gradient fill + 1 px hairline), every
label is Inter, and all coordinates are logical pixels.  ``layout`` is the single
source of truth for geometry: drawing and hit-testing both use it, so they can
never disagree.
"""
from __future__ import annotations

import math
import time

from . import theme as T
from ..core.config import ACCENT

WHITE = T.TEXT
DIM = T.TEXT_3
HOVER = (1.0, 1.0, 1.0, 0.09)

TIPS = {
    "sidebar": "Toggle sidebar", "map": "Map style", "locate": "Recenter",
    "fullscreen": "Full screen  F11", "compass": "Reset north",
    "zoom_in": "Zoom in", "zoom_out": "Zoom out", "search": "Search  /",
}


def _in(r, x, y):
    return r[0] <= x <= r[0] + r[2] and r[1] <= y <= r[1] + r[3]


def _wrap(ui):          # kept for older call sites; sizes are real pixels now
    return ui


# ------------------------------------------------------------------ keycaps
def _key_tokens(key):
    """'W A S D' -> W A S D;  'WASD / arrows' -> WASD, arrows;  'Shift' -> Shift."""
    out = []
    for part in key.split(" / "):
        part = part.strip()
        bits = part.split()
        if len(bits) > 1 and all(len(b) == 1 for b in bits):
            out.extend(bits)
        else:
            out.append(part)
    return out


def keycaps(ui, x, y, key, size=11, dry=False):
    """Draw ``key`` as one or more keycap chips starting at x, y.  Returns the width."""
    cx = x
    for i, tok in enumerate(_key_tokens(key)):
        if i:
            cx += 4
        if dry:
            tw = ui.text_w(tok, size, weight="semibold")
            cx += max(size + 10, tw + 12)
        else:
            cx += ui.kbd(cx, y, tok, size)
    return cx - x


class MapsChrome:
    SIDEBAR_W = 264
    BTN = 40
    ROW_H = 40
    CITY_H = 32
    TIP_DELAY = 0.45

    def __init__(self):
        self.sidebar_open = True
        self.guides_open = False
        self.navigation_active = False
        self._tip_name = None
        self._tip_t = 0.0

    # ------------------------------------------------------------------ layout
    def layout(self, w, h, mode, n_cities=0):
        """Ordered list of (action, rect).  Earlier entries win hit tests."""
        out = []
        B, M = self.BTN, T.MARGIN
        gx, gy = w - M - B, M
        out.append(("map", (gx, gy, B, B)))
        out.append(("locate", (gx, gy + B, B, B)))
        out.append(("fullscreen", (gx, gy + 2 * B, B, B)))
        out.append(("compass", (gx, gy + 3 * B + 12, B, B)))
        zx, zy = w - M - B, h - M - 2 * B
        out.append(("zoom_in", (zx, zy, B, B)))
        out.append(("zoom_out", (zx, zy + B, B, B)))
        if mode == "map":
            sw, sx, sy, sh = self.SIDEBAR_W, M, M, h - 2 * M
            if self.sidebar_open:
                out.append(("sidebar", (sx + sw - 12 - 32, sy + 12, 32, 32)))
                rx, rw, ry = sx + 12, sw - 24, sy + 64
                step = self.ROW_H + 4
                out.append(("search", (rx, ry, rw, self.ROW_H)))
                out.append(("guides", (rx, ry + step, rw, self.ROW_H)))
                out.append(("directions", (rx, ry + 2 * step, rw, self.ROW_H)))
                go_y = sy + sh - 12 - 64
                route_y = ry + 3 * step + 12
                if self.navigation_active and not self.guides_open and go_y > route_y + 124:
                    out.append(("route_cancel", (rx, route_y + 84, rw, 28)))
                if self.guides_open and n_cities:
                    cy = ry + 3 * step + 30
                    avail = go_y - 10 - cy
                    ch = max(18.0, min(float(self.CITY_H), avail / n_cities))
                    for i in range(n_cities):
                        out.append((f"city:{i}", (rx, cy + i * ch, rw, ch - 2)))
                out.append(("go", (rx, go_y, rw, 64)))
                out.append(("sidebar_bg", (sx, sy, sw, sh)))
            else:
                out.append(("sidebar", (M, M, 40, 40)))
        else:
            out.append(("search", (M, M, 52, 56)))
        return out

    def hit(self, pos, w, h, mode, n_cities=0):
        for name, r in self.layout(w, h, mode, n_cities):
            if _in(r, *pos):
                return name
        return None

    def over_ui(self, pos, w, h, mode, n_cities=0):
        return self.hit(pos, w, h, mode, n_cities) is not None

    def clickable(self, pos, w, h, mode, n_cities=0):
        """True over something that reacts to a click (for the hand cursor)."""
        act = self.hit(pos, w, h, mode, n_cities)
        return act is not None and act != "sidebar_bg"

    # ------------------------------------------------------------------ icons
    @staticmethod
    def icon_search(ui, cx, cy, s, col):
        ui.ring(cx - s * 0.10, cy - s * 0.10, s * 0.34, 1.8, col)
        ui.line(cx + s * 0.15, cy + s * 0.15, cx + s * 0.40, cy + s * 0.40, 2.0, col)

    @staticmethod
    def icon_guides(ui, cx, cy, s, col):
        q, g = s * 0.30, s * 0.07
        for dx in (-1, 1):
            for dy in (-1, 1):
                x = cx + (g if dx > 0 else -g - q)
                y = cy + (g if dy > 0 else -g - q)
                ui.rrect(x, y, q, q, 3, col)

    @staticmethod
    def icon_directions(ui, cx, cy, s, col):
        ui.line(cx - s * 0.22, cy + s * 0.36, cx - s * 0.22, cy - s * 0.02, 2.0, col)
        ui.line(cx - s * 0.22, cy - s * 0.02, cx + s * 0.10, cy - s * 0.02, 2.0, col)
        ui.tri((cx + s * 0.08, cy - s * 0.26), (cx + s * 0.08, cy + s * 0.22),
               (cx + s * 0.40, cy - s * 0.02), col)

    @staticmethod
    def icon_sidebar(ui, cx, cy, s, col):
        ui.box(cx - s * 0.42, cy - s * 0.32, s * 0.84, s * 0.64, 3.0, None, col, bw=1.8)
        ui.line(cx - s * 0.12, cy - s * 0.32, cx - s * 0.12, cy + s * 0.32, 1.8, col)

    @staticmethod
    def icon_map(ui, cx, cy, s, col):
        a, b = s * 0.42, s * 0.14
        top = [(cx - a, cy - s * 0.26), (cx - b, cy - s * 0.38), (cx + b, cy - s * 0.26), (cx + a, cy - s * 0.38)]
        bot = [(cx - a, cy + s * 0.38), (cx - b, cy + s * 0.26), (cx + b, cy + s * 0.38), (cx + a, cy + s * 0.26)]
        ui.outline_poly(top + list(reversed(bot)), 1.8, col)
        ui.line(top[1][0], top[1][1], bot[1][0], bot[1][1], 1.6, col)
        ui.line(top[2][0], top[2][1], bot[2][0], bot[2][1], 1.6, col)

    @staticmethod
    def icon_locate(ui, cx, cy, s, col):
        t = (cx + s * 0.34, cy - s * 0.34)
        a = (cx - s * 0.30, cy + s * 0.10)
        b = (cx - s * 0.10, cy + s * 0.30)
        m = (cx - s * 0.08, cy + s * 0.08)
        ui.tri(t, a, m, col)
        ui.tri(t, m, b, col)

    @staticmethod
    def icon_fullscreen(ui, cx, cy, s, col, exit_=False):
        d, k = s * 0.36, s * 0.18
        for sx in (-1, 1):
            for sy in (-1, 1):
                px, py = cx + sx * d, cy + sy * d
                ex = -sx if exit_ else sx
                ey = -sy if exit_ else sy
                if exit_:
                    px, py = cx + sx * (d - k * 0.4), cy + sy * (d - k * 0.4)
                ui.line(px, py, px - ex * k, py, 2.0, col)
                ui.line(px, py, px, py - ey * k, 2.0, col)

    @staticmethod
    def icon_arrow_up_right(ui, cx, cy, s, col):
        ui.line(cx - s * 0.30, cy + s * 0.30, cx + s * 0.30, cy - s * 0.30, 2.0, col)
        ui.line(cx - s * 0.02, cy - s * 0.30, cx + s * 0.30, cy - s * 0.30, 2.0, col)
        ui.line(cx + s * 0.30, cy - s * 0.30, cx + s * 0.30, cy + s * 0.02, 2.0, col)

    @staticmethod
    def _plus(ui, cx, cy, s, col):
        ui.line(cx - s * 0.30, cy, cx + s * 0.30, cy, 2.0, col)
        ui.line(cx, cy - s * 0.30, cx, cy + s * 0.30, 2.0, col)

    @staticmethod
    def _minus(ui, cx, cy, s, col):
        ui.line(cx - s * 0.30, cy, cx + s * 0.30, cy, 2.0, col)

    # ----------------------------------------------------------------- tooltips
    def _track_hover(self, ui, name):
        if name != self._tip_name:
            self._tip_name, self._tip_t = name, 0.0
        else:
            self._tip_t += ui.dt

    def _tooltip(self, ui, name, rect, w, side="left"):
        if name != self._tip_name or self._tip_t < self.TIP_DELAY or name not in TIPS:
            return
        text = TIPS[name]
        tw = ui.text_w(text, T.T_SMALL, weight="medium")
        bw, bh = tw + 20, 28
        fade = min(1.0, (self._tip_t - self.TIP_DELAY) * 8.0)
        if side == "left":
            x, y = rect[0] - 8 - bw, rect[1] + (rect[3] - bh) / 2
        else:
            x, y = rect[0] + rect[2] + 8, rect[1] + (rect[3] - bh) / 2
        x = max(8, x)
        ui.box(round(x), round(y), round(bw), bh, T.R_BUTTON, ((0.20, 0.20, 0.21, 0.97 * fade),
                                                                (0.11, 0.11, 0.12, 0.97 * fade)),
               (1, 1, 1, 0.30 * fade), shadow=12, shadow_dy=4, shadow_alpha=0.4 * fade)
        ui.text_c(text, x + bw / 2, y + bh / 2, T.T_SMALL, (1, 1, 1, fade), anchor="center",
                  weight="medium")

    # ------------------------------------------------------------------ pieces
    def _button_group(self, ui, rects, names, mouse, icons, active=None):
        x, y, w, _ = rects[0]
        total = sum(r[3] for r in rects)
        ui.container(x, y, w, total, "button")
        for i, (r, ic, nm) in enumerate(zip(rects, icons, names)):
            hv = ui.anim(("hv", nm), 1.0 if _in(r, *mouse) else 0.0, 20.0)
            if hv > 0.01:
                ui.box(r[0] + 3, r[1] + 3, r[2] - 6, r[3] - 6, T.R_BUTTON - 2, (1, 1, 1, 0.13 * hv))
            if i > 0:
                ui.divider(r[0] + 8, r[1], r[2] - 16, (1, 1, 1, 0.16))
            col = ACCENT if active == i else WHITE
            ic(ui, r[0] + r[2] / 2, r[1] + r[3] / 2, 20, col)

    def _compass(self, ui, r, mouse, north_vec=(0.0, -1.0)):
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        hv = ui.anim(("hv", "compass"), 1.0 if _in(r, *mouse) else 0.0, 20.0)
        ui.container(r[0], r[1], r[2], r[3], "button", r=r[2] / 2, hover=hv)
        ang = math.atan2(north_vec[0], -north_vec[1])
        rad = r[2] / 2
        for i in range(24):
            a = ang + i * math.pi / 12.0
            r0, r1 = rad - 4.0, rad - (7.5 if i % 6 == 0 else 6.0)
            ui.line(cx + r0 * math.sin(a), cy - r0 * math.cos(a),
                    cx + r1 * math.sin(a), cy - r1 * math.cos(a),
                    1.3 if i % 6 == 0 else 1.0, (1, 1, 1, 0.65 if i % 6 == 0 else 0.28))
        ui.text_c("N", cx, cy - 1, T.T_SMALL, (1.0, 0.36, 0.33, 1.0), anchor="center", weight="bold")

    # ------------------------------------------------------------------ shared cluster
    def draw_controls(self, ui, w, h, mode, mouse, fullscreen=False, north_vec=(0.0, -1.0)):
        rects = {n: r for n, r in self.layout(w, h, mode)}
        self._button_group(
            ui, [rects["map"], rects["locate"], rects["fullscreen"]], ("map", "locate", "fullscreen"),
            mouse, [self.icon_map, self.icon_locate,
                    (lambda u, x, y, s, c: self.icon_fullscreen(u, x, y, s, c, fullscreen))])
        self._compass(ui, rects["compass"], mouse, north_vec)
        zi, zo = rects["zoom_in"], rects["zoom_out"]
        self._button_group(ui, [zi, zo], ("zoom_in", "zoom_out"), mouse, [self._plus, self._minus])

    def _finish_tooltips(self, ui, w, h, mode, mouse):
        name = self.hit(mouse, w, h, mode)
        self._track_hover(ui, name if name in TIPS else None)
        if name in TIPS:
            rect = {n: r for n, r in self.layout(w, h, mode)}.get(name)
            if rect:
                self._tooltip(ui, name, rect, w, "right" if name in ("sidebar", "search") else "left")

    def pill(self, ui, x, y, text, size=T.T_BODY, color=T.TEXT_2, pad=12, bold=False):
        """A small one-line container; returns (width, height)."""
        tw = ui.text_w(text, size, weight="semibold" if bold else "medium")
        hgt = size + 16
        ui.container(x, y, tw + 2 * pad, hgt, "panel", r=T.R_PILL, shadow=False)
        ui.text_c(text, x + pad, y + hgt / 2, size, color, weight="semibold" if bold else "medium")
        return tw + 2 * pad, hgt

    def _field(self, ui, x, y, w, h, query, placeholder="Search places"):
        ui.container(x, y, w, h, "field", focus=True)
        self.icon_search(ui, x + 18, y + h / 2, 16, DIM)
        on = int(time.time() * 2) % 2 == 0
        cx = x + 34
        if query:
            shown = ui.fit(query, T.T_BASE, w - 56)
            cx += ui.text_c(shown, cx, y + h / 2, T.T_BASE, WHITE, weight="medium") + 1
        else:
            ui.text_c(placeholder, cx + 3, y + h / 2, T.T_BASE, T.TEXT_3)
        if on:                       # drawn, not a glyph: Inter has no box-drawing characters
            ui.rect(round(cx), round(y + h / 2 - 8), 1.5, 16, (1, 1, 1, 0.92))

    # ------------------------------------------------------------------ map screen
    def draw_map_ui(self, ui, w, h, mouse, cities, marker, typing, query, fullscreen,
                    tiles_dark=True, can_enter_3d=False, navigation=None):
        rects = {n: r for n, r in self.layout(w, h, "map", len(cities))}
        M, sw = T.MARGIN, self.SIDEBAR_W
        if self.sidebar_open:
            sb = rects["sidebar_bg"]
            ui.container(*sb, "panel", r=14)
            # brand row
            ui.box(sb[0] + 16, sb[1] + 14, 30, 30, 8, T.ACCENT_FILL, (1, 1, 1, 0.9))
            self.icon_locate(ui, sb[0] + 31, sb[1] + 29, 16, WHITE)
            ui.text_c("Maps", sb[0] + 58, sb[1] + 29, T.T_HEAD, WHITE, weight="semibold")
            tg = rects["sidebar"]
            hv = ui.anim(("hv", "sidebar"), 1.0 if _in(tg, *mouse) else 0.0, 20.0)
            ui.box(*tg, T.R_BUTTON, (1, 1, 1, 0.10 * hv))
            self.icon_sidebar(ui, tg[0] + 16, tg[1] + 16, 18, T.TEXT_2)
            rows = (("search", "Search", self.icon_search), ("guides", "Guides", self.icon_guides),
                    ("directions", "Directions", self.icon_directions))
            for name, label, ic in rows:
                r = rects[name]
                hv = ui.anim(("hv", name), 1.0 if _in(r, *mouse) else 0.0, 20.0)
                if name == "search" and typing:
                    self._field(ui, *r, query)
                    continue
                on = name == "guides" and self.guides_open
                if hv > 0.01 or on:
                    ui.box(*r, T.R_BUTTON, (1, 1, 1, max(0.10 * hv, 0.07 if on else 0.0)))
                ui.box(r[0] + 6, r[1] + 6, 28, 28, 7, T.INSET_FILL, T.BORDER_SOFT)
                ic(ui, r[0] + 20, r[1] + 20, 16, T.TEXT)
                ui.text_c(label, r[0] + 46, r[1] + r[3] / 2, T.T_BASE, WHITE, weight="medium")
            if self.guides_open and cities:
                c0 = rects["city:0"]
                ui.text("TRAVEL GUIDES", c0[0] + 8, c0[1] - 22, T.T_CAPTION, T.TEXT_3, weight="semibold")
                for i, (name, _, _) in enumerate(cities):
                    r = rects[f"city:{i}"]
                    sel = marker is not None and marker[2] == name
                    hv = ui.anim(("hv", f"city{i}"), 1.0 if _in(r, *mouse) else 0.0, 20.0)
                    if hv > 0.01:
                        ui.box(*r, T.R_BUTTON - 1, (ACCENT[0], ACCENT[1], ACCENT[2], 0.30 * hv))
                    elif sel:
                        ui.box(*r, T.R_BUTTON - 1, (1, 1, 1, 0.08))
                    ui.text_c(name, r[0] + 12, r[1] + r[3] / 2, T.T_BODY, T.TEXT, weight="regular")
                    ui.text_c(str((i + 1) % 10), r[0] + r[2] - 12, r[1] + r[3] / 2, T.T_SMALL,
                              T.TEXT_3, anchor="right", weight="medium")
            if navigation is not None and navigation.destination is not None and "route_cancel" in rects:
                rc = rects["route_cancel"]
                cy = rc[1] - 84
                ui.text("DRIVING ROUTE", rc[0] + 8, cy, T.T_CAPTION, T.TEXT_3, weight="semibold")
                ui.text(ui.fit(navigation.destination.name, T.T_LABEL, rc[2] - 16, weight="semibold"),
                        rc[0] + 8, cy + 20, T.T_LABEL, T.TEXT, weight="semibold")
                ui.text(ui.fit(navigation.instruction, T.T_SMALL, rc[2] - 16),
                        rc[0] + 8, cy + 43, T.T_SMALL, T.TEXT_2)
                if navigation.route is not None:
                    distance = (f"{navigation.remaining / 1000.:.1f} km" if navigation.remaining >= 1000.
                                else f"{int(round(navigation.remaining / 10.) * 10)} m")
                    minutes = max(1, int(math.ceil(navigation.eta_seconds / 60.)))
                    summary = "Route complete" if navigation.status == "arrived" else f"{minutes} min · {distance} · estimated"
                    ui.text(summary, rc[0] + 8, cy + 62, T.T_SMALL, T.TEXT_3)
                ui.container(*rc, "button", r=7, shadow=False, hover=float(_in(rc, *mouse)))
                ui.text_c("End route", rc[0] + rc[2] * .5, rc[1] + rc[3] * .5,
                          T.T_SMALL, T.TEXT_2, anchor="center", weight="medium")
            # travel call-to-action
            r = rects["go"]
            hv = ui.anim(("hv", "go"), 1.0 if _in(r, *mouse) else 0.0, 20.0)
            if marker:
                ui.container(*r, "accent", r=T.R_PANEL - 2)
                if hv > 0.01:
                    ui.box(*r, T.R_PANEL - 2, (1, 1, 1, 0.12 * hv))
                ui.text("Travel to", r[0] + 14, r[1] + 12, T.T_SMALL, (1, 1, 1, 0.80), weight="medium")
                ui.text(ui.fit(marker[2], T.T_LABEL + 1, r[2] - 64), r[0] + 14, r[1] + 31, T.T_LABEL + 1,
                        WHITE, weight="semibold")
                self.icon_arrow_up_right(ui, r[0] + r[2] - 24, r[1] + r[3] / 2, 18, WHITE)
            else:
                ui.container(*r, "inset", r=T.R_PANEL - 2, hover=hv)
                ui.text("Pick a place", r[0] + 14, r[1] + 13, T.T_LABEL, WHITE, weight="semibold")
                ui.text(ui.fit("Click the map \u00b7 double-click to travel", T.T_SMALL, r[2] - 28),
                        r[0] + 14, r[1] + 36, T.T_SMALL, T.TEXT_3)
            tgt = rects["sidebar"]
        else:
            r = rects["sidebar"]
            hv = ui.anim(("hv", "sidebar"), 1.0 if _in(r, *mouse) else 0.0, 20.0)
            ui.container(*r, "button", hover=hv)
            self.icon_sidebar(ui, r[0] + r[2] / 2, r[1] + r[3] / 2, 20, WHITE)
            if typing:
                ui.container(r[0] + r[2] + 8, r[1], 280, r[3], "panel", r=T.R_BUTTON)
                self._field(ui, r[0] + r[2] + 12, r[1] + 4, 272, r[3] - 8, query)
        self.draw_controls(ui, w, h, "map", mouse, fullscreen)
        # attribution, tucked left of the zoom pair
        att = "\u00a9 OpenStreetMap contributors"
        aw = ui.text_w(att, T.T_CAPTION, weight="medium") + 20
        ax, ay = w - T.MARGIN - self.BTN - 10 - aw, h - T.MARGIN - 24
        ui.container(ax, ay, aw, 24, "panel", r=T.R_BUTTON - 2, shadow=False)
        ui.text_c(att, ax + aw / 2, ay + 12, T.T_CAPTION, T.TEXT_3, anchor="center", weight="medium")
        self._finish_tooltips(ui, w, h, "map", mouse)

    # ------------------------------------------------------------------ 3D screen
    def draw_scene_ui(self, ui, w, h, mouse, label, coords, clock, sub, typing, query,
                      fullscreen, north_vec):
        M, B, ch = T.MARGIN, self.BTN, 56
        # time + conditions card, left of the control cluster
        sub_fit = ui.fit(sub, T.T_SMALL, min(360, max(120, w * 0.30)))
        tcw = int(max(ui.text_w(clock, T.T_HEAD, weight="semibold"),
                      ui.text_w(sub_fit, T.T_SMALL, weight="medium")) + 32)
        tx = w - M - B - 12 - tcw
        ui.container(tx, M, tcw, ch, "panel", r=T.R_PANEL)
        ui.text(clock, tx + 16, M + 8, T.T_HEAD, WHITE, weight="semibold")
        ui.text(sub_fit, tx + 16, M + 33, T.T_SMALL, T.TEXT_3, weight="medium")
        # place card with the search button
        cw = int(max(200, min(380, tx - 12 - M)))
        ui.container(M, M, cw, ch, "panel", r=T.R_PANEL)
        sr = (M + 8, M + 10, 36, 36)
        hv = ui.anim(("hv", "search"), 1.0 if _in((M, M, 52, ch), *mouse) else 0.0, 20.0)
        ui.container(*sr, "button", hover=hv, shadow=False)
        self.icon_search(ui, sr[0] + 18, sr[1] + 18, 17, WHITE)
        if typing:
            self._field(ui, M + 52, M + 8, cw - 60, ch - 16, query)
        else:
            ui.text(ui.fit(label or "Scene", T.T_LABEL + 1, cw - 72, weight="semibold"),
                    M + 58, M + 9, T.T_LABEL + 1, WHITE, weight="semibold")
            ui.text(coords, M + 58, M + 32, T.T_SMALL, T.TEXT_3, weight="medium")
        self.draw_controls(ui, w, h, "scene", mouse, fullscreen, north_vec)
        self._finish_tooltips(ui, w, h, "scene", mouse)

    # ------------------------------------------------------------------ HUD blocks
    def hint_bar(self, ui, x, y, max_w, items):
        """A row of keycap + label pairs in one container.  Items that don't fit are
        dropped from the end, but the last item (the shortcuts entry) always stays.
        Returns the width drawn."""
        size = T.T_SMALL
        widths = []
        for key, label in items:
            widths.append(keycaps(ui, 0, 0, key, 11, dry=True) + 6 + ui.text_w(label, size, weight="medium"))
        gap, pad = 14, 12
        keep = list(range(len(items)))
        while len(keep) > 1 and sum(widths[i] for i in keep) + gap * (len(keep) - 1) + 2 * pad > max_w:
            keep.pop(len(keep) - 2 if len(keep) > 2 else 0)
        total = sum(widths[i] for i in keep) + gap * (len(keep) - 1) + 2 * pad
        hh = 36
        ui.container(x, y, total, hh, "panel", r=T.R_PANEL - 2, shadow=False)
        cx = x + pad
        for i in keep:
            key, label = items[i]
            kw = keycaps(ui, cx, y + (hh - 21) / 2, key, 11)
            ui.text_c(label, cx + kw + 6, y + hh / 2, size, T.TEXT_2, weight="medium")
            cx += widths[i] + gap
        return total

    def status_card(self, ui, x, bottom, rows, progress=None, shimmer=0.0, min_w=0):
        """rows: [(text, colour, size, weight)].  Anchored by its bottom edge."""
        pad, lead = 14, 6
        tw = max(ui.text_w(t, s, weight=wt) for t, _, s, wt in rows)
        w = max(min_w, tw + 2 * pad)
        h = sum(s + lead for _, _, s, _ in rows) + 2 * pad - lead + (6 if progress is not None else 0)
        y = bottom - h
        ui.container(x, y, w, h, "panel", r=T.R_PANEL - 2, shadow=False)
        cy = y + pad
        for t, col, s, wt in rows:
            ui.text(t, x + pad, cy, s, col, weight=wt)
            cy += s + lead
        if progress is not None:
            ui.bar(x + pad, y + h - pad + 2, w - 2 * pad, 3, progress, ACCENT, shimmer)
        return w, h

    def speed_card(self, ui, x, y, name, kmh, gear, state, signal=0, braking=False, frac=0.0, unit="km/h"):
        """Driving instrument cluster.  ``signal``: -1 left, +1 right, 0 off."""
        w, h = 236, 124
        ui.container(x, y, w, h, "panel", r=T.R_PANEL)
        ui.text(name.upper(), x + 16, y + 14, T.T_CAPTION, T.TEXT_3, weight="semibold")
        sp = str(kmh)
        sw = ui.text(sp, x + 16, y + 30, T.T_DISPLAY, WHITE, weight="bold")
        ui.text(unit, x + 16 + sw + 6, y + 30 + 24, T.T_BASE, T.TEXT_3, weight="medium")
        gb = (x + w - 16 - 44, y + 14, 44, 44)
        ui.container(*gb, "button", shadow=False)
        ui.text_c(gear, gb[0] + gb[2] / 2, gb[1] + gb[3] / 2, T.T_HEAD, ACCENT, anchor="center",
                  weight="bold")
        blink = int(time.time() * 2.6) % 2 == 0
        for sgn, ax in ((-1, gb[0] - 24), (1, gb[0] - 8)):
            lit = signal == sgn and blink
            col = (1.0, 0.72, 0.18, 1.0) if lit else (1, 1, 1, 0.14)
            cy = gb[1] + gb[3] / 2
            if sgn < 0:
                ui.tri((ax + 12, cy - 7), (ax + 12, cy + 7), (ax, cy), col)
            else:
                ui.tri((ax, cy - 7), (ax, cy + 7), (ax + 12, cy), col)
        ui.bar(x + 16, y + 78, w - 32, 4, max(0.0, min(1.0, frac)),
               (1.0, 0.36, 0.33, 1.0) if braking else ACCENT)
        ui.text_c(ui.fit(state, T.T_SMALL, w - 32), x + 16, y + 98, T.T_SMALL,
                  T.WARN if state.startswith("Blocked") else T.TEXT_3, weight="medium")
        return w, h

    def chips(self, ui, right, y, items, min_x=0):
        """Right-aligned row of small state chips [(text, colour)], most important first.
        Chips that would run past ``min_x`` are dropped.  Returns the width used."""
        x = right
        for text, col in items:
            tw = ui.text_w(text, T.T_CAPTION, weight="medium") + 20
            if x - tw < min_x:
                break
            x -= tw
            ui.container(x, y, tw, 24, "panel", r=7, shadow=False)
            ui.text_c(text, x + tw / 2, y + 12, T.T_CAPTION, col, anchor="center", weight="medium")
            x -= 6
        return right - x
