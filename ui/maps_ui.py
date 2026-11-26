"""Interface chrome shared by the world-map and 3D views, in the Apple "liquid glass" language.

* Map screen: a floating glass sidebar (brand, Search / Guides / Directions, an expandable
  list of places, a travel call-to-action), a grouped control cluster (map style / locate /
  full screen), a compass and a zoom pair.
* 3D screen: a place card with search, a time + conditions card, the same controls, plus the
  HUD building blocks (key-hint bar, status card, speed card).

``layout`` is the single source of truth for geometry: drawing and hit-testing both use it,
so they can never disagree.  All coordinates are logical pixels.
"""
from __future__ import annotations

import math
import time

from . import theme as T
from ..core.config import ACCENT
from .ui_kit import (_in, _key_tokens, caps, caps_w, activity, alpha, button, chip, glass, icon_tile, inside, keycaps, mix, progress,
                     search_field, spring, surface)

# Re-export for backward compatibility
from .theme import (
    TEXT_PRIMARY as WHITE,
    TEXT_TERTIARY as DIM,
)

HOVER = (1.0, 1.0, 1.0, 0.08)

TIPS = {
    "sidebar": "Toggle sidebar", "map": "Map style", "locate": "Recenter",
    "fullscreen": "Full screen  F11", "compass": "Reset north",
    "zoom_in": "Zoom in", "zoom_out": "Zoom out", "search": "Search  /",
}


def _wrap(ui):          # kept for older call sites; sizes are real pixels now
    return ui


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
        out.append(("compass", (gx, gy + 3 * B + 10, B, B)))
        if mode == "map":
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
            out.append(("search", (M, M, 52, 52)))
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
        tw = ui.text_w(text, T.T_CAPTION_2, weight="medium")
        bw, bh = tw + 24, 28
        fade = min(1.0, (self._tip_t - self.TIP_DELAY) * 8.0)
        if side == "left":
            x, y = rect[0] - 10 - bw, rect[1] + (rect[3] - bh) / 2
        else:
            x, y = rect[0] + rect[2] + 10, rect[1] + (rect[3] - bh) / 2
        x = max(8, x)
        surface(ui, round(x), round(y), round(bw), bh, bh / 2,
                ((0.16, 0.16, 0.18, 0.96 * fade), (0.10, 0.10, 0.12, 0.96 * fade)),
                (1, 1, 1, 0.14 * fade), None)
        ui.text_c(text, x + bw / 2, y + bh / 2, T.T_CAPTION_2, (1, 1, 1, fade), anchor="center", weight="medium")

    # ------------------------------------------------------------------ pieces
    def _button_group(self, ui, rects, names, mouse, icons, active=None):
        x, y, w, _ = rects[0]
        total = sum(r[3] for r in rects)
        glass(ui, x, y, w, total, 14, "hud")
        for i, (r, ic, nm) in enumerate(zip(rects, icons, names)):
            hv = spring(ui, ("hv", nm), 1.0 if _in(r, *mouse) else 0.0, T.SPRING_GENTLE)
            if hv > 0.01:
                ui.box(r[0] + 3, r[1] + 3, r[2] - 6, r[3] - 6, 10, (1.0, 1.0, 1.0, 0.12 * hv))
            if i > 0:
                ui.box(r[0] + 10, r[1], r[2] - 20, 1, 0.5, T.SEPARATOR)
            col = T.ACCENT_BLUE if active == i else T.TEXT_PRIMARY
            ic(ui, r[0] + r[2] / 2, r[1] + r[3] / 2, 20, col)

    def _compass(self, ui, r, mouse, north_vec=(0.0, -1.0)):
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        hv = spring(ui, ("hv", "compass"), 1.0 if _in(r, *mouse) else 0.0, T.SPRING_GENTLE)
        rad = r[2] / 2
        surface(ui, r[0], r[1], r[2], r[3], rad, T.GLASS, mix(T.BORDER_DEFAULT, T.BORDER_EMPHASIZED, hv),
                T.SHADOW_2, specular=False)
        ang = math.atan2(north_vec[0], -north_vec[1])
        for i in range(24):
            a = ang + i * math.pi / 12.0
            r0, r1 = rad - 4.0, rad - (7.5 if i % 6 == 0 else 6.0)
            col = (1, 1, 1, 0.65 if i % 6 == 0 else 0.26)
            ui.line(cx + r0 * math.sin(a), cy - r0 * math.cos(a),
                    cx + r1 * math.sin(a), cy - r1 * math.cos(a),
                    1.3 if i % 6 == 0 else 1.0, col)
        ui.text_c("N", cx, cy - 1, T.T_CAPTION_2, T.SYS_RED, anchor="center", weight="bold")

    # ------------------------------------------------------------------ shared cluster
    def draw_controls(self, ui, w, h, mode, mouse, fullscreen=False, north_vec=(0.0, -1.0)):
        rects = {n: r for n, r in self.layout(w, h, mode)}
        self._button_group(
            ui, [rects["map"], rects["locate"], rects["fullscreen"]], ("map", "locate", "fullscreen"),
            mouse, [self.icon_map, self.icon_locate,
                    (lambda u, x, y, s, c: self.icon_fullscreen(u, x, y, s, c, fullscreen))])
        self._compass(ui, rects["compass"], mouse, north_vec)
        if "zoom_in" in rects:
            zi, zo = rects["zoom_in"], rects["zoom_out"]
            self._button_group(ui, [zi, zo], ("zoom_in", "zoom_out"), mouse, [self._plus, self._minus])

    def _finish_tooltips(self, ui, w, h, mode, mouse):
        name = self.hit(mouse, w, h, mode)
        self._track_hover(ui, name if name in TIPS else None)
        if name in TIPS:
            rect = {n: r for n, r in self.layout(w, h, mode)}.get(name)
            if rect:
                self._tooltip(ui, name, rect, w, "right" if name in ("sidebar", "search") else "left")

    def pill(self, ui, x, y, text, size=T.T_BODY, color=T.TEXT_SECONDARY, pad=14, bold=False):
        """A small one-line capsule; returns (width, height)."""
        tw = ui.text_w(text, size, weight="semibold" if bold else "medium")
        hgt = size + 16
        glass(ui, x, y, tw + 2 * pad, hgt, hgt / 2, "hud")
        ui.text_c(text, x + pad, y + hgt / 2, size, color, weight="semibold" if bold else "medium")
        return tw + 2 * pad, hgt

    def _field(self, ui, x, y, w, h, query, placeholder="Search places"):
        search_field(ui, x, y, w, h, query, placeholder, True, time.time())

    # ------------------------------------------------------------------ map screen
    def draw_map_ui(self, ui, w, h, mouse, cities, marker, typing, query, fullscreen,
                    tiles_dark=True, can_enter_3d=False, navigation=None):
        rects = {n: r for n, r in self.layout(w, h, "map", len(cities))}
        M, sw = T.MARGIN, self.SIDEBAR_W
        if self.sidebar_open:
            sb = rects["sidebar_bg"]
            glass(ui, *sb, 26, "menu")
            caps(ui, "// Maps", sb[0] + 20, sb[1] + 28, T.T_TITLE_2, T.TEXT_PRIMARY, "bold", vcenter=True)
            tg = rects["sidebar"]
            hv = spring(ui, ("hv", "sidebar"), 1.0 if _in(tg, *mouse) else 0.0, T.SPRING_GENTLE)
            ui.box(tg[0], tg[1], 32, 32, 10, mix((1.0, 1.0, 1.0, 0.10), T.SYS_BLUE, hv))
            self.icon_sidebar(ui, tg[0] + 16, tg[1] + 16, 18, T.TEXT_SECONDARY)
            tiles = {"guides": T.TILE_ORANGE, "directions": T.TILE_BLUE}
            rows = (("search", "Search", self.icon_search), ("guides", "Guides", self.icon_guides),
                    ("directions", "Directions", self.icon_directions))
            for name, label, ic in rows:
                r = rects[name]
                hv = spring(ui, ("hv", name), 1.0 if _in(r, *mouse) else 0.0, T.SPRING_GENTLE)
                if name == "search":
                    if typing:
                        self._field(ui, *r, query)
                    else:
                        ui.box(*r, r[3] / 2, T.FIELD_FILL if hv < 0.5 else T.MATERIAL_THICK)
                        self.icon_search(ui, r[0] + 20, r[1] + r[3] / 2, 16, T.TEXT_TERTIARY)
                        ui.text_c(label, r[0] + 38, r[1] + r[3] / 2, T.T_BODY + 1, T.TEXT_TERTIARY)
                    continue
                on = name == "guides" and self.guides_open
                if hv > 0.01 or on:
                    ui.box(*r, 12, T.GROUP_SELECTED if on else ((1, 1, 1, 0.08 * hv),) * 2)
                tx, ty = r[0] + 6, r[1] + (r[3] - 28) / 2
                ui.box(tx, ty, 28, 28, 7, tiles[name], None)
                ic(ui, tx + 14, ty + 14, 16, (1, 1, 1, 1))
                ui.text_c(label, r[0] + 46, r[1] + r[3] / 2, T.T_BODY + 1, T.TEXT_PRIMARY, weight="medium")
            if self.guides_open and cities:
                c0 = rects["city:0"]
                caps(ui, "// Travel guides", c0[0] + 8, c0[1] - 22, T.T_CAPTION_1, T.TEXT_TERTIARY, "medium")
                for i, (name, _, _) in enumerate(cities):
                    r = rects[f"city:{i}"]
                    sel = marker is not None and marker[2] == name
                    hv = spring(ui, ("hv", f"city{i}"), 1.0 if _in(r, *mouse) else 0.0, T.SPRING_GENTLE)
                    if hv > 0.01:
                        ui.box(*r, 10, T.GROUP_HOVER)
                    elif sel:
                        ui.box(*r, 10, T.GROUP)
                    ui.text_c(name, r[0] + 12, r[1] + r[3] / 2, T.T_BODY, T.TEXT_PRIMARY)
                    ui.text_c(str((i + 1) % 10), r[0] + r[2] - 12, r[1] + r[3] / 2, T.T_CAPTION_2,
                              T.TEXT_TERTIARY, anchor="right", weight="medium")
            if navigation is not None and navigation.destination is not None and "route_cancel" in rects:
                rc = rects["route_cancel"]
                cy = rc[1] - 84
                ui.text("Driving route", rc[0] + 8, cy, T.T_CAPTION_2, T.TEXT_SECONDARY, weight="semibold")
                ui.text(ui.fit(navigation.destination.name, T.T_LABEL, rc[2] - 16, weight="semibold"),
                        rc[0] + 8, cy + 20, T.T_LABEL, T.TEXT_PRIMARY, weight="semibold")
                ui.text(ui.fit(navigation.instruction, T.T_CAPTION_2, rc[2] - 16),
                        rc[0] + 8, cy + 43, T.T_CAPTION_2, T.TEXT_SECONDARY)
                if navigation.route is not None:
                    distance = (f"{navigation.remaining / 1000.:.1f} km" if navigation.remaining >= 1000.
                                else f"{int(round(navigation.remaining / 10.) * 10)} m")
                    minutes = max(1, int(math.ceil(navigation.eta_seconds / 60.)))
                    summary = "Route complete" if navigation.status == "arrived" else f"{minutes} min · {distance} · estimated"
                    ui.text(summary, rc[0] + 8, cy + 62, T.T_CAPTION_2, T.TEXT_TERTIARY)
                button(ui, rc, "End route", mouse, key="route_end", kind="secondary", size=T.T_CAPTION_2)
            # travel call-to-action
            r = rects["go"]
            hv = spring(ui, ("hv", "go"), 1.0 if _in(r, *mouse) else 0.0, T.SPRING_GENTLE)
            if marker:
                top = mix(T.ACCENT_BLUE_FILL[0], (1, 1, 1, 1), 0.12 * hv)
                bot = mix(T.ACCENT_BLUE_FILL[1], (1, 1, 1, 1), 0.12 * hv)
                ui.box(*r, 20, (top, bot))
                ui.line(r[0] + 20, r[1] + 0.5, r[0] + r[2] - 20, r[1] + 0.5, 1.0, (1, 1, 1, 0.30))
                ui.text("Travel to", r[0] + 18, r[1] + 12, T.T_CAPTION_2, (1, 1, 1, 0.78), weight="medium")
                ui.text(ui.fit(marker[2], T.T_LABEL + 1, r[2] - 70), r[0] + 18, r[1] + 31, T.T_LABEL + 1,
                        T.TEXT_PRIMARY, weight="semibold")
                self.icon_arrow_up_right(ui, r[0] + r[2] - 28, r[1] + r[3] / 2, 18, T.TEXT_PRIMARY)
            else:
                ui.box(*r, 20, T.GROUP_HOVER if hv > 0.5 else T.GROUP)
                ui.text("Pick a place", r[0] + 18, r[1] + 13, T.T_LABEL, T.TEXT_PRIMARY, weight="semibold")
                ui.text(ui.fit("Click the map · double-click to travel", T.T_CAPTION_2, r[2] - 32),
                        r[0] + 18, r[1] + 36, T.T_CAPTION_2, T.TEXT_TERTIARY)
        else:
            r = rects["sidebar"]
            hv = spring(ui, ("hv", "sidebar"), 1.0 if _in(r, *mouse) else 0.0, T.SPRING_GENTLE)
            surface(ui, *r, 14, T.GLASS, mix(T.BORDER_DEFAULT, T.BORDER_EMPHASIZED, hv), T.SHADOW_2)
            self.icon_sidebar(ui, r[0] + r[2] / 2, r[1] + r[3] / 2, 20, T.TEXT_PRIMARY)
            if typing:
                glass(ui, r[0] + r[2] + 8, r[1], 280, r[3], r[3] / 2, "hud")
                self._field(ui, r[0] + r[2] + 12, r[1] + 4, 272, r[3] - 8, query)
        self.draw_controls(ui, w, h, "map", mouse, fullscreen)
        # attribution, tucked left of the zoom pair
        att = "© OpenStreetMap contributors"
        aw = ui.text_w(att, T.T_CAPTION_2, weight="medium") + 24
        ax, ay = w - T.MARGIN - self.BTN - 10 - aw, h - T.MARGIN - 24
        ui.box(ax, ay, aw, 24, 12, (0.0, 0.0, 0.0, 0.34))
        ui.text_c(att, ax + aw / 2, ay + 12, T.T_CAPTION_2, T.TEXT_SECONDARY, anchor="center", weight="medium")
        self._finish_tooltips(ui, w, h, "map", mouse)

    # ------------------------------------------------------------------ 3D screen
    def draw_scene_ui(self, ui, w, h, mouse, label, coords, clock, sub, typing, query,
                      fullscreen, north_vec):
        M, B, ch = T.MARGIN, self.BTN, 52
        # time + conditions: one compact pill, clock large, conditions quiet beside it
        sub_fit = ui.fit(sub, T.T_CAPTION_2, min(300, max(100, w * 0.24)))
        cw_clock = ui.text_w(clock, T.T_TITLE_2, weight="semibold")
        tcw = int(cw_clock + ui.text_w(sub_fit, T.T_CAPTION_2, weight="medium") + 18 + 36)
        tx = w - M - B - 12 - tcw
        glass(ui, tx, M, tcw, ch, ch / 2, "hud")
        ui.text_c(clock, tx + 18, M + ch / 2, T.T_TITLE_2, T.TEXT_PRIMARY, weight="semibold")
        ui.text_c(sub_fit, tx + 18 + cw_clock + 12, M + ch / 2 + 1, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")
        # place pill with search
        cwid = int(max(200, min(380, tx - 12 - M)))
        glass(ui, M, M, cwid, ch, ch / 2, "hud")
        hv = spring(ui, ("hv", "search"), 1.0 if _in((M, M, 52, ch), *mouse) else 0.0, T.SPRING_GENTLE)
        if hv > 0.01:
            ui.box(M + 6, M + 6, ch - 12, ch - 12, (ch - 12) / 2, (1.0, 1.0, 1.0, 0.12 * hv))
        self.icon_search(ui, M + 26, M + ch / 2, 17, T.TEXT_SECONDARY)
        if typing:
            self._field(ui, M + 50, M + 8, cwid - 60, ch - 16, query)
        else:
            ui.text(ui.fit(label or "Scene", T.T_LABEL, cwid - 74, weight="semibold"),
                    M + 52, M + 9, T.T_LABEL, T.TEXT_PRIMARY, weight="semibold")
            ui.text(coords, M + 52, M + 29, T.T_CAPTION_1, T.TEXT_TERTIARY, weight="medium")
        self.draw_controls(ui, w, h, "scene", mouse, fullscreen, north_vec)
        self._finish_tooltips(ui, w, h, "scene", mouse)

    # ------------------------------------------------------------------ HUD blocks
    def hint_bar(self, ui, x, y, max_w, items, fade=1.0, center=False):
        """A row of keycap + label pairs in one glass capsule.  Items that don't fit are
        dropped from the end, but the last item (the shortcuts entry) always stays.
        Returns the width drawn."""
        size = T.T_CAPTION_2
        widths = []
        for key, label in items:
            widths.append(keycaps(ui, 0, 0, key, 11, dry=True) + 6 + ui.text_w(label, size, weight="medium"))
        gap, pad = 16, 14
        keep = list(range(len(items)))
        while len(keep) > 1 and sum(widths[i] for i in keep) + gap * (len(keep) - 1) + 2 * pad > max_w:
            keep.pop(len(keep) - 2 if len(keep) > 2 else 0)
        total = sum(widths[i] for i in keep) + gap * (len(keep) - 1) + 2 * pad
        hh = 38
        if center:
            x = x + (max_w - total) / 2
        if fade < 1.0:
            return self._hint_bar_faded(ui, x, y, items, keep, widths, gap, pad, hh, total, size, fade)
        glass(ui, x, y, total, hh, hh / 2, "hud")
        cx = x + pad
        for i in keep:
            key, label = items[i]
            kw = keycaps(ui, cx, y + (hh - 21) / 2, key, 11)
            ui.text_c(label, cx + kw + 6, y + hh / 2, size, T.TEXT_SECONDARY, weight="medium")
            cx += widths[i] + gap
        return total

    def _hint_bar_faded(self, ui, x, y, items, keep, widths, gap, pad, hh, total, size, fade):
        """The same bar while it fades out: panel and text scaled by ``fade`` (keycaps are dropped)."""
        ui.box(x, y, total, hh, hh / 2, ((0.07, 0.075, 0.09, 0.55 * fade),) * 2, (1, 1, 1, 0.08 * fade))
        cx = x + pad
        for i in keep:
            key, label = items[i]
            ui.text_c(f"{key}", cx, y + hh / 2, size, (1, 1, 1, 0.85 * fade), weight="semibold")
            kw = ui.text_w(key, size, weight="semibold")
            ui.text_c(label, cx + kw + 6, y + hh / 2, size, (1, 1, 1, 0.55 * fade), weight="medium")
            cx += widths[i] + gap - (widths[i] - kw - 6 - ui.text_w(label, size, weight="medium"))
        return total

    def status_card(self, ui, x, bottom, rows, progress=None, shimmer=0.0, min_w=0):
        """rows: [(text, colour, size, weight)].  Anchored by its bottom edge."""
        from .ui_kit import progress as progress_bar
        pad, lead = 16, 6
        tw = max(ui.text_w(t, s, weight=wt) for t, _, s, wt in rows)
        w = max(min_w, tw + 2 * pad)
        h = sum(s + lead for _, _, s, _ in rows) + 2 * pad - lead + (8 if progress is not None else 0)
        y = bottom - h
        glass(ui, x, y, w, h, 18, "hud")
        cy = y + pad
        for t, col, s, wt in rows:
            ui.text(t, x + pad, cy, s, col, weight=wt)
            cy += s + lead
        if progress is not None:
            progress_bar(ui, x + pad, y + h - pad + 1, w - 2 * pad, 4, progress, T.ACCENT_BLUE, shimmer)
        return w, h

    def speed_card(self, ui, x, y, name, kmh, gear, state, signal=0, braking=False, frac=0.0, unit="km/h"):
        """Driving instrument cluster.  ``signal``: -1 left, +1 right, 0 off."""
        from .ui_kit import progress as progress_bar
        w, h = 236, 124
        glass(ui, x, y, w, h, 22, "hud")
        ui.text(ui.fit(name, T.T_CAPTION_2, w - 100, weight="semibold"), x + 18, y + 14, T.T_CAPTION_2,
                T.TEXT_TERTIARY, weight="semibold")
        sp = str(kmh)
        sw = ui.text(sp, x + 18, y + 30, 44, T.TEXT_PRIMARY, weight="bold")
        ui.text(unit, x + 18 + sw + 6, y + 56, T.T_BODY, T.TEXT_TERTIARY, weight="medium")
        gcx, gcy = x + w - 18 - 20, y + 14 + 20
        ui.box(gcx - 20, gcy - 20, 40, 40, 12, (1.0, 1.0, 1.0, 0.10), (1.0, 1.0, 1.0, 0.08))
        ui.text_c(gear, gcx, gcy, T.T_HEAD, T.TEXT_PRIMARY, anchor="center", weight="bold")
        blink = int(time.time() * 2.6) % 2 == 0
        for sgn, ax in ((-1, gcx - 20 - 36), (1, gcx - 20 - 18)):
            lit = signal == sgn and blink
            col = T.SYS_BLUE if lit else (1, 1, 1, 0.12)
            if sgn < 0:
                ui.tri((ax + 12, gcy - 7), (ax + 12, gcy + 7), (ax, gcy), col)
            else:
                ui.tri((ax, gcy - 7), (ax, gcy + 7), (ax + 12, gcy), col)
        progress_bar(ui, x + 18, y + 94, w - 36, 5, max(0.0, min(1.0, frac)),
                     T.SYS_RED if braking else T.ACCENT_BLUE)
        ui.text_c(ui.fit(state, T.T_CAPTION_2, w - 36), x + 18, y + 110, T.T_CAPTION_2,
                  T.WARN_AMBER if state.startswith("Blocked") else T.TEXT_TERTIARY, weight="medium")
        return w, h

    def chips(self, ui, right, y, items, min_x=0):
        """Right-aligned row of small state chips [(text, colour)], most important first.
        Chips that would run past ``min_x`` are dropped.  Returns the width used."""
        x = right
        for text, col in items:
            tw = caps_w(ui, text, T.T_CAPTION_1) + 22
            if x - tw < min_x:
                break
            x -= tw
            chip(ui, x, y, text, col, T.T_CAPTION_1, h=24, pad=11, fill=(0.0, 0.0, 0.0, 0.7))
            x -= 6
        return right - x
