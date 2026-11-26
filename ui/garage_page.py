"""The Garage: a vehicle spawner with a card grid, class tabs, paint colours and live specs.

Cars are listed from the same tables the traffic and physics use (``vehicle_instances``),
so lengths, widths and masses on the cards are the real numbers.  Only passenger-class
vehicles can be driven by the player; buses and trucks are shown locked, the same way the
physics treats them.

The host (``App``) provides, duck-typed:

    garage_spawn(kind, rgb, enter) -> (ok, message)   place a car on a clear road nearby
    garage_count()  -> int                              how many cars the player has spawned
    garage_clear()                                      remove them all
    garage_enter    -> bool                             "get in after spawning" preference
"""
from __future__ import annotations

import math
import time

from . import icons
from . import theme as T
from ..vehicles.driving import DRIVABLE_KINDS
from .menu_ui import Page, frame_rect
from .ui_kit import (alpha, button, card, chip, close_button, divider, ease_out_expo, focus_ring, glass, group,
                     heading, inside, keycaps, mix, panel_header, scrollbar, sheet, spring, surface, toast,
                     toggle, toggle_rect)
from ..vehicles.vehicle_instances import HEIGHTS, LENGTHS, MASSES, VEHICLE_NAMES, WHEELBASES, WHEEL_RADII, WIDTHS

# (name, linear RGB as the renderer takes it)
PALETTE = (
    ("Marine blue", (0.025, 0.22, 0.65)),
    ("Pearl white", (0.80, 0.80, 0.78)),
    ("Onyx black", (0.012, 0.012, 0.014)),
    ("Racing red", (0.60, 0.02, 0.03)),
    ("Steel grey", (0.18, 0.19, 0.21)),
    ("Silver", (0.50, 0.52, 0.55)),
    ("Forest green", (0.02, 0.20, 0.09)),
    ("Solar orange", (0.85, 0.26, 0.02)),
    ("Taxi yellow", (0.85, 0.55, 0.02)),
    ("Violet", (0.22, 0.05, 0.42)),
    ("Teal", (0.01, 0.32, 0.36)),
    ("Rose", (0.62, 0.16, 0.30)),
)

# class tab -> kinds, in display order (indices into VEHICLE_NAMES)
CLASSES = (
    ("All", None),
    ("Cars", (0, 2, 5)),
    ("SUVs", (1, 8)),
    ("Utility", (7, 6)),
    ("Heavy", (3, 4)),
)
KIND_CLASS = {0: "Saloon", 1: "SUV", 2: "Hybrid hatchback", 3: "City bus", 4: "Box truck", 5: "Taxi",
              6: "Panel van", 7: "Pickup", 8: "Compact SUV"}

# Side-view silhouettes, front to the right.  Points are (x / length, y / height), y up from the
# ground; every polygon is convex (Overlay2D.poly fans from the centroid).
# roles: body = paint, cabin = paint (slightly darker), glass, trim = dark plastic / light bar.
_SEDAN_BODY = [(.03, .22), (.03, .46), (.09, .55), (.80, .58), (.96, .47), (.985, .30), (.97, .22)]
PROFILES = {
    0: [("body", _SEDAN_BODY), ("cabin", [(.27, .55), (.39, 1.0), (.65, 1.0), (.80, .56)]),
        ("glass", [(.30, .60), (.405, .93), (.50, .93), (.50, .60)]),
        ("glass", [(.52, .60), (.52, .93), (.645, .93), (.765, .60)])],
    1: [("body", [(.03, .26), (.03, .54), (.08, .62), (.82, .62), (.97, .52), (.985, .32), (.97, .26)]),
        ("cabin", [(.20, .62), (.25, 1.0), (.76, 1.0), (.88, .62)]),
        ("glass", [(.23, .66), (.27, .93), (.49, .93), (.49, .66)]),
        ("glass", [(.51, .66), (.51, .93), (.745, .93), (.85, .66)])],
    2: [("body", [(.03, .22), (.03, .50), (.10, .58), (.80, .56), (.97, .45), (.985, .28), (.97, .22)]),
        ("cabin", [(.22, .56), (.36, 1.0), (.62, 1.0), (.82, .56)]),
        ("glass", [(.26, .60), (.37, .93), (.49, .93), (.49, .60)]),
        ("glass", [(.51, .60), (.51, .93), (.61, .93), (.77, .60)])],
    3: [("body", [(.015, .14), (.015, .97), (.035, 1.0), (.965, 1.0), (.99, .96), (.99, .14)]),
        ("glass", [(.04, .56), (.04, .90), (.20, .90), (.20, .56)]),
        ("glass", [(.22, .56), (.22, .90), (.38, .90), (.38, .56)]),
        ("glass", [(.40, .56), (.40, .90), (.56, .90), (.56, .56)]),
        ("glass", [(.58, .56), (.58, .90), (.74, .90), (.74, .56)]),
        ("glass", [(.82, .56), (.82, .90), (.975, .90), (.975, .40)]),
        ("trim", [(.015, .14), (.015, .26), (.99, .26), (.99, .14)])],
    4: [("body", [(.02, .24), (.02, 1.0), (.66, 1.0), (.66, .24)]),
        ("cabin", [(.69, .24), (.69, .78), (.76, .92), (.93, .92), (.985, .62), (.985, .24)]),
        ("glass", [(.77, .86), (.92, .86), (.955, .66), (.77, .66)]),
        ("trim", [(.02, .14), (.02, .26), (.985, .26), (.985, .14)])],
    5: [("body", _SEDAN_BODY), ("cabin", [(.27, .55), (.39, 1.0), (.65, 1.0), (.80, .56)]),
        ("glass", [(.30, .60), (.405, .93), (.50, .93), (.50, .60)]),
        ("glass", [(.52, .60), (.52, .93), (.645, .93), (.765, .60)]),
        ("sign", [(.45, 1.0), (.46, 1.07), (.58, 1.07), (.59, 1.0)])],
    6: [("body", [(.02, .20), (.02, .92), (.06, 1.0), (.78, 1.0), (.97, .62), (.99, .42), (.985, .20)]),
        ("glass", [(.46, .93), (.72, .93), (.72, .66), (.46, .66)]),
        ("glass", [(.76, .93), (.935, .66), (.76, .66)]),
        ("trim", [(.02, .20), (.02, .30), (.985, .30), (.985, .20)])],
    7: [("body", [(.02, .26), (.02, .52), (.06, .56), (.60, .56), (.78, .62), (.97, .54), (.985, .30), (.97, .26)]),
        ("trim", [(.04, .56), (.04, .64), (.50, .64), (.50, .56)]),
        ("cabin", [(.50, .58), (.56, 1.0), (.76, 1.0), (.85, .62)]),
        ("glass", [(.545, .66), (.58, .93), (.64, .93), (.64, .66)]),
        ("glass", [(.66, .66), (.66, .93), (.745, .93), (.805, .66)])],
    8: [("body", [(.03, .24), (.03, .52), (.08, .60), (.80, .60), (.96, .50), (.985, .30), (.97, .24)]),
        ("cabin", [(.24, .60), (.33, 1.0), (.72, 1.0), (.84, .60)]),
        ("glass", [(.27, .64), (.345, .93), (.50, .93), (.50, .64)]),
        ("glass", [(.52, .64), (.52, .93), (.71, .93), (.81, .64)])],
}
EXTRA_AXLE = {4: 0.09}


def display_rgb(rgb):
    """Linear paint -> an sRGB-ish colour for the 2D preview."""
    return tuple(min(1.0, max(0.0, c)) ** (1 / 2.2) for c in rgb)


def draw_vehicle(ui, kind, x, y, w, h, rgb, dim=1.0, t=0.0):
    """Draw a side-view silhouette of ``kind`` centred in the box ``(x, y, w, h)``."""
    kind = int(kind) % len(LENGTHS)
    L, H, R = float(LENGTHS[kind]), float(HEIGHTS[kind]) * 1.08, float(WHEEL_RADII[kind])
    k = min(w * 0.94 / L, h * 0.88 / H)
    gx0 = x + (w - L * k) / 2
    ground = y + h / 2 + H * k / 2
    paint = display_rgb(rgb)
    base = (paint[0], paint[1], paint[2], dim)
    dark = (paint[0] * .82, paint[1] * .82, paint[2] * .82, dim)
    glass = (0.10, 0.15, 0.20, 0.96 * dim)
    trim = (0.05, 0.055, 0.06, 0.96 * dim)
    sign = (1.0, 0.82, 0.30, dim)
    cols = {"body": base, "cabin": dark, "glass": glass, "trim": trim, "sign": sign}
    # soft ground shadow
    ui.box(gx0 - L * k * .02, ground - 3, L * k * 1.04, 8, 4, (0, 0, 0, 0.30 * dim))
    for role, pts in PROFILES[kind]:
        ui.poly([(gx0 + px * L * k, ground - py * (H / 1.08) * k) for px, py in pts], cols[role])
    # glass highlight: a thin bright edge across the top of each pane
    for role, pts in PROFILES[kind]:
        if role == "glass":
            top = sorted(pts, key=lambda p: -p[1])[:2]
            a, b = top
            ui.line(gx0 + a[0] * L * k, ground - a[1] * (H / 1.08) * k + 1,
                    gx0 + b[0] * L * k, ground - b[1] * (H / 1.08) * k + 1, 1.0, (1, 1, 1, 0.22 * dim))
    # wheels
    over = (1.0 - float(WHEELBASES[kind]) / L) / 2.0
    axles = [over, 1.0 - over]
    if kind in EXTRA_AXLE:
        axles.append(over + EXTRA_AXLE[kind])
    r = R * k
    for ax in axles:
        wx, wy = gx0 + ax * L * k, ground - r
        ui.circle(wx, wy, r * 1.20, (0.02, 0.02, 0.025, 0.96 * dim))
        ui.circle(wx, wy, r, (0.045, 0.045, 0.05, dim))
        ui.circle(wx, wy, r * 0.60, (0.62, 0.64, 0.68, dim))
        ui.circle(wx, wy, r * 0.22, (0.16, 0.17, 0.19, dim))
        spin = t * 2.0
        for i in range(5):
            a = spin + i * math.tau / 5
            ui.line(wx, wy, wx + math.cos(a) * r * 0.58, wy + math.sin(a) * r * 0.58, 1.0, (0.2, 0.2, 0.23, dim))
    # lamps
    ui.circle(gx0 + 0.972 * L * k, ground - 0.40 * (H / 1.08) * k, max(1.5, 0.018 * L * k), (1.0, 0.96, 0.80, dim))
    ui.circle(gx0 + 0.028 * L * k, ground - 0.40 * (H / 1.08) * k, max(1.4, 0.015 * L * k), (0.95, 0.12, 0.10, dim))


def no_entry(ui, cx, cy, r, color):
    ui.ring(cx, cy, r, 1.8, color)
    d = r * 0.70
    ui.line(cx - d, cy + d, cx + d, cy - d, 1.8, color)


class GaragePage(Page):
    name = "garage"
    title = "Garage"

    def __init__(self, menu):
        super().__init__(menu)
        self.tab = 0
        self.kind = 0
        self.color = 0
        self.scroll = 0.0
        self.scroll_max = 0.0
        self.view = (0.0, 0.0, 0.0, 0.0)
        self.cols = 3
        self.toast = ("", 0.0, False)
        self._cards = []
        self._reveal = False

    # ------------------------------------------------------------- data
    def kinds(self):
        want = CLASSES[self.tab][1]
        return list(range(len(VEHICLE_NAMES))) if want is None else list(want)

    def on_open(self):
        self.scroll = 0.0
        if self.kind not in self.kinds():
            self.kind = self.kinds()[0]
        self.toast = ("", 0.0, False)

    def _say(self, text, ok):
        self.toast = (text, time.time() + 3.2, ok)

    @property
    def rgb(self):
        return PALETTE[self.color][1]

    # ------------------------------------------------------------- layout
    def _geometry(self, w, h):
        fx, fy, fw, fh = frame_rect(w, h, 1120, 700)
        fy += self.menu.oy()
        pad = 32.0
        detail_h = 168.0 if fh >= 680 else 150.0
        detail_y = fy + fh - 18 - detail_h
        tabs_y = fy + 86
        vy = tabs_y + 46
        vh = detail_y - 12 - vy
        cw = fw - 2 * pad
        self.cols = 5 if cw >= 1000 else 4 if cw >= 800 else 3 if cw >= 560 else 2
        return dict(frame=(fx, fy, fw, fh), pad=pad, detail=(fx + pad, detail_y, cw, detail_h),
                    tabs_y=tabs_y, vx=fx + pad, vy=vy, vw=cw, vh=vh)

    # --------------------------------------------------------------- draw
    def draw(self, ui, w, h, mouse):
        host = self.menu.host
        g = self._geometry(w, h)
        fx, fy, fw, fh = g["frame"]
        pad = g["pad"]
        now = time.time()

        sheet(ui, fx, fy, fw, fh)
        self.hits.add((fx, fy, fw, fh), "panel")

        # Header
        panel_header(ui, fx + pad, fy + 26, fw - 240, "Garage", "Call a vehicle to a clear road near you", icon="car", tile=T.TILE_ORANGE)

        cr = close_button(ui, fx + fw - pad - 32, fy + 30, mouse)
        self.hits.add(cr, "close")

        count = host.garage_count() if hasattr(host, "garage_count") else 0
        if count:
            chip(ui, fx + fw - pad - 32 - 12, fy + 36, f"{count} spawned", T.TEXT_SECONDARY, anchor="right", icon="car")

        self._draw_tabs(ui, g, mouse)
        self._draw_grid(ui, g, mouse, now)
        self._draw_detail(ui, g, mouse, now, count)

        # Toast notification
        text, until, ok = self.toast
        if text and now < until:
            toast(ui, fx + fw / 2, fy + 28, text, ok)

    def _draw_tabs(self, ui, g, mouse):
        x, y = g["vx"], g["tabs_y"]
        for i, (label, kinds) in enumerate(CLASSES):
            n = len(VEHICLE_NAMES) if kinds is None else len(kinds)
            lw = ui.text_w(label, T.T_BODY, weight="semibold")
            tw = lw + 22 + 28
            r = (x, y + 2, tw, 32.0)
            sel = i == self.tab
            hov = spring(ui, ("gar_tab", i), float(inside(r, *mouse)), T.SPRING_GENTLE)
            self.hits.add(r, "tab", i)

            if sel:
                ui.box(*r, 16, (1.0, 1.0, 1.0, 0.15))
            elif hov > 0.01:
                ui.box(*r, 16, (1.0, 1.0, 1.0, 0.07 * hov))

            ui.text_c(label, r[0] + 14, r[1] + 16, T.T_BODY, T.TEXT_PRIMARY if sel else mix(T.TEXT_SECONDARY, T.TEXT_PRIMARY, hov),
                      weight="semibold")
            ui.text_c(str(n), r[0] + 14 + lw + 7, r[1] + 16, T.T_CAPTION_2, T.TEXT_SECONDARY if sel else T.TEXT_TERTIARY,
                      weight="medium")
            x += tw + 4

    def _draw_grid(self, ui, g, mouse, now):
        vx, vy, vw, vh = g["vx"], g["vy"], g["vw"], g["vh"]
        kinds = self.kinds()
        cols, gap = self.cols, 12.0
        rows = max(1, math.ceil(len(kinds) / cols))
        cw = (vw - gap * (cols - 1)) / cols
        ch = max(112.0, min(158.0, (vh - gap * (rows - 1)) / rows))
        total = rows * ch + (rows - 1) * gap
        self.scroll_max = max(0.0, total - vh)

        if self._reveal and self.kind in kinds:
            row = kinds.index(self.kind) // cols
            top, bot = row * (ch + gap), row * (ch + gap) + ch
            if top < self.scroll:
                self.scroll = top
            elif bot > self.scroll + vh:
                self.scroll = bot - vh
        self._reveal = False
        self.scroll = max(0.0, min(self.scroll_max, self.scroll))
        self.view = (vx - 6, vy - 4, vw + 12, vh + 8)

        ui.push_clip(*self.view)
        self._cards = []
        for i, kind in enumerate(kinds):
            r, c = divmod(i, cols)
            rect = (vx + c * (cw + gap), vy + r * (ch + gap) - self.scroll, cw, ch)
            self._cards.append((kind, rect))
            if rect[1] > vy + vh or rect[1] + ch < vy:
                continue
            self._draw_card(ui, kind, rect, mouse, now, i)
        ui.pop_clip()
        scrollbar(ui, vx + vw + 8, vy, vh, vh, total, self.scroll)

    def _draw_card(self, ui, kind, rect, mouse, now, index):
        x, y, w, h = rect
        locked = kind not in DRIVABLE_KINDS
        sel = kind == self.kind
        hov = spring(ui, ("gar_card", kind), float(inside(rect, *mouse) and inside(self.view, *mouse)), T.SPRING_GENTLE)
        self.hits.add(rect, "card", kind, clip=self.view)

        e = ease_out_expo((self.menu.t - index * 0.03) / 0.30)
        dim = (0.45 if locked else 1.0) * e

        card(ui, x, y, w, h, selected=sel, hovered=hov > 0.01, focused=False,
             enabled=not locked, r=18, tint=display_rgb(self.rgb) + (1.0,) if sel else None)

        art_h = h - 50
        draw_vehicle(ui, kind, x + 14, y + 8, w - 28, art_h - 4, self.rgb, dim=dim, t=now if sel else 0.0)

        ui.text(ui.fit(VEHICLE_NAMES[kind], T.T_BODY, w - 90, weight="semibold"), x + 16, y + h - 38, T.T_BODY,
                T.TEXT_PRIMARY if not locked else T.TEXT_TERTIARY, weight="semibold")
        ui.text(KIND_CLASS[kind], x + 16, y + h - 20, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="regular")
        ui.text(f"{LENGTHS[kind]:.1f} m", x + w - 16, y + h - 38, T.T_CAPTION_2, T.TEXT_SECONDARY, anchor="right",
                weight="medium")

        if locked:
            no_entry(ui, x + w - 26, y + 26, 11, alpha(T.ERROR_RED, 0.85))
            ui.text("Not drivable", x + w - 16, y + h - 20, T.T_CAPTION_2, T.ERROR_RED, anchor="right",
                    weight="medium")

    def _draw_detail(self, ui, g, mouse, now, count):
        host = self.menu.host
        x, y, w, h = g["detail"]
        kind = self.kind
        locked = kind not in DRIVABLE_KINDS

        group(ui, x, y, w, h, 22)

        art = 250.0 if w >= 760 else 170.0
        draw_vehicle(ui, kind, x + 12, y + 12, art, h - 24, self.rgb, dim=0.5 if locked else 1.0, t=now)

        ix = x + art + 30
        right_w = 206.0
        info_w = x + w - right_w - 28 - ix

        ui.text(ui.fit(VEHICLE_NAMES[kind], T.T_HEAD, info_w, weight="bold"), ix, y + 16, T.T_HEAD, T.TEXT_PRIMARY,
                weight="bold")
        ui.text(f"{KIND_CLASS[kind]}  ·  {PALETTE[self.color][0]}", ix, y + 42, T.T_CAPTION_2, T.TEXT_TERTIARY,
                weight="regular")

        specs = (("Length", f"{LENGTHS[kind]:.2f} m"), ("Width", f"{WIDTHS[kind]:.2f} m"),
                 ("Height", f"{HEIGHTS[kind]:.2f} m"), ("Weight", f"{int(round(MASSES[kind], -1)):,} kg"))
        sw = info_w / 4
        for i, (k, v) in enumerate(specs):
            sx = ix + i * sw
            ui.text(k, sx, y + 62, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="regular")
            ui.text(ui.fit(v, T.T_BODY, sw - 8), sx, y + 78, T.T_BODY, T.TEXT_PRIMARY, weight="semibold")

        # Paint swatches
        sx0, sy0 = ix, y + h - 36
        step = min(30.0, info_w / len(PALETTE))
        rad = min(10.0, (step - 5) / 2)
        for i, (name, rgb) in enumerate(PALETTE):
            r = (sx0 + i * step, sy0, step - 2, 24.0)
            cx_, cy_ = r[0] + rad + 2, r[1] + 12
            hov = spring(ui, ("gar_sw", i), float(inside(r, *mouse)), T.SPRING_GENTLE)
            self.hits.add(r, "swatch", i)
            col = display_rgb(rgb)
            rr = rad + hov * 0.8
            ui.rect(cx_ - rr, cy_ - rr, 2 * rr, 2 * rr, (col[0], col[1], col[2], 1.0))
            if i == self.color:
                ui.box(cx_ - rad - 4, cy_ - rad - 4, 2 * rad + 8, 2 * rad + 8, 0.0, None, T.TEXT_PRIMARY, bw=2.0)

        # Actions
        bx = x + w - right_w - 16
        enter = bool(getattr(host, "garage_enter", True))
        row = (bx, y + 14, right_w, 28.0)
        ui.text_c("Drive right away", bx, y + 28, T.T_BODY, T.TEXT_SECONDARY, weight="medium")
        toggle(ui, row, enter, key="gar_enter", focus=False, enabled=not locked)
        if not locked:
            self.hits.add(toggle_rect(row), "enter")

        call = (bx, y + 56, right_w, 46.0)
        label = "Call vehicle" if not locked else "Not drivable"
        button(ui, call, label, mouse, key="gar_call", kind="primary" if not locked else "secondary",
               icon="car" if not locked else "lock", disabled=locked, size=T.T_BODY + 1)
        if not locked:
            self.hits.add(call, "call")

        clr = (bx, y + 112, right_w, 36.0)
        button(ui, clr, f"Remove mine ({count})" if count else "Nothing spawned", mouse, key="gar_clear",
               kind="tertiary", icon="trash", disabled=not count, size=T.T_BODY)
        if count:
            self.hits.add(clr, "clear")

    # -------------------------------------------------------------- input
    def prompts(self):
        return [("← ↑ ↓ →", "Browse"), ("Q E", "Class"), ("C", "Paint"), ("Enter", "Call vehicle"),
                ("Esc", "Back")]

    def _call(self):
        host = self.menu.host
        if self.kind not in DRIVABLE_KINDS:
            self._say("Buses and trucks can't be driven", False)
            return
        ok, msg = host.garage_spawn(self.kind, self.rgb, bool(getattr(host, "garage_enter", True)))
        if ok:
            self.menu.resume()
        else:
            self._say(msg, False)

    def _set_tab(self, i):
        i %= len(CLASSES)
        if i != self.tab:
            self.tab = i
            self.scroll = 0.0
            if self.kind not in self.kinds():
                self.kind = self.kinds()[0]

    def _move(self, dx, dy):
        kinds = self.kinds()
        i = kinds.index(self.kind) if self.kind in kinds else 0
        cols = self.cols
        if dx:
            j = max(0, min(len(kinds) - 1, i + dx))
        else:
            j = i + dy * cols
            if not 0 <= j < len(kinds):
                j = i
        if j != i:
            self.kind = kinds[j]
            self._reveal = True

    def key(self, ev, pg):
        k = ev.key
        host = self.menu.host
        if k in (pg.K_LEFT, pg.K_a):
            self._move(-1, 0)
        elif k in (pg.K_RIGHT, pg.K_d):
            self._move(1, 0)
        elif k in (pg.K_UP, pg.K_w):
            self._move(0, -1)
        elif k in (pg.K_DOWN, pg.K_s):
            self._move(0, 1)
        elif k == pg.K_q:
            self._set_tab(self.tab - 1)
        elif k in (pg.K_e, pg.K_TAB):
            self._set_tab(self.tab + (-1 if getattr(ev, "mod", 0) & pg.KMOD_SHIFT else 1))
        elif k == pg.K_c:
            back = getattr(ev, "mod", 0) & pg.KMOD_SHIFT
            self.color = (self.color + (-1 if back else 1)) % len(PALETTE)
        elif k == pg.K_g:
            host.garage_enter = not bool(getattr(host, "garage_enter", True))
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
            self._call()
        elif k == pg.K_DELETE and hasattr(host, "garage_count") and host.garage_count():
            host.garage_clear()
            self._say("Removed your spawned vehicles", True)
        else:
            return False
        return True

    def wheel(self, dy, pos):
        if inside(self.view, *pos):
            self.scroll = max(0.0, min(self.scroll_max, self.scroll - dy * 48.0))

    def hover(self, pos):
        hit = self.hits.at(pos)
        if hit and hit[0] == "card":
            self.kind = hit[1]

    def press(self, pos):
        hit = self.hits.at(pos)
        if not hit:
            return False
        tag, data, _ = hit
        host = self.menu.host
        if tag == "close":
            self.menu.back()
        elif tag == "tab":
            self._set_tab(data)
        elif tag == "card":
            if data == self.kind and data in DRIVABLE_KINDS and self._double(pos):
                self._call()
            else:
                self.kind = data
        elif tag == "swatch":
            self.color = data
        elif tag == "enter":
            host.garage_enter = not bool(getattr(host, "garage_enter", True))
        elif tag == "call":
            self._call()
        elif tag == "clear":
            host.garage_clear()
            self._say("Removed your spawned vehicles", True)
        return True

    _last_click = (0.0, (0.0, 0.0))

    def _double(self, pos):
        now = time.time()
        t0, p0 = self._last_click
        self._last_click = (now, pos)
        return now - t0 < 0.45 and math.hypot(pos[0] - p0[0], pos[1] - p0[1]) < 10