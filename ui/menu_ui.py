"""The menu system: title / pause menu, travel, about, and the shared page machinery.

``MenuSystem`` owns a stack of full-screen pages drawn over the live 3D scene.  It is
an immediate-mode UI: each frame a page draws itself and registers hit rectangles in
a ``HitMap``; mouse events are resolved against what was drawn on the previous frame,
so what you see is always what you can click.  Keyboard navigation (arrows, Enter,
Esc, Q/E) works everywhere and mirrors the mouse.

The host (``App``) provides, duck-typed:

    ui_sound                      click() / toggle(on) / open() / close() / error()
    menu_info()      -> dict      location, coords, time, weather, mode, vehicle, world, loading
    menu_resume()                 leave the menu and play
    menu_open_map(), menu_search(), menu_travel(index), menu_quit()
    version                       text shown in the footer
"""
from __future__ import annotations

import math

from . import icons
from . import theme as T
from ..core.config import CITIES
from .ui_kit import (alpha, brackets, button, chip, close_button, ease_out, heading, inside, keycaps,
                     logo_mark, mix, panel_header, paragraph, prompt_bar, scrim)


class HitMap:
    """Rectangles registered while drawing: ``(rect, tag, data, clip)``."""

    def __init__(self):
        self.items = []

    def clear(self):
        self.items = []

    def add(self, rect, tag, data=None, clip=None):
        self.items.append((tuple(rect), tag, data, clip))

    def at(self, pos):
        if pos is None:
            return None
        for rect, tag, data, clip in reversed(self.items):
            if inside(rect, *pos) and (clip is None or inside(clip, *pos)):
                return tag, data, rect
        return None


def frame_rect(w, h, max_w=1040, max_h=680, margin=36, bottom=70):
    """A centred panel that leaves room for the prompt bar."""
    pw = min(max_w, w - 2 * margin)
    ph = min(max_h, h - bottom - 24)
    return ((w - pw) / 2, max(20.0, (h - bottom - ph) / 2), pw, ph)


class Page:
    """Base class.  Pages draw, register hits, and react to navigation."""
    name = ""
    title = ""

    def __init__(self, menu):
        self.menu = menu
        self.hits = HitMap()
        self.focus = 0
        self.mouse = (-1.0, -1.0)

    # lifecycle -----------------------------------------------------------
    def on_open(self):
        pass

    def draw(self, ui, w, h, mouse):
        raise NotImplementedError

    def prompts(self):
        return [("Esc", "Back")]

    # input ---------------------------------------------------------------
    def key(self, ev, pg):
        return False

    def press(self, pos):
        return False

    def drag(self, pos):
        pass

    def release(self):
        pass

    def hover(self, pos):
        pass

    def wheel(self, dy, pos):
        pass

    def hot(self, pos):
        """True when the pointer is over something clickable (for the hand cursor)."""
        hit = self.hits.at(pos)
        return hit is not None and hit[0] not in ("bg", "panel")

    # helpers -------------------------------------------------------------
    def click(self):
        self.menu.sound("click")


# ------------------------------------------------------------------ confirm
class ConfirmDialog:
    def __init__(self, title, message, yes="Quit", no="Cancel", danger=True, on_yes=None):
        self.title, self.message, self.yes, self.no = title, message, yes, no
        self.danger, self.on_yes = danger, on_yes
        self.focus = 1                      # default to the safe choice
        self.rects = {}

    def draw(self, ui, w, h, mouse):
        ui.rect(0, 0, w, h, (0.0, 0.0, 0.0, 0.55))
        pw, ph = min(440, w - 48), 190
        x, y = (w - pw) / 2, (h - ph) / 2 - 20
        ui.container(x, y, pw, ph, "popup", r=T.R_PANEL + 2)
        ui.text(self.title, x + 26, y + 24, T.T_TITLE + 2, T.TEXT, weight="bold")
        paragraph(ui, self.message, x + 26, y + 60, pw - 52, T.T_BASE, T.TEXT_2, weight="regular", max_lines=3)
        bw, bh = 128, 38
        no_r = (x + pw - 26 - bw * 2 - 12, y + ph - 26 - bh, bw, bh)
        yes_r = (x + pw - 26 - bw, y + ph - 26 - bh, bw, bh)
        self.rects = {"no": no_r, "yes": yes_r}
        button(ui, no_r, self.no, mouse, key="cf_no", focus=self.focus == 1)
        button(ui, yes_r, self.yes, mouse, key="cf_yes", kind="danger" if self.danger else "accent",
               focus=self.focus == 0)

    def key(self, ev, pg, menu):
        k = ev.key
        if k in (pg.K_LEFT, pg.K_RIGHT, pg.K_TAB, pg.K_UP, pg.K_DOWN):
            self.focus ^= 1
            menu.sound("click")
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
            self._finish(menu, self.focus == 0)
        elif k in (pg.K_ESCAPE, pg.K_n):
            self._finish(menu, False)
        elif k == pg.K_y:
            self._finish(menu, True)
        return True

    def press(self, pos, menu):
        for name, r in self.rects.items():
            if inside(r, *pos):
                self._finish(menu, name == "yes")
                return True
        return True

    def hover(self, pos):
        for name, r in self.rects.items():
            if inside(r, *pos):
                self.focus = 0 if name == "yes" else 1

    def _finish(self, menu, yes):
        menu.confirm = None
        menu.sound("click")
        if yes and self.on_yes is not None:
            self.on_yes()


# --------------------------------------------------------------------- system
class MenuSystem:
    def __init__(self, host, store):
        from .about_page import AboutPage
        from .garage_page import GaragePage
        from .settings_page import SettingsPage
        self.host, self.store = host, store
        self.page = None                 # name of the open page, or None when closed
        self.stack = []
        self.title = False               # True on the start screen, before the first "play"
        self.confirm = None
        self.t = 0.0                     # seconds since the current page opened
        self.pages = {"main": MainPage(self), "settings": SettingsPage(self, store),
                      "travel": TravelPage(self), "about": AboutPage(self), "garage": GaragePage(self)}
        self.mouse = (-1.0, -1.0)
        self.pressed = False
        self.size = (1280.0, 720.0)

    # -------------------------------------------------------------- state
    @property
    def is_open(self):
        return self.page is not None

    @property
    def current(self):
        return self.pages.get(self.page)

    def add_page(self, name, page):
        self.pages[name] = page

    def sound(self, name, *args):
        snd = getattr(self.host, "ui_sound", None)
        fn = getattr(snd, name, None) if snd is not None else None
        if fn is not None:
            fn(*args)

    def open(self, page="main", *, title=None):
        if title is not None:
            self.title = bool(title)
        was_open = self.is_open
        self.page, self.stack, self.confirm, self.t = page, [], None, 0.0
        self.pressed = False
        self.current.on_open()
        if not was_open:
            self.sound("open")

    def go(self, page):
        if page not in self.pages:
            return
        if self.page is not None:
            self.stack.append(self.page)
        self.page, self.t, self.pressed = page, 0.0, False
        self.current.on_open()
        self.sound("click")

    def back(self):
        if self.stack:
            self.page, self.t, self.pressed = self.stack.pop(), 0.0, False
            self.current.on_open()
            self.sound("close")
        elif self.page == "main" and self.title:
            self.ask_quit()
        else:
            self.close()

    def close(self):
        if not self.is_open:
            return
        self.page, self.stack, self.confirm, self.pressed = None, [], None, False
        self.store.save()
        self.sound("close")

    def ask_quit(self):
        self.confirm = ConfirmDialog("Quit OSM 3D?", "Your settings are saved automatically.",
                                     yes="Quit", no="Keep playing", on_yes=self.host.menu_quit)
        self.sound("open")

    def resume(self):
        self.title = False
        self.close()
        self.host.menu_resume()

    # --------------------------------------------------------------- frame
    def draw(self, ui, w, h, mouse, dt):
        self.t += dt
        self.mouse, self.size = mouse, (w, h)
        page = self.current
        if page is None:
            return
        page.hits.clear()
        page.mouse = mouse
        fade = ease_out(self.t / 0.22)
        scrim(ui, w, h, fade, flat=self.page != "main")
        page.draw(ui, w, h, mouse)
        prompt_bar(ui, w, h, page.prompts())
        if self.confirm is not None:
            self.confirm.draw(ui, w, h, mouse)

    def oy(self):
        """Vertical slide-in offset for the current page (logical px)."""
        return (1.0 - ease_out(self.t / 0.28)) * 16.0

    # --------------------------------------------------------------- input
    def handle(self, ev, pg, pos):
        """Route one pygame event.  ``pos`` is the logical pointer position or ``None``."""
        page = self.current
        if page is None:
            return False
        t = ev.type
        if t == pg.KEYDOWN:
            if self.confirm is not None:
                return self.confirm.key(ev, pg, self)
            if page.key(ev, pg):
                return True
            if ev.key == pg.K_ESCAPE:
                self.back()
            return True
        if t == pg.MOUSEBUTTONDOWN:
            if ev.button != 1 or pos is None:
                return True
            if self.confirm is not None:
                return self.confirm.press(pos, self)
            self.pressed = True
            page.press(pos)
            return True
        if t == pg.MOUSEBUTTONUP:
            self.pressed = False
            page.release()
            return True
        if t == pg.MOUSEMOTION:
            if pos is not None:
                self.mouse = pos
                if self.confirm is not None:
                    self.confirm.hover(pos)
                else:
                    page.hover(pos)
                    if self.pressed and ev.buttons[0]:
                        page.drag(pos)
            return True
        if t == pg.MOUSEWHEEL:
            if self.confirm is None:
                page.wheel(ev.y, self.mouse)
            return True
        return False

    def cursor(self, pos):
        page = self.current
        if page is None or pos is None:
            return False
        if self.confirm is not None:
            return any(inside(r, *pos) for r in self.confirm.rects.values())
        return page.hot(pos)


# ------------------------------------------------------------------ main menu
class MainPage(Page):
    name = "main"
    ITEMS = (
        ("resume", "Start exploring", "Jump into the city", "play"),
        ("garage", "Garage", "Spawn and drive any vehicle", "car"),
        ("travel", "Travel", "Visit a city or open the world map", "globe"),
        ("settings", "Settings", "Graphics, camera, world, audio and more", "sliders"),
        ("controls", "Controls", "Keyboard and mouse reference", "keyboard"),
        ("about", "About", "Version, data sources and credits", "info"),
        ("quit", "Quit", "Close OSM 3D", "power"),
    )

    def on_open(self):
        self.focus = 0

    def _items(self):
        info = self.menu.host.menu_info()
        out = []
        for ident, label, desc, icon in self.ITEMS:
            enabled, note = True, ""
            if ident == "resume":
                if not self.menu.title:
                    label, desc = "Resume", "Return to the city"
                elif not info.get("world"):
                    label, desc = ("Open world map", "Choose where to go") if not info.get("loading") \
                        else ("Loading…", "Your city is being prepared")
                    icon = "globe" if not info.get("loading") else "clock"
                    enabled = not info.get("loading")
            elif ident == "garage" and not info.get("world"):
                enabled, note = False, "Load a city first"
            out.append((ident, label, desc, icon, enabled, note))
        return out

    # ------------------------------------------------------------- drawing
    def draw(self, ui, w, h, mouse):
        host, menu = self.menu.host, self.menu
        info = host.menu_info()
        wide = w >= 880
        x = 72.0 if wide else 28.0
        item_w = min(430.0, w - 2 * x)
        top = 64.0
        # wordmark
        logo_mark(ui, x, top, 46.0)
        ui.text("OSM 3D", x + 62, top - 2, T.TITLE_FONT, T.TEXT, weight="bold")
        ui.text("City explorer", x + 64, top + 36, T.T_BODY, T.TEXT_3, weight="medium")
        if menu.title is False:
            chip(ui, x + 64 + ui.text_w("City explorer", T.T_BODY, weight="medium") + 12, top + 36,
                 "PAUSED", T.WARN, T.T_CAPTION, h=20, pad=8)

        items = self._items()
        avail = h - 70 - (top + 120) - 20
        item_h = max(40.0, min(56.0, avail / len(items) - 6))
        gap = 6.0
        y0 = top + 120
        for i, (ident, label, desc, icon, enabled, note) in enumerate(items):
            e = ease_out((menu.t - i * 0.045) / 0.32)
            ix = x - (1.0 - e) * 34.0
            r = (ix, y0 + i * (item_h + gap), item_w, item_h)
            self.hits.add(r, "item", i) if enabled else None
            hov = ui.anim(("main", i), float(inside(r, *mouse) and enabled), 18.0)
            sel = self.focus == i
            self._draw_item(ui, r, label, desc, icon, enabled, note, sel, hov, e)
        # footer
        ui.text(f"v{getattr(host, 'version', '1.0.0')}", x, h - 98, T.T_SMALL, T.TEXT_4, weight="medium")
        if wide:
            self._draw_session(ui, w, h, info)

    def _draw_item(self, ui, r, label, desc, icon, enabled, note, sel, hov, e):
        x, y, w, h = r
        if sel and enabled:
            ui.box(x, y, w, h, T.R_CARD, ((T.ACCENT[0], T.ACCENT[1], T.ACCENT[2], 0.30),
                                         (T.ACCENT[0], T.ACCENT[1], T.ACCENT[2], 0.08)),
                   alpha(T.SIGNAL, 0.55), bw=1.0)
            ui.box(x, y + 8, 3.5, h - 16, 1.75, T.SIGNAL)
        elif hov > 0.01:
            ui.box(x, y, w, h, T.R_CARD, ((1, 1, 1, 0.06 * hov), (1, 1, 1, 0.025 * hov)))
        col = T.TEXT if enabled else T.TEXT_4
        isz = 24
        ix = x + 18 + isz / 2
        tile = h - 16
        ui.box(x + 12, y + 8, tile, tile, 9, (1, 1, 1, 0.07 if enabled else 0.03))
        icons.draw(ui, icon, x + 12 + tile / 2, y + h / 2, isz, T.SIGNAL if (sel and enabled) else col, 1.7)
        tx = x + 12 + tile + 14
        big = h >= 50
        ui.text(label, tx, y + (9 if big else 5), T.T_LABEL + (2 if big else 0), col, weight="semibold")
        sub = note if note else desc
        if big:
            ui.text(ui.fit(sub, T.T_SMALL, w - (tx - x) - 12), tx, y + h - 22, T.T_SMALL,
                    T.WARN if (note and not enabled) else T.TEXT_3, weight="medium")

    def _draw_session(self, ui, w, h, info):
        pw = 292.0
        x, y = w - 72 - pw, 64.0
        rows = [("Location", info.get("location") or "–", "pin"),
                ("Local time", info.get("time") or "–", "clock"),
                ("Weather", info.get("weather") or "–", "cloud"),
                ("Mode", info.get("mode") or "–", "route")]
        if info.get("vehicle"):
            rows.append(("Vehicle", info["vehicle"], "car"))
        ph = 58 + len(rows) * 44
        ui.container(x, y, pw, ph, "panel", r=T.R_PANEL)
        heading(ui, "Current session", x + 20, y + 18)
        if info.get("coords"):
            ui.text(info["coords"], x + pw - 20, y + 18, T.T_CAPTION, T.TEXT_4, anchor="right", weight="medium")
        for i, (k, v, ic) in enumerate(rows):
            ry = y + 48 + i * 44
            if i:
                ui.divider(x + 20, ry - 3, pw - 40)
            icons.draw(ui, ic, x + 30, ry + 19, 18, T.TEXT_3, 1.5)
            ui.text(k, x + 50, ry + 4, T.T_CAPTION, T.TEXT_3, weight="medium")
            ui.text(ui.fit(v, T.T_BASE, pw - 70), x + 50, ry + 19, T.T_BASE, T.TEXT, weight="semibold")

    # --------------------------------------------------------------- input
    def prompts(self):
        return [("↑ ↓", "Navigate"), ("Enter", "Select"),
                ("Esc", "Quit" if self.menu.title else "Resume")]

    def _activate(self, ident):
        m = self.menu
        m.sound("click")
        if ident == "resume":
            info = m.host.menu_info()
            if m.title and not info.get("world"):
                m.title = False
                m.close()
                m.host.menu_open_map()
            else:
                m.resume()
        elif ident == "garage":
            m.go("garage")
        elif ident == "travel":
            m.go("travel")
        elif ident == "settings":
            m.go("settings")
        elif ident == "controls":
            m.go("settings")
            m.pages["settings"].select("controls")
        elif ident == "about":
            m.go("about")
        elif ident == "quit":
            m.ask_quit()

    def key(self, ev, pg):
        items = self._items()
        n = len(items)
        k = ev.key
        if k in (pg.K_DOWN, pg.K_s, pg.K_TAB):
            step = -1 if (k == pg.K_TAB and getattr(ev, "mod", 0) & pg.KMOD_SHIFT) else 1
            self._move(items, step)
        elif k in (pg.K_UP, pg.K_w):
            self._move(items, -1)
        elif k == pg.K_HOME:
            self.focus = 0
        elif k == pg.K_END:
            self.focus = n - 1
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
            ident, enabled = items[self.focus][0], items[self.focus][4]
            if enabled:
                self._activate(ident)
            else:
                self.menu.sound("error")
        elif k == pg.K_ESCAPE:
            self.menu.back()
        else:
            return False
        return True

    def _move(self, items, step):
        n = len(items)
        for _ in range(n):
            self.focus = (self.focus + step) % n
            if items[self.focus][4]:
                break
        self.menu.sound("click")

    def hover(self, pos):
        hit = self.hits.at(pos)
        if hit and hit[0] == "item" and hit[1] != self.focus:
            self.focus = hit[1]

    def press(self, pos):
        hit = self.hits.at(pos)
        if hit and hit[0] == "item":
            self.focus = hit[1]
            self._activate(self._items()[hit[1]][0])
            return True
        return False


# --------------------------------------------------------------------- travel
class TravelPage(Page):
    name = "travel"

    def on_open(self):
        self.focus = 0
        self.cols = 2

    def _grid(self, w, h):
        fx, fy, fw, fh = frame_rect(w, h, 920, 640)
        fy += self.menu.oy()
        cols = 2 if fw < 760 else 2
        pad = 28
        cw = (fw - 2 * pad - 14) / cols
        ch = 62
        top = fy + 100
        rects = []
        for i in range(len(CITIES)):
            c, r = i % cols, i // cols
            rects.append((fx + pad + c * (cw + 14), top + r * (ch + 10), cw, ch))
        return (fx, fy, fw, fh), rects, cols, top + math.ceil(len(CITIES) / cols) * (ch + 10)

    def draw(self, ui, w, h, mouse):
        (fx, fy, fw, fh), rects, cols, bottom = self._grid(w, h)
        self.cols = cols
        ui.container(fx, fy, fw, fh, "popup", r=T.R_PANEL + 2)
        panel_header(ui, fx + 28, fy + 26, fw - 100, "Travel", "Jump to a city, or pick any place on the world map",
                     icon="globe")
        cr = close_button(ui, fx + fw - 28 - 32, fy + 26, mouse)
        self.hits.add(cr, "close")
        ui.divider(fx + 28, fy + 88, fw - 56)
        for i, (name, lat, lon) in enumerate(CITIES):
            r = rects[i]
            sel = self.focus == i
            hov = ui.anim(("trv", i), float(inside(r, *mouse)), 18.0)
            self.hits.add(r, "city", i)
            ui.box(*r, T.R_CARD, T.CARD_FILL_FOCUS if sel else T.CARD_FILL,
                   alpha(T.SIGNAL, 0.7) if sel else mix(T.BORDER_SOFT, T.BORDER_PANEL, hov))
            if sel:
                brackets(ui, r[0], r[1], r[2], r[3], 9, T.SIGNAL, 1.5, inset=-3)
            icons.draw(ui, "pin", r[0] + 30, r[1] + r[3] / 2, 22, T.SIGNAL if sel else T.TEXT_2, 1.6)
            ui.text(name, r[0] + 56, r[1] + 11, T.T_LABEL, T.TEXT, weight="semibold")
            ui.text(f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}  {abs(lon):.2f}°{'E' if lon >= 0 else 'W'}",
                    r[0] + 56, r[1] + 33, T.T_SMALL, T.TEXT_3, weight="medium")
            if i < 10:
                keycaps(ui, r[0] + r[2] - 14 - 21, r[1] + (r[3] - 21) / 2, str((i + 1) % 10), 11)
        by = bottom + 8
        bw = 190
        b1 = (fx + 28, by, bw, 40)
        b2 = (fx + 28 + bw + 12, by, bw + 20, 40)
        self.hits.add(b1, "map")
        self.hits.add(b2, "search")
        button(ui, b1, "Open world map", mouse, key="trv_map", kind="accent", icon="globe",
               focus=self.focus == len(CITIES))
        button(ui, b2, "Search for a place", mouse, key="trv_search", icon="pin",
               focus=self.focus == len(CITIES) + 1)

    def prompts(self):
        return [("↑ ↓ ← →", "Navigate"), ("Enter", "Travel"), ("1 – 0", "Quick travel"),
                ("Esc", "Back")]

    def _travel(self, i):
        self.menu.sound("click")
        self.menu.title = False
        self.menu.close()
        self.menu.host.menu_travel(i)

    def _extra(self, idx):
        self.menu.sound("click")
        self.menu.title = False
        self.menu.close()
        (self.menu.host.menu_open_map if idx == 0 else self.menu.host.menu_search)()

    def key(self, ev, pg):
        k, n = ev.key, len(CITIES)
        total = n + 2
        f = self.focus
        if k in (pg.K_RIGHT, pg.K_d):
            f = f + 1 if f < n - 1 or f >= n else f
            f = min(f, total - 1)
        elif k in (pg.K_LEFT, pg.K_a):
            f = max(0, f - 1)
        elif k in (pg.K_DOWN, pg.K_s):
            f = f + self.cols if f + self.cols < n else (n if f < n else min(f + 1, total - 1))
        elif k in (pg.K_UP, pg.K_w):
            f = f - self.cols if (f < n and f - self.cols >= 0) else (f if f < n else (n - 1 if f == n else n))
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
            if f < n:
                self._travel(f)
            else:
                self._extra(f - n)
            return True
        elif pg.K_1 <= k <= pg.K_9 or k == pg.K_0:
            self._travel(9 if k == pg.K_0 else k - pg.K_1)
            return True
        else:
            return False
        if f != self.focus:
            self.focus = f
            self.menu.sound("click")
        return True

    def hover(self, pos):
        hit = self.hits.at(pos)
        if hit and hit[0] == "city":
            self.focus = hit[1]
        elif hit and hit[0] in ("map", "search"):
            self.focus = len(CITIES) + (0 if hit[0] == "map" else 1)

    def press(self, pos):
        hit = self.hits.at(pos)
        if not hit:
            return False
        tag, data, _ = hit
        if tag == "city":
            self._travel(data)
        elif tag in ("map", "search"):
            self._extra(0 if tag == "map" else 1)
        elif tag == "close":
            self.menu.back()
        return True
