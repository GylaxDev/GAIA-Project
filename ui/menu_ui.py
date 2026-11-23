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
from . import ui_kit
from ..core.config import CITIES
from .loading_ui import APP_NAME, TAGLINE
from .backdrop import Backdrop
from .ui_kit import (alpha, bar, button, caps, caps_w, card, chip, close_button, divider, dot_grid, ease_out_cubic,
                     ease_out_expo, focus_ring, frame, glass, group, heading, icon_tile, inside, keycaps, logo_mark,
                     material_box, mix, panel_header, paragraph, prompt_bar, scrim, scrollbar, separator, sheet,
                     spring, surface)

# Re-export for other modules
from .ui_kit import (button as _button, toggle, toggle_rect, slider, segmented, stepper,
                     chip as _chip, close_button as _close_button, logo_mark as _logo_mark,
                     panel_header as _panel_header, prompt_bar as _prompt_bar, scrim as _scrim)


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
    ph = min(max_h, h - bottom - 48)
    return ((w - pw) / 2, max(44.0, (h - bottom - ph) / 2 + 12), pw, ph)


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
        ui.rect(0, 0, w, h, (0.0, 0.0, 0.0, 0.72))
        pw = min(420, w - 48)
        lines = ui_kit.wrap(ui, self.message, T.T_BODY, pw - 56, "regular", 3)
        lead = 20
        ph = 30 + 26 + 14 + len(lines) * lead + 26 + 40 + 26
        x, y = (w - pw) / 2, (h - ph) / 2 - 20

        ui.rect(x, y, pw, ph, T.PANEL_BG)
        frame(ui, x, y, pw, ph, T.LINE, title=self.title)
        paragraph(ui, self.message, x + 28, y + 34, pw - 56, T.T_BODY, T.TEXT_SECONDARY, lead=lead, max_lines=3)

        gap, bh = 10, 38
        bw = (pw - 56 - gap) / 2
        by = y + ph - 26 - bh
        no_r = (x + 28, by, bw, bh)
        yes_r = (x + 28 + bw + gap, by, bw, bh)
        self.rects = {"no": no_r, "yes": yes_r}

        button(ui, no_r, self.no, mouse, key="cf_no", kind="secondary", focus=self.focus == 1)
        button(ui, yes_r, self.yes, mouse, key="cf_yes",
               kind="destructive" if self.danger else "primary", focus=self.focus == 0)

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
        self.backdrop = Backdrop()
        self.title = False               # True on the start screen, before the first "play"
        self.confirm = None
        self.t = 0.0                     # seconds since the current page opened
        self.clock = 0.0                 # seconds since the menu was created (drives ambient motion)
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
        self.confirm = ConfirmDialog(f"Leave {APP_NAME}?", "Your settings are saved automatically. See you soon!",
                                     yes="Quit", no="Keep playing", on_yes=self.host.menu_quit)
        self.sound("open")

    def resume(self):
        self.title = False
        self.close()
        self.host.menu_resume()

    # --------------------------------------------------------------- frame
    def draw(self, ui, w, h, mouse, dt):
        self.t += dt
        self.clock += dt
        self.mouse, self.size = mouse, (w, h)
        page = self.current
        if page is None:
            return
        page.hits.clear()
        page.mouse = mouse
        ui_kit.POINTER["down"] = self.pressed

        ui_kit.prepare(ui)
        host = self.host
        info = host.menu_info()
        ver = getattr(host, "version", "")
        ui_kit.CHROME["version"] = f"v{ver}" if ver else ""
        ui_kit.CHROME["user"] = info.get("user") or "guest"

        fade = ease_out_expo(self.t / 0.22)
        opaque = (not info.get("world")) or self.title        # nothing to show behind us: the dot globe
        tiles = self._tiles()
        r = min(w, h) * 0.27
        if opaque and self.page == "main":
            self.backdrop.draw(ui, w, h, self.clock, globe=(w / 2, h * 0.43, r), tiles=tiles)
        elif opaque:
            self.backdrop.draw(ui, w, h, self.clock, dim=0.9 if self.page == "travel" else 0.28,
                               globe=(w * 0.80, h * 0.60, min(w, h) * 0.46), rings=False, tiles=tiles,
                               show_globe=self.page != "travel")
        else:
            scrim(ui, w, h, fade, flat=True)
            dot_grid(ui, 0, 0, w, h, 28.0, (1.0, 1.0, 1.0, 0.10), 0.8)
        self._user_line(ui, info)
        page.draw(ui, w, h, mouse)
        prompt_bar(ui, w, h, page.prompts())

        if self.confirm is not None:
            self.confirm.draw(ui, w, h, mouse)

    def _tiles(self):
        """The host's OpenStreetMap tile manager, if it offers one (used to build the globe)."""
        src = getattr(self.host, "osm_tiles", None)
        try:
            return src() if callable(src) else src
        except Exception:
            return None

    @staticmethod
    def _user_line(ui, info):
        name = ui_kit.CHROME["user"]
        ui.ring(24, 13, 4.2, 1.4, ui_kit.WHITE)
        ui.line(15.5, 27, 17, 21, 1.4, ui_kit.WHITE)
        ui.line(17, 21, 31, 21, 1.4, ui_kit.WHITE)
        ui.line(31, 21, 32.5, 27, 1.4, ui_kit.WHITE)
        ui.text_c(name, 50, 20, T.T_SUBHEAD, ui_kit.WHITE, weight="medium")

    def oy(self):
        """Vertical slide-in offset for the current page (logical px)."""
        return (1.0 - ease_out_expo(self.t / 0.28)) * 16.0

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
        ("quit", "Quit", f"Close {APP_NAME}", "power"),
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
        x = 56.0 if wide else 28.0

        # Wordmark
        bob = 0.0
        logo_mark(ui, x, 62, 46.0)
        caps(ui, APP_NAME, x + 62, 66, 26, T.TEXT_PRIMARY, "bold", track=0.16)
        sub_w = ui.text("// " + self._greeting(info), x + 62, 96, T.T_BODY - 1, T.WD_GREEN, weight="medium")
        if menu.title is False:
            chip(ui, x + 62 + sub_w + 14, 95, "Paused", T.WD_AMBER, T.T_CAPTION_1, h=20, pad=8)

        # Menu: left-aligned bars, as wide as their text; the selected one is wider and blue
        items = self._items()
        bar_h, pitch = 34.0, 46.0
        pitch = min(pitch, (h - T.HINT_BAR_H - 150 - 170) / max(1, len(items)))
        bottom = h - T.HINT_BAR_H - 26
        y0 = bottom - len(items) * pitch + (pitch - bar_h)
        for i, (ident, label, desc, icon, enabled, note) in enumerate(items):
            tw = ui.text_w(label, T.T_SUBHEAD, weight="medium")
            slide = (1.0 - ease_out_expo((menu.t - 0.04 * i) / 0.30)) * -60.0
            base_w = max(168.0, tw + 40)
            ry = y0 + i * pitch
            r = (x + slide, ry, base_w, bar_h)
            if enabled:
                self.hits.add((x, ry, base_w + 26, bar_h), "item", i)
            hov = spring(ui, ("main", i), float(inside((x, ry, base_w + 26, bar_h), *mouse) and enabled), T.SPRING_GENTLE)
            lit = (self.focus == i and enabled) or hov > 0.5
            self._draw_item(ui, r, label, enabled, note, lit, tw)

        if wide:
            self._draw_session(ui, w, h, info)
            focused = items[self.focus] if 0 <= self.focus < len(items) else None
            if focused:
                self._draw_hint_panel(ui, w, h, focused)

    @staticmethod
    def _greeting(info):
        try:
            hour = int(str(info.get("time", "")).split(":")[0])
        except ValueError:
            return TAGLINE
        part = ("Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 18
                else "Good evening" if 18 <= hour < 23 else "Still up")
        return f"{part}, explorer"

    def _draw_item(self, ui, r, label, enabled, note, lit, tw):
        x, y, w, h = r
        if lit:
            w += 26
            ui.rect(x, y, w, h, T.BAR_SELECTED)
        else:
            ui.rect(x, y, w, h, T.BAR if enabled else (0.05, 0.05, 0.05, 1.0))
        col = T.TEXT_PRIMARY if enabled else T.TEXT_QUATERNARY
        ui.text_c(label, x + (36 if lit else 20), y + h / 2, T.T_SUBHEAD, col, weight="medium")
        if note and not enabled:
            ui.text_c(note, x + w + 12, y + h / 2, T.T_CAPTION_2, T.WD_AMBER, weight="medium")

    def _draw_session(self, ui, w, h, info):
        pw = 330.0
        x, y = w - 56 - pw, 70.0
        rows = [("Location", info.get("location")), ("Coordinates", info.get("coords")),
                ("Local time", info.get("time")), ("Weather", info.get("weather")),
                ("Mode", info.get("mode")), ("Vehicle", info.get("vehicle"))]
        rows = [(k, v) for k, v in rows if v and v != "–"]
        rh = 32
        ph = 30 + max(1, len(rows)) * rh + 12
        ui.rect(x, y, pw, ph, T.PANEL_BG)
        frame(ui, x, y, pw, ph, T.LINE, title="Current session")
        if not rows:
            ui.text("No city loaded yet.", x + 20, y + 34, T.T_BODY, T.TEXT_SECONDARY)
            ui.text("Open the world map or pick a", x + 20, y + 58, T.T_CAPTION_2 + 1, T.TEXT_TERTIARY)
            ui.text("destination to start exploring.", x + 20, y + 76, T.T_CAPTION_2 + 1, T.TEXT_TERTIARY)
            return
        for i, (k, v) in enumerate(rows):
            ry = y + 26 + i * rh
            caps(ui, k, x + 20, ry + rh / 2, T.T_CAPTION_1, T.TEXT_TERTIARY, "medium", vcenter=True)
            ui.text_c(ui.fit(str(v), T.T_BODY, pw - 150), x + pw - 20, ry + rh / 2, T.T_BODY, T.WD_GREEN,
                      anchor="right", weight="medium")

    def _draw_hint_panel(self, ui, w, h, item):
        ident, label, desc, icon, enabled, note = item
        pw, ph = 420.0, 108.0
        x, y = w - 56 - pw, h - T.HINT_BAR_H - 26 - ph
        ui.rect(x, y, pw, ph, T.PANEL_BG)
        caps(ui, label, x + 20, y + 18, T.T_SUBHEAD, T.TEXT_PRIMARY, "bold")
        paragraph(ui, note if (note and not enabled) else desc, x + 20, y + 48, pw - 40, T.T_BODY - 1,
                  T.TEXT_SECONDARY, lead=19, max_lines=2)
        if enabled:
            keycaps(ui, x + 20, y + ph - 34, "Enter", 11)
            caps(ui, "Access", x + 20 + 60, y + ph - 23.5, T.T_CAPTION_2 + 1, T.TEXT_PRIMARY, "bold", vcenter=True)

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
        self.cols = 1
        self.view = None                 # (lat, lon) the globe is turning towards

    def _grid(self, w, h):
        x = 56.0 if w >= 880 else 28.0
        bw = min(380.0, w - 2 * x)
        top = 104.0
        bar_h, pitch = 34.0, min(42.0, (h - T.HINT_BAR_H - top - 96) / max(1, len(CITIES)))
        rects = [(x, top + i * pitch, bw, bar_h) for i in range(len(CITIES))]
        return rects, top + len(CITIES) * pitch + 6, x, bw

    def draw(self, ui, w, h, mouse):
        rects, bottom, x, bw = self._grid(w, h)
        self.cols = 1
        menu = self.menu
        caps(ui, "// Travel", x, 62, 22, T.TEXT_PRIMARY, "bold", track=0.16)

        # The globe turns to the highlighted city
        cur = CITIES[self.focus] if 0 <= self.focus < len(CITIES) else None
        if cur is not None:
            if self.view is None:
                self.view = (cur[1], cur[2])
            lat, lon = self.view
            dlon = (cur[2] - lon + 180.0) % 360.0 - 180.0
            k = min(1.0, ui.dt * 5.0)
            self.view = (lat + (cur[1] - lat) * k, lon + dlon * k)
        if w >= 880:
            r = min(w, h) * 0.27
            gx, gy = w * 0.68, h * 0.46
            if getattr(menu, "backdrop", None) is not None and self.view is not None:
                menu.backdrop.globe.draw(ui, gx, gy, r, menu.clock, view=self.view)
            ui.rect(gx - 1, gy - 14, 2, 28, T.WD_GREEN)            # target reticle
            ui.rect(gx - 14, gy - 1, 28, 2, T.WD_GREEN)
            ui.ring(gx, gy, 9 + 3 * math.sin(menu.clock * 3.0), 1.5, T.WD_GREEN)
            if cur is not None:
                caps(ui, cur[0], gx, gy - r - 54, 24, T.TEXT_PRIMARY, "bold", anchor="center", track=0.12)
                ui.text(f"{abs(cur[1]):.2f}°{'N' if cur[1] >= 0 else 'S'}   {abs(cur[2]):.2f}°{'E' if cur[2] >= 0 else 'W'}",
                        gx, gy - r - 22, T.T_BODY, T.WD_GREEN, anchor="center", weight="medium")

        for i, (name, lat, lon) in enumerate(CITIES):
            r_ = rects[i]
            sel = self.focus == i
            hov = spring(ui, ("trv", i), float(inside(r_, *mouse)), T.SPRING_GENTLE)
            self.hits.add(r_, "city", i)
            lit = sel or hov > 0.5
            wdt = r_[2] if lit else max(168.0, ui.text_w(name, T.T_SUBHEAD, weight="medium") + 40)
            ui.rect(r_[0], r_[1], wdt, r_[3], T.BAR_SELECTED if lit else T.BAR)
            ui.text_c(name, r_[0] + (36 if lit else 20), r_[1] + r_[3] / 2, T.T_SUBHEAD, T.TEXT_PRIMARY, weight="medium")
            if i < 10:
                keycaps(ui, r_[0] + wdt - 14 - 21 if lit else r_[0] + wdt + 10, r_[1] + (r_[3] - 21) / 2,
                        str((i + 1) % 10), 11)

        by = bottom + 8
        b1 = (x, by, 190.0, 36.0)
        b2 = (x + 202, by, 210.0, 36.0)
        self.hits.add(b1, "map")
        self.hits.add(b2, "search")
        button(ui, b1, "Open world map", mouse, key="trv_map", kind="secondary", icon="globe",
               focus=self.focus == len(CITIES))
        button(ui, b2, "Search for a place", mouse, key="trv_search", kind="secondary", icon="pin",
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