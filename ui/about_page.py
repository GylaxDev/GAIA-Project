"""About screen: version, what the app is, where its data comes from, and system details."""
from __future__ import annotations

from . import icons
from . import theme as T
from .menu_ui import Page, frame_rect
from .ui_kit import alpha, button, chip, close_button, heading, inside, logo_mark, paragraph

CREDITS = (
    ("Map data", "© OpenStreetMap contributors", "Buildings, roads and places, ODbL"),
    ("Terrain", "AWS Open Data elevation tiles", "Terrarium-encoded DEM, no key needed"),
    ("Stars", "HYG v4.1 star catalogue", "David Nash / Astronomy Nexus, CC BY-SA 4.0"),
    ("Typeface", "Inter", "The Inter Project Authors, SIL Open Font License 1.1"),
    ("Engine sound", "Recorded V8 loops", "Baked from user-supplied recordings"),
)


class AboutPage(Page):
    name = "about"
    title = "About"

    def on_open(self):
        self.focus = 0

    def draw(self, ui, w, h, mouse):
        host = self.menu.host
        fx, fy, fw, fh = frame_rect(w, h, 940, 620)
        fy += self.menu.oy()
        ui.container(fx, fy, fw, fh, "popup", r=T.R_PANEL + 2)
        self.hits.add((fx, fy, fw, fh), "panel")
        cr = close_button(ui, fx + fw - 28 - 32, fy + 24, mouse)
        self.hits.add(cr, "close")

        # identity block
        pad = 30
        logo_mark(ui, fx + pad, fy + 28, 56.0)
        ui.text("OSM 3D", fx + pad + 74, fy + 26, T.T_HEAD + 8, T.TEXT, weight="bold")
        ui.text(f"Version {getattr(host, 'version', '1.0.0')}", fx + pad + 76, fy + 62, T.T_BODY, T.TEXT_3,
                weight="medium")
        body_w = fw - 2 * pad
        paragraph(ui, "Explore real cities in 3D. Buildings, streets and terrain stream in from OpenStreetMap "
                      "while traffic, pedestrians, weather and a physically based sky bring them to life. "
                      "Fly over, walk the pavements, or take any car for a drive.",
                  fx + pad, fy + 104, min(body_w, 640), T.T_BASE, T.TEXT_2, lead=22)
        ui.divider(fx + pad, fy + 176, body_w)

        two = fw >= 760
        colw = (body_w - (36 if two else 0)) / (2 if two else 1)
        # credits
        heading(ui, "Data & credits", fx + pad, fy + 196)
        y = fy + 222
        for kind, name, note in CREDITS:
            ui.text(kind.upper(), fx + pad, y + 2, T.T_CAPTION - 1, T.TEXT_4, weight="semibold")
            ui.text(ui.fit(name, T.T_BASE, colw - 110), fx + pad + 96, y - 1, T.T_BASE, T.TEXT, weight="semibold")
            ui.text(ui.fit(note, T.T_SMALL, colw - 110), fx + pad + 96, y + 19, T.T_SMALL, T.TEXT_3)
            y += 48
        # system
        sx = fx + pad + (colw + 36 if two else 0)
        sy = fy + 196 if two else y + 14
        heading(ui, "This session", sx, sy)
        diag = host.diagnostics() if hasattr(host, "diagnostics") else {}
        rows = (("Graphics card", diag.get("gpu", "–")), ("Frame rate", diag.get("fps", "–")),
                ("City chunks", diag.get("chunks", "–")), ("Geometry memory", diag.get("vram", "–")),
                ("Traffic", diag.get("traffic", "–")), ("Cache folder", diag.get("cache", "–")))
        ry = sy + 26
        for k, v in rows:
            ui.text(k, sx, ry + 1, T.T_SMALL, T.TEXT_3, weight="medium")
            ui.text(ui.fit(str(v), T.T_BODY, colw - 120), sx + colw, ry, T.T_BODY, T.TEXT_2, anchor="right",
                    weight="semibold")
            ui.divider(sx, ry + 25, colw, (1, 1, 1, 0.06))
            ry += 33
        # actions
        by = fy + fh - 28 - 36
        b_log = (fx + pad, by, 170.0, 36.0)
        b_ctl = (fx + pad + 182, by, 170.0, 36.0)
        self.hits.add(b_log, "log")
        self.hits.add(b_ctl, "controls")
        button(ui, b_log, "Streaming log", mouse, key="ab_log", kind="ghost", icon="terminal",
               focus=self.focus == 0)
        button(ui, b_ctl, "Controls", mouse, key="ab_ctl", kind="ghost", icon="keyboard", focus=self.focus == 1)

    def prompts(self):
        return [("← →", "Choose"), ("Enter", "Select"), ("Esc", "Back")]

    def _do(self, which):
        host = self.menu.host
        self.menu.sound("click")
        if which == 0:
            if hasattr(host, "menu_show_log"):
                host.menu_show_log()            # the log card is drawn above the menu
        else:
            self.menu.go("settings")
            self.menu.pages["settings"].select("controls")

    def key(self, ev, pg):
        k = ev.key
        if k in (pg.K_LEFT, pg.K_RIGHT, pg.K_a, pg.K_d, pg.K_TAB):
            self.focus ^= 1
            self.menu.sound("click")
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE):
            self._do(self.focus)
        else:
            return False
        return True

    def hover(self, pos):
        hit = self.hits.at(pos)
        if hit and hit[0] in ("log", "controls"):
            self.focus = 0 if hit[0] == "log" else 1

    def press(self, pos):
        hit = self.hits.at(pos)
        if not hit:
            return False
        if hit[0] == "close":
            self.menu.back()
        elif hit[0] in ("log", "controls"):
            self._do(0 if hit[0] == "log" else 1)
        return True
