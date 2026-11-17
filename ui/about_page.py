"""About screen: version, what the app is, where its data comes from, and system details."""
from __future__ import annotations

from . import icons
from . import theme as T
from .menu_ui import Page, frame_rect
from .ui_kit import (button, close_button, group, heading, inside, logo_mark, paragraph, separator, sheet)

CREDITS = (
    ("Map data", "© OpenStreetMap contributors", "Buildings, roads and places, ODbL"),
    ("Terrain", "AWS Open Data elevation tiles", "Terrarium-encoded DEM, no key needed"),
    ("Stars", "HYG v4.1 star catalogue", "David Nash / Astronomy Nexus, CC BY-SA 4.0"),
    ("Typeface", "Inter", "The Inter Project Authors, SIL Open Font License 1.1"),
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

        sheet(ui, fx, fy, fw, fh)
        self.hits.add((fx, fy, fw, fh), "panel")

        cr = close_button(ui, fx + fw - 32 - 32, fy + 30, mouse)
        self.hits.add(cr, "close")

        # Identity block
        pad = 32
        logo_mark(ui, fx + pad, fy + 28, 60.0)
        ui.text("OSM 3D", fx + pad + 78, fy + 30, T.T_DISPLAY - 2, T.TEXT_PRIMARY, weight="bold")
        ui.text(f"Version {getattr(host, 'version', '1.0.0')}", fx + pad + 79, fy + 64, T.T_BODY, T.TEXT_TERTIARY,
                weight="medium")

        body_w = fw - 2 * pad
        paragraph(ui, "Explore real cities in 3D. Buildings, streets and terrain stream in from OpenStreetMap "
                      "while traffic, pedestrians, weather and a physically based sky bring them to life. "
                      "Fly over, walk the pavements, or take any car for a drive.",
                  fx + pad, fy + 108, min(body_w, 640), T.T_CALLOUT, T.TEXT_SECONDARY, lead=22)

        two = fw >= 760
        gap = 24
        colw = (body_w - (gap if two else 0)) / (2 if two else 1)
        top = fy + 196

        # Credits
        heading(ui, "Data & credits", fx + pad + 6, top)
        rh = 52
        gy = top + 24
        group(ui, fx + pad, gy, colw, rh * len(CREDITS))
        for i, (kind, name, note) in enumerate(CREDITS):
            ry = gy + i * rh
            ui.text(ui.fit(name, T.T_CALLOUT, colw - 130, weight="medium"), fx + pad + 16, ry + 9, T.T_CALLOUT,
                    T.TEXT_PRIMARY, weight="medium")
            ui.text(ui.fit(note, T.T_CAPTION_2, colw - 130), fx + pad + 16, ry + 30, T.T_CAPTION_2, T.TEXT_TERTIARY)
            ui.text_c(kind, fx + pad + colw - 16, ry + rh / 2, T.T_CAPTION_2, T.TEXT_TERTIARY, anchor="right",
                      weight="medium")
            if i < len(CREDITS) - 1:
                separator(ui, fx + pad + 16, ry + rh - 0.5, colw - 32)

        # System info
        sx = fx + pad + (colw + gap if two else 0)
        sy = top if two else gy + rh * len(CREDITS) + 20
        heading(ui, "This session", sx + 6, sy)
        diag = host.diagnostics() if hasattr(host, "diagnostics") else {}
        rows = (("Graphics card", diag.get("gpu", "–")), ("Frame rate", diag.get("fps", "–")),
                ("City chunks", diag.get("chunks", "–")), ("Geometry memory", diag.get("vram", "–")),
                ("Traffic", diag.get("traffic", "–")), ("Cache folder", diag.get("cache", "–")))
        kh = 44
        gy2 = sy + 24
        group(ui, sx, gy2, colw, kh * len(rows))
        for i, (k, v) in enumerate(rows):
            ry = gy2 + i * kh
            ui.text_c(k, sx + 16, ry + kh / 2, T.T_BODY, T.TEXT_SECONDARY)
            ui.text_c(ui.fit(str(v), T.T_BODY, colw - 150), sx + colw - 16, ry + kh / 2, T.T_BODY, T.TEXT_PRIMARY,
                      anchor="right", weight="medium")
            if i < len(rows) - 1:
                separator(ui, sx + 16, ry + kh - 0.5, colw - 32)

        # Actions
        by = fy + fh - 28 - 40
        b_log = (fx + pad, by, 176.0, 40.0)
        b_ctl = (fx + pad + 188, by, 150.0, 40.0)
        self.hits.add(b_log, "log")
        self.hits.add(b_ctl, "controls")
        button(ui, b_log, "Streaming log", mouse, key="ab_log", kind="secondary", icon="terminal",
               focus=self.focus == 0)
        button(ui, b_ctl, "Controls", mouse, key="ab_ctl", kind="secondary", icon="keyboard", focus=self.focus == 1)

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