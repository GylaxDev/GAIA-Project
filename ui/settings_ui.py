"""Live eye-adaptation controls and the F1 shortcut panel, in logical pixels."""
from __future__ import annotations

from . import theme as T
from .maps_ui import _in, keycaps


class SettingsPanel:
    SLIDERS = {
        "brightness": ("ae_brightness", -1.0, 2.0, 0.05),
        "strength": ("ae_strength", 0.0, 1.0, 0.05),
        "motion_blur": ("motion_blur", 0.0, 1.5, 0.05),
    }

    def __init__(self, shortcuts):
        self.shortcuts = shortcuts
        self.tab = "settings"
        self.dragging = None
        self.scroll_offset = 0.0

    def layout(self, ui, w, h):
        """One geometry source for rendering, clicks, dragging and scrolling."""
        pad = 24 if w >= 500 else 16
        pw = min(860 if self.tab == "shortcuts" else 580, w - 2 * T.MARGIN)
        row = 26 if h > 700 else 22
        size = T.T_BASE if h > 700 else T.T_BODY
        half = (len(self.shortcuts) + 1) // 2
        cols = ((self.shortcuts[:half], self.shortcuts[half:]) if pw >= 760
                else (self.shortcuts,))
        key_w = max(keycaps(ui, 0, 0, key, 11, dry=True)
                    for _, rows in self.shortcuts for key, _ in rows) + 16
        content_h = (max(sum(len(rows) * row + 30 for _, rows in col) for col in cols)
                     if self.tab == "shortcuts" else 446)
        ph = min(content_h + 108 + pad, h - 2 * T.MARGIN)
        x, y = (w - pw) / 2, (h - ph) / 2
        body = (x + pad, y + 108, pw - 2 * pad, ph - 108 - pad)
        bx, by, bw, bh = body
        rects = {
            "panel": (x, y, pw, ph),
            "settings_tab": (x + pad, y + 58, 110, 32),
            "shortcuts_tab": (x + pad + 118, y + 58, 110, 32),
            "close": (x + pw - pad - 30, y + 18, 30, 30),
            "toggle": (bx + bw - 68, by, 68, 30),
            "brightness": (bx + 8, by + 66, bw - 16, 24),
            "strength": (bx + 8, by + 124, bw - 16, 24),
            "motion_blur": (bx + 8, by + 268, bw - 16, 24),
            "fov_toggle": (bx + bw - 68, by + 330, 68, 30),
            "reset": (bx, by + bh - 34, 132, 32),
        }
        return dict(rects=rects, body=body, pad=pad, cols=cols, row=row, size=size,
                    key_w=key_w, scroll_max=max(0.0, content_h - bh))

    def hit(self, ui, pos, w, h, renderer):
        rects = self.layout(ui, w, h)["rects"]
        actions = ["close", "settings_tab", "shortcuts_tab"]
        if self.tab == "settings":
            actions += ["toggle", "fov_toggle", "reset"]
            # The motion-blur slider stays live even with eye adaptation off.
            actions += [name for name in self.SLIDERS
                        if name == "motion_blur" or renderer.ae_enabled]
        return next((name for name in actions if _in(rects[name], *pos)), None)

    def press(self, ui, pos, w, h, renderer):
        self.release()
        action = self.hit(ui, pos, w, h, renderer)
        if action in ("settings_tab", "shortcuts_tab"):
            self.tab = action.removesuffix("_tab")
            self.scroll_offset = 0.0
        elif action == "toggle":
            renderer.ae_enabled = not renderer.ae_enabled
            renderer.reset_adaptation()
        elif action == "fov_toggle":
            renderer.speed_fov = not renderer.speed_fov
        elif action == "reset":
            renderer.speed_fov = True
            renderer.reset_eye_adaptation_settings()
            renderer.motion_blur = renderer.MOTION_BLUR_DEFAULT
        elif action in self.SLIDERS:
            self.dragging = action
            self.drag(ui, pos, w, h, renderer)
        return action

    def drag(self, ui, pos, w, h, renderer):
        if self.dragging is None:
            return
        rect = self.layout(ui, w, h)["rects"][self.dragging]
        attr, lo, hi, step = self.SLIDERS[self.dragging]
        frac = max(0.0, min(1.0, (pos[0] - rect[0]) / max(1.0, rect[2])))
        value = round((lo + (hi - lo) * frac) / step) * step
        setattr(renderer, attr, round(max(lo, min(hi, value)), 2))
        # Do not reset the meter while dragging: retain smooth adaptation.

    def release(self):
        self.dragging = None

    def scroll_by(self, delta, ui, w, h):
        if self.tab != "shortcuts":
            return
        layout = self.layout(ui, w, h)
        self.scroll_offset = max(0.0, min(layout["scroll_max"],
                                         self.scroll_offset - delta * layout["row"] * 3))

    def draw(self, ui, w, h, mouse, renderer, show_hud):
        layout = self.layout(ui, w, h)
        rects, pad = layout["rects"], layout["pad"]
        x, y, pw, ph = rects["panel"]
        ui.container(x, y, pw, ph, "popup", r=T.R_PANEL + 2)
        ui.text("Settings & shortcuts", x + pad, y + 22, T.T_TITLE + 1, T.TEXT, weight="semibold")
        if pw >= 550:
            state = f"{renderer.aa_samples}\u00d7 MSAA" if renderer.aa_samples else "No AA"
            state += "  \u00b7  HUD " + ("on" if show_hud else "off")
            ui.text_c(state, rects["close"][0] - 14, y + 33, T.T_SMALL, T.TEXT_3,
                      anchor="right", weight="medium")
        for name, label in (("settings_tab", "Settings"), ("shortcuts_tab", "Shortcuts")):
            r = rects[name]
            hover = ui.anim(("settings", name), float(_in(r, *mouse)), 20.0)
            ui.container(*r, "button", active=name == self.tab + "_tab", hover=hover, shadow=False)
            ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2, T.T_BODY, T.TEXT,
                      anchor="center", weight="medium")
        r = rects["close"]
        hover = ui.anim(("settings", "close"), float(_in(r, *mouse)), 20.0)
        ui.container(*r, "inset", hover=hover)
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
        ui.line(cx - 4, cy - 4, cx + 4, cy + 4, 1.5, T.TEXT_2)
        ui.line(cx - 4, cy + 4, cx + 4, cy - 4, 1.5, T.TEXT_2)
        ui.divider(x + pad, y + 98, pw - 2 * pad)
        if self.tab == "settings":
            self._draw_controls(ui, layout, mouse, renderer)
        else:
            self._draw_shortcuts(ui, layout)

    def _draw_controls(self, ui, layout, mouse, renderer):
        rects = layout["rects"]
        x, y, w, h = layout["body"]
        ui.text_c("Eye adaptation", x, y + 15, T.T_LABEL, T.TEXT, weight="semibold")
        r = rects["toggle"]
        hover = ui.anim(("settings", "toggle"), float(_in(r, *mouse)), 20.0)
        ui.container(*r, "button", active=renderer.ae_enabled, hover=hover, shadow=False)
        ui.text_c("On" if renderer.ae_enabled else "Off", r[0] + r[2] / 2, r[1] + r[3] / 2,
                  T.T_BODY, T.TEXT, anchor="center", weight="semibold")
        color = T.TEXT_2 if renderer.ae_enabled else T.TEXT_4
        for name, label, value in (("brightness", "Brightness", f"{renderer.ae_brightness:+.2f} stops"),
                                   ("strength", "Adaptation strength", f"{renderer.ae_strength:.0%}")):
            r = rects[name]
            ui.text(label, x, r[1] - 20, T.T_BODY, color, weight="medium")
            ui.text(value, x + w, r[1] - 20, T.T_SMALL, color, anchor="right", weight="medium")
            attr, lo, hi, _ = self.SLIDERS[name]
            frac = max(0.0, min(1.0, (getattr(renderer, attr) - lo) / (hi - lo)))
            track_color = T.ACCENT if renderer.ae_enabled else T.TEXT_4
            ui.bar(r[0], r[1] + 10, r[2], 4, frac, color=track_color)
            ui.circle(r[0] + r[2] * frac, r[1] + 12, 7, color)
        hint = ("Raise brightness to lift shaded streets." if renderer.ae_enabled
                else "Turn it on (or press H) to preview changes.")
        if not renderer.hdr_scene or not renderer.ae_ok:
            hint = "Eye adaptation is unavailable without HDR support."
        ui.text(ui.fit(hint, T.T_SMALL, w), x, y + 172, T.T_SMALL, T.TEXT_3)
        ui.text(ui.fit("Brightness biases the target; strength scales the response.", T.T_SMALL, w),
                x, y + 192, T.T_SMALL, T.TEXT_3)

        ui.divider(x, y + 212, w)
        ui.text_c("Motion blur", x, y + 222, T.T_LABEL, T.TEXT, weight="semibold")
        r = rects["motion_blur"]
        value = float(renderer.motion_blur)
        attr, lo, hi, _ = self.SLIDERS["motion_blur"]
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        on = value > 0.0
        ui.text("Shutter", x, r[1] - 20, T.T_BODY, T.TEXT_2, weight="medium")
        ui.text("Off" if not on else f"{value:.2f}\u00d7", x + w, r[1] - 20, T.T_SMALL,
                T.TEXT_2, anchor="right", weight="medium")
        ui.bar(r[0], r[1] + 10, r[2], 4, frac, color=T.ACCENT if on else T.TEXT_4)
        ui.circle(r[0] + r[2] * frac, r[1] + 12, 7, T.TEXT_2 if on else T.TEXT_4)
        ui.text(ui.fit("Smears fast camera and car motion; 0 turns it off.", T.T_SMALL, w),
                x, y + 300, T.T_SMALL, T.TEXT_3)
        ui.divider(x, y + 320, w)
        ui.text_c("Chase camera FOV effect", x, y + 345, T.T_LABEL, T.TEXT, weight="semibold")
        r = rects["fov_toggle"]
        hover = ui.anim(("settings", "fov_toggle"), float(_in(r, *mouse)), 20.0)
        ui.container(*r, "button", active=renderer.speed_fov, hover=hover, shadow=False)
        ui.text_c("On" if renderer.speed_fov else "Off", r[0] + r[2] / 2, r[1] + r[3] / 2,
                  T.T_BODY, T.TEXT, anchor="center", weight="semibold")
        ui.text(ui.fit("Widens the view as the car speeds up.", T.T_SMALL, w),
                x, y + 372, T.T_SMALL, T.TEXT_3)
        r = rects["reset"]
        hover = ui.anim(("settings", "reset"), float(_in(r, *mouse)), 20.0)
        ui.container(*r, "button", hover=hover, shadow=False)
        ui.text_c("Reset defaults", r[0] + r[2] / 2, r[1] + r[3] / 2, T.T_BODY, T.TEXT_2,
                  anchor="center", weight="medium")
        ui.text_c("Esc / F1 to close", x + w, r[1] + r[3] / 2, T.T_SMALL, T.TEXT_3,
                  anchor="right", weight="medium")

    def _draw_shortcuts(self, ui, layout):
        x, y, w, h = layout["body"]
        row, size, key_w = layout["row"], layout["size"], layout["key_w"]
        self.scroll_offset = min(self.scroll_offset, layout["scroll_max"])
        cols, gap = layout["cols"], 32
        col_w = (w - gap * (len(cols) - 1)) / len(cols)
        for ci, col in enumerate(cols):
            cx, cy = x + ci * (col_w + gap), y - self.scroll_offset
            for title, rows in col:
                if y <= cy and cy + 18 <= y + h:
                    ui.text(title.upper(), cx, cy, T.T_CAPTION, T.TEXT_3, weight="semibold")
                cy += 22
                for key, what in rows:
                    if y <= cy and cy + row <= y + h:
                        keycaps(ui, cx, cy + (row - 21) / 2, key, 11)
                        ui.text_c(ui.fit(what, size, col_w - key_w), cx + key_w,
                                  cy + row / 2, size, T.TEXT_2, weight="regular")
                    cy += row
                cy += 8
            if ci == 0 and len(cols) == 2:
                ui.vdivider(x + col_w + gap / 2, y, h)
