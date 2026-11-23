"""Live eye-adaptation controls and the F1 shortcut panel, in logical pixels."""
from __future__ import annotations

from . import theme as T
from .ui_kit import (_in, caps, mix, button, close_button, divider, heading, keycaps, sheet, slider, spring, toggle)


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
        size = T.T_CALLOUT if h > 700 else T.T_BODY
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
        sheet(ui, x, y, pw, ph)
        caps(ui, "// Settings & shortcuts", x + pad, y + 22, T.T_TITLE_2 - 2, T.TEXT_PRIMARY, "bold")
        if pw >= 550:
            state = f"{renderer.aa_samples}× MSAA" if renderer.aa_samples else "No AA"
            state += "  ·  HUD " + ("on" if show_hud else "off")
            ui.text_c(state, rects["close"][0] - 14, y + 33, T.T_CAPTION_2, T.TEXT_TERTIARY,
                      anchor="right", weight="medium")

        # Two-tab capsule control
        a, b = rects["settings_tab"], rects["shortcuts_tab"]
        track = (a[0], a[1], b[0] + b[2] - a[0], a[3])
        ui.box(*track, track[3] / 2, T.FIELD)
        for name, label in (("settings_tab", "Settings"), ("shortcuts_tab", "Shortcuts")):
            r = rects[name]
            hover = spring(ui, ("settings", name), float(_in(r, *mouse)), T.SPRING_GENTLE)
            active = name == self.tab + "_tab"
            if active:
                ui.rect(*r, T.WD_BLUE)
            ui.text_c(label, r[0] + r[2] / 2, r[1] + r[3] / 2, T.T_BODY,
                      T.TEXT_PRIMARY if active else mix(T.TEXT_SECONDARY, T.TEXT_PRIMARY, 0.6 * hover),
                      anchor="center", weight="semibold")
        r = rects["close"]
        close_button(ui, r[0], r[1], mouse, key="f1_close", size=r[2])
        if self.tab == "settings":
            self._draw_controls(ui, layout, mouse, renderer)
        else:
            self._draw_shortcuts(ui, layout)

    def _row_title(self, ui, x, y, text):
        ui.text_c(text, x, y, T.T_CALLOUT, T.TEXT_PRIMARY, weight="semibold")

    def _slider_row(self, ui, x, w, r, label, value, frac, key, live, shade):
        ui.text(label, x, r[1] - 22, T.T_BODY, shade, weight="medium")
        ui.text(value, x + w, r[1] - 21, T.T_CAPTION_2, shade, anchor="right", weight="medium")
        slider(ui, r, frac, key=key, enabled=live)

    def _draw_controls(self, ui, layout, mouse, renderer):
        rects = layout["rects"]
        x, y, w, h = layout["body"]
        self._row_title(ui, x, y + 15, "Eye adaptation")
        toggle(ui, rects["toggle"], bool(renderer.ae_enabled), key="f1_ae")
        color = T.TEXT_SECONDARY if renderer.ae_enabled else T.TEXT_QUATERNARY
        for name, label, value in (("brightness", "Brightness", f"{renderer.ae_brightness:+.2f} stops"),
                                   ("strength", "Adaptation strength", f"{renderer.ae_strength:.0%}")):
            attr, lo, hi, _ = self.SLIDERS[name]
            frac = max(0.0, min(1.0, (getattr(renderer, attr) - lo) / (hi - lo)))
            self._slider_row(ui, x, w, rects[name], label, value, frac, f"f1_{name}", renderer.ae_enabled, color)
        hint = ("Raise brightness to lift shaded streets." if renderer.ae_enabled
                else "Turn it on (or press H) to preview changes.")
        if not renderer.hdr_scene or not renderer.ae_ok:
            hint = "Eye adaptation is unavailable without HDR support."
        ui.text(ui.fit(hint, T.T_CAPTION_2, w), x, y + 172, T.T_CAPTION_2, T.TEXT_TERTIARY)
        ui.text(ui.fit("Brightness biases the target; strength scales the response.", T.T_CAPTION_2, w),
                x, y + 192, T.T_CAPTION_2, T.TEXT_TERTIARY)

        divider(ui, x, y + 212, w)
        self._row_title(ui, x, y + 228, "Motion blur")
        r = rects["motion_blur"]
        value = float(renderer.motion_blur)
        attr, lo, hi, _ = self.SLIDERS["motion_blur"]
        frac = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        on = value > 0.0
        self._slider_row(ui, x, w, r, "Shutter", "Off" if not on else f"{value:.2f}×", frac, "f1_blur", True,
                         T.TEXT_SECONDARY)
        ui.text(ui.fit("Smears fast camera and car motion; 0 turns it off.", T.T_CAPTION_2, w),
                x, y + 300, T.T_CAPTION_2, T.TEXT_TERTIARY)
        divider(ui, x, y + 320, w)
        self._row_title(ui, x, y + 345, "Chase camera FOV effect")
        toggle(ui, rects["fov_toggle"], bool(renderer.speed_fov), key="f1_fov")
        ui.text(ui.fit("Widens the view as the car speeds up.", T.T_CAPTION_2, w),
                x, y + 372, T.T_CAPTION_2, T.TEXT_TERTIARY)
        r = rects["reset"]
        button(ui, r, "Reset defaults", mouse, key="f1_reset", kind="secondary")
        ui.text_c("Esc / F1 to close", x + w, r[1] + r[3] / 2, T.T_CAPTION_2, T.TEXT_TERTIARY,
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
                    heading(ui, title, cx, cy)
                cy += 22
                for key, what in rows:
                    if y <= cy and cy + row <= y + h:
                        keycaps(ui, cx, cy + (row - 21) / 2, key, 11)
                        ui.text_c(ui.fit(what, size, col_w - key_w), cx + key_w,
                                  cy + row / 2, size, T.TEXT_SECONDARY, weight="regular")
                    cy += row
                cy += 8
