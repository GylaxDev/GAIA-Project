"""Watch Dogs themed Settings Page - cyberpunk/hacker aesthetic.

Replaces the graphite settings with:
- Deep tech-black panels with cyan gradients
- Signature corner brackets on focus/hover
- Holographic scanline overlays
- Technical typography with tracking
- Glitch-style transitions
- Orange/Green/Purple accent variants for categories
"""
from __future__ import annotations

import math

from . import theme as T
from .menu_ui import Page, frame_rect
from .ui_kit import (heading, keycaps, toggle, slider, button, segmented, stepper,
                     panel_header, close_button, chip, mix, ease_out, inside, toggle_rect)
from .watchdogs_theme import (
    lerp, lerp_color, ease_out as wd_ease_out, stagger,
    WD_CYAN, WD_CYAN_DIM, WD_CYAN_BRIGHT, WD_CYAN_SOFT, WD_CYAN_FAINT,
    WD_ORANGE, WD_GREEN, WD_PURPLE,
    WD_BG_DEEP, WD_BG_DARK, WD_BG_MEDIUM, WD_BG_LIGHT,
    WD_PANEL_GRADIENT, WD_CARD_GRADIENT, WD_BUTTON_GRADIENT,
    WD_BUTTON_HOVER_GRADIENT, WD_BUTTON_PRESS_GRADIENT,
    WD_ACCENT_GRADIENT, WD_ACCENT_HOVER_GRADIENT,
    WD_BORDER_PRIMARY, WD_BORDER_SECONDARY, WD_BORDER_SUBTLE, WD_BORDER_FOCUS,
    WD_TEXT_PRIMARY, WD_TEXT_SECONDARY, WD_TEXT_MUTED, WD_TEXT_DISABLED,
    WD_OK, WD_WARN, WD_ERROR,
    WD_R_SMALL, WD_R_MEDIUM, WD_R_LARGE, WD_R_PILL, WD_R_CARD,
    WD_MARGIN, WD_GAP, WD_GAP_SM,
    WD_T_MICRO, WD_T_CAPTION, WD_T_SMALL, WD_T_BODY, WD_T_BASE, WD_T_LABEL, WD_DISPLAY,
    WD_ANIM_FAST, WD_ANIM_NORMAL, WD_ANIM_SLOW,
    WD_SHADOW_MEDIUM, WD_SHADOW_STRONG,
)
from .watchdogs_ui import (
    wd_brackets, wd_scanlines, wd_gradient_panel, wd_gradient_button,
    wd_toggle, wd_slider, wd_segmented, wd_card, wd_heading,
    wd_pill_chip, wd_scrim, wd_progress_bar,
)
from .watchdogs_icons import draw_wd


# Category accent colors for different setting groups
CAT_ACCENTS = {
    "display": WD_CYAN,
    "audio": WD_ORANGE,
    "gameplay": WD_GREEN,
    "controls": WD_PURPLE,
    "system": WD_CYAN,
    "graphics": WD_CYAN,
}

ROW_H, ROW_H_PLAIN = 56.0, 44.0
SECTION_H, SECTION_GAP = 30.0, 8.0
FOOTER_H = 56.0


class WatchDogsSettingsPage(Page):
    """Watch Dogs styled settings page with cyberpunk aesthetic."""
    name = "settings"
    title = "SYSTEM CONFIGURATION"

    def __init__(self, menu, store):
        super().__init__(menu)
        self.store = store
        self.cat = 0
        self.focus = 0
        self.scroll = {}
        self.scroll_max = 0.0
        self.drag_id = None
        self.slider_rects = {}
        self.row_pos = {}
        self._reveal = None
        self.view = (0.0, 0.0, 0.0, 0.0)
        self._anim_time = 0.0

    @property
    def category(self):
        cats = self.store.categories
        return cats[max(0, min(self.cat, len(cats) - 1))]

    def on_open(self):
        self.drag_id = None
        self.focus = 0
        self._reveal = None
        self._anim_time = 0.0

    def _set_category(self, index, quiet=False):
        n = len(self.store.categories)
        index %= max(1, n)
        if index != self.cat:
            self.cat = index
            self.focus = 0
            self._reveal = None
            self._anim_time = 0.0
            if not quiet:
                self.menu.sound("click")
        self.drag_id = None

    def _interactive(self):
        cat = self.category
        return [s for s in cat.settings() if s.kind != "info" and s.is_enabled()]

    def _geometry(self, w, h):
        fx, fy, fw, fh = frame_rect(w, h, 1180, 740)
        fy += self.menu.oy()
        compact = fw < 800
        rail_w = 72.0 if compact else 240.0
        cx = fx + 16 + rail_w + 28
        cw = fx + fw - 34 - cx
        vy = fy + 100
        vh = fh - 100 - FOOTER_H
        return dict(frame=(fx, fy, fw, fh), compact=compact, rail_w=rail_w, cx=cx, cw=cw,
                    vy=vy, vh=vh, footer_y=fy + fh - FOOTER_H)

    def _plan(self, cat, ui, cw):
        plan, y = [], 0.0
        if cat.custom == "controls":
            return self._plan_controls(ui, cw)
        for sec in cat.sections:
            plan.append(("section", sec.title, y, SECTION_H))
            y += SECTION_H
            for s in sec.rows:
                rh = ROW_H if s.desc else ROW_H_PLAIN
                plan.append(("row", s, y, rh))
                y += rh
            y += SECTION_GAP
        return plan, y

    def _plan_controls(self, ui, cw):
        shortcuts = getattr(self.menu.host, "SHORTCUTS", ())
        two = cw >= 700
        cols = 2 if two else 1
        row = 28.0
        key_w = max((self._key_width(ui, key) for _, rows in shortcuts for key, _ in rows), default=60.0) + 16
        heights = [22.0 + len(rows) * row + 12.0 for _, rows in shortcuts]
        columns, col_h = [[] for _ in range(cols)], [0.0] * cols
        for (title, rows), gh in zip(shortcuts, heights):
            k = col_h.index(min(col_h))
            columns[k].append((title, rows, col_h[k]))
            col_h[k] += gh
        plan = [("controls", dict(columns=columns, key_w=key_w, row=row, cols=cols), 0.0, max(col_h or [0.0]))]
        return plan, max(col_h or [0.0])

    @staticmethod
    def _key_width(ui, key):
        from .maps_ui import keycaps as kc
        return kc(ui, 0, 0, key, 10, dry=True)

    def draw(self, ui, w, h, mouse):
        self._anim_time += 1.0 / 60.0
        g = self._geometry(w, h)
        fx, fy, fw, fh = g["frame"]
        cats = self.store.categories
        cat = self.category
        cat_accent = CAT_ACCENTS.get(cat.id, WD_CYAN)

        # Full-screen scrim with scanlines
        wd_scrim(ui, w, h, 1.0)

        # Main frame with WD gradient
        wd_gradient_panel(ui, fx, fy, fw, fh, WD_PANEL_GRADIENT,
                          border=True, border_color=lerp_color(WD_BORDER_SECONDARY, cat_accent, 0.3),
                          radius=WD_R_LARGE, shadow=True)

        self.hits.add((fx, fy, fw, fh), "panel")

        # Rail (category sidebar)
        self._draw_rail(ui, g, mouse, cat_accent)

        # Content area
        cx, cw = g["cx"], g["cw"]

        # Header with category label
        self._draw_header(ui, cx, fy, cw, cat, cat_accent, mouse)

        # Divider
        ui.divider(cx, fy + 92, cw, lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.4))

        # Settings content
        plan, total = self._plan(cat, ui, cw)
        vy, vh = g["vy"], g["vh"]
        self.view = (cx - 12, vy, cw + 24, vh)
        self.scroll_max = max(0.0, total - vh)
        off = max(0.0, min(self.scroll_max, self.scroll.get(cat.id, 0.0)))
        self.row_pos = {item[1].id: (item[2], item[3]) for item in plan if item[0] == "row"}
        if self._reveal is not None and self._reveal in self.row_pos:
            ry, rh = self.row_pos[self._reveal]
            if ry < off:
                off = ry - 6
            elif ry + rh > off + vh:
                off = ry + rh - vh + 6
            off = max(0.0, min(self.scroll_max, off))
        self._reveal = None
        self.scroll[cat.id] = off

        ui.push_clip(*self.view)
        interactive = self._interactive()
        self.focus = max(0, min(self.focus, max(0, len(interactive) - 1)))
        focused = interactive[self.focus] if interactive else None

        for kind, obj, y, rh in plan:
            top = vy + y - off
            if top > vy + vh or top + rh < vy:
                continue
            if kind == "section":
                self._draw_section(ui, obj, cx, top, cw, cat_accent)
            elif kind == "row":
                self._draw_row(ui, obj, (cx, top, cw, rh), mouse, obj is focused, cat_accent)
            else:
                self._draw_controls(ui, obj, cx, top, cw, vy, vh, cat_accent)
        ui.pop_clip()

        # Scrollbar with WD style
        self._draw_scrollbar(ui, cx + cw + 8, vy, vh, vh, total, off, cat_accent)

        # Footer
        self._draw_footer(ui, g, mouse, cat, cat_accent)

    def _draw_rail(self, ui, g, mouse, cat_accent):
        fx, fy, fw, fh = g["frame"]
        rail_w, compact = g["rail_w"], g["compact"]
        rx, ry, rh = fx + 16, fy + 16, fh - 32

        # Rail background
        ui.box(rx, ry, rail_w, rh, WD_R_CARD - 2,
               (0.01, 0.015, 0.02, 0.4), lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.2))

        if compact:
            draw_wd(ui, "wd_settings", rx + rail_w / 2, ry + 36, 24, cat_accent, 1.8)
        else:
            # Category icon
            ui.box(rx + 16, ry + 16, 36, 36, 10, WD_CYAN_FAINT, alpha(cat_accent, 0.3))
            draw_wd(ui, "wd_settings", rx + 34, ry + 34, 22, cat_accent, 1.8)
            # Title
            wd_heading(ui, "SYSTEM", rx + 60, ry + 20, WD_TEXT_MUTED, WD_T_CAPTION)
            wd_heading(ui, "CONFIG", rx + 60, ry + 36, WD_TEXT_PRIMARY, WD_T_LABEL)

        item_h, gap = 40.0, 3.0
        y = ry + (60 if compact else 80)

        for i, c in enumerate(cats):
            r = (rx + 6, y + i * (item_h + gap), rail_w - 12, item_h)
            sel = i == self.cat
            hov = ui.anim(("wd_set_cat", i), float(inside(r, *mouse)), WD_ANIM_FAST)
            self.hits.add(r, "cat", i)

            accent = CAT_ACCENTS.get(c.id, WD_CYAN)

            if sel:
                ui.box(*r, WD_R_MEDIUM, WD_CARD_GRADIENT[0], WD_CARD_GRADIENT[1])
                ui.box(*r, WD_R_MEDIUM, None, lerp_color(WD_BORDER_SECONDARY, accent, 0.5), bw=1.5 / ui.scale)
                # Left accent bar
                ui.box(r[0] - 2, r[1] + 6, 2.5, r[3] - 12, 1.25, accent)
                # Corner brackets
                wd_brackets(ui, r[0], r[1], r[2], r[3], arm=8.0, color=accent, animated=True, phase=1.0)
            elif hov > 0.01:
                ui.box(*r, WD_R_MEDIUM,
                       lerp_color(WD_BG_DARK, accent, 0.1 * hov),
                       lerp_color(WD_BORDER_SUBTLE, accent, 0.3 * hov))

            col = lerp_color(WD_TEXT_PRIMARY, accent, 0.3) if sel else WD_TEXT_SECONDARY
            draw_wd(ui, c.icon, r[0] + (r[2] / 2 if compact else 22), r[1] + r[3] / 2, 20, col, 1.6)
            if not compact:
                ui.text_c(ui.fit(c.label, WD_T_BASE, r[2] - 52), r[0] + 40, r[1] + r[3] / 2, WD_T_BASE, col,
                          weight="semibold" if sel else "medium")

        # Navigation hint
        if not compact and ry + rh - 24 > y + len(cats) * (item_h + gap):
            kw = keycaps(ui, rx + 16, ry + rh - 30, "Q", 10)
            kw2 = keycaps(ui, rx + 16 + kw + 4, ry + rh - 30, "E", 10)
            ui.text_c("SWITCH CATEGORY", rx + 16 + kw + kw2 + 12, ry + rh - 21, WD_T_MICRO, WD_TEXT_MUTED, weight="medium")

    def _draw_header(self, ui, cx, fy, cw, cat, cat_accent, mouse):
        # Category icon badge
        ui.box(cx, fy + 20, 48, 48, 12, WD_CYAN_FAINT, alpha(cat_accent, 0.3))
        draw_wd(ui, cat.icon, cx + 24, fy + 44, 26, cat_accent, 1.8)

        # Labels
        wd_heading(ui, "CATEGORY", cx + 64, fy + 22, WD_TEXT_MUTED, WD_T_CAPTION)
        ui.text(cat.label, cx + 64, fy + 42, WD_T_HEAD + 2, WD_TEXT_PRIMARY, weight="bold")

        # Description
        ui.text(ui.fit(cat.blurb, WD_T_SMALL, cw - 120), cx + 64, fy + 70, WD_T_SMALL, WD_TEXT_SECONDARY, weight="medium")

        # Close button
        cr = (fx + cw - 28 - 32, fy + 20, 32, 32)
        cr = close_button(ui, cr[0], cr[1], mouse, size=32)
        self.hits.add(cr, "close")

    def _draw_section(self, ui, title, cx, top, cw, cat_accent):
        # Section heading with accent line
        wd_heading(ui, title, cx, top + 12, WD_TEXT_MUTED, WD_T_CAPTION)
        # Accent line extending from text
        tw = ui.text_w(title.upper(), WD_T_CAPTION, weight="semibold")
        ui.line(cx + tw + 16, top + 22, cx + cw, top + 22, 1.0 / ui.scale,
                lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.4))

    def _draw_row(self, ui, s, rect, mouse, focused, cat_accent):
        x, y, w, h = rect
        enabled = s.is_enabled()
        view_inside = inside(self.view, *mouse) if mouse else False
        hov = ui.anim(("wd_set_row", s.id),
                      float(inside(rect, *mouse) and enabled and view_inside), WD_ANIM_FAST)
        if enabled:
            self.hits.add(rect, "row", s, clip=self.view)

        # Row background on hover/focus
        if focused and enabled:
            ui.box(x - 8, y + 2, w + 16, h - 4, WD_R_MEDIUM, WD_CARD_GRADIENT[0], WD_CARD_GRADIENT[1])
            ui.box(x - 8, y + 2, w + 16, h - 4, WD_R_MEDIUM, None,
                   lerp_color(WD_BORDER_PRIMARY, cat_accent, 0.5), bw=1.5 / ui.scale)
            # Left accent bar
            ui.box(x - 10, y + 8, 2.5, h - 16, 1.25, cat_accent)
        elif hov > 0.01:
            ui.box(x - 8, y + 2, w + 16, h - 4, WD_R_MEDIUM,
                   lerp_color(WD_BG_DARK, cat_accent, 0.08 * hov),
                   lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.2 * hov))

        ctrl_w = min(360.0, w * 0.5)
        label_w = w - ctrl_w - 24
        col = WD_TEXT_PRIMARY if enabled else WD_TEXT_DISABLED
        ty = y + (10 if s.desc else 14)

        # Label with optional note chip
        lw = ui.text(ui.fit(s.label, WD_T_BASE, label_w, weight="semibold"), x, ty, WD_T_BASE, col, weight="semibold")
        if s.note:
            chip_w = chip(ui, x + min(lw, label_w) + 10, ty - 1, s.note, WD_TEXT_MUTED, WD_T_CAPTION - 1, h=16, pad=6)

        if s.desc:
            ui.text(ui.fit(s.desc, WD_T_SMALL, label_w), x, y + 30, WD_T_SMALL,
                    WD_TEXT_MUTED if enabled else WD_TEXT_DISABLED, weight="regular")

        ctrl = (x + w - ctrl_w, y, ctrl_w, h)

        if s.kind == "toggle":
            wd_toggle(ui, ctrl[0], ctrl[1], ctrl[2], ctrl[3],
                      bool(s.value()),
                      hover=ui.anim(("wd_tgl", s.id), float(enabled and inside(toggle_rect(ctrl), *mouse) and view_inside), WD_ANIM_FAST),
                      enabled=enabled)
            if enabled:
                self.hits.add(toggle_rect(ctrl), "toggle", s, clip=self.view)
        elif s.kind == "slider":
            vw = 72.0
            track = (ctrl[0] + 8, y, ctrl_w - vw - 16, h)
            self.slider_rects[s.id] = track
            hv = ui.anim(("wd_sl", s.id), float(enabled and (self.drag_id is s.id or inside(track, *mouse) and view_inside)), WD_ANIM_FAST)
            wd_slider(ui, track[0], track[1], track[2], track[3],
                      s.fraction(), hover=hv, color=cat_accent)
            ui.text_c(s.text(), x + w, y + h / 2, WD_T_BODY,
                      WD_TEXT_SECONDARY if enabled else WD_TEXT_DISABLED,
                      anchor="right", weight="semibold")
            if enabled:
                self.hits.add((track[0] - 8, y, track[2] + 16, h), "slider", s, clip=self.view)
        elif s.kind == "choice":
            self._draw_choice(ui, s, ctrl, mouse, focused, enabled, cat_accent)
        elif s.kind == "action":
            bw = max(100.0, ui.text_w(s.button, WD_T_BODY, weight="semibold") + 36)
            r = (x + w - bw, y + (h - 30) / 2, bw, 30.0)
            wd_gradient_button(ui, r[0], r[1], r[2], r[3], s.button,
                               color=cat_accent, hover=ui.anim(("wd_act", s.id), float(enabled and inside(r, *mouse) and view_inside), WD_ANIM_FAST),
                               disabled=not enabled)
            if enabled:
                self.hits.add(r, "action", s, clip=self.view)
        else:  # info
            ui.text_c(ui.fit(s.text(), WD_T_BASE, ctrl_w), x + w, y + h / 2, WD_T_BASE, WD_TEXT_SECONDARY,
                      anchor="right", weight="medium")

        # Divider
        ui.line(x, y + h - 1, x + w, y + h - 1, 1.0 / ui.scale,
                lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.15))

    def _draw_choice(self, ui, s, ctrl, mouse, focused, enabled, cat_accent):
        labels = [lbl for _, lbl in s.choices]
        idx = s.choice_index()
        need = sum(ui.text_w(t, WD_T_SMALL, weight="semibold") + 22 for t in labels) + 6
        if len(labels) <= 6 and need <= ctrl[2]:
            rects = wd_segmented(ui, ctrl[0], ctrl[1], ctrl[2], ctrl[3],
                                 labels, idx,
                                 hover_idx=-1,  # would need mouse check per segment
                                 enabled=enabled)
            if enabled:
                for i, r in enumerate(rects):
                    self.hits.add(r, "seg", (s, i), clip=self.view)
        else:
            arrows = stepper(ui, ctrl, labels[idx] if labels else "", mouse, key=s.id,
                             focus=focused and enabled, enabled=enabled, width=min(240.0, ctrl[2]))
            if enabled:
                self.hits.add(arrows[0], "step", (s, -1), clip=self.view)
                self.hits.add(arrows[1], "step", (s, +1), clip=self.view)

    def _draw_controls(self, ui, spec, cx, top, cw, vy, vh, cat_accent):
        cols, row, key_w = spec["cols"], spec["row"], spec["key_w"]
        gap = 32.0
        col_w = (cw - gap * (cols - 1)) / cols
        for ci, groups in enumerate(spec["columns"]):
            x = cx + ci * (col_w + gap)
            for title, rows, y0 in groups:
                y = top + y0
                if y < vy + vh and y + 20 > vy:
                    wd_heading(ui, title, x, y, WD_TEXT_MUTED, WD_T_CAPTION)
                y += 22
                for key, what in rows:
                    if y < vy + vh and y + row > vy:
                        keycaps(ui, x, y + (row - 19) / 2, key, 10)
                        ui.text_c(ui.fit(what, WD_T_BODY, col_w - key_w), x + key_w, y + row / 2, WD_T_BODY,
                                  WD_TEXT_SECONDARY, weight="regular")
                    y += row
            if ci == 0 and cols == 2:
                ui.line(x + col_w + gap / 2, top, x + col_w + gap / 2, top + spec_height(spec),
                        1.0 / ui.scale, lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.2))

    def _draw_scrollbar(self, ui, x, y, vh, view_h, content_h, offset, cat_accent):
        if content_h <= view_h + 1:
            return
        th = max(32.0, vh * view_h / content_h)
        ty = y + (vh - th) * (offset / max(1.0, content_h - view_h))
        # Scrollbar track
        ui.box(x, y, 3, vh, 1.5, WD_BG_DARK, WD_BORDER_SUBTLE)
        # Thumb with accent
        ui.box(x, ty, 3, th, 1.5, lerp_color(WD_CYAN_DIM, cat_accent, 0.4), lerp_color(WD_CYAN, cat_accent, 0.6))

    def _draw_footer(self, ui, g, mouse, cat, cat_accent):
        fx, fy, fw, fh = g["frame"]
        cx, cw, fyy = g["cx"], g["cw"], g["footer_y"]

        # Divider
        ui.line(cx, fyy + 6, cx + cw, fyy + 6, 1.0 / ui.scale,
                lerp_color(WD_BORDER_SUBTLE, cat_accent, 0.2))

        if cat.custom is None:
            r = (cx, fyy + 14, 180.0, 32.0)
            self.hits.add(r, "reset")
            wd_gradient_button(ui, r[0], r[1], r[2], r[3], "RESET TO DEFAULTS",
                               color=WD_ORANGE, hover=ui.anim(("wd_rst", cat.id), float(inside(r, *mouse)), WD_ANIM_FAST))

        # Hint text
        if cat.custom == "controls":
            txt = "HOLD Q WHILE ON FOOT TO PROFILE NEARBY ENTITIES"
        else:
            txt = "CHANGES APPLY INSTANTLY AND ARE SAVED AUTOMATICALLY"
        ui.text_c(ui.fit(txt, WD_T_CAPTION, cw - 190), cx + cw, fyy + 32, WD_T_CAPTION, WD_TEXT_MUTED,
                  anchor="right", weight="medium")

    def prompts(self):
        return [("↑ ↓", "SELECT"), ("← →", "ADJUST"), ("ENTER", "TOGGLE"), ("TAB", "NEXT CATEGORY"), ("ESC", "BACK")]

    def _commit(self, s, value, sound=None):
        self.store.change(s, value)
        if sound == "toggle":
            self.menu.sound("toggle", bool(s.value()))
        elif sound:
            self.menu.sound(sound)

    def _nudge(self, s, direction):
        if s.kind == "slider":
            step = s.step if s.step > 0 else (s.hi - s.lo) / 20.0
            self._commit(s, s.quantize(float(s.value() or 0.0) + direction * step))
        elif s.kind == "choice" and s.choices:
            i = max(0, min(len(s.choices) - 1, s.choice_index() + direction))
            if i != s.choice_index():
                self._commit(s, s.choices[i][0], "click")
        elif s.kind == "toggle":
            want = direction > 0
            if bool(s.value()) != want:
                self._commit(s, want, "toggle")

    def _activate(self, s):
        if s.kind == "toggle":
            self._commit(s, not bool(s.value()), "toggle")
        elif s.kind == "choice" and s.choices:
            i = (s.choice_index() + 1) % len(s.choices)
            self._commit(s, s.choices[i][0], "click")
        elif s.kind == "action" and s.action is not None:
            self.menu.sound("click")
            s.action()

    def key(self, ev, pg):
        k = ev.key
        rows = self._interactive()
        if k in (pg.K_q, pg.K_e, pg.K_TAB):
            back = k == pg.K_q or (k == pg.K_TAB and getattr(ev, "mod", 0) & pg.KMOD_SHIFT)
            self._set_category(self.cat + (-1 if back else 1))
        elif k in (pg.K_DOWN, pg.K_s) and rows:
            self.focus = min(len(rows) - 1, self.focus + 1)
            self._reveal = rows[self.focus].id
            self.menu.sound("click")
        elif k in (pg.K_UP, pg.K_w) and rows:
            self.focus = max(0, self.focus - 1)
            self._reveal = rows[self.focus].id
            self.menu.sound("click")
        elif k in (pg.K_LEFT, pg.K_a) and rows:
            self._nudge(rows[min(self.focus, len(rows) - 1)], -1)
        elif k in (pg.K_RIGHT, pg.K_d) and rows:
            self._nudge(rows[min(self.focus, len(rows) - 1)], +1)
        elif k in (pg.K_RETURN, pg.K_KP_ENTER, pg.K_SPACE) and rows:
            self._activate(rows[min(self.focus, len(rows) - 1)])
        elif k in (pg.K_PAGEDOWN, pg.K_PAGEUP):
            self._scroll(self.view[3] * 0.8 * (1 if k == pg.K_PAGEDOWN else -1))
        elif k in (pg.K_HOME, pg.K_END):
            self.scroll[self.category.id] = 0.0 if k == pg.K_HOME else self.scroll_max
        else:
            return False
        self.store.save()
        return True

    def _scroll(self, delta):
        cid = self.category.id
        self.scroll[cid] = max(0.0, min(self.scroll_max, self.scroll.get(cid, 0.0) + delta))

    def wheel(self, dy, pos):
        self._scroll(-dy * 50.0)

    def hover(self, pos):
        hit = self.hits.at(pos)
        if hit and hit[0] in ("row", "toggle", "slider", "seg", "step", "action"):
            s = hit[1][0] if hit[0] in ("seg", "step") else hit[1]
            rows = self._interactive()
            if s in rows:
                self.focus = rows.index(s)

    def _slide(self, pos):
        s = self.drag_id
        if s is None or not s.is_enabled():
            return
        r = self.slider_rects.get(s.id)
        if r is None:
            return
        frac = max(0.0, min(1.0, (pos[0] - r[0]) / max(1.0, r[2])))
        v = s.quantize(s.lo + (s.hi - s.lo) * frac)
        if v != s.value():
            self.store.change(s, v)

    def press(self, pos):
        hit = self.hits.at(pos)
        if not hit:
            return False
        tag, data, _ = hit
        if tag == "cat":
            self._set_category(data)
        elif tag == "close":
            self.menu.back()
        elif tag == "toggle":
            self._commit(data, not bool(data.value()), "toggle")
        elif tag == "slider":
            self.drag_id = data
            self._slide(pos)
        elif tag == "seg":
            s, i = data
            if i != s.choice_index():
                self._commit(s, s.choices[i][0], "click")
        elif tag == "step":
            s, d = data
            i = (s.choice_index() + d) % len(s.choices)
            self._commit(s, s.choices[i][0], "click")
        elif tag == "action":
            self.menu.sound("click")
            data.action()
        elif tag == "reset":
            self.store.reset_category(self.category)
            self.menu.sound("click")
        elif tag == "row":
            pass
        self.store.save()
        return True

    def drag(self, pos):
        self._slide(pos)

    def release(self):
        if self.drag_id is not None:
            self.drag_id = None
            self.store.save()

    def hot(self, pos):
        hit = self.hits.at(pos)
        return hit is not None and hit[0] not in ("bg", "panel", "row")


def spec_height(spec):
    return max((sum(22.0 + len(rows) * spec["row"] + 12.0 for _, rows, _ in groups)
                for groups in spec["columns"]), default=0.0)