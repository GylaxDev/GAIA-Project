"""The Settings screen: category tabs along the top, a scrolling list of live rows, and an info panel.

The page only draws and routes input.  Every value lives in a ``settings_store.Setting``
(built by ``settings_pages.build``), so a slider here and a keyboard shortcut elsewhere can
never disagree, and the store handles persistence.

Keyboard: Up/Down move between settings, Left/Right change the focused one, Enter or Space
toggles / runs it, Tab or Q/E switch category, PgUp/PgDn scroll, Esc goes back.
"""
from __future__ import annotations

import math

from . import icons
from . import theme as T
from .menu_ui import Page, frame_rect
from . import ui_kit
from .ui_kit import (alpha, bar, button, caps, caps_w, chip, close_button, ease_out_expo, frame, glass, heading,
                     inside, keycaps, mix, paragraph, scrollbar, segmented, slider, spring, stepper, toggle,
                     toggle_rect)

ROW_H, ROW_H_PLAIN = 46.0, 46.0
SECTION_H, SECTION_GAP = 30.0, 10.0
FOOTER_H = 0.0
TAB_Y, TAB_H = 96.0, 42.0
LIST_Y = 172.0


class SettingsPage(Page):
    name = "settings"
    title = "Settings"

    def __init__(self, menu, store):
        super().__init__(menu)
        self.store = store
        self.cat = 0                       # index into store.categories
        self.focus = 0                     # index into the current page's interactive rows
        self.scroll = {}                   # category id -> pixels scrolled
        self.scroll_max = 0.0
        self.drag_id = None
        self.slider_rects = {}
        self.row_pos = {}                  # setting id -> (y relative to content top, height)
        self._reveal = None
        self.view = (0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------ navigation
    @property
    def category(self):
        cats = self.store.categories
        return cats[max(0, min(self.cat, len(cats) - 1))]

    def on_open(self):
        self.drag_id = None
        self.focus = 0
        self._reveal = None

    def select(self, cat_id):
        for i, c in enumerate(self.store.categories):
            if c.id == cat_id:
                self._set_category(i)
                return

    def _set_category(self, index):
        n = len(self.store.categories)
        index %= max(1, n)
        if index != self.cat:
            self.cat = index
            self.focus = 0
            self._reveal = None
        self.drag_id = None

    def _interactive(self):
        cat = self.category
        return [s for s in cat.settings() if s.kind != "info" and s.is_enabled()]

    # ----------------------------------------------------------- layout math
    def _geometry(self, w, h):
        x0 = 56.0 if w >= 880 else 24.0
        wide = w >= 980
        list_w = (w - 2 * x0) * (0.62 if wide else 1.0)
        panel_x = x0 + list_w + 44
        vy = LIST_Y
        vh = h - T.HINT_BAR_H - vy - 18
        return dict(frame=(0.0, 0.0, w, h), compact=not wide, rail_w=0.0, cx=x0, cw=list_w, vy=vy, vh=vh,
                    footer_y=vy + vh, panel=(panel_x, vy, w - x0 - panel_x, 218.0) if wide else None, x0=x0)

    def _plan(self, cat, ui, cw):
        """Flat list of ``(kind, obj, y, h)`` for a category, plus the total height."""
        plan, y = [], 0.0
        if cat.custom == "controls":
            return self._plan_controls(ui, cw)
        for sec in cat.sections:
            heights = [ROW_H if r.desc else ROW_H_PLAIN for r in sec.rows]
            plan.append(("section", (sec.title, sum(heights)), y, SECTION_H))
            y += SECTION_H
            for s, rh in zip(sec.rows, heights):
                plan.append(("row", s, y, rh))
                y += rh
            y += SECTION_GAP
        return plan, y

    def _plan_controls(self, ui, cw):
        shortcuts = getattr(self.menu.host, "SHORTCUTS", ())
        two = cw >= 700
        cols = 2 if two else 1
        row = 36.0
        key_w = max((self._key_width(ui, key) for _, rows in shortcuts for key, _ in rows), default=60.0) + 16
        # distribute groups over columns, balancing heights
        heights = [30.0 + len(rows) * row + 16.0 for _, rows in shortcuts]
        columns, col_h = [[] for _ in range(cols)], [0.0] * cols
        for (title, rows), gh in zip(shortcuts, heights):
            k = col_h.index(min(col_h))
            columns[k].append((title, rows, col_h[k]))
            col_h[k] += gh
        plan = [("controls", dict(columns=columns, key_w=key_w, row=row, cols=cols), 0.0, max(col_h or [0.0]))]
        return plan, max(col_h or [0.0])

    @staticmethod
    def _key_width(ui, key):
        from .ui_kit import keycaps as kc
        return kc(ui, 0, 0, key, 11, dry=True)

    # ------------------------------------------------------------------ draw
    def draw(self, ui, w, h, mouse):
        g = self._geometry(w, h)
        cat = self.category
        self._draw_tabs(ui, g, mouse, w)

        cx, cw = g["cx"], g["cw"]
        plan, total = self._plan(cat, ui, cw)
        vy, vh = g["vy"], g["vh"]
        self.view = (cx - 4, vy - 6, cw + 8, vh + 6)
        self.scroll_max = max(0.0, total - vh)
        off = max(0.0, min(self.scroll_max, self.scroll.get(cat.id, 0.0)))
        self.row_pos = {item[1].id: (item[2], item[3]) for item in plan if item[0] == "row"}
        if self._reveal is not None and self._reveal in self.row_pos:
            ry, rh = self.row_pos[self._reveal]
            if ry < off:
                off = ry - 40
            elif ry + rh > off + vh:
                off = ry + rh - vh + 12
            off = max(0.0, min(self.scroll_max, off))
        self._reveal = None
        self.scroll[cat.id] = off

        ui.push_clip(*self.view)
        interactive = self._interactive()
        self.focus = max(0, min(self.focus, max(0, len(interactive) - 1)))
        focused = interactive[self.focus] if interactive else None

        for index, (kind, obj, y, rh) in enumerate(plan):
            top = vy + y - off
            if top > vy + vh or top + rh < vy - 6:
                continue
            if kind == "section":
                if obj[0]:
                    caps(ui, obj[0], cx, top + 6, T.T_FOOTNOTE, T.TEXT_TERTIARY, "semibold")
                # One rounded card holds the whole group, like a Settings list.
                glass(ui, cx, top + SECTION_H - 6, cw, obj[1] + 6, T.R_CARD, "menu")
            elif kind == "row":
                nxt = plan[index + 1][0] if index + 1 < len(plan) else None
                self._draw_row(ui, obj, (cx, top, cw, rh), mouse, obj is focused, last=nxt != "row")
            else:
                self._draw_controls(ui, obj, cx, top, cw, vy, vh)

        ui.pop_clip()
        scrollbar(ui, cx + cw + 10, vy, vh, vh, total, off)
        self._draw_panel(ui, g, mouse, cat, focused)

    def _draw_tabs(self, ui, g, mouse, w):
        x = g["x0"]
        ui.text("Settings", x, 40, 30, T.TEXT_PRIMARY, weight="semibold")
        x = g["x0"]
        pad = 22.0
        widths = [ui.text_w(c.label, T.T_CALLOUT, weight="medium") + 2 * pad
                  for c in self.store.categories]
        room = w - x - g["x0"]
        if sum(widths) > room:
            pad = max(8.0, pad * room / sum(widths))
            widths = [ui.text_w(c.label, T.T_CALLOUT, weight="medium") + 2 * pad
                      for c in self.store.categories]
        if sum(widths) > room:
            k = room / sum(widths)
            widths = [v * k for v in widths]
        total = sum(widths)
        ui.box(x, TAB_Y, total, TAB_H, TAB_H / 2, (1.0, 1.0, 1.0, 0.075), (1.0, 1.0, 1.0, 0.07))
        for i, c in enumerate(self.store.categories):
            r = (x, TAB_Y, widths[i], TAB_H)
            sel = i == self.cat
            hov = spring(ui, ("set_cat", i), float(inside(r, *mouse)), T.SPRING_GENTLE)
            self.hits.add(r, "cat", i)
            if sel:
                ui.box(r[0] + 2, r[1] + 2, r[2] - 4, r[3] - 4, (r[3] - 4) / 2,
                       (1.0, 1.0, 1.0, 0.94), None, shadow=8.0, shadow_dy=2.0, shadow_alpha=0.22)
            elif hov > 0.01:
                ui.box(r[0] + 2, r[1] + 2, r[2] - 4, r[3] - 4, (r[3] - 4) / 2,
                       (1.0, 1.0, 1.0, 0.07 * hov))
            ui.text_c(c.label, r[0] + r[2] / 2, r[1] + r[3] / 2, T.T_CALLOUT,
                      (0.06, 0.06, 0.07, 0.95) if sel else mix(T.TEXT_SECONDARY, T.WHITE, hov),
                      anchor="center", weight="semibold")
            x += widths[i]

    # ------------------------------------------------------------------ rows
    def _draw_row(self, ui, s, rect, mouse, focused, last=False):
        x, y, w, h = rect
        bh = h
        enabled = s.is_enabled()
        r = (x + 4, y + 2, w - 8, bh - 4)
        hov = spring(ui, ("set_row", s.id),
                     float(inside(r, *mouse) and enabled and inside(self.view, *mouse)), T.SPRING_GENTLE)
        lit = focused and enabled
        if enabled:
            self.hits.add(r, "row", s, clip=self.view)
        if hov > 0.01:
            ui.box(*r, T.R_5, (1.0, 1.0, 1.0, 0.07 * hov))
        if lit:
            ui.box(*r, T.R_5, (T.SYS_BLUE[0] * 0.55 + 0.10, T.SYS_BLUE[1] * 0.55 + 0.12,
                               T.SYS_BLUE[2] * 0.55 + 0.18, 0.90), (1.0, 1.0, 1.0, 0.12))
        elif not last:
            ui.rect(x + 18, y + bh - 1, w - 36, 1.0 / ui.scale, T.SEPARATOR)

        col = T.TEXT_PRIMARY if enabled else T.TEXT_QUATERNARY
        ctrl_w = min(390.0, w * 0.56)
        label_w = w - ctrl_w - 34
        lw = ui.text_c(ui.fit(s.label, T.T_CALLOUT, label_w, weight="medium"), x + 18, y + bh / 2, T.T_CALLOUT, col,
                       weight="medium")
        if s.note:
            chip(ui, x + 18 + min(lw, label_w) + 12, y + (bh - 20) / 2, s.note,
                 (1.0, 1.0, 1.0, 0.9) if lit else T.SYS_ORANGE, T.T_CAPTION_1, h=20, pad=8)

        ctrl = (x + w - ctrl_w - 14, y, ctrl_w, bh)
        key = s.id
        if s.kind == "toggle":
            toggle(ui, ctrl, bool(s.value()), key=key, focus=lit, enabled=enabled)
            if enabled:
                self.hits.add(toggle_rect(ctrl), "toggle", s, clip=self.view)
        elif s.kind == "slider":
            vw = 64.0
            track = (ctrl[0] + vw + 22, y, ctrl_w - vw - 22, bh)
            self.slider_rects[s.id] = track
            ui.text_c(s.text(), ctrl[0] + vw, y + bh / 2, T.T_BODY, T.TEXT_PRIMARY if enabled else T.TEXT_QUATERNARY,
                      anchor="right", weight="medium")
            slider(ui, track, s.fraction(), key=key, enabled=enabled)
            if enabled:
                self.hits.add((track[0] - 8, y, track[2] + 16, bh), "slider", s, clip=self.view)
        elif s.kind == "choice":
            self._draw_choice(ui, s, ctrl, mouse, lit, enabled)
        elif s.kind == "action":
            bw = max(96.0, ui.text_w(s.button, T.T_BODY, weight="medium") + 36)
            rr = (x + w - bw - 14, y + 5, bw, bh - 10)
            button(ui, rr, s.button, mouse, key=("act", s.id), kind="secondary", focus=lit, disabled=not enabled,
                   size=T.T_BODY)
            if enabled:
                self.hits.add(rr, "action", s, clip=self.view)
        else:                                       # info
            ui.text_c(ui.fit(s.text(), T.T_BODY, ctrl_w), x + w - 18, y + bh / 2, T.T_BODY, T.TEXT_SECONDARY,
                      anchor="right", weight="medium")

    def _draw_choice(self, ui, s, ctrl, mouse, focused, enabled):
        labels = [lbl for _, lbl in s.choices]
        idx = s.choice_index()
        need = sum(ui.text_w(t, T.T_CAPTION_2, weight="medium") + 24 for t in labels)
        if len(labels) <= 6 and need <= ctrl[2]:
            rects = segmented(ui, ctrl, labels, idx, mouse, key=s.id, focus=focused and enabled, enabled=enabled,
                              size=T.T_CAPTION_2 + 1)
            if enabled:
                for i, r in enumerate(rects):
                    self.hits.add(r, "seg", (s, i), clip=self.view)
        else:
            arrows = stepper(ui, ctrl, labels[idx] if labels else "", mouse, key=s.id,
                             focus=focused and enabled, enabled=enabled, width=min(250.0, ctrl[2]),
                             size=T.T_CAPTION_2 + 1)
            if enabled:
                self.hits.add(arrows[0], "step", (s, -1), clip=self.view)
                self.hits.add(arrows[1], "step", (s, +1), clip=self.view)

    def _draw_controls(self, ui, spec, cx, top, cw, vy, vh):
        cols, row, key_w = spec["cols"], spec["row"], spec["key_w"]
        gap = 24.0
        col_w = (cw - gap * (cols - 1)) / cols
        for ci, groups in enumerate(spec["columns"]):
            x = cx + ci * (col_w + gap)
            for title, rows, y0 in groups:
                y = top + y0
                gh = len(rows) * row
                if y < vy + vh and y + 30 + gh > vy:
                    caps(ui, title, x, y + 4, T.T_FOOTNOTE, T.TEXT_TERTIARY, "semibold")
                    glass(ui, x, y + 26, col_w, gh + 2, T.R_CARD, "menu")
                    for i, (key, what) in enumerate(rows):
                        ry = y + 28 + i * row
                        if i < len(rows) - 1:
                            ui.rect(x + 16, ry + row - 2, col_w - 32, 1.0 / ui.scale, T.SEPARATOR)
                        keycaps(ui, x + 12, ry + (row - 3 - 21) / 2, key, 11)
                        ui.text_c(ui.fit(what, T.T_BODY, col_w - 12 - key_w - 14), x + 12 + key_w, ry + (row - 3) / 2,
                                  T.T_BODY, T.TEXT_PRIMARY)

    def _draw_panel(self, ui, g, mouse, cat, focused):
        panel = g["panel"]
        bx = g["cx"] + g["cw"] + 44 if panel else g["cx"]
        if panel:
            px, py, pw, ph = panel
            glass(ui, px, py, pw, ph, T.R_CARD + 4, "menu")
            title = focused.label if focused is not None else cat.label
            ui.text(ui.fit(title, T.T_TITLE_2, pw - 44), px + 22, py + 20, T.T_TITLE_2,
                    T.TEXT_PRIMARY, weight="semibold")
            if cat.custom == "controls":
                text = "Every action, grouped by what it does. Press any key to try it."
            elif focused is not None:
                text = focused.desc or cat.blurb
            else:
                text = cat.blurb
            paragraph(ui, text, px + 22, py + 54, pw - 44, T.T_BODY, T.TEXT_SECONDARY, lead=21, max_lines=7)
            if focused is not None and focused.note:
                ui.text(focused.note, px + 22, py + ph - 34, T.T_CAPTION_2, T.SYS_ORANGE, weight="medium")
            by = py + ph + 14
        else:
            by = g["vy"] + g["vh"] + 4
            px, pw = g["cx"], 220.0
        if cat.custom is None:
            r = (px if panel else g["cx"], by, 180.0, 34.0) if panel else (g["cx"] + g["cw"] - 180.0, 92.0, 180.0, 28.0)
            self.hits.add(r, "reset")
            button(ui, r, "Reset page", mouse, key="set_reset", kind="secondary", icon="reset", size=T.T_BODY)

    # ------------------------------------------------------------------ input
    def prompts(self):
        return [("↑ ↓", "Select"), ("← →", "Adjust"), ("Enter", "Toggle"), ("Tab", "Next page"), ("Esc", "Back")]

    def _commit(self, s, value):
        self.store.change(s, value)

    def _nudge(self, s, direction):
        """Left / Right on the focused setting."""
        if s.kind == "slider":
            step = s.step if s.step > 0 else (s.hi - s.lo) / 20.0
            self._commit(s, s.quantize(float(s.value() or 0.0) + direction * step))
        elif s.kind == "choice" and s.choices:
            i = max(0, min(len(s.choices) - 1, s.choice_index() + direction))
            if i != s.choice_index():
                self._commit(s, s.choices[i][0])
        elif s.kind == "toggle":
            want = direction > 0
            if bool(s.value()) != want:
                self._commit(s, want)

    def _activate(self, s):
        if s.kind == "toggle":
            self._commit(s, not bool(s.value()))
        elif s.kind == "choice" and s.choices:
            i = (s.choice_index() + 1) % len(s.choices)
            self._commit(s, s.choices[i][0])
        elif s.kind == "action" and s.action is not None:
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
        elif k in (pg.K_UP, pg.K_w) and rows:
            self.focus = max(0, self.focus - 1)
            self._reveal = rows[self.focus].id
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
        self._scroll(-dy * 54.0)

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
                self._commit(s, s.choices[i][0])
        elif tag == "step":
            s, d = data
            i = (s.choice_index() + d) % len(s.choices)
            self._commit(s, s.choices[i][0])
        elif tag == "action":
            data.action()
        elif tag == "reset":
            self.store.reset_category(self.category)
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
    return max((sum(26.0 + len(rows) * spec["row"] + 14.0 for _, rows, _ in groups)
                for groups in spec["columns"]), default=0.0)

