"""The Settings screen: a category rail on the left, a scrolling page of live controls on the right.

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
from .ui_kit import (alpha, button, chip, close_button, divider, ease_out_expo, focus_ring, group, heading,
                     icon_tile, inside, keycaps, mix, panel_header, scrollbar, segmented, separator, sheet,
                     slider, spring, stepper, surface, toggle, toggle_rect)

ROW_H, ROW_H_PLAIN = 58.0, 48.0
SECTION_H, SECTION_GAP = 30.0, 16.0
FOOTER_H = 60.0
GROUP_PAD = 16.0      # inner horizontal padding of a grouped card
TILE_CYCLE = (T.TILE_BLUE, T.TILE_PURPLE, T.TILE_RED, T.TILE_ORANGE, T.TILE_GREEN, T.TILE_TEAL,
              T.TILE_INDIGO, T.TILE_PINK, T.TILE_GRAY)


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
                self._set_category(i, quiet=True)
                return

    def _set_category(self, index, quiet=False):
        n = len(self.store.categories)
        index %= max(1, n)
        if index != self.cat:
            self.cat = index
            self.focus = 0
            self._reveal = None
            if not quiet:
                self.menu.sound("click")
        self.drag_id = None

    def _interactive(self):
        cat = self.category
        return [s for s in cat.settings() if s.kind != "info" and s.is_enabled()]

    # ----------------------------------------------------------- layout math
    def _geometry(self, w, h):
        fx, fy, fw, fh = frame_rect(w, h, 1120, 700)
        fy += self.menu.oy()
        compact = fw < 760
        rail_w = 64.0 if compact else 216.0
        cx = fx + 20 + rail_w + 28
        cw = fx + fw - 32 - cx
        vy = fy + 96
        vh = fh - 96 - FOOTER_H
        return dict(frame=(fx, fy, fw, fh), compact=compact, rail_w=rail_w, cx=cx, cw=cw,
                    vy=vy, vh=vh, footer_y=fy + fh - FOOTER_H)

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
        row = 34.0
        key_w = max((self._key_width(ui, key) for _, rows in shortcuts for key, _ in rows), default=60.0) + 16
        # distribute groups over columns, balancing heights
        heights = [26.0 + len(rows) * row + 12.0 + 18.0 for _, rows in shortcuts]
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
        fx, fy, fw, fh = g["frame"]
        cat = self.category

        sheet(ui, fx, fy, fw, fh)
        self.hits.add((fx, fy, fw, fh), "panel")

        self._draw_rail(ui, g, mouse)

        cx, cw = g["cx"], g["cw"]
        panel_header(ui, cx, fy + 26, cw - 48, cat.label, cat.blurb)
        cr = close_button(ui, fx + fw - 32 - 32, fy + 30, mouse)
        self.hits.add(cr, "close")

        plan, total = self._plan(cat, ui, cw)
        vy, vh = g["vy"], g["vh"]
        self.view = (cx - 12, vy, cw + 24, vh)
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

        for i, (kind, obj, y, rh) in enumerate(plan):
            top = vy + y - off
            extent = rh + (obj[1] if kind == "section" else 0.0)
            if top > vy + vh or top + extent < vy:
                continue
            if kind == "section":
                title, gh = obj
                if title:
                    heading(ui, title, cx + 6, top + 4)
                group(ui, cx, top + rh, cw, gh)
            elif kind == "row":
                last = i + 1 >= len(plan) or plan[i + 1][0] != "row"
                self._draw_row(ui, obj, (cx, top, cw, rh), mouse, obj is focused, last)
            else:
                self._draw_controls(ui, obj, cx, top, cw, vy, vh)

        ui.pop_clip()
        scrollbar(ui, cx + cw + 12, vy, vh, vh, total, off)

        self._draw_footer(ui, g, mouse, cat)

    def _draw_rail(self, ui, g, mouse):
        fx, fy, fw, fh = g["frame"]
        rail_w, compact = g["rail_w"], g["compact"]
        rx, ry = fx + 20, fy + 26

        if not compact:
            ui.text("Settings", rx + 8, ry + 4, T.T_TITLE_3, T.TEXT_PRIMARY, weight="bold")

        item_h, gap = 42.0, 3.0
        y = ry + (0 if compact else 52)

        for i, c in enumerate(self.store.categories):
            r = (rx, y + i * (item_h + gap), rail_w, item_h)
            sel = i == self.cat
            hov = spring(ui, ("set_cat", i), float(inside(r, *mouse)), T.SPRING_GENTLE)
            self.hits.add(r, "cat", i)

            if sel:
                ui.box(*r, 12, (1.0, 1.0, 1.0, 0.11))
            elif hov > 0.01:
                ui.box(*r, 12, (1.0, 1.0, 1.0, 0.06 * hov))

            tile = TILE_CYCLE[i % len(TILE_CYCLE)]
            ts = 28.0
            tx = r[0] + (r[2] - ts) / 2 if compact else r[0] + 8
            icon_tile(ui, tx, r[1] + (r[3] - ts) / 2, ts, c.icon, tile)
            if not compact:
                ui.text_c(ui.fit(c.label, T.T_CALLOUT, r[2] - 56), r[0] + 46, r[1] + r[3] / 2, T.T_CALLOUT,
                          T.TEXT_PRIMARY if sel else T.TEXT_SECONDARY, weight="semibold" if sel else "medium")

        if not compact and ry + fh - 52 - 36 > y + len(self.store.categories) * (item_h + gap):
            by = fy + fh - 20 - 21
            kw = keycaps(ui, rx + 8, by, "Q", 11)
            kw2 = keycaps(ui, rx + 8 + kw + 4, by, "E", 11)
            ui.text_c("Switch page", rx + 8 + kw + kw2 + 12, by + 10.5, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")

    # ------------------------------------------------------------------ rows
    def _draw_row(self, ui, s, rect, mouse, focused, last=False):
        gx, y, gw, h = rect                       # the row spans the whole group width
        x, w = gx + GROUP_PAD, gw - 2 * GROUP_PAD
        enabled = s.is_enabled()
        hov = spring(ui, ("set_row", s.id),
                     float(inside(rect, *mouse) and enabled and inside(self.view, *mouse)), T.SPRING_GENTLE)

        if enabled:
            self.hits.add(rect, "row", s, clip=self.view)
        if focused and enabled:
            ui.box(gx + 6, y + 2, gw - 12, h - 4, 12, (1.0, 1.0, 1.0, 0.085))
        elif hov > 0.01:
            ui.box(gx + 6, y + 2, gw - 12, h - 4, 12, (1.0, 1.0, 1.0, 0.045 * hov))
        if not last:
            separator(ui, x, y + h - 0.5, w)

        ctrl_w = min(340.0, w * 0.5)
        label_w = w - ctrl_w - 24
        col = T.TEXT_PRIMARY if enabled else T.TEXT_QUATERNARY
        ty = y + (10 if s.desc else 14)

        lw = ui.text(ui.fit(s.label, T.T_CALLOUT, label_w, weight="medium"), x, ty, T.T_CALLOUT, col, weight="medium")
        if s.note:
            chip(ui, x + min(lw, label_w) + 10, ty, s.note, T.TEXT_TERTIARY, T.T_CAPTION_1, h=18, pad=7)
        if s.desc:
            ui.text(ui.fit(s.desc, T.T_CAPTION_2, label_w), x, y + 33, T.T_CAPTION_2,
                    T.TEXT_TERTIARY if enabled else T.TEXT_QUATERNARY, weight="regular")

        ctrl = (x + w - ctrl_w, y, ctrl_w, h)
        key = s.id

        if s.kind == "toggle":
            toggle(ui, ctrl, bool(s.value()), key=key, focus=focused and enabled, enabled=enabled)
            if enabled:
                self.hits.add(toggle_rect(ctrl), "toggle", s, clip=self.view)
        elif s.kind == "slider":
            vw = 56.0
            track = (ctrl[0] + 12, y, ctrl_w - vw - 24, h)
            self.slider_rects[s.id] = track
            hv = spring(ui, ("set_sl", s.id), float(enabled and (self.drag_id is s.id or inside(track, *mouse))),
                        T.SPRING_GENTLE)
            slider(ui, track, s.fraction(), key=key, hover=hv, focus=focused and enabled, enabled=enabled)
            ui.text_c(s.text(), x + w, y + h / 2, T.T_BODY, T.TEXT_SECONDARY if enabled else T.TEXT_QUATERNARY,
                      anchor="right", weight="medium")
            if enabled:
                self.hits.add((track[0] - 12, y, track[2] + 24, h), "slider", s, clip=self.view)
        elif s.kind == "choice":
            self._draw_choice(ui, s, ctrl, mouse, focused, enabled)
        elif s.kind == "action":
            bw = max(88.0, ui.text_w(s.button, T.T_BODY, weight="semibold") + 32)
            r = (x + w - bw, y + (h - 32) / 2, bw, 32.0)
            button(ui, r, s.button, mouse, key=("act", s.id), kind="secondary", focus=focused and enabled,
                   disabled=not enabled, size=T.T_BODY)
            if enabled:
                self.hits.add(r, "action", s, clip=self.view)
        else:                                       # info
            ui.text_c(ui.fit(s.text(), T.T_BODY, ctrl_w), x + w, y + h / 2, T.T_BODY, T.TEXT_SECONDARY,
                      anchor="right", weight="regular")

    def _draw_choice(self, ui, s, ctrl, mouse, focused, enabled):
        labels = [lbl for _, lbl in s.choices]
        idx = s.choice_index()
        need = sum(ui.text_w(t, T.T_CAPTION_2, weight="semibold") + 26 for t in labels) + 6
        if len(labels) <= 6 and need <= ctrl[2]:
            rects = segmented(ui, ctrl, labels, idx, mouse, key=s.id, focus=focused and enabled, enabled=enabled,
                              size=T.T_CAPTION_2)
            if enabled:
                for i, r in enumerate(rects):
                    self.hits.add(r, "seg", (s, i), clip=self.view)
        else:
            arrows = stepper(ui, ctrl, labels[idx] if labels else "", mouse, key=s.id,
                             focus=focused and enabled, enabled=enabled, width=min(230.0, ctrl[2]),
                             size=T.T_CAPTION_2)
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
                gh = len(rows) * row + 12
                if y < vy + vh and y + 26 + gh > vy:
                    heading(ui, title, x + 6, y + 2)
                    group(ui, x, y + 26, col_w, gh)
                    for i, (key, what) in enumerate(rows):
                        ry = y + 26 + 6 + i * row
                        keycaps(ui, x + 14, ry + (row - 21) / 2, key, 11)
                        ui.text_c(ui.fit(what, T.T_BODY, col_w - 14 - key_w - 14), x + 14 + key_w, ry + row / 2,
                                  T.T_BODY, T.TEXT_SECONDARY, weight="regular")
                        if i < len(rows) - 1:
                            separator(ui, x + 14, ry + row - 0.5, col_w - 28)

    def _draw_footer(self, ui, g, mouse, cat):
        fx, fy, fw, fh = g["frame"]
        cx, cw, fyy = g["cx"], g["cw"], g["footer_y"]

        if cat.custom is None:
            r = (cx - 4, fyy + 10, 160.0, 36.0)
            self.hits.add(r, "reset")
            button(ui, r, "Reset page", mouse, key="set_reset", kind="tertiary", icon="reset", size=T.T_BODY)

        txt = "Changes apply instantly and are saved automatically"
        if cat.custom == "controls":
            txt = "Hold Q while on foot to profile nearby people"
        ui.text_c(ui.fit(txt, T.T_CAPTION_2, cw - 180), cx + cw, fyy + 28, T.T_CAPTION_2, T.TEXT_TERTIARY,
                  anchor="right", weight="regular")

    # ------------------------------------------------------------------ input
    def prompts(self):
        return [("↑ ↓", "Select"), ("← →", "Adjust"), ("Enter", "Toggle"), ("Tab", "Next page"), ("Esc", "Back")]

    def _commit(self, s, value, sound=None):
        self.store.change(s, value)
        if sound == "toggle":
            self.menu.sound("toggle", bool(s.value()))
        elif sound:
            self.menu.sound(sound)

    def _nudge(self, s, direction):
        """Left / Right on the focused setting."""
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
    return max((sum(26.0 + len(rows) * spec["row"] + 14.0 for _, rows, _ in groups)
                for groups in spec["columns"]), default=0.0)