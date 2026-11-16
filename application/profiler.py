"""A "profiler" overlay: hold the scan key on foot and the crowd reveals who they are.

Everyone in view within scanning range gets a small diamond over their head; the one
nearest the screen centre locks on with a bigger reticle and a card showing the
procedurally generated profile (name, quirk, age, occupation, income) and a little
avatar painted from their actual skin / hair / clothing colours.  The profile is a
deterministic function of the person's seed (pedestrians.identity_for), so the same
stranger always has the same card.  All of it is fiction.
"""
from __future__ import annotations

import math

import numpy as np

SKIN = ((.96, .82, .70), (.93, .76, .60), (.85, .64, .45), (.76, .54, .36),
        (.62, .41, .26), (.50, .32, .20), (.38, .24, .15), (.28, .17, .11))
HAIR = ((.06, .05, .05), (.15, .09, .06), (.30, .18, .09), (.42, .19, .10),
        (.78, .64, .38), (.62, .62, .63), (.86, .86, .85), (.11, .08, .06))
TOP = ((.88, .88, .86), (.05, .05, .06), (.18, .18, .20), (.08, .12, .28), (.62, .10, .10), (.28, .32, .18),
       (.72, .55, .12), (.10, .45, .46), (.45, .45, .46), (.20, .30, .48), (.72, .62, .48), (.35, .07, .12),
       (.10, .28, .16), (.78, .45, .55), (.78, .36, .10), (.13, .11, .095))

WHITE = (1., 1., 1., 1.)
INK = (.04, .04, .05, 1.)


def _rgba(rgb, a=1.):
    return (rgb[0], rgb[1], rgb[2], a)


def project(vp, point, w, h):
    """World point -> logical screen position, or None behind the camera."""
    c = vp @ np.array([point[0], point[1], point[2], 1.])
    if c[3] <= .2:
        return None
    return (c[0] / c[3] * .5 + .5) * w, (1. - (c[1] / c[3] * .5 + .5)) * h


class Profiler:
    RANGE = 34.

    def __init__(self):
        self.active = False
        self.focus = None                 # Ped currently locked
        self._show = 0.                   # 0..1 card reveal
        self._last_seed = None

    def update(self, dt, held):
        self.active = bool(held)
        self._show = min(1., self._show + dt * 6.) if (self.active and self.focus is not None) \
            else max(0., self._show - dt * 8.)
        if not self.active:
            self.focus = None

    # ------------------------------------------------------------------ drawing
    def draw(self, ui, w, h, camera, crowd, eye):
        if not self.active:
            return
        vp = camera.projection(w / max(h, 1)) @ camera.view()
        cx, cy = w * .5, h * .5
        candidates = []
        for p in crowd.living():
            dist = math.hypot(p.x - eye[0], p.y - eye[1])
            if dist > self.RANGE:
                continue
            pos = project(vp, crowd.head_position(p), w, h)
            if pos is None or not (0 <= pos[0] <= w and 0 <= pos[1] <= h):
                continue
            candidates.append((p, pos, dist))

        # lock-on: nearest to the crosshair, with hysteresis toward the current target
        best, best_d = None, h * .20
        for p, pos, _ in candidates:
            d = math.hypot(pos[0] - cx, pos[1] - cy) * (.7 if p is self.focus else 1.)
            if d < best_d:
                best, best_d = p, d
        self.focus = best

        ui.circle(cx, cy, 2.2, (1., 1., 1., .85))
        for p, (sx, sy), dist in candidates:
            if p is self.focus:
                continue
            a = max(.25, 1. - dist / self.RANGE)
            self._diamond(ui, sx, sy - 4., 6., (1., 1., 1., a * .9), filled=False)
        for p, (sx, sy), dist in candidates:
            if p is self.focus:
                self._card(ui, w, h, p, sx, sy, crowd)

    @staticmethod
    def _diamond(ui, x, y, r, color, filled=True, width=1.6):
        pts = [(x, y - r), (x + r, y), (x, y + r), (x - r, y)]
        if filled:
            ui.poly(pts, color)
        else:
            ui.outline_poly(pts, width, color, True)

    def _card(self, ui, w, h, ped, sx, sy, crowd):
        t = self._show
        if t <= .01:
            return
        ident = crowd.identity(ped)
        self._diamond(ui, sx, sy - 6., 13., (1., 1., 1., t), filled=False, width=2.2)
        self._diamond(ui, sx, sy - 6., 4., (1., 1., 1., t), filled=True)

        info = (f"Age: {ident.age}", f"Occupation: {ident.occupation}", f"Income: {ident.income_text}")
        trait_w = ui.text_w(ident.trait, 20, True) + 38
        info_w = max(ui.text_w(s, 13, False, "medium") for s in info) + 22
        name_w = ui.text_w(ident.name, 13, True) + 22
        body_w = max(trait_w, info_w, name_w)
        card_w = 88 + body_w
        card_h = 12 + 24 + 38 + 22 * len(info)
        x0 = sx + 46 + (1. - t) * 14
        if x0 + card_w > w - 16:
            x0 = sx - 46 - card_w
        y0 = min(max(sy - 30, 16), h - card_h - 16)

        # leader line
        near_x = x0 if x0 > sx else x0 + card_w
        ui.line(sx + (13 if x0 > sx else -13), sy - 6, near_x, y0 + 30, 1.4, (1., 1., 1., .7 * t))

        # header + avatar
        ui.text("PROFILER.APP v0.2xx[beta]", x0, y0 - 1, 9, (.82, .84, .88, .9 * t), weight="medium")
        ax, ay, av = x0, y0 + 12, 78.
        self._avatar(ui, ax, ay, av, ped, t)
        for i in range(5):
            ui.rect(ax + i * 15, ay + av + 5, 10, 6, (.9, .9, .92, .85 * t))

        # name tab, quirk banner, facts
        bx = x0 + 88
        ui.rect(bx, y0 + 12, name_w, 24, (1., 1., 1., .96 * t))
        ui.text_c(ident.name, bx + 11, y0 + 24, 13, (.04, .04, .05, t), bold=True)
        ty = y0 + 36
        ui.poly([(bx - 12, ty + 19), (bx, ty), (bx + trait_w, ty), (bx + trait_w, ty + 38), (bx, ty + 38)],
                (1., 1., 1., .97 * t))
        ui.text_c(ident.trait, bx + 18, ty + 20, 20, (.04, .04, .05, t), bold=True)
        fy = ty + 38
        for i, line in enumerate(info):
            ui.rect(bx, fy + i * 22, info_w, 22, (.02, .02, .03, .74 * t))
            ui.text_c(line, bx + 11, fy + i * 22 + 11, 13, (.95, .96, .98, t), weight="medium")

    def _avatar(self, ui, x, y, s, ped, t):
        row = ped.row
        skin = _rgba(SKIN[int(row[12])], t)
        hair = _rgba(HAIR[int(row[13])], t)
        top = _rgba(TOP[int(row[14])], t)
        ui.rect(x, y, s, s, (.06, .07, .09, .92 * t))
        ui.rect(x, y + s - 1, s, 1, (1., 1., 1., .35 * t))
        cx = x + s * .5
        # shoulders, neck, head, hair
        ui.rrect(cx - 32, y + s - 26, 64, 40, 14, top)
        ui.rect(cx - 6, y + s - 40, 12, 18, skin)
        ui.circle(cx, y + 30, 19, hair)
        if ped.acc & (128 | 256):                    # long hair / bun: hair frames the face
            ui.rect(cx - 19, y + 30, 6, 26, hair)
            ui.rect(cx + 13, y + 30, 6, 26, hair)
        ui.circle(cx, y + 36, 16, skin)
        ui.rect(cx - 15, y + 19, 30, 7, hair)
        if ped.acc & 1024:                           # glasses
            ui.rect(cx - 14, y + 33, 28, 4, (.04, .04, .05, t))
        else:
            ui.circle(cx - 6, y + 35, 1.5, (.05, .05, .06, t))
            ui.circle(cx + 6, y + 35, 1.5, (.05, .05, .06, t))
