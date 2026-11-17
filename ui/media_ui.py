"""The now-playing card: cover preview, track text, progress and transport controls."""
from __future__ import annotations

import io
import math
import time

from . import theme as T
from .ui_kit import _in, glass, progress, spring
from ..render.overlay import make_texture

CARD_W, CARD_H = 380.0, 100.0
ART = 72.0
PAD = 12.0


def clock_text(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class NowPlayingPanel:
    """Draws itself only while the media session has a current track."""

    def __init__(self, session=None):
        self.session = session
        self._art_tex = None
        self._art_src = None
        self._art_bad = None
        self._art_px = 0
        self._track_key: tuple = ()
        self._gl = None

    # ----------------------------------------------------------------- queries
    def snapshot(self):
        return self.session.snapshot() if self.session is not None else None

    @property
    def available(self) -> bool:
        return self.session is not None and self.session.available

    def layout(self, w, h):
        snap = self.snapshot()
        if snap is None or w < CARD_W + 2 * T.MARGIN + 64:
            return None
        x = w - T.MARGIN - 40 - 12 - CARD_W
        y = T.MARGIN + 56 + 8 + 24 + 10
        text_x = x + PAD + ART + 14
        right = x + CARD_W - PAD
        buttons = {
            "previous": (right - 76, y + 66, 24, 26),
            "toggle": (right - 50, y + 66, 24, 26),
            "next": (right - 24, y + 66, 24, 26),
        }
        return dict(card=(x, y, CARD_W, CARD_H),
                    art=(x + PAD, y + 14, ART, ART),
                    text=(text_x, y, right - text_x, CARD_H),
                    bar=(text_x, y + 70, buttons["previous"][0] - 10 - text_x, 4),
                    buttons=buttons)

    def hit(self, pos, w, h):
        layout = self.layout(w, h)
        if layout is None:
            return None
        return next((name for name, rect in layout["buttons"].items() if _in(rect, *pos)), None)

    def press(self, pos, w, h):
        action = self.hit(pos, w, h)
        if action is None or self.session is None:
            return None
        if action == "toggle":
            self.session.toggle_play_pause()
        elif action == "next":
            self.session.next_track()
        elif action == "previous":
            self.session.previous_track()
        return action

    def close(self):
        if self.session is not None:
            self.session.stop()
        self._release_art()
        self._gl = None

    # ------------------------------------------------------------------ drawing
    def draw(self, ui, w, h, mouse):
        snap = self.snapshot()
        if snap is None:
            return
        layout = self.layout(w, h)
        if layout is None:
            return
        self._gl = getattr(ui, "gl", self._gl)
        if snap.track_key != self._track_key:
            self._release_art()
            self._track_key = snap.track_key

        x, y, cw, ch = layout["card"]
        ax, ay, aw, ah = layout["art"]
        tx, _, tw, _ = layout["text"]
        now = getattr(ui, "now", time.time())

        glass(ui, x, y, cw, ch, 22, "hud")

        # cover ----------------------------------------------------------------
        tex = self._ensure_art(ui, snap.art, int(round(aw * getattr(ui, "scale", 1.0))))
        if tex is not None:
            ui.image(tex, ax, ay, aw, ah, radius=12.0)
            ui.box(ax, ay, aw, ah, 12.0, fill=None, border=T.BORDER_SUBTLE, bw=1.0)
        else:
            ui.box(ax, ay, aw, ah, 12.0,
                   fill=((0.30, 0.30, 0.34, 0.98), (0.16, 0.16, 0.19, 0.98)),
                   border=T.BORDER_SUBTLE)
            cx, cy = ax + aw / 2, ay + ah / 2
            ui.ring(cx, cy, 18, 1.5, T.TEXT_QUATERNARY)
            ui.circle(cx, cy, 4.0, T.TEXT_QUATERNARY)
            ui.circle(cx, cy, 1.4, T.TEXT_TERTIARY)

        # state + source -------------------------------------------------------
        state = "Paused" if snap.paused else "Now playing"
        caption = state + (f"  ·  {snap.source}" if snap.source else "")
        if snap.playing:
            phase = now * 7.0
            for i in range(3):
                bh = 3.0 + (0.5 + 0.5 * math.sin(phase + i * 1.9)) * 7.0
                ui.rect(tx + i * 4, y + 23 - bh, 2.1, bh, T.ACCENT_BLUE)
        else:
            ui.circle(tx + 4, y + 19, 2.2, T.TEXT_QUATERNARY)
        ui.text(ui.fit(caption, T.T_CAPTION_2, tw - 20, weight="semibold"),
                tx + 18, y + 11, T.T_CAPTION_2, T.TEXT_TERTIARY, weight="semibold")

        # track text -----------------------------------------------------------
        ui.text(ui.fit(snap.title, T.T_BASE, tw, weight="semibold"), tx, y + 28,
                T.T_BASE, T.TEXT_PRIMARY, weight="semibold")
        byline = snap.byline or snap.source or "Windows media session"
        ui.text(ui.fit(byline, T.T_CAPTION_2, tw, weight="medium"), tx, y + 48,
                T.T_CAPTION_2, T.TEXT_TERTIARY, weight="medium")

        # progress / live ------------------------------------------------------
        bx, by, bw, bh = layout["bar"]
        if snap.duration > 0.0:
            # ``ui.now`` is wall time for animation; media positions use the
            # monotonic clock captured alongside the snapshot.
            position = snap.position_at(time.monotonic())
            frac = position / snap.duration
            progress(ui, bx, by, bw, bh, frac, (1.0, 1.0, 1.0, 0.85))
            ui.text(clock_text(position), bx, y + 78, T.T_CAPTION_2,
                    T.TEXT_QUATERNARY, weight="medium")
            ui.text(clock_text(snap.duration), bx + bw, y + 78, T.T_CAPTION_2,
                    T.TEXT_QUATERNARY, anchor="right", weight="medium")
        else:
            ui.rect(bx, by + bh / 2 - 0.5, bw, 1.0, T.HAIRLINE)
            ui.text("Live", bx, y + 78, T.T_CAPTION_2, T.TEXT_QUATERNARY, weight="medium")

        # transport ------------------------------------------------------------
        for name, rect in layout["buttons"].items():
            hover = spring(ui, ("now-playing", name), float(_in(rect, *mouse)), T.SPRING_GENTLE)
            if hover > 0.01:
                ui.box(rect[0] - 1, rect[1], rect[2] + 2, rect[3], 9, (1.0, 1.0, 1.0, 0.12 * hover))
            self._icon(ui, name, rect, T.TEXT_PRIMARY if hover > 0.5 else T.TEXT_SECONDARY, snap.playing)

    @staticmethod
    def _icon(ui, name, rect, color, playing):
        cx, cy = rect[0] + rect[2] / 2, rect[1] + rect[3] / 2
        if name == "previous":
            ui.rect(cx - 6, cy - 5, 2, 10, color)
            ui.tri((cx + 5, cy - 5), (cx + 5, cy + 5), (cx - 3, cy), color)
        elif name == "next":
            ui.tri((cx - 5, cy - 5), (cx - 5, cy + 5), (cx + 3, cy), color)
            ui.rect(cx + 4, cy - 5, 2, 10, color)
        elif playing:
            ui.rect(cx - 4, cy - 5, 3, 10, color)
            ui.rect(cx + 2, cy - 5, 3, 10, color)
        else:
            ui.tri((cx - 3.5, cy - 5.5), (cx - 3.5, cy + 5.5), (cx + 5, cy), color)

    # ------------------------------------------------------------------- cover
    def _ensure_art(self, ui, data, px):
        if data is None or data is self._art_bad or px <= 0:
            return None
        if (self._art_tex is not None and self._art_src is data
                and self._art_px == px):
            return self._art_tex
        try:
            import pygame
            surface = pygame.image.load(io.BytesIO(data))
            if surface.get_size() != (px, px):
                surface = pygame.transform.smoothscale(surface, (px, px))
            rgba = pygame.image.tostring(surface, "RGBA", False)
            tex = make_texture(ui.gl, px, px, rgba)
        except Exception:
            self._art_bad = data
            return None
        self._release_art()
        self._art_tex, self._art_src, self._art_px = tex, data, px
        return tex

    def _release_art(self):
        if self._art_tex is not None and self._gl is not None:
            try:
                self._gl.glDeleteTextures([self._art_tex])
            except Exception:
                pass
        self._art_tex = self._art_src = self._art_bad = None
        self._art_px = 0