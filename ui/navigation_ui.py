"""Destination picker and Apple-Maps-style live guidance card, in logical pixels."""
from __future__ import annotations

import math
import time

from . import theme as T
from .map_draw import RED, maneuver_symbol, place_symbol
from .maps_ui import MapsChrome, _in
from .minimap import MinimapPanel
from ..world.places import CATEGORY_LABELS, scene_places


def distance_text(metres):
    if metres >= 1000.: return f"{metres / 1000.:.1f} km"
    return f"{max(0, int(round(metres / 10.) * 10))} m"


class NavigationPanel:
    def __init__(self):
        self.open = False
        self.query = ""
        self.selection = 0
        self.items = ()
        self._places = ()
        self._origin = (0., 0.)
        self._rects = {}

    def refresh(self, scene, origin):
        places = scene_places(scene)
        if places is not self._places or self._origin != tuple(origin[:2]):
            selected = self.items[self.selection].key if 0 <= self.selection < len(self.items) else None
            self._places, self._origin = places, tuple(origin[:2])
            self._filter()
            self.selection = next((i for i, p in enumerate(self.items) if p.key == selected), self.selection)

    def _filter(self):
        query = self.query.strip().casefold()
        items = [p for p in self._places if not query or query in p.name.casefold()
                 or query in CATEGORY_LABELS.get(p.category, p.category).casefold()]
        items.sort(key=lambda p: (math.hypot(p.x - self._origin[0], p.y - self._origin[1]), p.key))
        self.items = tuple(items)
        self.selection = min(self.selection, max(0, len(self.items) - 1))

    def show(self, scene, origin, place=None):
        self.open, self.query, self.selection = True, place.name if place else "", 0
        self.refresh(scene, origin)
        self._filter()
        if place is not None:
            self.selection = next((i for i, p in enumerate(self.items) if p.key == place.key), 0)

    def key(self, ev, pg):
        if ev.key == pg.K_ESCAPE:
            self.open = False
            return "close", None
        if ev.key in (pg.K_RETURN, pg.K_KP_ENTER) and self.items:
            return "destination", self.items[self.selection]
        if ev.key in (pg.K_DOWN, pg.K_UP):
            step = 1 if ev.key == pg.K_DOWN else -1
            self.selection = max(0, min(max(0, len(self.items) - 1), self.selection + step))
        elif ev.key == pg.K_BACKSPACE:
            self.query, self.selection = self.query[:-1], 0
            self._filter()
        elif ev.unicode and ev.unicode.isprintable() and len(self.query) < 80:
            self.query, self.selection = self.query + ev.unicode, 0
            self._filter()
        return None, None

    def scroll_by(self, steps):
        self.selection = max(0, min(max(0, len(self.items) - 1), self.selection - int(steps)))

    def clear(self):
        self.open = False
        self.query, self.selection = "", 0
        self.items = self._places = ()
        self._rects.clear()

    def hit(self, pos):
        for action, rect in self._rects.items():
            if _in(rect, *pos): return action
        return None

    def draw(self, ui, w, h, mouse, navigation, scene, origin, status_h=0.):
        self._rects = {}
        if self.open: self.refresh(scene, origin)
        if w < 520 or h < 380:
            if self.open: self._picker(ui, w, h, mouse)
            return
        rect = (T.MARGIN, T.MARGIN + 72, 140., 32.)
        self._rects["open"] = rect
        hover = float(_in(rect, *mouse))
        ui.container(*rect, "button", r=9, hover=hover)
        MapsChrome.icon_directions(ui, rect[0] + 17, rect[1] + 16, 16, T.ACCENT)
        ui.text_c("Directions", rect[0] + 34, rect[1] + 16, T.T_BODY, T.TEXT, weight="medium")
        ui.text_c("F6", rect[0] + 126, rect[1] + 16, T.T_CAPTION, T.TEXT_3, anchor="right")
        inset = MinimapPanel().layout(w, h, status_h)
        if navigation.destination is not None and inset is not None:
            guide = (inset[0], inset[1] - 112., inset[2], 104.)
            if guide[1] >= rect[1] + rect[3] + 12.:
                self._guidance(ui, guide, mouse, navigation)
        if self.open:
            self._picker(ui, w, h, mouse)

    def _guidance(self, ui, rect, mouse, nav):
        x, y, w, h = rect
        ui.container(*rect, "panel", r=14)
        ui.box(x + 1, y + 1, w - 2, 54, 13,
               fill=((0.055, .29, .21, 1.), (.035, .20, .16, 1.)))
        maneuver = nav.next_maneuver
        kind = "arrive" if nav.status == "arrived" else maneuver.kind if maneuver else 0
        if nav.status in ("routing", "rerouting"):
            ui.spinner(x + 27, y + 28, 8., getattr(ui, "now", 0.))
        else:
            maneuver_symbol(ui, x + 27, y + 28, kind, 14.)
        ui.text(ui.fit(nav.instruction, T.T_BODY, w - 62, weight="semibold"),
                x + 50, y + 11, T.T_BODY, T.TEXT, weight="semibold")
        if nav.status == "navigating":
            sub = ("Now" if nav.turn_distance < 10. else f"In {distance_text(nav.turn_distance)}")
            if maneuver.road: sub += " · " + maneuver.road
        elif nav.status == "arrived": sub = "Park when it is safe"
        else: sub = "Loaded-road navigation"
        ui.text(ui.fit(sub, T.T_SMALL, w - 62), x + 50, y + 33, T.T_SMALL, T.TEXT_2, weight="medium")
        ui.circle(x + 13, y + 68, 3., RED)
        ui.text_c(ui.fit(nav.destination.name, T.T_BODY, w - 60, weight="semibold"),
                  x + 23, y + 68, T.T_BODY, T.TEXT, weight="semibold")
        if nav.status == "arrived":
            trip = "Route complete"
        elif nav.route is not None:
            minutes = max(1, int(math.ceil(nav.eta_seconds / 60.)))
            arrival = time.strftime("%H:%M", time.localtime(time.time() + nav.eta_seconds))
            trip = f"Est. {minutes} min · {distance_text(nav.remaining)} · {arrival}"
        else:
            trip = {"waiting": "Load nearby streets to continue",
                    "outside": "Choose a place in the loaded area",
                    "offroad": "Move closer to a loaded street",
                    "no_route": "Try another reachable place",
                    "error": "Try choosing the destination again"}.get(nav.status, "Finding a connected route")
        ui.text_c(trip, x + 12, y + 88, T.T_SMALL, T.TEXT_3, weight="medium")
        cancel = (x + w - 32, y + 59, 24., 24.)
        self._rects["cancel"] = cancel
        ui.circle(cancel[0] + 12, cancel[1] + 12, 11., (1., 1., 1., .13 if _in(cancel, *mouse) else .06))
        self._close_icon(ui, cancel)
        self._rects["guidance_bg"] = rect

    @staticmethod
    def _close_icon(ui, rect):
        cx, cy = rect[0] + rect[2] * .5, rect[1] + rect[3] * .5
        ui.line(cx - 3, cy - 3, cx + 3, cy + 3, 1.6, T.TEXT_2)
        ui.line(cx - 3, cy + 3, cx + 3, cy - 3, 1.6, T.TEXT_2)

    def _picker(self, ui, w, h, mouse):
        x, y = T.MARGIN, T.MARGIN + (72 if h >= 380 else 0)
        cw = min(294., max(160., w - 2 * T.MARGIN))
        count = min(6, max(1, int((h - y - T.MARGIN - 142.) / 46.)), len(self.items))
        start = min(max(0, self.selection - count + 1), max(0, len(self.items) - count))
        ch = 146. + max(1, count) * 46.
        # Picker is modal and painted last. Its targets replace the obscured card.
        self._rects = {}
        close = (x + cw - 38, y + 10, 26., 26.)
        self._rects["close"] = close
        ui.container(x, y, cw, ch, "popup", r=14)
        ui.text("Where to?", x + 16, y + 13, T.T_LABEL, T.TEXT, weight="semibold")
        self._close_icon(ui, close)
        ui.text("Nearby places · Enter to navigate", x + 16, y + 38, T.T_SMALL, T.TEXT_3)
        field = (x + 12, y + 60, cw - 24, 34.)
        MapsChrome()._field(ui, *field, self.query, "Search nearby places")
        self._rects["field"] = field
        if not self.items:
            ui.text("No matching places loaded", x + 16, y + 112, T.T_BODY, T.TEXT_2)
            ui.text("Right-click the minimap to drop a pin", x + 16, y + 134, T.T_SMALL, T.TEXT_3)
        for row, index in enumerate(range(start, start + count)):
            place = self.items[index]
            rect = (x + 10, y + 102 + row * 46, cw - 20, 44.)
            self._rects[f"place:{index}"] = rect
            if index == self.selection or _in(rect, *mouse):
                ui.box(*rect, 8, fill=(.08, .45, .90, .18))
            place_symbol(ui, rect[0] + 20, rect[1] + 22, place.category, 12.)
            ui.text(ui.fit(place.name, T.T_BODY, rect[2] - 56, weight="semibold"),
                    rect[0] + 42, rect[1] + 6, T.T_BODY, T.TEXT, weight="semibold")
            distance = math.hypot(place.x - self._origin[0], place.y - self._origin[1])
            sub = f"{CATEGORY_LABELS.get(place.category, 'Place')} · {distance_text(distance)}"
            ui.text(ui.fit(sub, T.T_SMALL, rect[2] - 56), rect[0] + 42, rect[1] + 25, T.T_SMALL, T.TEXT_3)
        button = (x + 12, y + ch - 38, cw - 24, 28.)
        self._rects["choose_map"] = button
        ui.container(*button, "button", r=8, shadow=False, hover=float(_in(button, *mouse)))
        MapsChrome.icon_map(ui, button[0] + 17, button[1] + 14, 15, T.TEXT_2)
        ui.text_c("Choose on map / drop a minimap pin", button[0] + 33, button[1] + 14,
                  T.T_SMALL, T.TEXT_2, weight="medium")
        self._rects["picker_bg"] = (x, y, cw, ch)
