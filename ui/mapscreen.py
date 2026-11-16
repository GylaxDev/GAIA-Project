"""The map screen: a flat, north-up Apple-Maps-style dark map with the chrome
from ``maps_ui`` on top (sidebar, control cluster, compass, zoom).

This replaces the old 3D globe + ``MapUI``.  It exposes the handful of methods
``App`` needs (``set_size``, ``update``, ``draw``, ``hit``, ``pick`` ...), and
reuses ``FlatMap`` for tiles, panning, cursor-anchored zoom and fly-to.
"""
from __future__ import annotations

import time
import math

import numpy as np

from .flatmap import FlatMap
from . import theme as T
from .maps_ui import MapsChrome, _in
from .map_draw import destination_pin, place_symbol, project_world, route_stroke
from ..world.places import scene_places

WATER = (0.125, 0.165, 0.44, 1.0)        # shows through where tiles haven't loaded yet
PAPER = (0.66, 0.78, 0.86, 1.0)          # same, for the plain (un-restyled) map


class MapScreen:
    def __init__(self, tiles, cities, invert=True):
        self.tiles = tiles
        self.cities = list(cities)
        self.invert = 1.0 if invert else 0.0
        self.flat = FlatMap(tiles)
        self.chrome = MapsChrome()
        self.marker = None
        self.press_ui = False
        self.W, self.H = 1280, 800
        self._place_hits = []
        self._place_labels = []
        # open on the Adriatic / central Europe, like Apple Maps' first view
        self.flat.jump_to(45.0, 13.0, 4.4)

    # ------------------------------------------------------------- delegation
    @property
    def flying(self):
        return self.flat.flying

    def set_scale(self, s):
        """Display scale: lets one logical pixel map onto ``s`` device pixels, and keeps
        map tiles at 1:1 with the framebuffer so they are not resampled."""
        s = max(0.5, float(s))
        self.flat.scale = s
        self.flat.T = 256.0 / s

    def set_size(self, w, h):
        self.W, self.H = max(int(w), 1), max(int(h), 1)
        self.flat.set_size(self.W, self.H)

    def update(self, dt):
        self.tiles.pump()
        self.flat.update(dt)

    def render(self, hours):
        """Nothing to do: the flat map is drawn entirely through the 2D overlay."""

    def pan(self, dx, dy): self.flat.pan(dx, dy)
    def end_drag(self): self.flat.end_drag()
    def zoom_by(self, steps, sx=None, sy=None): self.flat.zoom_by(steps, sx, sy)
    def zoom_step(self, sign): self.flat.zoom_by(2.5 * sign)
    def jump_to(self, lat, lon, zoom=None): self.flat.jump_to(lat, lon, zoom)
    def fly_to(self, lat, lon, zoom=None, dur=0.9): self.flat.fly_to(lat, lon, zoom, dur)
    def pick(self, sx, sy): return self.flat.pick(sx, sy)
    def screen_to_latlon(self, sx, sy): return self.flat.screen_to_latlon(sx, sy)
    def city_near(self, sx, sy, cities=None): return self.flat.city_near(sx, sy, cities or self.cities)
    def toggle_style(self): self.invert = 0.0 if self.invert > 0.5 else 1.0

    # ------------------------------------------------------------------ input
    def hit(self, pos):
        return self.chrome.hit(pos, self.W, self.H, "map", len(self.cities))

    def blocks(self, pos):
        return self.hit(pos) is not None

    def place_near(self, pos):
        symbol = next((place for x, y, place in self._place_hits
                       if math.hypot(pos[0] - x, pos[1] - y) <= 12.), None)
        return symbol or next((place for rect, place in self._place_labels if _in(rect, *pos)), None)

    def _place_names(self, ui, occupied):
        """Names must not cover another pin, city label or sidebar control."""
        occupied.extend((x - 12., y - 12., 24., 24.) for x, y, _ in self._place_hits)
        for name, lat, lon in self.cities:
            x, y, visible = self.flat.project(lat, lon)
            if visible:
                occupied.append((x - 6., y - 10., ui.text_w(name, T.T_BODY, weight="medium") + 20., 20.))
        for x, y, place in self._place_hits[:8]:
            shown = ui.fit(place.name, T.T_SMALL, 150., weight="medium")
            width = ui.text_w(shown, T.T_SMALL, weight="medium") + 4.
            for lx, ly in ((x + 14., y - 9.), (x - 14. - width, y - 9.),
                           (x - width * .5, y - 30.), (x - width * .5, y + 14.)):
                rect = (lx, ly, width, 18.)
                if not (16. < lx and lx + width < self.W - 66. and 16. < ly and ly + 18. < self.H - 24.):
                    continue
                if any(self.chrome.over_ui(point, self.W, self.H, "map", len(self.cities))
                       for point in ((lx, ly), (lx + width, ly + 18.))):
                    continue
                if any(lx < ox + ow and lx + width > ox and ly < oy + oh and ly + 18. > oy
                       for ox, oy, ow, oh in occupied):
                    continue
                ui.text_c(shown, lx + 2., ly + 9., T.T_SMALL, T.TEXT, weight="medium", halo=True)
                self._place_labels.append((rect, place))
                occupied.append(rect)
                break

    def _marker_label(self, ui):
        """The popup that floats above a manually chosen marker, or ``None``."""
        if self.marker is None: return None
        label = ui.fit(self.marker[2], T.T_BASE, 320, weight="semibold")
        width = ui.text_w(label, T.T_BASE, weight="semibold") + 28.
        x, y, visible = self.flat.project(self.marker[0], self.marker[1])
        if not visible: return None
        return (round(x - width * .5), round(y - 54.), round(width), 30.)

    def _city_names(self, ui, occupied):
        """City labels yield to place labels, controls and each other."""
        placed = []
        for name, lat, lon in self.cities:
            x, y, visible = self.flat.project(lat, lon)
            if not visible: continue
            shown = ui.fit(name, T.T_BODY, 240., weight="medium")
            width = ui.text_w(shown, T.T_BODY, weight="medium") + 20.
            # (text x, rect left, vertical y, anchor)
            for tx, lx, ly, anchor in ((x + 10., x + 10., y - 9., "left"),
                                       (x - 10. - width, x - 10. - width, y - 9., "right"),
                                       (x, x - width * .5, y - 26., "center"),
                                       (x, x - width * .5, y + 12., "center")):
                rect = (lx, ly, width, 18.)
                if not (8. < lx and lx + width < self.W - 56. and 8. < ly and ly + 18. < self.H - 8.):
                    continue
                if any(self.chrome.over_ui(point, self.W, self.H, "map", len(self.cities))
                       for point in ((lx, ly), (lx + width, ly + 18.))):
                    continue
                if any(lx < ox + ow and lx + width > ox and ly < oy + oh and ly + 18. > oy
                       for ox, oy, ow, oh in occupied + placed):
                    continue
                ui.text_c(shown, tx, y, T.T_BODY, (1, 1, 1, 0.98), anchor=anchor, weight="medium", halo=True)
                placed.append(rect)
                break

    def _navigation(self, ui, navigation, scene, projection, position, marker_rect=None):
        """Draw route/places/pins; return the label-occupancy rects for ``_city_names``."""
        self._place_hits.clear()
        self._place_labels.clear()
        if projection is None: return []
        flat = self.flat
        occupied = [marker_rect] if marker_rect is not None else []
        if position is not None:
            x, y = project_world(flat, np.asarray([position[:2]]), projection)[0]
            occupied.append((x - 10., y - 10., 20., 20.))
        if navigation is not None and navigation.destination is not None:
            x, y = project_world(flat, np.asarray([navigation.destination.position]), projection)[0]
            occupied.append((x - 14., y - 14., 28., 32.))
        clip = (0., 0., float(self.W), float(self.H))
        if navigation is not None and navigation.route is not None:
            route_stroke(ui, project_world(flat, navigation.route.tail(navigation.progress), projection), clip)
        if flat.zoom >= 14.:
            places = scene_places(scene)
            if places:
                pixels = project_world(flat, np.asarray([p.position for p in places]), projection)
                order = np.argsort(np.linalg.norm(pixels - (self.W * .5, self.H * .5), axis=1))
                for index in order:
                    x, y = map(float, pixels[index])
                    if not (16. < x < self.W - 66 and 16. < y < self.H - 24): continue
                    if self.chrome.over_ui((x, y), self.W, self.H, "map", len(self.cities)): continue
                    if any(math.hypot(x - px, y - py) < 27. for px, py, _ in self._place_hits): continue
                    place = places[index]
                    place_symbol(ui, x, y, place.category, 10.)
                    self._place_hits.append((x, y, place))
                    if len(self._place_hits) >= 60: break
                self._place_names(ui, occupied)
        if navigation is not None and navigation.destination is not None:
            point = project_world(flat, np.asarray([navigation.destination.position]), projection)[0]
            if -20 < point[0] < self.W + 20 and -20 < point[1] < self.H + 20:
                destination_pin(ui, *point, size=11.)
        if position is not None:
            point = project_world(flat, np.asarray([position[:2]]), projection)[0]
            if 0 < point[0] < self.W and 0 < point[1] < self.H:
                ui.circle(*point, 8., (1., 1., 1., 1.))
                ui.circle(*point, 5.5, (.08, .52, 1., 1.))
        return occupied

    # ------------------------------------------------------------------ draw
    def draw(self, ui, mouse, loading=False, typing=False, query="", fullscreen=False,
             can_enter_3d=False, navigation=None, scene=None, projection=None, position=None):
        flat, W, H = self.flat, self.W, self.H
        self.chrome.navigation_active = navigation is not None and navigation.destination is not None
        flat.draw(ui, self.invert, WATER if self.invert > 0.5 else PAPER)
        goal = None
        if navigation is not None and navigation.destination is not None and projection is not None:
            goal = projection.to_latlon(*navigation.destination.position)
        show_marker = bool(self.marker) and not (
            goal is not None and math.hypot(self.marker[0] - goal[0], self.marker[1] - goal[1]) < 1e-6)
        marker_rect = self._marker_label(ui) if show_marker else None
        occupied = self._navigation(ui, navigation, scene, projection, position, marker_rect)
        # place dots: white ring with a dark core; labels get a thin dark halo so they
        # stay readable on the tiles without a blurry drop shadow
        for name, lat, lon in self.cities:
            x, y, vis = flat.project(lat, lon)
            if vis:
                ui.circle(x, y, 5.0, (0.04, 0.05, 0.08, 0.60))
                ui.circle(x, y, 4.0, (1, 1, 1, 1.0))
                ui.circle(x, y, 2.2, (0.10, 0.14, 0.20, 1.0))
        self._city_names(ui, occupied)
        if show_marker:
            lat, lon, label = self.marker
            x, y, vis = flat.project(lat, lon)
            if vis:
                pulse = (time.time() * 0.9) % 1.0
                ui.circle(x, y, 10 + 22 * pulse, (0.04, 0.52, 1.0, 0.30 * (1.0 - pulse)))
                ui.circle(x, y, 11.5, (0, 0, 0, 0.25))
                ui.circle(x, y, 10, (1, 1, 1, 1))
                ui.circle(x, y, 7.4, (0.04, 0.52, 1.0, 1.0))
                lab = ui.fit(label, T.T_BASE, 320, weight="semibold")
                tw = ui.text_w(lab, T.T_BASE, weight="semibold")
                bx, by = round(x - tw / 2 - 14), round(y - 54)
                ui.container(bx, by, round(tw + 28), 30, "popup", r=T.R_BUTTON)
                ui.text_c(lab, x, by + 15, T.T_BASE, T.TEXT, anchor="center", weight="semibold")
        if self.tiles.offline:
            msg = "Map tiles unavailable (offline?)"
            tw = ui.text_w(msg, T.T_BASE, weight="medium")
            ui.container(round(W / 2 - tw / 2 - 16), 16, round(tw + 32), 34, "popup", r=T.R_BUTTON,
                         shadow=False)
            ui.text_c(msg, W / 2, 33, T.T_BASE, T.WARN, anchor="center", weight="medium")
        self.chrome.draw_map_ui(ui, W, H, mouse, self.cities, self.marker, typing, query,
                                 fullscreen, tiles_dark=self.invert > 0.5,
                                 can_enter_3d=can_enter_3d, navigation=navigation)
