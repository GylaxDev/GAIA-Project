"""A flat, perspective-projected minimap for the scene and driving views.

The inset is a 2D map seen in perspective: building footprints and roads are
coplanar with the ground, then projected through a pitched, heading-up camera.
Nothing is extruded, so the map reads like Apple Maps' tilted "2D" view
instead of a pile of dark blocks.

Geometry is prepared once per resident tile and kept as compact float32
triangles:

* building footprints are the roof triangles of the tile's building mesh,
  flattened onto the ground plane and grouped by connected component so each
  building keeps its own tint;
* roads come from the tile's ``RoadPath`` list as ground-plane ribbons.

Per frame the inset only culls, projects and colours the faces near the focus.
It allocates a few hundred kilobytes of float32 working arrays, where the old
per-frame extrusion built tens of megabytes of float64 wall geometry for every
resident tile (and retained it, which is what pushed the process into swap).

No extra framebuffer or scene pass is involved and the main renderer is never
touched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import theme as T
from ..vehicles.driving_surface import _mesh_triangles
from .maps_ui import _in
from .map_draw import destination_pin, place_symbol, route_stroke
from ..world.places import scene_places

CARD_W, CARD_H = 244.0, 176.0
MAP_RADIUS = 250.0
PITCH = math.radians(52.0)
DISTANCE = 172.0
AHEAD = 22.0
CELL = 0.6

# Footprints smaller than this are chimneys, vents and roof clutter.
MIN_ROOF_AREA = 4.0
MIN_BUILDING_AREA = 16.0

# Apple-Maps dark palette: graphite blocks, light asphalt, blue-black ground.
BG_TOP = (0.082, 0.095, 0.115, 1.0)
BG_BOTTOM = (0.042, 0.051, 0.066, 1.0)
BLOCK = np.array((0.250, 0.268, 0.300), np.float32)
ROAD_MINOR = np.array((0.170, 0.190, 0.225), np.float32)
ROAD_MAJOR = np.array((0.330, 0.360, 0.410), np.float32)
FOG = np.array((0.045, 0.055, 0.072), np.float32)

_SKIP_ROADS = frozenset((
    "footway", "pedestrian", "path", "steps", "cycleway", "bridleway",
    "track", "corridor", "platform", "construction", "proposed",
))
_MAJOR_ROADS = frozenset((
    "motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link",
    "trunk_link", "primary_link", "secondary_link", "tertiary_link",
))


def _rect_overlap(a, b):
    return (a[0] < b[0] + b[2] and a[0] + a[2] > b[0]
            and a[1] < b[1] + b[3] and a[1] + a[3] > b[1])


@dataclass
class Massing:
    """Flat building footprints in tile-local ground coordinates."""

    tris: np.ndarray        # (F, 3, 2) local xy
    centres: np.ndarray     # (F, 2) face centroid for cheap culling
    kind: np.ndarray        # (F,) 0 building
    tint: np.ndarray        # (F,) per-building brightness

    @property
    def empty(self) -> bool:
        return len(self.tris) == 0

    @classmethod
    def blank(cls) -> "Massing":
        return cls(np.empty((0, 3, 2), np.float32), np.empty((0, 2), np.float32),
                   np.empty((0,), np.int8), np.empty((0,), np.float32))


def _components(xy: np.ndarray, cell: float = CELL):
    """Group triangles that share a quantised XY vertex."""
    q = np.round(np.asarray(xy, np.float64) / cell).astype(np.int64)
    unique, inverse = np.unique(q, axis=0, return_inverse=True)
    vid = inverse.reshape(-1, 3).astype(np.int64)
    n = len(unique)
    if n == 0:
        return unique * cell, vid, np.empty((0,), np.int64)
    e0 = np.concatenate((vid[:, 0], vid[:, 1], vid[:, 2]))
    e1 = np.concatenate((vid[:, 1], vid[:, 2], vid[:, 0]))
    parent = np.arange(n, dtype=np.int64)
    for _ in range(120):
        old = parent
        np.minimum.at(parent, e0, old[e1])
        np.minimum.at(parent, e1, old[e0])
        parent = parent[parent]
        if np.array_equal(parent, old):
            break
    return unique * cell, vid, parent


def _face_area(flat: np.ndarray) -> np.ndarray:
    a = flat[:, 1] - flat[:, 0]
    b = flat[:, 2] - flat[:, 0]
    return 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def build_massing(mesh, driving=None) -> Massing:
    """Flatten one resident tile's roofs into ground-plane footprints.

    Runs in the tile worker (or once for the offline demo), never on the
    render thread.  Returns an empty massing when the tile has no buildings.
    """
    if mesh is None or not len(getattr(mesh, "indices", ())):
        return Massing.blank()
    try:
        tri = np.asarray(_mesh_triangles(mesh), np.float32)
    except Exception:
        return Massing.blank()
    if len(tri) == 0:
        return Massing.blank()

    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal, axis=1)
    length[length < 1e-9] = 1.0
    roof = (normal[:, 2] / length) > 0.55
    flat = np.ascontiguousarray(tri[roof, :, :2], np.float32)
    if len(flat) == 0:
        return Massing.blank()

    area = _face_area(flat)
    keep = area >= MIN_ROOF_AREA
    flat, area = flat[keep], area[keep]
    if len(flat) == 0:
        return Massing.blank()

    _points, vid, parent = _components(flat.reshape(-1, 2))
    if len(parent) == 0:
        return Massing.blank()
    building = parent[vid[:, 0]]
    total = np.bincount(building, weights=area.astype(np.float64),
                        minlength=len(parent))
    keep = total[building] >= MIN_BUILDING_AREA
    flat, building = flat[keep], building[keep]
    if len(flat) == 0:
        return Massing.blank()

    # Stable per-building brightness: hash the component id, no extra state.
    table = (np.arange(len(parent), dtype=np.uint64) * np.uint64(2654435761))
    table &= np.uint64(0xFFFFFFFF)
    tint_table = (0.90 + 0.14 * (table.astype(np.float64) / 4294967295.0)).astype(np.float32)
    return Massing(flat, flat.mean(axis=1, dtype=np.float32),
                   np.zeros(len(flat), np.int8), tint_table[building])


class MinimapPanel:
    """The bottom-left inset.  Draws only in scene/driving views."""

    def __init__(self):
        self._rect = None
        self._cam = None
        self._focus_px = None
        self._place_hits = []

    # ------------------------------------------------------------------ layout
    def layout(self, w, h, status_h=0.0):
        if w < 520 or h < 380:
            return None
        row_h = max(36.0, float(status_h or 0.0))
        bottom = h - T.MARGIN - row_h - 12.0
        height = min(CARD_H, bottom - T.MARGIN - 72.0)
        if height < 116.0:
            return None
        width = CARD_W
        return (T.MARGIN, bottom - height, width, height)

    def hit(self, pos):
        return self._rect is not None and _in(self._rect, *pos)

    def clear(self):
        self._rect = None
        self._cam = None
        self._place_hits.clear()

    def place_at(self, pos):
        for cx, cy, place, label in self._place_hits:
            if math.hypot(pos[0] - cx, pos[1] - cy) <= 11. or (label is not None and _in(label, *pos)):
                return place
        return None

    def pick(self, pos):
        """Screen point -> world ground XY, for a right-click destination pin."""
        if not self.hit(pos) or self._cam is None: return None
        cam = self._cam
        ray = (cam["forward"] + cam["right"] * ((pos[0] - cam["cx"]) / cam["focal"])
               + cam["up"] * ((cam["cy"] - pos[1]) / cam["focal"]))
        if abs(ray[2]) < 1e-9: return None
        t = (cam["plane_z"] - cam["eye"][2]) / ray[2]
        if t <= 0.: return None
        point = cam["eye"] + ray * t
        return tuple(point[:2]) if np.isfinite(point).all() else None

    # ------------------------------------------------------------------ camera
    def _camera(self, focus, heading, rect):
        x, y, w, h = rect
        fx, fy = math.cos(heading), math.sin(heading)
        plane_z = float(focus[2])
        target = np.array([focus[0] + fx * AHEAD, focus[1] + fy * AHEAD,
                           plane_z + 1.0], np.float64)
        eye = np.array([target[0] - fx * DISTANCE * math.cos(PITCH),
                        target[1] - fy * DISTANCE * math.cos(PITCH),
                        plane_z + DISTANCE * math.sin(PITCH)], np.float64)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        focal = h * 1.16
        return dict(eye=eye, forward=forward, right=right, up=up, focal=focal,
                    cx=x + w * 0.5, cy=y + h * 0.56, plane_z=plane_z)

    @staticmethod
    def _project_flat(xy, cam):
        """Project ground-plane points (..., 2) through the pitched camera.

        Every feature lies on the same Z plane, so the depth offset is a
        scalar and the whole projection stays in float32.
        """
        eye, right, up, fwd = cam["eye"], cam["right"], cam["up"], cam["forward"]
        ex, ey, ez = float(eye[0]), float(eye[1]), cam["plane_z"] - float(eye[2])
        dx = xy[..., 0] - ex
        dy = xy[..., 1] - ey
        xc = dx * float(right[0]) + dy * float(right[1]) + ez * float(right[2])
        yc = dx * float(up[0]) + dy * float(up[1]) + ez * float(up[2])
        zc = dx * float(fwd[0]) + dy * float(fwd[1]) + ez * float(fwd[2])
        safe = np.maximum(zc, 1.0)
        scale = np.float32(cam["focal"])
        sx = cam["cx"] + scale * (xc / safe)
        sy = cam["cy"] - scale * (yc / safe)
        return np.stack((sx, sy), axis=-1), zc

    # ------------------------------------------------------------------- draw
    def draw(self, ui, w, h, mouse, scene, focus, heading, status_h=0.0, navigation=None):
        rect = self.layout(w, h, status_h)
        self._place_hits.clear()
        if rect is None:
            self.clear()
            return
        self._rect = rect
        x, y, cw, ch = rect
        hover = ui.anim(("minimap", "hover"), float(_in(rect, *mouse)), 16.0)
        ui.container(x, y, cw, ch, "panel", r=14)
        ui.box(x + 1, y + 1, cw - 2, ch - 2, 13, fill=(BG_TOP, BG_BOTTOM))

        cam = self._camera(focus, heading, rect)
        self._cam = cam
        mark = self._project_flat(np.asarray(focus[:2], np.float32)[None, :], cam)[0][0]
        self._focus_px = (float(np.clip(mark[0], x + 18, x + cw - 18)),
                          float(np.clip(mark[1], y + 24, y + ch - 24)))

        verts, colors = self._batch(scene, focus, heading, rect, cam)
        if len(verts):
            ui.triangles(verts, colors, clip=rect, clip_radius=13)

        if navigation is not None and navigation.route is not None:
            points = navigation.route.tail(navigation.progress)
            projected, depth = self._project_flat(points[:, :2].astype(np.float32), cam)
            route_stroke(ui, projected, rect, (depth[:-1] > 1.) & (depth[1:] > 1.), radius=13.)

        self._places(ui, scene, focus, rect, cam, mouse, navigation)
        if navigation is not None and navigation.destination is not None:
            self._destination(ui, navigation.destination, rect, cam)

        self._puck(ui, self._focus_px[0], self._focus_px[1])
        self._chrome(ui, rect, heading, hover)
        border = T.BORDER_FOCUS if hover > 0.5 else T.BORDER_PANEL
        ui.box(x, y, cw, ch, 14, fill=None, border=border, bw=1.0)

    def _places(self, ui, scene, focus, rect, cam, mouse, navigation):
        places = scene_places(scene)
        if not places: return
        x, y, w, h = rect
        candidates = sorted((p for p in places if math.hypot(p.x - focus[0], p.y - focus[1]) < MAP_RADIUS + 60.),
                            key=lambda p: (math.hypot(p.x - focus[0], p.y - focus[1]), p.key))
        occupied = [(self._focus_px[0] - 21, self._focus_px[1] - 24, 42., 42.)]
        labels = 0
        for place in candidates:
            if navigation is not None and navigation.destination is not None and place.key == navigation.destination.key:
                continue
            point, depth = self._project_flat(np.asarray(place.position, np.float32)[None], cam)
            cx, cy = map(float, point[0])
            if depth[0] <= 1. or not (x + 14 < cx < x + w - 14 and y + 37 < cy < y + h - 32):
                continue
            if any(_in(r, cx, cy) for r in occupied): continue
            place_symbol(ui, cx, cy, place.category, 7.)
            label_rect = None
            hover = math.hypot(mouse[0] - cx, mouse[1] - cy) <= 12.
            if labels < 3 or hover:
                shown = ui.fit(place.name, T.T_CAPTION, min(84., x + w - cx - 16.))
                tw = ui.text_w(shown, T.T_CAPTION, weight="medium")
                candidate = (cx + 10, cy - 7, tw + 3, 15.)
                if tw >= 18 and not any(_rect_overlap(candidate, r) for r in occupied):
                    ui.text_c(shown, cx + 10, cy, T.T_CAPTION, T.TEXT_2, weight="medium", halo=True)
                    label_rect, labels = candidate, labels + 1
                    occupied.append(candidate)
            occupied.append((cx - 11, cy - 11, 22., 22.))
            self._place_hits.append((cx, cy, place, label_rect))
            if len(self._place_hits) >= 10: break

    def _destination(self, ui, destination, rect, cam):
        x, y, w, h = rect
        point, depth = self._project_flat(np.asarray(destination.position, np.float32)[None], cam)
        if not np.isfinite(point).all(): return
        px, py = map(float, point[0])
        inside = depth[0] > 1. and x + 18 < px < x + w - 18 and y + 39 < py < y + h - 32
        if not inside:
            # Keep an off-screen goal on the inset edge, rather than losing it.
            direction = np.array((px - self._focus_px[0], py - self._focus_px[1]))
            if depth[0] <= 1.: direction *= -1.
            if np.linalg.norm(direction) < 1e-6: direction[:] = (0., -1.)
            direction /= np.linalg.norm(direction)
            centre = np.asarray(self._focus_px)
            limits = ((x + 18, x + w - 18), (y + 40, y + h - 32))
            distances = [(limits[i][1 if direction[i] > 0 else 0] - centre[i]) / direction[i]
                         for i in range(2) if abs(direction[i]) > 1e-6]
            px, py = centre + direction * max(0., min(distances))
            ui.circle(px, py, 9., (.035, .02, .04, .95))
            normal = np.array((-direction[1], direction[0]))
            tip = np.array((px, py)) + direction * 6.
            back = np.array((px, py)) - direction * 3.
            ui.tri(tuple(tip), tuple(back + normal * 4.), tuple(back - normal * 4.), (1., .31, .34, 1.))
        else:
            destination_pin(ui, px, py, 8.)
            shown = ui.fit(destination.name, T.T_CAPTION, min(110., w - 36), weight="semibold")
            tw = ui.text_w(shown, T.T_CAPTION, weight="semibold")
            lx = max(x + 10, min(px - tw * .5, x + w - tw - 10))
            ui.text(shown, lx, py - 25, T.T_CAPTION, T.TEXT, weight="semibold", halo=True)

    # ------------------------------------------------------------- geometry
    @staticmethod
    def _flat(placed):
        """World-space flat geometry for one resident tile, built once.

        Returns ``(tris, kind, tint, centres)`` or ``None``.  ``kind`` is
        0 for a building footprint, 1 for a minor road, 2 for a major road.
        """
        cached = getattr(placed, "_minimap_flat", None)
        if cached is not None:
            return cached

        origin = np.asarray(placed.origin, np.float64)
        scale = np.asarray(placed.scale, np.float64)
        ox, oy = np.float32(origin[0]), np.float32(origin[1])
        sx, sy = np.float32(scale[0]), np.float32(scale[1])
        parts, kinds, tints = [], [], []

        mass = getattr(placed, "minimap", None)
        if mass is not None and not mass.empty:
            tris = mass.tris.astype(np.float32, copy=True)
            tris[:, :, 0] = tris[:, :, 0] * sx + ox
            tris[:, :, 1] = tris[:, :, 1] * sy + oy
            parts.append(tris)
            kinds.append(mass.kind)
            tints.append(mass.tint)

        roads, road_kind = _road_geometry(placed, origin, scale)
        if len(roads):
            parts.append(roads)
            kinds.append(road_kind)
            tints.append(np.ones(len(roads), np.float32))

        if not parts:
            # Cache the empty case too: otherwise a road-less tile would
            # rebuild (and reallocate) its nothing every single frame.
            result = (np.empty((0, 3, 2), np.float32), np.empty((0,), np.int8),
                      np.empty((0,), np.float32), np.empty((0, 2), np.float32))
        else:
            tris = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            kind = np.concatenate(kinds)
            tint = np.concatenate(tints).astype(np.float32, copy=False)
            result = (tris, kind, tint, tris.mean(axis=1, dtype=np.float32))
        try:
            placed._minimap_flat = result
        except AttributeError:
            pass
        return result

    def _batch(self, scene, focus, heading, rect, cam=None):
        x, y, w, h = rect
        cam = cam if cam is not None else self._camera(focus, heading, rect)
        empty = (np.empty((0, 2), np.float32), np.empty((0, 4), np.float32))
        tiles = getattr(scene, "tiles", None) if scene is not None else None
        if not tiles:
            return empty

        focus_xy = np.asarray(focus[:2], np.float64)
        reach = MAP_RADIUS + 60.0
        tris, kind, tint = [], [], []
        for placed in tiles.values():
            geo = self._flat(placed)
            if geo is None:
                continue
            t, k, sh, centres = geo
            near = np.hypot(centres[:, 0] - focus_xy[0],
                            centres[:, 1] - focus_xy[1]) <= reach
            if not near.any():
                continue
            tris.append(t[near])
            kind.append(k[near])
            tint.append(sh[near])
        if not tris:
            return empty

        tris = np.concatenate(tris, axis=0)
        kind = np.concatenate(kind)
        tint = np.concatenate(tint)
        screen, depth = self._project_flat(tris, cam)
        finite = np.isfinite(screen).all(axis=(1, 2))
        if not finite.all():
            tris, screen, depth = tris[finite], screen[finite], depth[finite]
            kind, tint = kind[finite], tint[finite]
        if len(tris) == 0:
            return empty

        sx, sy = screen[:, :, 0], screen[:, :, 1]
        inside = ((sx.max(axis=1) >= x - 24) & (sx.min(axis=1) <= x + w + 24)
                  & (sy.max(axis=1) >= y - 24) & (sy.min(axis=1) <= y + h + 24))
        if not inside.all():
            tris, screen, depth = tris[inside], screen[inside], depth[inside]
            kind, tint = kind[inside], tint[inside]
        if len(tris) == 0:
            return empty

        # colours ------------------------------------------------------------
        base = np.empty((len(kind), 3), np.float32)
        base[:] = BLOCK
        base[kind == 1] = ROAD_MINOR
        base[kind == 2] = ROAD_MAJOR
        colour = base * tint[:, None]
        fade = np.clip((depth.mean(axis=1) - 55.0) / (MAP_RADIUS * 1.7), 0.0, 1.0)[:, None] * 0.85
        colour = colour * (1.0 - fade) + FOG[None, :] * fade

        colours = np.empty((len(kind), 3, 4), np.float32)
        colours[:, :, :3] = colour[:, None, :]
        colours[:, :, 3] = 1.0
        # Roads first so blocks paint over the ribbons that run beneath them.
        order = np.argsort(-kind, kind="stable")
        return (screen[order].reshape(-1, 2).astype(np.float32, copy=False),
                colours[order].reshape(-1, 4))

    # ------------------------------------------------------------------ chrome
    def _puck(self, ui, cx, cy):
        pulse = 0.5 + 0.5 * math.sin(getattr(ui, "now", 0.0) * 2.6)
        ui.circle(cx, cy, 12.0 + 2.0 * pulse, (0.04, 0.50, 1.0, 0.16 + 0.10 * pulse))
        ui.tri((cx, cy - 17.0), (cx - 5.5, cy - 5.0), (cx + 5.5, cy - 5.0),
               (0.10, 0.56, 1.0, 0.92))
        ui.circle(cx, cy, 7.2, (1.0, 1.0, 1.0, 1.0))
        ui.circle(cx, cy, 5.1, (0.03, 0.47, 1.0, 1.0))

    def _chrome(self, ui, rect, heading, hover):
        x, y, w, h = rect
        # 2D badge: flat ground plane, only the camera is perspective.
        ui.container(x + 10, y + 10, 30, 18, "inset", r=5, shadow=False)
        ui.text_c("2D", x + 25, y + 19, T.T_CAPTION, T.TEXT_2,
                  anchor="center", weight="semibold")
        # compass
        cx, cy = x + w - 22, y + 21
        ui.circle(cx, cy, 12.5, (0.02, 0.03, 0.05, 0.52))
        ui.ring(cx, cy, 12.5, 1.0, (1.0, 1.0, 1.0, 0.18), seg=20)
        vx, vy = -math.cos(heading), -math.sin(heading)
        ui.line(cx, cy, cx + vx * 8.5, cy + vy * 8.5, 2.0, (1.0, 1.0, 1.0, 0.92))
        ui.circle(cx + vx * 8.5, cy + vy * 8.5, 2.2, (1.0, 0.32, 0.28, 1.0))
        ui.text_c("N", cx + vx * 9.5, cy + vy * 9.5 - 0.5, T.T_CAPTION,
                  (1.0, 1.0, 1.0, 0.95), anchor="center", weight="semibold")

        # scale bar, picked from a nice round distance that fits the card
        px_per_m = self._cam["focal"] / DISTANCE if self._cam else 1.0
        metres = next((m for m in (100.0, 50.0, 25.0, 10.0) if m * px_per_m <= 62.0), 10.0)
        length = metres * px_per_m
        bx, by = x + w - 12, y + h - 15
        ui.line(bx - length, by, bx, by, 1.5, (1.0, 1.0, 1.0, 0.72))
        for tx in (bx - length, bx):
            ui.line(tx, by - 3, tx, by + 2, 1.5, (1.0, 1.0, 1.0, 0.72))
        label = f"{int(metres)} m"
        ui.text(label, bx - length, by - 14, T.T_CAPTION, T.TEXT_3,
                weight="medium")

        if hover > 0.5:
            ui.box(x + 8, y + h - 26, 74, 18, 6, fill=(0.015, 0.022, 0.034, 0.78))
            ui.text_c("Open map", x + 16, y + h - 17, T.T_CAPTION,
                      T.TEXT_2, weight="medium")


def _road_geometry(placed, origin, scale):
    """World-space road ribbons for one resident tile, as flat triangles."""
    tris, kinds = [], []
    for path in getattr(placed, "paths", ()) or ():
        if (getattr(path, "highway", "") in _SKIP_ROADS
                or getattr(path, "tunnel", False)
                or getattr(path, "covered", False)):
            continue
        pts = np.asarray(path.points, np.float64)
        if len(pts) < 2:
            continue
        px = (pts[:, 0] * scale[0] + origin[0]).astype(np.float32)
        py = (pts[:, 1] * scale[1] + origin[1]).astype(np.float32)
        dx, dy = np.diff(px), np.diff(py)
        span = np.hypot(dx, dy)
        keep = span >= 1.0
        if not keep.any():
            continue
        dx, dy, span = dx[keep], dy[keep], span[keep]
        x0, y0 = px[:-1][keep], py[:-1][keep]
        x1, y1 = px[1:][keep], py[1:][keep]
        half = np.float32(max(2.0, min(15.0, float(getattr(path, "width", 6.0)) * 0.5)))
        nx, ny = (-dy / span * half).astype(np.float32), (dx / span * half).astype(np.float32)
        a = np.stack((x0 + nx, y0 + ny), axis=1)
        b = np.stack((x1 + nx, y1 + ny), axis=1)
        c = np.stack((x1 - nx, y1 - ny), axis=1)
        d = np.stack((x0 - nx, y0 - ny), axis=1)
        tri = np.empty((len(span) * 2, 3, 2), np.float32)
        tri[0::2] = np.stack((a, b, c), axis=1)
        tri[1::2] = np.stack((a, c, d), axis=1)
        tris.append(tri)
        kinds.append(np.full(len(tri),
                             2 if getattr(path, "highway", "") in _MAJOR_ROADS else 1,
                             np.int8))
    if not tris:
        return np.empty((0, 3, 2), np.float32), np.empty((0,), np.int8)
    return np.concatenate(tris, axis=0), np.concatenate(kinds)
