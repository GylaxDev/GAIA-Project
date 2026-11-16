"""Procedural pedestrian crowd that lives on the city's sidewalks.

* ``SidewalkNetwork`` derives walkable polylines from the road paths of every loaded
  tile (a band beside each carriageway, exactly where ``urban.build_sidewalks`` lays the
  pavement), built incrementally under a small per-frame time budget while tiles stream.
* ``PedestrianCrowd`` spawns people out of sight around the viewer, walks them along
  those polylines at their own pace, hands them on to the next street at corners (a
  straight link across the junction), lets them idle, phone, text, open umbrellas in the
  rain, glance at the player, sidestep each other and run from fast cars.
* Every person is generated from one integer seed: appearance (palette indices, hair,
  height, accessories) and a ctOS-style profile (``identity_for``).

The crowd only produces a (N, 24) float32 instance array; drawing is
``pedestrian_renderer.PedestrianRenderer``.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np

from .pedestrian_gait import gait
from .pedestrian_models import (ACC_UMBRELLA, ACC_PHONE_EAR, ACC_PHONE_LOOK, ACC_BACKPACK, ACC_COAT,
                                ACC_SKIRT, ACC_HAIR_SHORT, ACC_HAIR_LONG, ACC_HAIR_BUN, ACC_CAP,
                                ACC_GLASSES, ACC_BAG, ACC_IDLE, ACC_COFFEE, ACC_WATCH, ACC_BEARD,
                                POSE_POCKETS, POSE_CROSSED, POSE_WATCH, POSE_TALK)
from .human_assets import ACC_FORMAL, ACC_HOOD, MALE, FEMALE
from .walker import (PED_STRIDE, SceneQuery, wrap, C_HEADING, C_PHASE, C_AMP, C_SPEED, C_FLAGS,
                     C_HEIGHT, C_WIDTH, C_HEADSCALE, C_SEED, C_SKIN, C_HAIR, C_TOP, C_BOTTOM, C_SHOE,
                     C_ACCENT, C_TOPSTYLE, C_BOTSTYLE, C_BULK, C_VIS, C_HEADYAW, C_NOD, C_VARIANT)
from ..vehicles.vehicle_instances import LENGTHS, WIDTHS

try:                                                   # same constants the pavement mesh uses
    from ..world.urban import SIDEWALK_WIDTH, SIDEWALK_HEIGHT
except Exception:                                      # pragma: no cover - headless fallback
    SIDEWALK_WIDTH, SIDEWALK_HEIGHT = 2.15, .17
SIDEWALK_RISE = SIDEWALK_HEIGHT + .064                 # walking surface above the road surface

PEDESTRIAN_WAYS = frozenset(("pedestrian", "living_street", "footway", "path", "corridor"))
NO_SIDEWALK = frozenset(("motorway", "motorway_link"))

WALK, IDLE, PANIC = 0, 1, 2

# ---------------------------------------------------------------------------------------
# Procedural identity (the ctOS profiler card)
# ---------------------------------------------------------------------------------------
_FIRST = ("Dani", "Marcus", "Priya", "Tomas", "Aiko", "Jonah", "Lena", "Omar", "Chloe", "Rafael", "Mei", "Gareth",
          "Nadia", "Felix", "Isla", "Kofi", "Sasha", "Ravi", "Elena", "Mateo", "Hana", "Declan", "Amara", "Viktor",
          "Zoe", "Arjun", "Carmen", "Niall", "Yuki", "Leila", "Tobias", "Imani", "Bruno", "Sofia", "Kenji", "Maya",
          "Anton", "Rosa", "Idris", "Greta", "Pablo", "Noor", "Callum", "Ingrid", "Devon", "Thea", "Samir", "Wren",
          "Luca", "Farah", "Jasper", "Ines", "Haruto", "Mina", "Dmitri", "Abby", "Tariq", "Lucia", "Evan", "Bao")
_LAST = ("Sagalla", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Brennan", "Castillo", "Novak", "Reyes",
         "Whitaker", "Ibrahim", "Kowalski", "Mendes", "Park", "Rossi", "Abara", "Fitzgerald", "Chen", "Duarte",
         "Petrov", "Holloway", "Nakamura", "Santos", "Vance", "Aydin", "Larsen", "Mukherjee", "Quinn", "Delgado",
         "Fraser", "Osei", "Varga", "Ito", "Bianchi", "Eriksen", "Navarro", "Sato", "Kapoor", "Beaumont",
         "Zielinski", "Adeyemi", "McAllister", "Ferreira", "Hollis", "Yilmaz", "Rahman", "Thorne", "Marlowe", "Cruz")
_TRAITS = ("Habitual procrastinator", "Chronic insomniac", "Aspiring novelist", "Coffee addict", "Compulsive over-sharer",
           "Cat person", "Perpetual optimist", "Karaoke champion", "Vinyl collector", "Always runs late",
           "Gym regular", "Online shopping addict", "Night owl", "Amateur chef", "Chess hustler", "Light sleeper",
           "Reality TV devotee", "Weekend hiker", "Serial hobbyist", "Pun enthusiast", "Podcast binger",
           "Crossword fanatic", "Plant parent", "Overthinker", "Impulse buyer", "Early riser", "Lapsed guitarist",
           "Loyal to one barista", "Hates Mondays", "Conspiracy-curious")
_JOBS = (("Registered Nurse", 54000, 91000), ("Line Cook", 26000, 41000), ("Software Developer", 68000, 150000),
         ("Barista", 21000, 33000), ("Bike Courier", 22000, 38000), ("Accountant", 52000, 96000),
         ("High School Teacher", 41000, 79000), ("Electrician", 47000, 88000), ("Graphic Designer", 38000, 82000),
         ("Paralegal", 42000, 74000), ("Bus Driver", 36000, 62000), ("Warehouse Supervisor", 39000, 66000),
         ("Pharmacist", 98000, 135000), ("Journalist", 34000, 78000), ("Retail Associate", 20000, 34000),
         ("Architect", 61000, 120000), ("Paramedic", 38000, 72000), ("Data Analyst", 58000, 110000),
         ("Dental Hygienist", 62000, 98000), ("Lab Technician", 36000, 68000), ("Taxi Dispatcher", 28000, 47000),
         ("Property Manager", 45000, 92000), ("Physiotherapist", 58000, 99000), ("Sound Engineer", 33000, 86000),
         ("Librarian", 38000, 68000), ("Hotel Concierge", 27000, 49000), ("Delivery Driver", 24000, 43000),
         ("Mechanic", 34000, 69000), ("Marketing Manager", 62000, 125000), ("Freelance Photographer", 18000, 72000),
         ("Postal Worker", 40000, 64000), ("Chef de Partie", 32000, 58000), ("City Planner", 58000, 104000),
         ("Event Coordinator", 36000, 70000), ("Translator", 32000, 76000), ("Building Inspector", 52000, 90000))


@dataclass
class Identity:
    surname: str
    given: str
    trait: str
    age: int
    occupation: str
    income: int
    seed: int

    @property
    def name(self):
        return f"{self.surname}, {self.given}"

    @property
    def income_text(self):
        return f"${self.income:,}"


def identity_for(seed, age, student=False):
    """Deterministic fictional profile for a person seed. Quirks only; nothing sensitive."""
    rng = random.Random(int(seed) * 2654435761 % (2 ** 32))
    if student:
        job, lo, hi = "University Student", 6000, 19000
    else:
        job, lo, hi = rng.choice(_JOBS)
    t = (rng.random() + rng.random()) * .5
    income = int(round((lo + (hi - lo) * t) / 500.) * 500)
    return Identity(rng.choice(_LAST), rng.choice(_FIRST), rng.choice(_TRAITS), int(age), job, income, int(seed))


# ---------------------------------------------------------------------------------------
# Sidewalk network
# ---------------------------------------------------------------------------------------
class Walkway:
    __slots__ = ("pts", "z", "arc", "length", "centre", "radius", "pl", "zl", "al", "uid", "kind", "side")

    def __init__(self, pts, z, kind, uid=0, side=0.):
        self.pts = np.ascontiguousarray(pts, np.float64)
        self.z = np.ascontiguousarray(z, np.float64)
        seg = np.hypot(*np.diff(self.pts, axis=0).T)
        self.arc = np.r_[0., np.cumsum(seg)]
        self.length = float(self.arc[-1])
        lo, hi = self.pts.min(axis=0), self.pts.max(axis=0)
        self.centre = (lo + hi) * .5
        self.radius = float(np.hypot(*(hi - lo)) * .5) + 1.
        self.pl, self.zl, self.al = self.pts.tolist(), self.z.tolist(), self.arc.tolist()
        self.uid, self.kind, self.side = uid, kind, side   # side: +1 buildings on the path's left, -1 right, 0 unknown


def _offset(points, d):
    """Parallel polyline at signed distance d (left of travel positive), mitre-limited."""
    t = np.diff(points, axis=0)
    t /= np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-9)[:, None]
    n = np.column_stack((-t[:, 1], t[:, 0]))
    v = np.empty_like(points)
    v[0], v[-1] = n[0], n[-1]
    if len(points) > 2:
        v[1:-1] = n[:-1] + n[1:]
    m = np.hypot(v[:, 0], v[:, 1])
    v /= np.maximum(m, 1e-6)[:, None]
    seg = np.vstack((n, n[-1:]))
    k = 1. / np.maximum((v * seg).sum(axis=1), .55)
    return points + v * (d * k)[:, None]


def tile_walkways(placed, seen):
    out = []
    for path in placed.paths:
        if getattr(path, 'tunnel', False) or getattr(path, 'covered', False):
            continue
        level = getattr(path, 'level', None)
        if level is not None and level < 0:
            continue
        pts = np.asarray(path.points, np.float64)
        if pts.ndim != 2 or len(pts) < 2 or not np.isfinite(pts).all():
            continue
        world = pts * placed.scale + placed.origin
        z = path.z
        z = np.zeros(len(pts)) if z is None else np.asarray(z, np.float64)
        if len(z) != len(pts):
            z = np.full(len(pts), float(z.mean()) if len(z) else 0.)
        width, total, kind = float(path.width), float(path.total), path.highway
        uid = int(getattr(path, 'uid', 0))
        lines = []
        if kind in PEDESTRIAN_WAYS:
            lines.append((world, z + .02, 'path', 0.))
        elif width >= 5. and total >= 30. and kind not in NO_SIDEWALK:
            sides = getattr(path, 'sidewalks', ()) or (True, True)
            dist = width * .5 + SIDEWALK_WIDTH * .5
            for side, sign in zip(sides, (1., -1.)):
                if side:
                    lines.append((_offset(world, sign * dist), z + SIDEWALK_RISE, 'sidewalk', sign))
        for line, lz, lkind, lside in lines:
            if float(np.hypot(*np.diff(line, axis=0).T).sum()) < 8.:
                continue
            mid = line[len(line) // 2]
            key = (round(float(mid[0]) / 2.), round(float(mid[1]) / 2.), lkind)
            if key in seen:                                   # a two-way road carries two records
                continue
            seen.add(key)
            out.append(Walkway(line, lz, lkind, uid, lside))
    return out


class SidewalkNetwork:
    CELL = 16.

    def __init__(self):
        self.tiles, self._ids, self._seen = {}, {}, {}
        self._pending = []
        self.walks = []
        self.centres = np.zeros((0, 2))
        self.radii = self.lengths = np.zeros(0)
        self.ends = {}

    @property
    def busy(self):
        return bool(self._pending)

    def sync(self, scene, budget=.003):
        keys = set(scene.tiles)
        dirty = False
        for key in list(self.tiles):
            if key not in keys or self._ids.get(key) != id(scene.tiles[key]):
                self.tiles.pop(key, None)
                self._ids.pop(key, None)
                self._seen.pop(key, None)
                dirty = True
        queued = set(self._pending)
        for key in keys:
            if key not in self.tiles and key not in queued:
                self._pending.append(key)
        start = time.perf_counter()
        while self._pending and time.perf_counter() - start < budget:
            key = self._pending.pop()
            placed = scene.tiles.get(key)
            if placed is None:
                continue
            seen = set()
            for other in self._seen.values():
                seen |= other
            self.tiles[key] = tile_walkways(placed, seen)
            self._seen[key] = seen
            self._ids[key] = id(placed)
            dirty = True
        if dirty:
            self._rebuild()

    def _rebuild(self):
        self.walks = [w for tile in self.tiles.values() for w in tile]
        n = len(self.walks)
        self.centres = np.array([w.centre for w in self.walks]).reshape(n, 2)
        self.radii = np.array([w.radius for w in self.walks])
        self.lengths = np.array([w.length for w in self.walks])
        self.ends = {}
        for i, w in enumerate(self.walks):
            for end, point in ((0, w.pl[0]), (1, w.pl[-1])):
                cell = (int(point[0] // self.CELL), int(point[1] // self.CELL))
                self.ends.setdefault(cell, []).append((i, end))

    def nearby_ends(self, x, y, radius):
        cx, cy = int(x // self.CELL), int(y // self.CELL)
        reach = int(radius // self.CELL) + 1
        found = []
        for gx in range(cx - reach, cx + reach + 1):
            for gy in range(cy - reach, cy + reach + 1):
                for i, end in self.ends.get((gx, gy), ()):
                    w = self.walks[i]
                    px, py = w.pl[0] if end == 0 else w.pl[-1]
                    d = math.hypot(px - x, py - y)
                    if d <= radius:
                        found.append((d, w, end))
        return found

    def length_near(self, x, y, radius):
        if not len(self.walks):
            return 0.
        d = np.hypot(self.centres[:, 0] - x, self.centres[:, 1] - y)
        return float(self.lengths[d - self.radii < radius].sum())


# ---------------------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------------------
class Ped:
    __slots__ = ("walk", "seg", "s", "dir", "lat", "lane", "lat_goal", "block", "speed", "vbase", "x", "y", "z", "off",
                 "walk_z", "heading", "phase", "amp", "sf", "vis", "fade", "mode", "timer", "walk_pose", "idle_pose",
                 "umb_thr", "gaze", "curious", "seed", "age", "student", "row", "link", "height", "leg", "jogger",
                 "idle_rate", "tx", "ty", "prev_walk", "acc", "width", "leader", "pair_side", "talk", "face", "wait", "wait_t",
                 "check_t", "glance_t", "glance", "bulk", "vwob", "mood", "window")


def _pick(rng, weights):
    return int(rng.choice(len(weights), p=np.asarray(weights, float) / float(np.sum(weights))))


class PedestrianCrowd:
    MAX_PEDS = 150
    SPAWN_MIN, SPAWN_MAX, DESPAWN = 22., 105., 128.
    LINK_RADIUS = 15.
    CROSSING_GAP = 5.5               # links longer than this cross a carriageway: wait for a gap

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.network = SidewalkNetwork()
        self.peds = []
        self.density = 1.0
        self.enabled = True
        self.time = 0.
        self._seed_counter = int(self.rng.integers(1, 2 ** 30))
        self._rr = 0
        self._frame = 0
        self._warm = 0.
        self._cars = None
        self._rain = 0.
        self.query = SceneQuery()
        self.rows = np.zeros((0, PED_STRIDE), np.float32)

    def clear(self):
        self.peds.clear()
        self.rows = np.zeros((0, PED_STRIDE), np.float32)

    # ------------------------------------------------------------------ creation
    def _look(self):
        rng = self.rng
        age = int(np.clip(rng.normal(37, 14), 18, 80))
        female = bool(rng.random() < .52)
        skin = _pick(rng, (.12, .18, .18, .16, .14, .10, .08, .04))
        long_hair = rng.random() < (.62 if female else .08)
        if age > 60 and rng.random() < .65:
            hair = int(rng.choice((5, 6)))
        elif skin >= 4:
            hair = int(rng.choice((0, 0, 0, 1, 1, 2)))
        else:
            hair = int(rng.choice((0, 1, 1, 2, 2, 3, 4, 4, 5)))
        acc = 0
        if long_hair:
            acc |= ACC_HAIR_LONG if rng.random() < .75 else ACC_HAIR_BUN
        elif rng.random() < .93:
            acc |= ACC_HAIR_SHORT
        if not female and not long_hair and age > 23 and rng.random() < .22:
            acc |= ACC_BEARD
        if rng.random() < .08:
            acc |= ACC_CAP
        if rng.random() < .16:
            acc |= ACC_GLASSES
        if rng.random() < (.30 if age < 26 else .10):
            acc |= ACC_BACKPACK
        elif rng.random() < .09:
            acc |= ACC_BAG
        if rng.random() < .33:
            acc |= ACC_WATCH
        coat = rng.random() < .10
        formal = not coat and rng.random() < .24
        hood = not formal and rng.random() < .18
        if coat:
            acc |= ACC_COAT
        elif formal:
            acc |= ACC_FORMAL
        elif female and rng.random() < .30:
            acc |= ACC_SKIRT
        if hood:
            acc |= ACC_HOOD
        height = float(np.clip(rng.normal(1.65 if female else 1.76, .075), 1.50, 1.96))
        build = float(np.clip(rng.normal(1.0 + .0012 * (age - 37), .075), .88, 1.22))
        return dict(age=age, skin=skin, hair=hair, acc=acc, coat=coat, height=height, bulk=build,
                    female=female, variant=FEMALE if female else MALE, formal=formal, hood=hood,
                    width=float(np.clip(rng.normal(1.0 if female else 1.03, .05), .90, 1.12)),
                    head=float(np.clip(rng.normal(1.0, .025), .94, 1.07)),
                    top=int(rng.integers(0, 15)), bottom=int(rng.integers(0, 15)),
                    shoe=int(rng.integers(0, 7)), accent=int(rng.integers(0, 15)),
                    long_sleeve=bool(coat or formal or hood or rng.random() < .45),
                    shorts=bool(not coat and not formal and not hood and not (acc & ACC_SKIRT)
                                and rng.random() < .10))

    def _make(self, walk, s, direction, fade=True, leader=None):
        rng = self.rng
        p = Ped()
        look = self._look()
        self._seed_counter += 1
        p.seed = self._seed_counter
        p.age, p.student = look['age'], bool((look['acc'] & ACC_BACKPACK) and look['age'] < 26)
        p.height, p.width, p.bulk = look['height'], look['width'], look['bulk']
        p.leg = .85 * look['height'] / 1.78
        p.walk, p.s, p.dir, p.seg, p.link, p.prev_walk = walk, float(s), int(direction), 0, None, None
        p.lane = float(rng.uniform(-.65, .65))
        p.lat = p.lat_goal = p.lane
        p.block, p.off = False, 0.
        p.leader, p.pair_side, p.talk, p.face, p.window = leader, 1., False, None, False
        p.wait, p.wait_t, p.check_t, p.glance_t, p.glance, p.mood = False, 0., 0., float(rng.uniform(1., 5.)), 0., 0.
        roll = rng.random()
        p.jogger = roll < .04 and look['age'] < 55
        slow = 1. - .004 * max(look['age'] - 45, 0)                    # older people walk a little slower
        if p.jogger:
            p.vbase = float(rng.uniform(2.7, 3.7))
        elif roll < .14:
            p.vbase = float(rng.uniform(1.7, 2.0)) * slow
        else:
            p.vbase = float(np.clip(rng.normal(1.30, .15), .90, 1.65)) * slow
        p.vwob = float(rng.uniform(0., 6.28))
        acc = look['acc']
        p.walk_pose = 0
        if not p.jogger:
            r = rng.random()
            if r < .07:
                p.walk_pose = ACC_PHONE_EAR
            elif r < .12:
                p.walk_pose = ACC_PHONE_LOOK
                p.vbase = min(p.vbase, 1.05)
            elif r < .27:
                p.walk_pose = POSE_POCKETS
            elif r < .34 and not (acc & ACC_BACKPACK):
                acc |= ACC_COFFEE
        p.idle_pose = 0
        p.idle_rate = 0. if p.jogger else float(rng.uniform(.004, .02))
        p.umb_thr = float(rng.uniform(.10, .45)) if rng.random() < .82 else 9.
        p.speed = p.vbase * .9
        p.mode, p.timer = WALK, 0.
        p.phase = float(rng.uniform(0, 2 * math.pi))
        p.amp = p.sf = 0.
        p.gaze, p.curious = 0., bool(rng.random() < .55)
        p.vis, p.fade = (0., 1.) if fade else (1., 0.)
        p.acc = acc
        row = np.zeros(PED_STRIDE, np.float32)
        row[C_HEIGHT], row[C_WIDTH], row[C_HEADSCALE] = look['height'], look['width'], look['head']
        row[C_SEED] = (p.seed * .6180339887) % 1.
        row[C_SKIN], row[C_HAIR], row[C_TOP], row[C_BOTTOM] = look['skin'], look['hair'], look['top'], look['bottom']
        row[C_SHOE], row[C_ACCENT] = look['shoe'], look['accent']
        row[C_TOPSTYLE], row[C_BOTSTYLE] = float(look['long_sleeve']), float(look['shorts'])
        row[C_BULK] = look['bulk']
        row[C_VARIANT] = look['variant']
        p.row = row
        p.x, p.y = self._position(p)
        p.walk_z, p.z, p.heading = self._z(p), self._z(p), 0.
        p.tx, p.ty = self._tangent(p)
        p.heading = math.atan2(p.ty, p.tx)
        return p

    def identity(self, ped):
        return identity_for(ped.seed, ped.age, ped.student)

    @staticmethod
    def _locate(p):
        walk, al = p.walk, p.walk.al
        seg = min(max(p.seg, 0), len(al) - 2)
        while seg < len(al) - 2 and p.s > al[seg + 1]:
            seg += 1
        while seg > 0 and p.s < al[seg]:
            seg -= 1
        p.seg = seg
        span = al[seg + 1] - al[seg]
        return seg, ((p.s - al[seg]) / span if span > 1e-6 else 0.)

    def _position(self, p):
        seg, t = self._locate(p)
        a, b = p.walk.pl[seg], p.walk.pl[seg + 1]
        bx, by = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        tx, ty = self._tangent(p)
        return bx - ty * p.lat, by + tx * p.lat

    def _z(self, p):
        seg, t = self._locate(p)
        z = p.walk.zl
        return z[seg] + (z[seg + 1] - z[seg]) * t

    def _tangent(self, p):
        seg, _ = self._locate(p)
        a, b = p.walk.pl[seg], p.walk.pl[seg + 1]
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.
        return dx / n * p.dir, dy / n * p.dir

    # ------------------------------------------------------------------ spawning
    def _spawn(self, anchor, lo, hi, fade=True):
        net = self.network
        if not len(net.walks):
            return 0
        d = np.hypot(net.centres[:, 0] - anchor[0], net.centres[:, 1] - anchor[1])
        ok = (d - net.radii < hi) & (d + net.radii > lo)
        if not ok.any():
            return 0
        idx = np.flatnonzero(ok)
        w = net.lengths[idx]
        walk = net.walks[int(self.rng.choice(idx, p=w / w.sum()))]
        for _ in range(6):
            s = float(self.rng.uniform(1.5, max(walk.length - 1.5, 1.6)))
            x = float(np.interp(s, walk.arc, walk.pts[:, 0]))
            y = float(np.interp(s, walk.arc, walk.pts[:, 1]))
            dist = math.hypot(x - anchor[0], y - anchor[1])
            if lo <= dist <= hi:
                if any(abs(q.x - x) < 1.4 and abs(q.y - y) < 1.4 for q in self.peds):
                    continue
                direction = 1 if self.rng.random() < .5 else -1
                lead = self._make(walk, s, direction, fade)
                self.peds.append(lead)
                if self.rng.random() < .24 and walk.length > 6.:            # a friend walking alongside
                    mate = self._make(walk, s, direction, fade, leader=lead)
                    mate.vbase = lead.vbase
                    mate.walk_pose = 0 if self.rng.random() < .7 else mate.walk_pose
                    mate.pair_side = 1. if self.rng.random() < .5 else -1.
                    lead.talk = mate.talk = True
                    lead.mood = mate.mood = float(self.rng.uniform(0., 6.28))
                    mate.curious = False
                    lead.walk_pose = lead.walk_pose if lead.walk_pose in (0, POSE_POCKETS) else 0
                    mate.acc &= ~ACC_COFFEE if self.rng.random() < .5 else mate.acc
                    self.peds.append(mate)
                    return 2
                return 1
        return 0

    # ------------------------------------------------------------------ corners
    def _arrive(self, p, end):
        walk = p.walk
        ex, ey = walk.pl[-1] if end else walk.pl[0]
        tx, ty = p.tx, p.ty
        best, best_score = None, -9.
        for d, other, e2 in self.network.nearby_ends(ex, ey, self.LINK_RADIUS):
            if other is walk or other is p.prev_walk:
                continue
            sx, sy = other.pl[0] if e2 == 0 else other.pl[-1]
            if d > .5:
                align = ((sx - ex) * tx + (sy - ey) * ty) / d
            else:
                i0 = 1 if e2 == 0 else -2
                ox, oy = other.pl[i0][0] - sx, other.pl[i0][1] - sy
                n = math.hypot(ox, oy) or 1.
                align = (ox * tx + oy * ty) / n
            score = align + float(self.rng.random()) * .9 - d * .02
            if align > -.35 and score > best_score:
                best, best_score = (other, e2, sx, sy), score
        if best is None:
            if self.rng.random() < .7:
                p.dir = -p.dir
                p.s = walk.length if end else 0.
                p.prev_walk = None
            else:
                p.fade = -1.
            return
        other, e2, sx, sy = best
        z0 = walk.zl[-1] if end else walk.zl[0]
        z1 = other.zl[0] if e2 == 0 else other.zl[-1]
        gap = math.hypot(sx - ex, sy - ey)
        p.link = [ex, ey, z0, sx, sy, z1, 0., max(gap, .01)]
        p.prev_walk = walk
        p.walk, p.dir = other, 1 if e2 == 0 else -1
        p.s, p.seg = (0. if e2 == 0 else other.length), (0 if e2 == 0 else len(other.pl) - 2)
        if gap > self.CROSSING_GAP:                       # a road to cross: stop at the kerb and look
            p.wait, p.wait_t, p.check_t = True, 0., 0.
            r = self.rng.random()
            p.idle_pose = (POSE_CROSSED if r < .22 else POSE_POCKETS if r < .38 else ACC_PHONE_LOOK if r < .55
                           else POSE_WATCH if r < .65 and (p.acc & ACC_WATCH) else 0)

    def _unsafe(self, link):
        """Would a car reach the crossing within the next few seconds?"""
        if self._cars is None or not len(self._cars):
            return False
        rows = self._cars
        a = np.array(link[0:2]); b = np.array(link[3:5])
        ab = b - a
        L2 = max(float(ab @ ab), 1e-6)
        pos = rows[:, 12:14].astype(np.float64)
        fwd = rows[:, 0:2].astype(np.float64)
        fwd /= np.maximum(np.hypot(fwd[:, 0], fwd[:, 1]), 1e-6)[:, None]
        sp = rows[:, 22].astype(np.float64)
        near = np.hypot(pos[:, 0] - a[0], pos[:, 1] - a[1]) < 45. + np.abs(sp) * 2.
        moving = np.abs(sp) > .8
        sel = near & moving
        if not sel.any():
            return False
        pos, fwd, sp = pos[sel], fwd[sel], sp[sel]
        t = np.linspace(0., 3.8, 9)
        path = pos[:, None, :] + fwd[:, None, :] * (sp[:, None, None] * t[None, :, None])      # sp is signed
        proj = np.clip(((path - a) @ ab) / L2, 0., 1.)
        closest = a + proj[..., None] * ab
        d = np.hypot(*(path - closest).transpose(2, 0, 1))
        return bool((d < 3.4).any())

    # ------------------------------------------------------------------- update
    def update(self, dt, scene, anchor, anchor_height=2., *, player=None, vehicles=None, rain=0.,
               night=0., camera_xy=None):
        """Advance the crowd. anchor = world xy to populate around; player = (xy, speed) or None;
        vehicles = (rows, kinds) of nearby cars; returns the instance array (also self.rows)."""
        dt = float(min(max(dt, 1e-4), .1))
        self.time += dt
        self._frame += 1
        self._rain = rain
        self._cars = vehicles[0] if vehicles is not None else None
        self.network.sync(scene)
        anchor = np.asarray(anchor[:2], np.float64)
        active = self.enabled and anchor_height < 160.

        if active and len(self.network.walks):
            near = self.network.length_near(anchor[0], anchor[1], self.SPAWN_MAX)
            target = min(self.MAX_PEDS * self.density * (1. - .78 * night) * (1. - .35 * rain), near / 16.)
            living = sum(1 for q in self.peds if q.fade >= 0.)
            self._warm += dt
            lo = self.SPAWN_MIN if self._warm > 2.5 or living > 12 else 9.
            for _ in range(5 if living < target * .6 else 2):
                if living >= target:
                    break
                living += self._spawn(anchor, lo, self.SPAWN_MAX)
            if living > target + 6:                         # night fell / rain came: thin out the farthest
                far = max((q for q in self.peds if q.fade >= 0. and q.leader is None),
                          key=lambda q: (q.x - anchor[0]) ** 2 + (q.y - anchor[1]) ** 2, default=None)
                if far is not None:
                    far.fade = -1.
        elif not active:
            for q in self.peds:
                q.fade = -1.

        if self.peds:
            self._interact(dt, anchor, player, vehicles)
            far2 = self.DESPAWN ** 2
            for i, p in enumerate(self.peds):
                if p.leader is not None:
                    continue
                if (p.x - anchor[0]) ** 2 + (p.y - anchor[1]) ** 2 > far2:
                    p.fade = -1.
                near_view = (p.x - anchor[0]) ** 2 + (p.y - anchor[1]) ** 2 < 55. ** 2
                if near_view or (self._frame + i) % 2 == 0:
                    self._step(p, dt if near_view else dt * 2., player, rain)
            alive = set(map(id, self.peds))
            for p in self.peds:                             # companions follow their leader
                if p.leader is not None:
                    if id(p.leader) not in alive or p.leader.fade < 0. and p.leader.vis <= 0.:
                        p.leader = None
                    else:
                        self._follow(p, dt, player)
            self._refine_ground(scene)
            keep = [p for p in self.peds if not (p.fade < 0. and p.vis <= 0.)]
            gone = {id(p) for p in self.peds} - {id(p) for p in keep}
            for p in keep:
                if p.leader is not None and id(p.leader) in gone:
                    p.leader = None
            self.peds = keep
        self._pack(rain)
        return self.rows

    # --------------------------------------------------- crowd-level interaction
    def _interact(self, dt, anchor, player, vehicles):
        peds = self.peds
        n = len(peds)
        P = np.array([(p.x, p.y) for p in peds])
        T = np.array([(p.tx, p.ty) if (p.mode != IDLE and p.link is None and p.leader is None) else (0., 0.)
                      for p in peds])
        extra = []
        if player is not None:
            extra.append((player[0][0], player[0][1]))
        Q = np.vstack((P, np.array(extra).reshape(-1, 2))) if extra else P
        D = Q[None, :, :] - P[:, None, :]
        dist = np.hypot(D[..., 0], D[..., 1])
        dist[np.arange(n), np.arange(n)] = 99.
        along = D[..., 0] * T[:, None, 0] + D[..., 1] * T[:, None, 1]
        cross = T[:, None, 0] * D[..., 1] - T[:, None, 1] * D[..., 0]
        reach = np.full(Q.shape[0], 1.5)
        if extra:
            reach[-1] = 2.2
        mask = (dist < reach[None, :]) & (along > -.2) & (np.abs(T).sum(axis=1)[:, None] > 0)
        # companions do not dodge each other
        for i, p in enumerate(peds):
            if p.leader is not None:
                j = peds.index(p.leader) if p.leader in peds else -1
                if j >= 0:
                    mask[i, j] = False
                    mask[j, i] = False
        sign = np.where(np.abs(cross) < .05, np.where(np.arange(Q.shape[0])[None, :] % 2 == 0, 1., -1.), np.sign(cross))
        push = -(sign * np.clip(reach[None, :] - dist, 0., 1.5) * .7 * mask).sum(axis=1)
        blocked = (mask & (along > .1) & (along < 1.2) & (np.abs(cross) < .5)).any(axis=1)
        for i, p in enumerate(peds):
            if p.mode == IDLE and p.window:
                continue
            p.lat_goal = float(np.clip(p.lane + push[i], -.95, .95))
            p.block = bool(blocked[i])

        if vehicles is not None and len(vehicles[0]):
            rows = vehicles[0]
            fast = np.abs(rows[:, 22]) > 2.2
            if fast.any():
                rows = rows[fast]
                kinds = np.asarray(vehicles[1])[fast]
                C = rows[:, 12:14].astype(np.float64)
                F = rows[:, 0:2].astype(np.float64)
                F /= np.maximum(np.hypot(F[:, 0], F[:, 1]), 1e-6)[:, None]
                rel = P[:, None, :] - C[None, :, :]
                f = rel[..., 0] * F[None, :, 0] + rel[..., 1] * F[None, :, 1]
                l = -rel[..., 0] * F[None, :, 1] + rel[..., 1] * F[None, :, 0]
                sp = np.abs(rows[:, 22])[None, :]
                half_l = np.array([LENGTHS[int(k)] for k in kinds])[None, :] * .5
                half_w = np.array([WIDTHS[int(k)] for k in kinds])[None, :] * .5
                danger = ((f > -half_l - .8) & (f < half_l + 2. + sp * .9) & (np.abs(l) < half_w + 1.5))
                for i in np.flatnonzero(danger.any(axis=1)):
                    p = peds[i]
                    if p.mode == PANIC or p.link is not None or p.leader is not None:
                        continue
                    j = int(np.argmax(danger[i]))
                    away = (P[i] - C[j])
                    ahead = away[0] * p.tx + away[1] * p.ty
                    if ahead < 0:
                        p.dir = -p.dir
                        p.tx, p.ty = -p.tx, -p.ty
                    p.mode, p.timer, p.idle_pose, p.wait = PANIC, 2.2, 0, False

    # ----------------------------------------------------------- per-person step
    def _step(self, p, dt, player, rain):
        rng = self.rng
        p.vis = min(1., p.vis + dt * 2.) if p.fade >= 0. else max(0., p.vis - dt * 1.8)
        if p.fade >= 0. and p.vis >= 1.:
            p.fade = 0.

        crossing = p.link is not None and not p.wait
        if p.mode == PANIC:
            p.timer -= dt
            want = 5.0
            if p.timer <= 0.:
                p.mode = WALK
        elif p.mode == IDLE:
            p.timer -= dt
            want = 0.
            if p.timer <= 0.:
                p.mode, p.idle_pose, p.window, p.face = WALK, 0, False, None
                p.lane = float(rng.uniform(-.65, .65))
        elif p.wait:
            want = 0.                                         # standing at the kerb
            p.wait_t += dt
            p.check_t -= dt
            if p.check_t <= 0.:
                p.check_t = .4
                if not self._unsafe(p.link) or p.wait_t > 28.:
                    p.wait = False
                    p.idle_pose = 0
        else:
            want = p.vbase * (1. + .07 * math.sin(self.time * .35 + p.vwob))
            if crossing:
                want = max(want, 1.35) * 1.15                 # cross briskly
            if p.idle_rate and p.link is None and p.leader is None and rng.random() < p.idle_rate * dt:
                p.mode, p.timer = IDLE, float(rng.uniform(5., 18.))
                r = rng.random()
                p.idle_pose = (ACC_PHONE_EAR if r < .22 else ACC_PHONE_LOOK if r < .45 else POSE_CROSSED if r < .58
                               else POSE_WATCH if r < .68 and (p.acc & ACC_WATCH) else POSE_POCKETS if r < .78 else 0)
                if p.walk.side != 0. and rng.random() < .35:            # window-shopping: face the buildings
                    p.window, p.idle_pose = True, (0 if p.idle_pose in (ACC_PHONE_LOOK, ACC_PHONE_EAR) else p.idle_pose)
                    seg, _ = self._locate(p)
                    a, b = p.walk.pl[seg], p.walk.pl[seg + 1]
                    dx, dy = b[0] - a[0], b[1] - a[1]
                    n = math.hypot(dx, dy) or 1.
                    p.face = math.atan2(dx / n * p.walk.side, -dy / n * p.walk.side)
                    p.lane = max(-.95, min(.95, .85 * p.walk.side * p.dir))
                    p.lat_goal = p.lane
            elif p.block:
                want *= .35
        accel = 2.2 if want > p.speed else 3.4
        p.speed += max(-accel * dt, min(accel * dt, want - p.speed))
        if p.speed < .02 and want == 0.:
            p.speed = 0.

        p.lat += max(-.9 * dt, min(.9 * dt, p.lat_goal - p.lat))
        move = p.speed * dt
        if p.link is not None:
            k = p.link
            if not p.wait:
                k[6] += move / k[7]
            t = min(k[6], 1.)
            p.x, p.y = k[0] + (k[3] - k[0]) * t, k[1] + (k[4] - k[1]) * t
            p.walk_z = k[2] + (k[5] - k[2]) * t
            dx, dy = k[3] - k[0], k[4] - k[1]
            n = math.hypot(dx, dy) or 1.
            p.tx, p.ty = dx / n, dy / n
            if k[6] >= 1.:
                p.link = None
                p.tx, p.ty = self._tangent(p)
        else:
            p.s += move * p.dir
            walk = p.walk
            if p.s >= walk.length:
                p.s = walk.length
                self._arrive(p, 1)
            elif p.s <= 0.:
                p.s = 0.
                self._arrive(p, 0)
            if p.link is None:
                p.tx, p.ty = self._tangent(p)
                p.x, p.y = self._position(p)
                p.walk_z = self._z(p)
        p.z = p.walk_z + p.off

        # heading: look ahead along the path so corners are rounded; face a point of interest when idle
        if p.mode == IDLE and p.face is not None:
            goal, rate = p.face, 3.5
        else:
            goal, rate = math.atan2(p.ty, p.tx), 8.
            if p.link is None and p.speed > .3:
                seg, _ = self._locate(p)
                nxt = min(seg + 2, len(p.walk.pl) - 1) if p.dir > 0 else max(seg - 1, 0)
                pt = p.walk.pl[nxt]
                ahead = math.atan2(pt[1] - p.y, pt[0] - p.x) if p.dir > 0 else math.atan2(pt[1] - p.y, pt[0] - p.x)
                if math.hypot(pt[0] - p.x, pt[1] - p.y) < 4. + p.speed:
                    goal = goal + wrap(ahead - goal) * .45
        p.heading = wrap(p.heading + wrap(goal - p.heading) * (1. - math.exp(-rate * dt)))
        self._gaze(p, dt, player)

        amp, sf, rate = gait(p.speed, p.height, p.width)
        p.amp += (amp - p.amp) * (1. - math.exp(-12. * dt))
        p.sf = sf
        p.phase = (p.phase + rate * dt) % (2 * math.pi)

    def _gaze(self, p, dt, player):
        """Where the head points: both ways at a kerb, at the player, at a friend, now and then around."""
        goal = 0.
        if p.wait:
            goal = .95 * math.sin(p.wait_t * 2.1)               # look left and right for traffic
        elif p.leader is not None or (p.talk and self._partner(p) is not None):
            other = p.leader if p.leader is not None else self._partner(p)
            dx, dy = other.x - p.x, other.y - p.y
            rel = wrap(math.atan2(dy, dx) - p.heading)
            if abs(rel) < 2.4 and math.sin(self.time * .8 + p.mood) > -.3:
                goal = max(-1.1, min(1.1, rel))
        elif player is not None and p.curious and p.mode != PANIC:
            dx, dy = player[0][0] - p.x, player[0][1] - p.y
            if math.hypot(dx, dy) < 9.:
                rel = wrap(math.atan2(dy, dx) - p.heading)
                if abs(rel) < 2.2:
                    goal = max(-1.1, min(1.1, rel))
        if goal == 0.:
            p.glance_t -= dt
            if p.glance_t <= 0.:
                p.glance_t = float(self.rng.uniform(2., 7.))
                p.glance = float(self.rng.uniform(-.75, .75)) if self.rng.random() < .6 else 0.
            goal = p.glance
        p.gaze += (goal - p.gaze) * (1. - math.exp(-5. * dt))

    def _partner(self, p):
        for q in self.peds:
            if q.leader is p:
                return q
        return None

    def _follow(self, p, dt, player):
        """A companion keeps beside their leader, matching pace and stops, and talks."""
        lead = p.leader
        p.vis = min(1., p.vis + dt * 2.) if p.fade >= 0. else max(0., p.vis - dt * 1.8)
        if p.fade >= 0. and p.vis >= 1.:
            p.fade = 0.
        p.fade = lead.fade if lead.fade < 0. else p.fade
        p.walk, p.s, p.dir, p.seg, p.walk_z = lead.walk, lead.s, lead.dir, lead.seg, lead.walk_z
        p.mode, p.timer = lead.mode, lead.timer
        p.tx, p.ty = lead.tx, lead.ty
        nx, ny = -lead.ty, lead.tx
        spacing = .85 if lead.mode != IDLE else 1.05
        p.x = lead.x + nx * p.pair_side * spacing - lead.tx * (.25 if lead.mode != IDLE else 0.)
        p.y = lead.y + ny * p.pair_side * spacing - lead.ty * (.25 if lead.mode != IDLE else 0.)
        p.z = p.walk_z + p.off
        p.speed += (lead.speed * (1. + .03 * math.sin(self.time * .7 + p.vwob)) - p.speed) * min(1., 4. * dt)
        if lead.mode == IDLE:
            goal = math.atan2(lead.y - p.y, lead.x - p.x)           # stand facing each other
            rate = 3.
            p.idle_pose = 0 if p.idle_pose in (ACC_PHONE_EAR,) else p.idle_pose
        else:
            goal, rate = math.atan2(p.ty, p.tx), 8.
        if lead.mode == IDLE and lead.face is None:
            lead.face = math.atan2(p.y - lead.y, p.x - lead.x)
        p.heading = wrap(p.heading + wrap(goal - p.heading) * (1. - math.exp(-rate * dt)))
        self._gaze(p, dt, player)
        amp, sf, rate = gait(p.speed, p.height, p.width)
        p.amp += (amp - p.amp) * (1. - math.exp(-12. * dt))
        p.sf = sf
        p.phase = (p.phase + rate * dt) % (2 * math.pi)

    def _refine_ground(self, scene, per_frame=4):
        """Calibrate walking height against the real ground, a few people per frame."""
        n = len(self.peds)
        for _ in range(min(per_frame, n)):
            self._rr = (self._rr + 1) % n
            p = self.peds[self._rr]
            ground, water = self.query.ground(scene, p.x, p.y, p.z + .1, rise=.6)
            if ground is not None and (water is None or water <= ground + .02):
                err = ground - p.walk_z
                p.off += max(-.15, min(.15, max(-.7, min(.7, err)) - p.off))

    # ---------------------------------------------------------------------- pack
    def _pack(self, rain):
        n = len(self.peds)
        if self.rows.shape[0] != n:
            self.rows = np.zeros((n, PED_STRIDE), np.float32)
        rows = self.rows
        for i, p in enumerate(self.peds):
            r = rows[i]
            r[:] = p.row
            r[0], r[1], r[2], r[C_HEADING] = p.x, p.y, p.z, p.heading
            r[C_PHASE], r[C_AMP], r[C_SPEED] = p.phase, p.amp, p.sf
            umbrella = rain > p.umb_thr and p.mode != PANIC
            acc = p.acc
            if umbrella:
                acc |= ACC_UMBRELLA
                acc &= ~ACC_COFFEE                           # the free hand holds the umbrella
            if p.mode == PANIC:
                acc &= ~(ACC_COFFEE | ACC_UMBRELLA)
            pose = p.idle_pose if (p.mode == IDLE or p.wait) else (p.walk_pose if p.mode == WALK else 0)
            if umbrella and pose in (ACC_PHONE_LOOK, POSE_POCKETS, POSE_CROSSED):
                pose = 0
            if acc & ACC_COFFEE and pose in (ACC_PHONE_LOOK, POSE_CROSSED, POSE_POCKETS):
                pose = 0
            # conversation: the partner who is "speaking" gestures
            if p.talk and not pose and not (acc & (ACC_COFFEE | ACC_UMBRELLA)) and p.mode != PANIC:
                if math.sin(self.time * .55 + p.mood + (3.1 if p.leader is not None else 0.)) > .15:
                    pose = POSE_TALK
            if pose == ACC_PHONE_EAR and umbrella:
                pass
            flags = acc | pose
            if p.speed < .1:
                flags |= ACC_IDLE
            r[C_FLAGS] = flags
            r[C_VIS], r[C_HEADYAW] = p.vis, p.gaze
            r[C_NOD] = .0

    # ------------------------------------------------------------------- queries
    def living(self):
        return [p for p in self.peds if p.fade >= 0. and p.vis > .6]

    def head_position(self, p):
        return (p.x, p.y, p.z + p.height * .93)
