"""Live, destination-based driving guidance over the resident OSM road network.

The navigation worker reuses traffic's directed centreline topology, not its
expensive lane/render splines. It never reads or mutates the AI worker's graph.
Only one request can run at a time; newer destinations/tile generations replace
obsolete work instead of filling a queue. No network or route search runs in a
frame. ETA is an estimate from road speed limits, not a live-traffic feed.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import heapq
import math
import time

import numpy as np

from ..world.data_fetcher import DRIVABLE_HIGHWAYS
from ..traffic.road_graph import (GraphTile, RoadGraph, STRAIGHT, TURN_LEFT, TURN_RIGHT,
                         UTURN, _allowed_turns, _arc, _slice, turn_kind)


OFF_ROUTE_METRES = 22.
ARRIVAL_METRES = 14.
TURN_LABELS = {STRAIGHT: "Continue", TURN_LEFT: "Turn left", TURN_RIGHT: "Turn right",
               UTURN: "Make a U-turn", "arrive": "Arrive at destination"}
STATUS_LABELS = {
    "idle": "Choose a destination", "routing": "Finding a route...",
    "rerouting": "Rerouting...", "waiting": "Waiting for loaded roads",
    "outside": "Destination outside loaded roads", "offroad": "Join a loaded road",
    "no_route": "No connected driving route", "error": "Navigation unavailable",
    "arrived": "You have arrived",
}


@dataclass(frozen=True)
class Destination:
    name: str
    x: float
    y: float
    key: str = ""

    @classmethod
    def from_place(cls, place):
        return cls(place.name, place.x, place.y, place.key)

    @property
    def position(self):
        return (self.x, self.y)


@dataclass(frozen=True)
class Maneuver:
    at: float
    kind: int | str
    road: str = ""


@dataclass(frozen=True)
class NavigationRoute:
    points: np.ndarray             # world XYZ; elevation prevents cross-deck arrival
    arc: np.ndarray
    seconds: np.ndarray
    maneuvers: tuple
    initial_road: str
    endpoint_offset: float          # place entrance may be set back from the road

    @property
    def length(self):
        return float(self.arc[-1])

    @property
    def duration(self):
        return float(self.seconds[-1])

    def tail(self, progress):
        progress = max(0., min(self.length, float(progress)))
        k = max(0, min(len(self.arc) - 2, int(np.searchsorted(self.arc, progress, side="right")) - 1))
        t = (progress - self.arc[k]) / max(self.arc[k + 1] - self.arc[k], 1e-9)
        return np.vstack((self.points[k] + (self.points[k + 1] - self.points[k]) * t,
                          self.points[k + 1:]))


@dataclass(frozen=True)
class _Snap:
    edge: int
    arc: float
    distance: float
    penalty: float = 0.


class NavigationNetwork:
    """Fractional-edge A*: respects one-way roads, source nodes and road grades."""

    def __init__(self, tiles):
        tiles = tuple(tiles)
        self.nodes, self.edges, self.lines = RoadGraph.topology(tiles)
        self.arcs = [_arc(line) for line in self.lines]
        self.paths = {(tile.key, i): path for tile in tiles for i, path in enumerate(tile.paths)}
        self.max_speed = max((edge.limit for edge in self.edges), default=30.)
        self.successors = []
        for edge in self.edges:
            node = self.nodes[edge.end]
            neighbours = {self.edges[i].start for i in node.incoming} | {self.edges[i].end for i in node.outgoing}
            specs = set().union(*(_allowed_turns(spec) for spec in edge.turn_specs)) if edge.turn_specs else None
            following = []
            for eid in node.outgoing:
                turn = turn_kind(self.edges[eid].heading_start - edge.heading_end)
                if turn == UTURN and len(neighbours) > 1:
                    continue
                if node.junction and specs is not None and turn not in specs:
                    continue
                following.append((eid, {STRAIGHT: 0., TURN_LEFT: 4., TURN_RIGHT: 1., UTURN: 14.}[turn]))
            self.successors.append(tuple(following))
        # Vectorised nearest-segment queries are only made by the worker.
        a, b, ids, starts, lengths = [], [], [], [], []
        for edge, line, arc in zip(self.edges, self.lines, self.arcs):
            a.append(line[:-1]); b.append(line[1:])
            ids.extend([edge.id] * (len(line) - 1))
            starts.append(arc[:-1]); lengths.append(np.diff(arc))
        self.a = np.concatenate(a) if a else np.empty((0, 3))
        self.d = np.concatenate(b) - self.a if b else np.empty((0, 3))
        self.ids = np.asarray(ids, np.int32)
        self.starts = np.concatenate(starts) if starts else np.empty(0)
        self.lengths = np.concatenate(lengths) if lengths else np.empty(0)
        self.den = np.maximum(np.sum(self.d[:, :2] ** 2, axis=1), 1e-9)

    def _snaps(self, position, heading=None):
        if not len(self.a): return ()
        xy = np.asarray(position[:2], np.float64)
        t = np.clip(np.sum((xy - self.a[:, :2]) * self.d[:, :2], axis=1) / self.den, 0., 1.)
        q = self.a + self.d * t[:, None]
        distance = np.linalg.norm(q[:, :2] - xy, axis=1)
        score = distance.copy()
        if heading is not None:
            if len(position) > 2:
                score += np.abs(q[:, 2] - float(position[2])) * 5.
            direction = np.arctan2(self.d[:, 1], self.d[:, 0])
            score += (1. - np.cos(direction - heading)) * 3.
        nearest = float(score.min())
        candidates = np.flatnonzero(score <= nearest + 7.)
        order = candidates[np.argsort(score[candidates], kind="stable")]
        seen, out = set(), []
        for k in order:
            eid = int(self.ids[k])
            if eid in seen: continue
            seen.add(eid)
            # Nearby unrelated streets cannot become a shortcut through a block.
            if distance[k] > float(distance.min()) + 4. and heading is None:
                continue
            penalty = float(distance[k]) / 4.
            if heading is not None:
                if len(position) > 2 and abs(q[k, 2] - position[2]) > 3.5:
                    continue
                angle = math.atan2(self.d[k, 1], self.d[k, 0]) - heading
                penalty += (1. - math.cos(angle)) * 12.
            out.append(_Snap(eid, float(self.starts[k] + self.lengths[k] * t[k]),
                             float(distance[k]), penalty))
            if len(out) >= 8: break
        return tuple(out)

    def find_route(self, origin, destination, heading=None):
        if not self.edges: return "waiting", None
        starts = self._snaps(origin, heading)
        goals = self._snaps(destination.position)
        if not goals or min(s.distance for s in goals) > 120.: return "outside", None
        if not starts or min(s.distance for s in starts) > 80.: return "offroad", None
        starts = tuple(s for s in starts if s.distance <= 80.)
        goals = tuple(s for s in goals if s.distance <= 120.)
        targets = {}
        for goal in goals: targets.setdefault(goal.edge, []).append(goal)
        # Heuristic is to a snapped endpoint, never through the destination building.
        positions = np.array([node.position[:2] for node in self.nodes])
        endpoints = [np.array([np.interp(g.arc, self.arcs[g.edge], self.lines[g.edge][:, i])
                               for i in range(2)]) for g in goals]
        heuristic = np.min([np.linalg.norm(positions - p, axis=1) for p in endpoints], axis=0) / self.max_speed
        best = [math.inf] * len(self.edges)
        previous = [-1] * len(self.edges)
        roots, heap = {}, []
        final_cost, final = math.inf, None
        for start in starts:
            edge = self.edges[start.edge]
            for goal in targets.get(start.edge, ()):
                if goal.arc >= start.arc:
                    cost = start.penalty + (goal.arc - start.arc) / edge.limit + goal.penalty
                    if cost < final_cost:
                        final_cost, final = cost, (start, goal, (start.edge,))
            cost = start.penalty + (edge.length - start.arc) / edge.limit
            if cost < best[edge.id]:
                best[edge.id], roots[edge.id] = cost, start
                heapq.heappush(heap, (cost + float(heuristic[edge.end]), cost, edge.id))
        while heap:
            estimate, cost, arrival = heapq.heappop(heap)
            if estimate >= final_cost: break
            if cost > best[arrival] + 1e-9: continue
            for eid, turn_penalty in self.successors[arrival]:
                edge = self.edges[eid]
                for goal in targets.get(eid, ()):
                    candidate = cost + turn_penalty + goal.arc / edge.limit + goal.penalty
                    if candidate < final_cost:
                        chain, current = [eid], arrival
                        while current >= 0:
                            chain.append(current)
                            current = previous[current]
                        final_cost, final = candidate, (roots[arrival], goal, tuple(reversed(chain)))
                candidate = cost + turn_penalty + edge.length / edge.limit
                if candidate + 1e-9 < best[eid]:
                    best[eid], previous[eid], roots[eid] = candidate, arrival, roots[arrival]
                    heapq.heappush(heap, (candidate + float(heuristic[edge.end]), candidate, eid))
        if final is None: return "no_route", None
        return "navigating", self._compile(*final, heading=heading)

    def _road_name(self, eid):
        edge = self.edges[eid]
        return getattr(self.paths[edge.source], "name", "") or edge.highway.replace("_", " ").title() + " road"

    def _compile(self, start, goal, chain, heading=None):
        points, seconds, turns = [], [], []
        elapsed = 0.
        for i, eid in enumerate(chain):
            edge = self.edges[eid]
            a = start.arc if i == 0 else 0.
            b = goal.arc if i == len(chain) - 1 else edge.length
            line = _slice(self.lines[eid], self.arcs[eid], a, b)
            arc = _arc(line)
            if i:
                previous = self.edges[chain[i - 1]]
                turn = turn_kind(edge.heading_start - previous.heading_end)
                if turn != STRAIGHT or self._road_name(eid) != self._road_name(previous.id):
                    turns.append((len(points) - 1, turn, self._road_name(eid)))
                elapsed += {STRAIGHT: 0., TURN_LEFT: 4., TURN_RIGHT: 1., UTURN: 14.}[turn]
            skip = 1 if points and np.linalg.norm(points[-1] - line[0]) < .25 else 0
            points.extend(line[skip:])
            seconds.extend((elapsed + arc[skip:] / edge.limit).tolist())
            elapsed += float(arc[-1]) / edge.limit
        if len(points) < 2:
            points = [points[0], points[0]]
            seconds = [0., 0.]
        points, seconds = np.asarray(points, np.float64), np.asarray(seconds, np.float64)
        arc = _arc(points)
        keep = np.r_[True, np.diff(arc) > 1e-6]
        # Keep a zero-length same-position route so it can become "arrived".
        if keep.sum() >= 2:
            remap = np.cumsum(keep) - 1
            points, arc, seconds = points[keep], arc[keep], seconds[keep]
            turns = [(int(remap[k]), kind, road) for k, kind, road in turns]
        maneuvers = [Maneuver(float(arc[k]), kind, road) for k, kind, road in turns]
        first = self.edges[chain[0]]
        initial = points[1, :2] - points[0, :2]
        initial_heading = math.atan2(initial[1], initial[0]) if np.linalg.norm(initial) > 1e-6 else first.heading_start
        if heading is not None:
            initial_turn = turn_kind(initial_heading - heading)
            if initial_turn != STRAIGHT:
                maneuvers.insert(0, Maneuver(0., initial_turn, self._road_name(first.id)))
        maneuvers.append(Maneuver(float(arc[-1]), "arrive"))
        for array in (points, arc, seconds): array.flags.writeable = False
        return NavigationRoute(points, arc, seconds, tuple(maneuvers), self._road_name(chain[0]), goal.distance)


def project_route(route, position, progress=0., lookahead=120.):
    """Track a local arc window so a crossing/loop cannot skip the whole trip."""
    a, d = route.points[:-1], np.diff(route.points, axis=0)
    t = np.clip(np.sum((np.asarray(position)[:2] - a[:, :2]) * d[:, :2], axis=1)
                / np.maximum(np.sum(d[:, :2] ** 2, axis=1), 1e-9), 0., 1.)
    q = a + d * t[:, None]
    along = route.arc[:-1] + np.diff(route.arc) * t
    distance = np.linalg.norm(q[:, :2] - np.asarray(position)[:2], axis=1)
    if len(position) > 2:
        vertical = np.abs(q[:, 2] - position[2])
        distance += np.maximum(0., vertical - 1.) * 4.
        distance = np.where(vertical > 3.5, np.maximum(distance, OFF_ROUTE_METRES + 1.), distance)
    eligible = (along >= progress - 30.) & (along <= progress + lookahead)
    if eligible.any():
        distance = np.where(eligible, distance, np.inf)
    k = int(np.argmin(distance + np.abs(along - progress) * .001))
    return float(along[k]), float(distance[k])


class NavigationController:
    """Render-thread state and progress; the single worker owns the cached network."""

    def __init__(self, threaded=True):
        self.threaded = threaded
        self._pool = self._future = self._request = None
        self._network = self._network_key = None
        self._scene = self._scene_key = None
        self._scene_epoch = self._version = 0
        self._need_route = False
        self._closed = False
        self._last_request = -math.inf
        self._request_position = None
        self._off_route_since = None
        self.destination = self.route = None
        self.status, self.error = "idle", None
        self.progress = 0.

    def set_destination(self, destination):
        if not np.isfinite(destination.position).all():
            raise ValueError("Destination coordinates must be finite")
        self.clear()
        self.destination = destination
        self.status, self._need_route = "routing", True

    def clear(self):
        self._version += 1
        if self._future is not None: self._future.cancel()
        self.destination = self.route = None
        self.status, self.error, self.progress = "idle", None, 0.
        self._need_route = False
        self._last_request = -math.inf
        self._request_position = self._off_route_since = None

    @property
    def remaining(self):
        return max(0., self.route.length - self.progress) if self.route is not None else 0.

    @property
    def eta_seconds(self):
        if self.route is None or self.status == "arrived": return 0.
        return max(0., self.route.duration - float(np.interp(self.progress, self.route.arc, self.route.seconds)))

    @property
    def next_maneuver(self):
        if self.route is None: return None
        for maneuver in self.route.maneuvers:
            if maneuver.at >= self.progress - 6.:
                return maneuver
        return self.route.maneuvers[-1]

    @property
    def instruction(self):
        if self.status != "navigating": return STATUS_LABELS.get(self.status, "Navigation")
        return TURN_LABELS[self.next_maneuver.kind]

    @property
    def turn_distance(self):
        maneuver = self.next_maneuver
        return max(0., maneuver.at - self.progress) if maneuver is not None else 0.

    def _work(self, key, tiles, origin, destination, heading):
        if key != self._network_key:
            self._network = NavigationNetwork(tiles)
            self._network_key = key
        return self._network.find_route(origin, destination, heading)

    def _take(self):
        if self._future is None or not self._future.done(): return
        future, request = self._future, self._request
        self._future = self._request = None
        if future.cancelled(): return
        if request != (self._version, self._scene_key) or self.destination is None:
            self._need_route = self.destination is not None
            return
        try:
            self.status, self.route = future.result()
            self.error = None
        except Exception as exc:
            self.status, self.route, self.error = "error", None, str(exc)
        self.progress, self._off_route_since = 0., None

    def update(self, scene, position, heading=None, speed=0., now=None):
        if self._closed: return
        now = time.monotonic() if now is None else float(now)
        if scene is not self._scene:
            self._scene, self._scene_epoch = scene, self._scene_epoch + 1
        key = (self._scene_epoch, getattr(scene, "generation", 0))
        if key != self._scene_key:
            self._scene_key = key
            self._need_route = self.destination is not None and self.status != "arrived"
        self._take()
        if self.destination is None: return
        if self.route is not None and self.status != "arrived":
            along, lateral = project_route(self.route, position, self.progress, max(120., abs(speed) * 8.))
            if lateral <= OFF_ROUTE_METRES:
                self.progress = along
                self._off_route_since = None
                end = self.route.points[-1]
                near_end = math.hypot(position[0] - end[0], position[1] - end[1]) < 25.
                same_deck = len(position) < 3 or abs(position[2] - end[2]) < 2.5
                if self.remaining <= ARRIVAL_METRES and lateral < ARRIVAL_METRES and near_end and same_deck:
                    self.status, self.progress, self._need_route = "arrived", self.route.length, False
            else:
                if self._off_route_since is None: self._off_route_since = now
                if now - self._off_route_since >= .8:
                    self.status, self._need_route = "rerouting", True
        elif self.status not in ("routing", "arrived") and self._request_position is not None:
            if np.linalg.norm(np.asarray(position[:2]) - self._request_position) >= 35.:
                self._need_route = True
        if not self._need_route or self._future is not None or now - self._last_request < 1.:
            return
        tiles = tuple(GraphTile(tile_key, tuple(p for p in placed.paths if p.highway in DRIVABLE_HIGHWAYS),
                                tuple(placed.origin), tuple(placed.scale))
                      for tile_key, placed in getattr(scene, "tiles", {}).items())
        self._last_request, self._need_route = now, False
        self._request_position = np.asarray(position[:2], np.float64).copy()
        self._request = (self._version, self._scene_key)
        if self.status != "rerouting": self.status = "routing"
        args = (key, tiles, tuple(position), self.destination, heading)
        if self.threaded:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="navigation")
            self._future = self._pool.submit(self._work, *args)
        else:
            self._future = Future()
            try: self._future.set_result(self._work(*args))
            except Exception as exc: self._future.set_exception(exc)
        self._take()

    def close(self):
        self._closed = True
        self.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        self._pool = self._future = self._request = None
        self._network = self._network_key = self._scene = None
