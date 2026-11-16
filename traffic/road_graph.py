"""Immutable, city-wide directed OSM segments, lane splines and turn routes.

Builds run on a worker. OSM node identity is authoritative; geometry intersection
is a fallback for older cached polylines without node IDs. Grade-separated ways
are never connected merely because they cross in the map. Every tile uses the
same world projection, including its x scale and road/deck elevation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import heapq
import math

import numpy as np

from .traffic_dynamics import (BRAKE_ENVELOPE, LATERAL_ACCELERATION, MIN_CURVE_SPEED,
                               road_priority, speed_limit)
from ..world.road_layout import RoadLayout, lane_counts, lane_offsets
from ..vehicles.vehicle_instances import LENGTHS

STRAIGHT, TURN_LEFT, TURN_RIGHT, UTURN = 0, 1, -1, 2
STOP_MARGIN = .75
BUMPER_HALF_WIDTH = 1.3  # conservative lateral body envelope, including buses
LANE_TRIM_FRACTION = .45  # leave a positive road core; short links need forward turn room


def turn_kind(angle):
    a = math.atan2(math.sin(angle), math.cos(angle))
    if abs(a) > 2.6:
        return UTURN
    return TURN_LEFT if a > .38 else TURN_RIGHT if a < -.38 else STRAIGHT


def _readonly(a):
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def _arc(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))]


def _interpolate(points, arc, values):
    return np.column_stack([np.interp(values, arc, points[:, k]) for k in range(3)])


def _slice(points, arc, a, b):
    inner = points[(arc > a + 1e-6) & (arc < b - 1e-6)]
    ends = _interpolate(points, arc, [a, b])
    return np.vstack((ends[:1], inner, ends[1:]))


def _smooth_line(points, radius=4.):
    """Round sharp vertices with tangent-continuous quadratic corner splines."""
    if len(points) < 3:
        return points
    out = [points[0]]
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        u, v = b - a, c - b
        lu, lv = np.linalg.norm(u[:2]), np.linalg.norm(v[:2])
        if min(lu, lv) < 1e-5:
            continue
        if float(u[:2] @ v[:2]) / (lu * lv) > .9994:
            out.append(b)
            continue
        trim = min(radius, lu * .28, lv * .28)
        entry, leave = b - u * (trim / lu), b + v * (trim / lv)
        out.append(entry)
        t = np.linspace(0., 1., max(5, int(trim * 2) + 1))[1:, None]
        out.extend((1 - t) ** 2 * entry + 2 * (1 - t) * t * b + t ** 2 * leave)
    out.append(points[-1])
    return np.asarray(out, np.float64)


def _lane_count(path, forward):
    return lane_counts(path)[0 if forward else 1]


def _allowed_turns(spec):
    turns = set()
    for name in str(spec).split(';'):
        if name in ('left', 'slight_left', 'sharp_left'):
            turns.add(TURN_LEFT)
        elif name in ('right', 'slight_right', 'sharp_right'):
            turns.add(TURN_RIGHT)
        elif name in ('through', 'straight'):
            turns.add(STRAIGHT)
        elif name == 'reverse':
            turns.add(UTURN)
    return frozenset(turns) if turns else frozenset((STRAIGHT, TURN_LEFT, TURN_RIGHT, UTURN))


@dataclass(frozen=True)
class GraphTile:
    key: tuple
    paths: tuple
    origin: tuple = (0., 0.)
    scale: tuple = (1., 1.)


@dataclass(frozen=True)
class RoadNode:
    id: int
    key: tuple
    position: np.ndarray
    incoming: tuple = ()
    outgoing: tuple = ()
    signal: bool = False
    junction: bool = False
    radius: float = 0.


@dataclass(frozen=True)
class RoadSegment:
    id: int
    key: tuple
    start: int
    end: int
    source: tuple                 # tile key, original path index
    forward: bool
    source_start: float           # original path arc at travel origin
    source_end: float
    length: float
    limit: float
    priority: int
    width: float
    heading_start: float
    heading_end: float
    highway: str
    lane_count: int
    turn_specs: tuple = ()
    lanes: tuple = ()
    oneway: bool | None = None    # None only for legacy diagnostic snapshots
    offsets: tuple = ()
    shared: bool = False          # two routing directions, one physical lane


@dataclass(frozen=True)
class LaneSpline:
    id: int
    key: tuple
    edge: int
    index: int
    points: np.ndarray
    arc: np.ndarray
    heading: np.ndarray
    pitch: np.ndarray
    curvature: np.ndarray
    curve_speed: np.ndarray
    length: float
    limit: float
    allowed: frozenset = frozenset((STRAIGHT, TURN_LEFT, TURN_RIGHT, UTURN))
    incoming: int = -1            # connections name their incoming/outgoing road lanes
    outgoing: int = -1
    node: int = -1
    turn: int = STRAIGHT
    start_trim: float = 0.
    end_trim: float = 0.

    @classmethod
    def create(cls, id, key, edge, index, points, limit, *, heading_hint=0., pitch_hint=0.,
               heading_end_hint=None, pitch_end_hint=None, seam=False, **kwargs):
        points = np.asarray(points, np.float64)
        endpoints = points[[0, -1]].copy()
        arc = _arc(points)
        keep = np.r_[True, np.diff(arc) > 1e-6]
        points, arc = points[keep], arc[keep]
        if len(points) < 2 or seam:
            # A straight seam has no geometry. A tiny virtual progress interval
            # keeps the sampler strictly ordered without moving/reversing the car.
            # Coincident XY samples may still have distinct deck elevations.
            # Coalescing must not replace the receiving lane's exact endpoint.
            points = endpoints
            length = max(float(arc[-1]), .001)
            arc = np.array([0., length])
            heading_end = heading_hint if heading_end_hint is None else heading_end_hint
            heading_end = heading_hint + math.atan2(math.sin(heading_end - heading_hint),
                                                    math.cos(heading_end - heading_hint))
            heading = np.array([heading_hint, heading_end])
            pitch = np.array([pitch_hint, pitch_hint if pitch_end_hint is None else pitch_end_hint])
            curvature, curve_speed = np.zeros(2), np.full(2, limit)
            return cls(id, key, edge, index, *[_readonly(a) for a in
                       (points, arc, heading, pitch, curvature, curve_speed)], length, float(limit), **kwargs)
        sample_arc = np.unique(np.r_[arc, np.linspace(0., arc[-1], max(2, int(arc[-1] / 1.5) + 1))])
        points = _interpolate(points, arc, sample_arc)
        arc = _arc(points)
        # Exact unique arc values can still interpolate to the same world point
        # after a rotated/translated connector is rounded. Coalesce those samples
        # before differentiating, keeping the connector's exact final endpoint.
        endpoint = points[-1].copy()
        points = points[np.r_[True, np.diff(arc) > 1e-6]]
        points[-1] = endpoint
        arc = _arc(points)
        d = np.gradient(points, arc, axis=0)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        pitch = np.arctan2(d[:, 2], np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9))
        if heading_end_hint is not None:
            # Finite differences of a sampled Bezier miss its exact end tangents.
            # Keep body orientation continuous when entering/leaving a connector.
            for k, hint in ((0, heading_hint), (-1, heading_end_hint)):
                heading[k] += math.atan2(math.sin(hint - heading[k]), math.cos(hint - heading[k]))
        if pitch_end_hint is not None:
            pitch[0], pitch[-1] = pitch_hint, pitch_end_hint
        curvature = np.gradient(heading, arc)
        curve_speed = np.sqrt(LATERAL_ACCELERATION / np.maximum(np.abs(curvature), 1e-6))
        # Tight bends are driven at a floor speed, never a crawl, and never above the limit.
        curve_speed = np.minimum(limit, np.maximum(curve_speed, MIN_CURVE_SPEED))
        # A backward comfortable-braking envelope anticipates each sharp vertex.
        for i in range(len(arc) - 2, -1, -1):
            curve_speed[i] = min(curve_speed[i], math.sqrt(curve_speed[i + 1] ** 2 + BRAKE_ENVELOPE * (arc[i + 1] - arc[i])))
        return cls(id, key, edge, index, *[_readonly(a) for a in
                   (points, arc, heading, pitch, curvature, curve_speed)], float(arc[-1]), float(limit), **kwargs)

    def project(self, position):
        """World point -> nearest progress; used only when streamed graph IDs change."""
        a, d = self.points[:-1, :2], np.diff(self.points[:, :2], axis=0)
        t = np.clip(np.sum((np.asarray(position)[:2] - a) * d, axis=1)
                    / np.maximum(np.sum(d * d, axis=1), 1e-9), 0., 1.)
        q = a + t[:, None] * d
        k = int(np.argmin(np.sum((q - np.asarray(position)[:2]) ** 2, axis=1)))
        return float(self.arc[k] + t[k] * (self.arc[k + 1] - self.arc[k])), float(np.linalg.norm(q[k] - np.asarray(position)[:2]))


class SplineTable:
    """A single searchsorted samples every car, including intersection connectors."""
    def __init__(self, splines):
        n = len(splines)
        self.length = np.array([s.length for s in splines])
        self.base = np.zeros(n)
        self.start = np.zeros(n, np.int64)
        self.last = np.zeros(n, np.int64)
        arrays = {name: [] for name in ('points', 'arc', 'heading', 'pitch', 'curvature', 'curve_speed')}
        glob, off, base = [], 0, 0.
        for s in splines:
            i = s.id
            self.base[i], self.start[i], self.last[i] = base, off, off + len(s.arc) - 2
            glob.append(s.arc + base)
            for name in arrays:
                arrays[name].append(getattr(s, name))
            base += s.length + 16.
            off += len(s.arc)
        for name, values in arrays.items():
            setattr(self, name, _readonly(np.concatenate(values) if values else np.empty((0, 3) if name == 'points' else 0)))
        self.glob = _readonly(np.concatenate(glob) if glob else np.empty(0))

    def sample(self, spline_ids, progress):
        ids = np.asarray(spline_ids, np.int64)
        p = np.clip(np.asarray(progress), 0., self.length[ids])
        i = np.searchsorted(self.glob, self.base[ids] + p, side='right') - 1
        i = np.minimum(np.maximum(i, self.start[ids]), self.last[ids])
        t = (p - self.arc[i]) / np.maximum(self.arc[i + 1] - self.arc[i], 1e-9)
        xyz = self.points[i] + (self.points[i + 1] - self.points[i]) * t[:, None]
        rest = tuple(getattr(self, name)[i] + (getattr(self, name)[i + 1] - getattr(self, name)[i]) * t
                     for name in ('heading', 'pitch', 'curvature', 'curve_speed'))
        return (xyz, *rest)

    def curve_limit(self, spline_ids, progress):
        """IDM only needs the braking envelope, not a second full body-pose sample."""
        ids = np.asarray(spline_ids, np.int64)
        p = np.clip(np.asarray(progress), 0., self.length[ids])
        i = np.searchsorted(self.glob, self.base[ids] + p, side='right') - 1
        i = np.minimum(np.maximum(i, self.start[ids]), self.last[ids])
        t = (p - self.arc[i]) / np.maximum(self.arc[i + 1] - self.arc[i], 1e-9)
        return self.curve_speed[i] + (self.curve_speed[i + 1] - self.curve_speed[i]) * t


@dataclass(frozen=True)
class Route:
    generation: int
    origin: int
    destination: int
    nodes: tuple
    edges: tuple
    turns: tuple
    allowed_lanes: tuple
    cost: float


class _PointRegistry:
    def __init__(self):
        self.grid, self.osm, self.positions, self.signals = {}, {}, {}, set()

    def add(self, point, path, node_id=0, signal=False):
        layer = int(getattr(path, 'layer', 0))
        grade = (layer, bool(getattr(path, 'bridge', False)), bool(getattr(path, 'tunnel', False)))
        if node_id:
            for key in self.osm.get(int(node_id), ()):
                if abs(self.positions[key][2] - point[2]) < 2.5:
                    if signal: self.signals.add(key)
                    return key
        bx, by = np.floor(point[:2]).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for key, other_id in self.grid.get((bx + dx, by + dy, grade), ()):
                    q = self.positions[key]
                    if node_id and other_id and node_id != other_id:
                        continue  # adjacent, distinct OSM nodes are not an intersection
                    if np.linalg.norm(q[:2] - point[:2]) < .20 and abs(q[2] - point[2]) < 2.5:
                        if signal: self.signals.add(key)
                        return key
        key = ('osm', int(node_id)) if node_id else ('xy', round(float(point[0]) * 20), round(float(point[1]) * 20), *grade)
        if key in self.positions:
            key = (*key, round(float(point[2]) * 2))
        self.positions[key] = np.asarray(point, np.float64)
        self.grid.setdefault((bx, by, grade), []).append((key, int(node_id)))
        if node_id: self.osm.setdefault(int(node_id), []).append(key)
        if signal: self.signals.add(key)
        return key


class RoadGraph:
    """Published snapshots are never mutated by routing or simulation threads."""
    def __init__(self, nodes, edges, lanes, connections, generation):
        self.generation = int(generation)
        self.nodes, self.edges, self.lanes = tuple(nodes), tuple(edges), tuple(lanes)
        self.connections = tuple(connections)
        self.splines = self.lanes + self.connections
        self.lane_count = len(lanes)
        self.table = SplineTable(self.splines)
        self.node_by_key = {n.key: n.id for n in nodes}
        self.node_positions = _readonly(np.array([n.position[:2] for n in nodes], np.float64).reshape(-1, 2))
        self.edge_by_key = {e.key: e.id for e in edges}
        self.spline_by_key = {s.key: s.id for s in self.splines}
        self.connection_by_pair = {(s.incoming, s.outgoing): s.id for s in connections}
        self.limits = _readonly(np.array([s.limit for s in self.splines]))
        self.curve_min = _readonly(np.array([s.curve_speed.min() for s in self.splines]))
        controlled = np.array([n.junction or n.signal for n in nodes], bool)
        receiving = {}
        for c in connections:
            previous = receiving.setdefault((c.node, c.outgoing), c.incoming)
            if previous != c.incoming:
                controlled[c.node] = True  # lane drops also need receiving-space leases
        self.controlled = _readonly(controlled)
        self.successors = {s.id: (s.outgoing,) for s in connections}
        self.predecessors = {}
        for s in connections:
            self.successors.setdefault(s.incoming, []).append(s.id)
            self.predecessors.setdefault(s.id, []).append(s.incoming)
            self.predecessors.setdefault(s.outgoing, []).append(s.id)
        self.transitions = {}
        self.lane_transitions = {}
        self.node_connections = {}
        self.source_edges = {}
        for edge in edges:
            self.source_edges.setdefault((*edge.source, edge.forward), []).append(edge.id)
        for s in connections:
            pair = (lanes[s.incoming].edge, lanes[s.outgoing].edge)
            self.transitions.setdefault(pair, []).append(s.id)
            self.lane_transitions.setdefault((*pair, s.incoming), s.id)
            self.node_connections.setdefault(s.node, []).append(s.id)
        # Published adjacency is immutable. Integer-only tuples can leave the
        # cyclic GC's candidate set; tens of thousands of permanent lists cannot.
        # Do not disable/freeze the process-wide collector or retain old graphs.
        for name in ('successors', 'predecessors', 'transitions', 'node_connections', 'source_edges'):
            setattr(self, name, {key: tuple(value) for key, value in getattr(self, name).items()})
        self.max_speed = max((e.limit for e in edges), default=25.)
        self._components()
        self._routing_topology()
        self._control_geometry()
        self._shared_geometry()

    def _shared_geometry(self):
        """One admission corridor through consecutive shared single-lane seams.

        Opposing directed records overlap intentionally. Group whole unbranched
        corridors, not each tiny OSM segment, so two cars cannot deadlock halfway
        along the same physical lane. Direction follows the corridor, not the
        possibly reversed source-way order.
        """
        groups = np.full(len(self.splines), -1, np.int64)
        directions = np.zeros(len(self.splines), np.int8)
        physical, incident = {}, {}
        for edge in self.edges:
            if not edge.shared: continue
            key = tuple(sorted((edge.start, edge.end)))
            physical.setdefault(key, []).append(edge.id)
        keys = tuple(physical)
        for i, (a, b) in enumerate(keys):
            incident.setdefault(a, []).append(i); incident.setdefault(b, []).append(i)
        parent = list(range(len(keys)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for node, segments in incident.items():
            if len(segments) == 2 and not self.nodes[node].junction:
                parent[root(segments[1])] = root(segments[0])
        corridors = {}
        for i in range(len(keys)): corridors.setdefault(root(i), []).append(i)
        for gid, segments in enumerate(corridors.values()):
            members, oriented = set(segments), {}
            pending = [(segments[0], *keys[segments[0]])]
            while pending:
                sid, a, b = pending.pop()
                if sid in oriented: continue
                oriented[sid] = (a, b)
                for node, at_start in ((a, True), (b, False)):
                    if len(incident[node]) != 2 or self.nodes[node].junction: continue
                    for other in incident[node]:
                        if other not in members or other in oriented: continue
                        x, y = keys[other]
                        end = y if x == node else x
                        pending.append((other, end, node) if at_start else (other, node, end))
            for sid in segments:
                a, b = oriented[sid]
                for eid in physical[keys[sid]]:
                    edge = self.edges[eid]
                    for lane in edge.lanes:
                        groups[lane], directions[lane] = gid, 1 if edge.start == a else -1
        memberships = [()] * len(self.splines)
        for lane in self.lanes:
            if groups[lane.id] >= 0: memberships[lane.id] = ((int(groups[lane.id]), int(directions[lane.id])),)
        for c in self.connections:
            memberships[c.id] = tuple(dict.fromkeys((*memberships[c.incoming], *memberships[c.outgoing])))
            if memberships[c.id]: groups[c.id], directions[c.id] = memberships[c.id][0]
        self.shared_group, self.shared_direction = _readonly(groups), _readonly(directions)
        self.shared_memberships = tuple(memberships)
        self.shared_corridors = len(corridors)

    def _control_geometry(self):
        """Physical bumper bounds, independent of short-link/capped spline trims.

        Offsets are signed arc distances from the incoming lane's endpoint.
        Negative offsets deliberately propagate a stop back across shape seams.
        A connector's entry arc is where its body's front reaches the crossing
        carriageway, not where an arbitrary trimmed spline happens to begin.
        """
        offsets = np.full(self.lane_count, 3.)
        entries = np.full(len(self.splines), np.inf)
        approach_offsets = np.zeros(len(self.splines))
        lane_entries = np.zeros(self.lane_count)
        planes = [()] * self.lane_count
        virtual = np.array([max(0., s.length - _arc(s.points)[-1]) for s in self.splines])
        for lane in self.lanes:
            edge = self.edges[lane.edge]
            node = self.nodes[edge.end]
            if not self.controlled[node.id]: continue
            p = lane.points[-1, :2]
            forward = np.array([math.cos(lane.heading[-1]), math.sin(lane.heading[-1])])
            right = np.array([forward[1], -forward[0]])
            bound = float((node.position[:2] - p) @ forward)
            crossing = []
            boundaries = []
            if node.junction:
                for eid in (*node.incoming, *node.outgoing):
                    other = self.edges[eid]
                    yaw = other.heading_end if other.end == node.id else other.heading_start
                    normal = np.array([-math.sin(yaw), math.cos(yaw)])
                    advance = float(normal @ forward)
                    if abs(advance) <= .2: continue
                    lateral = BUMPER_HALF_WIDTH * abs(float(normal @ right))
                    crossing.append((-other.width * .5 - lateral
                                     - math.copysign(1., advance) * float(normal @ (p - node.position[:2])))
                                    / abs(advance))
                    boundaries.append((normal * math.copysign(1., advance), other.width * .5))
                bound = min(crossing) if crossing else bound - node.radius
            elif not node.signal:
                bound = 0.  # controlled lane drop: the receiving merge begins here
            lane_entries[lane.id] = bound
            planes[lane.id] = tuple(boundaries) if boundaries else ((forward, node.radius if node.junction else 0.),)
            offsets[lane.id] = bound - STOP_MARGIN - self._virtual_stop_back(lane.id, bound - STOP_MARGIN, virtual)
        for c in self.connections:
            if not self.controlled[c.node]: continue
            lane = self.lanes[c.incoming]
            bound = lane_entries[lane.id]
            entry = bound
            if bound > 0.:
                forward = np.array([math.cos(lane.heading[-1]), math.sin(lane.heading[-1])])
                projected = (c.points[:, :2] - lane.points[-1, :2]) @ forward
                hits = np.flatnonzero(projected >= bound)
                if len(hits):
                    k = int(hits[0])
                    entry = (0. if k == 0 else float(c.arc[k - 1] + (c.arc[k] - c.arc[k - 1])
                             * (bound - projected[k - 1]) / max(projected[k] - projected[k - 1], 1e-9)))
                else:
                    entry = min(bound, c.length)
                if np.ptp(c.heading) > 1e-6:
                    entry = self._body_entry(c, lane, planes[lane.id], entry)
            entry -= self._virtual_stop_back(lane.id, entry - STOP_MARGIN, virtual)
            entries[c.id] = entry
            # Use the same signed connector arc on both sides of its seam.
            # Projected metres and curved-connector metres are not interchangeable.
            approach_offsets[c.id] = entry - STOP_MARGIN
        self.stop_offsets = _readonly(offsets)
        self.controlled_ends = _readonly(np.array([self.controlled[self.edges[lane.edge].end]
                                                  for lane in self.lanes], bool))
        self.entry_progress = _readonly(entries)
        self.approach_offsets = _readonly(approach_offsets)
        self.max_stop_back = max(0., -float(offsets.min())) if len(offsets) else 0.

    def _virtual_stop_back(self, lane, front, virtual):
        """Do not count zero-distance sampler intervals as physical stop room.

        A negative bumper bound can cross several coincident shape seams. Their
        millimetre progress intervals remain in routing/odometer semantics, but
        cannot move a stopped body closer to the crossing carriageway.
        """
        reach = max(0., .5 * float(LENGTHS.max()) - front)
        pending, seen, maximum = [(lane, reach, 0.)], {}, 0.
        for _ in range(128):
            if not pending: break
            current, remaining, debt = pending.pop()
            physical = self.table.length[current] - virtual[current]
            maximum = max(maximum, debt)
            if remaining <= physical: continue
            debt += virtual[current]
            maximum = max(maximum, debt)
            remaining -= physical
            if seen.get(current, -1.) >= remaining: continue
            seen[current] = remaining
            pending.extend((previous, remaining, debt) for previous in self.predecessors.get(current, ()))
        # If a pathological cycle exceeds the bounded walk, retain the safe
        # upper bound for its remaining virtual intervals, not invented storage.
        if pending: maximum += sum(float(virtual[current]) for current, _, _ in pending) + 128. * float(virtual.max(initial=0.))
        return maximum

    def _body_entry(self, connection, lane, planes, entry):
        """Earliest conservative front-body contact in the actual pose sampler.

        A front point on the centreline misses the outside corner of a turning
        bus. Bound each interpolated position/yaw interval by the rectangle's
        support function, then refine the first possible contact. One signed
        entry arc is conservative for every production body length.
        """
        c = connection
        half = np.unique(.5 * LENGTHS)[:, None]
        radius, peak = np.hypot(half, BUMPER_HALF_WIDTH), np.arctan2(BUMPER_HALF_WIDTH, half)
        node = self.nodes[c.node].position[:2]
        def support(a, b):
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            ends = np.maximum(half * np.abs(np.cos(a)) + BUMPER_HALF_WIDTH * np.abs(np.sin(a)),
                              half * np.abs(np.cos(b)) + BUMPER_HALF_WIDTH * np.abs(np.sin(b)))
            critical = ((np.ceil((lo - peak) / math.pi) <= np.floor((hi - peak) / math.pi))
                        | (np.ceil((lo + peak) / math.pi) <= np.floor((hi + peak) / math.pi)))
            return np.where(critical, radius, ends)
        for normal, width in planes:
            projected = (c.points[:, :2] - node) @ normal + width
            heading = c.heading - math.atan2(normal[1], normal[0])
            possible = (np.maximum(projected[:-1], projected[1:])[None, :]
                        + support(heading[:-1][None, :], heading[1:][None, :]) >= 0.)
            has_hit = possible.any(axis=1)
            if not has_hit.any(): continue
            indices = possible.argmax(axis=1)
            a, b = c.arc[indices], c.arc[indices + 1]
            pa, pb = projected[indices], projected[indices + 1]
            ya, yb = heading[indices], heading[indices + 1]
            lo, hi = np.zeros(len(half)), np.ones(len(half))
            for _ in range(18):
                mid = .5 * (lo + hi)
                p0, pm = pa + (pb - pa) * lo, pa + (pb - pa) * mid
                y0, ym = ya + (yb - ya) * lo, ya + (yb - ya) * mid
                # Each row has its own body length and bracket.
                envelope = np.maximum(p0, pm) + np.diag(support(y0[None, :], ym[None, :]))
                safe = envelope < 0.
                lo, hi = np.where(safe, mid, lo), np.where(safe, hi, mid)
            centres = a + (b - a) * lo
            # Bodies already touching at spline origin entered on the incoming
            # straight extension; do not replace its signed upstream bound.
            at_origin = projected[0] + half[:, 0] * abs(math.cos(heading[0])) + BUMPER_HALF_WIDTH * abs(math.sin(heading[0])) >= 0.
            forward = np.array([math.cos(lane.heading[-1]), math.sin(lane.heading[-1])])
            advance = float(normal @ forward)
            upstream = (-float((lane.points[-1, :2] - node) @ normal) - width
                        - BUMPER_HALF_WIDTH * abs(math.sin(heading[0]))) / max(advance, 1e-9)
            candidates = np.where(at_origin, upstream, centres + half[:, 0])
            entry = min(entry, float(candidates[has_hit].min()))
        return entry

    @classmethod
    def topology(cls, tiles):
        """Directed centrelines shared by traffic splines and lightweight navigation.

        Only this stage establishes OSM identity, tile seams and grade-separated
        junctions. Navigation does not need a second city's worth of AI splines.
        """
        registry, sources, owners = _PointRegistry(), [], {}
        iterable = tiles.values() if isinstance(tiles, dict) else tiles
        for tile in iterable:
            if not isinstance(tile, GraphTile):
                tile = GraphTile(tile['key'], tuple(tile['paths']), tuple(tile['origin']), tuple(tile['scale']))
            for pi, path in enumerate(tile.paths):
                p = np.asarray(path.points, np.float64)
                if len(p) < 2 or not np.isfinite(p).all():
                    continue
                z = getattr(path, 'z', None)
                z = np.asarray(z, np.float64) if z is not None and len(z) == len(p) else np.zeros(len(p))
                xyz = np.c_[p * np.asarray(tile.scale) + np.asarray(tile.origin), z]
                arc = _arc(xyz)
                keep = np.r_[True, np.diff(arc) > 1e-4]
                xyz, arc = xyz[keep], arc[keep]
                if len(xyz) < 2 or arc[-1] < .5:
                    continue
                local_arc = np.asarray(path.arc)[keep]
                raw_ids = getattr(path, 'node_ids', ())
                ids = np.asarray(raw_ids, np.int64)[keep] if len(raw_ids) == len(p) else np.zeros(len(xyz), np.int64)
                signals = set(getattr(path, 'signal_nodes', ()))
                keys = [registry.add(q, path, int(n), int(n) in signals) for q, n in zip(xyz, ids)]
                source_id = len(sources)
                for key in set(keys): owners.setdefault(key, set()).add(source_id)
                sources.append(dict(tile=tile.key, pi=pi, path=path, xyz=xyz, arc=arc,
                                    local_arc=local_arc, ids=ids, keys=keys, cuts=[]))
        for src in sources:
            for i, key in enumerate(src['keys']):
                if i in (0, len(src['keys']) - 1) or src['ids'][i] or len(owners[key]) > 1:
                    src['cuts'].append((float(src['arc'][i]), key))
        cls._geometry_junctions(sources, registry)
        node_keys = dict.fromkeys(key for src in sources for _, key in src['cuts'])
        node_ids = {key: i for i, key in enumerate(node_keys)}
        nodes = [RoadNode(i, key, _readonly(registry.positions[key]), signal=key in registry.signals)
                 for key, i in node_ids.items()]
        incoming, outgoing = [[] for _ in nodes], [[] for _ in nodes]
        edges, centre = [], []
        for src in sources:
            path = src['path']
            layout = RoadLayout.from_road(path)
            cuts = sorted(src['cuts'], key=lambda item: item[0])
            unique = []
            for cut in cuts:
                if not unique or cut[0] - unique[-1][0] > .20:
                    unique.append(cut)
            for (a, ka), (b, kb) in zip(unique[:-1], unique[1:]):
                if b - a < .25 or ka == kb:
                    continue
                line = _slice(src['xyz'], src['arc'], a, b)
                la, lb = np.interp([a, b], src['arc'], src['local_arc'])
                for forward in (True, False):
                    if not forward and path.oneway:
                        continue
                    if not (layout.forward if forward else layout.backward):
                        continue
                    q = line if forward else line[::-1]
                    start, end = node_ids[ka if forward else kb], node_ids[kb if forward else ka]
                    hs = math.atan2(q[1, 1] - q[0, 1], q[1, 0] - q[0, 0])
                    he = math.atan2(q[-1, 1] - q[-2, 1], q[-1, 0] - q[-2, 0])
                    eid = len(edges)
                    key = (src['tile'], src['pi'], nodes[start].key, nodes[end].key, forward)
                    edges.append(RoadSegment(eid, key, start, end, (src['tile'], src['pi']), forward,
                                 float(la if forward else lb), float(lb if forward else la), b - a,
                                 speed_limit(path.highway, getattr(path, 'speed_limit', 0.)),
                                  road_priority(path.highway), layout.width, hs, he, path.highway,
                                  layout.forward if forward else layout.backward,
                                  tuple(getattr(path, 'turn_lanes' if forward else 'turn_lanes_backward', ())),
                                   oneway=bool(path.oneway), offsets=layout.offsets(forward), shared=layout.shared))
                    centre.append(q)
                    incoming[end].append(eid); outgoing[start].append(eid)
        for i, node in enumerate(nodes):
            neighbours = {edges[e].start for e in incoming[i]} | {edges[e].end for e in outgoing[i]}
            incident = [edges[e] for e in incoming[i] + outgoing[i]]
            junction = len(neighbours) >= 3
            major = junction and len(neighbours) >= 3 and any(e.lane_count >= 2 and e.priority >= 2 for e in incident)
            # Motorway junctions merge, rather than inventing traffic lights.
            surface = any(e.highway not in ('motorway', 'trunk', 'motorway_link', 'trunk_link') for e in incident)
            radius = min(14., max((e.width * .5 + 1. for e in incident), default=0.)) if junction else 0.
            if not junction:
                bend = max((abs(math.atan2(math.sin(edges[b].heading_start - edges[a].heading_end),
                                          math.cos(edges[b].heading_start - edges[a].heading_end)))
                            for a in incoming[i] for b in outgoing[i]
                            if edges[b].end != edges[a].start), default=0.)
                if bend > .001:
                    radius = 4.  # round ordinary shape vertices too, not just boxes
            nodes[i] = replace(node, incoming=tuple(incoming[i]), outgoing=tuple(outgoing[i]),
                                junction=junction, radius=radius, signal=node.signal or (major and surface))
        return nodes, edges, centre

    @classmethod
    def build(cls, tiles, generation=1):
        nodes, edges, centre = cls.topology(tiles)
        nodes = cls._assign_signals(nodes, edges)
        nodes = cls._seam_setbacks(nodes, edges)
        lanes = []
        for ei, edge in enumerate(edges):
            q = _smooth_line(centre[ei], 7. if edge.priority >= 4 else 4.)
            arc = _arc(q)
            d = np.gradient(q[:, :2], axis=0)
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            side = np.c_[d[:, 1], -d[:, 0], np.zeros(len(d))]
            def setback(node):
                return node.radius + 3. if node.junction or node.signal else node.radius
            start_trim = min(setback(nodes[edge.start]), arc[-1] * LANE_TRIM_FRACTION)
            end_trim = min(setback(nodes[edge.end]), arc[-1] * LANE_TRIM_FRACTION)
            lane_ids = []
            for ordinal, offset in enumerate(cls._lane_offsets(edge, edges, nodes)):
                shifted = q + side * offset
                shifted_arc = _arc(shifted)
                a = min(start_trim, shifted_arc[-1] * LANE_TRIM_FRACTION)
                b = max(shifted_arc[-1] - end_trim, shifted_arc[-1] * (1. - LANE_TRIM_FRACTION))
                line = _slice(shifted, shifted_arc, a, b)
                allowed = _allowed_turns(edge.turn_specs[ordinal]) if nodes[edge.end].junction and ordinal < len(edge.turn_specs) else frozenset((STRAIGHT, TURN_LEFT, TURN_RIGHT, UTURN))
                if nodes[edge.end].junction and not edge.turn_specs and edge.lane_count >= 3:
                    allowed = frozenset((TURN_LEFT,)) if ordinal == 0 else frozenset((TURN_RIGHT,)) if ordinal == edge.lane_count - 1 else frozenset((STRAIGHT,))
                lid = len(lanes)
                lanes.append(LaneSpline.create(lid, ('lane', edge.key, ordinal), ei, ordinal, line,
                                              edge.limit, allowed=allowed, start_trim=a, end_trim=shifted_arc[-1] - b))
                lane_ids.append(lid)
            edges[ei] = replace(edge, lanes=tuple(lane_ids))
        connections = cls._connections(nodes, edges, lanes)
        return cls(nodes, edges, lanes, connections, generation)

    @staticmethod
    def _assign_signals(nodes, edges):
        """Approach-tagged OSM lights share their nearby junction's controller.

        Standalone crossings keep their own controller. Only connected road
        edges within 40 m are followed; geometric proximity is not enough.
        """
        assigned, relocated = set(), set()
        for node in nodes:
            if not node.signal or node.junction:
                continue
            queue, seen = [(0., node.id)], {node.id}
            while queue:
                distance, current = heapq.heappop(queue)
                if distance > 40.:
                    break
                if current != node.id and nodes[current].junction:
                    assigned.add(current); relocated.add(node.id)
                    break
                for eid in nodes[current].outgoing:
                    target = edges[eid].end
                    if target not in seen:
                        seen.add(target)
                        heapq.heappush(queue, (distance + edges[eid].length, target))
        return [replace(n, signal=(n.signal and n.id not in relocated) or n.id in assigned) for n in nodes]

    @staticmethod
    def _oneway(edge, edges, nodes):
        if edge.oneway is not None:
            return edge.oneway
        return not any(edges[e].source == edge.source and edges[e].end == edge.start
                       for e in nodes[edge.end].outgoing)

    @classmethod
    def _lane_offsets(cls, edge, edges, nodes):
        if edge.offsets:
            return edge.offsets
        oneway = cls._oneway(edge, edges, nodes)
        opposite = 0 if oneway else next((edges[e].lane_count for e in nodes[edge.end].outgoing
                                         if edges[e].source == edge.source and edges[e].end == edge.start), edge.lane_count)
        return lane_offsets(edge.width, edge.lane_count, opposite)

    @classmethod
    def _seam_setbacks(cls, nodes, edges):
        """Give bends/lane-width changes actual forward room, not lateral loops."""
        offsets = [cls._lane_offsets(e, edges, nodes) for e in edges]
        result = []
        for node in nodes:
            radius = node.radius
            if not node.junction:
                for incoming in node.incoming:
                    a = edges[incoming]
                    for outgoing in node.outgoing:
                        b = edges[outgoing]
                        if b.end == a.start:
                            continue
                        angle = abs(math.atan2(math.sin(b.heading_start - a.heading_end),
                                               math.cos(b.heading_start - a.heading_end)))
                        if angle >= 2.6:
                            continue
                        shift = max(abs(x - offsets[outgoing][min(i, b.lane_count - 1)])
                                    for i, x in enumerate(offsets[incoming]))
                        if shift > .05:
                            radius = max(radius, 8., shift * 3.)
                        if angle > .001:
                            inner = max(abs(x) for x in (*offsets[incoming], *offsets[outgoing]))
                            radius = max(radius, (inner + 4.) * math.tan(angle * .5))
            result.append(replace(node, radius=min(radius, 40.)))
        return result

    @staticmethod
    def _geometry_junctions(sources, registry):
        if len(sources) < 2:
            return
        # Shapely is already required by the OSM importer, but not by route search.
        from shapely.geometry import LineString
        from shapely import STRtree
        lines = [LineString(s['xyz'][:, :2]) for s in sources]
        tree = STRtree(lines)
        pairs = tree.query(lines, predicate='intersects')
        for i, j in zip(*pairs):
            if j <= i:
                continue
            a, b = sources[int(i)], sources[int(j)]
            # Real OSM ways share a node at junctions. Do not infer a crossing
            # between two fully identified ways (e.g. an untagged overpass).
            if a['ids'].any() and b['ids'].any():
                continue
            pa, pb = a['path'], b['path']
            grade_a = (getattr(pa, 'layer', 0), getattr(pa, 'bridge', False), getattr(pa, 'tunnel', False))
            grade_b = (getattr(pb, 'layer', 0), getattr(pb, 'bridge', False), getattr(pb, 'tunnel', False))
            if grade_a != grade_b:
                continue
            cut = lines[int(i)].intersection(lines[int(j)])
            points = [cut] if cut.geom_type == 'Point' else list(cut.geoms) if cut.geom_type == 'MultiPoint' else []
            for p in points:
                sa, sb = lines[int(i)].project(p), lines[int(j)].project(p)
                qa = _interpolate(a['xyz'], a['arc'], [sa])[0]
                qb = _interpolate(b['xyz'], b['arc'], [sb])[0]
                if abs(qa[2] - qb[2]) > 2.5:
                    continue
                key = registry.add((qa + qb) * .5, pa)
                a['cuts'].append((float(sa), key)); b['cuts'].append((float(sb), key))

    @staticmethod
    def _connections(nodes, edges, lanes):
        connections = []
        for node in nodes:
            neighbours = {edges[e].start for e in node.incoming} | {edges[e].end for e in node.outgoing}
            for incoming in node.incoming:
                a = edges[incoming]
                for outgoing in node.outgoing:
                    b = edges[outgoing]
                    turn = turn_kind(b.heading_start - a.heading_end)
                    if turn == UTURN and len(neighbours) > 1:
                        continue
                    for in_lane in a.lanes:
                        la = lanes[in_lane]
                        if turn not in la.allowed:
                            continue
                        if not node.junction and turn != UTURN:
                            # A bend in one continuous road is not a junction
                            # turn: parallel lanes keep their identity through it.
                            target = b.lanes[min(la.index, len(b.lanes) - 1)]
                        elif turn in (TURN_LEFT, UTURN):
                            target = b.lanes[0]
                        elif turn == TURN_RIGHT:
                            target = b.lanes[-1]
                        else:
                            target = b.lanes[min(la.index, len(b.lanes) - 1)]
                        lb = lanes[target]
                        p, q = la.points[-1], lb.points[0]
                        ta = np.array([math.cos(la.heading[-1]), math.sin(la.heading[-1]), math.tan(la.pitch[-1])])
                        tb = np.array([math.cos(lb.heading[0]), math.sin(lb.heading[0]), math.tan(lb.pitch[0])])
                        distance = float(np.linalg.norm(q[:2] - p[:2]))
                        if turn == UTURN and distance < 8.:
                            # A narrow dead end cannot accommodate a rolling
                            # U-turn. End/recycle the trip instead of spinning.
                            continue
                        delta_heading = math.atan2(math.sin(lb.heading[0] - la.heading[-1]),
                                                   math.cos(lb.heading[0] - la.heading[-1]))
                        seam = turn == STRAIGHT and distance < .05 and abs(delta_heading) < .02
                        cid = len(lanes) + len(connections)
                        key = ('turn', la.key, lb.key)
                        t = None if seam else np.linspace(0., 1., max(5, int(distance * 1.5) + 1))[:, None]
                        # Long handles can fold an otherwise forward lane-width
                        # transition. Try shorter ACTUAL curves, not a heading
                        # filter or looser admission bounds. Unsafe links still
                        # have no connector, and geometric braking is unchanged.
                        fractions = (.55,) if seam or turn == UTURN else (.55, .45, .35, .25, .15, .1)
                        for fraction in fractions:
                            control = distance * fraction
                            if turn == UTURN: control = max(control, 4.)
                            if seam:
                                line = np.vstack((p, q))
                            else:
                                cp1, cp2 = p + ta * control, q - tb * control
                                line = ((1 - t) ** 3 * p + 3 * (1 - t) ** 2 * t * cp1
                                        + 3 * (1 - t) * t ** 2 * cp2 + t ** 3 * q)
                            connection = LaneSpline.create(cid, key, -1, 0, line, min(a.limit, b.limit),
                                                           heading_hint=la.heading[-1], pitch_hint=la.pitch[-1],
                                                           heading_end_hint=lb.heading[0], pitch_end_hint=lb.pitch[0], seam=seam,
                                                           incoming=in_lane, outgoing=target, node=node.id, turn=turn)
                            relative = connection.heading - connection.heading[0]
                            if ((turn != UTURN and np.ptp(connection.heading) > math.pi)
                                    or relative.min() < min(0., delta_heading) - .52
                                    or relative.max() > max(0., delta_heading) + .52
                                    or np.max(np.abs(connection.curvature)) > 2.):
                                # No loops, full rotations or sub-half-metre pivots.
                                continue
                            connections.append(connection)
                            break
        return connections

    def _components(self):
        parent = list(range(len(self.nodes)))
        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for e in self.edges:
            parent[root(e.end)] = root(e.start)
        self.component = np.array([root(i) for i in range(len(parent))], np.int64)
        self.component_nodes = {}
        for i, component in enumerate(self.component):
            self.component_nodes.setdefault(int(component), []).append(i)
        self.component_nodes = {key: tuple(value) for key, value in self.component_nodes.items()}
        self.origins = np.array([n.id for n in self.nodes if n.outgoing], np.int64)
        weights = np.array([sum(self.edges[e].length * self.edges[e].lane_count for e in self.nodes[i].outgoing) for i in self.origins])
        self.origin_weights = weights / weights.sum() if len(weights) else weights

    def _routing_topology(self):
        """Arrival-edge adjacency and its directed, turn-legal condensation graph.

        An undirected road component is not a reachable destination set. Missing
        connectors, one-ways and dedicated lanes can strand a driver inside it;
        repeatedly searching the whole city for those destinations starves physics.
        These immutable records are shared by searches, not rebuilt for every trip.
        """
        self._edge_times = tuple(e.length / e.limit for e in self.edges)
        penalties = {STRAIGHT: 0., TURN_LEFT: 4., TURN_RIGHT: 1., UTURN: 14.}
        successors = []
        reverse = [[] for _ in self.edges]
        for edge in self.edges:
            following = []
            for eid in self.nodes[edge.end].outgoing:
                connections = self.transitions.get((edge.id, eid))
                if connections is not None:
                    turn = self.splines[connections[0]].turn
                    following.append((eid, self.edges[eid].end, self._edge_times[eid] + penalties[turn]))
                    reverse[eid].append(edge.id)
            successors.append(tuple(following))
        self._route_successors = tuple(successors)
        # Iterative Kosaraju: long OSM chains must not overflow Python recursion.
        seen, order = bytearray(len(self.edges)), []
        for start in range(len(self.edges)):
            if seen[start]: continue
            pending = [(start, False)]
            while pending:
                eid, finished = pending.pop()
                if finished:
                    order.append(eid)
                elif not seen[eid]:
                    seen[eid] = 1
                    pending.append((eid, True))
                    pending.extend((following, False) for following, _, _ in reversed(successors[eid])
                                   if not seen[following])
        component = [-1] * len(self.edges)
        count = 0
        for start in reversed(order):
            if component[start] >= 0: continue
            component[start] = count
            pending = [start]
            while pending:
                eid = pending.pop()
                for previous in reverse[eid]:
                    if component[previous] < 0:
                        component[previous] = count
                        pending.append(previous)
            count += 1
        children, destinations = [set() for _ in range(count)], [set() for _ in range(count)]
        for edge in self.edges:
            own = component[edge.id]
            destinations[own].add(edge.end)
            children[own].update(component[eid] for eid, _, _ in successors[edge.id] if component[eid] != own)
        self._route_component = tuple(component)
        self._route_children = tuple(tuple(sorted(c)) for c in children)
        self._route_nodes = tuple(tuple(sorted(n)) for n in destinations)

    def destination_components(self, origin, incoming=-1):
        """Small immutable key for the legal first arrivals of a random request."""
        if not 0 <= origin < len(self.nodes): return ()
        return tuple(sorted({self._route_component[eid] for eid in self.nodes[origin].outgoing
                             if incoming < 0 or (incoming, eid) in self.transitions}))

    def reachable_destinations(self, components):
        """Uniform destination pool, including downstream one-way/sink components."""
        seen, nodes, pending = set(components), set(), list(components)
        while pending:
            current = pending.pop()
            nodes.update(self._route_nodes[current])
            for child in self._route_children[current]:
                if child not in seen:
                    seen.add(child); pending.append(child)
        return tuple(sorted(nodes))

    def compile_route(self, edges, origin=None, destination=None, cost=None):
        edges = tuple(int(e) for e in edges)
        if not edges:
            if origin is None or destination is None or origin != destination:
                return None
            return Route(self.generation, origin, destination, (origin,), (), (), (), 0.)
        for a, b in zip(edges[:-1], edges[1:]):
            if (a, b) not in self.transitions:
                return None
        nodes = (self.edges[edges[0]].start, *[self.edges[e].end for e in edges])
        turns, allowed = [], []
        for i, edge in enumerate(edges):
            if i + 1 < len(edges):
                connections = self.transitions[(edge, edges[i + 1])]
                turns.append(self.splines[connections[0]].turn)
                allowed.append(tuple(dict.fromkeys(self.splines[c].incoming for c in connections)))
            else:
                turns.append(STRAIGHT); allowed.append(self.edges[edge].lanes)
        cost = sum(self.edges[e].length / self.edges[e].limit for e in edges) if cost is None else cost
        return Route(self.generation, nodes[0], nodes[-1], tuple(nodes), edges, tuple(turns), tuple(allowed), float(cost))

    def find_route(self, origin, destination, *, incoming=-1, algorithm='astar', _cooperate=None):
        """A* on directed (node, arrival-segment) states; Dijkstra uses h=0.

        Arrival state prevents illegal U-turns and routes that cannot be driven
        from any of the incoming segment's dedicated lanes.
        """
        if algorithm not in ('astar', 'dijkstra'):
            raise ValueError('algorithm must be astar or dijkstra')
        origin, destination = int(origin), int(destination)
        if not (0 <= origin < len(self.nodes) and 0 <= destination < len(self.nodes)):
            return None
        if self.component[origin] != self.component[destination]:
            return None
        if origin == destination:
            return self.compile_route((), origin, destination)
        # One vector distance calculation is substantially cheaper than creating
        # a NumPy vector/norm for every heap relaxation in a city-wide search.
        if algorithm == 'astar':
            delta = self.node_positions - self.node_positions[destination]
            heuristic = np.hypot(delta[:, 0], delta[:, 1]) / self.max_speed
        else:
            heuristic = None
        incoming = int(incoming)
        # A reached state is uniquely identified by its arrival edge. Compact
        # indexed lists avoid millions of (node, edge) tuple/dict allocations.
        best, previous = [math.inf] * len(self.edges), [-1] * len(self.edges)
        if 0 <= incoming < len(self.edges) and self.edges[incoming].end == origin:
            best[incoming] = 0.
        queue = []
        penalties = {STRAIGHT: 0., TURN_LEFT: 4., TURN_RIGHT: 1., UTURN: 14.}
        for eid in self.nodes[origin].outgoing:
            connections = self.transitions.get((incoming, eid)) if incoming >= 0 else None
            if incoming >= 0 and connections is None: continue
            turn = self.splines[connections[0]].turn if connections is not None else STRAIGHT
            cost = self._edge_times[eid] + penalties[turn]
            if cost + 1e-9 >= best[eid]: continue
            best[eid] = cost
            node = self.edges[eid].end
            estimate = float(heuristic[node]) if heuristic is not None else 0.
            heapq.heappush(queue, (cost + estimate, cost, node, eid))
        end, expanded = -1, 0
        while queue:
            _, cost, node, arrival = heapq.heappop(queue)
            if cost > best[arrival] + 1e-9:
                continue
            if node == destination:
                end = arrival; break
            for eid, following, travel in self._route_successors[arrival]:
                candidate = cost + travel
                if candidate + 1e-9 < best[eid]:
                    best[eid], previous[eid] = candidate, arrival
                    estimate = float(heuristic[following]) if heuristic is not None else 0.
                    heapq.heappush(queue, (candidate + estimate, candidate, following, eid))
            expanded += 1
            if _cooperate is not None and expanded % 256 == 0:
                _cooperate()
        if end < 0:
            return None
        state, edges = end, []
        while state >= 0:
            edges.append(state); state = previous[state]
        return self.compile_route(edges[::-1], cost=best[end])

    def random_trip(self, rng, *, origin=None, incoming=-1, algorithm='astar'):
        if not len(self.origins):
            return None
        source = int(rng.choice(self.origins, p=self.origin_weights)) if origin is None else int(origin)
        if not (0 <= source < len(self.nodes)):
            return None
        destinations = self.reachable_destinations(self.destination_components(source, incoming))
        if not destinations: return None
        for _ in range(16):
            dest = int(rng.choice(destinations))
            if dest == source:
                continue
            route = self.find_route(source, dest, incoming=incoming, algorithm=algorithm)
            if route is not None and route.edges:
                return route
        # Directed sinks/disconnected small islands cannot hang the spawn queue.
        for eid in self.nodes[source].outgoing:
            if incoming < 0 or (incoming, eid) in self.transitions:
                return self.compile_route((eid,))
        return None

    def next_connection(self, route, cursor, lane_id):
        if route is None or cursor + 1 >= len(route.edges):
            return -1
        return self.lane_transitions.get((route.edges[cursor], route.edges[cursor + 1], lane_id), -1)
